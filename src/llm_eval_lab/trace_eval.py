from __future__ import annotations

import json
from dataclasses import asdict
from typing import Iterable, Protocol

from .checks import token_coverage
from .models import (
    AgentEvalCase,
    AgentEvalRegressionReport,
    AgentEvalReport,
    AgentRunEvalResult,
    CapturedRun,
    EvaluationContext,
    StructuredCheck,
    TraceEvent,
)
from .validation import fingerprint, number, unique_ids


class RunRubricJudge(Protocol):
    @property
    def configuration(self) -> dict: ...

    def grade(
        self, run: CapturedRun, case: AgentEvalCase
    ) -> tuple[StructuredCheck, ...]: ...


class HeuristicRunRubricJudge:
    """Lexical diagnostics only. Semantic criteria remain unknown without a judge."""

    configuration = {"kind": "lexical-diagnostic", "version": 2}

    def grade(
        self, run: CapturedRun, case: AgentEvalCase
    ) -> tuple[StructuredCheck, ...]:
        checks = [
            StructuredCheck(
                "reference_overlap",
                "diagnostic",
                None,
                None,
                f"token coverage={token_coverage(run.output, case.reference):.3f}; not a correctness verdict",
                weight=0,
            )
        ]
        checks.extend(
            StructuredCheck(
                c.check_id,
                "quality",
                None,
                None,
                "Semantic evaluation requires a configured judge or human review.",
                source="rubric",
                weight=c.weight,
                must_pass=c.must_pass,
            )
            for c in case.rubric
        )
        return tuple(checks)


def _artifact_present(value: object) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple, dict, set)):
        return bool(value)
    return True


def parse_trace_jsonl(lines: str | Iterable[str]) -> tuple[TraceEvent, ...]:
    """Read normalized trace JSONL; never silently discard malformed events."""
    events = []
    for lineno, line in enumerate(
        lines.splitlines() if isinstance(lines, str) else lines, 1
    ):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError("event must be an object")
            events.append(TraceEvent.from_dict(payload))
        except (ValueError, TypeError) as exc:
            raise ValueError(f"trace line {lineno}: {exc}") from exc
    return tuple(events)


def _ordered_subsequence(actual: list[str], expected: tuple[str, ...]) -> bool:
    position = 0
    for event_type in actual:
        if position < len(expected) and event_type == expected[position]:
            position += 1
    return position == len(expected)


def validate_checks(checks: tuple[StructuredCheck, ...]) -> None:
    unique_ids((c.check_id for c in checks), "check ids")
    for c in checks:
        if c.category not in {
            "outcome",
            "process",
            "quality",
            "efficiency",
            "risk",
            "diagnostic",
        }:
            raise ValueError(f"invalid category: {c.category}")
        if c.source not in {"deterministic", "rubric", "llm", "human"}:
            raise ValueError(f"invalid source: {c.source}")
        number(c.weight, "check weight")
        if c.category == "diagnostic" and (c.weight != 0 or c.must_pass):
            raise ValueError("diagnostics cannot affect scores or gates")
        if type(c.must_pass) is not bool or (c.must_pass and c.weight == 0):
            raise ValueError("must-pass checks need positive weights")
        if c.passed is None:
            if c.score is not None:
                raise ValueError("unknown checks must have null scores")
        elif type(c.passed) is not bool:
            raise ValueError("check pass must be bool or null")
        else:
            number(c.score, "check score", maximum=1)
        if not isinstance(c.notes, str) or not c.notes.strip():
            raise ValueError("checks require explanatory notes")
        if not all(isinstance(e, str) and e for e in c.evidence):
            raise ValueError("invalid evidence references")


def _validate_case(case: AgentEvalCase) -> None:
    unique_ids([case.case_id], "case ids")
    if not case.prompt.strip():
        raise ValueError("case prompt must be nonempty")
    e = case.expectation
    for name in ("max_tool_calls", "max_total_tokens"):
        value = getattr(e, name)
        if value is not None and (type(value) is not int or value < 0):
            raise ValueError(f"{name} must be a nonnegative integer")
    for needed, forbidden in (
        (e.required_tools, e.forbidden_tools),
        (e.required_skills, e.forbidden_skills),
    ):
        if set(needed) & set(forbidden):
            raise ValueError("required and forbidden behavior overlaps")
    if case.rubric:
        unique_ids((c.check_id for c in case.rubric), "rubric ids")
    for c in case.rubric:
        number(c.weight, "rubric weight")
        if c.weight <= 0 or not c.description.strip() or type(c.must_pass) is not bool:
            raise ValueError("invalid binary rubric")
    if (
        not case.accepted_outputs
        and not case.rubric
        and not (e.required_artifacts or e.artifact_values)
    ):
        raise ValueError("define an output, artifact or semantic success criterion")
    if any(not isinstance(v, str) or not v.strip() for v in case.accepted_outputs):
        raise ValueError("accepted outputs must be nonempty strings")


def evaluate_captured_run(
    run: CapturedRun,
    case: AgentEvalCase,
    rubric_judge: RunRubricJudge | None = None,
) -> AgentRunEvalResult:
    _validate_case(case)
    if run.prompt != case.prompt:
        raise ValueError("captured prompt differs from case prompt")
    if not run.run_id.strip() or not isinstance(run.output, str):
        raise ValueError("invalid captured run")
    if run.status not in {"completed", "failed", "cancelled", "timeout"}:
        raise ValueError("unknown run status")
    for event in run.trace:
        TraceEvent.from_dict(asdict(event))
    for value in run.usage.values():
        if type(value) is not int or value < 0:
            raise ValueError("usage must contain nonnegative integers")
    event_types = [event.event_type for event in run.trace]
    sequences = [event.sequence for event in run.trace]
    e = case.expectation
    checks: list[StructuredCheck] = []

    def add(cid, category, passed, notes, weight=1.0, must_pass=True, evidence=()):
        checks.append(
            StructuredCheck(
                cid,
                category,
                passed,
                None if passed is None else float(passed),
                notes,
                weight=weight,
                must_pass=must_pass,
                evidence=tuple(evidence),
            )
        )

    add(
        "outcome_success",
        "outcome",
        run.status == "completed",
        f"status={run.status}",
        2,
        evidence=("status",),
    )
    add(
        "trace_order",
        "process",
        _ordered_subsequence(event_types, e.required_event_order),
        f"required subsequence={e.required_event_order}",
        2,
        evidence=tuple(f"trace:{s}" for s in sequences),
    )
    add(
        "trace_sequence",
        "process",
        bool(sequences) and sequences == sorted(set(sequences)),
        "Sequence must be nonempty, strictly increasing and unique; gaps are permitted.",
        1.5,
    )
    add(
        "required_artifacts",
        "outcome",
        all(
            k in run.artifacts and _artifact_present(run.artifacts[k])
            for k in e.required_artifacts
        ),
        f"required={e.required_artifacts}",
        1.5,
        evidence=tuple(
            f"artifact:{k}" for k in e.required_artifacts if k in run.artifacts
        ),
    )
    add(
        "no_failed_event",
        "process",
        not any(t.endswith(".failed") for t in event_types),
        "Strict policy: failed events require review even if the run recovered.",
        1.5,
    )
    if e.artifact_values:
        add(
            "artifact_values",
            "outcome",
            all(
                k in run.artifacts and run.artifacts[k] == v
                for k, v in e.artifact_values.items()
            ),
            "Captured final-state values must match the declared contract.",
            evidence=tuple(
                f"artifact:{k}" for k in e.artifact_values if k in run.artifacts
            ),
        )
    if case.accepted_outputs:
        add(
            "output_contract",
            "outcome",
            run.output.strip() in case.accepted_outputs,
            "Exact output contract; paraphrases need a semantic rubric.",
            evidence=("output",),
        )
    for kind, event_type, key, required, forbidden in (
        ("tool", "tool.started", "tool", e.required_tools, e.forbidden_tools),
        ("skill", "skill.invoked", "skill", e.required_skills, e.forbidden_skills),
    ):
        events = [event for event in run.trace if event.event_type == event_type]
        names = [event.payload.get(key) for event in events]
        known = all(isinstance(name, str) and bool(name.strip()) for name in names)
        evidence = tuple(f"trace:{event.sequence}" for event in events)
        if required:
            verdict = (
                True
                if all(name in names for name in required)
                else (False if known else None)
            )
            add(
                f"required_{kind}s",
                "process",
                verdict,
                f"required={required}; observed={names}",
                evidence=evidence,
            )
        if forbidden:
            verdict = (
                False
                if any(name in forbidden for name in names)
                else (True if known else None)
            )
            add(
                f"forbidden_{kind}s",
                "risk",
                verdict,
                f"forbidden={forbidden}; observed={names}",
                evidence=evidence,
            )
    if e.max_tool_calls is not None:
        calls = event_types.count("tool.started")
        add(
            "tool_efficiency",
            "efficiency",
            calls <= e.max_tool_calls,
            f"tool_calls={calls}; budget={e.max_tool_calls}",
            must_pass=False,
        )
    if e.max_total_tokens is not None:
        tokens = run.usage.get("total_tokens")
        add(
            "token_efficiency",
            "efficiency",
            None if tokens is None else tokens <= e.max_total_tokens,
            f"total_tokens={tokens}; budget={e.max_total_tokens}; missing usage is unknown",
            must_pass=False,
        )
    quality = (rubric_judge or HeuristicRunRubricJudge()).grade(run, case)
    # A plugin may add diagnostics, but cannot omit or weaken the declared rubric.
    for criterion in case.rubric:
        matches = [c for c in quality if c.check_id == criterion.check_id]
        if len(matches) != 1 or (
            matches[0].must_pass,
            matches[0].weight,
            matches[0].category,
        ) != (criterion.must_pass, criterion.weight, "quality"):
            raise ValueError(f"judge violated rubric contract: {criterion.check_id}")
    checks.extend(quality)
    validate_checks(tuple(checks))
    total_weight = sum(c.weight for c in checks)
    score = 100 * sum((c.score or 0) * c.weight for c in checks) / total_weight
    passed = all(c.passed is True for c in checks if c.must_pass)
    return AgentRunEvalResult(
        case.case_id, passed, round(score, 2), tuple(checks), run.run_id
    )


def _metrics(results: tuple[AgentRunEvalResult, ...]) -> dict[str, float]:
    unique_ids((r.case_id for r in results), "case ids")
    values: dict[str, list[StructuredCheck]] = {}
    for result in results:
        validate_checks(result.checks)
        weight = sum(c.weight for c in result.checks)
        if weight <= 0:
            raise ValueError("report has no scored checks")
        expected_score = round(
            100 * sum((c.score or 0) * c.weight for c in result.checks) / weight, 2
        )
        expected_pass = all(c.passed is True for c in result.checks if c.must_pass)
        if (
            type(result.overall_pass) is not bool
            or result.overall_pass != expected_pass
            or result.score != expected_score
        ):
            raise ValueError("result summary does not match checks")
        for c in result.checks:
            if c.category != "diagnostic":
                values.setdefault(c.check_id, []).append(c)
    metrics = {}
    for name, checks in values.items():
        metrics[f"check/{name}"] = sum(c.score or 0 for c in checks) / len(checks)
        metrics[f"unknown/{name}"] = sum(c.passed is None for c in checks) / len(checks)
        metrics[f"applicable/{name}"] = float(len(checks))
    scored = [c for r in results for c in r.checks if c.category != "diagnostic"]
    metrics["overall_score"] = sum(r.score for r in results) / len(results)
    metrics["pass_rate"] = sum(r.overall_pass for r in results) / len(results)
    metrics["unknown_rate"] = sum(c.passed is None for c in scored) / len(scored)
    return dict(sorted(metrics.items()))


def evaluate_agent_runs(
    runs: dict[str, CapturedRun],
    cases: list[AgentEvalCase],
    rubric_judge: RunRubricJudge | None = None,
    *,
    context: EvaluationContext | None = None,
    subject: dict[str, str] | None = None,
) -> AgentEvalReport:
    unique_ids((c.case_id for c in cases), "case ids")
    if set(runs) != {c.case_id for c in cases}:
        raise ValueError(
            "captured runs must match case ids exactly (missing or extra runs)"
        )
    unique_ids((run.run_id for run in runs.values()), "run ids")
    judge = rubric_judge or HeuristicRunRubricJudge()
    config = judge.configuration
    if not isinstance(config, dict) or not config:
        raise ValueError(
            "judge configuration must identify its implementation, model and settings"
        )
    context = context or EvaluationContext()
    if not context.environment.strip() or context.protocol != "agent-eval-v2":
        raise ValueError("invalid environment or evaluation protocol")
    manifest = json.loads(
        json.dumps(
            {
                "cases": [asdict(c) for c in sorted(cases, key=lambda c: c.case_id)],
                "context": asdict(context),
                "judge": config,
            },
            allow_nan=False,
        )
    )
    identity = fingerprint(manifest)
    results = tuple(
        evaluate_captured_run(runs[c.case_id], c, judge)
        for c in sorted(cases, key=lambda c: c.case_id)
    )
    return AgentEvalReport(
        results, _metrics(results), identity, dict(subject or {}), manifest
    )


def compare_agent_eval_reports(
    baseline: AgentEvalReport,
    candidate: AgentEvalReport,
    max_overall_drop: float = 0.0,
    max_metric_drops: dict[str, float] | None = None,
    minimum_metrics: dict[str, float] | None = None,
) -> AgentEvalRegressionReport:
    number(max_overall_drop, "max_overall_drop", maximum=100)
    for report in (baseline, candidate):
        unique_ids((r.case_id for r in report.results), "case ids")
        if (
            not report.evaluation_manifest
            or fingerprint(report.evaluation_manifest) != report.evaluation_fingerprint
        ):
            raise ValueError("evaluation manifest is missing or altered")
        if _metrics(report.results) != report.metrics:
            raise ValueError("report metrics do not match checks")
    if {r.case_id for r in baseline.results} != {r.case_id for r in candidate.results}:
        raise ValueError("baseline 与 candidate 必须使用相同 case_id 集合")
    if (
        not baseline.evaluation_fingerprint
        or baseline.evaluation_fingerprint != candidate.evaluation_fingerprint
    ):
        raise ValueError(
            "evaluation conditions differ: cases, rubric, judge or environment"
        )
    for left, right in zip(
        sorted(baseline.results, key=lambda r: r.case_id),
        sorted(candidate.results, key=lambda r: r.case_id),
    ):

        def shape(result):
            return sorted(
                (c.check_id, c.category, c.weight, c.must_pass, c.source)
                for c in result.checks
            )

        if shape(left) != shape(right):
            raise ValueError("judge check contracts differ between runs")
    names = baseline.metrics.keys()
    if names != candidate.metrics.keys():
        raise ValueError("metric sets differ")
    deltas = {n: candidate.metrics[n] - baseline.metrics[n] for n in names}
    reasons = []
    for result in candidate.results:
        for c in result.checks:
            if c.must_pass and c.passed is not True:
                reasons.append(f"{result.case_id}/{c.check_id}: {c.verdict}")
    if deltas["overall_score"] < -max_overall_drop:
        reasons.append("overall_score dropped beyond tolerance")
    for name, limit in (max_metric_drops or {}).items():
        number(limit, f"drop limit {name}")
        if (
            name not in names
            or name.startswith(("unknown/", "applicable/"))
            or name == "unknown_rate"
        ):
            raise ValueError(f"unknown or non-higher-is-better metric: {name}")
        if deltas[name] < -limit:
            reasons.append(f"{name}: regression exceeds {limit}")
    for name, floor in (minimum_metrics or {}).items():
        number(floor, f"minimum {name}")
        if (
            name not in names
            or name.startswith(("unknown/", "applicable/"))
            or name == "unknown_rate"
        ):
            raise ValueError(f"unknown or non-higher-is-better metric: {name}")
        if candidate.metrics[name] < floor:
            reasons.append(f"{name}: below minimum {floor}")
    return AgentEvalRegressionReport(
        baseline, candidate, dict(sorted(deltas.items())), not reasons, tuple(reasons)
    )


def result_as_structured_json(result: AgentRunEvalResult) -> str:
    payload = asdict(result)
    payload["schema_version"] = 2
    payload["checks"] = [
        dict(
            id=c.check_id,
            category=c.category,
            verdict=c.verdict,
            score=c.score,
            notes=c.notes,
            source=c.source,
            weight=c.weight,
            must_pass=c.must_pass,
            evidence=list(c.evidence),
        )
        for c in result.checks
    ]
    return json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False)
