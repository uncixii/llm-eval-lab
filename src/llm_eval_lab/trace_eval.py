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
    StructuredCheck,
    TraceEvent,
)


class RunRubricJudge(Protocol):
    def grade(self, run: CapturedRun, case: AgentEvalCase) -> tuple[StructuredCheck, ...]: ...


class HeuristicRunRubricJudge:
    """本地 deterministic substitute；生产环境可替换为 structured LLM judge。"""

    def grade(self, run: CapturedRun, case: AgentEvalCase) -> tuple[StructuredCheck, ...]:
        relevance = token_coverage(run.output, case.reference) if case.reference else float(bool(run.output))
        has_evidence = bool(run.artifacts) and bool(run.output.strip())
        return (
            StructuredCheck(
                "output_relevance",
                "quality",
                relevance >= 0.5,
                relevance,
                f"reference token coverage={relevance:.3f}",
                source="rubric",
                weight=1.5,
                must_pass=True,
            ),
            StructuredCheck(
                "artifact_grounding",
                "quality",
                has_evidence,
                float(has_evidence),
                "输出应有 captured artifact 支撑。",
                source="rubric",
                weight=1.5,
                must_pass=True,
            ),
        )


def _artifact_present(value: object) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple, dict, set)):
        return bool(value)
    return True


def parse_trace_jsonl(lines: str | Iterable[str]) -> tuple[TraceEvent, ...]:
    source = lines.splitlines() if isinstance(lines, str) else lines
    events = []
    for line in source:
        if not line.strip():
            continue
        payload = json.loads(line)
        if "event_type" in payload or "type" in payload:
            events.append(TraceEvent.from_dict(payload))
    return tuple(events)


def _ordered_subsequence(actual: list[str], expected: tuple[str, ...]) -> bool:
    position = 0
    for event_type in actual:
        if position < len(expected) and event_type == expected[position]:
            position += 1
    return position == len(expected)


def evaluate_captured_run(
    run: CapturedRun,
    case: AgentEvalCase,
    rubric_judge: RunRubricJudge | None = None,
) -> AgentRunEvalResult:
    event_types = [event.event_type for event in run.trace]
    sequences = [event.sequence for event in run.trace]
    sequence_ok = (
        bool(sequences)
        and sequences == sorted(sequences)
        and len(sequences) == len(set(sequences))
    )
    expectation = case.expectation
    event_order_ok = _ordered_subsequence(event_types, expectation.required_event_order)
    artifacts_ok = all(
        name in run.artifacts and _artifact_present(run.artifacts[name])
        for name in expectation.required_artifacts
    )
    no_error = not any(event_type.endswith(".failed") for event_type in event_types)
    tool_calls = sum(event_type == "tool.started" for event_type in event_types)
    tool_efficiency = expectation.max_tool_calls is None or tool_calls <= expectation.max_tool_calls
    total_tokens = int(run.usage.get("total_tokens", 0))
    token_efficiency = expectation.max_total_tokens is None or total_tokens <= expectation.max_total_tokens
    checks = (
        StructuredCheck("outcome_success", "outcome", run.status == "completed", float(run.status == "completed"), f"status={run.status}", weight=2.0, must_pass=True),
        StructuredCheck("trace_order", "process", event_order_ok, float(event_order_ok), "required events 应按顺序出现。", weight=2.0, must_pass=True),
        StructuredCheck("trace_sequence", "process", sequence_ok, float(sequence_ok), "sequence 必须严格递增且不可重复。", weight=1.5, must_pass=True),
        StructuredCheck("required_artifacts", "outcome", artifacts_ok, float(artifacts_ok), f"required={expectation.required_artifacts}", weight=1.5, must_pass=True),
        StructuredCheck("no_failed_event", "process", no_error, float(no_error), "trace 中不应出现 failed event。", weight=1.5, must_pass=True),
        StructuredCheck("tool_efficiency", "efficiency", tool_efficiency, 1.0 if tool_efficiency else expectation.max_tool_calls / max(tool_calls, 1), f"tool_calls={tool_calls}", weight=1.0),
        StructuredCheck("token_efficiency", "efficiency", token_efficiency, 1.0 if token_efficiency else expectation.max_total_tokens / max(total_tokens, 1), f"total_tokens={total_tokens}", weight=1.0),
    ) + (rubric_judge or HeuristicRunRubricJudge()).grade(run, case)
    total_weight = sum(check.weight for check in checks) or 1.0
    score = 100.0 * sum(check.score * check.weight for check in checks) / total_weight
    overall_pass = all(check.passed for check in checks if check.must_pass)
    return AgentRunEvalResult(case.case_id, overall_pass, round(score, 2), checks)


def evaluate_agent_runs(
    runs: dict[str, CapturedRun],
    cases: list[AgentEvalCase],
    rubric_judge: RunRubricJudge | None = None,
) -> AgentEvalReport:
    missing = [case.case_id for case in cases if case.case_id not in runs]
    if missing:
        raise ValueError(f"缺少 captured runs: {missing}")
    results = tuple(
        evaluate_captured_run(runs[case.case_id], case, rubric_judge)
        for case in cases
    )
    if not results:
        return AgentEvalReport((), {})
    check_ids = {check.check_id for result in results for check in result.checks}
    metrics = {
        check_id: sum(
            next(check.score for check in result.checks if check.check_id == check_id)
            for result in results
        ) / len(results)
        for check_id in check_ids
    }
    metrics["overall_score"] = sum(result.score for result in results) / len(results)
    metrics["pass_rate"] = sum(result.overall_pass for result in results) / len(results)
    return AgentEvalReport(results, dict(sorted(metrics.items())))


def compare_agent_eval_reports(
    baseline: AgentEvalReport,
    candidate: AgentEvalReport,
    max_overall_drop: float = 0.0,
    max_metric_drops: dict[str, float] | None = None,
    minimum_metrics: dict[str, float] | None = None,
) -> AgentEvalRegressionReport:
    baseline_ids = {result.case_id for result in baseline.results}
    candidate_ids = {result.case_id for result in candidate.results}
    if baseline_ids != candidate_ids:
        raise ValueError("baseline 与 candidate 必须使用相同 case_id 集合")
    names = baseline.metrics.keys() | candidate.metrics.keys()
    deltas = {
        name: candidate.metrics.get(name, 0.0) - baseline.metrics.get(name, 0.0)
        for name in names
    }
    passed = deltas.get("overall_score", 0.0) >= -max_overall_drop
    for name, allowed_drop in (max_metric_drops or {}).items():
        passed = passed and deltas.get(name, 0.0) >= -allowed_drop
    for name, floor in (minimum_metrics or {}).items():
        passed = passed and candidate.metrics.get(name, 0.0) >= floor
    return AgentEvalRegressionReport(
        baseline,
        candidate,
        dict(sorted(deltas.items())),
        passed,
    )


def result_as_structured_json(result: AgentRunEvalResult) -> str:
    payload = asdict(result)
    payload["checks"] = [
        {
            "id": check["check_id"],
            "category": check["category"],
            "pass": check["passed"],
            "score": check["score"],
            "notes": check["notes"],
            "source": check["source"],
        }
        for check in payload["checks"]
    ]
    return json.dumps(payload, ensure_ascii=False, indent=2)
