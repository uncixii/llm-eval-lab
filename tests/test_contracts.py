import json
from dataclasses import replace
from pathlib import Path

import jsonschema
import pytest

from llm_eval_lab import (
    AgentEvalCase,
    CapturedRun,
    EvaluationContext,
    TraceEvent,
    TraceExpectation,
    compare_agent_eval_reports,
    evaluate_agent_runs,
    evaluate_captured_run,
    parse_trace_jsonl,
    result_as_structured_json,
    summarize_trials,
)


def case():
    return AgentEvalCase(
        "c",
        "synthetic question",
        "yes",
        TraceExpectation(max_tool_calls=1, max_total_tokens=10),
        accepted_outputs=("yes",),
        rubric=(),
    )


def run():
    return CapturedRun(
        "r",
        "synthetic question",
        "yes",
        "completed",
        (TraceEvent(1, "tool.started", {"tool": "read"}),),
        {},
        {"total_tokens": 5},
    )


def report(c=None, r=None, **kwargs):
    return evaluate_agent_runs({"c": r or run()}, [c or case()], **kwargs)


def test_critical_failure_cannot_be_offset_by_efficiency():
    baseline = replace(
        run(),
        trace=tuple(TraceEvent(i, "tool.started") for i in range(100)),
        usage={"total_tokens": 1000},
    )
    candidate = replace(
        run(), trace=(TraceEvent(1, "run.started"), TraceEvent(1, "run.completed"))
    )
    left, right = report(r=baseline), report(r=candidate)
    assert right.metrics["overall_score"] > left.metrics["overall_score"]
    gate = compare_agent_eval_reports(left, right)
    assert not gate.passed
    assert "c/trace_sequence: fail" in gate.reasons


def test_default_semantics_abstain_on_negation_and_irrelevant_artifacts():
    c = AgentEvalCase("c", "synthetic question", "需要执行阈值保护")
    r = replace(run(), output="不需要执行阈值保护", artifacts={"irrelevant": "hello"})
    result = evaluate_captured_run(r, c)
    assert not result.overall_pass
    check = next(c for c in result.checks if c.check_id == "answer_supported")
    assert check.verdict == "unknown" and check.score is None


def test_missing_usage_is_unknown_not_zero():
    r = report(r=replace(run(), usage={}))
    assert r.metrics["unknown/token_efficiency"] == 1
    assert r.metrics["check/token_efficiency"] == 0


@pytest.mark.parametrize(
    "mutation",
    [
        lambda c: replace(c, reference="different"),
        lambda c: replace(c, accepted_outputs=("yes", "certainly")),
        lambda c: replace(c, expectation=TraceExpectation(max_tool_calls=5)),
    ],
)
def test_same_id_does_not_allow_different_contract(mutation):
    with pytest.raises(ValueError, match="conditions differ"):
        compare_agent_eval_reports(report(), report(c=mutation(case())))


def test_different_environment_rejected():
    with pytest.raises(ValueError, match="conditions differ"):
        compare_agent_eval_reports(
            report(), report(context=EvaluationContext(environment="another"))
        )


@pytest.mark.parametrize(
    "cases,runs",
    [
        ([], {}),
        ([case(), case()], {"c": run()}),
        ([case()], {}),
        ([case()], {"c": run(), "extra": run()}),
    ],
)
def test_invalid_dataset_rejected(cases, runs):
    with pytest.raises(ValueError):
        evaluate_agent_runs(runs, cases)


def test_unknown_metric_and_nan_tolerance_rejected():
    with pytest.raises(ValueError, match="metric"):
        compare_agent_eval_reports(report(), report(), max_metric_drops={"typo": 0})
    with pytest.raises(ValueError):
        compare_agent_eval_reports(report(), report(), max_overall_drop=float("nan"))


def test_tool_identity_and_negative_skill_control():
    c = replace(
        case(),
        expectation=TraceExpectation(
            required_tools=("read",), forbidden_skills=("create-project",)
        ),
    )
    good = run()
    assert evaluate_captured_run(good, c).overall_pass
    bad = replace(
        good,
        trace=good.trace
        + (TraceEvent(2, "skill.invoked", {"skill": "create-project"}),),
    )
    assert not evaluate_captured_run(bad, c).overall_pass
    missing_name = replace(good, trace=(TraceEvent(1, "tool.started"),))
    checks = evaluate_captured_run(missing_name, c).checks
    assert (
        next(c for c in checks if c.check_id == "required_tools").verdict == "unknown"
    )


def test_final_state_contract_rejects_claimed_success():
    c = replace(case(), expectation=TraceExpectation(artifact_values={"saved": True}))
    assert not evaluate_captured_run(
        replace(run(), artifacts={"saved": False}), c
    ).overall_pass


@pytest.mark.parametrize(
    "text", ["{}", "[]", '{"sequence":true,"type":"x"}', '{"type":"x"}', "not json"]
)
def test_trace_parser_rejects_malformed_lines(text):
    with pytest.raises(ValueError, match="line 1"):
        parse_trace_jsonl(text)


def test_schema_validates_pass_and_unknown():
    schema = json.loads(
        (Path(__file__).parents[1] / "evals/rubric_result.schema.json").read_text()
    )
    for c in [case(), AgentEvalCase("c", run().prompt, "reference")]:
        jsonschema.validate(
            json.loads(result_as_structured_json(evaluate_captured_run(run(), c))),
            schema,
        )


def test_applicability_is_not_assumed_identical_across_cases():
    c2 = replace(case(), case_id="second", expectation=TraceExpectation())
    r = evaluate_agent_runs(
        {"c": run(), "second": replace(run(), run_id="r2")}, [case(), c2]
    )
    assert r.metrics["applicable/token_efficiency"] == 1


def test_trial_summary_keeps_flakiness_and_rejects_reused_runs():
    good = report()
    bad = report(r=replace(run(), run_id="r2", output="no"))
    summary = summarize_trials([good, bad])
    assert summary["per_case"]["c"]["pass_rate"] == 0.5
    assert not summary["per_case"]["c"]["all_trials_passed"]
    with pytest.raises(ValueError, match="duplicate"):
        summarize_trials([good, good])


def test_unknown_required_quality_blocks_gate_even_without_score_drop():
    c = AgentEvalCase("c", run().prompt, "yes")
    unknown = report(c=c)
    gate = compare_agent_eval_reports(unknown, unknown)
    assert not gate.passed
    assert "c/answer_supported: unknown" in gate.reasons


def test_manifest_and_result_summary_mutation_are_rejected():
    original = report()
    altered = replace(original, evaluation_manifest={"changed": True})
    with pytest.raises(ValueError, match="manifest"):
        compare_agent_eval_reports(original, altered)
    altered = replace(
        original, results=(replace(original.results[0], overall_pass=False),)
    )
    with pytest.raises(ValueError, match="summary"):
        compare_agent_eval_reports(original, altered)


def test_judge_cannot_weaken_required_rubric():
    from llm_eval_lab.models import StructuredCheck

    class WeakJudge:
        def grade(self, run, case):
            return (
                StructuredCheck(
                    "answer_supported",
                    "quality",
                    True,
                    1,
                    "unsubstantiated",
                    must_pass=False,
                ),
            )

    with pytest.raises(ValueError, match="rubric contract"):
        evaluate_captured_run(run(), AgentEvalCase("c", run().prompt), WeakJudge())
