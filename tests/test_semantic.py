from dataclasses import replace

import pytest

from llm_eval_lab import (
    AgentEvalCase,
    CapturedRun,
    Label,
    StructuredRunJudge,
    TraceEvent,
    compare_labels,
)
from llm_eval_lab.trace_eval import compare_agent_eval_reports, evaluate_agent_runs


class Client:
    def __init__(self, payloads):
        self.payloads, self.calls = payloads, []

    def judge(self, request, validation_errors):
        self.calls.append((request, validation_errors))
        return self.payloads[min(len(self.calls) - 1, len(self.payloads) - 1)]


def judge(client, **kwargs):
    return StructuredRunJudge(
        client, judge_id="synthetic-stub", model="none", prompt_version="v1", **kwargs
    )


def inputs():
    return CapturedRun(
        "r", "q", "no", "completed", (TraceEvent(1, "run.completed"),)
    ), AgentEvalCase("c", "q", "yes")


def payload(verdict="fail", refs=None):
    return {
        "answer_supported": {
            "verdict": verdict,
            "reason": "Contradicts the synthetic reference.",
            "evidence": refs if refs is not None else ["output", "reference"],
        }
    }


def test_structured_judge_receives_evidence_and_retries_invalid_reference():
    client = Client([payload(refs=["artifact:invented"]), payload()])
    run, case = inputs()
    checks = judge(client).grade(run, case)
    assert checks[0].verdict == "fail"
    assert client.calls[1][1]
    assert client.calls[0][0]["evidence"]["reference"] == "yes"


@pytest.mark.parametrize(
    "bad",
    [
        {},
        payload(refs=[]),
        payload(refs=["status"]),
        payload(verdict="maybe"),
        payload(refs=[{}]),
    ],
)
def test_invalid_judge_output_becomes_unknown(bad):
    run, case = inputs()
    checks = judge(Client([bad])).grade(run, case)
    assert checks[0].verdict == "unknown"


def test_judge_settings_are_part_of_comparison_identity():
    run, case = inputs()
    left = evaluate_agent_runs(
        {"c": run}, [case], judge(Client([payload()]), settings={"temperature": 0})
    )
    right = evaluate_agent_runs(
        {"c": run}, [case], judge(Client([payload()]), settings={"temperature": 1})
    )
    with pytest.raises(ValueError, match="conditions differ"):
        compare_agent_eval_reports(left, right)


def test_calibration_exposes_false_passes_and_unknown_coverage():
    refs = [Label("a", "quality", "fail"), Label("b", "quality", "pass")]
    actual = [Label("a", "quality", "pass"), Label("b", "quality", "unknown")]
    result = compare_labels(refs, actual)
    assert result["agreement_on_known"] == 0
    assert result["jointly_known_coverage"] == 0.5
    assert result["false_passes"] == [{"case_id": "a", "check_id": "quality"}]
    assert (
        compare_labels([Label("a", "q", "unknown")], [Label("a", "q", "unknown")])[
            "agreement_on_known"
        ]
        is None
    )


def test_calibration_rejects_duplicates_and_misalignment():
    label = Label("a", "q", "pass")
    with pytest.raises(ValueError):
        compare_labels([label, label], [label])
    with pytest.raises(ValueError):
        compare_labels([label], [replace(label, case_id="b")])


def test_gateway_judge_transport_and_fixed_provider_policy():
    import json

    from llm_eval_lab import GatewayJudgeClient, ResilientModelGateway

    class SyntheticProvider:
        name = "synthetic"

        def complete(self, prompt, model):
            request = json.loads(prompt)
            assert request["request"]["evidence"]["output"] == "no"
            return json.dumps(payload())

    with ResilientModelGateway([SyntheticProvider()]) as gateway:
        client = GatewayJudgeClient(gateway, "synthetic/model")
        grader = StructuredRunJudge(
            client, judge_id="synthetic", model="synthetic/model", prompt_version="v1"
        )
        run, case = inputs()
        assert grader.grade(run, case)[0].verdict == "fail"
        assert len(client.completions) == 1
        with pytest.raises(ValueError, match="model"):
            StructuredRunJudge(client, judge_id="x", model="wrong", prompt_version="v1")
    with ResilientModelGateway([SyntheticProvider(), SyntheticProvider()]) as gateway:
        with pytest.raises(ValueError, match="one provider"):
            GatewayJudgeClient(gateway, "synthetic/model")
