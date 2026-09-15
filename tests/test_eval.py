from llm_eval_lab.models import EvalCase
from llm_eval_lab.judge import LLMAsJudge
from llm_eval_lab.regression import compare_reports, evaluate_dataset
from llm_eval_lab.rubric import default_rubric


def test_candidate_regression_gate_passes_when_quality_improves() -> None:
    cases = [EvalCase("1", "问题", "答案包含关键指标。")]
    rubric = default_rubric()
    baseline = evaluate_dataset(cases, {"1": "答案。"}, rubric)
    candidate = evaluate_dataset(cases, {"1": "答案包含关键指标。[依据]"}, rubric)
    report = compare_reports(baseline, candidate)
    assert report.passed
    assert report.delta > 0


def test_empty_response_is_penalized() -> None:
    case = EvalCase("1", "问题", "参考答案")
    report = evaluate_dataset([case], {"1": ""}, default_rubric())
    assert report.aggregate == 0.0


def test_llm_judge_retries_invalid_structured_output() -> None:
    class Client:
        def __init__(self) -> None:
            self.calls = 0
            self.references = []
            self.validation_errors = []

        def judge(self, prompt, response, reference, rubric, validation_errors=()):
            self.calls += 1
            self.references.append(reference)
            self.validation_errors.append(validation_errors)
            if self.calls == 1:
                return {"relevance": {"score": 2.0, "reason": "invalid"}}
            return {
                criterion.name: {"score": 0.8, "reason": "structured"}
                for criterion in rubric.criteria
            }

    client = Client()
    scores = LLMAsJudge(client, max_retries=1).evaluate(
        EvalCase("case", "prompt", "reference"),
        "response",
        default_rubric(),
    )
    assert client.calls == 2
    assert client.references == ["reference", "reference"]
    assert client.validation_errors[1]
    assert all(score.score == 0.8 for score in scores)


def test_regression_rejects_mismatched_case_sets() -> None:
    rubric = default_rubric()
    left = evaluate_dataset([EvalCase("left", "q", "a")], {"left": "a"}, rubric)
    right = evaluate_dataset([EvalCase("right", "q", "a")], {"right": "a"}, rubric)
    try:
        compare_reports(left, right)
    except ValueError as exc:
        assert "相同 case_id" in str(exc)
    else:
        raise AssertionError("不同 eval case 集合不可直接比较")
