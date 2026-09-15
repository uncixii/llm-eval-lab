from __future__ import annotations

from typing import Any, Protocol

from .checks import token_coverage
from .models import EvalCase, JudgeScore, Rubric


class JudgeClient(Protocol):
    def judge(
        self,
        prompt: str,
        response: str,
        reference: str,
        rubric: Rubric,
        validation_errors: tuple[str, ...] = (),
    ) -> dict[str, Any]: ...


class JudgeOutputError(ValueError):
    pass


class HeuristicJudge:
    """可重复的 deterministic judge，作为 LLM-as-a-Judge 的 baseline。"""

    def evaluate(self, case: EvalCase, response: str, rubric: Rubric) -> tuple[JudgeScore, ...]:
        coverage = token_coverage(response, case.reference)
        return (
            JudgeScore("relevance", coverage, f"reference token coverage={coverage:.3f}"),
            JudgeScore(
                "groundedness",
                min(1.0, coverage + 0.1) if coverage else 0.0,
                "使用 reference token overlap 作为保守 proxy",
            ),
            JudgeScore("format", 1.0 if response.strip() else 0.0, "非空回答检查"),
        )


class LLMAsJudge:
    """LLM judge adapter；外部 client 只需返回 criterion -> score/reason。"""

    def __init__(self, client: JudgeClient, max_retries: int = 1) -> None:
        self.client = client
        self.max_retries = max_retries

    def evaluate(self, case: EvalCase, response: str, rubric: Rubric) -> tuple[JudgeScore, ...]:
        last_error: JudgeOutputError | None = None
        payload: dict[str, Any] = {}
        validation_errors: tuple[str, ...] = ()
        for _ in range(self.max_retries + 1):
            payload = self.client.judge(
                case.prompt,
                response,
                case.reference,
                rubric,
                validation_errors,
            )
            try:
                self._validate(payload, rubric)
                break
            except JudgeOutputError as exc:
                last_error = exc
                validation_errors = (str(exc),)
        else:
            raise last_error or JudgeOutputError("LLM judge 输出无效")
        return tuple(
            JudgeScore(
                criterion=criterion.name,
                score=float(payload.get(criterion.name, {}).get("score", 0.0)),
                reason=str(payload.get(criterion.name, {}).get("reason", "")),
                source="llm",
            )
            for criterion in rubric.criteria
        )

    @staticmethod
    def _validate(payload: dict[str, Any], rubric: Rubric) -> None:
        if not isinstance(payload, dict):
            raise JudgeOutputError("judge 输出必须是 object")
        for criterion in rubric.criteria:
            item = payload.get(criterion.name)
            if not isinstance(item, dict):
                raise JudgeOutputError(f"缺少 criterion: {criterion.name}")
            score = item.get("score")
            if not isinstance(score, (int, float)) or isinstance(score, bool):
                raise JudgeOutputError(f"{criterion.name}.score 必须是数字")
            if not 0.0 <= float(score) <= 1.0:
                raise JudgeOutputError(f"{criterion.name}.score 必须在 0 到 1 之间")
            if not isinstance(item.get("reason"), str):
                raise JudgeOutputError(f"{criterion.name}.reason 必须是字符串")
