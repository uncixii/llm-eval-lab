from __future__ import annotations

from .models import Rubric, RubricCriterion


def default_rubric() -> Rubric:
    return Rubric(
        name="qa_quality",
        criteria=(
            RubricCriterion("relevance", "回答是否直接覆盖问题和 reference 的核心信息", 0.4),
            RubricCriterion("groundedness", "回答是否只使用给定 reference 能支持的信息", 0.4),
            RubricCriterion("format", "回答是否满足结构或引用要求", 0.2),
        ),
    )

