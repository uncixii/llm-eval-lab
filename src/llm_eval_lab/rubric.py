from __future__ import annotations

from .models import Rubric, RubricCriterion


def default_rubric() -> Rubric:
    """Cheap diagnostics, deliberately not labelled relevance or groundedness."""
    return Rubric(
        "lexical_diagnostics",
        (
            RubricCriterion(
                "reference_overlap",
                "Reference token coverage; not semantic correctness",
                0.8,
            ),
            RubricCriterion("nonempty_output", "Output is nonempty", 0.2),
        ),
    )
