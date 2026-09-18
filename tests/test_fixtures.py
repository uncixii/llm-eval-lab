from pathlib import Path

import pytest

from llm_eval_lab import evaluate_captured_run
from llm_eval_lab.fixtures import load_synthetic_cases


@pytest.mark.parametrize(
    "case,run,expected",
    load_synthetic_cases(
        Path(__file__).parents[1] / "evals/synthetic_behavior_cases.json"
    ),
)
def test_synthetic_behavior_contracts(case, run, expected):
    result = evaluate_captured_run(run, case)
    assert result.overall_pass == expected["overall_pass"]
    assert sorted(c.check_id for c in result.checks if c.passed is False) == sorted(
        expected["failed_checks"]
    )
    assert sorted(
        c.check_id
        for c in result.checks
        if c.passed is None and c.category != "diagnostic"
    ) == sorted(expected["unknown_checks"])
