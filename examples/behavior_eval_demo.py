"""Replay synthetic good/bad/unknown cases; no model calls or agent execution."""

from pathlib import Path

from llm_eval_lab import Label, compare_labels, evaluate_captured_run
from llm_eval_lab.fixtures import load_synthetic_cases


def main():
    path = Path(__file__).resolve().parents[1] / "evals/synthetic_behavior_cases.json"
    for case, run, _ in load_synthetic_cases(path):
        result = evaluate_captured_run(run, case)
        issues = [
            f"{c.check_id}={c.verdict}"
            for c in result.checks
            if c.category != "diagnostic" and c.passed is not True
        ]
        print(
            case.case_id,
            f"pass={result.overall_pass}",
            ", ".join(issues) or "contract satisfied",
        )
    # Hand-authored labels illustrate the audit API, not measured human/model agreement.
    gold = [
        Label("synthetic-negation", "supported", "fail"),
        Label("synthetic-ambiguous", "supported", "unknown"),
    ]
    naive = [
        Label("synthetic-negation", "supported", "pass"),
        Label("synthetic-ambiguous", "supported", "unknown"),
    ]
    print("Synthetic calibration illustration:", compare_labels(gold, naive))


if __name__ == "__main__":
    main()
