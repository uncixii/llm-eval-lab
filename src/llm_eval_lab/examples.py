from __future__ import annotations

from .models import EvalCase
from .regression import compare_reports, evaluate_dataset
from .rubric import default_rubric


def main() -> None:
    cases = [
        EvalCase("case-1", "热管理要求是什么？", "需要监控冷却液温度并执行阈值保护。"),
        EvalCase("case-2", "异响如何排查？", "先确认振动频段，再检查轴承间隙。"),
    ]
    baseline = {"case-1": "需要监控温度。", "case-2": "先确认振动频段。"}
    candidate = {
        "case-1": "需要监控冷却液温度并执行阈值保护。依据 synthetic 规范。",
        "case-2": "先确认振动频段，再检查轴承间隙。[synthetic]",
    }
    rubric = default_rubric()
    baseline_report = evaluate_dataset(cases, baseline, rubric)
    candidate_report = evaluate_dataset(cases, candidate, rubric)
    report = compare_reports(baseline_report, candidate_report)
    print(f"baseline={baseline_report.aggregate:.3f}")
    print(f"candidate={candidate_report.aggregate:.3f}")
    print(f"delta={report.delta:.3f}, passed={report.passed}")


if __name__ == "__main__":
    main()
