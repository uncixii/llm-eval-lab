from __future__ import annotations

from collections.abc import Mapping, Sequence

from .judge import HeuristicJudge
from .models import EvalCase, EvalReport, EvalResult, RegressionReport, Rubric


def _weighted_average(scores, rubric: Rubric) -> float:
    weights = {criterion.name: criterion.weight for criterion in rubric.criteria}
    total_weight = sum(weights.values()) or 1.0
    return sum(score.score * weights.get(score.criterion, 0.0) for score in scores) / total_weight


def evaluate_dataset(
    cases: Sequence[EvalCase],
    responses: Mapping[str, str],
    rubric: Rubric,
    judge: HeuristicJudge | None = None,
) -> EvalReport:
    judge = judge or HeuristicJudge()
    results = []
    for case in cases:
        scores = judge.evaluate(case, responses.get(case.case_id, ""), rubric)
        results.append(EvalResult(case.case_id, scores, _weighted_average(scores, rubric)))
    aggregate = sum(result.overall for result in results) / len(results) if results else 0.0
    return EvalReport(tuple(results), aggregate)


def compare_reports(
    baseline: EvalReport,
    candidate: EvalReport,
    threshold: float = -0.02,
    max_metric_drops: dict[str, float] | None = None,
) -> RegressionReport:
    baseline_ids = {result.case_id for result in baseline.results}
    candidate_ids = {result.case_id for result in candidate.results}
    if baseline_ids != candidate_ids:
        raise ValueError("baseline 与 candidate 必须使用相同 case_id 集合")
    delta = candidate.aggregate - baseline.aggregate
    criteria = {
        score.criterion
        for report in (baseline, candidate)
        for result in report.results
        for score in result.scores
    }

    def criterion_average(report: EvalReport, criterion: str) -> float:
        values = [
            score.score
            for result in report.results
            for score in result.scores
            if score.criterion == criterion
        ]
        return sum(values) / len(values) if values else 0.0

    metric_deltas = {
        criterion: criterion_average(candidate, criterion) - criterion_average(baseline, criterion)
        for criterion in criteria
    }
    metric_deltas["aggregate"] = delta
    passed = delta >= threshold
    for name, allowed_drop in (max_metric_drops or {}).items():
        passed = passed and metric_deltas.get(name, 0.0) >= -allowed_drop
    return RegressionReport(
        baseline, candidate, delta, passed, threshold, dict(sorted(metric_deltas.items()))
    )
