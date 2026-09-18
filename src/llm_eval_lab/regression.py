from __future__ import annotations

from collections.abc import Mapping, Sequence

from .judge import HeuristicJudge
from .models import EvalCase, EvalReport, EvalResult, RegressionReport, Rubric
from .validation import fingerprint, number, unique_ids


def _weighted_average(scores, rubric: Rubric) -> float:
    weights = {criterion.name: criterion.weight for criterion in rubric.criteria}
    total_weight = sum(weights.values()) or 1.0
    return (
        sum(score.score * weights.get(score.criterion, 0.0) for score in scores)
        / total_weight
    )


def evaluate_dataset(
    cases: Sequence[EvalCase],
    responses: Mapping[str, str],
    rubric: Rubric,
    judge: HeuristicJudge | None = None,
    *,
    judge_config: dict | None = None,
) -> EvalReport:
    unique_ids((c.case_id for c in cases), "case ids")
    unique_ids((c.name for c in rubric.criteria), "rubric criteria")
    for criterion in rubric.criteria:
        number(criterion.weight, "rubric weight")
    if sum(c.weight for c in rubric.criteria) <= 0:
        raise ValueError("rubric must have positive total weight")
    if set(responses) != {c.case_id for c in cases}:
        raise ValueError("responses must match case ids exactly")
    if judge is not None and not isinstance(judge, HeuristicJudge) and not judge_config:
        raise ValueError("custom judge requires explicit judge_config")
    judge = judge or HeuristicJudge()
    identity = fingerprint(
        {
            "cases": sorted(cases, key=lambda c: c.case_id),
            "rubric": rubric,
            "judge": judge_config or {"kind": "lexical-diagnostic-v2"},
        }
    )
    results = []
    for case in cases:
        scores = judge.evaluate(case, responses.get(case.case_id, ""), rubric)
        if {s.criterion for s in scores} != {c.name for c in rubric.criteria} or len(
            scores
        ) != len(rubric.criteria):
            raise ValueError("judge criteria differ from rubric")
        for score in scores:
            number(score.score, "judge score", maximum=1)
        results.append(
            EvalResult(case.case_id, scores, _weighted_average(scores, rubric))
        )
    aggregate = (
        sum(result.overall for result in results) / len(results) if results else 0.0
    )
    return EvalReport(tuple(results), aggregate, identity)


def compare_reports(
    baseline: EvalReport,
    candidate: EvalReport,
    threshold: float = -0.02,
    max_metric_drops: dict[str, float] | None = None,
) -> RegressionReport:
    for report in (baseline, candidate):
        unique_ids((r.case_id for r in report.results), "case ids")
    if (
        not baseline.evaluation_fingerprint
        or baseline.evaluation_fingerprint != candidate.evaluation_fingerprint
    ):
        raise ValueError(
            "baseline 与 candidate 必须使用相同 case_id、rubric、judge 配置"
        )
    if not isinstance(threshold, (float, int)) or not -1 <= threshold <= 1:
        raise ValueError("invalid threshold")
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
        criterion: criterion_average(candidate, criterion)
        - criterion_average(baseline, criterion)
        for criterion in criteria
    }
    metric_deltas["aggregate"] = delta
    passed = delta >= threshold
    for name, allowed_drop in (max_metric_drops or {}).items():
        number(allowed_drop, "allowed_drop", maximum=1)
        if name not in metric_deltas:
            raise ValueError(f"unknown metric: {name}")
        passed = passed and metric_deltas[name] >= -allowed_drop
    return RegressionReport(
        baseline,
        candidate,
        delta,
        passed,
        threshold,
        dict(sorted(metric_deltas.items())),
    )
