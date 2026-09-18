"""Descriptive repeated-trial reporting without claiming statistical significance."""

from __future__ import annotations

from .models import AgentEvalReport
from .trace_eval import _metrics
from .validation import fingerprint, unique_ids


def summarize_trials(reports: list[AgentEvalReport]) -> dict:
    if not reports:
        raise ValueError("at least one trial is required")
    identity = reports[0].evaluation_fingerprint
    subject = reports[0].subject
    if not identity or any(
        r.evaluation_fingerprint != identity or r.subject != subject for r in reports
    ):
        raise ValueError("trials must share evaluation conditions and subject")
    for report in reports:
        if fingerprint(
            report.evaluation_manifest
        ) != report.evaluation_fingerprint or report.metrics != _metrics(
            report.results
        ):
            raise ValueError("invalid trial report")
    unique_ids(
        (r.run_id for report in reports for r in report.results), "trial run ids"
    )
    ids = {r.case_id for r in reports[0].results}
    if any({r.case_id for r in report.results} != ids for report in reports):
        raise ValueError("trial case sets differ")
    per_case = {}
    for cid in sorted(ids):
        results = [
            next(r for r in report.results if r.case_id == cid) for report in reports
        ]
        successes = sum(r.overall_pass for r in results)
        per_case[cid] = dict(
            trials=len(results),
            successes=successes,
            pass_rate=successes / len(results),
            all_trials_passed=successes == len(results),
            any_trial_passed=successes > 0,
            score_min=min(r.score for r in results),
            score_max=max(r.score for r in results),
        )
    return dict(
        trial_count=len(reports),
        case_count=len(ids),
        per_case=per_case,
        note="Descriptive observed trials only; no confidence or production reliability claim.",
    )
