"""Audit agreement on aligned labels, retaining abstentions and disagreements."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Label:
    case_id: str
    check_id: str
    verdict: str


def compare_labels(reference: list[Label], candidate: list[Label]) -> dict:
    def index(labels):
        result = {}
        for label in labels:
            key = (label.case_id, label.check_id)
            if (
                not all(key)
                or key in result
                or label.verdict not in {"pass", "fail", "unknown"}
            ):
                raise ValueError("invalid or duplicate calibration label")
            result[key] = label.verdict
        if not result:
            raise ValueError("empty calibration labels")
        return result

    left, right = index(reference), index(candidate)
    if left.keys() != right.keys():
        raise ValueError("calibration labels must align by case and criterion")
    known = [k for k in left if left[k] != "unknown" and right[k] != "unknown"]
    false_passes = [k for k in left if left[k] == "fail" and right[k] == "pass"]
    return {
        "count": len(left),
        "jointly_known_count": len(known),
        "jointly_known_coverage": len(known) / len(left),
        "agreement_on_known": sum(left[k] == right[k] for k in known) / len(known)
        if known
        else None,
        "reference_unknown_rate": sum(v == "unknown" for v in left.values())
        / len(left),
        "candidate_unknown_rate": sum(v == "unknown" for v in right.values())
        / len(right),
        "false_passes": [
            dict(case_id=k[0], check_id=k[1]) for k in sorted(false_passes)
        ],
        "disagreements": [
            dict(case_id=k[0], check_id=k[1], reference=left[k], candidate=right[k])
            for k in sorted(left)
            if left[k] != right[k]
        ],
    }
