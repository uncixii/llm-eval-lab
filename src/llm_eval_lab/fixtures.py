"""Load explicit synthetic contracts and captured runs, without executing tools."""

from __future__ import annotations

import json
from pathlib import Path

from .models import (
    AgentEvalCase,
    BinaryCriterion,
    CapturedRun,
    TraceEvent,
    TraceExpectation,
)


def load_synthetic_cases(path: str | Path):
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        raise ValueError("fixture must be a nonempty array")
    result = []
    for row in rows:
        if row.get("provenance") != "synthetic":
            raise ValueError("fixture must explicitly declare synthetic provenance")
        case_data = dict(row["case"])
        expectation = dict(case_data.pop("expectation", {}))
        for key in (
            "required_event_order",
            "required_artifacts",
            "required_tools",
            "forbidden_tools",
            "required_skills",
            "forbidden_skills",
        ):
            if key in expectation:
                expectation[key] = tuple(expectation[key])
        for key in ("accepted_outputs", "tags"):
            if key in case_data:
                case_data[key] = tuple(case_data[key])
        if "rubric" in case_data:
            case_data["rubric"] = tuple(
                BinaryCriterion(**c) for c in case_data["rubric"]
            )
        case = AgentEvalCase(**case_data, expectation=TraceExpectation(**expectation))
        run_data = dict(row["run"])
        run_data["trace"] = tuple(TraceEvent.from_dict(e) for e in run_data["trace"])
        result.append((case, CapturedRun(**run_data), row["expected"]))
    return result
