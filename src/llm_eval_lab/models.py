from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class EvalCase:
    case_id: str
    prompt: str
    reference: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RubricCriterion:
    name: str
    description: str
    weight: float = 1.0


@dataclass(frozen=True)
class Rubric:
    name: str
    criteria: tuple[RubricCriterion, ...]


@dataclass(frozen=True)
class JudgeScore:
    criterion: str
    score: float
    reason: str
    source: str = "deterministic"


@dataclass(frozen=True)
class EvalResult:
    case_id: str
    scores: tuple[JudgeScore, ...]
    overall: float


@dataclass(frozen=True)
class EvalReport:
    results: tuple[EvalResult, ...]
    aggregate: float


@dataclass(frozen=True)
class RegressionReport:
    baseline: EvalReport
    candidate: EvalReport
    delta: float
    passed: bool
    threshold: float
    metric_deltas: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class TraceEvent:
    sequence: int
    event_type: str
    payload: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TraceEvent":
        return cls(
            sequence=int(payload.get("sequence", 0)),
            event_type=str(payload.get("event_type", payload.get("type", "unknown"))),
            payload=dict(payload.get("payload", payload.get("item", {}))),
        )


@dataclass(frozen=True)
class CapturedRun:
    run_id: str
    prompt: str
    output: str
    status: str
    trace: tuple[TraceEvent, ...]
    artifacts: dict[str, Any] = field(default_factory=dict)
    usage: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class TraceExpectation:
    required_event_order: tuple[str, ...] = ()
    required_artifacts: tuple[str, ...] = ()
    max_tool_calls: int | None = None
    max_total_tokens: int | None = None


@dataclass(frozen=True)
class AgentEvalCase:
    case_id: str
    prompt: str
    reference: str = ""
    expectation: TraceExpectation = field(default_factory=TraceExpectation)


@dataclass(frozen=True)
class StructuredCheck:
    check_id: str
    category: str
    passed: bool
    score: float
    notes: str
    source: str = "deterministic"
    weight: float = 1.0
    must_pass: bool = False


@dataclass(frozen=True)
class AgentRunEvalResult:
    case_id: str
    overall_pass: bool
    score: float
    checks: tuple[StructuredCheck, ...]


@dataclass(frozen=True)
class AgentEvalReport:
    results: tuple[AgentRunEvalResult, ...]
    metrics: dict[str, float]


@dataclass(frozen=True)
class AgentEvalRegressionReport:
    baseline: AgentEvalReport
    candidate: AgentEvalReport
    metric_deltas: dict[str, float]
    passed: bool
