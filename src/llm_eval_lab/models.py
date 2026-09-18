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
    evaluation_fingerprint: str = ""


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
        sequence = payload.get("sequence")
        event_type = payload.get("event_type", payload.get("type"))
        body = payload.get("payload", payload.get("item", {}))
        if type(sequence) is not int or sequence < 0:
            raise ValueError("trace sequence must be an explicit nonnegative integer")
        if (
            not isinstance(event_type, str)
            or not event_type.strip()
            or not isinstance(body, dict)
        ):
            raise ValueError("invalid trace event type or payload")
        return cls(sequence, event_type, body)


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
    required_tools: tuple[str, ...] = ()
    forbidden_tools: tuple[str, ...] = ()
    required_skills: tuple[str, ...] = ()
    forbidden_skills: tuple[str, ...] = ()
    artifact_values: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BinaryCriterion:
    check_id: str
    description: str
    must_pass: bool = True
    weight: float = 1.0


@dataclass(frozen=True)
class EvaluationContext:
    environment: str = "synthetic-v1"
    protocol: str = "agent-eval-v2"


@dataclass(frozen=True)
class AgentEvalCase:
    case_id: str
    prompt: str
    reference: str = ""
    expectation: TraceExpectation = field(default_factory=TraceExpectation)
    accepted_outputs: tuple[str, ...] = ()
    rubric: tuple[BinaryCriterion, ...] = (
        BinaryCriterion(
            "answer_supported",
            "Does the answer satisfy the task without contradicting the reference or captured evidence?",
        ),
    )
    tags: tuple[str, ...] = ()


@dataclass(frozen=True)
class StructuredCheck:
    check_id: str
    category: str
    passed: bool | None
    score: float | None
    notes: str
    source: str = "deterministic"
    weight: float = 1.0
    must_pass: bool = False
    evidence: tuple[str, ...] = ()

    @property
    def verdict(self) -> str:
        return "unknown" if self.passed is None else ("pass" if self.passed else "fail")


@dataclass(frozen=True)
class AgentRunEvalResult:
    case_id: str
    overall_pass: bool
    score: float
    checks: tuple[StructuredCheck, ...]
    run_id: str = ""


@dataclass(frozen=True)
class AgentEvalReport:
    results: tuple[AgentRunEvalResult, ...]
    metrics: dict[str, float]
    evaluation_fingerprint: str = ""
    subject: dict[str, str] = field(default_factory=dict)
    evaluation_manifest: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AgentEvalRegressionReport:
    baseline: AgentEvalReport
    candidate: AgentEvalReport
    metric_deltas: dict[str, float]
    passed: bool
    reasons: tuple[str, ...] = ()
