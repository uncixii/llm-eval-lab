"""llm-eval-lab 的公开接口。"""

from .calibration import Label, compare_labels
from .judge import HeuristicJudge, JudgeOutputError, LLMAsJudge
from .models import (
    AgentEvalCase,
    BinaryCriterion,
    CapturedRun,
    EvalCase,
    EvalReport,
    EvaluationContext,
    Rubric,
    RubricCriterion,
    TraceEvent,
    TraceExpectation,
)
from .providers import (
    AttemptRecord,
    CompletionResult,
    GatewayError,
    LiteLLMProvider,
    ProviderResponse,
    ResilientModelGateway,
)
from .regression import compare_reports, evaluate_dataset
from .semantic import GatewayJudgeClient, StructuredRunJudge
from .trace_eval import (
    compare_agent_eval_reports,
    evaluate_agent_runs,
    evaluate_captured_run,
    parse_trace_jsonl,
    result_as_structured_json,
)
from .trials import summarize_trials

__all__ = [
    "AttemptRecord",
    "EvalCase",
    "EvalReport",
    "AgentEvalCase",
    "CapturedRun",
    "HeuristicJudge",
    "JudgeOutputError",
    "LLMAsJudge",
    "CompletionResult",
    "GatewayError",
    "LiteLLMProvider",
    "ProviderResponse",
    "ResilientModelGateway",
    "Rubric",
    "RubricCriterion",
    "TraceEvent",
    "TraceExpectation",
    "compare_agent_eval_reports",
    "compare_reports",
    "evaluate_dataset",
    "evaluate_agent_runs",
    "evaluate_captured_run",
    "parse_trace_jsonl",
    "result_as_structured_json",
]

__all__ += [
    "BinaryCriterion",
    "EvaluationContext",
    "StructuredRunJudge",
    "GatewayJudgeClient",
    "Label",
    "compare_labels",
    "summarize_trials",
]
