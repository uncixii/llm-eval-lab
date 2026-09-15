from llm_eval_lab import (
    AgentEvalCase,
    CapturedRun,
    TraceEvent,
    TraceExpectation,
    compare_agent_eval_reports,
    evaluate_agent_runs,
    evaluate_captured_run,
    parse_trace_jsonl,
    result_as_structured_json,
)


def make_run(event_types: tuple[str, ...]) -> CapturedRun:
    return CapturedRun(
        "run-1",
        "查询交付周期",
        "delivery_cycle 平均交付天数",
        "completed",
        tuple(TraceEvent(index, event) for index, event in enumerate(event_types, 1)),
        {"semantic_model": "delivery_cycle", "sql": "SELECT ..."},
        {"total_tokens": 100},
    )


def make_case() -> AgentEvalCase:
    return AgentEvalCase(
        "case-1",
        "查询交付周期",
        "delivery_cycle 平均交付天数",
        TraceExpectation(
            required_event_order=("run.started", "tool.started", "tool.completed", "run.completed"),
            required_artifacts=("semantic_model", "sql"),
            max_tool_calls=1,
            max_total_tokens=200,
        ),
    )


def test_trace_eval_scores_process_outcome_quality_and_efficiency() -> None:
    result = evaluate_captured_run(
        make_run(("run.started", "tool.started", "tool.completed", "run.completed")),
        make_case(),
    )
    assert result.overall_pass
    assert {check.category for check in result.checks} == {"outcome", "process", "quality", "efficiency"}
    assert '"overall_pass": true' in result_as_structured_json(result)


def test_trace_regression_exposes_failed_order_metric() -> None:
    case = make_case()
    bad = make_run(("tool.started", "run.started", "tool.completed", "run.completed"))
    good = make_run(("run.started", "tool.started", "tool.completed", "run.completed"))
    baseline = evaluate_agent_runs({"case-1": bad}, [case])
    candidate = evaluate_agent_runs({"case-1": good}, [case])
    report = compare_agent_eval_reports(baseline, candidate)
    assert report.metric_deltas["trace_order"] == 1.0
    assert report.passed


def test_parse_trace_jsonl() -> None:
    events = parse_trace_jsonl(
        '{"sequence":1,"event_type":"run.started"}\n'
        '{"type":"item.completed","item":{"type":"tool"}}\n'
    )
    assert events[0].event_type == "run.started"
    assert events[1].event_type == "item.completed"


def test_trace_eval_rejects_duplicate_sequence_and_empty_artifact() -> None:
    original = make_run(("run.started", "tool.started", "tool.completed", "run.completed"))
    run = CapturedRun(
        original.run_id,
        original.prompt,
        original.output,
        original.status,
        tuple(TraceEvent(1, event.event_type) for event in original.trace),
        {"semantic_model": "", "sql": None},
        original.usage,
    )
    result = evaluate_captured_run(run, make_case())
    assert not result.overall_pass
    assert not next(check for check in result.checks if check.check_id == "trace_sequence").passed
    assert not next(check for check in result.checks if check.check_id == "required_artifacts").passed
