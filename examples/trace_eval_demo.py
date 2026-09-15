from llm_eval_lab import (
    AgentEvalCase,
    CapturedRun,
    TraceEvent,
    TraceExpectation,
    evaluate_captured_run,
    result_as_structured_json,
)


def main() -> None:
    run = CapturedRun(
        run_id="synthetic-run-01",
        prompt="查询本月交付周期",
        output="匹配 delivery_cycle，并基于查询结果返回平均交付天数。",
        status="completed",
        trace=(
            TraceEvent(1, "run.started"),
            TraceEvent(2, "semantic_model.selected", {"model": "delivery_cycle"}),
            TraceEvent(3, "tool.started", {"tool": "query_data"}),
            TraceEvent(4, "tool.completed", {"ok": True}),
            TraceEvent(5, "run.completed"),
        ),
        artifacts={"semantic_model": "delivery_cycle", "generated_sql": "SELECT ..."},
        usage={"total_tokens": 320},
    )
    case = AgentEvalCase(
        "synthetic-case-01",
        run.prompt,
        reference="delivery_cycle 平均交付天数",
        expectation=TraceExpectation(
            required_event_order=("run.started", "semantic_model.selected", "tool.started", "tool.completed", "run.completed"),
            required_artifacts=("semantic_model", "generated_sql"),
            max_tool_calls=1,
            max_total_tokens=500,
        ),
    )
    print(result_as_structured_json(evaluate_captured_run(run, case)))


if __name__ == "__main__":
    main()
