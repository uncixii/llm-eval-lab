# llm-eval-lab

面向 LLM/Agent 应用的 structured evaluation、trace grading 与 regression testing reference implementation。

```text
Prompt / Eval Case
        ↓
Captured Run
  ├── ordered trace events
  ├── final output
  ├── artifacts
  └── token/tool usage
        ↓
Deterministic Checks
        ↓
Rubric-based Structured Grading
        ↓
Per-check Metrics
        ↓
Baseline vs Candidate Regression Gate
```

这里不是只评最终回答。Agent 即使给出了看似正确的答案，也可能选错 tool、走错顺序、发生多余循环，或消耗不必要的 tokens；这些过程问题需要通过 trace 才能定位。

## 核心能力

- `CapturedRun` / `TraceEvent`：统一保存 prompt、output、ordered trace、artifacts 和 usage。
- `TraceExpectation`：声明 required event order、required artifacts、最大 tool calls 和 token budget。
- `deterministic checks`：检查 outcome、event order、sequence 完整性、failed event、非空 artifact、tool-call 与 token efficiency。
- `Rubric`：把 relevance、groundedness、format 等定性标准结构化并显式配置权重。
- `HeuristicJudge` / `HeuristicRunRubricJudge`：提供零依赖、可重复 baseline。
- `LLMAsJudge` / `RunRubricJudge`：保留 LLM-as-a-Judge adapter；judge 同时接收 prompt、response、reference 和 rubric，格式错误重试时会收到 validation error。
- `evaluate_agent_runs`：输出逐 case checks 以及 outcome/process/quality/efficiency 指标。
- `compare_agent_eval_reports`：比较 baseline/candidate，支持 overall、关键 metric 最大退化值和最低分门槛。
- `ResilientModelGateway`：统一 completion provider 接口，支持 timeout、指数 backoff、provider 内重试、跨 provider fallback、attempt trace、latency、token usage 和成本估算。
- `LiteLLMProvider`：可选接入 LiteLLM，通过 model identifier 路由 OpenAI、Anthropic、Gemini、DeepSeek 等 API。

## 四类评分目标

| 类别 | 示例 |
| --- | --- |
| Outcome | 任务是否完成、必需 artifact 是否存在 |
| Process | tool 是否选择正确、关键事件是否按顺序发生、是否出现失败事件 |
| Quality | 输出是否相关、是否有 artifact/observation 支撑 |
| Efficiency | tool call 是否反复、token usage 是否超出预算 |

每个 check 都包含 `id/category/pass/score/notes/source`。`evals/rubric_result.schema.json` 给出机器可读的 JSON Schema，可供 CI 稳定解析，而不是依赖自由文本结论。Outcome、trace integrity、required artifacts 和核心 quality checks 默认属于 must-pass 条件，不能被其他高分抵消。

## 为什么 deterministic 与 rubric 要同时存在

适合代码直接判断的问题，例如是否执行成功、是否产生 SQL、tool 调用次数和 trace 顺序，应优先使用 deterministic checks：结果稳定、便宜、容易 debug。

回答是否真正理解业务语义、是否有充分 grounding 等开放性问题，再交给 rubric-based judge。LLM judge 不能替代硬性检查，而是补充普通断言难以覆盖的 semantic assertion。

## 量化改进

报告不只包含一个 aggregate score，还会保留 `trace_order`、`trace_sequence`、`required_artifacts`、`tool_efficiency`、`output_relevance` 等分项指标。对 baseline 与 candidate 运行相同 eval cases 后，regression report 会输出每项 delta，因此可以回答“提升发生在哪一步”，也可以在 CI 中配置不可退化门禁。

多 API workflow 不与某一家 SDK 耦合：

```python
gateway = ResilientModelGateway([primary_provider, fallback_provider])
result = gateway.complete(prompt, model="provider/model-name")
```

gateway 不把不同 provider 的 SDK response 暴露给上层；provider 可以返回纯文本或带 usage 的 `ProviderResponse`。成本按照外部传入的 model pricing 配置计算，避免在代码中固化可能变化的价格。

安装 `llm-eval-lab[providers]` 后可使用 `LiteLLMProvider`。默认 tests 使用 synthetic providers，不调用外部 API，也不需要任何密钥。

## Clean-room 声明

这是基于通用 LLM evaluation 工程问题抽象出的 clean-room reference implementation，不包含任何前公司 proprietary code、真实内部数据、真实表名、真实 URL、内部 Prompt 或内部架构资产。示例内容均为 synthetic data。

## 运行

```bash
python -m pip install -e ".[dev]"
python examples/demo.py
python examples/trace_eval_demo.py
python examples/multi_provider_demo.py
pytest -q
```

`examples/demo.py` 展示 final-output rubric 与 baseline/candidate diff；`examples/trace_eval_demo.py` 展示 captured Agent run 的结构化过程评分。

第一版优先保证可解释、可重复和易替换。baseline 与 candidate 必须使用相同 `case_id` 集合，缺失 captured run 会被显式拒绝，避免用不同数据集制造虚假提升。接入真实 judge model 时，建议保留 deterministic checks 作为硬约束，再把开放式质量判断交给 `LLMAsJudge` 或 `RunRubricJudge`。
