# 使用与复现

要求 Python 3.10+。核心库不依赖模型 SDK，测试和所有默认示例不需要 API 密钥。

```bash
python -m pip install -e ".[dev]"
pytest -q
python examples/demo.py
python examples/trace_eval_demo.py
python examples/behavior_eval_demo.py
python examples/multi_provider_demo.py
```

`demo.py` 展示便宜的文本诊断；`trace_eval_demo.py` 展示精确任务约定；`behavior_eval_demo.py` 回放带 synthetic 来源声明的正例、负例和未知案例。生成报告应保存到仓库之外；不要提交真实业务 trace、密钥、私有 prompt 或临时评测输出。

## 定义任务并比较版本

```python
from llm_eval_lab import (
    AgentEvalCase, CapturedRun, TraceEvent, TraceExpectation,
    EvaluationContext, evaluate_agent_runs, compare_agent_eval_reports,
)

case = AgentEvalCase(
    case_id="synthetic-config", prompt="读取模拟配置，只返回状态。",
    accepted_outputs=("enabled",), rubric=(),
    expectation=TraceExpectation(
        required_tools=("read_config",), forbidden_tools=("send_report",),
        artifact_values={"enabled": True}, max_total_tokens=100,
    ),
)
run = CapturedRun(
    run_id="synthetic-v1-run", prompt=case.prompt, output="enabled", status="completed",
    trace=(
        TraceEvent(1, "tool.started", {"tool": "read_config"}),
        TraceEvent(2, "tool.completed", {"tool": "read_config"}),
        TraceEvent(3, "run.completed"),
    ),
    artifacts={"enabled": True}, usage={"total_tokens": 50},
)
context = EvaluationContext(environment="synthetic-config-fixture-v1")
baseline = evaluate_agent_runs({case.case_id: run}, [case], context=context,
                               subject={"agent_revision": "baseline"})
# 实际使用中应替换为候选版本重新执行后捕获的 run；这里仅展示 API。
from dataclasses import replace
candidate = evaluate_agent_runs(
    {case.case_id: replace(run, run_id="synthetic-v2-run")}, [case], context=context,
    subject={"agent_revision": "candidate"},
)
gate = compare_agent_eval_reports(
    baseline, candidate,
    max_metric_drops={"pass_rate": 0},
    minimum_metrics={"check/token_efficiency": 1},
)
print(gate.passed, gate.reasons)
```

`rubric=()` 适用于精确任务约定足够完整的情况。如果任务要求自然语言解释，应定义语义 rubric，而不是把参考答案中的字词当成事实正确性。

完整报告可以用 `dataclasses.asdict(report)` 序列化，包含检查和 manifest。逐条结果使用 `result_as_structured_json` 输出 JSON v2，对应 `evals/rubric_result.schema.json`。CI 中可在 `gate.passed` 为 False 时退出 1；评测异常也必须返回非零，不能忽略。

## 接入语义 judge

```python
from llm_eval_lab import BinaryCriterion, StructuredRunJudge

semantic_case = replace(
    case, accepted_outputs=(), reference="模拟配置 enabled=true。",
    rubric=(BinaryCriterion("answer_supported", "回答是否准确描述配置且没有与证据矛盾的陈述？"),),
)

class SyntheticClient:
    def judge(self, request, validation_errors):
        # 仅演示协议；固定返回不是模型能力，不应用于真实判分。
        return {"answer_supported": {
            "verdict": "unknown", "reason": "该示例未配置真实语义评测器。", "evidence": [],
        }}

judge = StructuredRunJudge(SyntheticClient(), judge_id="synthetic-client",
                          model="none", prompt_version="example-v1")
report = evaluate_agent_runs({case.case_id: run}, [semantic_case], judge)
```

自定义 client 需要返回每项 rubric 的 `verdict/reason/evidence`，收到 `validation_errors` 后可修正结构。真实 judge 可以通过可选网关适配器接入：

```bash
python -m pip install -e ".[providers]"
```

```python
import os
from llm_eval_lab import LiteLLMProvider, ResilientModelGateway, GatewayJudgeClient

# 显式配置供应商凭据和实际模型标识；不会自动选择或调用模型。
model = os.environ["EVAL_JUDGE_MODEL"]
with ResilientModelGateway(
    [LiteLLMProvider(timeout_seconds=20)], timeout_seconds=25,
    retries_per_provider=1, max_in_flight=2,
) as gateway:
    client = GatewayJudgeClient(gateway, model)
    judge = StructuredRunJudge(client, judge_id="litellm-single-provider",
                              model=model, prompt_version="binary-evidence-v1")
    report = evaluate_agent_runs(
        {case.case_id: run}, [semantic_case], judge,
        context=EvaluationContext(environment="your-recorded-environment-version"),
    )
    # client.completions 保存模型调用的尝试记录和费用完整性。
```

这段是可选真实服务接入方式，默认 CI 不执行它。接入前先用少量可公开案例校准 rubric；SDK/模型版本、环境和 judge 设置应显式记录。不要把模型默认解码参数的变化当成候选 Agent 的改进。

## 标注对齐与重复运行

`compare_labels(reference, candidate)` 接收两组 `Label(case_id, check_id, verdict)`，必须逐项对齐。查看误放行和未知比例，再检查已知部分的一致率；不能只报告一个较高的一致率。

`summary = summarize_trials([trial_1, trial_2, trial_3])` 汇总同一版本的独立捕获结果。各次使用不同 run ID、相同 subject 和 environment，执行器负责重置环境。该函数不重新执行 Agent，也不计算统计显著性。

## 从旧接口迁移

- 默认文本 rubric 改为 `reference_overlap/nonempty_output`，不再用词汇重合冒充 relevance/groundedness。
- 默认 Agent 语义质量返回 unknown；精确约定任务显式设置 `rubric=()`，开放任务接入 judge。
- 指标名使用 `check/<id>`；空集、重复 ID、多余或缺失 run、比较条件不一致会抛出错误。
- JSON v2 用三态 verdict 替换布尔 pass，保留权重、硬性条件和证据。
- 自定义运行 judge 实现 `configuration` 与 `grade`；自定义文本 judge 调用 `evaluate_dataset` 时提供 `judge_config`。
