# 评测协议与设计取舍

## 任务、观测和评价的边界

`AgentEvalCase` 描述 prompt、reference、预期行为、允许的精确输出、语义 rubric 和分组标签。`CapturedRun` 描述外部执行器捕获的事实。评测器不启动 Agent、不运行 trace 中的命令，也不访问 artifact 所指向的路径。调用方应记录工具可见的输入、输出和最终状态，无需采集模型隐藏推理。

结果约定至少包含精确输出、产物或语义标准之一。默认任务包含 `answer_supported` 语义标准；默认 judge 无法判断它，返回未知。如果任务本身可以被精确输出或最终状态完整定义，可以显式设置 `rubric=()`。这时通过只表示满足已声明的确定性约定。

`artifact_values` 对捕获的最终状态做值相等检查，例如模拟配置的 `enabled=True`。它依赖外部采集器提供可信状态，不验证数据库本身，也不保证输出所声称的操作真实发生。不要从模型回答反推 artifact 再将其作为独立证据。

## 行为检查的力度

- 事件顺序匹配必需步骤子序列，允许中间出现合理步骤，避免过度约束方案。
- sequence 必须非负、严格递增且唯一；允许间隔，不声称能发现所有丢失日志。
- `tool.started` 的 `payload.tool` 和 `skill.invoked` 的 `payload.skill` 用于核对名称。启用名称约束却缺少名称时，不能证明符合要求的部分返回未知。已观察到的禁止名称直接失败。
- 禁止调用检查依赖完整 trace；本库无法证明上游没有漏采。完整性、工具调用 ID 配对和并发 span 的归一化由采集适配层负责。
- 默认 `.failed` 事件使过程检查失败，即使最终恢复。它是严格回归策略，不声称合理重试都属于业务失败；有恢复语义的业务应设计独立检查和相应协议版本，而不是删掉失败日志。
- 工具数与 token 预算是软指标；需要将资源预算作为准入条件时，设置对应 metric 最低值为 1。缺失 token usage 是未知，零 token 必须明确记录。

`parse_trace_jsonl` 只接受本项目的规范化事件协议，每行必须有显式 sequence。它不是原始 Codex / OpenTelemetry 日志的通用解析器，不会默默跳过不认识的记录。

## 二元 rubric 与未知

每项 `BinaryCriterion` 包含稳定 ID、单一可判断标准、权重和是否必须通过。建议将“回答质量高”拆为具体问题，例如“结论是否与给定证据矛盾”。不要为追求指标数量将同一事实重复计分。

`StructuredRunJudge` 向 client 提供 rubric 及可寻址证据：`output`、`reference`、`artifact:<key>`、`trace:<sequence>`。client 返回每项 verdict、reason 和 evidence ID。确定判断至少需要一项实质证据，仅引用 prompt 或运行 status 不够。所有字段、rubric ID 和引用均在本地校验；未知允许空证据。

结构错误会携带校验信息重试，耗尽后保留为未知。网络异常直接抛出，表示评测运行未完成，不当成被测 Agent 答错。证据引用有效只证明可追溯，不证明推理成立。引用式协议和“不服从证据内指令”的提示可以减少误判入口，但不是对 judge 提示注入的可靠隔离；需要专门反例和人工审计。

未配置语义 judge 时，`HeuristicRunRubricJudge` 为语义 rubric 返回未知；`reference_overlap` 权重为零且类别为 diagnostic，不参与指标与通过判定。默认文本评测 `HeuristicJudge` 同样只提供词汇覆盖与非空诊断。文本路径的回归通过只是这些诊断未退化，不等于语义正确；需要硬性门禁时使用 Agent 路径。

## 报告和比较协议

Agent 报告保存完整 `evaluation_manifest` 及其 SHA-256，包括排序后的用例内容、rubric、judge configuration 和 `EvaluationContext`。内置协议版本为 `agent-eval-v2`。`subject` 单独记录被测模型、Prompt/Skill 版本或代码提交，允许两边不同。默认环境是明确的 `synthetic-v1`；真实运行必须换成实际环境快照标识。

指纹用于发现配置差异，不是对日志的真实性签名。自定义 judge 必须如实记录实现版本、模型、提示版本、解码设置及任何影响结果的参数；无法由这个库自动发现未声明的外部变化。环境标识也需要调用方维护，不能用同一个字符串掩盖工具或数据变更。

比较前拒绝空样本、重复 case/run ID、缺失或多余 run、不同 prompt、不同指纹、不同检查结构和摘要与检查不一致。未知指标名、负限额、NaN / Infinity 不能变成宽松策略。不同任务可以有不同 rubric，单项指标分母仅为适用任务数，并单独保存 `applicable/<id>`。

指标使用显式命名空间：`check/<id>` 是分数均值，`unknown/<id>` 是未知比例，`applicable/<id>` 是适用数量。另有 `overall_score`（0–100）、`pass_rate`（0–1）和 `unknown_rate`（0–1）。未知评分保留 null，聚合时按零贡献、保留权重和分母处理，因此不能通过弃权抬高总分。报告 JSON v2 保留 verdict、weight、must_pass、evidence，区别于旧版只输出布尔 pass 的格式。

门禁顺序：

1. 验证评测条件一致和报告完整。
2. 候选方案任何必须通过的检查失败或未知，直接拒绝，并列出 case/check 原因。
3. 检查总分容忍退化、指定高分为优的指标最大退化和最低值。

这是严格准入策略：既有缺陷不因 baseline 也失败而自动获准。它不是自动部署授权。软指标未知不会单独拒绝；若预算数据必须可用，应为对应 `check/` 指标配置最低值 1。未知比例和适用数量不接受“越高越好”的门槛配置。

## 校准与重复试验

`compare_labels` 用 `(case_id, check_id)` 对齐标注。它同时展示双方未知率、共同可判断覆盖率、已知部分一致率、误放行和分歧列表。双方都未知不能算作高质量的一致判断；没有共同已知样本时，一致率为 null。标签可以来自两位人工标注者，也可以来自人工与 judge。合成示例里的手写标签仅说明分析方法，不能称为人机一致性实验。

真实校准建议按场景分层抽样，让标注者独立判断再裁决分歧，先改清 rubric 再评机器；保留校准集之外的用例，尤其关注“参考为失败、机器判通过”的误放行。没有业务风险依据时，不默认设置一个通用合格一致率。

`summarize_trials` 要求同一 subject、评测条件和样本集合，每次使用不同 run ID。摘要显示成功频率、是否每次成功、是否至少成功一次和分数范围，不把这些描述值称为置信区间或无偏 pass@k。它不能证明运行独立；执行器需要重置环境并记录种子等条件。比较版本时应保持试验次数一致，逐次使用严格门禁，不能挑最好的一次展示。

## 网关的可靠性与费用语义

同步 provider 接口返回文本或带 usage 的 `ProviderResponse`。网关采用有限次数重试、指数退避和可选跨 provider 回退。每个网关实例共享有界线程池，没有无限排队；超时调用在真正结束前继续占用容量。实例应用作 context manager 并复用，不要为每次请求创建新实例绕过上限。

线程等待超时不会终止已运行的网络调用，`close()` 也不能强制终止；挂死的 provider 可能影响 Python 进程退出。`LiteLLMProvider` 额外设置传输 timeout 并关闭 SDK 内部重试；需要硬性终止的场景应在独立进程或外部执行器隔离。本库未实现该隔离能力。

`CompletionResult.usage` 是最终成功响应的 usage；`attempt_trace` 保留每次已知 usage 与费用。`estimated_cost_usd` 汇总可估算尝试，全部未知时为 null；任何尝试缺失账单或价格时 `cost_complete=False`。这不是供应商结算账单。失败但已返回 usage 的空响应也计入已知成本；超时后迟到的账单无法完整获知。全部失败时 `GatewayError` 仍保留 attempt trace。

`GatewayJudgeClient` 限定单一 provider，禁止隐式更换裁判，并保存 completion records 供审计。通用网关会重试 SDK 异常，但没有区分供应商特定的永久错误、限流窗口或实现全局 deadline；大规模接入应在 provider 适配层补充相应策略。可选 LiteLLM 适配器不代表已完成真实服务验证。
