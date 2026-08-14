# DeepSeek V4 全链路模型路由与结构化交接设计

**日期：** 2026-08-14  
**状态：** 已确认设计，待实施  
**范围：** 个股分析主图、CLI、默认配置、大盘预警 LLM 慢路径

## 1. 目标

将 TradingAgents 的默认 LLM 从 MiniMax M3 切换为 DeepSeek V4，并在全部 LLM 调用中显式启用 Think。按照任务复杂度在 `deepseek-v4-pro` 与 `deepseek-v4-flash` 之间进行角色路由，同时限制跨 Agent 上下文，避免一次个股分析因完整报告和思考内容反复叠加而失控。

本次迁移遵循以下边界：

1. LLM 负责框架内的语义理解、权衡、反证和结论表达。
2. 价格、因子、估值、日期、风险闸门、仓位上限和证据资格仍由确定性代码负责。
3. Think 只在单个 Agent 的一次调用及其工具循环内使用，不作为跨 Agent 的正式投研证据。
4. Agent 间传递结构化结论、证据、反证、不确定性和失效条件，不传递原始思维链。
5. DeepSeek 故障时不自动回退 MiniMax，避免同一报告混用供应商而失去可复现性。

## 2. 官方能力约束

DeepSeek 官方 API 使用 `https://api.deepseek.com`，模型标识为：

- `deepseek-v4-pro`
- `deepseek-v4-flash`

两个模型均支持 OpenAI Chat Completions、Think、工具调用和 1M 上下文。Think 默认开启，但系统必须显式传递 `thinking.type=enabled`，避免供应商默认值变化导致行为漂移。

Think 模式下：

- 常规角色使用 `reasoning_effort=high`。
- 最终研究决策和组合决策使用 `reasoning_effort=max`。
- `temperature` 等采样参数不生效，因此继续保留角色温度配置仅用于兼容其他供应商，不把温度差异作为 DeepSeek 行为保证。
- 同一 Agent 的工具调用子轮次必须保留并回传 `reasoning_content`。
- 非工具调用的 Agent 边界不得传播或持久化 `reasoning_content`。

## 3. 方案选择

采用“角色分层 + 全量 Think + 结构化交接”方案。

未采用仅替换全局 quick/deep 模型的最小方案，因为当前基础分析师共享同一模型开关，宏观、画像和共识节点固定使用 quick 模型，无法体现角色重要性；同时下游多次拼接四份完整报告，迁移后仍会浪费上下文。

未采用全角色 Pro，因为新闻提取、舆情归纳、辩论和辅助信号任务调用频繁，Flash 已能完成明确边界任务，全部使用 Pro 会增加延迟和费用而缺少稳定收益。

## 4. 模型路由

### 4.1 Pro 角色

| 角色 | Think 强度 | 默认最大输出 | 原因 |
|---|---:|---:|---|
| 市场/技术面分析师 | high | 24K | 直接决定未来三天趋势和入场时机 |
| 基本面分析师 | high | 24K | 财报、估值、周期和质量判断复杂 |
| 宏观环境分析师 | high | 24K | 负责市场环境和行业外部风险 |
| 股票画像分析师 | high | 24K | 决定股票类型、估值方法和报告权重 |
| 研究经理 RM | max | 32K | 汇总证据、反证并形成研究结论 |
| 组合经理 PM | max | 32K | 形成最终评级、仓位和交易动作 |
| 大盘预警解释慢路径 | max | 16K | 仅在规则触发时调用，优先保证解释质量 |

### 4.2 Flash 角色

| 角色 | Think 强度 | 默认最大输出 | 原因 |
|---|---:|---:|---|
| 新闻分析师 | high | 16K | 事实提取、事件分类和时效校验 |
| 舆情分析师 | high | 16K | 来源噪声高，结构化归纳优先 |
| 共识分析师 | high | 16K | 基于已整理证据做归并 |
| 多头/空头研究员 | high | 16K | 调用频繁，任务是有边界的证据攻防 |
| 激进/中立/保守风控 | high | 16K | 各自执行明确风险检查 |
| 信号提取、反思等辅助调用 | high | 12K | 任务简单且不直接拥有最终决策权 |

Trader 节点当前已废弃，不为其新增运行时实例；配置中保留兼容映射，防止历史入口恢复时失去明确模型策略。

## 5. 配置设计

新增 DeepSeek 提供商配置：

```text
LLM_PROVIDER=deepseek
DEEPSEEK_API_KEY=
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_PRO_MODEL=deepseek-v4-pro
DEEPSEEK_FLASH_MODEL=deepseek-v4-flash
DEEPSEEK_THINKING=enabled
```

密钥只写入项目根目录被 Git 忽略的 `.env`。`.env.example` 只增加空占位，不包含真实值。

默认配置新增：

- `llm_role_policy`：角色到 `pro/flash`、Think 强度和输出预算的映射。
- `deepseek_thinking_enabled`：默认 `True`。
- `deepseek_reasoning_effort`：常规默认 `high`。
- `deepseek_context_budget_chars`：不同阶段的输入字符预算。
- `deepseek_max_tokens_flash/pro/decision`：默认输出预算。

现有 `deep_think_llm`、`quick_think_llm` 和角色布尔开关继续保留，用于其他供应商和手动回滚；当 `llm_provider=deepseek` 且存在 `llm_role_policy` 时，角色策略优先。

`DEFAULT_CONFIG`、批量分析入口 `main.py`、交互式 CLI 和大盘预警定时入口均以 DeepSeek 为默认提供商。只有用户显式设置其他 `LLM_PROVIDER` 时才进入兼容路径，不能因为 MiniMax 环境变量仍存在而静默选择 MiniMax。

## 6. 客户端边界

新增独立 `DeepSeekClient`，而不是把 DeepSeek 特殊参数继续堆进通用 OpenAI 客户端。职责包括：

1. 从 `DEEPSEEK_API_KEY` 读取密钥。
2. 默认使用官方基础地址。
3. 校验 Pro/Flash 模型标识。
4. 显式发送 `thinking`、`reasoning_effort` 和 `max_tokens`。
5. 保留现有壁钟超时和重试包装。
6. 保持 `reasoning_content` 位于 AIMessage 附加字段，使 LangGraph 工具循环能够原样回传。
7. 最终返回正文时只标准化 `content`，不得把 Think 拼入报告、状态、日志或数据库。

通用日志只记录输入摘要、最终正文长度、模型、耗时和 token usage；不记录密钥和原始 `reasoning_content`。

## 7. 跨 Agent 结构化交接

### 7.1 标准交接包

每个需要进入下游决策的 LLM 节点必须提供或被确定性代码投影成以下结构：

```yaml
role: market|fundamentals|news|sentiment|macro|profile|consensus|research|risk
as_of: YYYY-MM-DD
conclusion: 一句话结论
confidence: high|medium|low
supporting_evidence:
  - evidence_id: E-xxx
    claim: 结论
    source_ref: 来源引用
counter_evidence:
  - evidence_id: E-xxx
    claim: 反证
uncertainties:
  - 未知项
invalidation_conditions:
  - 失效条件
next_agent_focus:
  - 下游需要重点验证的问题
quality_status: complete|partial|invalid
```

优先复用现有 `SUMMARY`、`research_evidence_ledger` 和 `ic_decision_packet`，不再创建一套平行证据体系。新模块只负责把不同 Agent 的已有结构投影为统一交接包并实施预算。

### 7.2 数据流

1. 四个基础分析师保留完整 Markdown 报告用于审计，但下游 LLM 默认只接收各自 `SUMMARY` 和交接包。
2. 宏观、股票画像、共识节点接收有界摘要；确定性画像计算仍可在代码内部读取完整报告，不将其原样放入 LLM prompt。
3. Research Evidence Officer 继续由纯 Python 编译证据账本和 IC 决策包。
4. Bull/Bear 只接收 IC 包、交接摘要、当前对手观点和有限历史，不重复接收四份完整报告。
5. RM 接收证据账本、分析师交接包和有界辩论摘要；完整报告仅保留在审计产物。
6. 风控辩手复用现有风险数据包，并只保留最新有效观点和结构化风险项。
7. PM 接收 RM 结论、风险共识、市场闸门、仓位约束和关键证据，不消费原始 Think。

## 8. 上下文预算

预算按信息优先级确定，而不是简单截断尾部：

1. 当前价格、数据日期、市场闸门和确定性硬约束。
2. 已验证核心证据及其反证。
3. Agent 结构化结论与失效条件。
4. 最新一轮多空和风险观点。
5. 补充叙述。

默认字符预算：

| 阶段 | 上限 |
|---|---:|
| 单份基础分析师交接包 | 12K |
| Macro/Profile/Consensus 输入 | 48K |
| Bull/Bear 单轮输入 | 56K |
| RM 输入 | 96K |
| 单个风险辩手输入 | 48K |
| PM 输入 | 96K |
| 大盘预警解释输入 | 32K |

若超出预算，先删除补充叙述和旧辩论，再压缩低优先级证据；市场闸门、日期、价格、硬风险和核心反证不得被裁剪。每次裁剪记录计数和原因，但不记录被删除的 Think。

## 9. 大盘预警迁移

现有预警系统保持“代码决策、LLM 解释”的职责边界：

- 确定性特征、概率、规则灯号、状态转换、推送策略和仓位门控不变。
- 将 MiniMax 专用慢路径替换为 DeepSeek Pro Think `max`。
- 只向 DeepSeek 发送经过裁剪的特征快照、相似历史样本和当前规则结论。
- DeepSeek 输出继续经过结构化 schema 校验，不能修改规则灯号或概率。
- 调用失败时保存 `reasoning_status=unavailable`，照常持久化规则结果；只有本来就需要推送的规则告警才推送，不因 LLM 失败额外打扰用户。

MiniMax 适配器和配置保留为显式手动回滚能力，但默认执行路径、定时任务和文档均改为 DeepSeek。

## 10. 错误处理

- 缺少 `DEEPSEEK_API_KEY`：启动前立即失败，错误信息不得包含密钥值。
- 401/403：不重试，明确提示鉴权失败。
- 429/5xx/网络超时：使用现有有界重试和指数退避。
- `finish_reason=length`、空正文、非法 YAML/JSON：执行一次同模型紧凑重试；仍失败则将节点标记为失败或部分，不切换供应商。
- 个股分析的市场、基本面、RM、PM 任一关键节点不可恢复失败时，不生成貌似完整的最终决策。
- 新闻、舆情等非决定性节点失败时允许继续，但证据完整度必须降级，PM 不得把缺失证据解释为中性。
- 大盘预警 LLM 失败不影响确定性规则结果。

## 11. 安全要求

1. 真实 API Key 不进入 Git、报告、测试快照、异常和 LLM 调用日志。
2. 测试中只使用假的 `sk-test-*`。
3. `.env` 更新后执行 Git 差异检查和密钥模式扫描。
4. 原始 `reasoning_content` 不跨 Agent、不写报告、不写数据库、不写 `llm_calls`。
5. 报告可以展示“推理依据摘要”，但必须来自结构化交接包和证据引用。

## 12. 测试与验收

### 12.1 单元测试

- DeepSeek 工厂创建、默认 URL、环境密钥和模型校验。
- 所有 DeepSeek 请求显式开启 Think。
- 角色到 Pro/Flash、high/max、输出预算映射正确。
- 工具调用子轮次保留 `reasoning_content`，Agent 结束后不进入状态和报告。
- 上下文打包遵循优先级和字符预算，硬约束不可被裁剪。
- 缺失/非法交接包降级为 partial，不能伪造 complete。
- 预警系统使用 DeepSeek Pro，且 LLM 失败不改变规则决策。
- 日志和报告密钥扫描、Think 标签扫描均为空。

### 12.2 集成测试

1. 使用 `/models` 或最小 Chat Completion 验证 API Key、Pro、Flash 和 Think 可用。
2. 使用带工具调用的最小请求验证 `reasoning_content` 回传兼容性。
3. 在 `.venv` 中对一只 A 股执行完整个股分析，检查所有阶段结束、模型路由符合设计、最终报告存在。
4. 检查完整报告价格日期、短期三天建议、一年期评级、仓位和市场闸门一致。
5. 执行一次大盘预警 dry-run，确认 DeepSeek Pro 解释可用且不会发送非预期通知。
6. 运行项目全部可执行测试文件和编译检查。

### 12.3 成功标准

- 运行日志中没有新的 MiniMax 调用。
- 所有实际 LLM 调用均为 DeepSeek V4 且 Think 已开启。
- Pro/Flash 角色路由与本设计一致。
- 单次个股分析下游 prompt 不再反复包含四份完整原始报告。
- 最终报告不出现 `<think>`、`reasoning_content`、原始思维链或 API Key。
- 完整个股回归和大盘预警 dry-run 均完成，项目测试无回归。

## 13. 非目标

- 不允许 LLM 改写确定性风险概率或规则灯号。
- 不将原始思维链作为可读报告内容。
- 不新增多供应商自动路由或自动回退。
- 不在本次迁移中调整股票评级算法、市场预警阈值或数据供应商。
- 不为已废弃 Trader 节点恢复运行流程。
