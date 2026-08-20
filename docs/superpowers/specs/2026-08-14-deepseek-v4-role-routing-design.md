# DeepSeek V4 全链路路由、结构化交接与投委会评级设计

**日期：** 2026-08-14  
**状态：** DeepSeek 路由与方案 B 已确认，完整修订稿待复核
**范围：** 个股分析主图、投委会评级、最终报告、CLI、默认配置、大盘预警 LLM 慢路径

## 1. 目标

将 TradingAgents 的默认 LLM 从 MiniMax M3 切换为 DeepSeek V4，并在全部 LLM 调用中显式启用 Think。按照任务复杂度在 `deepseek-v4-pro` 与 `deepseek-v4-flash` 之间进行角色路由，同时限制跨 Agent 上下文，避免一次个股分析因完整报告和思考内容反复叠加而失控。个股长期评级同步从“目标价偏离主导”改为可审计的四支柱投委会矩阵，使前置 Agent 的专业结论能够真实影响评级或交易动作。

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
| 组合经理 PM | max | 32K | 继承研究评级并形成最终仓位和交易动作 |
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

### 7.1 协作原则：统一信封，不统一专业载荷

线下成熟投研团队不会要求技术、基本面、新闻、宏观和风控分析师填写同一张完整表。统一的应当是日期、来源、事实与观点分层、质量状态和追责方式；专业判断内容必须服从各角色的研究职责。

设计参考三类公开的机构实践：CFA 研究规范要求区分事实与观点，并让估值、推荐和风险拥有合理依据；PIMCO 将宏观讨论转化为风险目标和组合约束；Capital Group 则保留分析师独立高置信观点和团队分歧，避免过早形成单一共识。对应资料为 [CFA company analysis](https://www.cfainstitute.org/insights/professional-learning/refresher-readings/2026/company-analysis-past-and-present)、[PIMCO investment process](https://www.pimco.com/ae/en/about-us/our-process) 和 [Capital System](https://www.capitalgroup.com/advisor/ca/en/investments/know-your-product.html)。

本项目采用两层交接：

1. **专业分析师交接单**：保留每个角色的独立观点，不提前求共识。
2. **投委会决策包**：由 Research Evidence Officer 和 RM 汇总，通过正式证据 ID 进入决策。

Research Evidence Officer 之前的专业研究角色只共享以下最小公共信封：

```yaml
HANDOFF:
  schema_version: agent-handoff-v1
  role: <角色>
  mandate: <本角色负责回答的问题>
  as_of: <分析截止时间>
  horizons: [<适用期限>]
  source_periods: [<财务/行情/事件期间>]

  specialist_view:
    conclusion: <一句话专业结论>
    direction: bullish / bearish / neutral / mixed / not_applicable
    conviction: high / medium / low
    materiality: high / medium / low

  change:
    basis: <前一交易日/前一报告期/前一事件/无可比基准>
    delta: <强化/弱化/反转/无变化/不可比较及原因>

  facts:
    - local_fact_key: <角色内临时键，不是最终证据 ID>
      metric_or_event: <事实名称>
      value: <数值或事件内容>
      period: <数据所属期间>
      observed_at: <行情/行为数据观察时间或 null>
      published_at: <公告/财报/新闻公开日期或 null>
      source_ref: <来源引用>
      source_tier: official / regulatory / mainstream / research / social / calculated

  interpretations:
    - inference: <专业解释>
      fact_keys: [<支持该解释的 local_fact_key>]
      assumptions: [<必要假设>]

  counterpoints: [<本角色发现的反证>]
  uncertainties: [<本角色无法确认的事项>]
  requested_checks: [<希望哪个下游角色继续核查什么>]

  quality:
    status: complete / partial / invalid
    missing_fields: [<缺失项>]
    conflicts: [<本角色内部冲突>]
```

其中 `interpretations` 是可审计的“事实 → 专业解释”桥梁，不是原始思维链。基础分析师不得生成 `E-xxx` 正式证据 ID；正式 ID 只能由 Research Evidence Officer 在校验日期、来源、质量和决策资格后确定性分配。

Research Evidence Officer 之后的 Bull/Bear、RM、风控和 PM 不再携带 `facts`，而使用更严格的决策信封：

```yaml
DECISION_HANDOFF:
  role: <角色>
  as_of: <分析截止时间>
  accepted_claim_ids: [<正式证据 ID>]
  challenged_claims:
    - claim_id: <正式证据 ID>
      challenge_type: stale / weak_source / causal_gap / conflicting / mispriced
      reason: <挑战理由>
  decision_judgments:
    - judgment: <本角色判断>
      claim_ids: [<正式证据 ID>]
      assumptions: [<必要假设>]
      affected_horizon: 3d / 3m / 12m / position
      decision_dimension: thesis / valuation / catalyst / expectation_gap / timing / crowding / macro / risk
      allowed_effect: foundation / support / oppose / cap / floor / timing_only / conviction_only / veto_request
      direction: bullish / bearish / neutral / mixed
      materiality: high / medium / low
      confidence: high / medium / low
  unresolved_dissent: [<未解决分歧>]
  conditions_to_revisit: [<重新评估条件>]
  quality_status: complete / partial / invalid
```

决策角色不得新增事实；发现新事实时只能提出 `requested_check`，由对应专业分析师或确定性节点补证后重新进入 Evidence Officer。

### 7.2 角色专属交接内容与粒度

#### 7.2.1 市场/技术面分析师

**职责问题：** 当前价格结构、未来三日方向、入场时机和关键价位是什么？

必须交接：

- 周线、日线和未来三日方向，明确是否多周期一致。
- 动量、波动率、量价、资金流和数据时点。
- 支撑、阻力、失效位以及这些价位的计算依据。
- `base / upside / downside` 三种短线结构及触发条件，不填写主观概率。
- 入场偏好：追随、回踩确认、突破确认、等待或规避。
- 与前一交易日相比的结构变化。

粒度上限 6K 字符。只能建议短期时机，不得给出一年期目标价、长期评级或最终仓位。

#### 7.2.2 基本面分析师

**职责问题：** 公司未来一年盈利质量、竞争力和估值锚是否支持持有？

必须交接：

- 财务期间、单位、累计/单季口径和数据质量标志。
- 营收、归母、扣非、ROE、现金流、应收、存货和杠杆的变化桥。
- 商业模式、竞争壁垒、客户集中度和治理风险。
- 未来 12 个月盈利驱动、关键假设及相对上一报告期的变化。
- 估值方法、可用基准、合理区间和不可比原因。
- 基本面 `bear / base / bull` 情景的假设与结果范围；概率留给 RM。
- 最重要的财务反证和 thesis breaker。

粒度上限 8K 字符。只能建议长期质量和估值方向，不得判断三日买点或绕过市场风险门。

#### 7.2.3 新闻/事件分析师

**职责问题：** 哪些新事件会在什么时间改变盈利、估值或风险？

必须交接：

- 事件标题、事件日期、发布日期、日期依据和原始来源。
- 可信度、验证状态、来源层级、相关期限和 thesis 相关度。
- 事件对收入、利润、估值或风险的一级与二级传导链。
- 催化剂日历、验证窗口、已定价程度和失效条件。
- 卖方盈利修正、评级和目标价必须标注 `as_of`，不同机构分歧不得平均抹平。
- 传闻与已验证事实分区，传闻不能进入硬决策。

粒度上限 8K 字符。不得把新闻情绪直接升级成投资评级，也不得使用分析日之后的信息。

#### 7.2.4 舆情分析师

**职责问题：** 投资者行为是否显示拥挤、脆弱或叙事变化？

必须交接：

- 样本量、来源覆盖、采样窗口和可代表性限制。
- 多空中性占比、7 日变化、KOL 数量与一致性。
- 核心叙事、叙事新增/退潮、传播集中度和拥挤方向。
- 传闻标记、情绪与价格背离、可能的反转触发条件。
- 情绪对短线波动和风险的影响，不延伸为公司基本面事实。

粒度上限 4K 字符。社交内容只能作为行为证据；没有官方或主流来源复核时，不得进入盈利、订单或估值硬证据。

#### 7.2.5 宏观环境分析师

**职责问题：** 自上而下环境如何改变该行业的风险溢价和胜率？

必须交接：

- 利率周期、流动性、风险偏好和宏观数据时点。
- 对目标行业的传导链，而不是泛化宏观评论。
- 三个月与一年两个期限的顺风/逆风判断。
- 已知宏观事件日历及可能影响方向。
- 对主题溢价和风险预算的调整建议，以及触发其改变的条件。
- 与公司底层事实冲突的宏观假设。

粒度上限 5K 字符。只能建议风险溢价和环境约束，不得直接决定个股评级。

#### 7.2.6 股票画像分析师

**职责问题：** 这只股票应采用什么研究框架、权重和估值方法？

必须交接：

- 市值、行业、风格、流动性、股票类型和业务暴露。
- `REPORT_WEIGHTS`、`DECISION_STYLE` 和主次估值方法。
- 主题阶段、周期位置、成长质量和数据完整度。
- 确定性 `SYS_*` 信号的结构化转录，不得重新解释或覆盖代码结论。
- LLM 选择与系统参照发生偏离时的差异和理由。
- 哪些专业报告对本股最重要，哪些字段不得进入决策。

粒度上限 8K 字符。它负责“怎么研究”，不负责“最终买卖”。报告权重影响阅读顺序，不影响证据真实性。

#### 7.2.7 市场共识分析师

**职责问题：** 市场已经相信什么、价格计入什么、还有哪些问题没有答案？

必须交接：

- 共识方向、强度、拥挤度及各来源分布。
- 已计价的盈利、增长、估值和催化假设，附来源与日期。
- 卖方、新闻、舆情之间的一致点和分歧点。
- 市场隐含预期与基础分析师预测的差值。
- 2-4 个可验证、可证伪的未回答问题。

粒度上限 5K 字符。不得用简单平均制造共识，不给最终推荐；其任务是建立“市场基准观点”，供 Bull/Bear 寻找预期差。

#### 7.2.8 Bull/Bear 研究员

**职责问题：** 在同一事实集上，最强的看多与看空投资论证分别是什么？

每轮必须交接：

- 核心论点和适用期限。
- 引用的正式证据 ID，不得引用未进入账本的信息。
- “证据 → 因果链 → 盈利/估值/价格影响”的可审计推理桥。
- 对手最强论点、接受的部分、反驳及反驳依据。
- 被挑战的证据 ID、挑战类型和需要补证的问题。
- 使本方观点失效的条件。

单轮上限 6K 字符。只保留当前轮和上一轮结构化摘要，不累计完整辩论全文；双方没有评级和仓位决定权。

#### 7.2.9 Research Manager

**职责问题：** 独立研究是否形成足够证据支持的短期观点和一年期建议？

必须形成真正的投委会研究包：

- `decision_request`：本次需要 PM 决定的问题。
- 三日趋势/入场观点与一年期研究评级，二者必须分开。
- 市场已计价内容、我们的 variant view 及其证据。
- `bear / base / bull` 情景：假设、概率、目标区间、潜在收益和关键触发点。
- 支持证据、反证和证据质量排序。
- 未解决分歧：谁不同意什么、为何尚未解决。
- 催化剂日历、thesis breaker、跟踪指标和复核事件。
- 因数据缺失而不能下结论的部分。

粒度上限 12K 字符。RM 可以形成研究评级，但不能覆盖代码计算的市场闸门、仓位上限、价格或估值输入。

#### 7.2.10 风控三方

**职责问题：** 在 RM 研究结论成立或失败时，组合面临什么损失路径？

每个角色必须交接：

- 风险渠道、期限、发生可能性和影响等级。
- 引用的正式证据 ID和市场风险字段。
- 压力情景、损失传导和触发条件。
- 对仓位、止损、流动性和事件窗口的建议。
- 对其他风控观点的异议及仍未解决的问题。

单方上限 5K 字符。激进方负责机会成本与少配风险，中立方负责情景平衡，保守方负责尾部风险；三方只能提出仓位上限或 veto 建议，不能修改确定性硬门。

#### 7.2.11 Portfolio Manager

**职责问题：** 在研究观点和硬风险约束下，现在具体做什么？

最终交付必须包含：

- 当前动作、未来三日操作和一年期评级。
- 新仓/持仓分别适用的仓位区间、入场条件、止损和减仓条件。
- 最大可接受损失、流动性约束和市场风险门。
- 采纳的证据 ID、拒绝的研究观点及拒绝原因。
- 对未解决分歧的保守处理方式。
- 催化剂前后复核节点和 thesis breaker。
- 数据不完整时的明确降级或 `WAIT`。

PM 是唯一拥有最终动作权的 LLM 角色，但不得突破确定性代码给出的风险门、仓位上限和证据资格。

#### 7.2.12 大盘预警解释器

**职责问题：** 为什么确定性规则在当前时点发出该灯号，哪些信息可能使风险升降？

只输出风险因果链、历史相似结构、支持/反对因素、不确定性和下一观察点。不得输出新的概率、改灯、改变推送条件或直接生成个股交易动作。

### 7.3 证据编号、决策权限与投委会包

专业分析师交接单中的 `local_fact_key` 只在本角色报告内有效。Research Evidence Officer 负责：

1. 校验来源、发布日期、数据期间、时效和质量。
2. 将合格事实映射为正式 `claim_id`。
3. 标记 `decision_eligible`、方向、期限、owner 和 falsifier。
4. 保留冲突，不自动裁决。
5. 编译供 Bull/Bear、RM、风控和 PM 使用的 `research_evidence_ledger` 与 `ic_decision_packet`。

权限边界如下：

| 角色 | 可建议 | 不可决定 |
|---|---|---|
| 市场 | 三日方向、入场结构、价位 | 一年期评级、最终仓位 |
| 基本面 | 盈利质量、估值和长期方向 | 三日买点、市场风险门 |
| 新闻/舆情 | 催化、情绪和事件风险 | 硬财务事实、最终评级 |
| 宏观 | 风险溢价和行业环境 | 个股买卖 |
| 画像/共识 | 研究框架、市场基准观点 | 最终推荐 |
| Bull/Bear | 独立论证和证据挑战 | 评级、仓位 |
| RM | 研究评级和情景概率 | 突破硬门和仓位上限 |
| 风控 | 风险建议、veto 建议 | 修改确定性规则 |
| PM | 最终动作和执行计划 | 覆盖确定性硬约束 |
| 预警解释器 | 解释和观察点 | 改概率、改灯、改推送 |

### 7.4 双层决策：研究评级与当前动作分离

最终报告必须同时给出两个不同问题的答案，不得用一个 `BUY/HOLD/SELL` 混合长期价值与短期时机：

1. **一年期研究评级 `research_rating`**：回答未来 12 个月相对当前价格是否值得配置，由 RM 形成投委会研究结论。
2. **当前交易动作 `trade_action`**：回答现在是否执行、执行多少和如何退出，由 PM 在市场结构与硬风险约束下决定。

允许出现 `research_rating=OVERWEIGHT`、`trade_action=WAIT`，表示长期方向成立但当前买点或市场风险不允许进场。市场技术面不得把一年期评级从看多改为看空；基本面也不得越权把短期 `WAIT` 改成立即买入。

### 7.5 四支柱投委会矩阵

#### 7.5.1 设计原则

现有“目标价偏离先机械映射评级、其他报告再有限修正”的方式会把大部分专业研究压缩成一个目标价，使前置 Agent 的贡献和分歧难以追溯。本次改为“证据门槛 + 四支柱矩阵 + 情景预期收益 + 风险边界”：

- 估值仍是必要支柱，但不再单独生成初始评级。
- 不采用 Agent 多数投票，也不把不同职责的方向票线性相加。
- 高质量分歧必须保留，不能通过平均分消失。
- 正向评级必须具备正的概率加权预期收益；负向评级必须具备负的概率加权预期收益或明确 thesis breaker。
- 研究风险和数据质量可以封顶或降级研究评级；风控委员会可以否决当前动作，但不能伪装成另一位专业分析师的结论。

#### 7.5.2 四个研究支柱

| 支柱 | 负责回答的问题 | 主责输入 | 不负责 |
|---|---|---|---|
| 经营与盈利质量 `thesis` | 盈利增长是否真实、可持续，竞争力是否改善 | 基本面、行业对照、公司画像 | 三日买点 |
| 估值与预期收益 `valuation` | 当前价格对应的情景收益和下行空间是否有吸引力 | 基本面估值、画像方法、宏观风险溢价、三情景目标 | 单独决定最终评级 |
| 催化与预期差 `catalyst` | 什么尚未计价，何时可能被验证 | 新闻、共识、盈利修正、事件日历 | 把传闻当成硬事实 |
| 持续性与风险 `durability` | thesis 能否穿越一年、失败路径和尾部风险是什么 | 宏观、资金、拥挤、治理、Bull/Bear、确定性风险数据 | 取代评级后的风控委员会和确定性风险硬门 |

股票画像可以按 `DECISION_STYLE` 调整支柱关注顺序和适用门槛，但不得把任一支柱权重设为零。不同风格的差异体现在证据要求和收益阈值，不使用一个全市场固定加权总分。

#### 7.5.3 RM 投委会输入与输出

RM 接收 Evidence Officer 校验后的证据账本，为每个支柱形成：

```yaml
IC_RECOMMENDATION_INPUT:
  pillars:
    thesis:
      state: strong / adequate / mixed / weak / invalid
      direction: bullish / bearish / neutral / mixed
      confidence: high / medium / low
      accepted_claim_ids: []
      challenged_claim_ids: []
      thesis_breakers: []
    valuation:
      state: attractive / fair / stretched / invalid
      scenario_expected_return_pct: <float|null>
      downside_pct: <float|null>
      payoff_ratio: <float|null>
      confidence: high / medium / low
      accepted_claim_ids: []
    catalyst:
      state: strong / visible / weak / absent / adverse
      priced_in: low / partial / high / unknown
      next_validation_date: <date|null>
      accepted_claim_ids: []
    durability:
      state: resilient / acceptable / fragile / broken
      hard_veto: true / false
      accepted_claim_ids: []
  unresolved_dissent: []
  evidence_quality: complete / partial / insufficient
```

RM 使用 Think 完成证据解释、因果链、情景假设和支柱状态判断；不得直接手写最终评级。这里的 `hard_veto` 仅指正式证据已经验证的研究 thesis breaker，与评级形成之后风控委员会对交易动作施加的 execution veto 不同。确定性工具 `compute_ic_recommendation` 校验枚举、证据资格、情景概率、预期收益、研究硬否决和评级边界后返回 `research_rating`、`rating_reason_codes` 与完整阶段留痕。

#### 7.5.4 确定性评级矩阵

评级工具遵守以下语义，具体收益阈值继续按股票风格、波动率和主题阶段动态生成：

| 评级 | 必要条件 |
|---|---|
| `BUY` | 预期收益达到高档阈值；`thesis` 为 strong/adequate；`catalyst` 至少 visible；无 hard veto；证据非 insufficient |
| `OVERWEIGHT` | 预期收益为正并达到配置阈值；经营 thesis 不为 weak/invalid；没有 broken durability；允许存在一项中等不确定性 |
| `HOLD` | 预期收益接近中性，或四支柱存在高质量冲突，或证据不足以支持方向性配置 |
| `UNDERWEIGHT` | 概率加权收益达到负向配置阈值，且经营、催化或持续性至少一项明显偏弱 |
| `SELL` | 负收益达到高档阈值并伴随 weak thesis / broken durability，或出现经过证据验证的 thesis breaker |

补充规则：

- `BUY/OVERWEIGHT` 的概率加权预期收益必须为正，`UNDERWEIGHT/SELL` 必须为负。仅当出现经过正式证据验证的 thesis breaker 时，允许在目标价数据暂时滞后的情况下输出 `SELL`，并强制标记 `THESIS_BREAK_OVERRIDE`；一般的 weak thesis 只能要求重算情景，不能绕过收益方向不变量。
- 未触发 `THESIS_BREAK_OVERRIDE` 时，经营与盈利为 `invalid`、估值情景为 `invalid` 或证据质量为 `insufficient`，最高只能 `HOLD`；经过正式证据验证的 thesis breaker 仍可进入 `SELL` 例外路径。
- 催化缺失不能单独触发负评级，但可阻止 `BUY`；负面催化可以加强 `UNDERWEIGHT/SELL`。
- 舆情只影响 `crowding`、短期脆弱性、评级封顶和仓位，不提供长期方向基础票。
- Bull/Bear 不投票，只能挑战证据、假设和因果链；挑战成立时由 RM 重评对应支柱。
- 市场技术面只生成三日结构和 `trade_action` 条件，不进入一年期评级方向。

#### 7.5.5 LLM 与代码职责

| 任务 | 执行者 |
|---|---|
| 识别事实含义、因果链、预期差、情景假设和 thesis breaker | 对应专业 Agent / RM，Think 开启 |
| 校验日期、来源、证据 ID、字段枚举和跨期穿越 | Evidence Officer / Python |
| 计算目标价、情景概率和、概率加权收益、赔率、阈值 | Python 工具 |
| 根据已确认支柱状态执行评级矩阵、封顶、否决和不变量检查 | `compute_ic_recommendation` |
| 决定当前动作、仓位、入场和止损 | PM + 现有确定性风险/执行工具 |

LLM 可以提出“为什么支柱应为 strong/weak”的有证据判断，代码负责保证相同输入得到相同评级结果。不得把原始 Think 当作评级输入。

现有 `compute_step6_final_rating` 不再作为长期评级的权威入口。实施时保留并复用其中已验证的动态收益阈值、拥挤封顶、数据质量降级和方向不变量逻辑，但删除“先按目标价偏离产生五档初始评级”的主导路径；最终权威结果只来自 `compute_ic_recommendation`，避免两套评级工具并存并互相覆盖。

### 7.6 决策贡献账本与用户报告

Evidence Officer 先生成通过资格校验的候选贡献，RM 只能从候选中选择作用和解释，最终由 Python 编译 `decision_contribution_ledger`。每条贡献必须包含：

```yaml
- role: fundamentals
  decision_dimension: thesis
  conclusion: <专业结论>
  accepted_effect: foundation / support / oppose / cap / floor / research_veto / timing_only / conviction_only / rejected
  direction: bullish / bearish / neutral / mixed
  materiality: high / medium / low
  confidence: high / medium / low
  claim_ids: []
  rejection_reason: <未采纳时必填>
```

同一结论只能在一个主决策维度中记一次，防止新闻、共识和 Bull/Bear 重复引用同一事实造成多重计分。PM 不得修改贡献归因，只能说明它如何把研究评级转成交易动作。

最终 `decision.md` 顶部必须先显示：

1. 一年期研究评级。
2. 当前交易动作、新建仓位和未来三日判断。
3. 四支柱结论及其对评级的作用。
4. 当前动作与长期评级不一致时的明确原因。
5. 2-4 条最重要的 Agent 贡献，以及被拒绝的高重要性观点。

用户版只展示结论、证据摘要和贡献，不展示工具调用、矩阵执行过程或原始 Think。示例：

```markdown
长期研究评级：OVERWEIGHT
当前交易动作：WAIT

评级依据：经营与盈利支持｜估值与预期收益支持｜催化中性｜持续性风险限制升至 BUY
暂不买入原因：三日结构尚未确认，市场风险门限制新增仓位
关键分歧：基本面认为盈利加速可持续；Bear 认为订单兑现节奏尚未验证
```

### 7.7 数据流

1. 四个基础分析师保留完整 Markdown 报告用于审计，同时输出角色专属 `HANDOFF`；下游 LLM 默认不再接收完整报告。
2. 宏观、股票画像和共识节点接收有界专业交接单；确定性画像计算仍可在代码内部读取完整报告，但不将其原样放入 LLM prompt。
3. Research Evidence Officer 由纯 Python 校验专业交接、编译正式证据账本和投委会初始包。
4. Bull/Bear 只接收 IC 包、正式证据、当前对手观点和有限历史。
5. RM 接收证据账本、角色交接包和有界辩论摘要，形成四支柱输入；Python 工具生成一年期研究评级与贡献账本。
6. 风控辩手接收 RM 包、贡献账本和风险数据包，并只保留最新有效观点。
7. PM 接收研究评级、贡献账本、风险共识、市场闸门和仓位约束，形成当前动作；PM 不重新计算长期评级。
8. 报告渲染器将研究评级、当前动作、四支柱贡献和关键分歧置于 `decision.md` 顶部。
9. 完整报告只进入审计产物；原始 Think 在任何 Agent 边界前丢弃。

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
| 市场分析师交接单 | 6K |
| 基本面分析师交接单 | 8K |
| 新闻分析师交接单 | 8K |
| 舆情分析师交接单 | 4K |
| Macro/Profile/Consensus 单份交接 | 5K / 8K / 5K |
| Bull/Bear 单轮输出 | 6K |
| RM 投委会研究包 | 12K |
| 单个风控交接单 | 5K |
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
- 基础研究角色不能生成正式证据 ID；Evidence Officer 分配的 ID 稳定、唯一且可追溯。
- Bull/Bear、RM、风控和 PM 引用不存在或不合格的证据 ID 时必须被剔除并降级。
- 每个角色的 handoff schema、粒度上限和权限边界分别验证；市场分析师不能输出长期目标价，基本面分析师不能决定三日买点，RM/PM 不能覆盖确定性硬门。
- 场景概率只由 RM 生成且合计为 100%；基础分析师只提供情景假设和结果范围。
- 四支柱缺失、非法枚举、重复归因和不合格证据必须被拒绝或确定性降级。
- `compute_ic_recommendation` 对同一输入必须返回同一评级；BUY/OVERWEIGHT、SELL 和 thesis breaker override 的不变量分别覆盖。
- 市场技术面只能改变 `trade_action`，不得改变 `research_rating`；基本面不得覆盖短期风险门。
- `decision_contribution_ledger` 必须标识采纳效果和拒绝原因，同一事实不得跨角色重复计分。
- 预警系统使用 DeepSeek Pro，且 LLM 失败不改变规则决策。
- 日志和报告密钥扫描、Think 标签扫描均为空。

### 12.2 集成测试

1. 使用 `/models` 或最小 Chat Completion 验证 API Key、Pro、Flash 和 Think 可用。
2. 使用带工具调用的最小请求验证 `reasoning_content` 回传兼容性。
3. 在 `.venv` 中对一只 A 股执行完整个股分析，检查所有阶段结束、模型路由符合设计、最终报告存在。
4. 检查完整报告价格日期、短期三天建议、一年期评级、仓位和市场闸门一致，并能看到四支柱及 Agent 贡献。
5. 构造“长期看多但短线 WAIT”“估值便宜但 thesis weak”“目标价滞后但 thesis breaker”三类回归样例，检查评级与动作分离。
6. 执行一次大盘预警 dry-run，确认 DeepSeek Pro 解释可用且不会发送非预期通知。
7. 运行项目全部可执行测试文件和编译检查。

### 12.3 成功标准

- 运行日志中没有新的 MiniMax 调用。
- 所有实际 LLM 调用均为 DeepSeek V4 且 Think 已开启。
- Pro/Flash 角色路由与本设计一致。
- 单次个股分析下游 prompt 不再反复包含四份完整原始报告。
- 最终长期评级不再由目标价偏离单独初始化，而由四支柱矩阵确定；目标价与情景收益仍是必要估值输入和方向不变量。
- 前置 Agent 的高重要性结论能在贡献账本和最终报告中追溯到采纳效果或拒绝原因。
- 最终报告不出现 `<think>`、`reasoning_content`、原始思维链或 API Key。
- 完整个股回归和大盘预警 dry-run 均完成，项目测试无回归。

## 13. 非目标

- 不允许 LLM 改写确定性风险概率或规则灯号。
- 不将原始思维链作为可读报告内容。
- 不新增多供应商自动路由或自动回退。
- 不调整市场预警阈值或数据供应商；股票评级算法仅按第 7.4-7.6 节改造为四支柱投委会矩阵。
- 不为已废弃 Trader 节点恢复运行流程。
