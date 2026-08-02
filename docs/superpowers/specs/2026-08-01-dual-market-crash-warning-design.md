# A 股与美股大盘骤跌概率预警系统设计

## 1. 背景

项目现有 `tradingagents.harness.market_risk_daily` 使用宽基指数趋势、20 日已实现波动率和三个指数是否站上 MA20，生成低、中、高、极高风险及 `OPEN/CONDITIONAL/WAIT` 门控。它适合识别趋势破坏和续跌状态，但无法独立承担骤跌前的概率预警。

2000-01-01 至 2026-07-31 的机械事件研究识别出 51 个 A 股急跌波段和 18 个美股急跌波段。约 59.6% 的 A 股三日暴跌正样本和 62.2% 的美股正样本发生在市场已回撤至少 5% 后。这说明现有风险模块仍有价值，但需要新增脆弱性、压力转折和概率校准能力，覆盖尚未明显下跌时的第一段风险。

本设计以 `docs/research/2026-08-01-a-us-market-crash-pattern-study.md` 及其事件数据为研究依据，并取代 `2026-07-14-tech-risk-radar-design.md` 作为市场级风险预警的主设计。旧科技风险设计中的盘前/盘中快照、数据时效和短期门控原则继续保留。

## 2. 目标

1. 分别估计 A 股和美股未来 1 个、3 个交易日的宽基急跌概率。
2. 区分尚未回撤 5% 的 `FIRST_SHOCK` 和已回撤 5% 的 `CONTINUATION` 阶段。
3. 将脆弱性积累、压力转折和续跌状态拆成独立证据组。
4. 盘前固定生成报告，盘中持续计算但只在橙灯、红灯或橙升红时推送。
5. 黄灯只提醒；橙灯限制追涨和新增仓位；红灯联动个股 `WAIT/REDUCE`。
6. 使用 M3 Think 处理因果推演、冲突证据和跨市场传导，代码负责计算、概率、状态机和硬性门控。
7. 所有结果可回测、可复现、可审计；数据不足不得伪装成低风险。
8. 任一数据源或 LLM 故障时，系统仍能落库并生成量化报告。

## 3. 非目标

- 不预测战争、疫情、恐袭或政策突变发生的准确时间。
- 不建设自动下单系统。
- 不在首版建设全市场订单簿或秒级高频模型。
- 不让 LLM 计算技术指标、训练概率模型或绕过硬性风险约束。
- 不因市场短期风险直接修改个股长期评级或一年期目标价。
- 不把规则分数包装成概率。

## 4. 设计原则

### 4.1 职责边界

- 代码处理确定性计算、数据质量、模型推断、状态转换、门控和审计。
- LLM 处理需要语义理解和逻辑推演的场景判断。
- LLM 可以在满足证据约束时上调一级，不能降低代码确认的橙灯或红灯。
- Think 过程不保存到最终报告，也不传递给后续 Agent；只保存结构化结论。

### 4.2 Clean Architecture

领域规则不依赖 Tushare、Yahoo、MiniMax、SQLite 或飞书。外部能力通过端口接入，适配器失败不得污染领域对象。

只采用实际降低耦合的模式：

- Ports and Adapters：隔离行情、模型、LLM、存储和通知。
- Strategy：A 股和美股使用不同特征策略。
- State Machine：管理绿、黄、橙、红及恢复。
- Repository：统一持久化快照、预测、判断和推送。
- Circuit Breaker：约束数据源及 LLM 连续失败。

不为简单对象增加抽象工厂、服务定位器或动态插件系统。

## 5. 总体架构

```text
MarketDataPort / ContextPort
          |
          v
DataQualityService
          |
          v
FeatureStrategy(A / US)
          |
          v
ProbabilityModelPort
          |
          v
QuantRiskAssessment
          |
          v
ReasoningPort(M3 Think)
          |
          v
WarningPolicy + StateMachine
          |
          v
Repository / Feishu / Individual-stock Gate
```

建议目录：

```text
tradingagents/harness/market_warning/
├── domain.py
├── ports.py
├── quality.py
├── features.py
├── probability.py
├── reasoning.py
├── policy.py
├── service.py
├── runner.py
└── adapters/
    ├── tushare_data.py
    ├── realtime_quote.py
    ├── us_market_data.py
    ├── minimax_reasoning.py
    ├── sqlite_repository.py
    └── feishu_notifier.py
```

`market_risk.py` 保留，作为续跌状态特征和兼容门控输入，不再继续膨胀为完整预警编排器。

## 6. 领域对象与接口

### 6.1 核心对象

`MarketDataPoint`

- `market`
- `symbol`
- `field`
- `value`
- `data_time`
- `fetched_at`
- `source`
- `quality_status`

`FeatureSnapshot`

- `market`
- `as_of_time`
- `session_slot`
- `feature_version`
- `features`
- `evidence`
- `data_quality`

`QuantRiskAssessment`

- `crash_1d_probability`
- `crash_3d_probability`
- `market_phase`
- `base_rate_1d`
- `base_rate_3d`
- `reliability_grade`
- `model_version`
- `calibration_version`
- `top_contributors`

`LLMContextAssessment`

- `market_scenario`
- `causal_chain`
- `supporting_evidence_ids`
- `conflicting_evidence_ids`
- `overlooked_risks`
- `recommended_risk_level`
- `confidence`
- `action_reason`
- `reasoning_status`

`FinalWarningDecision`

- `baseline_level`
- `final_level`
- `state_transition`
- `entry_gate`
- `new_position_cap_pct`
- `holding_action`
- `push_required`
- `decision_reasons`
- `data_status`

### 6.2 端口

- `MarketDataPort`：按市场和时间获取点时可见数据。
- `MarketContextPort`：提供可选的政策、信用和事件摘要。
- `ProbabilityModelPort`：接收特征快照并返回校准概率。
- `ReasoningPort`：接收压缩上下文并返回结构化推理结论。
- `WarningRepository`：保存和读取所有状态。
- `WarningNotifier`：发送盘前报告和升级提醒。
- `ClockPort`：为交易日历、测试和重放提供可注入时间。

## 7. 数据输入与质量

### 7.1 A 股盘前

使用项目现有 Tushare 权限：

- 上证综指、深证成指、沪深 300、中证 500、创业板和科创 50。
- 点时股票日线截面，用于上涨家数、MA20/MA50 宽度、创新低家数和行业同步性。
- 指数估值、换手率和成交额。
- 融资融券汇总及变化。
- Shibor 和可用资金面数据。
- 北向资金历史仅作辅助，数据停止或口径变化时不得作为核心特征。
- 涨跌停、跌停扩散和大额资金流向。

融资、估值等数据按实际披露时间进入特征；无法证明当时可见的数据至少滞后一日。

### 7.2 A 股盘中

- 宽基指数和科技/高 Beta ETF 实时价格。
- 可可靠获得的实时股票截面宽度。
- 同步下跌比例、中位涨跌幅、行业扩散和异常振幅。
- 主源和备用源的价格、时间交叉验证。

真实宽度不可得时标记缺失，不用三个指数投票冒充全市场宽度。

### 7.3 美股盘前与盘中

- S&P 500、Nasdaq、Russell 2000。
- SOXX 或等价半导体代理。
- VIX、VIX3M 及期限结构。
- HYG、LQD 和 Treasury 代理。
- 可获得的美元、信用利差和金融压力指标。
- 跨指数宽度和相对强弱。

Yahoo 可用于历史研究和首版影子运行，不能作为生产盘中硬门控的唯一数据源。只有一个来源时，美股盘中结果标记为影子状态，不联动个股硬门控。

### 7.4 数据状态

- `fresh`：核心输入齐全且时间一致。
- `partial`：非核心因子缺失，可计算但降低可靠度。
- `conflicted`：双源价格或时间超过容差。
- `stale`：超过当前会话允许时效。
- `insufficient`：无法形成有效判断。

`conflicted/stale/insufficient` 不得输出正常绿灯，LLM 不得自行补数。

## 8. 确定性特征

### 8.1 公共特征组

`regime`

- 1/5/20/60/120/252 日收益。
- 20/60/252 日回撤。
- MA20/MA50/MA200 距离和斜率。
- 5/20/60 日已实现波动率及短长波动比。
- 日内振幅、收盘位置和成交异常。

`breadth_and_dispersion`

- 上涨股票比例。
- MA20/MA50 上方比例。
- 创 20/60 日新低比例。
- 行业同步下跌比例。
- 大小盘、科技与宽基相对强弱。

`transition`

- 短波动加速。
- 异常振幅和弱收盘。
- 下跌扩散速度。
- 信用、波动和价格的同步恶化。

### 8.2 A 股特征

- 20 日融资余额增长和融资买入拥挤度。
- 融资余额从高位开始收缩。
- 换手率和估值历史分位。
- 涨跌停差、跌停扩散和连板退潮。
- Shibor 水平、变化和期限压力。

### 8.3 美股特征

- HYG 相对 LQD 的 5/20 日强弱。
- VIX 水平、变化及 `VIX/VIX3M`。
- Russell、Nasdaq、SOXX 相对 S&P 的扩散。
- Treasury、美元和信用压力代理。

每个特征必须声明：数据源、可用时间、缺失策略、方向、单位和版本。

## 9. 标签与模型

### 9.1 首版标签

标签在首次模型训练前冻结：

| 市场 | 1 日急跌 | 3 日急跌 |
|---|---|---|
| A 股 | 下一交易日收益不高于 -4% | 未来 3 日最差累计收益不高于 -6% |
| 美股 | 下一交易日收益不高于 -3% | 未来 3 日最差累计收益不高于 -5% |

`FIRST_SHOCK`：预测时点 20 日回撤大于 -5%。

`CONTINUATION`：预测时点 20 日回撤不大于 -5%。

首版不为两个阶段分别训练模型。每个市场、每个周期一个模型，共四个模型；阶段和交互特征进入同一个模型，避免稀有样本再次拆分。

### 9.2 基线模型

使用正则化逻辑回归：

- 保留真实基准概率。
- 不使用 SMOTE 或合成暴跌样本。
- 缺失值处理和标准化参数只在训练窗口拟合。
- 类别权重如用于训练，最终必须重新校准概率。

项目需要明确增加 `scikit-learn` 依赖。模型和预处理器必须作为同一版本化 pipeline 保存。

### 9.3 挑战模型

LightGBM 只作为离线挑战者，不纳入首版生产依赖。只有连续多个样本外窗口满足全部晋级条件才替换基线：

- Brier Score 更低。
- AUPRC 更高。
- 相同月均预警次数下召回率更高。
- 不能仅依靠单个危机阶段获胜。
- 校准没有系统性高估。

### 9.4 时间序列评估

- 开发期：2000-2012。
- 验证期：2013-2019。
- 首轮隔离测试：2020-2026-07-31。
- 使用扩展窗口或滚动窗口，不使用随机切分。
- 3 日标签的训练与验证边界至少隔离 3 个交易日。
- 概率使用滚动 Platt 校准；样本不足时不使用易过拟合的非参数校准。
- 完成冻结测试后，生产模型可重训至最新可用日期，但必须保留测试报告和训练截止日期。

### 9.5 可靠度与漂移

每次预测附带：

- `reliability_grade`：A/B/C/UNAVAILABLE。
- 近期缺失率和数据漂移。
- 近期概率分布与训练期偏移。
- 最近一次模型训练和校准日期。

模型或数据漂移超过阈值时，禁止红灯仅由概率单独触发，需要硬信号确认。

## 10. M3 Think 推理层

### 10.1 调用时机

- 两市盘前固定报告调用一次。
- 盘中普通轮询不调用。
- 代码首次产生候选橙灯、红灯或橙升红时调用。
- 可选市场上下文识别到重大政策、信用或外部事件时调用。

### 10.2 输入

LLM 只接收压缩后的结构化上下文：

- 当前和前一快照概率、阶段及变化。
- 数据质量和可靠度。
- 前若干项特征贡献及 `evidence_id`。
- 硬信号和反证。
- 可选事件摘要及其时间。
- 当前状态和允许的状态转换。

不发送数千行原始行情或其他 Agent 的完整推理过程。

### 10.3 输出契约

M3 使用 Think 模式，但只返回严格 JSON。结构字段对应 `LLMContextAssessment`。校验要求：

- `recommended_risk_level` 必须属于 GREEN/YELLOW/ORANGE/RED。
- `confidence` 位于 0 到 1。
- 证据 ID 必须存在于输入快照。
- 因果链和反证不能为空。

LLM 仅当 `confidence >= 0.70` 且引用至少两个有效证据时可上调一级。不能降低代码确认的橙灯或红灯，不能在数据不足时输出绿灯。

### 10.4 故障降级

- 超时、关键词屏蔽、空响应和非法 JSON 最多进行一次格式修复重试。
- 第二次失败后立即使用量化基线。
- 量化快照、落库和飞书报告不得依赖 LLM 成功。
- 报告标注推理不可用，但不展示底层错误或 Think 内容。
- 连续失败触发 circuit breaker，冷却期内不再重复调用。

## 11. 灯色与状态机

### 11.1 初始映射

阈值使用各市场、各周期训练窗口的基准率，不能根据隔离测试集回调：

| 状态 | 条件 | 个股联动 |
|---|---|---|
| GREEN | 低于基准概率 2 倍且无硬风险 | OPEN，不额外限仓 |
| YELLOW | 达到基准 2 倍或脆弱性明显积累 | 仅提醒，不修改仓位 |
| ORANGE | 达到基准 4 倍且至少一个压力转折信号 | LIMITED，禁止追涨，新增单票仓位上限 3% |
| RED | 达到基准 8 倍，或踩踏/流动性硬触发 | WAIT，新增仓位 0%，持仓可建议 REDUCE |
| UNKNOWN | 数据冲突、陈旧或不足 | WAIT，新增仓位 0%；只表示无法可靠判断，不等同于市场红灯 |

1 日和 3 日概率取更严格的映射结果。硬触发器必须版本化，并在实施计划中逐项定义；首版至少覆盖市场级异常振幅、跌停/同步下跌扩散、信用与波动同步恶化。核心数据源失效只能进入 UNKNOWN，不能作为市场红灯证据。

### 11.2 状态转换

- 升级立即生效。
- 同一等级不重复推送。
- ORANGE/RED 降级需要连续两次有效快照满足恢复条件。
- 盘中恢复只落库并更新个股门控，不发送恢复消息。
- 下一次盘前报告说明恢复及持续时间。
- LLM 上调结果也必须经过状态机，不能跳过幂等和恢复规则。

### 11.3 与个股分析集成

`FinalWarningDecision` 生成统一 `effective_gate`，与个股结构、买入时机和现有 `market_risk` 取最严格结果：

- YELLOW 不修改个股长期评级或仓位。
- ORANGE 只约束短期动作和新增仓位。
- RED 对短期新仓强制 WAIT，已持仓场景允许 REDUCE。
- 长期 BUY/OVERWEIGHT/HOLD 等评级保持独立。

## 12. 调度与推送

### 12.1 统一 Runner

定时系统每 5 分钟唤醒一次 runner。runner 使用交易所日历、市场时区和幂等记录判断是否实际执行，替代按北京时间硬编码多条 cron。

A 股：

- 08:30 启动盘前任务，允许等待和重试 T-1 数据，目标 08:35 前推送。
- 09:35-11:25、13:05-14:55 每 5 分钟内部计算。

美股：

- 美东 08:30 盘前固定报告，自动处理冬夏令时。
- 开盘 5 分钟后至收盘前每 5 分钟内部计算。

盘中仅在首次进入 ORANGE/RED 或 ORANGE 升 RED 时推送。夜间美股升级仍立即推送。

### 12.2 幂等

幂等键至少包含：

- market
- as_of_date
- session_slot 或 evaluation_time_bucket
- decision_level
- transition
- model_version

重复 runner、重试和进程恢复不得生成重复飞书消息。

### 12.3 报告

盘前报告固定顺序：

1. 当前灯色与动作。
2. 未来 1 日/3 日急跌概率和基准概率。
3. FIRST_SHOCK 或 CONTINUATION。
4. 与前一交易日变化。
5. 前三项量化贡献。
6. M3 场景判断、因果链和核心反证。
7. 数据时间、可靠度和模型版本。

盘中升级报告只显示变化、触发证据和立即动作。

## 13. 存储

在现有 harness SQLite 中新增独立表：

`market_warning_feature_snapshots`

- 唯一键：market、as_of_time、feature_version。
- 保存数据状态、特征 JSON、证据 JSON 和源时间摘要。

`market_warning_predictions`

- 唯一键：feature_snapshot_id、model_version、horizon。
- 保存概率、基准率、阶段、可靠度和贡献。

`market_warning_reasoning`

- 保存结构化 LLM 结论、状态、模型名和错误分类。
- 不保存 Think 原文。

`market_warning_decisions`

- 保存基线灯色、最终灯色、转换、门控、仓位上限和原因。

`market_warning_alerts`

- 保存幂等键、推送状态、发送时间和错误摘要。

`market_warning_model_registry`

- 保存模型版本、训练截止日期、特征版本、校准版本、指标和文件校验值。

模型 pipeline 保存在 `harness_data/models/market_warning/`，不写入数据库 blob。

## 14. 错误处理与可观测性

- 每次 runner 输出结构化运行摘要。
- 数据获取、特征、模型、LLM、数据库和通知分别记录阶段状态。
- 推送失败不回滚快照和决策。
- 模型文件缺失或校验失败时，不得临时训练；使用 UNKNOWN 或已登记的上一个稳定模型。
- 数据源和 LLM circuit breaker 状态写入运行摘要。
- 日志必须包含实际数据时间，但不得输出 API 密钥、访问令牌或完整敏感响应。

## 15. 测试与验收

### 15.1 单元测试

- 市场和阶段标签。
- 点时可见数据及披露滞后。
- 特征方向、缺失和异常值处理。
- 概率映射与可靠度。
- LLM JSON 校验、证据引用和上调限制。
- 状态升级、连续两次恢复及幂等。
- 个股门控不修改长期评级。

### 15.2 集成测试

- Tushare、实时源和美股适配器的 fresh/partial/stale/conflicted。
- 交易日历、节假日和美股冬夏令时。
- SQLite 全流程和重复 runner。
- 飞书固定报告、升级报告及失败重试。
- M3 超时、关键词屏蔽、空响应、非法 JSON 和 circuit breaker。

### 15.3 模型验收

- 证明无随机切分和标签边界穿越。
- Brier Score 优于恒定基准概率。
- AUPRC 高于正样本基准。
- 在相同月均橙/红预警次数下，召回率优于旧 `market_risk`。
- 结果不能仅由 2008、2015 或 2020 单个危机贡献。
- 校准图和分箱实际发生率无明显系统性高估。
- 橙/红月均推送次数受控，并在报告中同时展示误报和漏报。

### 15.4 端到端验收

- A 股和美股盘前各生成一份完整报告。
- 盘中无升级时不推送。
- 候选 ORANGE/RED 能触发 M3，并在失败时正常降级。
- 红灯正确联动个股 WAIT/REDUCE。
- 任一 LLM 故障场景下量化报告生成率为 100%。
- 数据陈旧不得再次出现旧缓存被标记为 fresh 的问题。

## 16. 首版交付范围

首版按以下顺序落地：

1. 领域对象、端口、数据库迁移和数据质量规则。
2. A 股/美股确定性特征及点时历史快照构建。
3. 四个逻辑回归 pipeline、校准、注册和隔离测试报告。
4. Warning Policy、状态机和个股门控集成。
5. M3 Think 结构化推理、校验和故障降级。
6. 统一 runner、盘前/事件驱动飞书模板和幂等。
7. A 股生产冒烟，美股单源时先影子运行。
8. 数据源满足生产要求后再开启美股盘中硬门控。

## 17. 成功标准

系统成功不等于“预测每一次暴跌”，而是：

- 用点时可见数据输出经过样本外校准的风险概率。
- 相比旧系统更早识别部分第一段风险，同时保持续跌识别能力。
- 明确展示误报、漏报和可靠度，不用高频报警伪造召回率。
- 在数据或 LLM 故障时保持可用和保守。
- 将市场预警稳定转化为可解释、可审计的短期动作约束。
