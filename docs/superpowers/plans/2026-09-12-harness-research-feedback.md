# Harness 研究复盘与参数实验 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 建成可信复盘、单参数对照实验、证据驱动迭代提案的最小闭环。

**Architecture:** 扩展现有 harness，沿用 SQLite、报告目录和每日/每周入口。数据与评价由确定性代码执行；实验单独存储；LLM 只消费精简证据包并生成提案，不直接更改生产配置。

**Tech Stack:** 项目 `.venv`、Python、SQLite、现有 pytest/交易日历/LLM 客户端与飞书交付能力；无新增服务。

**Spec:** `docs/superpowers/specs/2026-09-12-harness-research-feedback-design.md`。

**当前执行范围：** 因报告样本较少，以下完整平台任务暂缓。优先采用 `2026-09-12-harness-small-sample-v1.md`，不得将本清单全部视为本次交付。

## Global Constraints

- 本次文档为计划；以下所有实施步骤尚未执行。
- 不改每日 21:00 与周六 10:00 调度，不改大盘预警职责和推送策略，不移除现有缓存预热。
- 不覆盖历史报告、不复制密钥或隐藏思考原文、不将试验数据写入生产统计。
- 保留工作区现有改动；当前 `main.py` 的用户修改不得被覆盖。
- WAIT 不等于预测下跌或已经买入；无持仓、无可判定成交就不虚构账户收益。
- 生成版本、输入时点、实际配置必须留档；历史不完整数据不得伪装成可重放样本。
- 所有实验先登记协议，旧日期 LLM 重跑只作诊断；未经验证的候选不改变主分析结果。
- 不新增依赖，优先复用现有 `exchange_calendars` 和 provider 配置；必要扩展另作说明。

---

## 文件职责

沿用 `archive.py` 提取与入库、`truth_fetcher.py` 真值获取、`backtest.py` 聚合、`review.py` 证据与文案、`weekly_review.py` 编排。禁止将这些全部重写成新框架。

按任务需要新增 `run_manifest.py`（生成档案）、`evaluation.py`（评价契约与口径）、`experiments.py`（实验登记与对照）、`iteration_review.py`（受限 LLM 提案）。策略参数定义位于 `tradingagents/strategy_parameters.py`，避免生产策略依赖 harness 包。

第一阶段交付任务 1-4；第二阶段交付任务 5-6；第三阶段交付任务 7；任务 8 验收完整闭环。每任务先补失败测试、实现最小改动、通过对应测试并独立审查后提交。

### 任务 1：生成时运行档案

修改：`main.py`、`tradingagents/graph/trading_graph.py`、`tradingagents/harness/archive.py`、`db.py`、`schema.sql`。

新增：`tradingagents/harness/run_manifest.py`、`test_run_manifest.py`。

接口：`start_manifest(report_dir, metadata, effective_config) -> dict`、`finish_manifest(report_dir, result) -> dict`；归档优先读取生成清单，缺失明确标 legacy。允许失败运行仅有清单、没有 decision.md。

- [ ] 测试：生成后 Git HEAD 改变不影响版本；相同配置稳定哈希；实际角色覆盖值入档；密钥脱敏；失败运行被计数；重复归档幂等。
- [ ] 用 `.venv/bin/python -m pytest tradingagents/harness/test_run_manifest.py -q` 验证测试先失败。
- [ ] 接入配置解析后、分析开始与成功/失败出口；记录相关输入快照/时间与最终交接摘要引用，不只记录文件路径或哈希。
- [ ] 成本/耗时复用已有计量位置，缺失值写未知而非零；增量迁移在数据库副本验证，保留旧记录。
- [ ] 同一测试通过后审查敏感信息与原有主分析行为，再提交本任务。

### 任务 2：市场时钟与真值状态

修改：`tradingagents/harness/truth_fetcher.py`、`price_cache.py`、`daily_update.py`、`extractor.py`。

新增：`tradingagents/harness/test_truth_fetcher.py`。

参考：`tradingagents/harness/market_warning/calendars.py` 的现有交易日历能力，只复用计算，不引入预警服务执行副作用。

接口：真值结果返回 `target_at`、`as_of`、`status`、`reason`、`benchmark_id`、`evaluation_version`。区分未到期、到期缺数据、源不支持、抓取失败和完成。

- [ ] 测试 A 股节假日、美股夏令时/休市/提前收市、周末报告、跨年日历范围；不要照搬当前日历 helper 的 2026 年固定上限。
- [ ] 测试已经过期但价格缓存不足必须为数据缺失；失败有界重试；不同市场基准与同一收益锚点；数据新鲜度不由退出码代替。
- [ ] 执行 `.venv/bin/python -m pytest tradingagents/harness/test_truth_fetcher.py -q`，先观察失败，再实现状态和时间规则。
- [ ] 保留旧 outcome 原始值，新评价单独版本化；到期日由市场日历决定，不用已有数据行数决定是否到期。
- [ ] 同一测试通过，使用旧美股缺数据记录做离线夹具，确认不再标正常未到期，审查后提交。

### 任务 3：拆分预测、操作、长期研究评价

修改：`tradingagents/harness/archive.py`、`truth_fetcher.py`、`backtest.py`、`db.py`、`schema.sql`。

新增：`tradingagents/harness/evaluation.py`、`test_evaluation.py`。

扩展：`tradingagents/harness/test_backtest_metrics.py`。

接口：`evaluate_run(prediction, manifest, truth, protocol) -> dict`，输出三个独立 scorecard、覆盖/成熟状态与诊断原因；显式区分研究信号收益代理和模拟执行收益。

- [ ] 先写核心失败测试：长期 OVERWEIGHT + WAIT + 新仓 0% 不生成多头交易；缺短期预测不拿长期评级补齐；同一报告多个期限不增加独立报告数。
- [ ] 补到期 3 交易日、3/6/12 日历月、未成交、无持仓、同日止盈止损顺序不明、公司行动/基准同口径的夹具。
- [ ] 执行 `.venv/bin/python -m pytest tradingagents/harness/test_evaluation.py tradingagents/harness/test_backtest_metrics.py -q` 确认先失败。
- [ ] 实现固定日频假设下的模拟与不确定分支；报告可用前的当日高低价不得用于可执行结果；无法精确执行的盘中样本明确降级。
- [ ] 聚合区分市场、可追溯版本、评价版本；记录报告数、决策组数、结果行数和弃权率，不再输出伪账户 PnL。
- [ ] 同组测试通过后，以 run 584 的中际旭创 WAIT 报告作为回归夹具，只验证语义和口径，不根据后来涨跌改评级，审查后提交。

### 任务 4：可信周报与问题证据包

修改：`tradingagents/harness/review.py`、`weekly_review.py`、`test_weekly_review.py`。

新增：`tradingagents/harness/test_review_evidence.py`。

接口：`build_review_packet(metrics, cases, health, experiment_status) -> dict`；此处不调用 LLM。案例包含原始 run/evidence ID、版本、正反例和已验证数值。

- [ ] 测试脚本成功但行情过期时健康度不能全绿；样本 5 不叫统计显著；旧版与当前版分开；重复切片合并；本周新增与累计分开。
- [ ] 执行 `.venv/bin/python -m pytest tradingagents/harness/test_review_evidence.py tradingagents/harness/test_weekly_review.py -q`，先失败再实施。
- [ ] 生成三套独立摘要、证据等级和最多三个问题；无足够证据输出继续观察，不硬给参数建议。
- [ ] 飞书文本使用短段落，区分本地留档路径与可交付附件，沿用已有通知适配；测试使用假发送器，避免误发。
- [ ] 测试通过、第一阶段副本冒烟通过后审查提交；此时即可上线可信复盘，不等待参数研究完成。

### 任务 5：只暴露首个参数，保持默认行为

修改：`tradingagents/dataflows/profile_calc.py`、`test_profile_calc.py`、`tradingagents/default_config.py` 以及查明后的 `compute_short_term_structure` 实际调用方。

新增：`tradingagents/strategy_parameters.py`、`tradingagents/dataflows/test_strategy_parameters.py`。

接口：`resolve_strategy_parameters(config) -> dict`；首个参数为 `entry.breakout_volume_ratio_min`，默认 1.3，首轮候选 1.1/1.3/1.5。生产策略使用参数解析结果，manifest 记录同一个结果。

- [ ] 先追踪全部实际调用点；先写默认不变、非法值拒绝、配置传递完整、参数确实改变临界样本的失败测试。
- [ ] 执行 `.venv/bin/python -m pytest tradingagents/dataflows/test_strategy_parameters.py tradingagents/dataflows/test_profile_calc.py -q` 验证失败。
- [ ] 用可选关键字参数/现有 config 注入替换这一处门槛，禁止同时抽取全部 ATR、资金流、估值常量。
- [ ] 加入破位、持续亏损、风险禁入不会因放宽量能而被绕开的测试；执行 `.venv/bin/python -m pytest tradingagents/agents/managers/test_entry_timing.py -q`。
- [ ] 以上测试通过、默认值与既有夹具结果一致后审查提交；此任务不改变默认策略。

### 任务 6：候选对照与前瞻影子记录

新增：`tradingagents/harness/experiments.py`、`test_experiments.py`。

修改：`tradingagents/harness/db.py`、`schema.sql`；复用任务 1/3 的清单与评价接口。

接口：`register_experiment(protocol) -> str`、`replay_local(experiment_id, snapshots) -> dict`、`compare_paired(experiment_id, outcomes) -> dict`。存储协议哈希、全部尝试、基线/候选配对、检查历史和晋级结论。

- [ ] 测试禁止未知参数、改变锁定测试协议、混用评价版本、输入穿越、缺失快照和实验写入生产统计。
- [ ] 测试局部结构变化不能自动记为 PM 策略改善；重复快照不增加独立样本；相关样本按协议分组；风险/覆盖边界失败不能晋级。
- [ ] 执行 `.venv/bin/python -m pytest tradingagents/harness/test_experiments.py -q`，先失败再实现单参数三候选重放。
- [ ] 提供时间顺序拆分、跨边界标签剔除、冻结测试窗口和配对不确定性统计；边界未预注册或样本不足只能进入观察状态。
- [ ] 影子运行先支持无需 LLM 的候选路径。需重跑下游 agent 时默认禁用付费调用，先记录受影响阶段与预算需求。
- [ ] 测试通过并输出一份真实输入上的敏感性报告；即使没有合格快照，也必须清楚输出不可重放原因，不伪造结果。审查后提交。

### 任务 7：每周有限调用的迭代提案

新增：`tradingagents/harness/iteration_review.py`、`test_iteration_review.py`。

修改：`tradingagents/harness/review.py`、`weekly_review.py`；读取现有 `tradingagents/agents/utils/research_evidence_node.py` 的贡献台账，不重建一套角色协议。

接口：`propose_iterations(packet, llm, budget) -> dict`；只消费任务 4 的证据包，输出 schema 校验后的建议或规则版降级摘要。

- [ ] 测试不存在的案例 ID、LLM 改写的统计值被拒绝；引用率不得标注因果贡献；无新证据不调用；超预算不调用；模型失败不阻塞确定性周报。
- [ ] 执行 `.venv/bin/python -m pytest tradingagents/harness/test_iteration_review.py -q`，先失败再实现。
- [ ] 复用统一 provider 配置；最多一次周复盘加一次格式修复，预留每次请求的最坏 token 成本，推理 token 计入总预算。
- [ ] 输出问题层次、正反例、代码/参数位置、最小改动、实验与回归要求、风险边界及回滚；输出“暂不调整”是正常结果。
- [ ] 用假客户端覆盖所有控制分支；联网小冒烟只进行一次提案生成，不批量重跑股票。审查通过后提交。

### 任务 8：完整验收、兼容性与交付

- [ ] 为前三阶段分别运行聚焦测试；完整确定性回归命令为 `.venv/bin/python -m pytest tradingagents/harness tradingagents/dataflows/test_profile_calc.py tradingagents/dataflows/test_strategy_parameters.py tradingagents/agents/managers/test_entry_timing.py tradingagents/agents/utils/test_handoff.py -q`。检查并隔离已有联网测试，不把网络不可用误报为策略失败。
- [ ] 在临时数据库副本中完成归档 -> 到期标签 -> 三套评价 -> 周报 -> 单参数实验 -> 提案；保留生产数据库、调度和报告不变。
- [ ] 历史夹具至少覆盖：中际旭创 WAIT、重复同快照报告、到期缺美股数据、实际新建仓、未知持仓、长期未到期、摘要失败。
- [ ] 显示一份手机友好的样例周报和一份结构化迭代提案；使用可用飞书附件能力交付需先验证发送路径，不能声称本地 Markdown 路径能在手机打开。
- [ ] 复用下一次正常股票分析验证清单采集，必要时才新增一次 `.venv` 真实股票冒烟；无异常耐心等待，检查最终报告、配置与费用，不频繁重复调用。
- [ ] 文档化运行方式、状态解释、预算和回滚；报告“工程验收完成”与“参数优势尚待前瞻验证”各自状态。
- [ ] 每阶段走已有授权的分支/PR/测试/合并流程；策略默认值仍需满足已批准实验协议的晋级条件，不随平台上线自动调整。

## 暂不实施

多参数大网格、自动优化所有 agent、以完整思考原文训练、历史 LLM 重跑冒充无穿越回测、分钟级虚拟精确成交、新购数据源、自动放松风控。待首个闭环有足够证据后另立计划。
