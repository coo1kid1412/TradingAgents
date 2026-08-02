# 大盘骤跌预警规则生产版 V1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 交付 A 股规则生产预警 V1：08:30 固定盘前报告，盘中每 10 分钟零 LLM 扫描，橙/红跨级立即推送，M3 仅在告警完成后补充解释；美股继续影子运行。

**Architecture:** 保留现有模型服务和概率持久化边界，新增独立的规则评估领域对象、纯函数规则引擎、规则服务与激活状态。数据、规则、状态机、持久化和推送组成快路径；影子模型与 M3 组成不影响告警结论的慢路径。规则通知和个股硬门控使用两个独立激活阶段，默认只开启通知灰度。

**Tech Stack:** Python 3.11、dataclasses、SQLite、pandas、Tushare Pro、exchange-calendars、standard-library unittest、现有飞书发送适配器、MiniMax M3。

## Global Constraints

- 设计依据：[规则生产版 V1 Spec](/Users/lailixiang/WorkSpace/QoderWorkspace/TradingAgents/.worktrees/dual-market-crash-warning/docs/superpowers/specs/2026-08-02-market-warning-rule-v1-design.md)。实现与 Spec 冲突时先修订 Spec，不在代码中暗改口径。
- 所有开发和回归命令使用项目 `.venv/bin/python`。
- 规则分数只能进入规则评估表和规则报告，禁止写入 `market_warning_predictions.probability`。
- 快路径禁止调用任何 LLM；M3 只能在确定性告警发送成功或已幂等确认后调用。
- A 股生产告警必须使用扫描时点可见的数据；T-1 数据只能作为盘前基线。
- Tushare 5000 积分不代表已开通全市场实时日线 `rt_k`。必须执行权限探针，不能假设可用。
- 横截面实时权限不足、覆盖率低于 80% 或陈旧超过 5 分钟时，禁止触发依赖市场宽度或跌停占比的红灯。
- `rule_v1/notify` 与 `rule_v1/gate` 分开激活；V1 首次部署只允许通知灰度，完成 10 个 A 股交易日审计后才能评估硬门控。
- 保留现有模型链路测试；新规则链路不得破坏旧模型回放、训练和报告能力。

---

## Task 1: 建立规则领域对象与持久化边界

**Files:**
- Modify: `tradingagents/harness/market_warning/domain.py`
- Modify: `tradingagents/harness/market_warning/ports.py`
- Modify: `tradingagents/harness/schema.sql`
- Modify: `tradingagents/harness/db.py`
- Modify: `tradingagents/harness/market_warning/adapters/sqlite_repository.py`
- Modify: `tradingagents/harness/market_warning/test_domain.py`
- Modify: `tradingagents/harness/market_warning/test_repository.py`

- [ ] **Step 1: 写领域对象失败测试**

  在 `test_domain.py` 覆盖：
  - `TriggeredRule` 的层级只能是 `VULNERABILITY/PRESSURE/CONTINUATION/HARD_TRIGGER`。
  - `severity_points` 必须是 0-2 的整数，`evidence_ids` 去重且不可为空。
  - `RuleRiskAssessment.risk_score` 必须是 0-10 的有限数，但不接受 `probability` 字段。
  - `FinalWarningDecision.decision_source` 只能是 `model` 或 `rule_v1`。
  - `RunnerResult` 可同时携带主规则评估和影子模型评估。

- [ ] **Step 2: 运行领域测试并确认失败**

  Run: `.venv/bin/python -m unittest tradingagents/harness/market_warning/test_domain.py -q`

- [ ] **Step 3: 最小实现领域对象**

  新增 `RuleLayer`、`DecisionSource`、`TriggeredRule`、`RuleRiskAssessment`；给 `FinalWarningDecision` 增加默认兼容的 `decision_source=DecisionSource.MODEL`；给 `RunnerResult` 增加 `rule_assessment` 和 `shadow_quant_assessment`，旧 `quant_assessment` 保留给模型链路。

- [ ] **Step 4: 写 schema 与 repository 失败测试**

  在 `test_repository.py` 断言：
  - 新表 `market_warning_rule_assessments` 与 `market_warning_rule_registry` 可幂等创建。
  - 旧数据库可迁移 `market_warning_decisions.decision_source`、`rule_assessment_id`、`shadow_prediction_ids_json`。
  - 规则决策无需伪造两条 prediction 记录即可保存和回放。
  - 规则评估 JSON 完整保留触发规则、缺失可选组和延迟。
  - 规则注册表同一时刻最多一个 `notification_active` 和一个 `gate_active` 版本。

- [ ] **Step 5: 运行 repository 测试并确认失败**

  Run: `.venv/bin/python -m unittest tradingagents/harness/market_warning/test_repository.py -q`

- [ ] **Step 6: 实现 schema、migration 与 repository API**

  `save_decision()` 改为接受 `prediction_ids=()`、`rule_assessment_id=None`、`shadow_prediction_ids=()`；按 `decision_source` 校验互斥关系：
  - `model` 必须有唯一的 1d/3d prediction。
  - `rule_v1` 必须有属于同一 feature snapshot 的 rule assessment，主 prediction 可为空。
  - 影子 prediction 只审计，不进入主决策来源。

  在 `ports.py` 增加 `save_rule_assessment()`、`register_rule_engine()`、`activate_rule_engine()`、`load_active_rule_engine()`、`load_rule_evaluation()` 协议。

- [ ] **Step 7: 运行领域与 repository 测试**

  Run: `.venv/bin/python -m unittest tradingagents/harness/market_warning/test_domain.py tradingagents/harness/market_warning/test_repository.py -q`

- [ ] **Step 8: 提交**

  Run: `git add tradingagents/harness/market_warning/domain.py tradingagents/harness/market_warning/ports.py tradingagents/harness/schema.sql tradingagents/harness/db.py tradingagents/harness/market_warning/adapters/sqlite_repository.py tradingagents/harness/market_warning/test_domain.py tradingagents/harness/market_warning/test_repository.py && git commit -m "feat: add rule warning domain and persistence"`

---

## Task 2: 实现带校验和的 A 股纯规则引擎

**Files:**
- Create: `tradingagents/harness/market_warning/rule_manifest_v1.json`
- Create: `tradingagents/harness/market_warning/rule_policy.py`
- Create: `tradingagents/harness/market_warning/test_rule_policy.py`

- [ ] **Step 1: 把 Spec 阈值固化成 JSON 清单**

  清单必须包含 `engine_version=rule-v1.0.0`、所有阈值、分层上限、灯号条件、必需/可选特征名和单位；运行时计算 SHA-256 并写入评估与注册表。

- [ ] **Step 2: 写逐条规则和边界失败测试**

  使用工厂构造固定 `FeatureSnapshot`，覆盖每条脆弱性、压力、续跌与硬触发规则；每个阈值至少测试 `threshold-epsilon`、`threshold`、`threshold+epsilon`。额外覆盖：
  - `new_low_20d_pct` 只展示、不计分。
  - 可选融资/换手缺失只降低可靠度，不产生 `UNKNOWN`。
  - 指数核心字段缺失产生 `UNKNOWN`。
  - 横截面无效时屏蔽 breadth/limit-down 红灯。
  - A 股规则拒绝美股 snapshot。
  - `risk_score` 只做审计，弱信号叠加不能机械产生红灯。

- [ ] **Step 3: 运行规则测试并确认失败**

  Run: `.venv/bin/python -m unittest tradingagents/harness/market_warning/test_rule_policy.py -q`

- [ ] **Step 4: 实现纯函数**

  实现：
  - `load_rule_manifest(path) -> RuleManifest`
  - `evaluate_a_share_rules(snapshot, manifest, previous_assessment=None) -> RuleRiskAssessment`
  - `manifest_sha256(path) -> str`

  评估器不得访问网络、数据库、环境变量或时钟；`evaluation_latency_ms` 由调用方写入，不在纯函数内读系统时间。

- [ ] **Step 5: 运行规则与现有 policy 测试**

  Run: `.venv/bin/python -m unittest tradingagents/harness/market_warning/test_rule_policy.py tradingagents/harness/market_warning/test_policy.py -q`

- [ ] **Step 6: 提交**

  Run: `git add tradingagents/harness/market_warning/rule_manifest_v1.json tradingagents/harness/market_warning/rule_policy.py tradingagents/harness/market_warning/test_rule_policy.py && git commit -m "feat: implement deterministic A-share warning rules"`

---

## Task 3: 建立规则历史评估与通知晋级门槛

**Files:**
- Create: `tradingagents/harness/market_warning/rule_evaluation.py`
- Create: `tradingagents/harness/market_warning/test_rule_evaluation.py`
- Modify: `tradingagents/harness/market_warning/readiness.py`
- Modify: `tradingagents/harness/market_warning/test_readiness.py`

- [ ] **Step 1: 写历史指标失败测试**

  固定小型 DataFrame，验证：
  - 2000-2012、2013-2019、2020-2026 时间切分。
  - 只统计 GREEN/YELLOW 进入 ORANGE/RED 的新告警，不把同级维持重复计数。
  - FIRST_SHOCK 与 CONTINUATION 分开统计 base rate、precision、recall、lift。
  - 月均告警次数、单一危机真阳性贡献占比、相同提醒预算对比。
  - 特征 `available_at > as_of_time` 时直接拒绝评估，防止数据穿越。

- [ ] **Step 2: 运行测试并确认失败**

  Run: `.venv/bin/python -m unittest tradingagents/harness/market_warning/test_rule_evaluation.py tradingagents/harness/market_warning/test_readiness.py -q`

- [ ] **Step 3: 实现评估 CLI 与结构化产物**

  CLI 输入已有历史特征文件与规则清单，输出 JSON：各阶段指标、首跌/续跌分组、提醒预算、危机贡献、清单 SHA-256，并明确标记 2020-2026 为 `previously_observed_holdout=true`。

  Run example: `.venv/bin/python -m tradingagents.harness.market_warning.rule_evaluation --market a_share --manifest tradingagents/harness/market_warning/rule_manifest_v1.json --output harness_data/models/market_warning/rule-v1-evaluation.json`

- [ ] **Step 4: 扩展 readiness 模式**

  `check_production_readiness(..., mode)` 支持：
  - `model`：保持四模型校验不变。
  - `rule_v1/notify`：要求冻结期 lift > 2、月均新橙/红 <= 6、清单 checksum 匹配、数据冒烟通过、运行时基准通过。
  - `rule_v1/gate`：在 notify 基础上要求危机贡献 <= 50%、10 个 A 股交易日 soak 完成且无严重数据故障。

- [ ] **Step 5: 运行测试**

  Run: `.venv/bin/python -m unittest tradingagents/harness/market_warning/test_rule_evaluation.py tradingagents/harness/market_warning/test_readiness.py -q`

- [ ] **Step 6: 提交**

  Run: `git add tradingagents/harness/market_warning/rule_evaluation.py tradingagents/harness/market_warning/test_rule_evaluation.py tradingagents/harness/market_warning/readiness.py tradingagents/harness/market_warning/test_readiness.py && git commit -m "feat: evaluate and gate rule warning engine"`

---

## Task 4: 探测 Tushare 实时权限并构建盘前横截面基线

**Files:**
- Create: `tradingagents/harness/market_warning/adapters/tushare_realtime_breadth.py`
- Create: `tradingagents/harness/market_warning/test_realtime_breadth.py`
- Modify: `tradingagents/harness/market_warning/adapters/realtime_quote.py`
- Modify: `tradingagents/harness/market_warning/test_data_adapters.py`

- [ ] **Step 1: 写权限探针失败测试**

  fake Tushare client 覆盖：`rt_k` 成功、接口不存在、权限拒绝、空数据、字段缺失、超时。探针返回结构化状态 `available/permission_denied/unavailable/invalid_payload`，不得把异常吞成“无风险”。

- [ ] **Step 2: 写基线与实时横截面失败测试**

  覆盖：
  - 盘前基线仅使用最近已完成交易日的 `daily`/`daily_basic`/`stock_basic`。
  - `stk_limit` 在 08:30 后单独加载并缓存；首个盘中扫描前重试一次。
  - `rt_k` 使用官方通配符一次或少量分片提取全市场，合并时每只股票只取不晚于扫描时点的最新记录。
  - 输出 `last/pre_close/data_time/ma20/low_20d/industry/down_limit/source`。
  - 覆盖率低于 80% 或最大数据时间陈旧超过 5 分钟时标记横截面不可用于红灯。
  - T-1 行情不能进入 `last` 充当盘中现价。

- [ ] **Step 3: 运行测试并确认失败**

  Run: `.venv/bin/python -m unittest tradingagents/harness/market_warning/test_realtime_breadth.py tradingagents/harness/market_warning/test_data_adapters.py -q`

- [ ] **Step 4: 实现缓存与批量适配器**

  实现：
  - `probe_rt_k_permission(pro, symbols, as_of_time)`
  - `build_premarket_baseline(pro, trade_date, cache_root)`
  - `load_realtime_cross_section(pro, baseline, as_of_time)`

  适配官方 `rt_k` 通配符全市场请求与 6000 行返回上限；默认使用 `3*.SZ,6*.SH,0*.SZ,9*.BJ`，必要时按交易所分片，所有批次记录实际来源与 `trade_time`。权限不足时抛出可分类异常 `RealtimePermissionUnavailable`，不回退到 T-1 横截面。`rt_min` 仅保留作个股诊断，不进入 V1 全市场快路径。

- [ ] **Step 5: 接入现有 `_cross_section_points`**

  `RealtimeAShareDataAdapter` 默认可注入新 loader；生产工厂只有在探针成功时启用，失败时仍提供指数快照并把横截面可靠度降级为 C/INSUFFICIENT，使 breadth 规则自动失效。

- [ ] **Step 6: 运行测试**

  Run: `.venv/bin/python -m unittest tradingagents/harness/market_warning/test_realtime_breadth.py tradingagents/harness/market_warning/test_data_adapters.py -q`

- [ ] **Step 7: 提交**

  Run: `git add tradingagents/harness/market_warning/adapters/tushare_realtime_breadth.py tradingagents/harness/market_warning/test_realtime_breadth.py tradingagents/harness/market_warning/adapters/realtime_quote.py tradingagents/harness/market_warning/test_data_adapters.py && git commit -m "feat: add validated realtime A-share breadth data"`

---

## Task 5: 实现规则快路径服务和状态机整合

**Files:**
- Create: `tradingagents/harness/market_warning/rule_service.py`
- Create: `tradingagents/harness/market_warning/test_rule_service.py`
- Modify: `tradingagents/harness/market_warning/policy.py`
- Modify: `tradingagents/harness/market_warning/test_policy.py`

- [ ] **Step 1: 写服务顺序和失败关闭测试**

  使用 spy ports 断言严格顺序：
  `load data -> quality -> features -> evaluate rules -> persist rule -> state transition -> persist decision -> write report -> notify -> optional shadow -> optional M3`。

  覆盖：
  - 首次 ORANGE、直接 RED、ORANGE->RED 推送。
  - YELLOW、同级维持、恢复不推送。
  - `UNKNOWN` 不清除最后确认的橙/红状态。
  - 橙/红需要两个连续有效扫描才能降级。
  - 规则、数据库或核心数据异常时输出 `UNKNOWN`，不输出 GREEN。
  - 影子模型异常不改变主决策与错误码。
  - M3 在通知后才运行，超时/异常/格式错误不改变灯号、仓位和推送结果。

- [ ] **Step 2: 运行测试并确认失败**

  Run: `.venv/bin/python -m unittest tradingagents/harness/market_warning/test_rule_service.py tradingagents/harness/market_warning/test_policy.py -q`

- [ ] **Step 3: 复用状态机，新增规则入口**

  在 `policy.py` 增加 `build_rule_decision(rule, previous)`，只把 `RuleRiskAssessment.risk_level` 送入现有恢复迟滞逻辑。动作映射固定：GREEN/YELLOW 不做硬限制，ORANGE 进入防守门，RED 关闭新增仓位；具体文案沿用现有项目约定。

- [ ] **Step 4: 实现 `RuleMarketWarningService`**

  服务以依赖注入方式接收数据、特征策略、规则评估器、repository、reporter、notifier、可选 shadow evaluator 与 post-alert reasoning。快路径用 `time.monotonic_ns()` 记录耗时；reasoning 延迟不计入快路径预算。

- [ ] **Step 5: 运行测试**

  Run: `.venv/bin/python -m unittest tradingagents/harness/market_warning/test_rule_service.py tradingagents/harness/market_warning/test_policy.py tradingagents/harness/market_warning/test_service.py -q`

- [ ] **Step 6: 提交**

  Run: `git add tradingagents/harness/market_warning/rule_service.py tradingagents/harness/market_warning/test_rule_service.py tradingagents/harness/market_warning/policy.py tradingagents/harness/market_warning/test_policy.py && git commit -m "feat: add fail-closed rule warning service"`

---

## Task 6: 提升规则报告和飞书告警可读性

**Files:**
- Modify: `tradingagents/harness/market_warning/reporting.py`
- Modify: `tradingagents/harness/market_warning/test_reporting.py`
- Modify: `tradingagents/harness/market_warning/adapters/feishu_notifier.py`
- Modify: `tradingagents/harness/market_warning/test_runner.py`

- [ ] **Step 1: 写报告内容失败测试**

  盘前报告首屏必须出现：灯号、立即操作、规则生产模式、数据时间、可靠度、规则分数不是概率。盘中告警必须用 `【橙灯：提前防守】` 或 `【红灯：风险确认】` 开头，并突出入场门、仓位上限、持仓动作、前三条触发规则和相对上一状态的变化。

  明确断言报告不包含：原始 Think、内部日志、`crash probability`、不可用模型的 0% 伪概率。

- [ ] **Step 2: 写幂等键失败测试**

  规则告警幂等键使用 `market/date/slot/level/transition/engine_version/manifest_sha256`；模型告警保持旧键兼容。盘前固定推送，盘中只按跨级推送。

- [ ] **Step 3: 运行测试并确认失败**

  Run: `.venv/bin/python -m unittest tradingagents/harness/market_warning/test_reporting.py tradingagents/harness/market_warning/test_runner.py -q`

- [ ] **Step 4: 实现双来源渲染和通知**

  `render_premarket_report`、`render_upgrade_report` 按 `decision_source` 分派，规则版只读取 `rule_assessment`。M3 最近解释放在报告末尾，缺失时整个区块省略。

- [ ] **Step 5: 运行测试**

  Run: `.venv/bin/python -m unittest tradingagents/harness/market_warning/test_reporting.py tradingagents/harness/market_warning/test_runner.py -q`

- [ ] **Step 6: 提交**

  Run: `git add tradingagents/harness/market_warning/reporting.py tradingagents/harness/market_warning/test_reporting.py tradingagents/harness/market_warning/adapters/feishu_notifier.py tradingagents/harness/market_warning/test_runner.py && git commit -m "feat: render actionable rule warning alerts"`

---

## Task 7: 实现 10 分钟调度、数据库租约与故障提醒

**Files:**
- Modify: `tradingagents/harness/schema.sql`
- Modify: `tradingagents/harness/market_warning/adapters/sqlite_repository.py`
- Modify: `tradingagents/harness/market_warning/runner.py`
- Modify: `tradingagents/harness/market_warning/test_repository.py`
- Modify: `tradingagents/harness/market_warning/test_runner.py`

- [ ] **Step 1: 写时间网格失败测试**

  A 股只在 08:30、09:35/09:45/.../11:25、13:05/13:15/.../14:55 到期；09:40、11:30、12:00、15:00 不到期。美股仍影子运行，不能触发 A 股生产推送或门控。

- [ ] **Step 2: 写租约与故障计数失败测试**

  覆盖：原子获得租约、未过期租约导致 `overlap_skipped`、过期租约可回收、异常也释放租约、连续三个应执行时点失败只发一次系统故障提醒、成功后清零。

- [ ] **Step 3: 运行测试并确认失败**

  Run: `.venv/bin/python -m unittest tradingagents/harness/market_warning/test_runner.py tradingagents/harness/market_warning/test_repository.py -q`

- [ ] **Step 4: 实现 runner 模式与租约**

  增加 CLI `--mode model|rule_v1`，默认读取激活配置；规则模式只组合 `RuleMarketWarningService`。租约键为 `market_warning_fast_scan:a_share`，租期 8 分钟；cron 可每 5 分钟唤醒，但非 10 分钟格点立即退出，不访问行情。

- [ ] **Step 5: 实现运行指标**

  持久化 `started_at/finished_at/latency_ms/status/error_class/overlap_skipped/llm_calls`。快路径测试必须断言 `llm_calls=0`。系统故障提醒使用独立幂等键和“预警系统数据故障”文案，不能显示为市场红灯。

- [ ] **Step 6: 运行测试**

  Run: `.venv/bin/python -m unittest tradingagents/harness/market_warning/test_runner.py tradingagents/harness/market_warning/test_repository.py -q`

- [ ] **Step 7: 提交**

  Run: `git add tradingagents/harness/schema.sql tradingagents/harness/market_warning/adapters/sqlite_repository.py tradingagents/harness/market_warning/runner.py tradingagents/harness/market_warning/test_repository.py tradingagents/harness/market_warning/test_runner.py && git commit -m "feat: schedule and lease ten-minute rule scans"`

---

## Task 8: 接入 MiniMax M3 告警后解释慢路径

**Files:**
- Modify: `tradingagents/harness/market_warning/adapters/minimax_reasoning.py`
- Modify: `tradingagents/harness/market_warning/reasoning.py`
- Modify: `tradingagents/harness/market_warning/test_reasoning.py`
- Modify: `tradingagents/harness/market_warning/test_rule_service.py`

- [ ] **Step 1: 写规则解释契约失败测试**

  输入只能含结构化特征、触发规则、反向证据和 evidence IDs；输出只接受场景、因果链、反向证据、遗漏风险。断言原始 Think 不进入返回对象和数据库。

- [ ] **Step 2: 写触发与超时测试**

  首次 ORANGE、首次 RED、ORANGE->RED 才触发；同级 30 分钟内不触发。通知失败时不运行 M3；通知已由幂等记录确认 sent 时允许运行。90 秒超时只记录 `reasoning_unavailable`。

- [ ] **Step 3: 运行测试并确认失败**

  Run: `.venv/bin/python -m unittest tradingagents/harness/market_warning/test_reasoning.py tradingagents/harness/market_warning/test_rule_service.py -q`

- [ ] **Step 4: 实现规则解释适配**

  复用现有 MiniMax M3 客户端和断路器，新增规则评估输入方法，不改变模型模式接口。任何 M3 推荐灯号字段都丢弃并记契约告警，决策对象在调用前后必须相等。

- [ ] **Step 5: 运行测试**

  Run: `.venv/bin/python -m unittest tradingagents/harness/market_warning/test_reasoning.py tradingagents/harness/market_warning/test_rule_service.py -q`

- [ ] **Step 6: 提交**

  Run: `git add tradingagents/harness/market_warning/adapters/minimax_reasoning.py tradingagents/harness/market_warning/reasoning.py tradingagents/harness/market_warning/test_reasoning.py tradingagents/harness/market_warning/test_rule_service.py && git commit -m "feat: add post-alert M3 market context"`

---

## Task 9: 分离通知激活与个股硬门控

**Files:**
- Modify: `tradingagents/harness/market_risk.py`
- Modify: `tradingagents/harness/test_market_risk.py`
- Modify: `tradingagents/harness/market_warning/readiness.py`
- Modify: `tradingagents/harness/market_warning/test_readiness.py`

- [ ] **Step 1: 写门控失败测试**

  覆盖：
  - `rule_v1/notify` 激活时个股只看到提示信息，不改变硬门控。
  - `rule_v1/gate` 激活且 A 股橙/红、数据新鲜可靠时才约束新增仓位。
  - GREEN/YELLOW、过期、可靠度不足、checksum 不匹配、US 影子决策都不硬门控。
  - `model` 模式继续要求同一版本四模型激活。

- [ ] **Step 2: 运行测试并确认失败**

  Run: `.venv/bin/python -m unittest tradingagents/harness/test_market_risk.py tradingagents/harness/market_warning/test_readiness.py -q`

- [ ] **Step 3: 修改市场风险读取逻辑**

  SQL 按 `decision_source` 分支加载：规则决策关联 active rule registry；模型决策保持 active model registry。返回结构增加 `decision_source`、`engine_version` 与 `notification_only`，不把规则分数放入 `probabilities`。

- [ ] **Step 4: 运行测试**

  Run: `.venv/bin/python -m unittest tradingagents/harness/test_market_risk.py tradingagents/harness/market_warning/test_readiness.py -q`

- [ ] **Step 5: 提交**

  Run: `git add tradingagents/harness/market_risk.py tradingagents/harness/test_market_risk.py tradingagents/harness/market_warning/readiness.py tradingagents/harness/market_warning/test_readiness.py && git commit -m "feat: gate stock analysis with activated rule warnings"`

---

## Task 10: 安装器、运维命令与灰度文档

**Files:**
- Create: `scripts/install_market_warning_rule_v1.py`
- Create: `scripts/probe_market_warning_data.py`
- Create: `tradingagents/harness/market_warning/test_installation.py`
- Modify: `docs/market_warning_operations.md`

- [ ] **Step 1: 写安装保护失败测试**

  安装器必须拒绝：未显式指定 `rule_v1/notify`、readiness 失败、清单 checksum 不一致、数据库路径不可写、飞书配置缺失。它不得自动启用 `rule_v1/gate`。

- [ ] **Step 2: 运行测试并确认失败**

  Run: `.venv/bin/python -m unittest tradingagents/harness/market_warning/test_installation.py -q`

- [ ] **Step 3: 实现权限/数据冒烟命令**

  `probe_market_warning_data.py` 输出机器可读 JSON，至少包含：指数实时数据、全市场 `rt_k` 权限、横截面覆盖率、`stk_limit` 可用性、最新完成交易日、每个来源的数据时间和是否满足 notify 门槛。日志不得打印 token。

- [ ] **Step 4: 实现安装器**

  安装器写入现有调度机制，08:30 固定盘前运行，盘中 cron 每 5 分钟唤醒，由 runner 自己判断 10 分钟格点。重复执行必须幂等；卸载选项只删除该 V1 生成的调度项。

- [ ] **Step 5: 写运维文档**

  文档包含：报告怎么看、橙/红操作含义、权限不足时的表现、手工 dry-run、查询最近运行、解除过期租约、停用通知、10 日灰度审计与 gate 激活命令。

- [ ] **Step 6: 运行测试**

  Run: `.venv/bin/python -m unittest tradingagents/harness/market_warning/test_installation.py -q`

- [ ] **Step 7: 提交**

  Run: `git add scripts/install_market_warning_rule_v1.py scripts/probe_market_warning_data.py tradingagents/harness/market_warning/test_installation.py docs/market_warning_operations.md && git commit -m "feat: add guarded rule warning deployment"`

---

## Task 11: 性能、回归与真实数据验收

**Files:**
- Create: `tradingagents/harness/market_warning/test_rule_benchmark.py`
- Modify: `docs/market_warning_operations.md`

- [ ] **Step 1: 构造 100 轮录制行情基准**

  使用固定录制快照运行完整快路径 100 次，排除一次冷启动，计算 P50/P95/max。断言 P95 < 30 秒、每轮 `llm_calls=0`、内存不随轮次单调增长、不会产生重复告警。

- [ ] **Step 2: 运行性能测试**

  Run: `.venv/bin/python -m unittest tradingagents/harness/market_warning/test_rule_benchmark.py -q`

- [ ] **Step 3: 跑市场预警完整测试集**

  Run: `.venv/bin/python -m unittest tradingagents/harness/market_warning tradingagents/harness/test_market_risk.py -q`

- [ ] **Step 4: 跑项目相关回归**

  Run: `.venv/bin/python -m unittest tests/test_model_validation.py tests/test_tushare_rate_limit_wait.py tests/test_ticker_symbol_handling.py -q`

- [ ] **Step 5: 在真实 `.env` 下执行数据权限探针**

  Run: `.venv/bin/python scripts/probe_market_warning_data.py --market a_share --json`

  验收：明确记录全市场 `rt_k` 是否开通和横截面覆盖率。若权限不足，V1 不得安装成生产 notify；保留已完成代码和历史评估，并在最终结果中给出精确的数据权限缺口。

- [ ] **Step 6: 运行历史规则评估并检查门槛**

  Run: `.venv/bin/python -m tradingagents.harness.market_warning.rule_evaluation --market a_share --manifest tradingagents/harness/market_warning/rule_manifest_v1.json --output harness_data/models/market_warning/rule-v1-evaluation.json`

  只有 `rule_v1/notify` readiness 通过才继续安装。历史日线结果不得描述为分钟级验证。

- [ ] **Step 7: 执行离线时点端到端冒烟**

  当前为非交易时段时，使用最近一个完成交易日的盘前时点和录制盘中快照；真实盘中价格验收留到下一交易日，不伪造“实时通过”。验证报告首屏、幂等告警和 SQLite 审计记录。

- [ ] **Step 8: 安装通知灰度或明确阻断**

  满足数据探针、历史门槛、性能和飞书配置时运行：

  Run: `.venv/bin/python scripts/install_market_warning_rule_v1.py --mode rule_v1/notify`

  然后 dry-run 下一个 A 股应执行时点，确认只有一个 due slot。任一门槛不通过则不安装，不绕过 readiness。

- [ ] **Step 9: 记录验收证据并提交**

  把不含密钥和个人标识的基准摘要、历史指标及数据权限状态更新到 `docs/market_warning_operations.md`。

  Run: `git add tradingagents/harness/market_warning/test_rule_benchmark.py docs/market_warning_operations.md && git commit -m "test: validate rule warning v1 end to end"`

---

## Task 12: 最终审查、发布与合并

**Files:**
- Review: all files changed on `codex/market-warning-rule-v1`

- [ ] **Step 1: 对照 Spec 审查实现**

  逐条核对运行时间、通知状态转换、规则阈值、数据新鲜度、零 LLM 快路径、M3 不改结论、notify/gate 分离和美股影子约束。搜索并拒绝 `TODO`、`TBD`、临时阈值和规则分数写入 probability 的实现。

  Run: `rg -n "TODO|TBD|risk_score.*probability|probability.*risk_score" tradingagents/harness/market_warning tradingagents/harness/market_risk.py scripts docs/market_warning_operations.md`

- [ ] **Step 2: 运行最终测试和 readiness**

  Run: `.venv/bin/python -m unittest tradingagents/harness/market_warning tradingagents/harness/test_market_risk.py tests/test_model_validation.py tests/test_tushare_rate_limit_wait.py tests/test_ticker_symbol_handling.py -q`

  Run: `.venv/bin/python -m tradingagents.harness.market_warning.readiness --mode rule_v1/notify`

- [ ] **Step 3: 检查变更范围**

  Run: `git status --short && git diff --check && git log --oneline origin/main..HEAD`

- [ ] **Step 4: 发布功能分支并创建 PR**

  Run: `git push -u origin codex/market-warning-rule-v1`

  PR 必须列出：规则不是概率、实时权限探针结果、历史评估门槛、性能结果、实际安装状态、未开启个股 gate 的原因和下一次交易日实盘验收项。

- [ ] **Step 5: CI 通过后合并 main**

  按用户长期授权，在检查通过后合并 PR，拉取远端 `main` 并确认本地与远端 commit 一致；不得把原工作区中无关的用户修改带入提交。
