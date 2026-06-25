# AI 主升浪评级克制放松设计

## 背景

当前评级链路已经把很多主观漂移收进确定性工具：

- `stock_profile_node` 先生成画像与 `SYS_VALUATION_REGIME`
- `research_manager` 在 Step 6 强制调用 `compute_step6_final_rating`
- `compute_step6_final_rating` 统一执行动态阈值、估值映射、regime 闸门、拥挤度、升降档、趋势叠加、极端防御和不变量终检
- `market_risk_daily` 每日生成市场风险快照，PM 侧已经用它约束短期动作和仓位

用户反馈的问题不是要把某些历史报告倒推出更乐观结论，而是：在 AI 算力链处于主升环境、且个股业绩或订单正在兑现时，系统不应因为估值偏贵或拥挤常态而过早保守；但会亏钱的股票仍不能投。

本设计的核心原则是把判断拆成三层：

1. 大环境是否允许进攻：由 `tradingagents.harness.market_risk_daily` 的市场快照判断。
2. 个股是否属于 AI 主升兑现票：由确定性画像/基本面/新闻/资金信号判断。
3. 是否存在不可投风险：由地雷、现金流、扣非、派发、下修、目标价隐含收益等硬约束排除。

## 目标

新增一个可审计的 AI 主升浪信号，使 A/B 类股票在合适环境下不被过度保守处理：

- A 类：中际旭创、天孚通信、新易盛等，业绩已经兑现的 AI 算力链龙头。
- B 类：工业富联、沪电股份等，订单、客户绑定或产能周期正在兑现的 AI 算力链核心票。

达成后：

- 市场环境 OPEN 且个股 confirmed 时，允许更顺畅地从 `HOLD` 升到 `OVERWEIGHT`，强确认下允许 `OVERWEIGHT` 升到 `BUY`。
- 市场环境 CONDITIONAL 时，只允许轻度升档，不允许追成 `BUY`。
- 市场环境 WAIT / 高风险 / 数据不足时，不做 AI 主升升档；长期 thesis 可保留，但 PM 入场和仓位必须受市场闸门限制。
- 纯 AI 叙事但兑现弱、现金流差、扣非亏损、资金派发或预期下修的股票不得被误抬。

## 非目标

- 不用历史样本结果反推规则。
- 不把所有 AI 概念股整体调乐观。
- 不绕过现有 `discipline`、拥挤多头禁 BUY、价格/目标价同号不变量。
- 不改变 PM 的市场风险仓位闸门；本设计只让 RM 的长期评级更懂 AI 主升，不负责短线买点。

## 新增概念

### 1. Market Risk Mode

从 `market_risk_snapshot` 派生一个市场模式，作为 AI 主升升档的外部总闸。

| 条件 | market_mode | AI 主升升档权限 |
|------|-------------|----------------|
| `entry_gate=OPEN` 且 `risk_level=低` 且 `t_plus_1_bias=偏多` | `risk_on` | 完整生效 |
| `entry_gate=CONDITIONAL` 或 `risk_level=中` | `conditional` | 只允许轻度升档 |
| `entry_gate=WAIT` 或 `risk_level in 高/极高/数据不足` 或快照缺失 | `risk_off` | 禁止升档 |

快照缺失必须视为 `risk_off`，不能假设低风险。

### 2. SYS_AI_MAIN_UPTREND

在 `profile_calc.py` 中新增确定性判断，输出：

```text
【SYS_AI_MAIN_UPTREND】 enabled=true | class=confirmed | reasons=...
```

合法字段：

- `enabled`: `true` / `false`
- `class`: `confirmed` / `early` / `none`
- `reasons`: 简短列出触发证据
- `blockers`: 简短列出未触发或被排除的原因

`confirmed` 表示业绩或订单已经进入兑现期；`early` 表示链条地位明确、趋势强，但兑现证据还不够完整。

## 个股触发规则

### 赛道条件

必须命中 AI 算力链硬科技之一：

- CPO / 光模块 / 光器件
- AI 服务器 / 算力设备 / 服务器代工
- AI PCB / 高速互联 / 交换机链条
- 液冷 / 数据中心基础设施
- AI 芯片、半导体核心链，但需避免把强周期存储自动当作范式成长

优先复用现有 `detect_paradigm_growth`、主营构成 `SYS_MAIN_BUSINESS`、行业/公司名单和新闻催化信息。

### 兑现条件

至少满足一条：

- 年度或最近季度净利润高增，建议 `net_profit_growth >= 0.40`。
- 营收高增，建议 `revenue_growth >= 0.30`。
- `earnings_revision == 上修`。
- 新闻或基本面报告中出现订单、核心客户绑定、产能放量、英伟达/云厂商链条绑定等硬证据。

### 趋势条件

至少满足一条：

- `momentum_score >= 65`。
- `theme_stage_inferred == acceleration`。
- 板块 RS 明显为正，且本股在主题内相对强。

### 排除条件

任一命中则 `enabled=false` 或最多 `class=none`：

- 地雷清单触发。
- `recurring_loss=True`。
- 扣非亏损、经营现金流持续恶化、现金快速消耗且无确定订单对冲。
- `valuation_regime=discipline`。
- `has_peak_signal=True`。
- 散户高接盘且价格极端，价格极端定义沿用已有 blowoff 逻辑：RSI 1Y 分位 >= 85 或获利盘 >= 85%。
- 主力资金持续恶化且无卖方上修。
- `earnings_revision == 下修`。
- 综合目标价中位低于现价，且没有新的已确认业绩数据说明目标价锚滞后。

## 评级影响

在 `compute_step6_final_rating` 中新增入参：

```python
ai_main_uptrend: bool = False
ai_main_uptrend_class: str = ""
market_mode: str = ""
```

新增调整步骤放在“对称升降档”之后、“趋势叠加”之前。原因：

- 对称升降档已经处理数据完整度、红旗和拐点。
- 趋势叠加仍保留原有 style/vote/catalyst 合成。
- AI 主升调整应是一个明确的中间证据层，而不是最后绕过边界。

### risk_on

`ai_main_uptrend_class=confirmed`：

- `HOLD -> OVERWEIGHT`
- `OVERWEIGHT -> BUY` 仅当额外满足任一强确认：
  - `momentum_score >= 80`
  - `news_catalyst_score > 0`
  - `earnings_revision == 上修`

`ai_main_uptrend_class=early`：

- 仅允许 `HOLD -> OVERWEIGHT`
- 不允许升到 `BUY`

### conditional

无论 `confirmed` 还是 `early`：

- 最多 `HOLD -> OVERWEIGHT`
- 不允许 `OVERWEIGHT -> BUY`

### risk_off

- 不做 AI 主升升档。
- 返回说明：市场风险快照不允许进攻，AI 主升信号仅保留为长期观察项。

### 边界约束

AI 主升调整必须受现有边界钳制：

- `discipline` 天花板不能越过。
- 拥挤多头禁 `BUY` 不能越过。
- `rating direction` 与目标价隐含收益同号不变量不能越过。
- `market_mode=risk_off` 时不能升档。

## 报告输出

`stock_profile` 末尾增加：

```text
【SYS_AI_MAIN_UPTREND】 enabled=true | class=confirmed | market_sensitive=true | reasons=...
```

`manager.md` Step 6 工具返回中增加：

```text
AI 主升：enabled=true/class=confirmed/market_mode=risk_on -> HOLD→OVERWEIGHT
```

`RM_SUMMARY` 增加：

```yaml
ai_main_uptrend:
  enabled: true
  class: confirmed
  market_mode: risk_on
  adjustment: "+1"
  blockers: []
```

## 测试计划

### 单元测试

新增或扩展：

- `tradingagents/dataflows/test_valuation_regime.py`
- `tradingagents/agents/managers/test_step6_final_rating.py`

覆盖：

- AI 主升 confirmed + risk_on：`HOLD -> OVERWEIGHT`。
- AI 主升 confirmed + risk_on + 强确认：`OVERWEIGHT -> BUY`。
- AI 主升 early + risk_on：只能 `HOLD -> OVERWEIGHT`。
- AI 主升 confirmed + conditional：不能 `OVERWEIGHT -> BUY`。
- AI 主升 confirmed + risk_off：不升档。
- `discipline` 天花板存在时，AI 主升不能越界。
- 目标价中位低于现价时，看多评级最终仍收敛 HOLD。
- 散户高接盘 + 价格极端时，不触发 AI 主升。
- 扣非亏损或经营现金流恶化时，不触发 AI 主升。

### 回归样本

正向观察样本：

- 中际旭创
- 天孚通信
- 新易盛
- 工业富联
- 沪电股份

条件样本：

- 澜起科技
- 兆易创新

这两类不能预设正负。它们用于验证同一类 AI/半导体资产在不同状态下能否切换：上修/兑现/无派发时可触发；派发/下修/周期顶部/估值纪律明确时不得触发。

负向样本：

- 淳中科技
- 非 AI 主升行业股票

负向样本用于验证纯叙事、兑现弱或非相关行业不会被误抬。

## 验收标准

- 规则不依赖具体历史样本名称得出结论，所有样本必须由信号自然触发。
- `market_risk_daily` 为 `risk_off` 时，没有任何 AI 主升升档。
- AI 主升升档后仍能被现有不变量收敛。
- 条件样本在不同 `valuation_regime` / 资金面 / 预期修正状态下能切换，而不是固定正负。
- PM 仍严格使用 `apply_market_risk_gate` 控制短线动作和仓位。

## 风险与回滚

主要风险：

- AI 主升识别过宽，把纯题材票误抬。
- 市场模式过强，导致 RM 长期评级随短期市场快照波动过大。
- `market_risk_daily` 只覆盖宽基风险，不能完全代表 AI 板块自身环境。

缓解：

- 个股排除条件优先级高于 AI 主升加分。
- `market_mode` 只控制升档权限，不直接降长期评级。
- 后续可新增 AI/算力主题 ETF 作为 market risk 的行业子快照，但本期不做。

回滚：

- 新增逻辑应集中在 `profile_calc.py`、`stock_profile_node.py`、`research_manager.py`、`rm_tools.py` 与测试文件。
- 出现误抬时可先关闭 `ai_main_uptrend` 入参消费，让画像信号保留但不影响评级。
