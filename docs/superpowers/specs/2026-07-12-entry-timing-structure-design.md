# 确定性短线结构与入场时机设计

## 1. 背景

项目已经形成两条相对稳定的决策链：

- 个股长期方向由基本面、估值 regime、资金、主题阶段和 AI 主升兑现资格共同决定。
- 大环境是否允许进攻由 `tradingagents.harness.market_risk_daily` 生成的快照决定。

当前缺口是“股票是否值得投”与“现在是否适合买”仍混在同一个 LLM 判断中。强趋势股票可能因为短期位置偏高被压低长期评级，也可能在量价衰竭时仍得到模糊的“逢低介入”建议。Market Analyst 已描述量价关系，但缺少可复现、可回测、可被下游稳定消费的短线结构真值。

本设计新增一层确定性 `short_term_structure`，独立给出入场时机，不直接改变五档长期评级。

## 2. 目标与非目标

### 2.1 目标

1. 使用日线 OHLCV 确定性识别短线趋势、回踩、突破准备、衰竭和破位。
2. 将长期评级与入场时机二维分离。
3. 使用 `market_risk_daily` 作为“立即介入”的市场总闸。
4. 保留亏损、业绩下修和纪律型估值等现有硬护栏。
5. 所有状态由明确数值计算，可单元测试和历史回测。

### 2.2 非目标

- 不因短线结构直接升降 `BUY / OVERWEIGHT / HOLD / UNDERWEIGHT / SELL`。
- 不修改 `compute_ai_main_uptrend_signal` 的资格或升档规则。
- 不以某只历史上涨股票反推阈值。
- 不新增 LLM 调用，不让 LLM 主观判断“均线是否有角度”。
- 不在本期重构 Trader、风险辩论或分析师权重。

## 3. 总体设计

数据流如下：

```text
日线 OHLCV
  -> compute_short_term_structure()
  -> stock_profile 的 SYS_SHORT_TERM_* 机读字段
  -> Portfolio Manager
       + 长期评级
       + valuation_regime / recurring_loss / earnings_revision
       + market_risk_daily -> market_mode
  -> entry_timing_action + 触发价位 + 原因
  -> 决策卡“入场时机”与正文解释
```

Market Analyst 负责解释结构和量价含义，但不产生最终真值。PM 必须优先采用 Python 机读字段；报告缺失字段时降级为保守等待，不得由 LLM 补猜。

## 4. 短线结构计算

### 4.1 输入要求

函数签名：

```python
compute_short_term_structure(price_df) -> dict
```

输入使用已有日线 OHLCV DataFrame，兼容大小写列名。最少需要 `Close` 和 `Volume`；`High`、`Low` 缺失时可计算趋势，但不能确认回踩或突破。少于 25 个有效交易日时返回 `insufficient_data`。

计算只使用截至分析日的数据，不得读取未来数据。

### 4.2 基础字段

```yaml
short_term_structure:
  as_of_date: YYYY-MM-DD
  close: 0.0
  ma10: 0.0
  ma20: 0.0
  ma10_slope_5d_pct: 0.0
  price_vs_ma10_pct: 0.0
  volume_ratio_5d_20d: 0.0
  distance_to_20d_high_pct: 0.0
  trend_up: false
  near_ma10: false
  pullback_volume_shrink: false
  breakout_with_volume: false
  breakout_ready: false
  volume_price_exhaustion: false
  effective_breakdown: false
  structure_class: insufficient_data
  reasons: []
```

定义：

- `MA10`、`MA20`：收盘价简单移动平均。
- `ma10_slope_5d_pct = (MA10_t / MA10_t-5 - 1) * 100`。
- `price_vs_ma10_pct = (Close_t / MA10_t - 1) * 100`。
- `volume_ratio_5d_20d = mean(Volume, 5) / mean(Volume, 20)`。
- `distance_to_20d_high_pct = (Close_t / max(High, 20) - 1) * 100`。

### 4.3 布尔信号

首期阈值采用透明先验，后续只能通过样本外回测调整：

| 信号 | 确定性条件 |
|---|---|
| `trend_up` | `Close >= MA20` 且 `MA10 >= MA20` 且 `ma10_slope_5d_pct >= 0.5%` |
| `near_ma10` | 最近 3 日最低价曾进入 `[MA10*0.97, MA10*1.03]`，且当前收盘不低于 `MA10*0.97` |
| `pullback_volume_shrink` | `near_ma10=true` 且 5/20 日量比 `<=0.80` |
| `breakout_with_volume` | 当前收盘高于此前 20 日最高收盘，且当日成交量/此前 20 日均量 `>=1.30` |
| `breakout_ready` | 距此前 20 日最高收盘不超过 `3%`，近 5 日振幅不超过 `8%`，5/20 日量比 `<=0.85`，且尚未有效突破 |
| `volume_price_exhaustion` | 当前收盘处于近 20 日最高价下方 `0%~3%`，但 5/20 日量比 `<0.70`；或价格创 20 日新高而成交量低于此前 20 日均量的 `0.80` |
| `effective_breakdown` | 连续 2 日收盘低于 `MA10*0.97`，且 `MA10` 斜率小于 `0`；或当前收盘低于 `MA20*0.97` |

涨跌停、停牌或成交量为零的行在量比计算前剔除；有效交易日不足则相应字段为 `null`，不得用零代替缺失。

### 4.4 状态分类与优先级

状态按以下优先级单选，前者覆盖后者：

1. `insufficient_data`：有效数据不足。
2. `broken`：`effective_breakdown=true`。
3. `exhaustion`：`volume_price_exhaustion=true`。
4. `breakout`：`breakout_with_volume=true` 且 `trend_up=true`。
5. `trend_pullback`：`trend_up=true`、`near_ma10=true`、`pullback_volume_shrink=true`。
6. `breakout_ready`：同名布尔信号为真且 `MA10` 斜率不为负。
7. `healthy_trend`：`trend_up=true`。
8. `neutral`：其余情况。

优先级确保破位和衰竭不能被“仍在均线上方”掩盖。`reasons` 必须写入实际触发的数值，不输出泛化描述。

## 5. 入场时机决策

### 5.1 标准输出

内部使用五个动作：

```text
BUY_NOW       立即介入
WAIT_PULLBACK 等回踩
WAIT_BREAKOUT 等突破
DO_NOT_ENTER  暂不介入
EXIT_WATCH    退出观察
```

为兼容现有决策卡四选一，`EXIT_WATCH` 在卡片中渲染为“不建议介入”，但正文必须明确“结构已破位，退出观察”；不得写成普通等待。

输出字段：

```yaml
entry_timing:
  action: BUY_NOW
  label: 立即介入
  trigger_price_low: null
  trigger_price_high: null
  invalidation_price: 0.0
  structure_class: trend_pullback
  market_mode: risk_on
  blocked_by: []
  reasons: []
```

### 5.2 个股结构到动作映射

在无硬护栏且市场环境允许时：

| `structure_class` | 基础动作 |
|---|---|
| `trend_pullback` | `BUY_NOW` |
| `breakout` | `BUY_NOW`，但若当日涨幅或距 MA10 偏离过大则改为 `WAIT_PULLBACK` |
| `healthy_trend` | `WAIT_PULLBACK` |
| `breakout_ready` | `WAIT_BREAKOUT` |
| `exhaustion` | `DO_NOT_ENTER` |
| `broken` | `EXIT_WATCH` |
| `neutral` / `insufficient_data` | `DO_NOT_ENTER` |

追高保护：若 `price_vs_ma10_pct > 8%`，任何 `BUY_NOW` 降为 `WAIT_PULLBACK`。若分析日单日涨幅 `>7%` 且成交量比 `>2.0`，也降为 `WAIT_PULLBACK`，避免将情绪加速日视为低风险入口。

### 5.3 市场总闸

继续复用 `derive_market_mode(market_risk_snapshot)`：

| `market_mode` | 约束 |
|---|---|
| `risk_on` | 允许 `BUY_NOW` |
| `conditional` | `trend_pullback` 或 `breakout` 产生的 `BUY_NOW` 一律降为 `WAIT_PULLBACK`；其他动作保持不变，不得抬升 |
| `risk_off` | 禁止 `BUY_NOW`；基础动作统一收敛为 `DO_NOT_ENTER`，`broken` 仍保持 `EXIT_WATCH` |

快照缺失继续按 `risk_off` 处理。PM 必须引用快照日期，过期快照按现有市场风险规则处理，不在本模块另造时效标准。

### 5.4 个股硬护栏

以下任一条件存在时，技术结构不能输出 `BUY_NOW`：

- `recurring_loss=true`。
- `earnings_revision=下修`。
- `valuation_regime=discipline`。
- AI 主升信号中的现有 blocker 包含 `peak`、资金持续恶化或散户高接盘叠加价格极端。
- 长期评级为 `UNDERWEIGHT` 或 `SELL`。

动作收敛规则也是确定性的：`recurring_loss`、`earnings_revision=下修`、`valuation_regime=discipline` 或上述 AI 主升硬 blocker 任一命中，基础动作收敛为 `DO_NOT_ENTER`；若结构本身为 `broken`，保持更强的 `EXIT_WATCH`。长期评级为 `UNDERWEIGHT` 或 `SELL` 时采用同一规则。

`HOLD` 可以输出 `WAIT_PULLBACK` 或 `WAIT_BREAKOUT`，原基础动作为 `BUY_NOW` 时固定降为 `WAIT_PULLBACK`，但不得输出 `BUY_NOW`。`BUY`/`OVERWEIGHT` 才有资格在其余条件满足时输出 `BUY_NOW`。这保证技术面只优化时机，不覆盖长期方向。

## 6. 触发价位生成

- `WAIT_PULLBACK`：目标区间为 `[MA10*0.98, MA10*1.02]`，按标的价格精度取整；下沿不得低于当前有效支撑位。若当前价已在区间内但被市场闸门降级，保留区间并注明等待市场转为 `risk_on`。
- `WAIT_BREAKOUT`：触发价为“此前 20 日最高收盘价 * 1.005”，且要求成交量达到此前 20 日均量的 `1.30` 倍。卡片展示具体价格，正文展示量能条件。
- `BUY_NOW`：失效位默认为 `min(MA20*0.97, 最近 10 日摆动低点*0.99)`；该值只作为技术结构失效参考，不替代 PM 的组合止损。
- `DO_NOT_ENTER` / `EXIT_WATCH`：不伪造买入触发价，可给出重新观察条件。

## 7. 组件改动边界

预计涉及：

- `tradingagents/dataflows/profile_calc.py`：新增纯函数及字段计算。
- `tradingagents/agents/utils/stock_profile_node.py`：调用函数并附加 `SYS_SHORT_TERM_*` 机读行。
- `tradingagents/agents/analysts/market_analyst.py`：要求解释确定性结构，禁止覆盖真值。
- `tradingagents/agents/managers/portfolio_manager.py`：读取结构、评级和 `market_mode`，确定入场动作并填入决策卡。
- 对应单元测试文件：覆盖结构分类、优先级、市场闸门和硬护栏。

若现有 PM 没有可靠的 Python 入口承载动作映射，应新增一个无 I/O 的 helper，并由 PM prompt 消费其机读输出；不能把映射规则只写在提示词中。

## 8. 降级与错误处理

- OHLCV 缺失或不足：`insufficient_data -> DO_NOT_ENTER`。
- Volume 缺失：不能判定回踩缩量、放量突破或量价衰竭；只允许 `healthy_trend / broken / neutral`。
- 市场风险快照缺失：`risk_off -> DO_NOT_ENTER`。
- LLM 未按格式转录：最终机读结果仍由 Python 真值提供；报告保存层应保留原始 `SYS_SHORT_TERM_*` 行便于排查。
- 数值异常（无穷、NaN、负成交量）：忽略异常行并在 `reasons` 标记数据不足，不能抛出导致整支股票分析中断的异常。

## 9. 测试与验收

### 9.1 单元测试

使用人工构造 OHLCV，至少覆盖：

1. 上升趋势中的 MA10 缩量回踩 -> `trend_pullback`。
2. 平台缩量接近前高 -> `breakout_ready`。
3. 放量有效突破 -> `breakout`。
4. 创新高但量能明显萎缩 -> `exhaustion`，且优先于 `healthy_trend`。
5. 连续跌破 MA10 或有效跌破 MA20 -> `broken`。
6. 少于 25 日或 Volume 缺失 -> 正确降级，不抛异常。
7. 同一输入重复计算结果完全一致。

入场映射至少覆盖：

1. `trend_pullback + BUY + risk_on` -> `BUY_NOW`。
2. 相同结构在 `conditional` -> 等待；在 `risk_off` -> `DO_NOT_ENTER`。
3. 距 MA10 超过 8% -> `WAIT_PULLBACK`。
4. `discipline`、亏损、下修任一存在 -> 不得 `BUY_NOW`。
5. `UNDERWEIGHT/SELL` 即使结构良好也不得买入。
6. `broken` 始终保持 `EXIT_WATCH`。

### 9.2 历史样本回放

样本按形态挑选，不按事后涨跌挑选。至少包含强趋势、缩量回踩、放量突破、量价衰竭和破位各 3 个截面，并固定分析截止日。验收关注：

- 无未来数据泄漏。
- 相同截面重复运行结构真值一致。
- `BUY_NOW` 只出现在 `risk_on + 正向长期评级 + 无硬护栏`。
- 结构衰竭和破位不被 AI 主升标签覆盖。
- 记录 T+5/T+10/T+20 最大有利和最大不利波动，为后续阈值校准积累数据；首期不自动调参。

### 9.3 全链路回归

在 `.venv` 下完成：

1. 运行新增和相关既有测试。
2. 选择一只正常交易、数据完整的 A 股完整分析。
3. 核对报告已输出结构类别、市场模式、入场动作和具体触发价。
4. 核对五档长期评级没有因本模块被直接改写。
5. 核对 M3 报告完整生成，决策卡入场字段与 Python 真值一致。

## 10. 成功标准

- 短线结构和入场动作均由确定性 Python 逻辑生成。
- 长期评级与入场时机二维独立。
- `market_risk_daily` 是唯一市场总闸，没有第二套大盘判断。
- 技术结构不能覆盖亏损、下修、纪律 regime 或负向长期评级。
- 新增测试全部通过，既有相关测试无回归。
- `.venv` 全链路股票分析成功，报告字段完整且可追溯到计算值。

## 11. 回滚策略

入场时机结果应通过独立字段接入。若线上表现异常，可停止 PM 对 `SYS_SHORT_TERM_*` 的消费，恢复原入场时机生成；结构计算和报告字段可保留用于继续观测，不影响长期评级、AI 主升资格或市场风险快照。
