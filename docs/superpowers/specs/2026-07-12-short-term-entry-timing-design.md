# 短线入场时机确定性信号设计

日期：2026-07-12
状态：待用户确认

## 1. 背景与目标

当前系统已经能够分别判断公司基本面、AI 主升兑现资格、估值制度和大盘风险，但“长期是否值得配置”与“当前是否适合入场”仍有耦合。典型结果是：公司逻辑成立但位置偏高时，最终评级容易被压成 HOLD；或者评级看多，却没有明确告诉用户应当立即介入、等待回踩还是等待突破。

本次新增一个 Python 确定性短线结构层 `short_term_structure`，输出独立的入场时机结论。首期只影响交易时机与执行建议，不直接改变长期评级。

目标：

1. 将长期评级与短线入场时机拆成两个正交结论。
2. 将“MA10 有角度、缩量回踩、放量突破、量价衰竭、趋势破位”等经验转成可复现指标。
3. 保留 `market_risk_daily` 作为大环境总闸。
4. 保留亏损、业绩下修、纪律型估值等基本面风险的优先级，技术形态不能覆盖风险红旗。
5. 输出可测试、可回测、可解释的结构化字段，避免依赖 LLM 图感。

非目标：

- 不预测短期涨跌幅或精确买点价格。
- 不因 MA10 信号直接升降 BUY/HOLD/SELL 评级。
- 不针对某一只历史上涨股票反推参数。
- 不在首期引入新的外部行情数据源或复杂机器学习模型。

## 2. 总体架构

数据流：

```text
OHLCV 日线
  -> compute_short_term_structure()
  -> SYS_SHORT_TERM_STRUCTURE 机读真值
  -> Market Analyst 解释结构与关键位
  -> Research Manager 生成独立 entry_timing
  -> Portfolio Manager 输出“长期评级 + 入场时机”决策卡

market_risk_daily --------------------^ 大环境总闸
基本面/估值硬红旗 --------------------^ 个股风险总闸
```

计算逻辑归属 `tradingagents/dataflows/profile_calc.py`，画像节点负责转录，Market Analyst 只解释，不得改写确定性分类。RM/PM 消费标准字段并应用总闸。

## 3. 输入与基础指标

输入优先使用至少 60 个交易日的 `Open/High/Low/Close/Volume`；结构计算的硬下限为 20 日，不足 20 日时返回 `insufficient_data`，不猜测。RSI 一年分位继续复用现有更长窗口的计算结果，缺失时只关闭依赖 RSI 的分支。

基础字段：

| 字段 | 定义 |
|---|---|
| `ma10` | 收盘价 10 日简单移动平均 |
| `ma20` | 收盘价 20 日简单移动平均 |
| `ma10_slope_5d_pct` | `ma10[t] / ma10[t-5] - 1` |
| `price_vs_ma10_pct` | `close[t] / ma10[t] - 1` |
| `price_vs_ma20_pct` | `close[t] / ma20[t] - 1` |
| `volume_ratio_5d_20d` | 近 5 日均量 / 近 20 日均量 |
| `volume_ratio_1d_20d` | 当日成交量 / 近 20 日均量 |
| `atr14_pct` | ATR14 / 当前收盘价 |
| `distance_from_prior_20d_high_pct` | 当前价相对“不含当日”的前 20 日最高收盘价的距离 |
| `rsi_percentile_1y` | 复用现有 RSI 一年分位 |

所有价格距离阈值先按 ATR 自适应，再设置上下限，避免用固定 2% 同时套用低波银行股和高波科技股：

```text
near_band = clamp(0.75 * atr14_pct, 1.5%, 4.0%)
break_band = clamp(1.00 * atr14_pct, 2.0%, 5.0%)
extended_band = clamp(2.00 * atr14_pct, 6.0%, 12.0%)
```

## 4. 短线结构分类

分类采用“风险状态优先”顺序，一只股票只能得到一个主分类。优先级从高到低：

`broken > exhaustion > trend_pullback > breakout_ready > healthy_trend > neutral`

### 4.1 broken：趋势破位

满足以下任一组：

- 收盘价低于 MA20，且 MA10 五日斜率小于等于 -1%；
- 连续 2 日收盘低于 MA10，最新价低于 MA10 超过 `near_band`，且 MA10 五日斜率小于 0；
- 前一日收盘不低于前一日 MA20、当日收盘低于 MA20，且 `volume_ratio_1d_20d >= 1.5`。

该状态表达短线结构已坏，不等价于公司长期逻辑失效。

### 4.2 exhaustion：量价衰竭或过热

满足以下任一组：

- 当前价距离 MA10 超过 `extended_band`，且 RSI 一年分位不低于 85；
- 当日收盘突破“不含当日”的前 20 日最高收盘价，但 `volume_ratio_1d_20d < 0.8`，形成无量新高；
- 近 5 日收盘涨幅不低于 `max(5%, 2 * atr14_pct)`、MA10 五日斜率大于 0，且现有 `has_vol_divergence=true`。

仅有高 RSI 或仅有缩量不能单独判衰竭，必须出现价格扩张/新高与量能或拥挤的组合证据。

### 4.3 trend_pullback：上升趋势中的缩量回踩

全部满足：

- MA10 五日斜率大于等于 1%；
- MA10 高于 MA20；
- 当前收盘价位于 MA10 的 `±near_band` 内，且未有效跌破 MA20；
- `volume_ratio_5d_20d <= 0.85`；
- 不属于 `broken` 或 `exhaustion`。

这是首期唯一可支持“分批介入”的积极结构，但仍受市场和基本面总闸限制。

### 4.4 breakout_ready：平台收敛，等待确认突破

全部满足：

- 当前价尚未突破“不含当日”的前 20 日最高收盘价，且向下距离不超过 `break_band`；
- 近 10 日收盘价振幅不超过 `max(8%, 3 * atr14_pct)`；
- `volume_ratio_5d_20d <= 0.85`；
- MA10 五日斜率不低于 0，且当前价不低于 MA20；
- 尚未出现当日放量突破确认。

该状态不是买入信号，只表示等待 `收盘突破前 20 日最高收盘价 + volume_ratio_1d_20d >= 1.3`。

### 4.5 healthy_trend：趋势健康但无低风险触发点

全部满足：

- MA10 五日斜率大于 0；
- MA10 高于 MA20，当前价高于 MA10；
- 不属于上述四类。

若距离 MA10 偏大，默认等待回踩；若距离适中但没有缩量回踩证据，保持观察，不把“趋势上涨”直接当作追涨理由。

### 4.6 neutral / insufficient_data

- `neutral`：没有明确优势结构，也没有明确破位或衰竭。
- `insufficient_data`：有效数据不足，后续不得将其翻译为积极入场建议。

## 5. 结构化输出契约

```yaml
short_term_structure:
  as_of_date: YYYY-MM-DD
  structure_class: trend_pullback | breakout_ready | healthy_trend | exhaustion | broken | neutral | insufficient_data
  ma10: 0.0
  ma20: 0.0
  ma10_slope_5d_pct: 0.0
  price_vs_ma10_pct: 0.0
  price_vs_ma20_pct: 0.0
  volume_ratio_5d_20d: 0.0
  volume_ratio_1d_20d: 0.0
  atr14_pct: 0.0
  distance_from_prior_20d_high_pct: 0.0
  breakout_confirmed: false
  reasons: []
  blockers: []
```

画像末尾增加单一机读行，避免 M3 在自然语言转述时丢字段：

```text
SYS_SHORT_TERM_STRUCTURE: class=trend_pullback | ma10_slope_5d_pct=2.3 | price_vs_ma10_pct=0.8 | volume_ratio_5d_20d=0.71 | breakout_confirmed=false
```

## 6. 入场时机映射

最终新增 `entry_timing`，与长期评级并列。允许值固定为 `分批介入 / 小仓试探 / 等回踩 / 等放量突破 / 暂不介入 / 退出观察 / 继续观察 / 数据不足`：

| 结构分类 | 基础入场时机 |
|---|---|
| `trend_pullback` | 分批介入 |
| `breakout_ready` | 等放量突破 |
| `healthy_trend` | 等回踩 |
| `exhaustion` | 暂不介入 |
| `broken` | 退出观察 |
| `neutral` | 继续观察 |
| `insufficient_data` | 数据不足 |

### 6.1 市场环境总闸

复用 `derive_market_mode(market_risk_snapshot)`：

- `risk_on`：允许按基础映射输出。
- `conditional`：`分批介入` 降为 `小仓试探`；其他状态不变。
- `risk_off`：任何积极状态统一降为 `暂不介入`；不得改变长期评级。
- 缺失市场快照按现有规则视为 `risk_off`。

### 6.2 个股风险总闸

以下任一条件出现时，禁止输出 `分批介入` 或 `小仓试探`：

- `recurring_loss=true`；
- `earnings_revision=下修`；
- `valuation_regime=discipline`；
- `has_peak_signal=true`、散户高接盘叠加 RSI 一年分位不低于 85，或资金状态为恶化/主力连续流出至少 3 日且无盈利上修；
- 有效 OHLCV 少于 20 日，或 Close/Volume 清洗后无法计算 MA20 与量比。

总闸只约束动作，不用技术状态覆盖或改写基本面事实。

## 7. Agent 职责

### 7.1 Market Analyst

- 必须引用 `SYS_SHORT_TERM_STRUCTURE` 的分类和数字。
- 解释支撑位、阻力位、量价关系及失效条件。
- 不得把 `healthy_trend` 擅自升级为立即买入。
- 不得自行修改确定性结构分类；若认为数据冲突，只能标注冲突。

### 7.2 Research Manager

- 先完成长期评级，再独立生成 `entry_timing`。
- 调用确定性映射工具应用市场和个股总闸。
- 输出长期观点与短期时机可能不同，例如 `OVERWEIGHT + 等回踩`。

### 7.3 Portfolio Manager

- 决策卡并列展示：长期评级、入场时机、触发条件、失效条件。
- 不得用短线结构将长期 SELL 自动抬高；也不得因一次短期破位自动把长期 BUY 降成 SELL。

## 8. 错误处理与降级

- OHLCV 缺列、非数值、排序异常：清洗后仍不足则 `insufficient_data`。
- 成交量为零或停牌：不计算量比积极信号，输出 `neutral` 或 `insufficient_data` 并记录原因。
- ATR 为零或缺失：使用固定 `near_band=2%`、`break_band=3%`、`extended_band=8%`，并记录降级原因。
- `market_risk_daily` 缺失：沿用 `risk_off`，不允许 LLM 猜测大盘状态。
- M3 输出缺失 `entry_timing`：PM/RM 后处理器根据确定性字段补齐，不因格式遗漏导致整份报告失败。

## 9. 测试与验收

### 9.1 单元测试

使用人工构造 OHLCV 分别覆盖：

1. MA10 上扬、MA10 高于 MA20、缩量回踩，输出 `trend_pullback`。
2. 平台缩量且接近 20 日高点，输出 `breakout_ready`。
3. 趋势向上但偏离 MA10，输出 `healthy_trend` 或 `exhaustion`，取决于 RSI 与量价组合。
4. 无量创新高，输出 `exhaustion`。
5. 放量跌破 MA20，输出 `broken`。
6. 数据不足、停牌量、ATR 缺失的降级行为。
7. 分类优先级冲突时，`broken/exhaustion` 优先于积极结构。

### 9.2 总闸测试

覆盖每个结构分类与 `risk_on/conditional/risk_off` 的映射；验证亏损、下修和 discipline 均能否决积极动作，但不改写长期评级。

### 9.3 报告回归

至少选择四类样本各一只：强趋势、缩量回踩、量价衰竭、趋势破位。验收的是同一输入下结构分类稳定、证据数字一致、总闸有效，而不是报告是否迎合股票后续涨跌。

在 `.venv` 下完成：

- 目标单元测试；
- 现有估值制度、AI 主升、RM 最终评级和 market risk 测试；
- 一只完整股票分析，检查 M3 是否稳定产出长期评级与 `entry_timing`，以及最终报告是否闭环。

## 10. 成功标准

1. 同一 OHLCV 输入重复运行得到完全相同的结构分类与数值。
2. 报告明确区分“长期评级”和“入场时机”，允许出现 `OVERWEIGHT + 等回踩`。
3. `risk_off`、亏损、下修或 discipline 下绝不出现积极入场动作。
4. Market Analyst、RM、PM 引用的结构分类一致，不因 M3 转述发生漂移。
5. 原有 AI 主升升档和评级测试无回归。
6. 首期不改变长期评级算术，只新增时机输出与执行约束。

## 11. 后续校准边界

首版阈值是可解释工程先验。上线后由 harness 按全部有效样本统一统计各分类的 T+5/T+10/T+20 收益、最大回撤和胜率，再决定是否校准。禁止根据单只股票或单次报告手工修改阈值。

只有当样本量和稳定性足够时，才讨论第二阶段：让 `trend_pullback + risk_on + 无硬红旗` 对仓位产生轻度影响；仍不建议直接改变长期评级。
