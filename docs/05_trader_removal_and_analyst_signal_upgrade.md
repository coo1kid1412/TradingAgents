# 优化 05：移除 Trader 节点 + 分析师信号结构化升级

> **目标读者**：实施该优化的 coding agent（Qoder）
> **预估工作量**：1-1.5 天
> **改动文件数**：12 个
> **新增 LLM 调用**：0；**减少 LLM 调用**：1（移除 Trader）
> **风险等级**：中（涉及 graph 拓扑改动 + 多 agent prompt 改动，需充分回归）

---

## 0. 一图看懂本次改动

```
改动前：
  分析师层(4) → Bull/Bear 辩论 → RM → Trader → 风控辩论(3) → PM
                                       ↑↑↑
                          删除此节点

改动后：
  分析师层(4) → Bull/Bear 辩论 → RM →────────→ 风控辩论(3) → PM
        ↑
   每个分析师 prompt 末尾强制输出 SUMMARY 块 + 5 项 prompt 细节增强
```

---

## 1. 背景与动机

### 1.1 Trader 节点的价值密度问题（A2）

Trader 节点当前职责：
1. 复述 RM 方向（约束："禁止重新判断方向"）
2. 评估止损位（实际 RM 已给止损）
3. 评估流动性与执行窗口（与下游 aggressive_debator（流动性风险）职责重叠）
4. 输出 `FINAL TRANSACTION PROPOSAL: **<RM 原始评级>**`

经分析，Trader 节点是：
- **冗余节点**：上述 2、3 项工作下游已重复做一次
- **格式化层**：第 4 项只是把 RM 评级原样输出
- **耗时点**：占整个 graph 1 个 LLM 调用 + 1 次 memory 检索

**结论**：移除 Trader，原职责重新分配——止损归 RM（已有），流动性归风控团队的 liquidity_analyst（已有），最终评级直接以 RM 输出为准。

### 1.2 分析师下游信号难提取问题（B1）

4 个分析师（market / news / fundamentals / social）当前输出都是自由 markdown：
- 下游 Bull/Bear/RM/PM/风控 都要重读全文 + 自己提炼
- LLM 阅读自由文本时**关键数字易漏**（如净利增速、行业 PE、KOL 一致性）
- 多个下游 agent 重复做 NLU，token 浪费

**解决**：每个分析师 prompt 末尾**强制输出 YAML SUMMARY 块**，下游优先从 SUMMARY 中提取关键事实。

> ⚠️ 重要约束：SUMMARY 是给 LLM 看的结构化文本，**不是给 Python 解析的 schema**。不要写 yaml.safe_load 解析层，让 LLM 自己读 YAML 文本即可。

### 1.3 分析师细节信号缺失（C）

- market：技术指标无历史分位，不知道"RSI=72"是顶部信号还是延续
- news：新闻无时间窗 + 已 priced-in 标注，RM 无法判断 alpha 来源
- social：仅定性叙事，无法进入 RM 的加权评分
- fundamentals：估值无行业对照
- bull/bear：论据无信心度自评

---

## 2. 总体范围

### 2.1 必须修改的文件（12 个）

**Graph 层**：
- `tradingagents/graph/setup.py` — 移除 Trader 节点 + 边重连
- `tradingagents/graph/trading_graph.py` — 移除 trader 节点构造调用（如有）

**Agent 层**：
- `tradingagents/agents/trader/trader.py` — **保留文件但停用导出**（向前兼容；可删但不在本 spec 内）
- `tradingagents/agents/managers/portfolio_manager.py` — 移除对 `trader_investment_plan` 的依赖
- `tradingagents/agents/managers/research_manager.py` — 第三步行动方案补全原 Trader 的"流动性评估"提示
- `tradingagents/agents/risk_mgmt/aggressive_debator.py` — 移除 `trader_investment_plan` 引用
- `tradingagents/agents/risk_mgmt/conservative_debator.py` — 同上
- `tradingagents/agents/risk_mgmt/neutral_debator.py` — 同上
- `tradingagents/agents/analysts/market_analyst.py` — 加 SUMMARY + 历史分位
- `tradingagents/agents/analysts/news_analyst.py` — 加 SUMMARY + 时间窗 + priced-in
- `tradingagents/agents/analysts/fundamentals_analyst.py` — 加 SUMMARY + 行业 PE 对照（基于已有数据）
- `tradingagents/agents/analysts/social_media_analyst.py` — 加 SUMMARY + 量化情绪
- `tradingagents/agents/researchers/bull_researcher.py` — 论据加信心度自评
- `tradingagents/agents/researchers/bear_researcher.py` — 同上

**State 层**：
- `tradingagents/agents/utils/agent_states.py` — `trader_investment_plan` 标记为 deprecated（保留字段以兼容旧报告）

**报告/CLI 层**：
- `main.py` — 检查 `_save_report` 是否引用 trader 报告，若引用则改为可选
- `cli/` 下任何引用 `trader_investment_plan` 的位置 — 改为可选

### 2.2 不修改的文件

- `tradingagents/graph/conditional_logic.py`（多空和风控辩论的轮数判断不涉及 Trader）
- `tradingagents/graph/propagation.py`、`signal_processing.py`、`reflection.py`、`checkpointer.py`
- 所有 `dataflows/` 下的数据接入层
- 所有 `llm_clients/`
- `tradingagents/agents/utils/agent_utils.py`（除非要新增工具，本 spec 不引入新工具）

---

## 3. 任务 A2：移除 Trader 节点

### 3.1 改动 A2-1：Graph 拓扑

**文件**：`tradingagents/graph/setup.py`

**Before**（参考第 150、203、204 行附近，确切行号以实际文件为准）：
```python
workflow.add_node("Trader", trader_node)
...
workflow.add_edge("Research Manager", "Trader")
workflow.add_edge("Trader", "Aggressive Analyst")
```

**After**：
```python
# Trader 节点已废弃，RM 输出直接进入风控辩论
# workflow.add_node("Trader", trader_node)  # REMOVED in 05
...
workflow.add_edge("Research Manager", "Aggressive Analyst")
# workflow.add_edge("Trader", "Aggressive Analyst")  # REMOVED in 05
```

如 `setup.py` 中 `trader_node` 参数已不再使用，从函数签名中移除该参数。同步检查并修改 `tradingagents/graph/trading_graph.py` 中调用 `setup` 的地方。

### 3.2 改动 A2-2：RM 承接"流动性评估"提示

**文件**：`tradingagents/agents/managers/research_manager.py`

在第三步"制定投资计划"的"行动方案"段落后、"价位一致性自检"之前，新增一小节：

```markdown
- **执行可行性自检**（原 Trader 职责，并入 RM）：
  - 简述该标的日均成交额量级（小盘 < 5000 万 / 中盘 5000 万-5 亿 / 大盘 >5 亿）
  - 建仓比例是否会对单日成交额造成 >5% 冲击？若是，建议分批节奏
  - 是否临近重大事件窗口（财报披露、解禁日、政策细则）？若 7 个交易日内有重大事件，标注"建议事件后再加仓"
  - 输出格式：`执行可行性：[流动性档位] / [是否需分批] / [近期事件标注]`
```

### 3.3 改动 A2-3：风控三人移除 Trader 引用

**3 个文件**：`aggressive_debator.py` / `conservative_debator.py` / `neutral_debator.py`

每个文件做相同的修改：

**Before**：
```python
trader_decision = state["trader_investment_plan"]
...
prompt = f"""...
**Research Manager 的投资方案（含评级、评分、风控审查指引）：**
{investment_plan}

**交易员的执行方案：**
{trader_decision}

**辩论要求**：
..."""
```

**After**：
```python
# trader_decision 已废弃（05 优化），改为只引用 RM 方案
...
prompt = f"""...
**Research Manager 的投资方案（含评级、评分、价位区间、执行可行性、风控审查指引）：**
{investment_plan}

**辩论要求**：
..."""
```

**注意**：保留 `state["trader_investment_plan"]` 的读取代码可选，但**不要再用**——若 state 中此键不存在，应当容错（用 `.get("trader_investment_plan", "")`），避免老 checkpoint 报错。

### 3.4 改动 A2-4：PM 移除 Trader 引用

**文件**：`tradingagents/agents/managers/portfolio_manager.py`

**Before**：
```python
trader_plan = state["trader_investment_plan"]
...
curr_situation = f"{research_plan}\n\n{trader_plan}"
past_memories = memory.get_memories(curr_situation, n_matches=3)
...
prompt = f"""...
- Research Manager's investment plan: **{research_plan}**
- Trader's transaction proposal: **{trader_plan}**
..."""
```

**After**：
```python
# trader_plan 已废弃（05 优化），所有职责并入 RM
...
curr_situation = research_plan
past_memories = memory.get_memories(curr_situation, n_matches=3)
...
prompt = f"""...
- Research Manager's investment plan: **{research_plan}**
..."""
```

PM prompt 中所有提到 "Trader 提供的是执行方案" / "Trader 标记的执行风险" 等段落需要重写或删除。统一改为：**RM 方案已包含执行可行性评估，PM 直接基于 RM 方案 + 风控辩论做最终决策**。

PM 决策规则保留（第二步评级调整 ±1 档的规则不变），仅去掉对 Trader 的依赖。

### 3.5 改动 A2-5：State Schema

**文件**：`tradingagents/agents/utils/agent_states.py`

`trader_investment_plan` 字段：
- **保留**（不要删除，避免旧 checkpoint 不兼容）
- 在字段定义旁加注释：`# DEPRECATED in 05: Trader node removed, kept for backward compat`

### 3.6 改动 A2-6：main.py 报告生成

**文件**：`main.py`

搜索 `trader_investment_plan` 或 `trader_plan` 的所有引用，把"必填"改为"可选输出"。

预期变化：报告目录 `reports/<ticker>_<timestamp>/` 中原本的 `4_trader/` 子目录将不再被写入；上游报告章节自然减少一节。

### 3.7 改动 A2-7：trader.py 文件本身

**文件**：`tradingagents/agents/trader/trader.py`

**保留文件**，但在文件顶部加一行废弃注释：
```python
"""DEPRECATED in optimization 05: Trader node removed from graph.

This file is kept for git history and potential future reuse.
The node's responsibilities have been redistributed:
- Direction & stop-loss → Research Manager
- Liquidity check → Liquidity Analyst (formerly 'aggressive_debator')
"""
```

不要从 `__init__.py` 移除 import（如果存在），避免遗漏的引用导致 ImportError；改为不在 graph 中实际调用即可。

### 3.8 A2 验收标准

1. ✅ `python -c "from tradingagents.graph.trading_graph import TradingAgentsGraph"` 无错
2. ✅ 跑完一支股票，`reports/<ticker>_*/` 目录下**不出现** `4_trader/decision.md`
3. ✅ LLM 调用日志 `llm_calls/` 不再有 "Trader" / "交易执行专家" 关键词的 prompt
4. ✅ PM 输出的"决策卡"9 个字段全部填写，止损位字段从 RM 方案中正确提取（不再来自 Trader）
5. ✅ Graph 边连接：RM 的下一节点直接是 Aggressive Analyst

---

## 4. 任务 B1：分析师 YAML SUMMARY

### 4.1 设计原则

- SUMMARY 是 **Markdown 代码块** 嵌入在分析师 markdown 报告的**末尾**，使用 ```yaml ... ``` 包裹
- 不要写 Python YAML 解析代码——SUMMARY 给下游 LLM 看的结构化提示，不是程序消费的 schema
- 每个字段必须填写——LLM 若数据缺失，应填 `null` 或 `"不适用"`，不允许省略字段
- 字段名固定、单位固定、取值集合受限（如 rating ∈ {正面, 中性, 负面}）

### 4.2 各分析师的 SUMMARY Schema

#### 4.2.1 market_analyst

放在原 prompt"### 五、风险提示"之后、`get_language_instruction()` 之前。

```yaml
SUMMARY:
  trend: 上行 / 下行 / 震荡           # 三选一
  momentum: 强 / 中 / 弱               # 三选一
  rsi_value: <数值>                    # 当前 RSI 值
  rsi_pct_1y: <0-100>                  # RSI 在过去 1 年的历史分位（详见 C1）
  macd_signal: bullish / bearish / neutral
  key_support: <数值>                  # 关键支撑位（A 股用元，美股用美元）
  key_resistance: <数值>               # 关键阻力位
  atr_pct: <0-100>                     # ATR / 当前价 × 100（波动率）
  volume_state: 放量 / 缩量 / 正常
  rating: BUY / HOLD / SELL
  confidence: <1-5>                    # 整体技术面信号的置信度
```

#### 4.2.2 news_analyst

放在原 prompt"末尾汇总表格"之后、`get_language_instruction()` 之前。

```yaml
SUMMARY:
  net_sentiment: 正面 / 负面 / 中性
  num_events_total: <整数>             # 总事件数（去重后）
  key_events:                          # 至多 5 条，按影响排序
    - title: <≤30 字>
      category: 公司 / 行业 / 宏观 / 机构
      horizon: 短期(≤1周) / 中期(1-3月) / 长期(>3月)
      priced_in_p: <0-100>             # 已 priced-in 概率（详见 C2）
      impact: +大 / +中 / +小 / 0 / -小 / -中 / -大
      credibility: 高 / 中 / 低
  research_consensus_rating: BUY / HOLD / SELL / null    # 若研报数据可用
  research_consensus_target_price: <数值或 null>
```

#### 4.2.3 fundamentals_analyst

放在原 prompt"### 八、关键指标汇总表"之后。

```yaml
SUMMARY:
  pe_ttm: <数值>                       # 必须使用「系统计算」值
  pe_zone: 高估 / 合理 / 低估
  pe_industry_median: <数值或 null>    # 若数据中含行业对照，否则 null（详见 C4）
  pe_vs_industry: 高于 / 接近 / 低于 / 不可比
  growth_yoy_revenue: <百分比>
  growth_yoy_profit: <百分比>
  roe: <百分比>
  debt_ratio: <百分比>
  fcf_quality: 高 / 中 / 低             # 经营性现金流 / 净利润比
  red_flags:                            # 列表，至多 3 条；无则填 []
    - <红旗描述，≤20 字>
  rating: 正面 / 中性 / 负面
```

#### 4.2.4 social_media_analyst

放在原 prompt"Markdown 汇总表格"之后。

```yaml
SUMMARY:
  net_sentiment: 偏多 / 偏空 / 分歧
  bull_post_pct: <0-100>               # 偏多帖子占比
  bear_post_pct: <0-100>               # 偏空帖子占比
  neutral_post_pct: <0-100>            # 中性占比；三者和必须 = 100
  kol_consensus: 一致看多 / 一致看空 / 分歧 / 无明显 KOL
  kol_count_observed: <整数>           # 观察到的 KOL 数量
  sentiment_trend_7d: <-100 ~ +100>    # 近 7 天情绪变化净值（粗估）
  key_narratives:                       # 列表，至多 3 条
    - <叙事主题，≤20 字>
  rating: BUY / HOLD / SELL             # 仅基于情绪信号的建议
```

### 4.3 实施细节（每个分析师文件）

**模板**：在每个分析师的 `system_message` 拼接末尾、`get_language_instruction()` 之前，加入一段强制要求：

```python
"## 强制输出：SUMMARY 块（位于报告末尾）\n"
"在报告所有正文章节和汇总表格之后，**必须**附加一个 YAML 代码块，"
"格式严格如下（字段名、单位、取值集合不可变）：\n\n"
"```yaml\n"
"<填入对应 schema>\n"
"```\n\n"
"## SUMMARY 规则\n"
"- 字段缺失时填 null 或 \"不适用\"，不允许省略字段名\n"
"- 取值必须落在 schema 允许的集合内（如 rating ∈ {...}）\n"
"- 数值字段保留 2 位小数；百分比字段直接填数字（不带 % 符号）\n"
"- 该 SUMMARY 块是下游 RM / 风控团队的核心信息源，宁缺勿错\n"
```

### 4.4 下游消费侧改动

**Bull / Bear / RM / PM / 3 个风控** 的 prompt 末尾（措辞规范之前）增加一段：

```python
"## 报告解读优先级\n"
"上游 4 个分析师的报告**末尾**都附有 YAML SUMMARY 块。请按以下优先级使用：\n"
"1. **优先**从 SUMMARY 中提取数值类事实（PE、增速、情绪占比、关键事件等）\n"
"2. **次之**阅读正文章节的定性论述以理解 SUMMARY 数值的来龙去脉\n"
"3. 若 SUMMARY 与正文矛盾，以正文为准并在评分理由中说明数据不一致\n"
```

### 4.5 B1 验收标准

1. ✅ 跑完一支股票，4 个分析师报告 `1_analysts/{market,news,fundamentals,social}.md` 末尾都有 ```yaml SUMMARY ... ``` 块
2. ✅ SUMMARY 字段无缺失（用 `grep -A 30 "^SUMMARY:" reports/*/1_analysts/*.md` 抽样目检）
3. ✅ 取值集合合法（如 `rating` 不出现 `STRONG_BUY` / `Strong Buy` 等非法值）
4. ✅ RM 报告中能观察到对 SUMMARY 数值的引用（如"参考 fundamentals SUMMARY 中 pe_ttm=19.5"）

---

## 5. 任务 C：分析师细节增强

### 5.1 C1：market_analyst 加历史分位

**文件**：`tradingagents/agents/analysts/market_analyst.py`

**改动点**：在 system_message 的 "### 二、核心技术指标分析" 段后，新增子要求：

```markdown
**每个核心指标必须输出"历史分位"**：
- 基于已获取的至少 365 天行情数据，估算当前指标值在过去 1 年的百分位
- 例如："RSI=72，处于过去 1 年 87 分位（高位区间）"
- 若数据不足 1 年（如新股），标注"数据不足，分位仅供参考"
- 在 SUMMARY 块的 `rsi_pct_1y` 字段填写该数值（0-100）
```

> 注：历史分位由 LLM 基于已获取的指标历史数据**估算**，不需要新增数据源工具。如果 Qoder 评估需要工具支持，可在后续优化中加 `get_indicator_percentile(ticker, indicator, current_value)`。

### 5.2 C2：news_analyst 加事件时间窗 + priced-in

**文件**：`tradingagents/agents/analysts/news_analyst.py`

**改动点**：在 `system_message` 的"## 行业分类"段之后、"## 输出格式"段之前，新增：

```markdown
## 事件特征标注（每条关键事件必须标注）
对所有进入汇总表格的事件，必须额外标注以下两项：

1. **时间窗口**（horizon）：
   - 短期（≤1 周）：交易日级别可观测影响，如财报披露、突发事件
   - 中期（1-3 月）：季度级影响，如政策细则、产品发布
   - 长期（>3 月）：结构性影响，如行业格局变化

2. **已 priced-in 概率**（priced_in_p）：0-100 的估计值
   - 90-100：已被市场充分定价，对未来股价边际影响小
   - 50-90：部分定价，仍有发酵空间
   - 0-50：尚未被定价，潜在 alpha 来源
   - 判断依据：事件首次披露日期、市场反应幅度、卖方研报覆盖度

汇总表格新增两列：| 时间窗 | 已定价概率 |
```

### 5.3 C3：social_media_analyst 加量化情绪

**文件**：`tradingagents/agents/analysts/social_media_analyst.py`

**改动点**：在 `system_message` 的"## 分析要求"段中，把现有"整体情绪"子项替换为：

```markdown
- **量化情绪指标**（必须输出数值）：
  - 多头帖子占比：基于采样到的帖子中明确表达看多观点的比例（0-100%）
  - 空头帖子占比：明确看空的比例
  - 中性占比：100% − 多头 − 空头
  - **三者之和必须等于 100%**
- **KOL 一致性**（必须输出）：
  - 识别样本中粉丝量大或发帖活跃的 KOL（≥3 个为有效观察）
  - 标注是"一致看多 / 一致看空 / 分歧"
  - 若 KOL 不足 3 个，标注"无明显 KOL"
- **7 日情绪变化净值**（必须输出）：
  - 比较最近 7 天与之前 7 天的多空占比变化
  - 输出区间 -100 ~ +100 的整数（正数 = 偏多增强；负数 = 偏空增强）
```

### 5.4 C4：fundamentals_analyst 加行业 PE 对照（基于现有数据，不引入新工具）

**文件**：`tradingagents/agents/analysts/fundamentals_analyst.py`

**改动点**：在 system_message 的 "### 二、估值分析" 段中，新增子要求：

```markdown
**行业 PE 对照（必须尝试，数据可用时强制输出）**：
- 检查 fundamentals 数据中是否含「行业 PE」、「行业平均 PE」、「同业可比 PE」等字段
- 若可用，在估值分析中显式输出："当前 PE=X 倍 vs 行业中位数 Y 倍，处于 [高于/接近/低于] 水平"
- 若 raw_data 中无行业数据，明确标注 "行业 PE 数据不可用，仅基于自身历史分位判断估值水位"，并在 SUMMARY 的 `pe_industry_median` 字段填 null
- **禁止**凭印象给出行业 PE（如"行业平均约 25 倍"——若无数据来源，不要写）
```

> 注：本任务**不引入** `get_industry_pe_distribution` 新工具，仅基于已有数据增强 prompt。后续若发现行业对照频繁缺失，可在 06 优化中加新工具。

### 5.5 C5：bull/bear 论据加信心度自评

**文件**：`tradingagents/agents/researchers/bull_researcher.py` + `bear_researcher.py`

**改动点**：在两个 researcher 的 prompt 中"Key points to focus on" 列表之后，新增：

```markdown
## 论据格式要求（强制）
每条论据必须使用以下格式输出：

> **论据 N**：<论据描述>
> - **证据类型**：Hard fact / Catalyst / 估值类比 / 情绪叙事
> - **信心度**：<1-5 分>（1=纯猜测，5=有强数据/事件支撑）
> - **依据**：<引用具体数据源，如"fundamentals SUMMARY 中 pe_ttm=19.5"或"news 中 Q2 业绩预告增 80%"或"market SUMMARY 中 rsi_pct_1y=87"等>

**信心度评分标准**：
- 5 分：有具体可验证的硬数据/已发生事件支撑
- 4 分：有间接数据 + 合理推断
- 3 分：有定性论述但无硬数据
- 2 分：基于行业经验或类比
- 1 分：直觉判断，无明确依据

**禁止**：每条论据若没有给出"依据"段落，视为无效论据，下游 RM 评分会按 0 处理。
```

### 5.6 RM 配套改动（C 任务的下游配套）

**文件**：`tradingagents/agents/managers/research_manager.py`

在第一步"按证据类型加权"段落，**新增信心度加权层**：

```markdown
**信心度加权（叠加于证据类型权重）**：
- Bull/Bear 提供的每条论据已自评信心度（1-5）
- 最终权重 = 证据类型权重 × (信心度 / 5)
- 例：Hard fact（基础权重 3）+ 信心度 4 → 实际权重 = 3 × 0.8 = 2.4
- 例：Catalyst（基础权重 2）+ 信心度 2 → 实际权重 = 2 × 0.4 = 0.8
- 若论据未标注信心度，按 3（默认中等）处理
```

### 5.7 C 任务验收标准

1. ✅ market 报告中能看到至少 3 个指标带"历史分位 X%"标注
2. ✅ news 报告的汇总表格新增"时间窗""已定价概率"两列，且每行都有填值
3. ✅ social 报告中 SUMMARY 的 `bull_post_pct + bear_post_pct + neutral_post_pct === 100`
4. ✅ fundamentals 报告若数据中含行业 PE 字段，估值分析必有对照；若不含，SUMMARY 中 `pe_industry_median` 为 null
5. ✅ Bull/Bear 报告中每条论据都有"信心度 X/5"标注
6. ✅ RM 报告中能看到信心度加权后的实际权重（如 "Hard fact × 0.8 = 2.4"）

---

## 6. 实施顺序与依赖

```
Phase 1（独立，可并行）
├── A2: Trader 移除（4 个文件改动）
└── C1-C5: 分析师细节增强（6 个文件改动，C5 改动 2 个 researcher）

Phase 2（依赖 Phase 1）
└── B1: 分析师 SUMMARY + 下游消费侧（10 个文件改动）

Phase 3（验收）
└── 完整跑通一支股票 + 抽查报告
```

**推荐顺序**：

1. **Day 1 上午**：A2 全部 6 处文件改动 + 跑通验证（确保 graph 没断）
2. **Day 1 下午**：B1 实施 4 个分析师 SUMMARY 块 + 下游 5 个 agent 的"读 SUMMARY"提示
3. **Day 2 上午**：C1-C5 五项细节增强
4. **Day 2 下午**：完整跑通 1 支 A 股 + 1 支美股，按验收标准 grep 抽查

---

## 7. 测试与验证

### 7.1 单元级验证（每个改动文件后立即做）

```bash
# 1. 语法检查
python -c "import py_compile; [py_compile.compile(f, doraise=True) for f in [
    'tradingagents/agents/managers/research_manager.py',
    'tradingagents/agents/managers/portfolio_manager.py',
    'tradingagents/graph/setup.py',
    # ...每个改动文件
]]; print('OK')"

# 2. import 检查
python -c "from tradingagents.graph.trading_graph import TradingAgentsGraph; print('import OK')"
```

### 7.2 集成级验证（A2 完成后做一次）

```bash
# 跑一支股票看 graph 是否正常
# 在 main.py 把 _TICKERS 改为只跑 1 支，运行
python main.py

# 验收点：
# - llm_calls/ 中无 "Trader" / "交易执行专家" 关键词
# - reports/<ticker>_*/ 目录下无 4_trader/
# - 总 LLM 调用数较改动前减少 1（多空辩论 2 轮 + 风控 1 轮 = 14 次预期；改动前是 15 次）
```

### 7.3 集成级验证（B1 + C 完成后做）

```bash
# 跑 1 支 A 股 + 1 支美股
# 抽查
grep -A 20 "^SUMMARY:" reports/*/1_analysts/market.md       # 应有 yaml 块
grep -A 30 "^SUMMARY:" reports/*/1_analysts/fundamentals.md
grep -E "信心度.*[1-5]" reports/*/2_research/bull.md         # 应能找到信心度标注
grep -E "历史分位|分位" reports/*/1_analysts/market.md       # 应能找到
grep -E "已定价概率|priced_in" reports/*/1_analysts/news.md  # 应能找到
```

### 7.4 决策一致性回归

在改动前后**同一日期同一标的**各跑 1 次，对比：
- PM 决策卡的"评级"应保持一致（允许 ±1 档差异，因为辩论有 LLM 随机性）
- 评级的"核心催化 / 核心风险"主题应保持相似
- 若评级方向完全翻转（如 BUY → SELL），需排查 prompt 改动是否引入了 bug

---

## 8. 风险与回滚

### 8.1 已知风险

| 风险 | 影响 | 缓解 |
|---|---|---|
| 移除 Trader 后 RM 的"执行可行性"段未起到补偿作用 | 决策落地性下降 | RM prompt 已新增"执行可行性自检"段（3.2），效果不佳则补充范例 |
| YAML SUMMARY 被 LLM 写错格式（如缺字段、取值越界）| 下游 LLM 解读混乱 | SUMMARY 规则严格 + 抽样目检 + 必要时给 1-2 个完整 example |
| 信心度自评导致 LLM 集中给高分（评分通胀）| RM 加权失真 | 在评分标准里强调"信心度 5 分必须有可验证硬数据" |
| 旧报告依赖 trader_investment_plan 字段做渲染 | 报告生成报错 | state schema 中保留字段（3.5），代码用 `.get` 容错 |

### 8.2 回滚策略

- 全部改动在一个 feature branch 实施，验收完成后再 merge
- 每个 Phase 单独 commit，便于按 Phase 回滚
- 若发现严重问题：`git revert <phase commit>`

---

## 9. 完成定义（DoD）

满足以下全部条件视为完成：

1. ✅ A2/B1/C 三组改动全部落地（12 个文件）
2. ✅ 跑通 2 支股票（1 A 股 + 1 美股），无报错
3. ✅ 7.3 节的所有 grep 验收点通过
4. ✅ PM 决策卡 9 个字段全部填写无 "TBD"
5. ✅ LLM 调用数较改动前减少 1（Trader 移除）
6. ✅ `docs/05_trader_removal_and_analyst_signal_upgrade_验收报告.md` 完成写作

---

## 10. 不在本 spec 范围内的事项

明确**不**做的事，避免 scope creep：

- ❌ 不引入新数据源工具（`get_industry_pe_distribution` 等留给后续优化）
- ❌ 不重命名风控 3 人（aggressive→liquidity 等，留给后续优化 06）
- ❌ 不引入 portfolio-level state（组合层视角留给后续优化 07）
- ❌ 不修改辩论轮数 / max_debate_rounds 等超参（这是 main.py 配置，不属本 spec）
- ❌ 不修改 LLM 客户端 / 422 重试机制 / 措辞规范常量
- ❌ 不修改 reports 目录结构或 `_save_report` 主体逻辑（除非 trader 章节缺失导致写入失败）

---

## 附录 A：SUMMARY YAML 完整示例（4 个分析师各一份）

### market_analyst SUMMARY 示例

```yaml
SUMMARY:
  trend: 上行
  momentum: 强
  rsi_value: 68.5
  rsi_pct_1y: 82
  macd_signal: bullish
  key_support: 285.50
  key_resistance: 320.00
  atr_pct: 3.20
  volume_state: 放量
  rating: BUY
  confidence: 4
```

### news_analyst SUMMARY 示例

```yaml
SUMMARY:
  net_sentiment: 正面
  num_events_total: 12
  key_events:
    - title: Q2 业绩预告同比 +85%
      category: 公司
      horizon: 短期(≤1周)
      priced_in_p: 30
      impact: +大
      credibility: 高
    - title: 行业新规将于 Q3 落地
      category: 行业
      horizon: 中期(1-3月)
      priced_in_p: 60
      impact: +中
      credibility: 中
  research_consensus_rating: BUY
  research_consensus_target_price: 350.00
```

### fundamentals_analyst SUMMARY 示例

```yaml
SUMMARY:
  pe_ttm: 19.52
  pe_zone: 合理
  pe_industry_median: 24.10
  pe_vs_industry: 低于
  growth_yoy_revenue: 35.20
  growth_yoy_profit: 28.10
  roe: 22.50
  debt_ratio: 42.30
  fcf_quality: 高
  red_flags:
    - 流动比率<1，短期偿债压力
  rating: 正面
```

### social_media_analyst SUMMARY 示例

```yaml
SUMMARY:
  net_sentiment: 偏多
  bull_post_pct: 58
  bear_post_pct: 22
  neutral_post_pct: 20
  kol_consensus: 一致看多
  kol_count_observed: 5
  sentiment_trend_7d: 15
  key_narratives:
    - Q2 业绩超预期预期升温
    - 行业新规带来订单增量
    - 估值已修复至合理区间
  rating: BUY
```

---

## 附录 B：Bull/Bear 论据信心度示例

### Bull 论据正确示例

> **论据 1**：公司 Q2 营收同比增速从 +35% 加速至 +42%，且毛利率提升 2.3 pct
> - **证据类型**：Hard fact
> - **信心度**：5/5
> - **依据**：fundamentals SUMMARY 中 growth_yoy_revenue=42.0；fundamentals 报告"成长性分析"段落引用财报披露数据
>
> **论据 2**：6 月新品发布会预期催化估值切换
> - **证据类型**：Catalyst
> - **信心度**：3/5
> - **依据**：news SUMMARY 中 key_events 第 2 条 "6 月 18 日新品发布会，priced_in_p=45"

### Bear 论据错误示例（必须避免）

> **论据 1**：估值偏高
> - **证据类型**：估值类比
> - **信心度**：4/5
> - **依据**：（无）← ❌ 缺依据，RM 评分时按 0 分处理

---

## 附录 C：移除 Trader 后 PM 输入字段变化对照表

| 字段 | 改动前 | 改动后 | 备注 |
|---|---|---|---|
| `state["investment_plan"]` | RM 方案 | RM 方案 + 执行可行性段落 | 内容增强 |
| `state["trader_investment_plan"]` | Trader 方案文本 | 空字符串（向前兼容）| Deprecated |
| `state["risk_debate_state"]` | 含 3 人辩论 | 含 3 人辩论（不变）| — |
| PM 检索 memory 用的 curr_situation | RM 方案 + Trader 方案 | RM 方案 | 简化 |
| PM 决策依据 | RM + Trader + Risk Debate | RM + Risk Debate | 简化 |

完。
