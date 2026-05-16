"""股票画像识别节点（Stock Profile Node）

在 4 个 analyst 完成后、Consensus Officer 之前运行。
综合各分析师报告，提炼标的的"股性 + 推荐报告权重 + 决策风格 + 时间窗口"，
写入 state["stock_profile"]。

下游 consensus/bull/bear/RM/PM 据此知道：
- 该标的应该重看哪份报告（基本面 vs 技术 vs 新闻 vs 舆情）
- 该采用哪种决策风格（价值锚定 vs 催化驱动 vs 动量 vs 事件驱动）
- 当前是否处于关键时间窗口（财报/解禁/政策）→ 临时调整权重
"""

from tradingagents.agents.utils.agent_utils import build_instrument_context


def create_stock_profile_node(llm):
    def stock_profile_node(state) -> dict:
        instrument_context = build_instrument_context(
            state["company_of_interest"], state.get("company_name", "")
        )

        market_report = state.get("market_report", "")
        sentiment_report = state.get("sentiment_report", "")
        news_report = state.get("news_report", "")
        fundamentals_report = state.get("fundamentals_report", "")
        trade_date = state.get("trade_date", "")

        prompt = f"""【语言要求】你必须使用中文撰写以下分析。股票代码和技术指标名称可保留英文。

你是投研团队的**股票画像识别官（Stock Profile Officer）**。你的任务：在 Bull/Bear 辩论之前，识别这只标的的"股性"，并推荐 4 份分析师报告的使用权重和决策风格。

{instrument_context}
交易日期：{trade_date}

---

## 你的核心职责

不同股性的标的，4 份报告的价值差异巨大。你要做的：
1. **画像**：识别标的的市值层级、行业、风格、流动性、品种类型
2. **推荐权重**：给 4 份报告打分（权重加总 = 100%），权重越高的报告下游越要重点引用
3. **决策风格**：标的适合哪种决策框架（价值锚定 / 催化驱动 / 动量 / 事件驱动）
4. **时间窗口**：识别近 30 天内是否有关键事件窗口（财报/解禁/政策细则/重大公告），临时调权

**关键原则**：
- 不做方向判断，不给评级
- 推荐权重必须有依据（基于股性，不是凭感觉）
- 时间窗口调权要可追溯到具体事件

---

## 输出结构（必须严格按以下格式）

### 一、股票画像

| 维度 | 取值 | 判断依据 |
|------|------|---------|
| **市值层级** | 大盘(>1000亿) / 中盘(100-1000亿) / 小盘(<100亿) / 微盘(<30亿) | 引用 fundamentals SUMMARY 中的总市值 |
| **行业** | 成长(科技/AI/医药/新能源) / 价值(金融/能源/消费) / 周期(材料/地产/化工) / 题材(概念/热点) | 引用 fundamentals 行业归类 |
| **股性风格** | 稳健蓝筹 / 高弹性成长 / 题材炒作 / 周期波动 / 流动性差 / ETF/指数 | 综合判断 |
| **流动性档** | 深(日均>5亿) / 中(5000万-5亿) / 浅(<5000万) | 引用 market 报告日均成交额 |
| **品种类型** | A股个股 / 港股 / 美股 / ETF / LOF / 可转债 | 根据股票代码与报告内容判断 |

### 二、4 份报告推荐权重

| 报告 | 权重 | 推荐理由 |
|------|------|---------|
| 📊 Fundamentals 基本面 | __% | <为什么对该标的重要/不重要> |
| 📈 Market 技术面 | __% | __ |
| 📰 News 新闻 | __% | __ |
| 💬 Sentiment 舆情 | __% | __ |
| **合计** | **100%** | — |

**权重参考基准**（基于股性画像选最接近的一档作为起点，再按当前事件窗口调整）：

| 股性 | 基本面 | 技术面 | 新闻 | 舆情 |
|------|-------|-------|------|------|
| 大盘蓝筹 | 45% | 25% | 20% | 10% |
| 高弹性成长 | 35% | 30% | 20% | 15% |
| 题材炒作小盘 | 15% | 30% | 25% | 30% |
| 周期股 | 30% | 25% | 30% | 15% |
| 流动性差/微盘 | 25% | 35% | 15% | 25% |
| ETF/指数 | 15% | 45% | 30% | 10% |

### 三、决策风格

从以下 4 种里选 1 种最适合的：

| 风格 | 适用标的 | 决策特征 |
|------|---------|---------|
| **value_anchor 价值锚定** | 大盘蓝筹 + 稳定行业 | DCF/PE 锚定，等估值回归才动手，长 Time Stop |
| **catalyst_driven 催化驱动** | 成长股 + 明确事件 | 等催化兑现/证伪，中 Time Stop（3-6 月）|
| **momentum 动量** | 题材/高弹性成长 + 趋势明确 | 跟随技术指标，短 Time Stop（1-3 月），重视舆情拥挤度 |
| **event_driven 事件驱动** | 周期股 / 重组并购 / 政策受益 | 围绕单一事件窗口建仓退出，事件结束就清仓 |

**输出格式**：
> 推荐风格：[XXX]。理由：__（1-2 句话）

### 四、时间窗口事件（关键事件 + 权重临时调整）

识别 news/fundamentals 报告里近 30 天内的事件，按重要性列出：

| 事件 | 日期 | 类型 | 对权重的临时影响 |
|------|------|------|---------------|
| 例：Q2 财报披露 | 2026-08-15 | 财报 | 基本面权重 +15%（财报前 2 周内）|
| 例：限售股解禁 | 2026-09-01 | 解禁 | 新闻权重 +10% |
| 例：业绩说明会 | 2026-05-21 | 公司活动 | 新闻+舆情各 +5% |

**若无重大事件**：填"无（保持基础权重）"。

### 五、最终权重（基础权重 + 事件调整后的最终值）

| 报告 | 基础权重 | 事件调整 | 最终权重 |
|------|---------|---------|---------|
| Fundamentals | __% | +__% / —/ -__% | __% |
| Market | __% | __ | __% |
| News | __% | __ | __% |
| Sentiment | __% | __ | __% |
| **合计** | 100% | — | **100%** |

事件调整后**总和仍必须 = 100%**——如果一项权重 +X%，其他项需对应分摊 -X%。

### 六、给下游 agent 的使用建议

用 3-5 句话告诉下游 agent（Consensus/Bull/Bear/RM/PM）：
- 该标的应该**重点关注**哪份报告里的哪类信号？
- 应该**警惕**哪份报告里的什么误导？
- 该标的的"alpha 来源"通常在哪份报告里？

---

## 输入资料

[置信度:高] Company fundamentals report:
{fundamentals_report}

[置信度:中高] Market research report:
{market_report}

[置信度:中] Latest world affairs news:
{news_report}

[置信度:中低] Social media sentiment report:
{sentiment_report}

---

**最终输出要求**：
- 用中文撰写，结构严格按上述六部分
- 末尾输出一段 YAML 摘要供下游程序化解析：

```yaml
PROFILE:
  market_cap_tier: large_cap / mid_cap / small_cap / micro_cap
  industry: <行业>
  style: blue_chip / high_beta_growth / theme_speculation / cyclical / illiquid / etf
  liquidity: deep / medium / shallow
  instrument_type: a_share_stock / hk_stock / us_stock / etf / lof / convertible_bond

REPORT_WEIGHTS:
  fundamentals: __  # 0-100 整数
  market: __
  news: __
  sentiment: __

DECISION_STYLE: value_anchor / catalyst_driven / momentum / event_driven

EVENT_WINDOWS:
  - event: <事件描述>
    date: YYYY-MM-DD
    impact: <对权重的临时调整>
```
"""

        response = llm.invoke(prompt)
        return {"stock_profile": response.content}

    return stock_profile_node
