"""宏观上下文识别节点（Macro Context Node）

在 4 个 analyst 完成后、Stock Profile Officer 之前运行。
综合 news + fundamentals 中的宏观信息，提炼"宏观环境对当前标的的影响"，
写入 state["macro_context"]。

下游 stock_profile/consensus/RM 据此知道：
- 当前利率周期阶段（宽松/紧缩/拐点）
- 流动性松紧（M2、社融、北向资金、融资余额方向）
- 主要宏观风险事件未来 30 天
- 当前 industry 相对宏观环境的方向（顺风/逆风/无关）

参考真实投研团队的 Macro Strategist 角色 —— 真实头部团队（高盛/桥水）都有专人跟踪宏观。
对 A 股高估值成长股（AI 算力等），宏观环境（流动性松紧、利率周期）是估值容忍度的关键决定因素。
"""

from tradingagents.agents.utils.agent_utils import build_instrument_context


def create_macro_context_node(llm):
    def macro_context_node(state) -> dict:
        instrument_context = build_instrument_context(
            state["company_of_interest"], state.get("company_name", "")
        )

        news_report = state.get("news_report", "")
        fundamentals_report = state.get("fundamentals_report", "")
        sentiment_report = state.get("sentiment_report", "")
        market_report = state.get("market_report", "")
        trade_date = state.get("trade_date", "")

        prompt = f"""【语言要求】你必须使用中文撰写以下分析。利率/政策/汇率等术语可保留英文缩写（如 PMI、CPI、HIBOR）。

你是投研团队的**宏观策略师（Macro Strategist）**。在 Bull/Bear 辩论之前，识别当前宏观环境对该标的的影响方向。

**重要**：你的判断会影响下游 stock_profile_node 对主题溢价（THEMATIC_PREMIUM）的判断 + RM Step 1 的行业景气度判断。**精准、可证伪、有依据**。

{instrument_context}

当前日期：{trade_date}

---

## 一、利率周期阶段（必输出）

从 news / fundamentals 报告中识别美联储/央行近期政策动作 + 利率指标信号，判断当前所处阶段：

| 阶段 | 触发信号 | 对成长股的影响 |
|------|---------|---------------|
| **宽松周期（dovish）** | 降息 / 降准 / 利率指标下行 / 央行明确转鸽 | ✅ 利好高估值成长股（贴现率下降）|
| **紧缩周期（hawkish）** | 加息 / 缩表 / 利率指标上行 / 央行强硬鹰派 | ❌ 利空高估值成长股（贴现率上升）|
| **拐点期（pivot）** | 政策方向不明 / 数据矛盾 / 市场对方向有大分歧 | ⚠️ 波动加大，主题溢价不稳定 |
| **平稳期（neutral）** | 政策已稳定一段时间 | 中性 |

**输出**：阶段 + 依据（引用 news 报告中的具体央行公告 / 利率数据 / 政策细则）

---

## 二、流动性松紧度（必输出，A 股关键）

A 股是**资金驱动市场**，流动性松紧直接决定估值上限。从 news + fundamentals + market 报告提取以下信号：

| 信号 | 含义 |
|------|------|
| **M2 / 社融增速** | M2 增速 < 名义 GDP → 流动性紧缩；M2 > 名义 GDP + 4pp → 流动性充裕 |
| **北向资金近期方向** | 连续 5+ 日净流入 / 净流出 |
| **融资余额变化** | 上升 → 杠杆资金活跃；下降 → 杠杆退场 |
| **新发基金规模** | 公募基金发行规模 |
| **板块成交活跃度** | 主题板块换手率 / 涨停板数量 |

**输出**：流动性松紧度（充裕 / 中性 / 紧缩 / 极端紧缩）+ 主要依据（≥2 个具体信号）

---

## 三、主要宏观风险事件（未来 30 天，必输出）

从 news 报告中识别未来 30 天内可能影响市场的宏观事件：

| 事件类型 | 例子 |
|---------|------|
| 央行议息 | 美联储 FOMC / 中国央行 MLF/LPR |
| 经济数据 | CPI / PMI / 非农 / 工业增加值 |
| 政策窗口 | 中央经济工作会议 / 两会 / 三中全会 / 政治局会议 |
| 地缘政治 | 美国大选 / 中美贸易谈判 / 关税调整 / 地区冲突 |
| 行业政策 | 行业监管细则 / 产业目录调整 / 财政补贴 |

**输出**：列出未来 30 天**已知**的关键事件（事件名 + 预计日期 + 对市场的预期方向）。无明确事件填"无重大已知事件"。

---

## 四、该标的所属 industry 相对宏观环境的方向（必输出）

基于一、二、三和 fundamentals 提到的标的 industry，判断当前宏观环境对该 industry 的方向：

| 方向 | 含义 |
|------|------|
| **强顺风** | 宏观环境对该 industry 多重利好叠加 |
| **顺风** | 宏观环境有利但程度温和 |
| **中性** | 宏观环境无明显方向 |
| **逆风** | 宏观环境对该 industry 不利 |
| **强逆风** | 宏观环境多重利空叠加 |

**输出**：方向 + 一句话理由（如"AI 算力受益于全球宽松周期 + 国内政策支持，处于强顺风"）

---

## 五、对下游 agent 的关键提示

**给 stock_profile_node 的提示**：
- 当前 macro_environment 是 [紧缩/宽松/中性/逆风/顺风]
- 主题溢价容忍度应该如何调整？
  - 流动性紧缩 + 利率上行 → 主题溢价容忍度应**收紧**（acceleration 阶段也最多给 +30% 而非 +50%）
  - 流动性宽松 + 利率下行 → 主题溢价容忍度应**放宽**（acceleration 阶段可给 +50% 甚至 +60%）

**给 RM 的提示**：
- 当前宏观环境是否支持高估值容忍？
- 行业景气度判断是否需要叠加宏观顺/逆风修正？

---

## 最终输出要求

- 用中文撰写，结构严格按上述五部分
- 每条判断必须有具体数据支撑（引用 news/fundamentals 报告中的原文，禁止凭印象）
- 末尾输出一段 YAML 摘要供下游程序化解析：

```yaml
MACRO_CONTEXT:
  rate_cycle: dovish / hawkish / pivot / neutral
  rate_cycle_evidence: <一句话依据>

  liquidity: 充裕 / 中性 / 紧缩 / 极端紧缩
  liquidity_signals:
    - <信号1>
    - <信号2>

  upcoming_events:
    - event: <事件名>
      date: <YYYY-MM-DD>
      expected_impact: 正面 / 负面 / 中性 / 不确定
    # 无事件填空列表 []

  industry_macro_direction: 强顺风 / 顺风 / 中性 / 逆风 / 强逆风
  industry_macro_rationale: <一句话>

  premium_adjustment_advice: 放宽 / 维持 / 收紧
  premium_adjustment_pct: <整数百分比，相对基础阈值的调整，如 +10、0、-15>
```

## 输入资料

[置信度:中] Latest news report:
{news_report}

[置信度:高] Company fundamentals report:
{fundamentals_report}

[置信度:中高] Market research report:
{market_report}

[置信度:中低] Social media sentiment report:
{sentiment_report}
"""

        response = llm.invoke(prompt)
        return {"macro_context": response.content}

    return macro_context_node
