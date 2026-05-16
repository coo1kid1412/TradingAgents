"""共识识别节点（Consensus Node）

在 4 个 analyst 完成后、Bull/Bear 辩论之前运行。
综合各分析师报告，提炼"当前股价隐含的市场预期"，写入 state["consensus_snapshot"]。

下游 Bull/Bear/RM 据此识别共识 vs 反共识论据：
- 共识论据：已被 price-in，alpha 价值低
- 反共识论据：共识忽视的点，alpha 价值高
"""

from tradingagents.agents.utils.agent_utils import build_instrument_context


def create_consensus_node(llm):
    def consensus_node(state) -> dict:
        instrument_context = build_instrument_context(
            state["company_of_interest"], state.get("company_name", "")
        )

        market_report = state.get("market_report", "")
        sentiment_report = state.get("sentiment_report", "")
        news_report = state.get("news_report", "")
        fundamentals_report = state.get("fundamentals_report", "")
        stock_profile = state.get("stock_profile", "")

        prompt = f"""【语言要求】你必须使用中文撰写以下分析。股票代码和技术指标名称可保留英文。

你是投研团队的**共识识别官**。你的唯一任务：基于 4 份分析师报告，提炼"当前市场对该标的的一致预期"，作为下游 Bull/Bear 辩论的校准基准。

{instrument_context}

---

## 股票画像（由画像识别官在你之前提炼，影响你的报告权重）

{stock_profile if stock_profile else "（画像缺失，按 4 份报告等权处理）"}

**重要**：画像里给了 4 份报告的推荐权重。**权重越高的报告，你在识别共识时越要重点引用其口径**。例如：
- 题材炒作小盘股：舆情权重高 → 共识 narrative 应主要来自舆情/新闻口径
- 大盘蓝筹：基本面权重高 → 共识应主要来自卖方一致预期 + 基本面数据

---

## 你的核心职责

识别"当前股价里已经 price-in 了什么"。下游 Bull/Bear 会根据你的输出判断：
- 哪些论据是**共识**（已被定价，alpha 价值低）
- 哪些论据是**反共识**（共识忽视的，alpha 价值高）

**关键原则**：
- 不做方向判断，不给评级，不预测
- 只提炼"市场当下相信什么"
- 用证据说话：卖方研报口径、舆情多空比、新闻 narrative、机构持仓变化

---

## 输出结构（必须严格按以下格式）

### 一、共识方向
[偏多 / 偏空 / 分歧明显 / 中性]，并用一句话说明判断依据。

### 二、核心 narrative（一段话）
用 2-3 句话描述"市场当前讲的故事"。例如：
> "市场已 price-in 该公司 2026 年净利润同比 +50%、AI 算力互联龙头地位、CXL 量产 Q4 落地，给到动态 PE 80-90 倍。"

### 三、已被定价的关键预期（列表）
列出 3-6 条"当前股价隐含的具体预期"，每条必须可证伪。格式：
- [预期内容] — 来源：[报告/数据]
- 例如：2026 年净利润同比 ≥ +50% — 来源：fundamentals SUMMARY + 卖方研报口径
- 例如：高盛目标价 363 元（隐含 +45% 上行）— 来源：news 报告
- 例如：三星 CXL 量产 Q4 如期落地 — 来源：sentiment + news

### 四、共识来源
- 卖方研报口径：__（评级数 + 平均目标价）
- 舆情口径：__（多空比 + 关键 KOL 倾向）
- 新闻 narrative：__（最近 N 天主流报道方向）
- 机构持仓变化：__（增持/减持）

### 五、关键不确定性（共识没有回答的问题）
列出 2-4 个"共识叙事**没回答**的问题"——这些是下游 Bull/Bear 寻找反共识论据的金矿。每条格式：
- [问题描述]——共识假设：__；实际数据/隐患：__
- 例如：扣非净利润增速能否跟上归母？共识假设：能跟上；实际数据：扣非仅 +20%，归母 +61%，差 41 个百分点
- 例如：融资余额 143 亿是否构成踩踏风险？共识假设：占市值 4.8% 不构成系统风险；隐患：绝对金额科创板前列

### 六、共识强度评估
- **共识强度**：[强 / 中 / 弱]
  - 强：卖方/舆情/新闻三方一致，且方向明确
  - 中：两方一致，一方分歧
  - 弱：三方分歧，市场对方向尚未形成共识
- **拥挤度警示**：若共识方向为偏多且共识强度为"强"，标注 **⚠️ 拥挤多头**；反之 **⚠️ 拥挤空头**。拥挤交易往往伴随反向风险。

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
- 每条预期/不确定性必须可观察可证伪，禁止泛化表述（"基本面良好"、"行业前景广阔"不可接受）
- 末尾输出一段 YAML 摘要供下游程序化解析：

```yaml
CONSENSUS_SUMMARY:
  direction: 偏多/偏空/分歧/中性
  strength: 强/中/弱
  crowded: yes/no
  priced_in:
    - <预期1>
    - <预期2>
  uncertainties:
    - <问题1>
    - <问题2>
```
"""

        response = llm.invoke(prompt)
        return {"consensus_snapshot": response.content}

    return consensus_node
