import json

from tradingagents.agents.utils.agent_utils import RISK_DEBATE_PHRASING_RULES
from tradingagents.agents.utils.risk_context import build_risk_data_packet
from tradingagents.agents.risk_mgmt.risk_response import invoke_risk_response
from tradingagents.agents.utils.handoff import decision_handoff_contract, pack_agent_context


def create_conservative_debator(llm):
    def conservative_node(state) -> dict:
        risk_debate_state = state["risk_debate_state"]
        history = risk_debate_state.get("history", "")
        conservative_history = risk_debate_state.get("conservative_history", "")

        current_aggressive_response = risk_debate_state.get("current_aggressive_response", "")
        current_neutral_response = risk_debate_state.get("current_neutral_response", "")

        # trader_decision 已废弃（optimization 05），改为只引用 RM 方案
        # trader_decision = state["trader_investment_plan"]  # DEPRECATED in 05
        investment_plan = state.get("investment_plan", "")
        risk_data_packet = build_risk_data_packet(state)
        bounded_input = pack_agent_context(
            [
                {"label": "RM 研究包", "content": investment_plan, "priority": "hard_constraint"},
                {"label": "确定性风险数据包", "content": risk_data_packet, "priority": "hard_constraint"},
                {"label": "流动性风控最新观点", "content": current_aggressive_response, "priority": "decision"},
                {"label": "尾部风控最新观点", "content": current_neutral_response, "priority": "decision"},
                {"label": "有限历史", "content": history, "priority": "narrative"},
            ],
            budget_chars=18_000,
        )

        prompt = f"""【语言要求】你必须使用中文进行以下所有风险辩论和分析。股票代码和技术指标名称可保留英文。

你是**事件风险与时机分析师**。你的职责是识别可能打乱投资执行方案的事件风险，而非重新判断投资方向。你需要回答："有什么事件可能在执行期间引爆？"

你的专项审查维度：
1. **公司层面事件**：近期是否有财报披露？业绩预告？重大公告？管理层变动？股权激励到期？大宗交易？
2. **行业/监管事件**：是否有行业政策变动？监管审查？反垄断调查？关税/贸易政策变化？
3. **宏观事件窗口**：央行议息？非农数据？地缘政治风险？这些宏观事件对标的的影响路径是什么？

**关键约束**：
- 你不是在做方向判断——那是投研团队的职责
- 你关注的是"有哪些时间炸弹可能在持仓期间爆炸"
- 不要为 RM 方案辩护——你的职责是找事件地雷，不是唱赞歌
- 事件日期必须来自官方公告/交易所预约或 IC 决策包中的合格证据；传闻不能写成确定日期
- 没有统计基率或可追溯来源时，禁止给精确概率；使用高/中/低并注明依据

**有界决策上下文（事件日期只能引用合格证据，禁止把传闻写成事实）：**
{bounded_input}

**辩论要求**：
如果还没有其他分析师的回应，请基于可用数据提出你自己的事件风险分析。

在辩论中，每轮发言必须包含：
- 识别到的事件风险（具体事件+预计时间窗口）
- 事件发生概率（高/中/低）
- 若事件发生，对 Trader 方案的冲击评估

积极回应其他分析师的观点，特别是当他们的流动性或尾部风险分析揭示了新的事件触发条件时，从事件时机角度补充你的评估。特别关注 RM 方案中"风控审查指引"提到的未决问题是否涉及事件风险。以中文口语化方式进行辩论。

正文控制在 800-1200 个中文字符内，优先保证末尾 RISK_VIEW 完整；不要输出冗长清单。

{RISK_DEBATE_PHRASING_RULES}

发言末尾必须输出：
```yaml
RISK_VIEW:
  role: event
  severity: high / medium / low / unknown
  cap_pct: <0-100 或 null>
  cap_basis: <引用的官方事件窗口及冲击依据；无则 null>
  evidence_ids: [<从风险数据包 reference_ids 逐字选择；也可用 RM-PLAN>]
  data_supported: true / false
```
只有事件窗口与冲击依据均可追溯时才可令 `data_supported=true`。禁止自造 evidence_ids。

**重要：请用中文进行风险辩论。** 股票代码和技术指标名称请保留英文原文。"""
        prompt += decision_handoff_contract(
            "conservative_risk", "识别公司、行业、监管与宏观事件窗口风险",
        )

        response = invoke_risk_response(llm, prompt, role="event")

        argument = f"Conservative Analyst: {response.content}"

        bounded_history = pack_agent_context(
            [{"label": "上一轮", "content": current_aggressive_response or current_neutral_response, "priority": "decision"},
             {"label": "当前轮", "content": argument, "priority": "decision"}],
            budget_chars=10_000,
        )
        new_risk_debate_state = {
            "history": bounded_history,
            "aggressive_history": risk_debate_state.get("aggressive_history", ""),
            "conservative_history": argument,
            "neutral_history": risk_debate_state.get("neutral_history", ""),
            "latest_speaker": "Conservative",
            "current_aggressive_response": risk_debate_state.get(
                "current_aggressive_response", ""
            ),
            "current_conservative_response": argument,
            "current_neutral_response": risk_debate_state.get(
                "current_neutral_response", ""
            ),
            "count": risk_debate_state["count"] + 1,
        }

        return {"risk_debate_state": new_risk_debate_state}

    return conservative_node
