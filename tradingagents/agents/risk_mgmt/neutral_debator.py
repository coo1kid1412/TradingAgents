import json

from tradingagents.agents.utils.agent_utils import RISK_DEBATE_PHRASING_RULES
from tradingagents.agents.utils.risk_context import build_risk_data_packet
from tradingagents.agents.risk_mgmt.risk_response import invoke_risk_response
from tradingagents.agents.utils.handoff import decision_handoff_contract, pack_agent_context


def create_neutral_debator(llm):
    def neutral_node(state) -> dict:
        risk_debate_state = state["risk_debate_state"]
        history = risk_debate_state.get("history", "")
        neutral_history = risk_debate_state.get("neutral_history", "")

        current_aggressive_response = risk_debate_state.get("current_aggressive_response", "")
        current_conservative_response = risk_debate_state.get("current_conservative_response", "")

        # trader_decision 已废弃（optimization 05），改为只引用 RM 方案
        # trader_decision = state["trader_investment_plan"]  # DEPRECATED in 05
        investment_plan = state.get("investment_plan", "")
        risk_data_packet = build_risk_data_packet(state)
        bounded_input = pack_agent_context(
            [
                {"label": "RM 研究包", "content": investment_plan, "priority": "hard_constraint"},
                {"label": "确定性风险数据包", "content": risk_data_packet, "priority": "hard_constraint"},
                {"label": "流动性风控最新观点", "content": current_aggressive_response, "priority": "decision"},
                {"label": "事件风控最新观点", "content": current_conservative_response, "priority": "decision"},
                {"label": "有限历史", "content": history, "priority": "narrative"},
            ],
            budget_chars=18_000,
        )

        prompt = f"""【语言要求】你必须使用中文进行以下所有风险辩论和分析。股票代码和技术指标名称可保留英文。

你是**尾部风险与压力测试分析师**。你的职责是评估投资方案在极端情景下的风险敞口，而非重新判断投资方向。你需要回答："最坏的情况下会亏多少？"

你的专项审查维度：
1. **压力测试（3 情景）**：
   - 乐观情景：催化兑现，预期收益
   - 基准情景：按当前趋势延续，预期收益/损失
   - 悲观情景：核心假设证伪（如业绩显著低于预期、政策转向、流动性枯竭），潜在最大下行幅度估算
2. **尾部风险识别**：是否存在"低概率但毁灭性"的风险？例如：集中度风险、关联方风险、财务造假风险、退市风险
3. **相关系风险**：该标的与大盘/行业的相关性如何？极端行情下相关性是否可能趋向 1（系统性风险）？

**关键约束**：
- 你不是在做方向判断——那是投研团队的职责
- 你关注的是"如果一切都错了，损失有多大"
- 尾部风险往往来自投研团队未充分考虑的维度，你需要主动思考"大家忽略了什么"
- 不要为 RM 方案辩护——你的职责是找极端风险，不是唱赞歌
- 压力情景价格/跌幅必须锚定 RM 情景、技术支撑或历史波动；禁止自行发明精确概率和“最大亏损”
- 没有组合 AUM 时，只分析单票价格风险，不得换算组合金额损失

**有界决策上下文（压力锚只能引用这里已有的数值）：**
{bounded_input}

**辩论要求**：
如果还没有其他分析师的回应，请基于可用数据提出你自己的尾部风险分析。

在辩论中，每轮发言必须包含：
- 压力测试情景（明确乐观/基准/悲观的假设和结果）
- 尾部风险点（具体）
- 建议的仓位/止损调整方案

积极回应其他分析师的观点，特别是当他们的流动性或事件风险分析暴露了新的尾部风险路径时，从极端情景角度补充你的评估。特别关注 RM 方案中"风控审查指引"提到的未决问题，这些往往是尾部风险的入口。以中文口语化方式进行辩论。

正文控制在 800-1200 个中文字符内，优先保证末尾 RISK_VIEW 完整；不要输出冗长清单。

{RISK_DEBATE_PHRASING_RULES}

发言末尾必须输出：
```yaml
RISK_VIEW:
  role: tail
  severity: high / medium / low / unknown
  cap_pct: <0-100 或 null>
  cap_basis: <引用的 RM 悲观情景/技术支撑/历史波动；无则 null>
  evidence_ids: [<从风险数据包 reference_ids 逐字选择；也可用 RM-PLAN>]
  data_supported: true / false
```
未引用可追溯压力锚时 `data_supported=false` 且 `cap_pct=null`。禁止自造 evidence_ids。

**重要：请用中文进行风险辩论。** 股票代码和技术指标名称请保留英文原文。"""
        prompt += decision_handoff_contract(
            "neutral_risk", "平衡情景并评估尾部损失路径与组合脆弱性",
        )

        response = invoke_risk_response(llm, prompt, role="tail")

        argument = f"Neutral Analyst: {response.content}"

        bounded_history = pack_agent_context(
            [{"label": "上一轮", "content": current_aggressive_response or current_conservative_response, "priority": "decision"},
             {"label": "当前轮", "content": argument, "priority": "decision"}],
            budget_chars=10_000,
        )
        new_risk_debate_state = {
            "history": bounded_history,
            "aggressive_history": risk_debate_state.get("aggressive_history", ""),
            "conservative_history": risk_debate_state.get("conservative_history", ""),
            "neutral_history": argument,
            "latest_speaker": "Neutral",
            "current_aggressive_response": risk_debate_state.get(
                "current_aggressive_response", ""
            ),
            "current_conservative_response": risk_debate_state.get("current_conservative_response", ""),
            "current_neutral_response": argument,
            "count": risk_debate_state["count"] + 1,
        }

        return {"risk_debate_state": new_risk_debate_state}

    return neutral_node
