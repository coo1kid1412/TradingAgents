import json

from tradingagents.agents.utils.agent_utils import RISK_DEBATE_PHRASING_RULES
from tradingagents.agents.utils.risk_context import build_risk_data_packet
from tradingagents.agents.risk_mgmt.risk_response import invoke_risk_response


def create_aggressive_debator(llm):
    def aggressive_node(state) -> dict:
        risk_debate_state = state["risk_debate_state"]
        history = risk_debate_state.get("history", "")
        aggressive_history = risk_debate_state.get("aggressive_history", "")

        current_conservative_response = risk_debate_state.get("current_conservative_response", "")
        current_neutral_response = risk_debate_state.get("current_neutral_response", "")

        # trader_decision 已废弃（optimization 05），改为只引用 RM 方案
        # trader_decision = state["trader_investment_plan"]  # DEPRECATED in 05
        investment_plan = state.get("investment_plan", "")
        risk_data_packet = build_risk_data_packet(state)

        prompt = f"""【语言要求】你必须使用中文进行以下所有风险辩论和分析。股票代码和技术指标名称可保留英文。

你是**流动性风险与执行分析师**。你的职责是从执行层面评估投资方案的风险，而非重新判断投资方向。你需要回答："这个方案在执行过程中会不会出问题？"

你的专项审查维度：
1. **流动性风险**：日均成交额能否支撑建议仓位？大额交易冲击成本如何？分批建仓是否现实？是否存在流动性枯竭的时段（如开盘/收盘/财报日）？
2. **止损可执行性**：硬止损位在极端行情下能否实际执行？涨跌停板是否阻碍止损？ATR 波动率是否超出正常范围？
3. **市场微结构风险**：该标的是否存在停牌风险？融资融券限制？做市商/流动性提供者是否充足？

**关键约束**：
- 你不是在做方向判断——那是投研团队的职责
- 你关注的是"即便方向正确，执行过程中会不会翻车"
- 不要为 RM 方案辩护——你的职责是找执行漏洞，不是唱赞歌
- 若没有组合 AUM、计划成交金额、日均成交额或 ATR，禁止假设账户规模、成交占比、冲击成本百分比；只能列为待补数据
- 只有存在报告内可追溯的成交额/ATR/涨跌停数据时，才允许给出仓位上限

**Research Manager 的投资方案（含评级、评分、价位区间、执行可行性、风控审查指引）：**
{investment_plan}

**确定性风险数据包（只引用这里已有的数值，禁止补造缺失字段）：**
```json
{json.dumps(risk_data_packet, ensure_ascii=False, indent=2, default=str)}
```

**辩论要求**：
以下是当前的对话历史：{history}
以下是事件风控分析师的最新论点：{current_conservative_response}
以下是尾部风控分析师的最新论点：{current_neutral_response}
如果还没有其他分析师的回应，请基于可用数据提出你自己的流动性风险分析。

在辩论中，每轮发言必须包含：
- 识别到的执行风险（具体）
- 风险严重程度（高/中/低）
- 建议的风险缓释措施

积极回应其他分析师的观点，特别是当他们的事件风险或尾部风险分析影响了执行可行性时，从流动性角度给出你的评估。特别关注 RM 方案中"风控审查指引"提到的未决问题是否涉及执行层面风险。以中文口语化方式进行辩论。

正文控制在 800-1200 个中文字符内，优先保证末尾 RISK_VIEW 完整；不要输出冗长清单。

{RISK_DEBATE_PHRASING_RULES}

发言末尾必须输出：
```yaml
RISK_VIEW:
  role: liquidity
  severity: high / medium / low / unknown
  cap_pct: <0-100 或 null>
  cap_basis: <引用的成交额/ATR/执行数据；无则 null>
  evidence_ids: [<从风险数据包 reference_ids 逐字选择；也可用 RM-PLAN>]
  data_supported: true / false
```
`data_supported=false` 时 `cap_pct` 必须为 null。禁止自造 evidence_ids。

**重要：请用中文进行风险辩论。** 股票代码和技术指标名称请保留英文原文。"""

        response = invoke_risk_response(llm, prompt, role="liquidity")

        argument = f"Aggressive Analyst: {response.content}"

        new_risk_debate_state = {
            "history": history + "\n" + argument,
            "aggressive_history": aggressive_history + "\n" + argument,
            "conservative_history": risk_debate_state.get("conservative_history", ""),
            "neutral_history": risk_debate_state.get("neutral_history", ""),
            "latest_speaker": "Aggressive",
            "current_aggressive_response": argument,
            "current_conservative_response": risk_debate_state.get("current_conservative_response", ""),
            "current_neutral_response": risk_debate_state.get(
                "current_neutral_response", ""
            ),
            "count": risk_debate_state["count"] + 1,
        }

        return {"risk_debate_state": new_risk_debate_state}

    return aggressive_node
