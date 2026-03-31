from tradingagents.agents.utils.agent_utils import build_instrument_context


def create_portfolio_manager(llm, memory):
    def portfolio_manager_node(state) -> dict:

        instrument_context = build_instrument_context(state["company_of_interest"], state.get("company_name", ""))

        history = state["risk_debate_state"]["history"]
        risk_debate_state = state["risk_debate_state"]
        market_research_report = state["market_report"]
        news_report = state["news_report"]
        fundamentals_report = state["fundamentals_report"]
        sentiment_report = state["sentiment_report"]
        trader_plan = state["investment_plan"]

        curr_situation = f"{market_research_report}\n\n{sentiment_report}\n\n{news_report}\n\n{fundamentals_report}"
        past_memories = memory.get_memories(curr_situation, n_matches=2)

        past_memory_str = ""
        for i, rec in enumerate(past_memories, 1):
            past_memory_str += rec["recommendation"] + "\n\n"

        prompt = f"""【语言要求】你必须使用中文撰写以下所有分析内容和回复。评级关键词（Buy/Overweight/Hold/Underweight/Sell）和股票代码可保留英文。

你是投资组合经理，负责综合风险分析团队的辩论，做出最终交易决策。

{instrument_context}

---

## 决策流程（必须严格按顺序完成）

### 第一步：方向判断
根据风险团队辩论内容，先判断当前标的的整体方向：
- **偏多 (Bullish)** — 上行风险回报比优于下行
- **偏空 (Bearish)** — 下行风险大于上行空间
- **中性 (Neutral)** — 多空完全均衡，无明确信号

### 第二步：确定评级强度
根据方向判断选择评级：

| 方向 | 信号强 | 信号中等 |
|------|--------|---------|
| 偏多 | **Buy** — 强烈看多，建仓或加仓 | **Overweight** — 温和看多，逐步增加敞口 |
| 偏空 | **Sell** — 强烈看空，离场或回避 | **Underweight** — 温和看空，减仓或部分止盈 |
| 中性 | — | **Hold** — 维持现有仓位 |

**重要约束**：
- Hold 仅在方向判断为"中性"时允许使用
- 如果方向偏多或偏空，**必须**选择对应方向的评级（Buy/Overweight 或 Sell/Underweight），不得选 Hold
- 如果选择 Hold，必须明确说明为什么当前不存在任何可操作的方向性信号

### 第三步：输出报告
1. **方向判断**：偏多 / 偏空 / 中性，附简要理由
2. **评级**：Buy / Overweight / Hold / Underweight / Sell
3. **执行摘要**：入场策略、仓位比例、关键风险位、时间周期
4. **投资论点**：基于辩论内容和历史教训的详细推理

---

**交易员的初步方案：** {trader_plan}

**历史教训：** {past_memory_str}

**风险团队辩论记录：**
{history}

---

**重要：请用中文撰写你的最终交易决策报告。** 评级关键词（Buy/Overweight/Hold/Underweight/Sell）和股票代码请保留英文原文。请以中文阐述你的投资论点和执行摘要，使用专业的投资组合管理术语。"""

        response = llm.invoke(prompt)

        new_risk_debate_state = {
            "judge_decision": response.content,
            "history": risk_debate_state["history"],
            "aggressive_history": risk_debate_state["aggressive_history"],
            "conservative_history": risk_debate_state["conservative_history"],
            "neutral_history": risk_debate_state["neutral_history"],
            "latest_speaker": "Judge",
            "current_aggressive_response": risk_debate_state["current_aggressive_response"],
            "current_conservative_response": risk_debate_state["current_conservative_response"],
            "current_neutral_response": risk_debate_state["current_neutral_response"],
            "count": risk_debate_state["count"],
        }

        return {
            "risk_debate_state": new_risk_debate_state,
            "final_trade_decision": response.content,
        }

    return portfolio_manager_node
