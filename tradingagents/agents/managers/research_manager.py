
from tradingagents.agents.utils.agent_utils import build_instrument_context


def create_research_manager(llm, memory):
    def research_manager_node(state) -> dict:
        instrument_context = build_instrument_context(state["company_of_interest"], state.get("company_name", ""))
        history = state["investment_debate_state"].get("history", "")
        market_research_report = state["market_report"]
        sentiment_report = state["sentiment_report"]
        news_report = state["news_report"]
        fundamentals_report = state["fundamentals_report"]

        investment_debate_state = state["investment_debate_state"]

        curr_situation = f"{market_research_report}\n\n{sentiment_report}\n\n{news_report}\n\n{fundamentals_report}"
        past_memories = memory.get_memories(curr_situation, n_matches=2)

        past_memory_str = ""
        for i, rec in enumerate(past_memories, 1):
            past_memory_str += rec["recommendation"] + "\n\n"

        prompt = f"""【语言要求】你必须使用中文撰写以下所有分析内容和回复。评级关键词（Buy/Sell/Hold）和股票代码可保留英文。

你是投资研究总监，负责综合多空双方的辩论，做出最终投资决策。

{instrument_context}

---

## 决策流程（必须严格按顺序完成）

### 第一步：量化评分
分别为多头和空头论据的说服力打分（1-10 分）：
- **多头得分 (Bull Score)**：____/10
- **空头得分 (Bear Score)**：____/10

评分标准：
- 9-10 分：论据有确凿的数据/事件支撑，逻辑严密，短期内大概率兑现
- 7-8 分：论据较有说服力，有明确的催化剂或数据支持
- 5-6 分：论据有一定道理，但缺乏关键数据或催化剂
- 3-4 分：论据较弱，存在明显逻辑漏洞
- 1-2 分：论据站不住脚

### 第二步：根据得分差决定方向
- 多头得分 > 空头得分 → **Buy**
- 空头得分 > 多头得分 → **Sell**
- 双方得分完全相同（差值 = 0）**且**近期无任何明确催化剂 → **Hold**

**重要约束**：Hold 仅在多空得分完全持平、且确实没有可操作信号时才允许使用。如果多空任何一方哪怕只高出 1 分，都必须选择对应方向（Buy 或 Sell），不要回避决策。

### 第三步：制定投资计划
- **你的决策**：Buy / Sell / Hold，以及得分依据
- **核心理由**：引用辩论中最有说服力的 2-3 个论据
- **行动方案**：具体的建仓/减仓策略、仓位比例建议、止损止盈位
- **风险提示**：当前决策最大的反面风险

---

## 历史教训
以下是过往类似场景的反思记录，请吸取教训避免重蹈覆辙：
\"{past_memory_str}\"

## 辩论记录
{history}

---

**重要：请用中文撰写你的投资决策和分析报告。** 评级关键词（Buy/Sell/Hold）和股票代码请保留英文原文。请以中文口语化方式阐述你的决策理由和详细投资计划。"""
        response = llm.invoke(prompt)

        new_investment_debate_state = {
            "judge_decision": response.content,
            "history": investment_debate_state.get("history", ""),
            "bear_history": investment_debate_state.get("bear_history", ""),
            "bull_history": investment_debate_state.get("bull_history", ""),
            "current_response": response.content,
            "count": investment_debate_state["count"],
        }

        return {
            "investment_debate_state": new_investment_debate_state,
            "investment_plan": response.content,
        }

    return research_manager_node
