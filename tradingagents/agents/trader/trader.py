import functools

from tradingagents.agents.utils.agent_utils import build_instrument_context


def create_trader(llm, memory):
    def trader_node(state, name):
        company_name = state["company_of_interest"]
        instrument_context = build_instrument_context(company_name, state.get("company_name", ""))
        investment_plan = state["investment_plan"]
        market_research_report = state["market_report"]
        sentiment_report = state["sentiment_report"]
        news_report = state["news_report"]
        fundamentals_report = state["fundamentals_report"]

        curr_situation = f"{market_research_report}\n\n{sentiment_report}\n\n{news_report}\n\n{fundamentals_report}"
        past_memories = memory.get_memories(curr_situation, n_matches=2)

        past_memory_str = ""
        if past_memories:
            for i, rec in enumerate(past_memories, 1):
                past_memory_str += rec["recommendation"] + "\n\n"
        else:
            past_memory_str = "No past memories found."

        context = {
            "role": "user",
            "content": f"Based on a comprehensive analysis by a team of analysts, here is an investment plan tailored for {company_name}. {instrument_context} This plan incorporates insights from current technical market trends, macroeconomic indicators, and social media sentiment. Use this plan as a foundation for evaluating your next trading decision.\n\nProposed Investment Plan: {investment_plan}\n\nLeverage these insights to make an informed and strategic decision.",
        }

        messages = [
            {
                "role": "system",
                "content": f"""【语言要求】你必须使用中文撰写以下所有交易分析和建议。股票代码和技术指标名称可保留英文。'FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL**' 结尾格式必须保留英文原文。

你是一名果断的交易员，根据研究团队的分析做出最终交易决策。

## 决策流程（必须按顺序完成）

### 第一步：判断市场方向
根据研究团队的投资计划，先明确判断当前标的的方向：
- **看多 (Bullish)**：基本面/技术面/情绪面整体偏正面
- **看空 (Bearish)**：基本面/技术面/情绪面整体偏负面

你必须先给出方向判断，不允许跳过此步骤。

### 第二步：确定操作建议
- 看多 → **BUY**（建仓或加仓）
- 看空 → **SELL**（离场或减仓）
- **HOLD 仅在以下极端情况允许**：多空信号完全矛盾且强度相当，短期内没有任何可操作的催化剂。选择 HOLD 时必须明确说明为什么既不能 BUY 也不能 SELL。

### 第三步：交易计划
- 具体建仓/减仓比例
- 入场/出场价位
- 止损止盈点位
- 时间周期

结论必须以 'FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL**' 结尾。

## 历史教训
{past_memory_str}

**重要：请用中文撰写你的交易分析和建议。** 但 'FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL**' 这个结尾格式必须保留英文原文，这是系统解析所必需的。股票代码和技术指标名称也请保留英文。""",
            },
            context,
        ]

        result = llm.invoke(messages)

        return {
            "messages": [result],
            "trader_investment_plan": result.content,
            "sender": name,
        }

    return functools.partial(trader_node, name="Trader")
