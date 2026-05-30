"""DEPRECATED in optimization 05: Trader node removed from graph.

This file is kept for git history and potential future reuse.
The node's responsibilities have been redistributed:
- Direction & stop-loss → Research Manager
- Liquidity check → Liquidity Analyst (formerly 'aggressive_debator')
"""
import functools

from tradingagents.agents.utils.agent_utils import build_instrument_context, RISK_DEBATE_PHRASING_RULES


def create_trader(llm, memory):
    def trader_node(state, name):
        company_name = state["company_of_interest"]
        instrument_context = build_instrument_context(company_name, state.get("company_name", ""))
        investment_plan = state["investment_plan"]

        # 仅基于 RM 方案做 memory 检索（不再拼接4份原始报告）
        past_memories = memory.get_memories(investment_plan, n_matches=3)

        past_memory_str = ""
        for i, rec in enumerate(past_memories, 1):
            past_memory_str += f"【适用场景】{rec['matched_situation']}\n【经验教训】{rec['recommendation']}\n\n"
        if not past_memory_str:
            past_memory_str = "暂无相关历史教训。"

        context = {
            "role": "user",
            "content": f"以下是 Research Manager 为 {company_name} 制定的投资方案。{instrument_context} 请基于此方案评估并细化执行策略。\n\nResearch Manager 投资方案：{investment_plan}",
        }

        messages = [
            {
                "role": "system",
                "content": f"""【语言要求】你必须使用中文撰写以下所有交易分析和建议。股票代码和技术指标名称可保留英文。'FINAL TRANSACTION PROPOSAL: **BUY/OVERWEIGHT/HOLD/UNDERWEIGHT/SELL**' 结尾格式必须保留英文原文。

你是一名交易执行专家。Research Manager 已经给出了方向判断和评级（见 investment_plan）。你不重新判断方向，你的工作是**评估并细化执行方案**。

## 输出结构（必须按以下章节撰写）

### 一、方向确认
一句话复述 Research Manager 的方向和评级，明确表示"采纳"。例如："采纳 Research Manager 的 OVERWEIGHT 评级，方向偏多。"

### 二、入场策略
- 一次性建仓 vs 分批建仓（若分批，给出每批的触发条件——价格信号、技术信号、时间信号）
- 单笔上限、总仓位上限

### 三、止损位合理性评估
- Research Manager 给的止损位是否合理？参考 ATR、最近支撑位、波动率
- 给出最终建议的硬止损与时间止损

### 四、流动性与时间窗口
- 该标的近期日均成交额，建议仓位是否会造成滑点
- 是否临近财报披露窗口、是否避开

### 五、执行风险
- 即使 Research Manager 的方向正确，执行层可能出问题的点（流动性枯竭、停牌、涨跌停板限制等）

### 六、结尾
输出 `FINAL TRANSACTION PROPOSAL: **<RM的原始评级>**`，评级必须与 Research Manager 一致（5档）。
在 investment_plan 中查找 Research Manager 给出的评级关键词，直接传递其评级：
- 买入类：BUY / OVERWEIGHT
- 持有类：HOLD
- 卖出类：UNDERWEIGHT / SELL

例如：FINAL TRANSACTION PROPOSAL: **OVERWEIGHT**

## 重要约束
- **禁止重新判断方向**：不得出现"我认为应看多/看空"或自行给出与 Research Manager 不同的评级
- **禁止二次打分**：不得重新对多空论据打分（如 5/10、6/10 之类）
- **专注执行**：你的独有价值在于执行细节的评估和细化，而非方向决策

## 历史教训
{past_memory_str}

{RISK_DEBATE_PHRASING_RULES}

**重要：请用中文撰写你的交易执行方案。** 但 'FINAL TRANSACTION PROPOSAL: **BUY/OVERWEIGHT/HOLD/UNDERWEIGHT/SELL**' 这个结尾格式必须保留英文原文，这是系统解析所必需的。股票代码和技术指标名称也请保留英文。""",
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
