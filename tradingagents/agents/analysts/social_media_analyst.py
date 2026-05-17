from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import time
import json
from tradingagents.agents.utils.agent_utils import (
    build_instrument_context,
    get_language_instruction,
    get_xueqiu_posts,
)
from tradingagents.dataflows.config import get_config


def create_social_media_analyst(llm):
    def social_media_analyst_node(state):
        current_date = state["trade_date"]
        instrument_context = build_instrument_context(state["company_of_interest"], state.get("company_name", ""))

        tools = [
            get_xueqiu_posts,
        ]

        system_message = (
            "你是一名专业舆情分析师，负责分析投资者社区情绪并撰写舆情分析报告。\n\n"
            "## 数据来源\n"
            "你拥有以下工具：\n"
            "1. **get_xueqiu_posts(query, start_date, end_date)** — 搜索雪球（中国最大投资社区）帖子和评论。"
            "最多调用 3 次，使用不同查询词：\n"
            "   - 股票代码（如 '600519'）\n"
            "   - 公司名称（如 '贵州茅台'）\n"
            "   - 常用昵称/简称（如 '茅子'、'茅台酒'）\n\n"
            "## 分析要求\n"
            "撰写全面报告，涵盖：\n"
            "- **量化情绪指标**（必须输出数值）：\n"
            "  - 多头帖子占比：基于采样到的帖子中明确表达看多观点的比例（0-100%）\n"
            "  - 空头帖子占比：明确看空的比例\n"
            "  - 中性占比：100% − 多头 − 空头\n"
            "  - **三者之和必须等于 100%**\n"
            "- **KOL 一致性**（必须输出）：\n"
            "  - 识别样本中粉丝量大或发帖活跃的 KOL（≥3 个为有效观察）\n"
            '  - 标注是"一致看多 / 一致看空 / 分歧"\n'
            '  - 若 KOL 不足 3 个，标注"无明显 KOL"\n'
            "- **7 日情绪变化净值**（必须输出）：\n"
            "  - 比较最近 7 天与之前 7 天的多空占比变化\n"
            "  - 输出区间 -100 ~ +100 的整数（正数 = 偏多增强；负数 = 偏空增强）\n"
            "- **核心叙事**：投资者讨论的主要话题（业绩、政策、行业趋势等）\n"
            "- **情绪变化**：帖子与评论之间情绪的显著变化\n"
            "- **关键意见领袖**：突出有影响力的发言者及其观点\n"
            "- **风险信号**：担忧、传闻或负面情绪集群\n"
            "- **可执行洞察**：对交易决策的具体建议\n\n"
            "在报告末尾附 Markdown 汇总表格（情绪、催化因素、风险）。"
            "\n\n**重要**：股票代码（如 AAPL）、技术指标名称请保留英文原文。"
            "Markdown 表格的表头请使用中文。请使用专业的金融和舆情分析术语。"
            + get_language_instruction()
            + "\n\n## 强制输出：SUMMARY 块（位于报告末尾）\n"
            "在报告所有正文章节和汇总表格之后，**必须**附加一个 YAML 代码块，"
            "格式严格如下（字段名、单位、取值集合不可变）：\n\n"
            "```yaml\n"
            "SUMMARY:\n"
            "  net_sentiment: 偏多 / 偏空 / 分歧\n"
            "  bull_post_pct: <0-100>\n"
            "  bear_post_pct: <0-100>\n"
            "  neutral_post_pct: <0-100>\n"
            "  kol_consensus: 一致看多 / 一致看空 / 分歧 / 无明显 KOL\n"
            "  kol_count_observed: <整数>\n"
            "  sentiment_trend_7d: <-100 ~ +100>\n"
            "  key_narratives:\n"
            "    - <叙事主题，≤20 字>\n"
            "  rating: BUY / HOLD / SELL                # 措辞评级（保守表达）\n"
            "  data_implied_direction: 偏多 / 偏空 / 中性  # 数据真实隐含方向（穿透措辞）\n"
            "  data_implied_reasoning: <≤30 字说明>\n"
            "```\n\n"
            "## SUMMARY 规则\n"
            '- 字段缺失时填 null 或 "不适用"，不允许省略字段名\n'
            "- 取值必须落在 schema 允许的集合内\n"
            "- bull_post_pct + bear_post_pct + neutral_post_pct 必须 = 100\n"
            "- 数值字段保留 2 位小数；百分比字段直接填数字（不带 % 符号）\n"
            "- 该 SUMMARY 块是下游 RM / 风控团队的核心信息源，宁缺勿错\n"
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "【语言要求】你必须使用中文撰写所有分析报告和回复内容。股票代码、技术指标名称和评级关键词可保留英文。\n\n"
                    "你是一个协作式 AI 助手。使用提供的工具推进分析。"
                    "如果你无法完全回答，其他助手会协助。"
                    "如果你或任何助手有最终交易建议 **BUY/HOLD/SELL**，"
                    "请在回复前加上 FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL**。"
                    "你可使用以下工具：{tool_names}。\n{system_message}"
                    "当前日期：{current_date}。{instrument_context}",
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

        prompt = prompt.partial(system_message=system_message)
        prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))
        prompt = prompt.partial(current_date=current_date)
        prompt = prompt.partial(instrument_context=instrument_context)

        chain = prompt | llm.bind_tools(tools)

        result = chain.invoke(state["messages"])

        report = ""

        if len(result.tool_calls) == 0:
            report = result.content

        return {
            "messages": [result],
            "sentiment_report": report,
        }

    return social_media_analyst_node
