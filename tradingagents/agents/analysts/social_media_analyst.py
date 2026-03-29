from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import time
import json
from tradingagents.agents.utils.agent_utils import (
    build_instrument_context,
    get_news,
    get_xueqiu_posts,
)
from tradingagents.dataflows.config import get_config


def create_social_media_analyst(llm):
    def social_media_analyst_node(state):
        current_date = state["trade_date"]
        instrument_context = build_instrument_context(state["company_of_interest"], state.get("company_name", ""))

        tools = [
            get_news,
            get_xueqiu_posts,
        ]

        system_message = (
            "You are a social media and sentiment analyst specializing in investor community analysis. "
            "Your job is to gauge public sentiment, identify emerging narratives, and detect sentiment shifts for a specific stock.\n\n"
            "## Data Sources\n"
            "You have two tools:\n"
            "1. **get_news(ticker, start_date, end_date)** — Fetches structured financial news (东方财富/新浪). Use the stock ticker as the query.\n"
            "2. **get_xueqiu_posts(query, start_date, end_date)** — Searches Xueqiu (雪球), China's largest investment community, for posts and comments. "
            "Use this for real social media sentiment. **Call it up to 3 times** with varied queries:\n"
            "   - Stock code (e.g. '600519')\n"
            "   - Chinese company name (e.g. '贵州茅台')\n"
            "   - Common nicknames/slang (e.g. '茅子', '茅台酒')\n\n"
            "## Analysis Requirements\n"
            "After gathering data, write a comprehensive report covering:\n"
            "- **Overall sentiment**: Bullish / Bearish / Mixed, with supporting evidence from posts and comments\n"
            "- **Key narratives**: What topics are investors discussing? (earnings, policy, industry trends, etc.)\n"
            "- **Sentiment shifts**: Any notable changes in tone between posts and comments\n"
            "- **Notable voices**: Highlight influential posters (large followers, verified accounts) and their views\n"
            "- **Risk signals**: Any concerns, rumors, or negative sentiment clusters\n"
            "- **Actionable insights**: Specific takeaways for trading decisions\n\n"
            "Append a Markdown table at the end summarizing key points (sentiment, catalysts, risks)."
            "\n\n**重要：请用中文撰写你的分析报告。** 股票代码（如 AAPL）、技术指标名称请保留英文原文。Markdown 表格的表头也请使用中文。请使用专业的金融和舆情分析术语。"
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "【语言要求】你必须使用中文撰写所有分析报告和回复内容。股票代码、技术指标名称和评级关键词可保留英文。\n\n"
                    "You are a helpful AI assistant, collaborating with other assistants."
                    " Use the provided tools to progress towards answering the question."
                    " If you are unable to fully answer, that's OK; another assistant with different tools"
                    " will help where you left off. Execute what you can to make progress."
                    " If you or any other assistant has the FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** or deliverable,"
                    " prefix your response with FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** so the team knows to stop."
                    " You have access to the following tools: {tool_names}.\n{system_message}"
                    "For your reference, the current date is {current_date}. {instrument_context}",
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
