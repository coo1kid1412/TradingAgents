from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from tradingagents.agents.utils.agent_utils import (
    build_instrument_context,
    get_announcements,
    get_cls_telegraph,
    get_global_news,
    get_insider_transactions,
    get_language_instruction,
    get_news,
    get_news_from_search,
    get_research_reports,
)
from tradingagents.dataflows.config import get_config


def create_news_analyst(llm):
    def news_analyst_node(state):
        current_date = state["trade_date"]
        instrument_context = build_instrument_context(state["company_of_interest"], state.get("company_name", ""))

        tools = [
            get_news,
            get_global_news,
            get_insider_transactions,
            get_announcements,
            get_cls_telegraph,
            get_research_reports,
            get_news_from_search,
        ]

        system_message = (
            "你是一名专业的新闻分析师，负责分析近期与目标股票和宏观经济相关的新闻、公告、研报及内部交易信息，"
            "撰写一份全面的新闻研究报告，为交易决策提供依据。\n\n"
            "## 可用工具\n"
            "你拥有以下七个数据工具，请根据需要合理调用：\n"
            "1. **get_news(ticker, curr_date)** — 获取个股近 10 天相关新闻（东方财富/新浪财经等），日期范围 T-10 到 T 由系统自动计算，只需传入当前分析日期\n"
            "2. **get_global_news(curr_date, look_back_days, limit)** — 获取宏观经济及全球财经新闻\n"
            "3. **get_insider_transactions(ticker)** — 获取董监高/大股东持股变动数据\n"
            "4. **get_announcements(ticker, curr_date)** — 获取公司公告（巨潮资讯：财报、股东大会、风险提示、资产重组、股权变动等），日期范围 T-10 到 T 由系统自动计算\n"
            "5. **get_cls_telegraph(curr_date, limit)** — 获取财联社电报快讯（实时市场重大事件、央行政策、大宗商品等）\n"
            "6. **get_research_reports(ticker, limit)** — 获取个股研报（东方财富：机构评级、盈利预测、目标价等）\n"
            "7. **get_news_from_search(ticker, query_hint)** — 通过 Brave Search 搜索实时网络新闻（支持所有市场：A股/港股/美股），"
            "返回过去7天内的 top 10 新闻（已自动排除百科类页面）。公司名称已自动解析，query_hint 仅用于补充关键词（如行业、事件），"
            "**不要**在 query_hint 中填写公司名称或股票代码\n\n"
            "## 分析流程\n"
            "请按照以下步骤展开分析：\n"
            "1. 调用 get_news + get_announcements 获取个股层面的新闻和公告\n"
            "2. 调用 get_news_from_search 搜索实时网络新闻，补充上一步可能遗漏的最新信息\n"
            "3. 调用 get_global_news + get_cls_telegraph 获取宏观和行业层面的信息\n"
            "4. 调用 get_insider_transactions 检查内部交易异动\n"
            "5. 调用 get_research_reports 获取机构研报观点和盈利预测\n\n"
            "**重要**：get_news_from_search 是独立的数据源，无论 get_news 是否成功都应调用，以获取更全面的实时新闻覆盖。\n\n"
            "## 信息去重与可信度评估\n"
            "- 多个数据源可能返回相同或相似的新闻，请自动合并去重，避免在报告中重复提及同一事件\n"
            "- 对每条关键信息标注**可信度**（高/中/低）：\n"
            "  - **高**：来自官方公告（巨潮资讯）、权威财经媒体的一手报道、机构研报\n"
            "  - **中**：来自二手转载、综合分析类报道、Brave Search 搜索到的主流财经媒体新闻\n"
            "  - **低**：来源不明、未经证实的传闻或小道消息\n\n"
            "## 行业分类\n"
            "请将所有采集到的新闻和信息按以下维度分类整理：\n"
            "- **公司层面**：财报、公告、高管变动、股权变动、并购重组\n"
            "- **行业层面**：行业政策、竞争对手动态、产业链上下游变化\n"
            "- **宏观层面**：货币政策、财政政策、国际贸易、大宗商品、汇率\n"
            "- **机构观点**：研报评级变化、盈利预测调整、目标价变动\n\n"
            "## 输出格式\n"
            "请用中文撰写报告，按上述四个维度分节组织内容。在报告末尾附上一个 Markdown 汇总表格，包含以下列：\n"
            "| 信息来源 | 日期 | 分类 | 关键内容 | 可信度 | 对股价影响 |\n"
            "股票代码、专有名词和评级关键词（BUY/SELL/HOLD）请保留英文原文。"
            + get_language_instruction()
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
            "news_report": report,
        }

    return news_analyst_node
