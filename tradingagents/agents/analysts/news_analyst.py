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
from tradingagents.dataflows.ticker_utils import is_a_share


def _get_available_tools(ticker: str):
    """根据目标市场返回可用的工具列表。
    
    A 股：全部工具（含公告、财联社、研报）
    港股/美股：仅通用工具（新闻、全球新闻、内部交易、网络搜索）
    """
    all_tools = [
        get_news,
        get_global_news,
        get_insider_transactions,
        get_news_from_search,
    ]
    
    # A 股专属工具
    a_share_only_tools = [
        get_announcements,
        get_cls_telegraph,
        get_research_reports,
    ]
    
    if is_a_share(ticker):
        return all_tools + a_share_only_tools
    return all_tools


def _get_tool_description(ticker: str) -> str:
    """根据目标市场返回工具描述。"""
    base_desc = (
        "## 可用工具\n"
        "你拥有以下数据工具，请根据需要合理调用：\n"
        "1. **get_news(ticker, curr_date)** — 获取个股近 10 天相关新闻（东方财富/新浪财经等），日期范围 T-10 到 T 由系统自动计算，只需传入当前分析日期\n"
        "2. **get_global_news(curr_date, look_back_days, limit)** — 获取宏观经济及全球财经新闻\n"
        "3. **get_insider_transactions(ticker)** — 获取董监高/大股东持股变动数据\n"
        "7. **get_news_from_search(ticker, query_hint)** — 通过 Brave Search 搜索实时网络新闻（支持所有市场：A股/港股/美股），"
        "返回过去7天内的 top 10 新闻（已自动排除百科类页面）。公司名称已自动解析，query_hint 仅用于补充关键词（如行业、事件），"
        "**不要**在 query_hint 中填写公司名称或股票代码\n"
    )
    
    if is_a_share(ticker):
        a_share_desc = (
            "4. **get_announcements(ticker, curr_date)** — 获取公司公告（巨潮资讯：财报、股东大会、风险提示、资产重组、股权变动等），日期范围 T-10 到 T 由系统自动计算\n"
            "5. **get_cls_telegraph(curr_date, limit)** — 获取财联社电报快讯（实时市场重大事件、央行政策、大宗商品等）\n"
            "6. **get_research_reports(ticker, limit)** — 获取个股研报（东方财富：机构评级、盈利预测、目标价等）\n"
        )
        return base_desc[:base_desc.rfind("\n7.")] + "\n" + a_share_desc + base_desc[base_desc.rfind("\n7."):]
    
    return base_desc


def _get_analysis_steps(ticker: str) -> str:
    """根据目标市场返回分析流程。"""
    if is_a_share(ticker):
        return (
            "## 分析流程\n"
            "请按照以下步骤展开分析：\n"
            "1. 调用 get_news + get_announcements 获取个股层面的新闻和公告\n"
            "2. 调用 get_news_from_search 搜索实时网络新闻，补充上一步可能遗漏的最新信息\n"
            "3. 调用 get_global_news + get_cls_telegraph 获取宏观和行业层面的信息\n"
            "4. 调用 get_insider_transactions 检查内部交易异动\n"
            "5. 调用 get_research_reports 获取机构研报观点和盈利预测\n"
        )
    else:
        return (
            "## 分析流程\n"
            "请按照以下步骤展开分析：\n"
            "1. 调用 get_news 获取个股层面的新闻\n"
            "2. 调用 get_news_from_search 搜索实时网络新闻，获取更全面的实时新闻覆盖\n"
            "3. 调用 get_global_news 获取宏观和行业层面的信息\n"
            "4. 调用 get_insider_transactions 检查内部交易异动\n"
        )


def create_news_analyst(llm):
    def news_analyst_node(state):
        current_date = state["trade_date"]
        ticker = state["company_of_interest"]
        instrument_context = build_instrument_context(ticker, state.get("company_name", ""))

        tools = _get_available_tools(ticker)
        tool_desc = _get_tool_description(ticker)
        analysis_steps = _get_analysis_steps(ticker)

        # Check if A-share to customize the role description
        is_a = is_a_share(ticker)
        role_desc = (
            "你是一名专业的新闻分析师，负责分析近期与目标股票和宏观经济相关的新闻、公告、研报及内部交易信息，"
            "撰写一份全面的新闻研究报告，为交易决策提供依据。\n\n"
            if is_a else
            "你是一名专业的新闻分析师，负责分析近期与目标股票和宏观经济相关的新闻及内部交易信息，"
            "撰写一份全面的新闻研究报告，为交易决策提供依据。\n\n"
        )

        credibility_note = (
            "- 对每条关键信息标注**可信度**（高/中/低）：\n"
            "  - **高**：来自官方公告（巨潮资讯）、权威财经媒体的一手报道、机构研报\n"
            "  - **中**：来自二手转载、综合分析类报道、Brave Search 搜索到的主流财经媒体新闻\n"
            "  - **低**：来源不明、未经证实的传闻或小道消息\n\n"
            if is_a else
            "- 对每条关键信息标注**可信度**（高/中/低）：\n"
            "  - **高**：来自 SEC  filings、权威财经媒体的一手报道、机构研报\n"
            "  - **中**：来自二手转载、综合分析类报道、Brave Search 搜索到的主流财经媒体新闻\n"
            "  - **低**：来源不明、未经证实的传闻或小道消息\n\n"
        )

        system_message = (
            role_desc
            + tool_desc + "\n"
            + analysis_steps + "\n\n"
            "**重要**：get_news_from_search 是独立的数据源，无论 get_news 是否成功都应调用，以获取更全面的实时新闻覆盖。\n\n"
            "## 信息去重与可信度评估\n"
            "- 多个数据源可能返回相同或相似的新闻，请自动合并去重，避免在报告中重复提及同一事件\n"
            + credibility_note
            + "## 行业分类\n"
            "请将所有采集到的新闻和信息按以下维度分类整理：\n"
            "- **公司层面**：财报、公告、高管变动、股权变动、并购重组\n"
            "- **行业层面**：行业政策、竞争对手动态、产业链上下游变化\n"
            "- **宏观层面**：货币政策、财政政策、国际贸易、大宗商品、汇率\n"
            + ("- **机构观点**：研报评级变化、盈利预测调整、目标价变动\n\n" if is_a else "")
            + "## 事件特征标注（每条关键事件必须标注）\n"
            "对所有进入汇总表格的事件，必须额外标注以下两项：\n\n"
            "1. **时间窗口**（horizon）：\n"
            "   - 短期（≤1 周）：交易日级别可观测影响，如财报披露、突发事件\n"
            "   - 中期（1-3 月）：季度级影响，如政策细则、产品发布\n"
            "   - 长期（>3 月）：结构性影响，如行业格局变化\n\n"
            "2. **已 priced-in 概率**（priced_in_p）：0-100 的估计值\n"
            "   - 90-100：已被市场充分定价，对未来股价边际影响小\n"
            "   - 50-90：部分定价，仍有发酵空间\n"
            "   - 0-50：尚未被定价，潜在 alpha 来源\n"
            "   - 判断依据：事件首次披露日期、市场反应幅度、卖方研报覆盖度\n\n"
            "汇总表格新增两列：| 时间窗 | 已定价概率 |\n\n"
            "## 事件二阶影响分析（必输出，识别传导链路）\n\n"
            "新闻分析的核心价值不是罗列事件，而是识别**事件的二阶传导**——某个直接影响 X 的事件，往往会通过供应链/竞争格局/政策延伸传导到 Y。\n\n"
            "对汇总表中的**关键事件（impact 绝对值 ≥ 中）**，必须标注二阶影响链路：\n\n"
            "> **示例**：'美国 H100 出口管制升级' → 直接影响：海外算力供给减少；二阶传导：(1) 国内算力替代加速 (2) 国产 GPU 厂商订单增加 (3) 国产光模块/PCB 厂商间接受益\n\n"
            "**输出格式**（每条关键事件必填）：\n"
            "- 事件：__\n"
            "- 直接影响：__\n"
            "- 二阶传导链路：(1) __ (2) __ (3) __\n"
            "- 对当前标的的相对位置：直接受益方 / 间接受益方 / 直接受损方 / 间接受损方 / 无关\n\n"
            "## 事件累积效应（必输出，识别累积趋势）\n\n"
            "单个小事件可能是噪音，多个同类事件叠加可能是趋势。对近 30 天的事件做累积模式识别：\n\n"
            "| 累积模式 | 触发条件 | 含义 |\n"
            "|---------|---------|------|\n"
            "| 多笔机构调研激增 | 近 30 日 ≥3 次机构集中调研 | 机构资金正在重新关注 |\n"
            "| 多次评级上调 | 近 30 日 ≥2 家卖方上调 | 卖方一致预期改善 |\n"
            "| 多次评级下调 | 近 30 日 ≥2 家卖方下调 | 卖方一致预期恶化 |\n"
            "| 多次内部人减持 | 近 30 日 ≥3 笔大股东/高管减持 | 内部人信号性卖出 |\n"
            "| 多次内部人增持 | 近 30 日 ≥3 笔增持 | 内部人信号性买入 |\n"
            "| 同类政策反复出台 | 多项政策指向同一方向 | 政策风向已定 |\n\n"
            "**累积效应判断**：列出本次识别到的累积模式（无则填'无显著累积'）+ 一句话解读。\n\n"
            "## 卖方研报/产业链调研纪要识别（高价值信号，必输出）\n\n"
            "真实头部投研团队会专门跟踪**渠道调研** —— 比一般公开新闻可信度高一档（机构掌握信息时间早 + 颗粒度细 + 接触管理层/上下游一手信息）。从 news 报告中识别以下关键词，提取出来单列：\n\n"
            "**关键词触发**：调研 / 产业链调研 / 渠道调研 / 管理层交流 / 分析师电话 / 业绩说明会 / 投资者交流 / 草根调研 / 终端走访 / 经销商反馈 / 上下游访谈\n\n"
            "**输出格式**（每条调研纪要必填）：\n\n"
            "| 调研类型 | 调研机构 | 时间 | 关键观点 | 可信度（高/中）| 对股价方向 |\n"
            "|---------|---------|------|---------|---------------|-----------|\n"
            "| 例：产业链调研 | 中金/招商等 | 2026-05-XX | 例：Q2 渠道库存 1.5 月（低于去年同期 2.0）| 高（一手）| +中 |\n"
            "| ... | ... | ... | ... | ... | ... |\n\n"
            "**调研纪要的特殊价值**：\n"
            "- 渠道层信号比公开新闻早 1-3 个月反映基本面变化\n"
            "- 多家机构同主题集中调研（如近 30 天 ≥3 次）→ 极强的机构关注度信号\n"
            "- 调研观点与卖方研报评级方向不一致 → 需特别警惕（可能机构内部认知分歧）\n\n"
            "若 news 报告中无调研类信号，明确填\"无渠道调研纪要识别到\"。**禁止**编造调研内容。\n\n"
            + "## 输出格式\n"
            "请用中文撰写报告，按上述维度分节组织内容。在报告末尾附上一个 Markdown 汇总表格，包含以下列：\n"
            "| 信息来源 | 日期 | 分类 | 关键内容 | 可信度 | 对股价影响 | 时间窗 | 已定价概率 |\n"
            "股票代码、专有名词和评级关键词（BUY/SELL/HOLD）请保留英文原文。"
            + get_language_instruction()
            + "\n\n## 强制输出：SUMMARY 块（位于报告末尾）\n"
            "在报告所有正文章节和汇总表格之后，**必须**附加一个 YAML 代码块，"
            "格式严格如下（字段名、单位、取值集合不可变）：\n\n"
            "```yaml\n"
            "SUMMARY:\n"
            "  net_sentiment: 正面 / 负面 / 中性          # 措辞（保守表达）\n"
            "  num_events_total: <整数>\n"
            "  key_events:\n"
            "    - title: <≤30 字>\n"
            "      category: 公司 / 行业 / 宏观 / 机构\n"
            "      event_date: <预期发生/验证日期：YYYY-MM-DD（精确）或 2026Q3（季度）或 未知>\n"
            "      horizon: 短期(≤1周) / 中期(1-3月) / 长期(>3月)\n"
            "      priced_in_p: <0-100>\n"
            "      impact: +大 / +中 / +小 / 0 / -小 / -中 / -大\n"
            "      credibility: 高 / 中 / 低\n"
            "      thesis_relevance: 核心 / 相关 / 边缘     # 该事件对投资逻辑(thesis)的相关度\n"
            "      second_order_chain: <≤50 字描述二阶传导链路，无则填 null>\n"
            "  cumulative_patterns:\n"
            "    - <识别到的累积模式，≤30 字>\n"
            "  research_consensus_rating: BUY / HOLD / SELL / null\n"
            "  research_consensus_target_price: <数值或 null>\n"
            "  data_implied_direction: 偏多 / 偏空 / 中性  # 数据真实隐含方向（穿透措辞）\n"
            "  data_implied_reasoning: <≤30 字说明>\n"
            "```\n\n"
            "## SUMMARY 规则\n"
            '- 字段缺失时填 null 或 "不适用"，不允许省略字段名\n'
            "- 取值必须落在 schema 允许的集合内\n"
            "- 数值字段保留 2 位小数；百分比字段直接填数字（不带 % 符号）\n"
            "- 该 SUMMARY 块是下游 RM / 风控团队的核心信息源，宁缺勿错\n"
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

        # 确定性催化信号注入（路2）：把 LLM 自产的结构化 key_events 聚合成 SYS_CATALYST 行，
        # 让"新闻催化"确定性进评级链（催化腿第5信号），不再靠 RM 当散文二次解读。
        if report:
            try:
                from tradingagents.dataflows.news_catalyst import (
                    aggregate_news_catalyst, aggregate_catalyst_calendar,
                )
                cat = aggregate_news_catalyst(report)
                if cat is not None:
                    report = report + (
                        f"\n\n<!-- ⚠️SYS_CATALYST｜Python 从 SUMMARY.key_events 确定性聚合，RM Step6 催化腿直读 -->\n"
                        f"SYS_CATALYST: direction={cat['direction']} | strength={cat['strength']}"
                        f" | score={cat['score']}"
                        f"（净催化分{cat['net']}，{cat['n_events']}个事件按 impact×可信度×(1-已定价)×时间窗 聚合"
                        + (f"；最近端：{cat['nearest']}" if cat['nearest'] else "")
                        + "）\n"
                    )
                # 催化日历（步骤1）：thesis 相关的有方向事件，按日期排——供 PM 时间止损/监控直读
                cal = aggregate_catalyst_calendar(report)
                if cal:
                    lines = "\n".join(
                        f"  - {c['date']} | {c['direction']} {c['impact']} | {c['thesis_relevance']}"
                        f" | {c['title']}" + (f"（已定价{c['priced_in_p']}%）" if c['priced_in_p'] not in (None, "null") else "")
                        for c in cal)
                    report = report + (
                        f"\n<!-- ⚠️SYS_CATALYST_CALENDAR｜Python 抽 thesis 相关催化事件按日期排，PM 时间止损/监控直读 -->\n"
                        f"SYS_CATALYST_CALENDAR:\n{lines}\n"
                    )
            except Exception:
                pass

        return {
            "messages": [result],
            "news_report": report,
        }

    return news_analyst_node
