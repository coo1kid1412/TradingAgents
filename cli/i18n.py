"""TradingAgents CLI 中文本地化模块。

集中管理所有英文→中文的映射字典和翻译辅助函数。
内部标识符（LangGraph 节点名、状态 key 等）保持英文不变，
仅在渲染层通过此模块转换为中文显示。
"""

# Agent 名称映射（英文节点名 → 中文显示名）
AGENT_NAMES = {
    "Market Analyst": "市场分析师",
    "Social Analyst": "舆情分析师",
    "News Analyst": "新闻分析师",
    "Fundamentals Analyst": "基本面分析师",
    "Bull Researcher": "多头研究员",
    "Bear Researcher": "空头研究员",
    "Research Manager": "研究主管",
    "Trader": "交易员",
    "Aggressive Analyst": "激进风控分析师",
    "Neutral Analyst": "中立风控分析师",
    "Conservative Analyst": "保守风控分析师",
    "Portfolio Manager": "投资组合经理",
}

# 团队名称映射
TEAM_NAMES = {
    "Analyst Team": "分析师团队",
    "Research Team": "研究团队",
    "Trading Team": "交易团队",
    "Risk Management": "风险管理团队",
    "Portfolio Management": "投资组合管理",
}

# 报告板块标题映射（section key → 中文标题）
REPORT_TITLES = {
    "market_report": "市场技术分析",
    "sentiment_report": "社交舆情分析",
    "news_report": "新闻资讯分析",
    "fundamentals_report": "基本面分析",
    "investment_plan": "研究团队决策",
    "trader_investment_plan": "交易团队方案",
    "final_trade_decision": "投资组合管理决策",
}

# 状态文本映射
STATUS_TEXT = {
    "pending": "等待中",
    "in_progress": "进行中",
    "completed": "已完成",
    "error": "错误",
}

# 状态颜色映射（用于 Rich 渲染）
STATUS_COLORS = {
    "pending": "yellow",
    "completed": "green",
    "error": "red",
}

# 消息类型映射
MESSAGE_TYPES = {
    "Tool": "工具",
    "Agent": "智能体",
    "System": "系统",
    "User": "用户",
    "Data": "数据",
    "Control": "控制",
}


def t(key: str, mapping: dict = None) -> str:
    """翻译辅助函数。在指定映射字典中查找 key，找不到则返回原文。

    如果未指定 mapping，依次在 AGENT_NAMES、TEAM_NAMES 中查找。
    """
    if mapping is not None:
        return mapping.get(key, key)

    for m in (AGENT_NAMES, TEAM_NAMES, REPORT_TITLES, STATUS_TEXT, MESSAGE_TYPES):
        if key in m:
            return m[key]
    return key
