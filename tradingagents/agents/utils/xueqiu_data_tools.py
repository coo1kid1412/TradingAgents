from langchain_core.tools import tool
from typing import Annotated
from tradingagents.dataflows.xueqiu_sentiment import fetch_xueqiu_posts


@tool
def get_xueqiu_posts(
    query: Annotated[str, "Search query - stock code, company name, or slang term"],
    start_date: Annotated[str, "Start date in yyyy-mm-dd format"],
    end_date: Annotated[str, "End date in yyyy-mm-dd format"],
) -> str:
    """Search Xueqiu (雪球) for stock-related posts, comments and investor sentiment.

    Use this tool to find social media discussions about a stock on Xueqiu,
    China's largest investment community. You can search by stock code (e.g. '600519'),
    company name (e.g. '贵州茅台'), or common slang/abbreviation (e.g. '茅子').
    Call this tool up to 3 times with different queries to get broader coverage.

    Args:
        query: Stock code, company name, or slang term to search for.
        start_date: Start date in yyyy-mm-dd format.
        end_date: End date in yyyy-mm-dd format.

    Returns:
        Formatted markdown string with posts, comments, and sentiment data.
    """
    try:
        return fetch_xueqiu_posts(query, start_date, end_date)
    except Exception as e:
        return f"获取雪球数据失败: {str(e)}"
