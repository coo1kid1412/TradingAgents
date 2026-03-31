from langchain_core.tools import tool
from typing import Annotated
from datetime import datetime, timedelta
from tradingagents.dataflows.interface import route_to_vendor
from tradingagents.dataflows.ticker_utils import is_a_share, _extract_code
from tradingagents.dataflows.brave_search import search_news


def _resolve_company_name(ticker: str) -> str:
    """尝试从 ticker resolver 获取公司名称，失败返回空字符串。"""
    if not is_a_share(ticker):
        return ""
    try:
        from tradingagents.dataflows.ticker_resolver import resolve_ticker
        resolved = resolve_ticker(ticker)
        # 排除 fallback 返回的纯代码占位名
        if resolved.name and resolved.name != resolved.code:
            return resolved.name
    except Exception:
        pass
    return ""


def _safe_route(method: str, *args, **kwargs) -> str:
    """调用 route_to_vendor 并兜底异常，确保 @tool 永远返回 str 而非抛异常。"""
    try:
        return route_to_vendor(method, *args, **kwargs)
    except Exception as e:
        return f"获取数据失败 ({method}): {e}"


_CN_ONLY_MSG = (
    "该工具仅支持中国 A 股市场（沪深京），"
    "不支持港股/美股。请使用 get_news 获取该股票的新闻信息。"
)

# 个股新闻固定回看天数
_NEWS_LOOKBACK_DAYS = 10

@tool
def get_news(
    ticker: Annotated[str, "Ticker symbol"],
    curr_date: Annotated[str, "Current analysis date in yyyy-mm-dd format"],
) -> str:
    """
    Retrieve news data for a given ticker symbol.
    Automatically fetches news from T-10 to T (T = curr_date).
    Args:
        ticker (str): Ticker symbol
        curr_date (str): Current analysis date in yyyy-mm-dd format
    Returns:
        str: A formatted string containing news data
    """
    try:
        end_dt = datetime.strptime(curr_date, "%Y-%m-%d")
        start_dt = end_dt - timedelta(days=_NEWS_LOOKBACK_DAYS)
        start_date = start_dt.strftime("%Y-%m-%d")
        end_date = curr_date
    except ValueError:
        start_date = curr_date
        end_date = curr_date
    return _safe_route("get_news", ticker, start_date, end_date)

@tool
def get_global_news(
    curr_date: Annotated[str, "Current date in yyyy-mm-dd format"],
    look_back_days: Annotated[int, "Number of days to look back"] = 7,
    limit: Annotated[int, "Maximum number of articles to return"] = 5,
) -> str:
    """
    Retrieve global news data.
    Uses the configured news_data vendor.
    Args:
        curr_date (str): Current date in yyyy-mm-dd format
        look_back_days (int): Number of days to look back (default 7)
        limit (int): Maximum number of articles to return (default 5)
    Returns:
        str: A formatted string containing global news data
    """
    return _safe_route("get_global_news", curr_date, look_back_days, limit)

@tool
def get_insider_transactions(
    ticker: Annotated[str, "ticker symbol"],
) -> str:
    """
    Retrieve insider transaction information about a company.
    Uses the configured news_data vendor.
    Args:
        ticker (str): Ticker symbol of the company
    Returns:
        str: A report of insider transaction data
    """
    return _safe_route("get_insider_transactions", ticker)

@tool
def get_announcements(
    ticker: Annotated[str, "Ticker symbol"],
    curr_date: Annotated[str, "Current analysis date in yyyy-mm-dd format"],
) -> str:
    """
    Retrieve company announcements / disclosures (公告) for a given ticker.
    Data source: 巨潮资讯 (cninfo.com.cn) via AKShare. Only supports A-shares.
    Covers: earnings reports, shareholder changes, risk warnings, M&A, etc.
    Automatically fetches announcements from T-10 to T (T = curr_date).
    Args:
        ticker (str): Ticker symbol
        curr_date (str): Current analysis date in yyyy-mm-dd format
    Returns:
        str: A formatted string containing announcement data
    """
    if not is_a_share(ticker):
        return f"get_announcements: {ticker} — {_CN_ONLY_MSG}"
    try:
        end_dt = datetime.strptime(curr_date, "%Y-%m-%d")
        start_dt = end_dt - timedelta(days=_NEWS_LOOKBACK_DAYS)
        start_date = start_dt.strftime("%Y-%m-%d")
        end_date = curr_date
    except ValueError:
        start_date = curr_date
        end_date = curr_date
    return _safe_route("get_announcements", ticker, start_date, end_date)

@tool
def get_cls_telegraph(
    curr_date: Annotated[str, "Current date in yyyy-mm-dd format"],
    limit: Annotated[int, "Maximum number of telegraphs to return"] = 30,
) -> str:
    """
    Retrieve real-time financial flash news from CLS (财联社电报).
    Fast-breaking market-moving news covering macro policy, central bank decisions,
    commodity prices, and corporate events.
    Args:
        curr_date (str): Current date in yyyy-mm-dd format
        limit (int): Maximum number of telegraphs (default 30)
    Returns:
        str: A formatted string containing telegraph data
    """
    return _safe_route("get_cls_telegraph", curr_date, limit)

@tool
def get_research_reports(
    ticker: Annotated[str, "Ticker symbol"],
    limit: Annotated[int, "Maximum number of reports to return"] = 20,
) -> str:
    """
    Retrieve analyst research reports (个股研报) for a given ticker.
    Data source: 东方财富. Only supports A-shares.
    Includes analyst ratings, earnings forecasts, target prices, and research institution names.
    Args:
        ticker (str): Ticker symbol
        limit (int): Maximum number of reports (default 20)
    Returns:
        str: A formatted string containing research report data
    """
    if not is_a_share(ticker):
        return f"get_research_reports: {ticker} — {_CN_ONLY_MSG}"
    return _safe_route("get_research_reports", ticker, limit)

@tool
def get_news_from_search(
    ticker: Annotated[str, "Ticker symbol or stock name"],
    query_hint: Annotated[str, "Additional keywords to refine the search (optional, company name is auto-resolved)"] = "",
) -> str:
    """
    Search real-time news from the web using Brave Search API.
    Independent data source complementing get_news. Returns top 10 results
    from the past 7 days, excluding encyclopedia pages.
    Supports ALL markets (A-shares, HK, US, etc.).
    For A-shares, the company name is auto-resolved — no need to provide it in query_hint.
    Args:
        ticker (str): Ticker symbol or stock name
        query_hint (str): Extra keywords to narrow search (optional; do NOT put stock name here, it is auto-resolved)
    Returns:
        str: Formatted list of recent news articles from web search
    """
    code = _extract_code(ticker) if is_a_share(ticker) else ticker
    company_name = _resolve_company_name(ticker)

    # 构建 query：公司名 + 代码 + 额外关键词 + "新闻"
    parts = []
    if company_name:
        parts.append(company_name)
    parts.append(code)
    if query_hint:
        # 过滤掉 query_hint 中与 code/company_name 重复的部分
        hint_clean = query_hint
        if company_name:
            hint_clean = hint_clean.replace(company_name, "")
        hint_clean = hint_clean.replace(code, "").strip()
        if hint_clean:
            parts.append(hint_clean)
    parts.append("新闻")
    query = " ".join(parts)
    try:
        return search_news(query, count=20, freshness="pw")
    except Exception as e:
        return f"Brave Search 调用失败: {e}"
