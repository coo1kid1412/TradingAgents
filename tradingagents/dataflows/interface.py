from typing import Annotated

# Import from vendor-specific modules
from .y_finance import (
    get_YFin_data_online,
    get_stock_stats_indicators_window,
    get_fundamentals as get_yfinance_fundamentals,
    get_balance_sheet as get_yfinance_balance_sheet,
    get_cashflow as get_yfinance_cashflow,
    get_income_statement as get_yfinance_income_statement,
    get_insider_transactions as get_yfinance_insider_transactions,
)
from .yfinance_news import get_news_yfinance, get_global_news_yfinance
from .alpha_vantage import (
    get_stock as get_alpha_vantage_stock,
    get_indicator as get_alpha_vantage_indicator,
    get_fundamentals as get_alpha_vantage_fundamentals,
    get_balance_sheet as get_alpha_vantage_balance_sheet,
    get_cashflow as get_alpha_vantage_cashflow,
    get_income_statement as get_alpha_vantage_income_statement,
    get_insider_transactions as get_alpha_vantage_insider_transactions,
    get_news as get_alpha_vantage_news,
    get_global_news as get_alpha_vantage_global_news,
)
from .akshare_vendor import (
    get_stock as get_akshare_stock,
    get_indicator as get_akshare_indicator,
    get_fundamentals as get_akshare_fundamentals,
    get_balance_sheet as get_akshare_balance_sheet,
    get_cashflow as get_akshare_cashflow,
    get_income_statement as get_akshare_income_statement,
    get_news as get_akshare_news,
    get_global_news as get_akshare_global_news,
    get_insider_transactions as get_akshare_insider_transactions,
    get_announcements as get_akshare_announcements,
    get_cls_telegraph as get_akshare_cls_telegraph,
    get_research_reports as get_akshare_research_reports,
)
from .tushare_vendor import (
    get_stock as get_tushare_stock,
    get_indicator as get_tushare_indicator,
    get_fundamentals as get_tushare_fundamentals,
    get_balance_sheet as get_tushare_balance_sheet,
    get_cashflow as get_tushare_cashflow,
    get_income_statement as get_tushare_income_statement,
    get_news as get_tushare_news,
    get_global_news as get_tushare_global_news,
    get_insider_transactions as get_tushare_insider_transactions,
)
from .alpha_vantage_common import AlphaVantageRateLimitError
from .vendor_errors import VendorRateLimitError, VendorUnavailableError
from .ticker_utils import is_a_share

try:
    from yfinance.exceptions import YFRateLimitError
except ImportError:
    YFRateLimitError = type(None)  # 如果 yfinance 未安装，不会匹配任何异常

# Configuration and routing logic
from .config import get_config

# Tools organized by category
TOOLS_CATEGORIES = {
    "core_stock_apis": {
        "description": "OHLCV stock price data",
        "tools": [
            "get_stock_data"
        ]
    },
    "technical_indicators": {
        "description": "Technical analysis indicators",
        "tools": [
            "get_indicators"
        ]
    },
    "fundamental_data": {
        "description": "Company fundamentals",
        "tools": [
            "get_fundamentals",
            "get_balance_sheet",
            "get_cashflow",
            "get_income_statement"
        ]
    },
    "news_data": {
        "description": "News and insider data",
        "tools": [
            "get_news",
            "get_global_news",
            "get_insider_transactions",
            "get_announcements",
            "get_cls_telegraph",
            "get_research_reports",
        ]
    }
}

VENDOR_LIST = [
    "akshare",
    "tushare",
    "yfinance",
    "alpha_vantage",
]

# A股按接口类型分别设置优先级
# 行情/财务类：Tushare 速度更快、数据更稳定 → tushare > akshare > yfinance
# 新闻/公告类：AKShare 数据更丰富（东方财富个股新闻、公告、电报、研报） → akshare > tushare > yfinance
_A_SHARE_MARKET_FINANCE_ORDER = ["tushare", "akshare", "yfinance"]
_A_SHARE_NEWS_ORDER = ["akshare", "tushare", "yfinance"]

_A_SHARE_METHOD_VENDOR_ORDER = {
    # 行情/财务类
    "get_stock_data": _A_SHARE_MARKET_FINANCE_ORDER,
    "get_indicators": _A_SHARE_MARKET_FINANCE_ORDER,
    "get_fundamentals": _A_SHARE_MARKET_FINANCE_ORDER,
    "get_balance_sheet": _A_SHARE_MARKET_FINANCE_ORDER,
    "get_cashflow": _A_SHARE_MARKET_FINANCE_ORDER,
    "get_income_statement": _A_SHARE_MARKET_FINANCE_ORDER,
    "get_insider_transactions": _A_SHARE_MARKET_FINANCE_ORDER,
    # 新闻/公告类
    "get_news": _A_SHARE_NEWS_ORDER,
    "get_global_news": _A_SHARE_NEWS_ORDER,
    "get_announcements": _A_SHARE_NEWS_ORDER,
    "get_cls_telegraph": _A_SHARE_NEWS_ORDER,
    "get_research_reports": _A_SHARE_NEWS_ORDER,
}

# Mapping of methods to their vendor-specific implementations
VENDOR_METHODS = {
    # core_stock_apis
    "get_stock_data": {
        "akshare": get_akshare_stock,
        "tushare": get_tushare_stock,
        "alpha_vantage": get_alpha_vantage_stock,
        "yfinance": get_YFin_data_online,
    },
    # technical_indicators
    "get_indicators": {
        "akshare": get_akshare_indicator,
        "tushare": get_tushare_indicator,
        "alpha_vantage": get_alpha_vantage_indicator,
        "yfinance": get_stock_stats_indicators_window,
    },
    # fundamental_data
    "get_fundamentals": {
        "akshare": get_akshare_fundamentals,
        "tushare": get_tushare_fundamentals,
        "alpha_vantage": get_alpha_vantage_fundamentals,
        "yfinance": get_yfinance_fundamentals,
    },
    "get_balance_sheet": {
        "akshare": get_akshare_balance_sheet,
        "tushare": get_tushare_balance_sheet,
        "alpha_vantage": get_alpha_vantage_balance_sheet,
        "yfinance": get_yfinance_balance_sheet,
    },
    "get_cashflow": {
        "akshare": get_akshare_cashflow,
        "tushare": get_tushare_cashflow,
        "alpha_vantage": get_alpha_vantage_cashflow,
        "yfinance": get_yfinance_cashflow,
    },
    "get_income_statement": {
        "akshare": get_akshare_income_statement,
        "tushare": get_tushare_income_statement,
        "alpha_vantage": get_alpha_vantage_income_statement,
        "yfinance": get_yfinance_income_statement,
    },
    # news_data
    "get_news": {
        "akshare": get_akshare_news,
        "tushare": get_tushare_news,
        "alpha_vantage": get_alpha_vantage_news,
        "yfinance": get_news_yfinance,
    },
    "get_global_news": {
        "akshare": get_akshare_global_news,
        "tushare": get_tushare_global_news,
        "yfinance": get_global_news_yfinance,
        "alpha_vantage": get_alpha_vantage_global_news,
    },
    "get_insider_transactions": {
        "akshare": get_akshare_insider_transactions,
        "tushare": get_tushare_insider_transactions,
        "alpha_vantage": get_alpha_vantage_insider_transactions,
        "yfinance": get_yfinance_insider_transactions,
    },
    # announcements (A-share only, 巨潮资讯)
    "get_announcements": {
        "akshare": get_akshare_announcements,
    },
    # CLS telegraph (财联社电报)
    "get_cls_telegraph": {
        "akshare": get_akshare_cls_telegraph,
    },
    # research reports (个股研报)
    "get_research_reports": {
        "akshare": get_akshare_research_reports,
    },
}

def get_category_for_method(method: str) -> str:
    """Get the category that contains the specified method."""
    for category, info in TOOLS_CATEGORIES.items():
        if method in info["tools"]:
            return category
    raise ValueError(f"方法 '{method}' 未在任何类别中找到")

def get_vendor(category: str, method: str = None) -> str:
    """Get the configured vendor for a data category or specific tool method.
    Tool-level configuration takes precedence over category-level.
    """
    config = get_config()

    # Check tool-level configuration first (if method provided)
    if method:
        tool_vendors = config.get("tool_vendors", {})
        if method in tool_vendors:
            return tool_vendors[method]

    # Fall back to category-level configuration
    return config.get("data_vendors", {}).get(category, "default")

def route_to_vendor(method: str, *args, **kwargs):
    """Route method calls to appropriate vendor implementation with fallback support.

    A股代码会按接口类型自动路由：
      - 行情/财务类：tushare → akshare → yfinance
      - 新闻/公告类：akshare → tushare → yfinance
    非A股代码按配置文件的 vendor 顺序路由。
    """
    if method not in VENDOR_METHODS:
        raise ValueError(f"Method '{method}' not supported")

    # 检测第一个参数是否为 A 股代码
    symbol = args[0] if args else kwargs.get("symbol", kwargs.get("ticker", ""))
    a_share_detected = is_a_share(symbol) if symbol else False

    if a_share_detected:
        # A股：按接口类型选择对应优先级顺序
        vendor_order = _A_SHARE_METHOD_VENDOR_ORDER.get(
            method, _A_SHARE_MARKET_FINANCE_ORDER
        )
        fallback_vendors = [
            v for v in vendor_order
            if v in VENDOR_METHODS[method]
        ]
    else:
        # 非A股：按配置路由
        category = get_category_for_method(method)
        vendor_config = get_vendor(category, method)
        primary_vendors = [v.strip() for v in vendor_config.split(',')]

        all_available_vendors = list(VENDOR_METHODS[method].keys())
        fallback_vendors = primary_vendors.copy()
        for vendor in all_available_vendors:
            if vendor not in fallback_vendors:
                fallback_vendors.append(vendor)

    last_error = None
    for vendor in fallback_vendors:
        if vendor not in VENDOR_METHODS[method]:
            continue

        vendor_impl = VENDOR_METHODS[method][vendor]
        impl_func = vendor_impl[0] if isinstance(vendor_impl, list) else vendor_impl

        try:
            return impl_func(*args, **kwargs)
        except (AlphaVantageRateLimitError, VendorRateLimitError,
                VendorUnavailableError, YFRateLimitError) as e:
            last_error = e
            continue  # 限流或不可用时 fallback 到下一个供应商
        except Exception as e:
            last_error = e
            continue  # 其他异常（参数错误、网络超时等）也 fallback

    raise RuntimeError(
        f"方法 '{method}' 所有数据供应商均失败"
        + (f"（最后一个错误：{last_error}）" if last_error else "")
    )