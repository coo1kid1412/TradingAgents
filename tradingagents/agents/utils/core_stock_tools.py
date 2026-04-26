from langchain_core.tools import tool
from typing import Annotated
from datetime import datetime, timedelta
from tradingagents.dataflows.interface import route_to_vendor

# 行情数据最低回看天数：确保覆盖中长期趋势
_MIN_STOCK_LOOKBACK_DAYS = 730


@tool
def get_stock_data(
    symbol: Annotated[str, "ticker symbol of the company"],
    start_date: Annotated[str, "Start date in yyyy-mm-dd format"],
    end_date: Annotated[str, "End date in yyyy-mm-dd format"],
) -> str:
    """
    Retrieve stock price data (OHLCV) for a given ticker symbol.
    Uses the configured core_stock_apis vendor.
    The date range is automatically extended to at least 365 days to ensure
    sufficient historical data for reliable trend analysis.
    Args:
        symbol (str): Ticker symbol of the company, e.g. AAPL, TSM
        start_date (str): Start date in yyyy-mm-dd format
        end_date (str): End date in yyyy-mm-dd format
    Returns:
        str: A formatted dataframe containing the stock price data for the specified ticker symbol in the specified date range.
    """
    # 强制最低回看 365 天
    try:
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        if (end_dt - start_dt).days < _MIN_STOCK_LOOKBACK_DAYS:
            start_date = (end_dt - timedelta(days=_MIN_STOCK_LOOKBACK_DAYS)).strftime("%Y-%m-%d")
    except ValueError:
        pass
    print(f"[get_stock_data] 正在获取 {symbol} 行情数据 ({start_date} ~ {end_date})...")
    try:
        result = route_to_vendor("get_stock_data", symbol, start_date, end_date)
        print(f"[get_stock_data] {symbol} 行情数据获取完成 ({len(result)} chars)")
        return result
    except Exception as e:
        return f"获取行情数据失败 ({symbol}): {e}"
