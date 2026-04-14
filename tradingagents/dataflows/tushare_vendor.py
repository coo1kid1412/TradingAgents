"""Tushare Pro 数据供应商 —— 专业 A 股数据接口。

Token 通过环境变量 TUSHARE_TOKEN 配置。
未配置 Token 时，所有方法抛出 TushareUnavailableError 以触发 fallback。
"""

import os
from typing import Annotated
from datetime import datetime, timedelta

import pandas as pd

from .ticker_utils import to_tushare_format, to_akshare_date, to_standard_date
from .vendor_errors import TushareRateLimitError, TushareUnavailableError

_ts_api = None


def _get_tushare_api():
    """获取或初始化 Tushare Pro API 实例（单例）。"""
    global _ts_api
    if _ts_api is not None:
        return _ts_api

    token = os.environ.get("TUSHARE_TOKEN", "").strip()
    if not token:
        raise TushareUnavailableError(
            "TUSHARE_TOKEN 环境变量未配置。"
            "请在 .env 文件中设置 TUSHARE_TOKEN=你的token。"
            "访问 https://tushare.pro/ 注册获取免费 Token。"
        )

    try:
        import tushare as ts
        ts.set_token(token)
        _ts_api = ts.pro_api()
        return _ts_api
    except ImportError:
        raise TushareUnavailableError("tushare 未安装，请运行: pip install tushare")
    except Exception as e:
        raise TushareUnavailableError(f"Tushare 初始化失败：{e}")


def _safe_call(func, *args, **kwargs):
    """包装 Tushare API 调用，捕获频率限制和权限错误。"""
    try:
        return func(*args, **kwargs)
    except (TushareUnavailableError, TushareRateLimitError):
        raise
    except Exception as e:
        msg = str(e).lower()
        if any(kw in msg for kw in ("每分钟", "rate", "频率", "too many")):
            raise TushareRateLimitError(f"Tushare 请求频率超限：{e}")
        if any(kw in msg for kw in ("积分", "权限", "point", "permission")):
            raise TushareUnavailableError(f"Tushare 积分不足或权限不够：{e}")
        raise TushareUnavailableError(f"Tushare API 调用失败：{e}")


# ---------------------------------------------------------------------------
# 1. get_stock
# ---------------------------------------------------------------------------
def get_stock(
    symbol: Annotated[str, "ticker symbol of the company"],
    start_date: Annotated[str, "Start date in yyyy-mm-dd format"],
    end_date: Annotated[str, "End date in yyyy-mm-dd format"],
) -> str:
    """获取 A 股日线行情（Tushare Pro）。"""
    pro = _get_tushare_api()
    ts_code = to_tushare_format(symbol)

    df = _safe_call(
        pro.daily,
        ts_code=ts_code,
        start_date=to_akshare_date(start_date),
        end_date=to_akshare_date(end_date),
    )

    if df is None or df.empty:
        return f"未找到股票 '{symbol}' 在 {start_date} 至 {end_date} 期间的数据"

    df = df.sort_values("trade_date")
    result = pd.DataFrame({
        "Date": df["trade_date"].apply(lambda x: to_standard_date(str(x))),
        "Open": df["open"].round(2),
        "High": df["high"].round(2),
        "Low": df["low"].round(2),
        "Close": df["close"].round(2),
        "Volume": (df["vol"] * 100).astype(int),  # 手 → 股
    })

    csv_string = result.to_csv(index=False)

    # 显示实际返回的数据范围，而非请求的数据范围
    actual_start = result["Date"].iloc[0]
    actual_end = result["Date"].iloc[-1]

    header = (
        f"# Stock data for {symbol}\n"
        f"# Actual date range: {actual_start} to {actual_end} "
        f"(requested: {start_date} to {end_date})\n"
        f"# Source: Tushare Pro\n"
        f"# Total records: {len(result)}\n"
        f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    )
    return header + csv_string


# ---------------------------------------------------------------------------
# 2. get_indicator
# ---------------------------------------------------------------------------
def get_indicator(
    symbol: Annotated[str, "ticker symbol of the company"],
    indicator: Annotated[str, "technical indicator (stockstats format)"],
    curr_date: Annotated[str, "current trading date YYYY-mm-dd"],
    look_back_days: Annotated[int, "days to look back"],
) -> str:
    """使用 Tushare 日线 + stockstats 计算技术指标。"""
    from .stockstats_utils import calculate_indicator_from_ohlcv

    pro = _get_tushare_api()
    ts_code = to_tushare_format(symbol)

    curr_dt = datetime.strptime(curr_date, "%Y-%m-%d")
    hist_start = curr_dt - timedelta(days=365)

    df = _safe_call(
        pro.daily,
        ts_code=ts_code,
        start_date=to_akshare_date(hist_start.strftime("%Y-%m-%d")),
        end_date=to_akshare_date(curr_date),
    )

    if df is None or df.empty:
        return f"未找到股票 '{symbol}' 的历史行情数据，无法计算指标"

    df = df.sort_values("trade_date")
    ohlcv = pd.DataFrame({
        "Date": df["trade_date"].apply(lambda x: to_standard_date(str(x))),
        "Open": df["open"],
        "High": df["high"],
        "Low": df["low"],
        "Close": df["close"],
        "Volume": (df["vol"] * 100).astype(int),
    })

    indicator_data = calculate_indicator_from_ohlcv(ohlcv, indicator)

    # 记录股票最早上市日期，用于区分"非交易日"和"尚未上市"
    first_listed_date = ohlcv["Date"].min() if not ohlcv.empty else None

    before = curr_dt - timedelta(days=look_back_days)
    lines = []
    current_dt = curr_dt
    while current_dt >= before:
        ds = current_dt.strftime("%Y-%m-%d")
        if ds in indicator_data:
            val = indicator_data[ds]
        elif first_listed_date and ds < first_listed_date:
            val = "N/A：股票尚未上市"
        else:
            val = "N/A：非交易日（周末或节假日）"
        lines.append(f"{ds}: {val}")
        current_dt -= timedelta(days=1)

    return (
        f"## {indicator} values from {before.strftime('%Y-%m-%d')} to {curr_date}:\n"
        f"## Source: Tushare Pro + stockstats\n\n"
        + "\n".join(lines)
        + "\n"
    )


# ---------------------------------------------------------------------------
# 3. get_fundamentals
# ---------------------------------------------------------------------------
def get_fundamentals(
    ticker: Annotated[str, "ticker symbol of the company"],
    curr_date: Annotated[str, "current date"] = None,
) -> str:
    """获取 A 股公司基本面（Tushare Pro）。"""
    pro = _get_tushare_api()
    ts_code = to_tushare_format(ticker)
    sections: list[str] = []

    # 公司基本信息
    try:
        basic = _safe_call(
            pro.stock_basic,
            ts_code=ts_code,
            fields="ts_code,symbol,name,area,industry,market,list_date",
        )
        if basic is not None and not basic.empty:
            row = basic.iloc[0]
            sections.append("## 公司基本信息")
            sections.append(f"代码: {row.get('ts_code', 'N/A')}")
            sections.append(f"名称: {row.get('name', 'N/A')}")
            sections.append(f"地区: {row.get('area', 'N/A')}")
            sections.append(f"行业: {row.get('industry', 'N/A')}")
            sections.append(f"市场: {row.get('market', 'N/A')}")
            sections.append(f"上市日期: {row.get('list_date', 'N/A')}")
            sections.append("")
    except (TushareUnavailableError, TushareRateLimitError):
        raise
    except Exception as e:
        sections.append(f"# 获取公司信息出错：{e}\n")

    # 财务指标
    try:
        fina = _safe_call(pro.fina_indicator, ts_code=ts_code, limit=4)
        if fina is not None and not fina.empty:
            sections.append("## 财务指标（最近4期）")
            sections.append(fina.to_csv(index=False))
    except (TushareUnavailableError, TushareRateLimitError):
        raise
    except Exception as e:
        sections.append(f"# 获取财务指标出错：{e}\n")

    # 估值指标（PE / PB 等）
    try:
        if curr_date:
            daily_basic = _safe_call(
                pro.daily_basic,
                ts_code=ts_code,
                trade_date=to_akshare_date(curr_date),
            )
        else:
            daily_basic = _safe_call(pro.daily_basic, ts_code=ts_code, limit=1)
        if daily_basic is not None and not daily_basic.empty:
            r = daily_basic.iloc[0]
            sections.append("## 估值指标")
            sections.append(f"PE(TTM): {r.get('pe_ttm', 'N/A')}")
            sections.append(f"PB: {r.get('pb', 'N/A')}")
            sections.append(f"PS(TTM): {r.get('ps_ttm', 'N/A')}")
            sections.append(f"总市值(万元): {r.get('total_mv', 'N/A')}")
            sections.append(f"流通市值(万元): {r.get('circ_mv', 'N/A')}")
            sections.append("")
    except (TushareUnavailableError, TushareRateLimitError):
        raise
    except Exception as e:
        sections.append(f"# 获取估值指标出错：{e}\n")

    header = (
        f"# Company Fundamentals for {ticker}\n"
        f"# Source: Tushare Pro\n"
        f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    )
    return header + "\n".join(sections)


# ---------------------------------------------------------------------------
# 4. get_balance_sheet
# ---------------------------------------------------------------------------
def get_balance_sheet(
    ticker: Annotated[str, "ticker symbol of the company"],
    freq: Annotated[str, "frequency: 'annual' or 'quarterly'"] = "quarterly",
    curr_date: Annotated[str, "current date"] = None,
) -> str:
    """获取资产负债表（需要 2000+ 积分）。"""
    pro = _get_tushare_api()
    ts_code = to_tushare_format(ticker)

    limit = 4 if freq == "quarterly" else 2
    df = _safe_call(pro.balancesheet, ts_code=ts_code, limit=limit)

    if df is None or df.empty:
        return f"未找到股票 '{ticker}' 的资产负债表数据"

    csv_string = df.to_csv(index=False)
    header = (
        f"# Balance Sheet for {ticker} ({freq})\n"
        f"# Source: Tushare Pro\n"
        f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    )
    return header + csv_string


# ---------------------------------------------------------------------------
# 5. get_cashflow
# ---------------------------------------------------------------------------
def get_cashflow(
    ticker: Annotated[str, "ticker symbol of the company"],
    freq: Annotated[str, "frequency: 'annual' or 'quarterly'"] = "quarterly",
    curr_date: Annotated[str, "current date"] = None,
) -> str:
    """获取现金流量表（需要 2000+ 积分）。"""
    pro = _get_tushare_api()
    ts_code = to_tushare_format(ticker)

    limit = 4 if freq == "quarterly" else 2
    df = _safe_call(pro.cashflow, ts_code=ts_code, limit=limit)

    if df is None or df.empty:
        return f"未找到股票 '{ticker}' 的现金流量表数据"

    csv_string = df.to_csv(index=False)
    header = (
        f"# Cash Flow for {ticker} ({freq})\n"
        f"# Source: Tushare Pro\n"
        f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    )
    return header + csv_string


# ---------------------------------------------------------------------------
# 6. get_income_statement
# ---------------------------------------------------------------------------
def get_income_statement(
    ticker: Annotated[str, "ticker symbol of the company"],
    freq: Annotated[str, "frequency: 'annual' or 'quarterly'"] = "quarterly",
    curr_date: Annotated[str, "current date"] = None,
) -> str:
    """获取利润表（需要 2000+ 积分）。"""
    pro = _get_tushare_api()
    ts_code = to_tushare_format(ticker)

    limit = 4 if freq == "quarterly" else 2
    df = _safe_call(pro.income, ts_code=ts_code, limit=limit)

    if df is None or df.empty:
        return f"未找到股票 '{ticker}' 的利润表数据"

    csv_string = df.to_csv(index=False)
    header = (
        f"# Income Statement for {ticker} ({freq})\n"
        f"# Source: Tushare Pro\n"
        f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    )
    return header + csv_string


# ---------------------------------------------------------------------------
# 7. get_news
# ---------------------------------------------------------------------------
def get_news(
    ticker: Annotated[str, "ticker symbol of the company"],
    start_date: Annotated[str, "Start date yyyy-mm-dd"],
    end_date: Annotated[str, "End date yyyy-mm-dd"],
) -> str:
    """获取新闻（需要单独权限）。"""
    pro = _get_tushare_api()

    try:
        df = _safe_call(
            pro.news,
            src="sina",
            start_date=to_akshare_date(start_date),
            end_date=to_akshare_date(end_date),
            limit=20,
        )
    except (TushareUnavailableError, TushareRateLimitError):
        raise

    if df is None or df.empty:
        return f"未找到 {start_date} 至 {end_date} 期间的新闻"

    csv_string = df.to_csv(index=False)
    header = (
        f"# News (Tushare)\n"
        f"# Date range: {start_date} to {end_date}\n"
        f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    )
    return header + csv_string


# ---------------------------------------------------------------------------
# 8. get_global_news
# ---------------------------------------------------------------------------
def get_global_news(
    curr_date: Annotated[str, "current date yyyy-mm-dd"],
    look_back_days: Annotated[int, "days to look back"] = 7,
    limit: Annotated[int, "max articles"] = 50,
) -> str:
    """获取全局财经新闻（Tushare）。"""
    pro = _get_tushare_api()

    curr_dt = datetime.strptime(curr_date, "%Y-%m-%d")
    start_dt = curr_dt - timedelta(days=look_back_days)

    try:
        df = _safe_call(
            pro.news,
            src="sina",
            start_date=to_akshare_date(start_dt.strftime("%Y-%m-%d")),
            end_date=to_akshare_date(curr_date),
            limit=limit,
        )
    except (TushareUnavailableError, TushareRateLimitError):
        raise

    if df is None or df.empty:
        return "未找到全局财经新闻"

    csv_string = df.to_csv(index=False)
    header = (
        f"# Global Financial News\n"
        f"# Source: Tushare Pro\n"
        f"# Date range: {look_back_days} days before {curr_date}\n\n"
    )
    return header + csv_string


# ---------------------------------------------------------------------------
# 9. get_insider_transactions
# ---------------------------------------------------------------------------
def get_insider_transactions(
    symbol: Annotated[str, "ticker symbol of the company"],
) -> str:
    """获取大股东/董监高持股变动（需要 2000+ 积分）。"""
    pro = _get_tushare_api()
    ts_code = to_tushare_format(symbol)

    try:
        df = _safe_call(pro.stk_holdertrade, ts_code=ts_code, limit=20)
    except (TushareUnavailableError, TushareRateLimitError):
        raise

    if df is None or df.empty:
        return f"未找到股票 '{symbol}' 的内部交易数据"

    csv_string = df.to_csv(index=False)
    header = (
        f"# Insider Transactions for {symbol}\n"
        f"# Source: Tushare Pro\n"
        f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    )
    return header + csv_string
