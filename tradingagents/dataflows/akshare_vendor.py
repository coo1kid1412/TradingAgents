"""AKShare 数据供应商 —— 免费开源中国 A 股数据接口。

所有函数签名与 y_finance.py / alpha_vantage.py 保持一致，
返回格式化字符串供 LLM Agent 直接使用。
"""

from typing import Annotated
from datetime import datetime, timedelta

import pandas as pd

from .ticker_utils import to_akshare_format, to_akshare_report_format, to_akshare_date, to_standard_date
from .vendor_errors import AKShareError


def _import_akshare():
    """延迟导入 akshare，避免项目启动时的额外开销。"""
    try:
        import akshare as ak
        return ak
    except ImportError:
        raise AKShareError("akshare 未安装，请运行: pip install akshare")


# AKShare 中文列名 → 标准英文列名
_OHLCV_COL_MAP = {
    "日期": "Date",
    "开盘": "Open",
    "最高": "High",
    "最低": "Low",
    "收盘": "Close",
    "成交量": "Volume",
}


# ---------------------------------------------------------------------------
# 1. get_stock
# ---------------------------------------------------------------------------
def get_stock(
    symbol: Annotated[str, "ticker symbol of the company"],
    start_date: Annotated[str, "Start date in yyyy-mm-dd format"],
    end_date: Annotated[str, "End date in yyyy-mm-dd format"],
) -> str:
    """获取 A 股 OHLCV 日线数据（前复权）。"""
    ak = _import_akshare()
    code = to_akshare_format(symbol)

    try:
        df = ak.stock_zh_a_hist(
            symbol=code,
            period="daily",
            start_date=to_akshare_date(start_date),
            end_date=to_akshare_date(end_date),
            adjust="qfq",
        )
    except Exception as e:
        raise AKShareError(f"AKShare 获取 {symbol} 行情数据失败：{e}")

    if df is None or df.empty:
        return f"未找到股票 '{symbol}' 在 {start_date} 至 {end_date} 期间的数据"

    df = df.rename(columns=_OHLCV_COL_MAP)
    keep = [c for c in ("Date", "Open", "High", "Low", "Close", "Volume") if c in df.columns]
    df = df[keep]

    for col in ("Open", "High", "Low", "Close"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").round(2)

    csv_string = df.to_csv(index=False)

    header = (
        f"# Stock data for {symbol} from {start_date} to {end_date}\n"
        f"# Source: AKShare (东方财富)\n"
        f"# Total records: {len(df)}\n"
        f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    )
    return header + csv_string


# ---------------------------------------------------------------------------
# 2. get_indicator
# ---------------------------------------------------------------------------
def get_indicator(
    symbol: Annotated[str, "ticker symbol of the company"],
    indicator: Annotated[str, "technical indicator name (stockstats format)"],
    curr_date: Annotated[str, "current trading date YYYY-mm-dd"],
    look_back_days: Annotated[int, "how many days to look back"],
) -> str:
    """使用 AKShare OHLCV 数据 + stockstats 计算技术指标。"""
    from .stockstats_utils import calculate_indicator_from_ohlcv

    ak = _import_akshare()
    code = to_akshare_format(symbol)

    curr_dt = datetime.strptime(curr_date, "%Y-%m-%d")
    hist_start = curr_dt - timedelta(days=365)

    try:
        df = ak.stock_zh_a_hist(
            symbol=code,
            period="daily",
            start_date=to_akshare_date(hist_start.strftime("%Y-%m-%d")),
            end_date=to_akshare_date(curr_date),
            adjust="qfq",
        )
    except Exception as e:
        raise AKShareError(f"AKShare 获取 {symbol} 指标数据失败：{e}")

    if df is None or df.empty:
        return f"未找到股票 '{symbol}' 的历史行情数据，无法计算指标"

    df = df.rename(columns=_OHLCV_COL_MAP)
    indicator_data = calculate_indicator_from_ohlcv(df, indicator)

    before = curr_dt - timedelta(days=look_back_days)
    lines = []
    current_dt = curr_dt
    while current_dt >= before:
        ds = current_dt.strftime("%Y-%m-%d")
        val = indicator_data.get(ds, "N/A：非交易日（周末或节假日）")
        lines.append(f"{ds}: {val}")
        current_dt -= timedelta(days=1)

    return (
        f"## {indicator} values from {before.strftime('%Y-%m-%d')} to {curr_date}:\n"
        f"## Source: AKShare + stockstats\n\n"
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
    """获取 A 股公司基本面信息。"""
    ak = _import_akshare()
    code = to_akshare_format(ticker)
    sections: list[str] = []

    # 公司基本信息
    try:
        info_df = ak.stock_individual_info_em(symbol=code)
        if info_df is not None and not info_df.empty:
            sections.append("## 公司基本信息")
            for _, row in info_df.iterrows():
                item = row.iloc[0] if len(row) > 0 else ""
                value = row.iloc[1] if len(row) > 1 else ""
                sections.append(f"{item}: {value}")
            sections.append("")
    except Exception as e:
        sections.append(f"# 获取公司信息出错：{e}\n")

    # 财务分析指标
    try:
        fa_df = ak.stock_financial_analysis_indicator(symbol=code)
        if fa_df is not None and not fa_df.empty:
            sections.append("## 财务分析指标（最近3期）")
            sections.append(fa_df.head(3).to_csv(index=False))
    except Exception as e:
        sections.append(f"# 获取财务分析指标出错：{e}\n")

    header = (
        f"# Company Fundamentals for {ticker}\n"
        f"# Source: AKShare (东方财富)\n"
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
    """获取 A 股资产负债表。"""
    ak = _import_akshare()
    code = to_akshare_format(ticker)

    try:
        df = ak.stock_balance_sheet_by_report_em(symbol=code)
    except Exception as e:
        raise AKShareError(f"AKShare 获取 {ticker} 资产负债表失败：{e}")

    if df is None or df.empty:
        return f"未找到股票 '{ticker}' 的资产负债表数据"

    limit = 4 if freq == "quarterly" else 2
    csv_string = df.head(limit).to_csv(index=False)

    header = (
        f"# Balance Sheet for {ticker} ({freq})\n"
        f"# Source: AKShare (东方财富)\n"
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
    """获取 A 股现金流量表。"""
    ak = _import_akshare()
    code = to_akshare_format(ticker)

    try:
        df = ak.stock_cash_flow_sheet_by_report_em(symbol=code)
    except Exception as e:
        raise AKShareError(f"AKShare 获取 {ticker} 现金流量表失败：{e}")

    if df is None or df.empty:
        return f"未找到股票 '{ticker}' 的现金流量表数据"

    limit = 4 if freq == "quarterly" else 2
    csv_string = df.head(limit).to_csv(index=False)

    header = (
        f"# Cash Flow for {ticker} ({freq})\n"
        f"# Source: AKShare (东方财富)\n"
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
    """获取 A 股利润表。"""
    ak = _import_akshare()
    code = to_akshare_report_format(ticker)

    try:
        df = ak.stock_profit_sheet_by_report_em(symbol=code)
    except Exception as e:
        raise AKShareError(f"AKShare 获取 {ticker} 利润表失败：{e}")

    if df is None or df.empty:
        return f"未找到股票 '{ticker}' 的利润表数据"

    limit = 4 if freq == "quarterly" else 2
    csv_string = df.head(limit).to_csv(index=False)

    header = (
        f"# Income Statement for {ticker} ({freq})\n"
        f"# Source: AKShare (东方财富)\n"
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
    """获取 A 股个股新闻。"""
    ak = _import_akshare()
    code = to_akshare_format(ticker)

    try:
        df = ak.stock_news_em(symbol=code)
    except Exception as e:
        raise AKShareError(f"AKShare 获取 {ticker} 新闻失败：{e}")

    if df is None or df.empty:
        return f"未找到股票 '{ticker}' 的相关新闻"

    df = df.head(20)
    csv_string = df.to_csv(index=False)

    header = (
        f"# News for {ticker}\n"
        f"# Source: AKShare (东方财富)\n"
        f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    )
    return header + csv_string


# ---------------------------------------------------------------------------
# 8. get_global_news
# ---------------------------------------------------------------------------
def get_global_news(
    curr_date: Annotated[str, "current date yyyy-mm-dd"],
    look_back_days: Annotated[int, "how many days to look back"] = 7,
    limit: Annotated[int, "max number of articles"] = 50,
) -> str:
    """获取中国财经全局新闻。"""
    ak = _import_akshare()

    df = None
    for fetch_fn in ("stock_info_global_em", "news_economic_baidu"):
        try:
            df = getattr(ak, fetch_fn)()
            if df is not None and not df.empty:
                break
        except Exception:
            continue

    if df is None or df.empty:
        raise AKShareError("AKShare 获取全局新闻失败：所有新闻接口均无数据")

    csv_string = df.head(limit).to_csv(index=False)

    header = (
        f"# Global Financial News (China)\n"
        f"# Source: AKShare\n"
        f"# Date range: {look_back_days} days before {curr_date}\n"
        f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    )
    return header + csv_string


# ---------------------------------------------------------------------------
# 9. get_insider_transactions
# ---------------------------------------------------------------------------
def get_insider_transactions(
    symbol: Annotated[str, "ticker symbol of the company"],
) -> str:
    """获取 A 股董监高/大股东持股变动。"""
    ak = _import_akshare()
    code = to_akshare_format(symbol)

    # 尝试多个可能的 AKShare 接口
    df = None
    tried_apis = []
    for api_name in ("stock_gdfx_free_holding_change_em", "stock_inner_trade_xq"):
        tried_apis.append(api_name)
        try:
            fn = getattr(ak, api_name, None)
            if fn is None:
                continue
            df = fn(symbol=code)
            if df is not None and not df.empty:
                break
        except Exception:
            continue

    if df is None or df.empty:
        return (
            f"# Insider Transactions for {symbol}\n"
            f"# Source: AKShare\n\n"
            f"AKShare 暂无 {symbol} 的内部交易数据，"
            f"建议通过 Tushare（stk_holdertrade 接口）获取。"
        )

    csv_string = df.head(20).to_csv(index=False)

    header = (
        f"# Insider Transactions for {symbol}\n"
        f"# Source: AKShare\n"
        f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    )
    return header + csv_string
