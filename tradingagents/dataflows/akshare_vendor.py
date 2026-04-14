"""AKShare 数据供应商 —— 免费开源中国 A 股数据接口。

所有函数签名与 y_finance.py / alpha_vantage.py 保持一致，
返回格式化字符串供 LLM Agent 直接使用。
"""

import logging
from typing import Annotated
from datetime import datetime, timedelta

import pandas as pd

from .ticker_utils import to_akshare_format, to_akshare_report_format, to_akshare_date, to_standard_date, is_etf_or_lof, _get_exchange
from .vendor_errors import AKShareError

logger = logging.getLogger(__name__)


def _import_akshare():
    """延迟导入 akshare，避免项目启动时的额外开销。"""
    try:
        import akshare as ak
        return ak
    except ImportError:
        raise AKShareError("akshare 未安装，请运行: pip install akshare")


# AKShare 中文列名 → 标准英文列名（东方财富源 stock_zh_a_hist / fund_etf_hist_em）
_OHLCV_COL_MAP = {
    "日期": "Date",
    "开盘": "Open",
    "最高": "High",
    "最低": "Low",
    "收盘": "Close",
    "成交量": "Volume",
}

# 新浪源 stock_zh_a_daily 已使用小写英文列名，映射为首字母大写
_OHLCV_COL_MAP_SINA = {
    "date": "Date",
    "open": "Open",
    "high": "High",
    "low": "Low",
    "close": "Close",
    "volume": "Volume",
}


def _get_ohlcv(ak, code: str, start_date: str, end_date: str) -> pd.DataFrame:
    """获取 A 股 OHLCV 数据，自动适配股票和 ETF/LOF。

    尝试顺序：
      1. stock_zh_a_hist  — 东方财富源，覆盖股票（6/0/3/688 等）
      2. stock_zh_a_daily — 新浪源 fallback（东方财富不可用时）
      3. fund_etf_hist_em — 东方财富源，覆盖 ETF（5xxxxx, 159xxx 等）
      4. fund_etf_hist_sina — 新浪源 ETF fallback

    Args:
        ak: akshare 模块
        code: 纯6位数字代码
        start_date: YYYYMMDD
        end_date: YYYYMMDD

    Returns:
        DataFrame with 标准英文列名 (Date, Open, High, Low, Close, Volume)，
        如果所有接口均无数据则返回空 DataFrame。
    """
    # 1. 先尝试东方财富行情接口（覆盖大部分标的）
    try:
        df = ak.stock_zh_a_hist(
            symbol=code,
            period="daily",
            start_date=start_date,
            end_date=end_date,
            adjust="qfq",
        )
        if df is not None and not df.empty:
            return df.rename(columns=_OHLCV_COL_MAP)
        else:
            logger.debug("AKShare stock_zh_a_hist(%s) 返回空数据", code)
    except Exception as e:
        logger.warning("AKShare stock_zh_a_hist(%s) 失败: %s", code, e)

    # 2. 新浪源 fallback（stock_zh_a_daily 需要 sh/sz 前缀格式）
    try:
        exchange = _get_exchange(code).lower()  # "SH" → "sh", "SZ" → "sz"
        sina_symbol = f"{exchange}{code}"
        df = ak.stock_zh_a_daily(
            symbol=sina_symbol,
            start_date=start_date,
            end_date=end_date,
            adjust="qfq",
        )
        if df is not None and not df.empty:
            return df.rename(columns=_OHLCV_COL_MAP_SINA)
        else:
            logger.debug("AKShare stock_zh_a_daily(%s) 返回空数据", sina_symbol)
    except Exception as e:
        logger.warning("AKShare stock_zh_a_daily(%s) 失败: %s", sina_symbol, e)

    # 3. ETF 行情接口（东方财富源）— stock_zh_a_hist 对 ETF 常返回空
    try:
        df = ak.fund_etf_hist_em(
            symbol=code,
            period="daily",
            start_date=start_date,
            end_date=end_date,
            adjust="qfq",
        )
        if df is not None and not df.empty:
            return df.rename(columns=_OHLCV_COL_MAP)
        else:
            logger.debug("AKShare fund_etf_hist_em(%s) 返回空数据", code)
    except Exception as e:
        logger.warning("AKShare fund_etf_hist_em(%s) 失败: %s", code, e)

    # 4. ETF 新浪源 fallback（fund_etf_hist_sina 需要 sh/sz 前缀，不支持日期范围）
    try:
        exchange = _get_exchange(code).lower()
        sina_symbol = f"{exchange}{code}"
        df = ak.fund_etf_hist_sina(symbol=sina_symbol)
        if df is not None and not df.empty:
            # fund_etf_hist_sina 返回全量历史，按日期范围过滤
            df = df.rename(columns=_OHLCV_COL_MAP_SINA)
            if "Date" in df.columns:
                df["Date"] = pd.to_datetime(df["Date"])
                start_dt = pd.to_datetime(start_date, format="%Y%m%d")
                end_dt = pd.to_datetime(end_date, format="%Y%m%d")
                df = df[(df["Date"] >= start_dt) & (df["Date"] <= end_dt)]
                df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")
            if not df.empty:
                return df
        else:
            logger.debug("AKShare fund_etf_hist_sina(%s) 返回空数据", sina_symbol)
    except Exception as e:
        logger.warning("AKShare fund_etf_hist_sina(%s) 失败: %s", sina_symbol, e)

    logger.error("AKShare _get_ohlcv(%s) 所有接口均失败，返回空 DataFrame", code)
    return pd.DataFrame()


# ---------------------------------------------------------------------------
# 1. get_stock
# ---------------------------------------------------------------------------
def get_stock(
    symbol: Annotated[str, "ticker symbol of the company"],
    start_date: Annotated[str, "Start date in yyyy-mm-dd format"],
    end_date: Annotated[str, "End date in yyyy-mm-dd format"],
) -> str:
    """获取 A 股 OHLCV 日线数据（前复权），支持股票和 ETF/LOF。"""
    ak = _import_akshare()
    code = to_akshare_format(symbol)

    df = _get_ohlcv(ak, code, to_akshare_date(start_date), to_akshare_date(end_date))

    if df.empty:
        return f"未找到股票 '{symbol}' 在 {start_date} 至 {end_date} 期间的数据"

    keep = [c for c in ("Date", "Open", "High", "Low", "Close", "Volume") if c in df.columns]
    df = df[keep]

    for col in ("Open", "High", "Low", "Close"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").round(2)

    csv_string = df.to_csv(index=False)

    # 显示实际返回的数据范围，而非请求的数据范围
    actual_start = df["Date"].iloc[0] if "Date" in df.columns else start_date
    actual_end = df["Date"].iloc[-1] if "Date" in df.columns else end_date

    header = (
        f"# Stock data for {symbol}\n"
        f"# Actual date range: {actual_start} to {actual_end} "
        f"(requested: {start_date} to {end_date})\n"
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
    """使用 AKShare OHLCV 数据 + stockstats 计算技术指标，支持股票和 ETF/LOF。"""
    from .stockstats_utils import calculate_indicator_from_ohlcv

    ak = _import_akshare()
    code = to_akshare_format(symbol)

    curr_dt = datetime.strptime(curr_date, "%Y-%m-%d")
    hist_start = curr_dt - timedelta(days=365)

    df = _get_ohlcv(
        ak, code,
        to_akshare_date(hist_start.strftime("%Y-%m-%d")),
        to_akshare_date(curr_date),
    )

    if df.empty:
        return f"未找到股票 '{symbol}' 的历史行情数据，无法计算指标"
    indicator_data = calculate_indicator_from_ohlcv(df, indicator)

    # 记录股票最早有数据的日期，用于区分"非交易日"和"尚未上市"
    first_available_date = df["Date"].min() if "Date" in df.columns and not df.empty else None

    before = curr_dt - timedelta(days=look_back_days)
    lines = []
    current_dt = curr_dt
    while current_dt >= before:
        ds = current_dt.strftime("%Y-%m-%d")
        if ds in indicator_data:
            val = indicator_data[ds]
        elif first_available_date and ds < first_available_date:
            val = "N/A：股票尚未上市"
        else:
            val = "N/A：非交易日（周末或节假日）"
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
    """获取 A 股公司基本面信息。ETF/LOF 走专属数据路径。"""
    ak = _import_akshare()
    code = to_akshare_format(ticker)

    if is_etf_or_lof(ticker):
        return _get_etf_fundamentals(ak, code, ticker)

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


def _get_etf_fundamentals(ak, code: str, ticker: str) -> str:
    """获取 ETF/LOF 基金基本面信息（fund_etf_spot_em 实时数据）。"""
    sections: list[str] = []

    try:
        spot_df = ak.fund_etf_spot_em()
        row = spot_df[spot_df["代码"] == code]
        if row.empty:
            # 尝试 LOF
            try:
                lof_df = ak.fund_lof_spot_em()
                row = lof_df[lof_df["代码"] == code]
            except Exception:
                pass

        if not row.empty:
            r = row.iloc[0]
            sections.append("## 基金实时行情")
            field_map = [
                ("名称", "名称"),
                ("最新价(元)", "最新价"),
                ("IOPV实时估值", "IOPV实时估值"),
                ("基金折价率(%)", "基金折价率"),
                ("涨跌额", "涨跌额"),
                ("涨跌幅(%)", "涨跌幅"),
                ("成交量(手)", "成交量"),
                ("成交额(元)", "成交额"),
                ("开盘价", "开盘价"),
                ("最高价", "最高价"),
                ("最低价", "最低价"),
                ("昨收", "昨收"),
                ("振幅(%)", "振幅"),
                ("换手率(%)", "换手率"),
                ("最新份额", "最新份额"),
                ("流通市值(元)", "流通市值"),
                ("总市值(元)", "总市值"),
            ]
            for label, col in field_map:
                if col in r.index and pd.notna(r[col]):
                    sections.append(f"{label}: {r[col]}")

            sections.append("")
            sections.append("## 资金流向")
            flow_fields = [
                ("主力净流入-净额", "主力净流入-净额"),
                ("主力净流入-净占比(%)", "主力净流入-净占比"),
                ("超大单净流入-净额", "超大单净流入-净额"),
                ("超大单净流入-净占比(%)", "超大单净流入-净占比"),
                ("大单净流入-净额", "大单净流入-净额"),
                ("大单净流入-净占比(%)", "大单净流入-净占比"),
                ("中单净流入-净额", "中单净流入-净额"),
                ("中单净流入-净占比(%)", "中单净流入-净占比"),
                ("小单净流入-净额", "小单净流入-净额"),
                ("小单净流入-净占比(%)", "小单净流入-净占比"),
            ]
            for label, col in flow_fields:
                if col in r.index and pd.notna(r[col]):
                    sections.append(f"{label}: {r[col]}")
        else:
            sections.append("未在 ETF/LOF 实时行情中找到该基金")
    except Exception as e:
        sections.append(f"# 获取 ETF 实时行情出错：{e}\n")

    header = (
        f"# ETF/LOF Fund Fundamentals for {ticker}\n"
        f"# Source: AKShare (东方财富 ETF 实时行情)\n"
        f"# 注意：ETF/LOF 是被动跟踪指数的基金产品，没有传统上市公司的财务报表\n"
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
    if is_etf_or_lof(ticker):
        return (
            f"# Balance Sheet for {ticker}\n\n"
            f"{ticker} 是 ETF/LOF 基金，不是上市公司，没有资产负债表。\n"
            f"如需了解该基金的基本信息，请使用 get_fundamentals 工具。"
        )

    ak = _import_akshare()
    code = to_akshare_report_format(ticker)

    try:
        df = ak.stock_balance_sheet_by_report_em(symbol=code)
    except Exception as e:
        return f"# Balance Sheet for {ticker}\n\nAKShare 获取资产负债表失败：{e}"

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
    if is_etf_or_lof(ticker):
        return (
            f"# Cash Flow for {ticker}\n\n"
            f"{ticker} 是 ETF/LOF 基金，不是上市公司，没有现金流量表。\n"
            f"如需了解该基金的基本信息，请使用 get_fundamentals 工具。"
        )

    ak = _import_akshare()
    code = to_akshare_report_format(ticker)

    try:
        df = ak.stock_cash_flow_sheet_by_report_em(symbol=code)
    except Exception as e:
        return f"# Cash Flow for {ticker}\n\nAKShare 获取现金流量表失败：{e}"

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
    if is_etf_or_lof(ticker):
        return (
            f"# Income Statement for {ticker}\n\n"
            f"{ticker} 是 ETF/LOF 基金，不是上市公司，没有利润表。\n"
            f"如需了解该基金的基本信息，请使用 get_fundamentals 工具。"
        )

    ak = _import_akshare()
    code = to_akshare_report_format(ticker)

    try:
        df = ak.stock_profit_sheet_by_report_em(symbol=code)
    except Exception as e:
        return f"# Income Statement for {ticker}\n\nAKShare 获取利润表失败：{e}"

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
    if is_etf_or_lof(symbol):
        return (
            f"# Insider Transactions for {symbol}\n\n"
            f"{symbol} 是 ETF/LOF 基金，没有董监高/内部交易数据。"
        )

    ak = _import_akshare()
    code = to_akshare_format(symbol)
    # 构建带交易所前缀的代码，用于匹配 stock_inner_trade_xq 返回的格式
    from .ticker_utils import _get_exchange
    prefixed_code = f"{_get_exchange(code)}{code}"

    # stock_inner_trade_xq() 返回全市场近期内部交易，按代码过滤
    df = None
    try:
        all_df = ak.stock_inner_trade_xq()
        if all_df is not None and not all_df.empty and "股票代码" in all_df.columns:
            df = all_df[all_df["股票代码"] == prefixed_code]
    except Exception:
        pass

    if df is None or df.empty:
        return (
            f"# Insider Transactions for {symbol}\n"
            f"# Source: AKShare (雪球)\n\n"
            f"近期暂无 {symbol} 的董监高/内部交易数据。"
        )

    csv_string = df.head(20).to_csv(index=False)

    header = (
        f"# Insider Transactions for {symbol}\n"
        f"# Source: AKShare (雪球)\n"
        f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    )
    return header + csv_string
    return header + csv_string


# ---------------------------------------------------------------------------
# 10. get_announcements
# ---------------------------------------------------------------------------
def get_announcements(
    ticker: Annotated[str, "ticker symbol of the company"],
    start_date: Annotated[str, "Start date yyyy-mm-dd"],
    end_date: Annotated[str, "End date yyyy-mm-dd"],
) -> str:
    """获取 A 股公司公告（来源：巨潮资讯）。"""
    if is_etf_or_lof(ticker):
        return (
            f"# Company Announcements for {ticker}\n\n"
            f"{ticker} 是 ETF/LOF 基金，巨潮资讯暂不支持查询 ETF 公告。\n"
            f"可通过 get_news 工具获取该基金的相关新闻。"
        )

    ak = _import_akshare()
    code = to_akshare_format(ticker)

    try:
        df = ak.stock_zh_a_disclosure_report_cninfo(
            symbol=code,
            market="沪深京",
            start_date=to_akshare_date(start_date),
            end_date=to_akshare_date(end_date),
        )
    except Exception as e:
        return (
            f"# Company Announcements for {ticker}\n\n"
            f"AKShare 获取公告数据失败：{e}"
        )

    if df is None or df.empty:
        return (
            f"# Company Announcements for {ticker}\n"
            f"# Source: AKShare (巨潮资讯)\n\n"
            f"未找到 {ticker} 在 {start_date} 至 {end_date} 期间的公告"
        )

    df = df.head(30)
    csv_string = df.to_csv(index=False)

    header = (
        f"# Company Announcements for {ticker}\n"
        f"# Source: AKShare (巨潮资讯)\n"
        f"# Date range: {start_date} to {end_date}\n"
        f"# Total results: {len(df)}\n"
        f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    )
    return header + csv_string


# ---------------------------------------------------------------------------
# 11. get_cls_telegraph
# ---------------------------------------------------------------------------
def get_cls_telegraph(
    curr_date: Annotated[str, "current date yyyy-mm-dd"],
    limit: Annotated[int, "max number of telegraphs"] = 30,
) -> str:
    """获取财联社电报快讯。"""
    ak = _import_akshare()

    try:
        df = ak.stock_info_global_cls(symbol="全部")
    except Exception as e:
        raise AKShareError(f"AKShare 获取财联社电报失败：{e}")

    if df is None or df.empty:
        return (
            f"# CLS Telegraph (财联社电报)\n"
            f"# Date: {curr_date}\n\n"
            f"暂无财联社电报数据"
        )

    df = df.head(limit)
    csv_string = df.to_csv(index=False)

    header = (
        f"# CLS Telegraph (财联社电报)\n"
        f"# Source: AKShare (财联社)\n"
        f"# Total results: {len(df)}\n"
        f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    )
    return header + csv_string


# ---------------------------------------------------------------------------
# 12. get_research_reports
# ---------------------------------------------------------------------------
def get_research_reports(
    ticker: Annotated[str, "ticker symbol of the company"],
    limit: Annotated[int, "max number of reports"] = 20,
) -> str:
    """获取个股研报（来源：东方财富）。"""
    if is_etf_or_lof(ticker):
        return (
            f"# Research Reports for {ticker}\n\n"
            f"{ticker} 是 ETF/LOF 基金，东方财富暂不提供 ETF 个股研报。\n"
            f"可通过 get_news 工具获取该基金的相关分析文章。"
        )

    ak = _import_akshare()
    code = to_akshare_format(ticker)

    try:
        df = ak.stock_research_report_em(symbol=code)
    except Exception as e:
        return (
            f"# Research Reports for {ticker}\n\n"
            f"AKShare 获取研报数据失败：{e}"
        )

    if df is None or df.empty:
        return (
            f"# Research Reports for {ticker}\n"
            f"# Source: AKShare (东方财富)\n\n"
            f"未找到 {ticker} 的研报数据"
        )

    df = df.head(limit)
    # 选择关键列，减少 token 消耗
    keep_cols = [c for c in (
        "股票代码", "股票简称", "报告名称", "东财评级", "机构",
        "2025-盈利预测-收益", "2025-盈利预测-市盈率",
        "2026-盈利预测-收益", "2026-盈利预测-市盈率",
        "行业", "日期",
    ) if c in df.columns]
    if keep_cols:
        df = df[keep_cols]
    csv_string = df.to_csv(index=False)

    header = (
        f"# Research Reports for {ticker}\n"
        f"# Source: AKShare (东方财富)\n"
        f"# Total results: {len(df)}\n"
        f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    )
    return header + csv_string
