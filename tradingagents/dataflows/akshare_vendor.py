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
from .financial_field_maps import (
    extract_and_format,
    AKSHARE_FUNDAMENTALS_MAP,
    AKSHARE_BALANCE_SHEET_MAP,
    AKSHARE_CASHFLOW_MAP,
    AKSHARE_INCOME_MAP,
)
from .valuation_utils import (
    compute_ttm_eps,
    compute_ttm_revenue_per_share,
    compute_valuation_metrics,
)

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
      1. stock_zh_a_daily — 新浪源（优先：不依赖 JS 解密，无 V8 崩溃风险）
      2. stock_zh_a_hist  — 东方财富源 fallback（需 JS 解密，偶发 py_mini_racer 崩溃）
      3. fund_etf_hist_sina — 新浪源 ETF fallback
      4. fund_etf_hist_em — 东方财富源 ETF fallback

    Args:
        ak: akshare 模块
        code: 纯6位数字代码
        start_date: YYYYMMDD
        end_date: YYYYMMDD

    Returns:
        DataFrame with 标准英文列名 (Date, Open, High, Low, Close, Volume)，
        如果所有接口均无数据则返回空 DataFrame。
    """
    exchange = _get_exchange(code).lower()  # "SH" → "sh", "SZ" → "sz"
    sina_symbol = f"{exchange}{code}"

    # 1. 新浪源（优先：不依赖 JS 解密，无 py_mini_racer V8 崩溃风险）
    try:
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

    # 2. 东方财富源 fallback（需 JS 解密，偶发 py_mini_racer V8 崩溃）
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

    # 3. ETF 新浪源 fallback（fund_etf_hist_sina 不支持日期范围，返回全量后过滤）
    try:
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

    # 4. ETF 东方财富源 fallback
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
        f"# Source: AKShare\n"
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
    # 获取 look_back_days 天的数据（加上一些缓冲用于指标计算），而不是固定 365 天
    hist_start = curr_dt - timedelta(days=look_back_days + 60)

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
    report_code = to_akshare_report_format(ticker)  # 财报接口需要 SH/SZ 前缀

    if is_etf_or_lof(ticker):
        return _get_etf_fundamentals(ak, code, ticker)

    sections: list[str] = []
    info_df = None      # 公司基本信息 DataFrame（EM源）
    fa_df = None        # 财务分析指标 DataFrame（Sina源）
    income_df = None    # 利润表 DataFrame（EM源）

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

    # 财务分析指标（Sina源，需指定 start_year 否则默认1900会返回空）
    try:
        start_year = str(max(datetime.now().year - 5, 2020))
        fa_df = ak.stock_financial_analysis_indicator(symbol=code, start_year=start_year)
        if fa_df is not None and not fa_df.empty:
            # 显示用排序副本（降序取最新3期），fa_df 保持原始升序供估值计算使用
            sections.append("## 财务分析指标（最近3期）")
            sections.append(extract_and_format(
                fa_df.sort_values("日期", ascending=False),
                AKSHARE_FUNDAMENTALS_MAP, period_col="日期", limit=3,
            ))
    except Exception as e:
        logger.warning("AKShare stock_financial_analysis_indicator(%s) 失败: %s", code, e)

    # 利润表（EM源）—— 始终获取，用于 PS(TTM) 计算和 fa_df 为空时的 fallback
    # 注意：stock_profit_sheet_by_report_em 需要 SH/SZ 前缀格式（如 SH688008），
    # 传纯6位代码会导致 AKShare 内部 NoneType 错误
    income_df = None  # 确保始终有定义
    try:
        income_result = ak.stock_profit_sheet_by_report_em(symbol=report_code)
        # AKShare 某些接口可能返回 None 而非 DataFrame
        if income_result is not None and hasattr(income_result, 'empty') and not income_result.empty:
            income_df = income_result
            # 如果 fa_df 为空，则用 income_df 显示财务指标
            if fa_df is None or fa_df.empty:
                sections.append("## 财务指标（最近3期，源自利润表）")
                sections.append(
                    extract_and_format(income_df, AKSHARE_INCOME_MAP, period_col="REPORT_DATE", limit=3)
                )
            else:
                # fa_df 有数据，income_df 仅用于 PS(TTM) 计算，不重复显示
                logger.debug("AKShare 利润表数据已获取，用于 PS(TTM) 计算")
        else:
            logger.warning("AKShare stock_profit_sheet_by_report_em(%s) 返回空数据或 None", report_code)
    except Exception as e:
        logger.warning("AKShare stock_profit_sheet_by_report_em(%s) 失败: %s", report_code, e)

    # --- 估值指标（系统自行计算） ---
    _append_valuation_section(sections, ak, ticker, code, curr_date, info_df, fa_df, income_df)

    header = (
        f"# Company Fundamentals for {ticker}\n"
        f"# Source: AKShare (东方财富)\n"
        f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    )
    return header + "\n".join(sections)


def _safe_float(val) -> float | None:
    """Safely convert a value to positive float, returning None on failure."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    try:
        n = float(val)
        return n if n > 0 else None
    except (ValueError, TypeError):
        return None


def _extract_from_info_df(info_df: pd.DataFrame | None, item_name: str) -> float | None:
    """从 stock_individual_info_em 的 DataFrame 中提取指定项目的值。"""
    if info_df is None or info_df.empty:
        return None
    for _, row in info_df.iterrows():
        item = str(row.iloc[0]).strip() if len(row) > 0 else ""
        if item == item_name:
            value = row.iloc[1] if len(row) > 1 else None
            return _safe_float(value)
    return None


def _get_shares_from_cninfo(ak, code: str, target_date: str | None) -> tuple[float | None, float | None]:
    """从巨潮（cninfo）股本变动接口拿总股本/流通股，作为东财 push2 失败时的独立兜底源。

    巨潮服务器与东财/新浪是完全不同的源，遇到东财 RemoteDisconnected 时仍可用。

    Args:
        ak: akshare module
        code: 不带前缀的股票代码（如 "603516"）
        target_date: 目标日期 "YYYY-MM-DD" 或 None（用今天）

    Returns:
        (total_shares, outstanding_shares) tuple，单位：股。失败时返回 (None, None)。
    """
    try:
        end_dt = target_date or datetime.now().strftime("%Y-%m-%d")
        # 兼容传入 YYYY-MM-DD 或 YYYYMMDD
        end_obj = datetime.strptime(end_dt.replace("-", ""), "%Y%m%d")
        # 取近 365 天的股本变动记录（覆盖最近一次定期报告披露）
        start_str = (end_obj - timedelta(days=365)).strftime("%Y%m%d")
        end_str = end_obj.strftime("%Y%m%d")

        df = ak.stock_share_change_cninfo(symbol=code, start_date=start_str, end_date=end_str)
        if df is None or df.empty:
            return None, None

        # 按变动日期降序取最新一条（最近一次披露的股本结构）
        if "变动日期" in df.columns:
            df = df.sort_values("变动日期", ascending=False)
        latest = df.iloc[0]

        # 字段单位：万股 → 转换为股
        total_wan = latest.get("总股本", None)
        circ_wan = latest.get("已流通股份", None)
        total_shares = float(total_wan) * 1e4 if total_wan is not None and not pd.isna(total_wan) else None
        outstanding_shares = float(circ_wan) * 1e4 if circ_wan is not None and not pd.isna(circ_wan) else None

        if total_shares is not None:
            logger.info(
                "巨潮 cninfo 兜底成功: %s 总股本=%.4f万股, 已流通=%.4f万股 (变动日期=%s)",
                code, total_shares / 1e4, (outstanding_shares or 0) / 1e4,
                latest.get("变动日期", "N/A"),
            )
        return total_shares, outstanding_shares
    except Exception as e:
        logger.debug("巨潮 stock_share_change_cninfo 也失败: %s", e)
        return None, None


def _append_valuation_section(
    sections: list[str],
    ak,
    ticker: str,
    code: str,
    curr_date: str | None,
    info_df: pd.DataFrame | None,
    fa_df: pd.DataFrame | None,
    income_df: pd.DataFrame | None,
):
    """从已有财务数据 + 行情数据自行计算估值指标，补足 AKShare 缺失的 PE/PB/PS/市值。

    数据获取优先级:
      - 收盘价: Sina OHLCV（最稳定）
      - 总股本/流通股: stock_individual_info_em（EM源，已在公司信息中获取）
      - EPS/BPS: fa_df（Sina源）> income_df（EM源）
      - PS: 需 income_df + 总股本
    """
    try:
        # --- 1. 获取收盘价（优先 Sina 源，最稳定） ---
        close_price = None
        try:
            end_dt = curr_date if curr_date else datetime.now().strftime("%Y-%m-%d")
            start_dt = (datetime.strptime(end_dt, "%Y-%m-%d") - timedelta(days=10)).strftime("%Y%m%d")
            end_ymd = to_akshare_date(end_dt)
            ohlcv = _get_ohlcv(ak, code, start_dt, end_ymd)
            if not ohlcv.empty and "Close" in ohlcv.columns:
                close_price = float(ohlcv["Close"].iloc[-1])
        except Exception as e:
            logger.debug("AKShare 获取收盘价失败: %s", e)

        if close_price is None:
            sections.append("## 估值指标（系统计算）")
            sections.append("无法获取收盘价，跳过估值计算")
            sections.append("")
            return

        # --- 2. 获取总股本和流通股本 ---
        total_shares = _extract_from_info_df(info_df, "总股本")
        outstanding_shares = _extract_from_info_df(info_df, "流通股")

        # 如果 info_df 没有总股本但有总市值，从总市值反算
        if total_shares is None:
            total_mv = _extract_from_info_df(info_df, "总市值")
            if total_mv is not None:
                total_shares = total_mv / close_price

        if outstanding_shares is None:
            circ_mv = _extract_from_info_df(info_df, "流通市值")
            if circ_mv is not None:
                outstanding_shares = circ_mv / close_price

        # 兜底：东财 push2 失败时，走巨潮（cninfo）独立源拿股本数据
        # 历史上重试同接口（同源 push2）几乎必失败，已改为换源
        if total_shares is None or outstanding_shares is None:
            cninfo_total, cninfo_circ = _get_shares_from_cninfo(ak, code, curr_date)
            if total_shares is None and cninfo_total is not None:
                total_shares = cninfo_total
            if outstanding_shares is None and cninfo_circ is not None:
                outstanding_shares = cninfo_circ

        # --- 3. 计算年度 EPS ---
        annual_eps = None
        # 3a. 从 fa_df（Sina 财务指标）
        if fa_df is not None and not fa_df.empty and "日期" in fa_df.columns:
            eps_col = None
            for col in fa_df.columns:
                if "每股收益" in str(col) and "摊薄" in str(col):
                    eps_col = col
                    break
            if eps_col is None:
                for col in fa_df.columns:
                    if "加权每股收益" in str(col):
                        eps_col = col
                        break
            if eps_col:
                annual_mask = fa_df["日期"].astype(str).str.replace("-", "", regex=False).str.endswith("1231")
                annual_rows = fa_df[annual_mask]
                if not annual_rows.empty:
                    eps_val = annual_rows.iloc[-1].get(eps_col, None)
                    if eps_val is not None and not pd.isna(eps_val):
                        annual_eps = float(eps_val)
        # 3b. 从 income_df（EM 利润表）
        if annual_eps is None and income_df is not None and not income_df.empty:
            if "BASIC_EPS" in income_df.columns and "REPORT_DATE" in income_df.columns:
                sorted_df = income_df.sort_values("REPORT_DATE")
                annual_mask = sorted_df["REPORT_DATE"].astype(str).str[:10].str.replace("-", "", regex=False).str.endswith("1231")
                annual_rows = sorted_df[annual_mask]
                if not annual_rows.empty:
                    eps_val = annual_rows.iloc[-1].get("BASIC_EPS", None)
                    if eps_val is not None and not pd.isna(eps_val):
                        annual_eps = float(eps_val)

        # --- 4. 计算 TTM EPS ---
        ttm_eps = None
        # 4a. 从 fa_df
        if fa_df is not None and not fa_df.empty:
            eps_col = None
            for col in fa_df.columns:
                if "每股收益" in str(col) and "摊薄" in str(col):
                    eps_col = col
                    break
            if eps_col is None:
                for col in fa_df.columns:
                    if "加权每股收益" in str(col):
                        eps_col = col
                        break
            if eps_col:
                ttm_eps = compute_ttm_eps(fa_df, eps_col=eps_col, date_col="日期")
        # 4b. 从 income_df
        if ttm_eps is None and income_df is not None and not income_df.empty:
            if "BASIC_EPS" in income_df.columns:
                ttm_eps = compute_ttm_eps(income_df, eps_col="BASIC_EPS", date_col="REPORT_DATE")

        # --- 5. 获取 BPS（每股净资产） ---
        bps = None
        # 5a. 从 fa_df（Sina 财务指标，优先"调整后"版本）
        if fa_df is not None and not fa_df.empty:
            bps_col = None
            # 优先: 每股净资产_调整后(元)
            for col in fa_df.columns:
                if "每股净资产" in str(col) and "调整后" in str(col):
                    bps_col = col
                    break
            # 次选: 任意含 每股净资产 的列
            if bps_col is None:
                for col in fa_df.columns:
                    if "每股净资产" in str(col):
                        bps_col = col
                        break
            if bps_col:
                # fa_df 升序，iloc[-1] 取最新报告期
                bps_val = fa_df.iloc[-1].get(bps_col, None)
                if bps_val is not None and not pd.isna(bps_val):
                    bps = float(bps_val)
        # 5b. 从资产负债表计算: BPS = TOTAL_PARENT_EQUITY / total_shares
        if bps is None and total_shares is not None:
            try:
                report_code = to_akshare_report_format(ticker)
                bs_df = ak.stock_balance_sheet_by_report_em(symbol=report_code)
                if bs_df is not None and not bs_df.empty and "TOTAL_PARENT_EQUITY" in bs_df.columns:
                    equity = bs_df.iloc[0].get("TOTAL_PARENT_EQUITY", None)
                    if equity is not None and not pd.isna(equity):
                        bps = float(equity) / total_shares
            except Exception:
                pass  # 非关键，获取不到不影响其他指标

        # --- 6. 计算 TTM 每股营业收入 ---
        ttm_rev_ps = None
        if income_df is not None and not income_df.empty and total_shares is not None:
            ttm_rev_ps = compute_ttm_revenue_per_share(
                income_df,
                revenue_col="OPERATE_INCOME",
                date_col="REPORT_DATE",
                total_shares=total_shares,
            )

        # --- 7. 调用公共函数计算估值指标 ---
        val = compute_valuation_metrics(
            close_price=close_price,
            ttm_eps=ttm_eps,
            annual_eps=annual_eps,
            bps=bps,
            total_shares=total_shares,
            outstanding_shares=outstanding_shares,
            ttm_revenue_per_share=ttm_rev_ps,
        )

        # --- 8. 格式化输出（匹配 Tushare 层格式） ---
        sections.append("## 估值指标（系统计算）")
        sections.append(f"收盘价(元): {close_price}")

        if val["dynamic_pe"] is not None:
            sections.append(f"动态PE(TTM): {val['dynamic_pe']}倍 (公式: 收盘价/TTM_EPS)")
        else:
            sections.append("动态PE(TTM): N/A (缺少TTM_EPS)")

        if val["static_pe"] is not None:
            sections.append(f"静态PE: {val['static_pe']}倍 (公式: 收盘价/年度EPS)")
        else:
            sections.append("静态PE: N/A (缺少年度EPS)")

        if val["pb"] is not None:
            sections.append(f"PB: {val['pb']}")
        else:
            sections.append("PB: N/A")

        if val["ps_ttm"] is not None:
            sections.append(f"PS(TTM): {val['ps_ttm']}")
        else:
            sections.append("PS(TTM): N/A")

        if val["total_mv_yi"] is not None:
            sections.append(f"总市值(亿元): {val['total_mv_yi']}")
        else:
            sections.append("总市值(亿元): N/A")

        if val["circ_mv_yi"] is not None:
            sections.append(f"流通市值(亿元): {val['circ_mv_yi']}")
        else:
            sections.append("流通市值(亿元): N/A")

        sections.append("")
    except Exception as e:
        logger.warning("AKShare 估值指标计算失败: %s", e)


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
    table = extract_and_format(df, AKSHARE_BALANCE_SHEET_MAP, period_col="REPORT_DATE", limit=limit)

    header = (
        f"# Balance Sheet for {ticker} ({freq})\n"
        f"# Source: AKShare (东方财富)\n"
        f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    )
    return header + table


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
    table = extract_and_format(df, AKSHARE_CASHFLOW_MAP, period_col="REPORT_DATE", limit=limit)

    header = (
        f"# Cash Flow for {ticker} ({freq})\n"
        f"# Source: AKShare (东方财富)\n"
        f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    )
    return header + table


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
    table = extract_and_format(df, AKSHARE_INCOME_MAP, period_col="REPORT_DATE", limit=limit)

    header = (
        f"# Income Statement for {ticker} ({freq})\n"
        f"# Source: AKShare (东方财富)\n"
        f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    )
    return header + table


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
    except Exception as e:
        # 接口调用失败时抛异常，让 route_to_vendor 继续降级到 yfinance
        raise AKShareError(f"stock_inner_trade_xq 调用失败: {e}")

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
