"""Tushare Pro 数据供应商 —— 专业 A 股数据接口。

Token 通过环境变量 TUSHARE_TOKEN 配置。
未配置 Token 时，所有方法抛出 TushareUnavailableError 以触发 fallback。
"""

import os
import re
import time
import logging
from typing import Annotated
from datetime import datetime, timedelta

import pandas as pd

from .ticker_utils import to_tushare_format, to_akshare_date, to_standard_date, is_etf_or_lof
from .vendor_errors import TushareRateLimitError, TushareUnavailableError
from .financial_field_maps import (
    extract_and_format,
    TUSHARE_FUNDAMENTALS_MAP,
    TUSHARE_BALANCE_SHEET_MAP,
    TUSHARE_CASHFLOW_MAP,
    TUSHARE_INCOME_MAP,
)

logger = logging.getLogger(__name__)

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


def _parse_retry_delay(error_msg: str) -> int | None:
    """从 Tushare 限流错误消息中解析重试等待时间（秒）。

    Tushare 错误格式示例:
      "抱歉，您访问接口(stock_basic)频率超限(1次/分钟)"
      "抱歉，您每分钟最多访问200次"

    Returns:
        等待秒数（含 5 秒缓冲），适用于分钟级限流；
        对于小时/天级限流返回 None（表示不应重试，直接 fallback）；
        解析失败时返回 65 秒（保守默认值，按分钟级处理）。
    """
    # 格式1: "1次/分钟" / "200次/分钟"
    match = re.search(r'(\d+)\s*次\s*/\s*(\S+)', error_msg)
    if match:
        count = int(match.group(1))
        unit = match.group(2)
    else:
        # 格式2: "每分钟最多访问200次" / "每小时最多访问10次"
        match = re.search(r'每(\S+?)最多.*?(\d+)\s*次', error_msg)
        if match:
            unit = match.group(1)
            count = int(match.group(2))
        else:
            logger.debug("无法解析限流频率，使用默认等待 65s: %s", error_msg)
            return 65  # 保守默认：1次/分钟 + 5s 缓冲

    if "分" in unit:
        interval = 60
    elif "小" in unit:
        # 小时级限流：重试无意义（需要等 3600+ 秒），直接 fallback
        logger.info("Tushare 小时级限流，跳过重试直接 fallback: %s", error_msg)
        return None
    elif "天" in unit or "日" in unit:
        # 天级限流：同上，直接 fallback
        logger.info("Tushare 天级限流，跳过重试直接 fallback: %s", error_msg)
        return None
    else:
        return 65

    wait = interval // count + 5  # 加 5 秒缓冲
    return min(max(wait, 5), 120)  # 分钟级限制在 5~120 秒之间


# 限流重试配置
_RATE_LIMIT_MAX_RETRIES = 2  # 最大重试次数


def _safe_call(func, *args, **kwargs):
    """包装 Tushare API 调用，捕获频率限制和权限错误。

    限流时自动重试：从错误消息中解析限流频率（如 "1次/分钟"→等65秒），
    计算合适的等待时间后重试，最多重试 _RATE_LIMIT_MAX_RETRIES 次。
    小时/天级限流不重试，直接抛异常触发 fallback。
    仅对分钟级频率限制重试，权限/积分类错误不重试。
    """
    retries = 0

    while True:
        try:
            return func(*args, **kwargs)
        except (TushareUnavailableError, TushareRateLimitError):
            raise
        except Exception as e:
            msg = str(e)
            msg_lower = msg.lower()

            if any(kw in msg_lower for kw in ("每分钟", "rate", "频率", "too many", "每小时", "每天")):
                delay = _parse_retry_delay(msg)
                if delay is None:
                    # 小时/天级限流，重试无意义，直接抛异常触发 fallback
                    raise TushareRateLimitError(
                        f"Tushare 限流级别过高（小时/天），跳过重试：{e}"
                    )
                if retries < _RATE_LIMIT_MAX_RETRIES:
                    retries += 1
                    logger.warning(
                        "Tushare 限流，第 %d 次重试（等待 %ds）: %s",
                        retries, delay, e,
                    )
                    time.sleep(delay)
                    continue
                raise TushareRateLimitError(
                    f"Tushare 请求频率超限（已重试 {retries} 次）：{e}"
                )

            if any(kw in msg_lower for kw in ("积分", "权限", "point", "permission")):
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
    """获取 A 股日线行情（Tushare Pro）。
    
    自动识别股票和基金（ETF/LOF），使用对应的接口：
    - 股票：pro.daily
    - 基金/ETF：pro.fund_daily
    """
    pro = _get_tushare_api()
    ts_code = to_tushare_format(symbol)
    
    # 检测是否为 ETF/LOF 基金，选择对应的接口
    is_fund = is_etf_or_lof(symbol)
    api_func = pro.fund_daily if is_fund else pro.daily
    data_source = "Tushare Pro (基金)" if is_fund else "Tushare Pro"

    df = _safe_call(
        api_func,
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
        f"# Source: {data_source}\n"
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

def _compute_ttm_eps(fina_df: pd.DataFrame) -> float | None:
    """从 fina_indicator 累计数据计算 TTM（滚动12个月）每股收益。

    TTM_EPS = 最新年报EPS - 上年同期Q1_EPS + 今年Q1_EPS

    中国财报披露规则:
      Q1(单季) → H1(累计) → 9M(累计) → Annual(累计)
      fina_indicator 的 eps 字段为累计值，故:
      TTM = Annual_EPS - prev_Q1_EPS + curr_Q1_EPS
    """
    if fina_df is None or fina_df.empty or "eps" not in fina_df.columns:
        return None
    if "end_date" not in fina_df.columns:
        return None

    df = fina_df.sort_values("end_date").copy()
    df["end_date"] = df["end_date"].astype(str)

    # 最新年报 EPS
    annual_mask = df["end_date"].str.endswith("1231")
    annual_rows = df[annual_mask]
    if annual_rows.empty:
        return None
    latest_annual = annual_rows.iloc[-1]
    annual_eps = latest_annual.get("eps", None)
    if annual_eps is None or annual_eps <= 0:
        return None

    ttm_eps = float(annual_eps)

    # + 最新 Q1 EPS
    q1_mask = df["end_date"].str.endswith("0331")
    if q1_mask.any():
        latest_q1 = df[q1_mask].iloc[-1]
        latest_q1_eps = latest_q1.get("eps", None)
        if latest_q1_eps is not None:
            ttm_eps = ttm_eps + float(latest_q1_eps)

    # - 上年同期 Q1 EPS
    if q1_mask.sum() >= 2:
        prev_q1 = df[q1_mask].iloc[-2]
        prev_q1_eps = prev_q1.get("eps", None)
        if prev_q1_eps is not None:
            ttm_eps = ttm_eps - float(prev_q1_eps)

    return ttm_eps if ttm_eps > 0 else None


def get_fundamentals(
    ticker: Annotated[str, "ticker symbol of the company"],
    curr_date: Annotated[str, "current date"] = None,
) -> str:
    """获取 A 股公司基本面（Tushare Pro）。

    三个子接口（stock_basic / fina_indicator / daily_basic）独立获取，
    单个接口限流或不可用时不阻塞其他接口——这样即使 stock_basic 限流，
    daily_basic 仍可返回 PE/PB/PS 等估值指标。
    """
    pro = _get_tushare_api()
    ts_code = to_tushare_format(ticker)
    sections: list[str] = []
    has_data = False  # 追踪是否至少有一个接口返回了有效数据
    fina = None  # 提升到函数级别，供后续 PE 计算使用

    # 公司基本信息
    try:
        basic = _safe_call(
            pro.stock_basic,
            ts_code=ts_code,
            fields="ts_code,symbol,name,area,industry,market,list_date",
        )
        if basic is not None and not basic.empty:
            has_data = True
            row = basic.iloc[0]
            sections.append("## 公司基本信息")
            sections.append(f"代码: {row.get('ts_code', 'N/A')}")
            sections.append(f"名称: {row.get('name', 'N/A')}")
            sections.append(f"地区: {row.get('area', 'N/A')}")
            sections.append(f"行业: {row.get('industry', 'N/A')}")
            sections.append(f"市场: {row.get('market', 'N/A')}")
            sections.append(f"上市日期: {row.get('list_date', 'N/A')}")
            sections.append("")
    except (TushareUnavailableError, TushareRateLimitError) as e:
        logger.warning("获取公司基本信息失败（接口限流或不可用），跳过: %s", e)
    except Exception as e:
        sections.append(f"# 获取公司信息出错：{e}\n")

    # 财务指标（精选关键字段，消除列名歧义）
    try:
        fina = _safe_call(pro.fina_indicator, ts_code=ts_code, limit=5)
        if fina is not None and not fina.empty:
            has_data = True
            sections.append("## 财务指标（最近4期）")
            sections.append(extract_and_format(fina, TUSHARE_FUNDAMENTALS_MAP, period_col="end_date", limit=5))
    except (TushareUnavailableError, TushareRateLimitError) as e:
        logger.warning("获取财务指标失败（接口限流或不可用），跳过: %s", e)
    except Exception as e:
        sections.append(f"# 获取财务指标出错：{e}\n")

    # 估值指标（PE / PB 等）—— 系统计算 PE，不依赖 API 的 pe_ttm
    close_price = None
    api_pe_ttm = None
    daily_basic = None
    try:
        if curr_date:
            daily_basic = _safe_call(
                pro.daily_basic,
                ts_code=ts_code,
                trade_date=to_akshare_date(curr_date),
            )
        else:
            daily_basic = _safe_call(pro.daily_basic, ts_code=ts_code, limit=1)
        # 非交易日时 trade_date 查询返回空，自动回退到 limit=1 获取最近交易日
        if (daily_basic is None or daily_basic.empty) and curr_date:
            logger.info("daily_basic 指定日期无数据（可能是非交易日），回退到 limit=1: %s", ts_code)
            daily_basic = _safe_call(pro.daily_basic, ts_code=ts_code, limit=1)
        if daily_basic is not None and not daily_basic.empty:
            has_data = True
            r = daily_basic.iloc[0]
            close_price = float(r["close"]) if pd.notna(r.get("close")) else None
            api_pe_ttm = float(r["pe_ttm"]) if pd.notna(r.get("pe_ttm")) else None
            sections.append(f"收盘价(元): {r.get('close', 'N/A')}")
    except (TushareUnavailableError, TushareRateLimitError) as e:
        logger.warning("获取估值指标失败（接口限流或不可用），跳过: %s", e)
    except Exception as e:
        sections.append(f"# 获取估值指标出错：{e}\n")

    # 系统计算 PE（核心修复：不再依赖 Tushare pe_ttm，自行计算确保准确性）
    try:
        sections.append("## PE估值（系统计算）")

        # 动态 PE(TTM)：收盘价 / TTM_EPS
        ttm_eps = _compute_ttm_eps(fina)
        if close_price and ttm_eps:
            dynamic_pe = round(close_price / ttm_eps, 2)
            sections.append(f"动态PE(系统计算): {dynamic_pe}倍 (公式: 收盘价/TTM_EPS)")
        else:
            sections.append("动态PE(系统计算): N/A (缺少收盘价或TTM_EPS)")

        # 静态 PE：收盘价 / 年度 EPS
        if fina is not None and not fina.empty and close_price:
            annual_mask = fina["end_date"].astype(str).str.endswith("1231")
            annual_rows = fina[annual_mask]
            if not annual_rows.empty:
                annual_eps = float(annual_rows.iloc[-1].get("eps", 0))
                if annual_eps > 0:
                    static_pe = round(close_price / annual_eps, 2)
                    sections.append(f"静态PE(系统计算): {static_pe}倍 (公式: 收盘价/年度EPS)")
                else:
                    sections.append("静态PE(系统计算): N/A (年度EPS<=0)")

        # API 参考值（仅供对比，不作为主要依据）
        if api_pe_ttm is not None:
            sections.append(f"PE(TTM/API参考): {round(api_pe_ttm, 4)}倍 (Tushare daily_basic 直接返回，仅供参考)")

        # 偏差警告
        if close_price and ttm_eps and api_pe_ttm is not None:
            calc_pe = close_price / ttm_eps
            if calc_pe > 0 and abs(api_pe_ttm - calc_pe) / calc_pe > 0.15:
                deviation = round(abs(api_pe_ttm - calc_pe) / calc_pe * 100)
                sections.append(f"⚠️ PE偏差警告: API值与系统计算值偏差 {deviation}%，以系统计算值为准")

        # PB / PS / 市值
        if daily_basic is not None and not daily_basic.empty:
            r = daily_basic.iloc[0]
            sections.append(f"PB: {r.get('pb', 'N/A')}")
            sections.append(f"PS(TTM): {r.get('ps_ttm', 'N/A')}")
            total_mv = r.get('total_mv', None)
            sections.append(f"总市值(万元): {total_mv if total_mv is not None else 'N/A'}")
            if total_mv is not None and pd.notna(total_mv):
                sections.append(f"总市值(亿元): {round(float(total_mv) / 10000, 2)}")
            sections.append(f"流通市值(万元): {r.get('circ_mv', 'N/A')}")
        sections.append("")
    except Exception as e:
        logger.warning("系统计算 PE 出错: %s", e)

    # 如果三个接口全部失败，抛出异常让 route_to_vendor fallback 到 AKShare
    if not has_data:
        raise TushareUnavailableError(
            f"Tushare get_fundamentals 所有子接口均未返回数据（{ticker}）"
        )

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

    table = extract_and_format(df, TUSHARE_BALANCE_SHEET_MAP, period_col="end_date", limit=limit)
    header = (
        f"# Balance Sheet for {ticker} ({freq})\n"
        f"# Source: Tushare Pro\n"
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
    """获取现金流量表（需要 2000+ 积分）。"""
    pro = _get_tushare_api()
    ts_code = to_tushare_format(ticker)

    limit = 4 if freq == "quarterly" else 2
    df = _safe_call(pro.cashflow, ts_code=ts_code, limit=limit)

    if df is None or df.empty:
        return f"未找到股票 '{ticker}' 的现金流量表数据"

    table = extract_and_format(df, TUSHARE_CASHFLOW_MAP, period_col="end_date", limit=limit)
    header = (
        f"# Cash Flow for {ticker} ({freq})\n"
        f"# Source: Tushare Pro\n"
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
    """获取利润表（需要 2000+ 积分）。"""
    pro = _get_tushare_api()
    ts_code = to_tushare_format(ticker)

    limit = 4 if freq == "quarterly" else 2
    df = _safe_call(pro.income, ts_code=ts_code, limit=limit)

    if df is None or df.empty:
        return f"未找到股票 '{ticker}' 的利润表数据"

    table = extract_and_format(df, TUSHARE_INCOME_MAP, period_col="end_date", limit=limit)
    header = (
        f"# Income Statement for {ticker} ({freq})\n"
        f"# Source: Tushare Pro\n"
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
