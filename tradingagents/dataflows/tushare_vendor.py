"""Tushare Pro 数据供应商 —— 专业 A 股数据接口。

Token 通过环境变量 TUSHARE_TOKEN 配置。
未配置 Token 时，所有方法抛出 TushareUnavailableError 以触发 fallback。
"""

import os
import re
import json
import time
import logging
from typing import Annotated, Optional
from datetime import datetime, timedelta

import pandas as pd

from tradingagents.dataflows.valuation_utils import compute_ttm_eps, compute_ttm_revenue_per_share

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

# ---------------------------------------------------------------------------
# 权限/限流缓存：记录已确认无权限或小时/天级限流的 Tushare 接口
# 避免同一运行周期内重复调用已知不可用的接口，直接触发 fallback
# ---------------------------------------------------------------------------
_DENIED_APIS: set[str] = set()

# ---------------------------------------------------------------------------
# fina_indicator 数据缓存：财务指标季度更新、极稳定 → 缓存后命中即用，不调 API（省 1次/小时限流额度）；
# 限流/不可用时回退旧缓存（旧增速 > 没增速）。这是让下游 SYS_GROWTH_YOY 稳定产出的总开关——
# earnings 腿确定性 + PEG 确定性输入都依赖它，否则一限流就全降级回 LLM 自选 → 评级摆动复发。
# ---------------------------------------------------------------------------
_FINA_CACHE_DIR = os.environ.get(
    "FINA_INDICATOR_CACHE_DIR",
    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                 "harness_data", "fina_indicator_cache"),
)
_FINA_CACHE_TTL_SEC = 7 * 24 * 3600  # 7 天内必不变（季度数据）；新鲜则直接用、不调 API


def _fina_cache_path(ts_code: str) -> str:
    return os.path.join(_FINA_CACHE_DIR, f"fina_{ts_code}.json")


def _read_fina_cache(ts_code: str, require_fresh: bool) -> Optional[pd.DataFrame]:
    """读 fina_indicator 缓存。require_fresh=True 时仅返回 TTL 内的；损坏/缺失返回 None。"""
    path = _fina_cache_path(ts_code)
    if not os.path.exists(path):
        return None
    if require_fresh and (time.time() - os.path.getmtime(path)) >= _FINA_CACHE_TTL_SEC:
        return None
    try:
        records = json.load(open(path, encoding="utf-8"))
        if not records:
            return None
        return pd.DataFrame(records)
    except Exception as e:
        logger.debug("fina_indicator 缓存读取失败 %s: %s", ts_code, e)
        return None


def _write_fina_cache(ts_code: str, fina: pd.DataFrame) -> None:
    try:
        os.makedirs(_FINA_CACHE_DIR, exist_ok=True)
        json.dump(fina.to_dict(orient="records"),
                  open(_fina_cache_path(ts_code), "w", encoding="utf-8"),
                  ensure_ascii=False, default=str)
    except Exception as e:
        logger.debug("fina_indicator 缓存写入失败 %s: %s", ts_code, e)


def _fetch_fina_indicator_cached(pro, ts_code: str) -> Optional[pd.DataFrame]:
    """fina_indicator 取数 + 数据缓存（新鲜缓存直接用；限流/失败回退旧缓存）。"""
    # 1) 新鲜缓存命中 → 直接用，不调 API（省限流额度；fina 季度才变）
    fresh = _read_fina_cache(ts_code, require_fresh=True)
    if fresh is not None:
        logger.info("fina_indicator 命中新鲜缓存(跳过 API)：%s", ts_code)
        return fresh
    # 2) 调 API
    try:
        fina = _safe_call(pro.fina_indicator, ts_code=ts_code, limit=5, api_name="fina_indicator")
    except (TushareUnavailableError, TushareRateLimitError):
        # 3) 限流/不可用 → 回退旧缓存（哪怕过期）；无缓存才把异常抛回原 fallback
        stale = _read_fina_cache(ts_code, require_fresh=False)
        if stale is not None:
            logger.info("fina_indicator 限流/不可用，回退旧缓存：%s", ts_code)
            return stale
        raise
    if fina is not None and not fina.empty:
        _write_fina_cache(ts_code, fina)
        return fina
    # API 返回空 → 试旧缓存兜底
    return _read_fina_cache(ts_code, require_fresh=False)


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


def _safe_call(func, *args, api_name: str = "", **kwargs):
    """包装 Tushare API 调用，捕获频率限制和权限错误。

    限流时自动重试：从错误消息中解析限流频率（如 "1次/分钟"→等65秒），
    计算合适的等待时间后重试，最多重试 _RATE_LIMIT_MAX_RETRIES 次。
    小时/天级限流不重试，直接抛异常触发 fallback。
    仅对分钟级频率限制重试，权限/积分类错误不重试。

    权限缓存：当接口返回权限不足或小时/天级限流时，将 api_name 记入
    _DENIED_APIS，后续调用直接跳过，避免重复无效请求。
    """
    # 预检：如果此接口已确认无权限或高级别限流，直接跳过
    if api_name and api_name in _DENIED_APIS:
        raise TushareUnavailableError(
            f"Tushare 接口 {api_name} 已确认不可用（权限不足或高级别限流），跳过调用"
        )

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
                    # 小时/天级限流，重试无意义，缓存并直接抛异常触发 fallback
                    if api_name:
                        _DENIED_APIS.add(api_name)
                    raise TushareRateLimitError(
                        f"Tushare 限流级别过高（小时/天），跳过重试：{e}"
                    )

                # 分钟级限流：即使等待时间超过壁钟超时（60s），也优先等待重试
                # 因为 Tushare 的 A 股数据质量通常优于 AKShare 和 yfinance
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
                # 权限/积分不足：缓存此接口，后续直接跳过
                if api_name:
                    _DENIED_APIS.add(api_name)
                    logger.info("已将 Tushare 接口 %s 加入不可用缓存", api_name)
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
        api_name="fund_daily" if is_fund else "daily",
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
    # 获取 look_back_days 天的数据（加上一些缓冲用于指标计算），而不是固定 365 天
    hist_start = curr_dt - timedelta(days=look_back_days + 60)

    df = _safe_call(
        pro.daily,
        ts_code=ts_code,
        start_date=to_akshare_date(hist_start.strftime("%Y-%m-%d")),
        end_date=to_akshare_date(curr_date),
        api_name="daily",
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

    已移至 valuation_utils.compute_ttm_eps，此函数保留为兼容入口。
    """
    return compute_ttm_eps(fina_df, eps_col="eps", date_col="end_date")


def _compute_ttm_revenue_per_share_fina(
    fina_df: pd.DataFrame,
    total_shares: float,
) -> float | None:
    """从 fina_indicator 累计数据计算 TTM 每股营业收入。

    Tushare 的 fina_indicator 包含 tob_operate_income（营业总收入）字段，
    使用与 compute_ttm_eps 相同的逻辑计算 TTM 营收，再除以总股本得到每股营收。

    Args:
        fina_df: fina_indicator 返回的 DataFrame
        total_shares: 总股本（单位：股）

    Returns:
        TTM 每股营业收入（元/股），计算失败返回 None
    """
    if fina_df is None or fina_df.empty:
        return None
    if total_shares is None or total_shares <= 0:
        return None

    # 检查是否存在营业总收入字段
    revenue_col = "tob_operate_income"
    if revenue_col not in fina_df.columns:
        logger.debug("fina_indicator 中未找到 %s 字段，无法计算 PS(TTM)", revenue_col)
        return None

    # 复用 valuation_utils 的计算逻辑
    return compute_ttm_revenue_per_share(
        fina_df,
        revenue_col=revenue_col,
        date_col="end_date",
        total_shares=total_shares,
    )


def _format_growth_indicators(fina) -> str:
    """从 fina_indicator df 抽确定性增速指标，输出固定格式行供 stock_profile parser 直读。

    字段（tushare fina_indicator 标准列）：
    - q_sales_yoy / q_netprofit_yoy：单季营收 / 归母净利 同比增速（%）
    - or_yoy / netprofit_yoy：累计营收 / 归母净利 同比增速（%，最近 1231 期≈年度）
    - dt_netprofit_yoy：扣非净利同比增速（%）——成长质量闸用
    - profit_dedt：扣除非经常性损益后净利润（绝对值）——判断主业是否亏损
    防御式：列缺失或解析失败 → 返回空（上层 parser 退回散文兜底）。
    """
    try:
        df = fina.sort_values("end_date")
        latest = df.iloc[-1]
        annual = df[df["end_date"].astype(str).str.endswith("1231")]
        annual_row = annual.iloc[-1] if not annual.empty else latest

        def _g(row, col):
            v = row.get(col) if hasattr(row, "get") else None
            try:
                return float(v) if v is not None and v == v else None
            except (ValueError, TypeError):
                return None

        rev_q = _g(latest, "q_sales_yoy")
        np_q = _g(latest, "q_netprofit_yoy")
        rev_a = _g(annual_row, "or_yoy")
        np_a = _g(annual_row, "netprofit_yoy")
        dt_a = _g(annual_row, "dt_netprofit_yoy")      # 扣非净利同比（年度）
        dedt_a = _g(annual_row, "profit_dedt")         # 扣非净利绝对值（最新年报，累计）
        dedt_l = _g(latest, "profit_dedt")             # 扣非净利绝对值（最新一期，累计 YTD）
        if all(x is None for x in (rev_q, np_q, rev_a, np_a)):
            return ""

        def _f(v):
            return f"{v:.2f}" if v is not None else "NA"

        out = (
            "\n【SYS_GROWTH_YOY｜tushare fina_indicator 确定性增速，下游直读勿改】 "
            f"营收YoY 单季={_f(rev_q)}% 年度={_f(rev_a)}% | "
            f"归母净利YoY 单季={_f(np_q)}% 年度={_f(np_a)}% | "
            f"扣非净利YoY 年度={_f(dt_a)}%\n"
        )
        # 成长质量闸（扣非口径）：用「最新一期」扣非判主业是否亏损（捕捉近期恶化——
        # 年报扣非可能为正但最新季报已转亏，如淳中 FY2024 扣非+2.8亿 vs Q1 2026 扣非亏损）。
        # 最新一期 OR 最新年报 任一为亏 → recurring_loss=yes（宁谨慎，仅封顶前瞻 PEG、回退保守）。
        dedt_use = dedt_l if dedt_l is not None else dedt_a
        if dedt_use is not None or dedt_a is not None:
            loss = ((dedt_l is not None and dedt_l <= 0)
                    or (dedt_a is not None and dedt_a <= 0))
            ref = dedt_l if dedt_l is not None else dedt_a
            period = "最新一期" if dedt_l is not None else "最新年报"
            sign = "正" if ref > 0 else ("负" if ref < 0 else "零")
            out += (
                "【SYS_GROWTH_QUALITY｜扣非口径成长质量，下游前瞻路由/盈利腿直读】 "
                f"扣非净利({period})={ref/1e8:.2f}亿({sign}) | 年报扣非={_f(dedt_a/1e8 if dedt_a is not None else None)}亿 | "
                f"recurring_loss={'yes' if loss else 'no'} | 扣非净利YoY年度={_f(dt_a)}%\n"
            )
        return out
    except Exception as e:
        logger.debug("_format_growth_indicators 失败: %s", e)
        return ""


def get_fundamentals(
    ticker: Annotated[str, "ticker symbol of the company"],
    curr_date: Annotated[str, "current date"] = None,
) -> str:
    """获取 A 股公司基本面（Tushare Pro）。

    三个子接口（stock_basic / fina_indicator / daily_basic）独立获取，
    单个接口限流或不可用时不阻塞其他接口——这样即使 stock_basic 限流，
    daily_basic 仍可返回 PE/PB/PS 等估值指标。

    当所有关键子接口均已确认不可用（权限不足或高级别限流）时，
    直接抛出 TushareUnavailableError 触发 fallback 到 AKShare，
    避免逐一尝试浪费时间。
    """
    # 预检：如果所有关键子接口都已确认不可用，直接触发 fallback
    _FUNDAMENTALS_CRITICAL_APIS = {"stock_basic", "fina_indicator", "daily_basic"}
    if _FUNDAMENTALS_CRITICAL_APIS.issubset(_DENIED_APIS):
        raise TushareUnavailableError(
            f"Tushare get_fundamentals 所有关键子接口已确认不可用"
            f"（{_FUNDAMENTALS_CRITICAL_APIS}），直接 fallback"
        )

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
            api_name="stock_basic",
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
        fina = _fetch_fina_indicator_cached(pro, ts_code)
        if fina is not None and not fina.empty:
            has_data = True
            sections.append("## 财务指标（最近4期）")
            sections.append(extract_and_format(fina, TUSHARE_FUNDAMENTALS_MAP, period_col="end_date", limit=5))
            # 确定性增速指标（复用本次 fina_indicator，不额外调接口）——供 stock_profile parser 直读，
            # 不再依赖 LLM 把增速写成各种散文格式（之前 parser 反复抓不到的根因）
            growth_line = _format_growth_indicators(fina)
            if growth_line:
                sections.append(growth_line)
    except (TushareUnavailableError, TushareRateLimitError) as e:
        logger.warning("获取财务指标失败（接口限流或不可用），跳过: %s", e)
    except Exception as e:
        sections.append(f"# 获取财务指标出错：{e}\n")

    # 利润表（用于 PS(TTM) 计算 fallback：当 fina_indicator 无 tob_operate_income 时使用）
    income_for_ps = None
    try:
        if fina is None or fina.empty or "tob_operate_income" not in fina.columns:
            logger.debug("fina_indicator 无 tob_operate_income 字段，尝试从 income 接口获取营业收入")
            income_for_ps = _safe_call(pro.income, ts_code=ts_code, limit=5, api_name="income")
            if income_for_ps is not None and not income_for_ps.empty:
                logger.debug("成功从 income 接口获取营业收入数据，可用于 PS(TTM) 计算")
    except (TushareUnavailableError, TushareRateLimitError) as e:
        logger.warning("获取利润表失败（接口限流或不可用），PS(TTM) 可能无法计算: %s", e)
    except Exception as e:
        logger.debug("获取利润表出错，PS(TTM) 可能无法计算: %s", e)

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
                api_name="daily_basic",
            )
        else:
            daily_basic = _safe_call(pro.daily_basic, ts_code=ts_code, limit=1, api_name="daily_basic")
        # 非交易日时 trade_date 查询返回空，自动回退到 limit=1 获取最近交易日
        if (daily_basic is None or daily_basic.empty) and curr_date:
            logger.info("daily_basic 指定日期无数据（可能是非交易日），回退到 limit=1: %s", ts_code)
            daily_basic = _safe_call(pro.daily_basic, ts_code=ts_code, limit=1, api_name="daily_basic")
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

        sections.append("")

    except Exception as e:
        logger.warning("系统计算 PE 出错: %s", e)

    # 系统计算 PS(TTM)（核心修复：不再依赖 Tushare ps_ttm，自行计算确保准确性）
    try:
        sections.append("## PS(TTM)估值（系统计算）")

        # 从 daily_basic 获取总股本（单位：股）
        total_shares = None
        if daily_basic is not None and not daily_basic.empty:
            r = daily_basic.iloc[0]
            # Tushare daily_basic 返回的 total_share 单位为股
            total_shares = float(r["total_share"]) if pd.notna(r.get("total_share")) else None

        # 系统计算 PS(TTM)：优先用 fina_indicator.tob_operate_income，fallback 到 income.revenue
        ttm_rev_ps = _compute_ttm_revenue_per_share_fina(fina, total_shares)
        revenue_source = "fina_indicator.tob_operate_income"
        if ttm_rev_ps is None and income_for_ps is not None and not income_for_ps.empty:
            # Fallback: 从 income 接口的 revenue 字段计算
            from tradingagents.dataflows.valuation_utils import compute_ttm_revenue_per_share
            if "revenue" in income_for_ps.columns:
                ttm_rev_ps = compute_ttm_revenue_per_share(
                    income_for_ps,
                    revenue_col="revenue",
                    date_col="end_date",
                    total_shares=total_shares,
                )
                revenue_source = "income.revenue(fallback)"
            elif "OPERATE_INCOME" in income_for_ps.columns:
                ttm_rev_ps = compute_ttm_revenue_per_share(
                    income_for_ps,
                    revenue_col="OPERATE_INCOME",
                    date_col="end_date",
                    total_shares=total_shares,
                )
                revenue_source = "income.OPERATE_INCOME(fallback)"

        if close_price and ttm_rev_ps:
            calc_ps = round(close_price / ttm_rev_ps, 2)
            sections.append(f"PS(TTM/系统计算): {calc_ps} (公式: 收盘价/TTM_每股营业收入, 数据源: {revenue_source})")
        else:
            sections.append(f"PS(TTM/系统计算): N/A (缺少收盘价、总股本或营业收入数据)")

        # API 参考值（仅供对比，不作为主要依据）
        if daily_basic is not None and not daily_basic.empty:
            r = daily_basic.iloc[0]
            api_ps_ttm = float(r["ps_ttm"]) if pd.notna(r.get("ps_ttm")) else None
            if api_ps_ttm is not None:
                sections.append(f"PS(TTM/API参考): {round(api_ps_ttm, 4)} (Tushare daily_basic 直接返回，仅供参考)")

        # 偏差警告
        if close_price and ttm_rev_ps and api_ps_ttm is not None:
            calc_ps = close_price / ttm_rev_ps
            if calc_ps > 0 and abs(api_ps_ttm - calc_ps) / calc_ps > 0.15:
                deviation = round(abs(api_ps_ttm - calc_ps) / calc_ps * 100)
                sections.append(f"⚠️ PS偏差警告: API值与系统计算值偏差 {deviation}%，以系统计算值为准")

        # PB / 市值（从 daily_basic 获取）
        if daily_basic is not None and not daily_basic.empty:
            r = daily_basic.iloc[0]
            sections.append(f"PB: {r.get('pb', 'N/A')}")
            total_mv = r.get('total_mv', None)
            sections.append(f"总市值(万元): {total_mv if total_mv is not None else 'N/A'}")
            if total_mv is not None and pd.notna(total_mv):
                sections.append(f"总市值(亿元): {round(float(total_mv) / 10000, 2)}")
            sections.append(f"流通市值(万元): {r.get('circ_mv', 'N/A')}")
        sections.append("")
    except Exception as e:
        logger.warning("系统计算 PS(TTM) 出错: %s", e)

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
    df = _safe_call(pro.balancesheet, ts_code=ts_code, limit=limit, api_name="balancesheet")

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
    df = _safe_call(pro.cashflow, ts_code=ts_code, limit=limit, api_name="cashflow")

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
    df = _safe_call(pro.income, ts_code=ts_code, limit=limit, api_name="income")

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
            api_name="news",
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
            api_name="news",
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
        df = _safe_call(pro.stk_holdertrade, ts_code=ts_code, limit=20, api_name="stk_holdertrade")
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


# ---------------------------------------------------------------------------
# 10. get_capital_flow —— 资金流主力/超大/大/中/小单 + 流通市值 + 龙虎榜 30 日计数
# ---------------------------------------------------------------------------
def get_capital_flow(
    symbol: Annotated[str, "ticker symbol"],
    end_date: Annotated[str, "current trading date YYYY-mm-dd"],
    lookback_days: Annotated[int, "trading-day lookback window for moneyflow"] = 120,
) -> dict:
    """资金流原始数据装配（tushare 主路径）。

    返回结构化 dict（注意：与现有 get_stock/get_fundamentals 等返回 str 的接口风格不同——
    这里需要保留 DataFrame 给下游 capital_flow_utils 做派生计算，避免格式化-反序列化的精度丢失）。

    内部串联 3 个 tushare 接口：
    - moneyflow_dc       : 主力/超大/大/中/小单净额 + 散户成交占比（rate 字段）
    - bak_daily          : 流通市值（亿元，单位是亿元，不是万元 —— 已在 5 只股 dry-run 中验证）
    - top_list           : 龙虎榜上榜记录 → 派生 30 日上榜次数

    任何子调用失败都不阻塞，对应字段返回 None，让 capital_flow_utils 用「数据不足」逻辑兜底。
    完全不可用时（pro 未初始化）抛 TushareUnavailableError 触发 fallback。

    Returns:
        dict 含字段（全名）：
        - moneyflow_df              : DataFrame（按 trade_date 升序），含 capital_flow_utils
                                       要求的标准列名（main_force_net_amount_yi 等）
        - circulating_market_value_yi : float 或 None
        - lhb_count_30d             : int 或 None
        - latest_trade_date         : 最新交易日 YYYYMMDD，用于北向新鲜度判断
        - data_source_breakdown     : dict 标注每个子项实际数据来源（"tushare" / "missing"）
    """
    pro = _get_tushare_api()
    ts_code = to_tushare_format(symbol)
    end_compact = to_akshare_date(end_date)

    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    start_dt = end_dt - timedelta(days=lookback_days * 2)  # ×2 抵消周末/节假日缩水
    start_compact = to_akshare_date(start_dt.strftime("%Y-%m-%d"))

    out: dict = {
        "moneyflow_df": None,
        "circulating_market_value_yi": None,
        "lhb_count_30d": None,
        "lhb_inst_net_buy_30d_yi": None,
        "latest_trade_date": end_compact,
        "data_source_breakdown": {
            "moneyflow": "missing",
            "circ_mv": "missing",
            "lhb": "missing",
            "lhb_inst": "missing",
        },
    }

    # 子调用 1: moneyflow_dc（主力/超大/大/中/小单 + rate 字段）
    try:
        mf_raw = _safe_call(
            pro.moneyflow_dc,
            ts_code=ts_code,
            start_date=start_compact,
            end_date=end_compact,
            api_name="moneyflow_dc",
        )
        if mf_raw is not None and not mf_raw.empty:
            mf = mf_raw.sort_values("trade_date").reset_index(drop=True)
            # 单位换算：tushare moneyflow_dc 的 *_amount 字段单位都是「万元」
            normalized = pd.DataFrame({
                "trade_date":                  mf["trade_date"].astype(str),
                "main_force_net_amount_yi":    (mf["net_amount"].astype(float) * 1e-4).round(4)
                    if "net_amount" in mf.columns else None,
                "extra_large_net_amount_yi":   (mf["buy_elg_amount"].astype(float) * 1e-4).round(4)
                    if "buy_elg_amount" in mf.columns else None,
                "large_net_amount_yi":         (mf["buy_lg_amount"].astype(float) * 1e-4).round(4)
                    if "buy_lg_amount" in mf.columns else None,
                "medium_buy_amount_rate_pct":  mf["buy_md_amount_rate"].astype(float)
                    if "buy_md_amount_rate" in mf.columns else None,
                "small_buy_amount_rate_pct":   mf["buy_sm_amount_rate"].astype(float)
                    if "buy_sm_amount_rate" in mf.columns else None,
                "close":                       mf["close"].astype(float)
                    if "close" in mf.columns else None,
            })
            # 取最新交易日（用于北向新鲜度对照）
            out["latest_trade_date"] = str(normalized["trade_date"].iloc[-1])
            # 子调用 1.b：再补一次 daily 拿 amount，用于 ddz_like_20d_pct 分母
            try:
                daily = _safe_call(
                    pro.daily,
                    ts_code=ts_code,
                    start_date=start_compact,
                    end_date=end_compact,
                    fields="trade_date,amount",
                    api_name="daily",
                )
                if daily is not None and not daily.empty:
                    daily = daily.sort_values("trade_date").reset_index(drop=True)
                    # daily.amount 单位为千元 → 亿元（除以 1e5）
                    daily["daily_amount_yi"] = (daily["amount"].astype(float) * 1e-5).round(4)
                    normalized = normalized.merge(
                        daily[["trade_date", "daily_amount_yi"]],
                        on="trade_date", how="left",
                    )
            except (TushareUnavailableError, TushareRateLimitError) as e:
                logger.info("get_capital_flow: pro.daily amount 调用失败，ddz_like_20d_pct 将不可计算: %s", e)
            out["moneyflow_df"] = normalized
            out["data_source_breakdown"]["moneyflow"] = "tushare"
    except (TushareUnavailableError, TushareRateLimitError) as e:
        logger.warning("get_capital_flow: moneyflow_dc 不可用: %s", e)

    # 子调用 2: bak_daily 拿流通市值（亿元，单位已是亿元，不限流）
    try:
        # 取最近 5 个交易日，避免节假日空数据
        bak_start = (end_dt - timedelta(days=10)).strftime("%Y%m%d")
        bak = _safe_call(
            pro.bak_daily,
            ts_code=ts_code,
            start_date=bak_start,
            end_date=end_compact,
            fields="trade_date,close,float_mv",
            api_name="bak_daily",
        )
        if bak is not None and not bak.empty:
            bak = bak.sort_values("trade_date")
            mv = bak.iloc[-1].get("float_mv")
            if pd.notna(mv) and float(mv) > 0:
                # bak_daily.float_mv 单位 = 亿元（dry-run 已验证：close × float_share）
                out["circulating_market_value_yi"] = round(float(mv), 2)
                out["data_source_breakdown"]["circ_mv"] = "tushare"
    except (TushareUnavailableError, TushareRateLimitError) as e:
        logger.warning("get_capital_flow: bak_daily 不可用: %s", e)

    # 子调用 3: top_list（龙虎榜 30 日上榜次数）
    try:
        lhb_start = (end_dt - timedelta(days=30)).strftime("%Y%m%d")
        # top_list 是按交易日批量查询，但支持 ts_code 过滤；fallback 直接 ts_code 单股查询
        lhb = _safe_call(
            pro.top_list,
            ts_code=ts_code,
            start_date=lhb_start,
            end_date=end_compact,
            api_name="top_list",
        )
        if lhb is not None and not lhb.empty:
            # 同一天可能有多条记录（多个上榜原因），按 trade_date 去重
            unique_days = lhb["trade_date"].astype(str).nunique() if "trade_date" in lhb.columns else len(lhb)
            out["lhb_count_30d"] = int(unique_days)
            out["data_source_breakdown"]["lhb"] = "tushare"
        else:
            # 30 天内未上榜 → 0 次（不是 missing）
            out["lhb_count_30d"] = 0
            out["data_source_breakdown"]["lhb"] = "tushare"
    except (TushareUnavailableError, TushareRateLimitError) as e:
        logger.info("get_capital_flow: top_list 不可用: %s", e)

    # 子调用 4: top_inst（龙虎榜机构成交明细 → 30 日机构席位净买，判方向）
    # 机构出货/游资派发同样会上榜，所以方向看机构净买，不看上榜次数。
    try:
        inst_start = (end_dt - timedelta(days=30)).strftime("%Y%m%d")
        inst = _safe_call(
            pro.top_inst,
            ts_code=ts_code,
            start_date=inst_start,
            end_date=end_compact,
            api_name="top_inst",
        )
        if inst is not None and not inst.empty and "net_buy" in inst.columns:
            # tushare top_inst.net_buy 单位为元 → 亿元（×1e-8）。方向取符号、不依赖精确单位；
            # 阈值在 capital_flow_utils 以亿元计，单位若有出入只影响"持平带"宽度（TODO: 首跑校准）。
            net_buy_yi = float(inst["net_buy"].astype(float).sum()) * 1e-8
            out["lhb_inst_net_buy_30d_yi"] = round(net_buy_yi, 4)
            out["data_source_breakdown"]["lhb_inst"] = "tushare"
        elif inst is not None and inst.empty:
            # 30 天内无机构席位明细 → 0（不是 missing）
            out["lhb_inst_net_buy_30d_yi"] = 0.0
            out["data_source_breakdown"]["lhb_inst"] = "tushare"
    except (TushareUnavailableError, TushareRateLimitError) as e:
        logger.info("get_capital_flow: top_inst 不可用: %s", e)

    # 主路径不可用时（moneyflow + circ_mv 都为空）抛错触发 fallback
    if out["moneyflow_df"] is None and out["circulating_market_value_yi"] is None:
        raise TushareUnavailableError(
            f"Tushare get_capital_flow 主路径全部失败 ({symbol})，触发 akshare fallback"
        )

    return out


# ---------------------------------------------------------------------------
# 11. get_holder_number —— 季报股东户数（stk_holdernumber）
# ---------------------------------------------------------------------------
def get_holder_number(
    symbol: Annotated[str, "ticker symbol"],
    lookback_quarters: Annotated[int, "number of recent quarters"] = 8,
) -> Optional[pd.DataFrame]:
    """季报股东户数序列（按 end_date 升序）。

    Returns:
        DataFrame，含字段：
        - end_date     报告期 YYYYMMDD
        - holder_num   股东户数（单位：户）
        无数据返回 None；接口不可用抛 TushareUnavailableError。
    """
    pro = _get_tushare_api()
    ts_code = to_tushare_format(symbol)

    df = _safe_call(
        pro.stk_holdernumber,
        ts_code=ts_code,
        api_name="stk_holdernumber",
    )
    if df is None or df.empty:
        return None

    if "end_date" not in df.columns or "holder_num" not in df.columns:
        return None

    df = df.dropna(subset=["holder_num"])
    if df.empty:
        return None

    df = df.sort_values("end_date").tail(lookback_quarters).reset_index(drop=True)
    return df[["end_date", "holder_num"]].copy()


# ---------------------------------------------------------------------------
# 12. get_north_hold —— 北向（港股通）个股持股序列
# ---------------------------------------------------------------------------
def get_north_hold(
    symbol: Annotated[str, "ticker symbol"],
    end_date: Annotated[str, "current trading date YYYY-mm-dd"],
    lookback_days: Annotated[int, "trading-day lookback window"] = 30,
) -> Optional[pd.DataFrame]:
    """北向资金个股持股序列（tushare hk_hold）。

    注意：自 2024-08-16 起中证取消日度个股北向持股披露，akshare 接口完全停滞；
    tushare hk_hold 同样可能返回空或只有历史快照。capital_flow_utils 的
    compute_northbound_metrics 会通过 latest_trade_date 与 northbound_latest_date
    对比新鲜度（>7 天判 stale），stale 时不参与 5 维投票。

    Returns:
        DataFrame，含字段：
        - trade_date          YYYYMMDD
        - hold_share_count    持股量（股）
        无数据返回 None；接口不可用抛 TushareUnavailableError。
    """
    pro = _get_tushare_api()
    ts_code = to_tushare_format(symbol)
    end_compact = to_akshare_date(end_date)
    start_compact = to_akshare_date(
        (datetime.strptime(end_date, "%Y-%m-%d") - timedelta(days=lookback_days * 2)).strftime("%Y-%m-%d")
    )

    df = _safe_call(
        pro.hk_hold,
        code=ts_code,
        start_date=start_compact,
        end_date=end_compact,
        api_name="hk_hold",
    )
    if df is None or df.empty:
        return None

    if "trade_date" not in df.columns or "vol" not in df.columns:
        return None

    df = df.sort_values("trade_date").reset_index(drop=True)
    return pd.DataFrame({
        "trade_date":       df["trade_date"].astype(str),
        "hold_share_count": df["vol"].astype(float),
    })
