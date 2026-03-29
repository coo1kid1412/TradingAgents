"""Ticker 解析/校验层 —— 将用户输入解析为结构化的股票信息。

支持：
  - A 股纯数字代码: "600519", "518880", "160723"
  - A 股带后缀: "600519.SH", "000858.SZ"
  - 港股: "00700.HK"
  - 美股: "AAPL"
  - 中文名模糊搜索: "贵州茅台", "茅台"

Fallback 链: AKShare → Tushare → yfinance
"""

import functools
import logging
import os
import re
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from .ticker_utils import is_a_share, _extract_code, _get_exchange

logger = logging.getLogger(__name__)

_CJK_RE = re.compile(r"[\u4e00-\u9fff]")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ResolvedTicker:
    """Ticker 解析结果。"""
    code: str           # 规范化代码: "600519", "AAPL", "00700"
    name: str           # "贵州茅台", "Apple Inc."
    exchange: str       # "SH"/"SZ"/"BJ"/"HK"/"US"/""
    market: str         # "a_share" / "hk" / "us" / "other"
    original_input: str # 用户原始输入


class TickerNotFoundError(ValueError):
    """Ticker 解析失败 —— 所有数据源均未找到对应标的。"""
    pass


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _import_akshare():
    try:
        import akshare as ak
        return ak
    except ImportError:
        return None


def _get_tushare_pro():
    """获取 tushare pro api 实例，失败返回 None。"""
    try:
        from .tushare_vendor import _get_tushare_api
        return _get_tushare_api()
    except Exception:
        return None


def _import_yfinance():
    try:
        import yfinance as yf
        return yf
    except ImportError:
        return None


def _detect_market(ticker: str) -> str:
    """粗略判断 ticker 属于哪个市场。"""
    t = ticker.strip().upper()
    if is_a_share(t):
        return "a_share"
    if t.endswith(".HK"):
        return "hk"
    # Pure alpha (AAPL, MSFT) or known US suffix
    if re.match(r"^[A-Z]{1,5}$", t) or t.endswith(".US") or t.endswith(".O") or t.endswith(".N"):
        return "us"
    return "other"


# ---------------------------------------------------------------------------
# A-share resolution strategies
# ---------------------------------------------------------------------------

def _resolve_a_share_akshare(code: str) -> Optional[Tuple[str, str]]:
    """用 ak.stock_individual_info_em 查询 A 股/ETF/LOF 名称。

    Returns (name, exchange) or None.
    """
    ak = _import_akshare()
    if ak is None:
        return None

    try:
        df = ak.stock_individual_info_em(symbol=code)
        if df is None or df.empty:
            return None
        name_row = df[df["item"] == "股票简称"]
        if name_row.empty:
            return None
        name = str(name_row["value"].values[0]).strip()
        if not name:
            return None
        exchange = _get_exchange(code)
        return (name, exchange)
    except Exception as e:
        logger.debug("akshare stock_individual_info_em(%s) failed: %s", code, e)
        return None


def _resolve_a_share_tushare_stock(code: str, exchange: str) -> Optional[Tuple[str, str]]:
    """用 tushare pro.stock_basic 查询股票。"""
    pro = _get_tushare_pro()
    if pro is None:
        return None

    ts_code = f"{code}.{exchange}"
    try:
        df = pro.stock_basic(ts_code=ts_code, fields="ts_code,name")
        if df is not None and not df.empty:
            name = str(df.iloc[0]["name"]).strip()
            return (name, exchange)
    except Exception as e:
        logger.debug("tushare stock_basic(%s) failed: %s", ts_code, e)

    return None


def _resolve_a_share_tushare_fund(code: str, exchange: str) -> Optional[Tuple[str, str]]:
    """用 tushare pro.fund_basic 查询 ETF/LOF/场内基金。"""
    pro = _get_tushare_pro()
    if pro is None:
        return None

    ts_code = f"{code}.{exchange}"
    try:
        df = pro.fund_basic(ts_code=ts_code, fields="ts_code,name")
        if df is not None and not df.empty:
            name = str(df.iloc[0]["name"]).strip()
            return (name, exchange)
    except Exception as e:
        logger.debug("tushare fund_basic(%s) failed: %s", ts_code, e)

    return None


# ---------------------------------------------------------------------------
# AKShare fund cache (ETF + LOF)
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=1)
def _load_fund_code_name_map() -> Dict[str, str]:
    """加载 AKShare ETF + LOF 代码→名称映射（缓存）。

    合并 fund_etf_spot_em() (约1400条) 和 fund_lof_spot_em() (约390条)。
    """
    ak = _import_akshare()
    if ak is None:
        return {}

    result: Dict[str, str] = {}
    for loader_name in ("fund_etf_spot_em", "fund_lof_spot_em"):
        try:
            loader = getattr(ak, loader_name)
            df = loader()
            if df is not None and not df.empty and "代码" in df.columns and "名称" in df.columns:
                for _, row in df[["代码", "名称"]].iterrows():
                    code = str(row["代码"]).strip()
                    name = str(row["名称"]).strip()
                    if code and name:
                        result[code] = name
            logger.debug("Loaded %s: %d items", loader_name, len(df) if df is not None else 0)
        except Exception as e:
            logger.warning("Failed to load %s: %s", loader_name, e)

    logger.info("Fund code-name cache loaded: %d entries (ETF + LOF)", len(result))
    return result


def _resolve_a_share_akshare_fund(code: str) -> Optional[Tuple[str, str]]:
    """在 AKShare ETF/LOF 缓存中查找代码。

    Returns (name, exchange) or None.
    """
    fund_map = _load_fund_code_name_map()
    name = fund_map.get(code)
    if name:
        exchange = _get_exchange(code)
        return (name, exchange)
    return None


# 上交所基金/ETF 前缀 — 5xxxxx 几乎全是基金，优先走 fund_basic
_SH_FUND_PREFIX = "5"


def _make_resolved(code: str, name: str, exchange: str,
                   user_input: str, source: str) -> ResolvedTicker:
    """构造 ResolvedTicker 并记录日志。"""
    logger.info("Resolved %s → %s (via %s)", user_input, name, source)
    return ResolvedTicker(code=code, name=name, exchange=exchange,
                          market="a_share", original_input=user_input)


def _resolve_a_share(user_input: str) -> ResolvedTicker:
    """A 股代码解析，根据前缀智能选择接口顺序。

    路由策略：
      - 5xxxxx (上交所ETF/基金): fund_basic → AKShare fund_cache → stock_info → stock_basic
      - 1xxxxx (深市ETF/LOF等):  stock_info → AKShare fund_cache → stock_basic → fund_basic
      - 其他 A 股代码:           stock_info → stock_basic → fund_basic → AKShare fund_cache
    """
    code = _extract_code(user_input)
    exchange = _get_exchange(code)

    if code[0] == _SH_FUND_PREFIX:
        # 5xxxxx: 大概率是 ETF/基金
        # fund_basic(tushare) → AKShare ETF/LOF缓存 → stock_individual_info_em → stock_basic
        strategies = [
            (lambda: _resolve_a_share_tushare_fund(code, exchange), "Tushare fund_basic"),
            (lambda: _resolve_a_share_akshare_fund(code), "AKShare fund_cache"),
            (lambda: _resolve_a_share_akshare(code), "AKShare stock_info"),
            (lambda: _resolve_a_share_tushare_stock(code, exchange), "Tushare stock_basic"),
        ]
    elif code[0] == "1":
        # 1xxxxx: 深市ETF/LOF/债券 — 先试通用接口，再加基金缓存兜底
        strategies = [
            (lambda: _resolve_a_share_akshare(code), "AKShare stock_info"),
            (lambda: _resolve_a_share_akshare_fund(code), "AKShare fund_cache"),
            (lambda: _resolve_a_share_tushare_stock(code, exchange), "Tushare stock_basic"),
            (lambda: _resolve_a_share_tushare_fund(code, exchange), "Tushare fund_basic"),
        ]
    else:
        # 6/0/3/688/8/4: 股票为主 → stock_individual_info_em 覆盖最广
        strategies = [
            (lambda: _resolve_a_share_akshare(code), "AKShare stock_info"),
            (lambda: _resolve_a_share_tushare_stock(code, exchange), "Tushare stock_basic"),
            (lambda: _resolve_a_share_tushare_fund(code, exchange), "Tushare fund_basic"),
            (lambda: _resolve_a_share_akshare_fund(code), "AKShare fund_cache"),
        ]

    for resolver, source in strategies:
        result = resolver()
        if result:
            name, ex = result
            return _make_resolved(code, name, ex, user_input, source)

    tried = " → ".join(s for _, s in strategies)
    raise TickerNotFoundError(
        f"未找到 A 股标的 '{user_input}'（代码 {code}）。"
        f"已尝试：{tried}，均无结果。请检查代码是否正确。"
    )


# ---------------------------------------------------------------------------
# Chinese name fuzzy search
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=1)
def _load_a_share_code_name_map() -> Dict[str, str]:
    """加载 A 股代码→名称映射表（缓存）。"""
    ak = _import_akshare()
    if ak is None:
        return {}

    try:
        df = ak.stock_info_a_code_name()
        if df is None or df.empty:
            return {}
        return dict(zip(df["code"].astype(str), df["name"].astype(str)))
    except Exception as e:
        logger.warning("Failed to load A-share code-name map: %s", e)
        return {}


def _fuzzy_search_akshare(query: str) -> Optional[ResolvedTicker]:
    """在 AKShare 股票列表中按名称模糊搜索。"""
    code_name = _load_a_share_code_name_map()
    if not code_name:
        return None

    # Priority: exact match > prefix match > substring match
    exact = [(c, n) for c, n in code_name.items() if n == query]
    if exact:
        code, name = exact[0]
        return ResolvedTicker(code=code, name=name, exchange=_get_exchange(code),
                              market="a_share", original_input=query)

    prefix = [(c, n) for c, n in code_name.items() if n.startswith(query)]
    if prefix:
        code, name = prefix[0]
        return ResolvedTicker(code=code, name=name, exchange=_get_exchange(code),
                              market="a_share", original_input=query)

    contains = [(c, n) for c, n in code_name.items() if query in n]
    if contains:
        if len(contains) > 1:
            candidates = ", ".join(f"{c}({n})" for c, n in contains[:5])
            logger.warning("模糊搜索 '%s' 匹配到 %d 个结果，取第一个: %s", query, len(contains), candidates)
        code, name = contains[0]
        return ResolvedTicker(code=code, name=name, exchange=_get_exchange(code),
                              market="a_share", original_input=query)

    return None


def _fuzzy_search_tushare(query: str) -> Optional[ResolvedTicker]:
    """用 Tushare 原生模糊搜索。"""
    pro = _get_tushare_pro()
    if pro is None:
        return None

    try:
        df = pro.stock_basic(name=query, fields="ts_code,symbol,name")
        if df is not None and not df.empty:
            row = df.iloc[0]
            code = str(row["symbol"])
            name = str(row["name"])
            ts_code = str(row["ts_code"])
            exchange = ts_code.split(".")[-1] if "." in ts_code else ""
            return ResolvedTicker(code=code, name=name, exchange=exchange,
                                  market="a_share", original_input=query)
    except Exception as e:
        logger.debug("tushare stock_basic(name=%s) failed: %s", query, e)

    return None


def _resolve_by_name(user_input: str) -> ResolvedTicker:
    """中文名称模糊搜索: akshare → tushare。"""
    result = _fuzzy_search_akshare(user_input)
    if result:
        logger.info("Fuzzy resolved '%s' → %s %s (via AKShare)", user_input, result.code, result.name)
        return result

    result = _fuzzy_search_tushare(user_input)
    if result:
        logger.info("Fuzzy resolved '%s' → %s %s (via Tushare)", user_input, result.code, result.name)
        return result

    raise TickerNotFoundError(
        f"未找到与 '{user_input}' 匹配的股票。"
        f"已尝试 AKShare 名称列表 + Tushare 模糊搜索，均无结果。"
    )


# ---------------------------------------------------------------------------
# Global (HK / US / other) resolution
# ---------------------------------------------------------------------------

def _resolve_global(user_input: str) -> ResolvedTicker:
    """全球市场解析: yfinance。"""
    yf = _import_yfinance()
    if yf is None:
        raise TickerNotFoundError(
            f"无法解析 '{user_input}'：yfinance 未安装。"
        )

    # Build candidate list: original input + HK leading-zero variants
    candidates = [user_input]
    t = user_input.strip().upper()
    if t.endswith(".HK"):
        code_part = t.replace(".HK", "")
        # yfinance HK stocks may need leading zeros stripped: 00700.HK → 0700.HK
        stripped = code_part.lstrip("0")
        if stripped != code_part and stripped:
            candidates.append(f"{stripped}.HK")
            # Also try with at least 4 digits: 00700 → 0700
            if len(stripped) < 4:
                candidates.append(f"{stripped.zfill(4)}.HK")

    last_err = None
    for candidate in candidates:
        try:
            ticker = yf.Ticker(candidate)
            info = ticker.info
            if not info or info.get("regularMarketPrice") is None:
                continue

            name = info.get("shortName") or info.get("longName") or ""

            # Detect market type from original input
            if t.endswith(".HK"):
                market = "hk"
                ex = "HK"
                code = t.replace(".HK", "")
            elif re.match(r"^[A-Z]{1,5}$", t):
                market = "us"
                ex = "US"
                code = t
            else:
                market = "other"
                ex = info.get("exchange", "")
                code = t.split(".")[0] if "." in t else t

            logger.info("Resolved %s → %s (via yfinance, candidate=%s)", user_input, name, candidate)
            return ResolvedTicker(code=code, name=name, exchange=ex,
                                  market=market, original_input=user_input)
        except Exception as e:
            last_err = e
            continue

    raise TickerNotFoundError(
        f"无法解析 '{user_input}'：yfinance 查询失败"
        + (f" ({last_err})" if last_err else "") + "。"
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def resolve_ticker(user_input: str) -> ResolvedTicker:
    """将用户输入解析为结构化的 ResolvedTicker。

    解析策略:
      1. A 股代码 (is_a_share) → akshare → tushare stock → tushare fund
      2. 中文名称 (含 CJK 字符) → akshare 模糊搜索 → tushare 模糊搜索
      3. 全球标的 (AAPL, 00700.HK) → yfinance

    Raises:
        TickerNotFoundError: 所有数据源均未找到。
    """
    user_input = user_input.strip()
    if not user_input:
        raise TickerNotFoundError("输入为空，请提供股票代码或名称。")

    # Path 1: A-share code
    if is_a_share(user_input):
        return _resolve_a_share(user_input)

    # Path 2: Chinese name fuzzy search
    if _CJK_RE.search(user_input):
        return _resolve_by_name(user_input)

    # Path 3: Global (HK / US / other)
    return _resolve_global(user_input)
