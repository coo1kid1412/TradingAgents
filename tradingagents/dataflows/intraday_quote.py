"""Validated same-day A-share quote overlay.

Completed daily bars remain authoritative. This module only supplies a
process-local provisional bar for an analysis dated today.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from datetime import datetime, time as clock_time
from typing import Optional
from zoneinfo import ZoneInfo

import pandas as pd
import requests

from .ticker_utils import _get_exchange, is_a_share, to_akshare_format, to_tushare_format

logger = logging.getLogger(__name__)

_SHANGHAI = ZoneInfo("Asia/Shanghai")
_CACHE_TTL_SECONDS = 20.0
_QUOTE_CACHE: dict[tuple[str, str], tuple[float, "IntradayQuote"]] = {}
_TUSHARE_RT_K_DISABLED = False


@dataclass(frozen=True)
class IntradayQuote:
    symbol: str
    name: str
    trade_date: str
    quote_time: datetime
    open: float
    high: float
    low: float
    last: float
    pre_close: float
    volume: int
    amount: float
    source: str


def _as_float(value) -> Optional[float]:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if pd.notna(result) else None


def _normalize_datetime(value) -> Optional[datetime]:
    try:
        parsed = pd.Timestamp(value).to_pydatetime()
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=_SHANGHAI)
    return parsed.astimezone(_SHANGHAI)


def _valid_prices(open_: float, high: float, low: float, last: float, pre_close: float) -> bool:
    values = (open_, high, low, last, pre_close)
    if any(v is None or v <= 0 for v in values):
        return False
    return high >= max(open_, low, last) and low <= min(open_, high, last)


def _parse_tushare_quote(frame: pd.DataFrame, symbol: str) -> Optional[IntradayQuote]:
    if frame is None or frame.empty:
        return None
    row = frame.iloc[0]
    quote_time = _normalize_datetime(row.get("trade_time"))
    if quote_time is None:
        return None
    open_ = _as_float(row.get("open"))
    high = _as_float(row.get("high"))
    low = _as_float(row.get("low"))
    last = _as_float(row.get("close"))
    pre_close = _as_float(row.get("pre_close"))
    if not _valid_prices(open_, high, low, last, pre_close):
        return None
    volume = _as_float(row.get("vol"))
    amount = _as_float(row.get("amount"))
    if volume is None or amount is None or volume < 0 or amount < 0:
        return None
    return IntradayQuote(
        symbol=to_akshare_format(symbol),
        name=str(row.get("name") or ""),
        trade_date=quote_time.date().isoformat(),
        quote_time=quote_time,
        open=open_, high=high, low=low, last=last, pre_close=pre_close,
        volume=int(volume), amount=float(amount), source="tushare_rt_k",
    )


_SINA_BODY_RE = re.compile(r'=\s*"(.*)"\s*;?\s*$')


def _parse_sina_response(text: str, symbol: str) -> Optional[IntradayQuote]:
    match = _SINA_BODY_RE.search(text.strip()) if text else None
    if not match or not match.group(1):
        return None
    fields = match.group(1).split(",")
    if len(fields) < 32:
        return None
    open_ = _as_float(fields[1])
    pre_close = _as_float(fields[2])
    last = _as_float(fields[3])
    high = _as_float(fields[4])
    low = _as_float(fields[5])
    volume = _as_float(fields[8])
    amount = _as_float(fields[9])
    quote_time = _normalize_datetime(f"{fields[30]} {fields[31]}")
    if quote_time is None or not _valid_prices(open_, high, low, last, pre_close):
        return None
    if volume is None or amount is None or volume < 0 or amount < 0:
        return None
    return IntradayQuote(
        symbol=to_akshare_format(symbol),
        name=fields[0].strip(),
        trade_date=quote_time.date().isoformat(),
        quote_time=quote_time,
        open=open_, high=high, low=low, last=last, pre_close=pre_close,
        volume=int(volume), amount=float(amount), source="sina_realtime",
    )


def is_quote_fresh(
    quote: IntradayQuote,
    analysis_date: str,
    now: Optional[datetime] = None,
) -> bool:
    now = now or datetime.now(_SHANGHAI)
    if now.tzinfo is None:
        now = now.replace(tzinfo=_SHANGHAI)
    else:
        now = now.astimezone(_SHANGHAI)

    if analysis_date != now.date().isoformat() or quote.trade_date != analysis_date:
        return False
    quote_time = quote.quote_time.astimezone(_SHANGHAI)
    if quote_time > now.replace(microsecond=0) or (now - quote_time).total_seconds() < -60:
        return False

    current = now.time()
    quoted = quote_time.time()
    if current < clock_time(9, 25):
        return False
    if clock_time(11, 30) < current < clock_time(13, 0):
        return clock_time(11, 25) <= quoted <= clock_time(11, 31)
    if current >= clock_time(15, 0):
        return quoted >= clock_time(14, 55)

    age_seconds = (now - quote_time).total_seconds()
    return 0 <= age_seconds <= 300


def _fetch_tushare_quote(symbol: str, tushare_api) -> Optional[IntradayQuote]:
    frame = tushare_api.rt_k(
        ts_code=to_tushare_format(symbol),
        fields=(
            "ts_code,name,pre_close,high,open,low,close,vol,amount,trade_time"
        ),
    )
    return _parse_tushare_quote(frame, symbol)


def _fetch_sina_quote(symbol: str) -> Optional[IntradayQuote]:
    code = to_akshare_format(symbol)
    exchange = _get_exchange(code).lower()
    response = requests.get(
        f"https://hq.sinajs.cn/list={exchange}{code}",
        headers={"Referer": "https://finance.sina.com.cn/"},
        timeout=5,
    )
    response.raise_for_status()
    return _parse_sina_response(response.content.decode("gbk", errors="replace"), symbol)


def fetch_intraday_quote(
    symbol: str,
    analysis_date: str,
    tushare_api=None,
    now: Optional[datetime] = None,
) -> Optional[IntradayQuote]:
    """Return a fresh quote for today's A-share analysis, otherwise ``None``."""
    global _TUSHARE_RT_K_DISABLED

    now = now or datetime.now(_SHANGHAI)
    if now.tzinfo is None:
        now = now.replace(tzinfo=_SHANGHAI)
    else:
        now = now.astimezone(_SHANGHAI)
    if not is_a_share(symbol) or analysis_date != now.date().isoformat():
        return None

    cache_key = (to_akshare_format(symbol), analysis_date)
    cached = _QUOTE_CACHE.get(cache_key)
    if cached and time.monotonic() - cached[0] <= _CACHE_TTL_SECONDS:
        if is_quote_fresh(cached[1], analysis_date, now):
            return cached[1]

    if tushare_api is not None and not _TUSHARE_RT_K_DISABLED:
        try:
            quote = _fetch_tushare_quote(symbol, tushare_api)
            if quote and is_quote_fresh(quote, analysis_date, now):
                _QUOTE_CACHE[cache_key] = (time.monotonic(), quote)
                return quote
        except Exception as exc:
            message = str(exc).lower()
            if "rt_k" in message and any(word in message for word in ("权限", "permission", "积分")):
                _TUSHARE_RT_K_DISABLED = True
                logger.info("Tushare rt_k unavailable; using realtime fallback")
            else:
                logger.warning("Tushare rt_k quote failed for %s: %s", symbol, exc)

    try:
        quote = _fetch_sina_quote(symbol)
    except Exception as exc:
        logger.warning("Sina realtime quote failed for %s: %s", symbol, exc)
        return None
    if quote and is_quote_fresh(quote, analysis_date, now):
        _QUOTE_CACHE[cache_key] = (time.monotonic(), quote)
        return quote
    return None


def _date_display(value) -> str:
    text = str(value)
    if len(text) >= 10 and "-" in text:
        return text[:10]
    if len(text) >= 8 and text[:8].isdigit():
        return f"{text[:4]}-{text[4:6]}-{text[6:8]}"
    return text


def parse_price_metadata(text: str) -> dict:
    """Parse freshness headers emitted by stock-data vendors."""
    meta = {"status": "unknown", "date": None, "time": None, "source": "unknown"}
    if not text:
        return meta

    range_match = re.search(r"^# Actual date range: .*? to (\d{4}-\d{2}-\d{2})", text, re.M)
    if range_match:
        meta["date"] = range_match.group(1)
    status_match = re.search(r"^# Price data status:\s*(\S+)", text, re.M)
    if status_match:
        meta["status"] = status_match.group(1)
    source_match = re.search(r"^# Latest bar source:\s*(.+?)\s*$", text, re.M)
    if source_match:
        meta["source"] = source_match.group(1).strip()
    else:
        vendor_match = re.search(r"^# Source:\s*(.+?)\s*$", text, re.M)
        if vendor_match:
            meta["source"] = vendor_match.group(1).strip()
    time_match = re.search(r"^# Latest quote time:\s*(.+?)\s*$", text, re.M)
    if time_match:
        meta["time"] = time_match.group(1).strip()
    return meta


def merge_intraday_quote(
    daily_df: pd.DataFrame,
    quote: Optional[IntradayQuote],
    analysis_date: str,
) -> tuple[pd.DataFrame, dict]:
    """Append a provisional quote to a Tushare daily frame when appropriate."""
    merged = daily_df.copy()
    target = analysis_date.replace("-", "")
    date_values = merged["trade_date"].astype(str) if "trade_date" in merged.columns else pd.Series(dtype=str)

    if target in set(date_values.str[:8]):
        return merged, {
            "status": "official_daily", "date": analysis_date,
            "time": None, "source": "tushare_daily",
        }

    if quote is not None and quote.trade_date == analysis_date:
        row = {column: None for column in merged.columns}
        row.update({
            "trade_date": target,
            "open": quote.open,
            "high": quote.high,
            "low": quote.low,
            "close": quote.last,
            "vol": quote.volume / 100.0,
            "amount": quote.amount / 1000.0,
        })
        merged = pd.concat([merged, pd.DataFrame([row])], ignore_index=True)
        merged = merged.sort_values("trade_date").reset_index(drop=True)
        return merged, {
            "status": "intraday_provisional",
            "date": quote.trade_date,
            "time": quote.quote_time.strftime("%Y-%m-%d %H:%M:%S"),
            "source": quote.source,
        }

    latest = None
    if not merged.empty and "trade_date" in merged.columns:
        latest = _date_display(merged.sort_values("trade_date")["trade_date"].iloc[-1])
    return merged, {
        "status": "t_minus_1", "date": latest,
        "time": None, "source": "tushare_daily",
    }


def _clear_quote_cache_for_tests() -> None:
    global _TUSHARE_RT_K_DISABLED
    _QUOTE_CACHE.clear()
    _TUSHARE_RT_K_DISABLED = False
