"""Validated full-market realtime breadth built from Tushare ``rt_k``."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Callable
from zoneinfo import ZoneInfo

import joblib
import pandas as pd


_SHANGHAI = ZoneInfo("Asia/Shanghai")
_FULL_MARKET_PATTERNS = "0*.SZ,2*.SZ,3*.SZ,6*.SH,4*.BJ,8*.BJ,9*.BJ"
_RT_K_FIELDS = "ts_code,pre_close,close,trade_time"
_PERMISSION_TERMS = ("permission", "denied", "forbidden", "权限", "积分")


class RealtimeBreadthError(RuntimeError):
    """Base class for explicit realtime breadth failures."""


class RealtimePermissionUnavailable(RealtimeBreadthError):
    """The configured Tushare account cannot use full-market ``rt_k``."""


class RealtimeDataUnavailable(RealtimeBreadthError):
    """The realtime provider returned an unusable payload."""


@dataclass(frozen=True)
class RealtimePermissionProbe:
    status: str
    row_count: int
    message: str | None = None


@dataclass(frozen=True)
class PremarketBreadthBaseline:
    trade_date: date
    completed_trade_date: date
    universe_size: int
    frame: pd.DataFrame

    def __post_init__(self) -> None:
        if self.completed_trade_date >= self.trade_date:
            raise ValueError("completed_trade_date must be before trade_date")
        if isinstance(self.universe_size, bool) or not isinstance(self.universe_size, int) or self.universe_size < 1:
            raise ValueError("universe_size must be a positive integer")
        required = {"ts_code", "pre_close", "ma20", "low_20d", "industry", "down_limit"}
        if not isinstance(self.frame, pd.DataFrame) or not required.issubset(self.frame.columns):
            raise ValueError("baseline frame is missing required columns")
        object.__setattr__(self, "frame", self.frame.copy())


def _permission_error(error: Exception) -> bool:
    message = str(error).lower()
    return any(term in message for term in _PERMISSION_TERMS)


def _valid_rt_k_payload(frame: Any) -> bool:
    return isinstance(frame, pd.DataFrame) and {
        "ts_code",
        "pre_close",
        "close",
        "trade_time",
    }.issubset(frame.columns)


def probe_rt_k_permission(
    pro: Any,
    as_of_time: datetime,
    symbols: str = "600000.SH,000001.SZ",
) -> RealtimePermissionProbe:
    method = getattr(pro, "rt_k", None)
    if not callable(method):
        return RealtimePermissionProbe("unavailable", 0, "rt_k interface is missing")
    try:
        frame = method(ts_code=symbols, fields=_RT_K_FIELDS)
    except Exception as error:
        status = "permission_denied" if _permission_error(error) else "unavailable"
        return RealtimePermissionProbe(status, 0, error.__class__.__name__)
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return RealtimePermissionProbe("unavailable", 0, "empty payload")
    if not _valid_rt_k_payload(frame):
        return RealtimePermissionProbe("invalid_payload", len(frame), "missing required fields")
    visible = _normalize_rt_k(frame, as_of_time)
    if visible.empty:
        return RealtimePermissionProbe("unavailable", 0, "no visible current-session rows")
    return RealtimePermissionProbe("available", len(visible))


def _cache_path(cache_root: Path, trade_date: date) -> Path:
    return cache_root / f"a-share-breadth-baseline-{trade_date:%Y%m%d}.joblib"


def _load_cached_baseline(path: Path, trade_date: date) -> PremarketBreadthBaseline | None:
    if not path.is_file():
        return None
    try:
        value = joblib.load(path)
    except Exception:
        return None
    return value if isinstance(value, PremarketBreadthBaseline) and value.trade_date == trade_date else None


def build_premarket_baseline(
    pro: Any,
    *,
    trade_date: date,
    previous_session: Callable[[date], date],
    cache_root: Path | str,
) -> PremarketBreadthBaseline:
    """Build T-1 price baselines while requesting current-day limit prices separately."""

    root = Path(cache_root)
    root.mkdir(parents=True, exist_ok=True)
    path = _cache_path(root, trade_date)
    cached = _load_cached_baseline(path, trade_date)
    if cached is not None:
        return cached
    completed = previous_session(trade_date)
    calendar = pro.trade_cal(
        exchange="",
        start_date=(completed - timedelta(days=45)).strftime("%Y%m%d"),
        end_date=completed.strftime("%Y%m%d"),
        is_open="1",
        fields="cal_date,is_open",
    )
    if not isinstance(calendar, pd.DataFrame) or "cal_date" not in calendar:
        raise RealtimeDataUnavailable("trade calendar is unavailable")
    sessions = (
        pd.to_datetime(calendar["cal_date"].astype(str), format="%Y%m%d", errors="coerce")
        .dropna()
        .dt.date
    )
    sessions = tuple(sorted(day for day in sessions if day <= completed)[-20:])
    if len(sessions) != 20 or sessions[-1] != completed:
        raise RealtimeDataUnavailable("twenty completed sessions are required")
    daily_frames = []
    for session in sessions:
        frame = pro.daily(
            trade_date=session.strftime("%Y%m%d"),
            fields="ts_code,trade_date,close",
        )
        if isinstance(frame, pd.DataFrame) and not frame.empty:
            daily_frames.append(frame[["ts_code", "trade_date", "close"]].copy())
    if not daily_frames:
        raise RealtimeDataUnavailable("daily cross sections are unavailable")
    history = pd.concat(daily_frames, ignore_index=True)
    history["close"] = pd.to_numeric(history["close"], errors="coerce")
    history["trade_date"] = pd.to_datetime(
        history["trade_date"].astype(str), format="%Y%m%d", errors="coerce"
    ).dt.date
    history = history.dropna(subset=["ts_code", "trade_date", "close"])
    history = history.loc[history["trade_date"].isin(sessions)].sort_values(
        ["ts_code", "trade_date"], kind="stable"
    )
    latest = history.loc[history["trade_date"] == completed, ["ts_code", "close"]].rename(
        columns={"close": "pre_close"}
    )
    rolling = history.groupby("ts_code", sort=False)["close"].agg(
        ma20="mean", low_20d="min", observations="count"
    )
    rolling.loc[rolling["observations"] < 20, ["ma20", "low_20d"]] = float("nan")
    frame = latest.merge(rolling.drop(columns="observations"), on="ts_code", how="left")
    basic = pro.stock_basic(
        exchange="",
        list_status="L",
        fields="ts_code,industry",
    )
    if isinstance(basic, pd.DataFrame) and {"ts_code", "industry"}.issubset(basic.columns):
        frame = frame.merge(
            basic[["ts_code", "industry"]].drop_duplicates("ts_code", keep="last"),
            on="ts_code",
            how="left",
        )
    else:
        frame["industry"] = None
    try:
        limits = pro.stk_limit(
            trade_date=trade_date.strftime("%Y%m%d"),
            fields="ts_code,down_limit",
        )
    except Exception:
        limits = pd.DataFrame()
    if isinstance(limits, pd.DataFrame) and {"ts_code", "down_limit"}.issubset(limits.columns):
        frame = frame.merge(
            limits[["ts_code", "down_limit"]].drop_duplicates("ts_code", keep="last"),
            on="ts_code",
            how="left",
        )
    else:
        frame["down_limit"] = float("nan")
    frame = frame.drop_duplicates("ts_code", keep="last").reset_index(drop=True)
    if frame.empty:
        raise RealtimeDataUnavailable("effective stock universe is empty")
    baseline = PremarketBreadthBaseline(
        trade_date=trade_date,
        completed_trade_date=completed,
        universe_size=len(frame),
        frame=frame,
    )
    joblib.dump(baseline, path)
    return baseline


def _trade_time(value: Any, trade_date: date) -> datetime | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    if len(text) <= 8 and ":" in text:
        text = f"{trade_date.isoformat()} {text}"
    parsed = pd.to_datetime(text, errors="coerce")
    if pd.isna(parsed):
        return None
    timestamp = pd.Timestamp(parsed)
    return (
        timestamp.tz_localize(_SHANGHAI).to_pydatetime()
        if timestamp.tzinfo is None
        else timestamp.tz_convert(_SHANGHAI).to_pydatetime()
    )


def _normalize_rt_k(frame: pd.DataFrame, as_of_time: datetime) -> pd.DataFrame:
    if not _valid_rt_k_payload(frame):
        return pd.DataFrame()
    local_as_of = as_of_time.astimezone(_SHANGHAI)
    rows = frame[["ts_code", "pre_close", "close", "trade_time"]].copy()
    rows["data_time"] = rows["trade_time"].map(
        lambda value: _trade_time(value, local_as_of.date())
    )
    rows["last"] = pd.to_numeric(rows["close"], errors="coerce")
    rows["pre_close"] = pd.to_numeric(rows["pre_close"], errors="coerce")
    rows = rows.dropna(subset=["ts_code", "data_time", "last"])
    rows = rows.loc[
        rows["data_time"].map(lambda value: value.date() == local_as_of.date())
        & rows["data_time"].map(lambda value: value <= local_as_of)
        & rows["data_time"].map(
            lambda value: (local_as_of - value).total_seconds() <= 5 * 60
        )
        & rows["last"].gt(0)
    ].copy()
    if rows.empty:
        return rows
    return (
        rows.sort_values(["ts_code", "data_time"], kind="stable")
        .drop_duplicates("ts_code", keep="last")
        .reset_index(drop=True)
    )


def _refresh_limits(pro: Any, baseline: PremarketBreadthBaseline) -> pd.DataFrame:
    current = baseline.frame.copy()
    if current["down_limit"].notna().any():
        return current
    try:
        limits = pro.stk_limit(
            trade_date=baseline.trade_date.strftime("%Y%m%d"),
            fields="ts_code,down_limit",
        )
    except Exception:
        return current
    if not isinstance(limits, pd.DataFrame) or not {"ts_code", "down_limit"}.issubset(limits.columns):
        return current
    current = current.drop(columns="down_limit").merge(
        limits[["ts_code", "down_limit"]].drop_duplicates("ts_code", keep="last"),
        on="ts_code",
        how="left",
    )
    return current


def load_realtime_cross_section(
    pro: Any,
    baseline: PremarketBreadthBaseline,
    as_of_time: datetime,
    *,
    patterns: str = _FULL_MARKET_PATTERNS,
) -> pd.DataFrame:
    if as_of_time.astimezone(_SHANGHAI).date() != baseline.trade_date:
        raise ValueError("baseline trade_date must match as_of_time")
    method = getattr(pro, "rt_k", None)
    if not callable(method):
        raise RealtimePermissionUnavailable("rt_k interface is missing")
    try:
        payload = method(ts_code=patterns, fields=_RT_K_FIELDS)
    except Exception as error:
        if _permission_error(error):
            raise RealtimePermissionUnavailable("rt_k permission is unavailable") from error
        raise RealtimeDataUnavailable("rt_k request failed") from error
    if not isinstance(payload, pd.DataFrame) or payload.empty:
        result = pd.DataFrame(columns=("ts_code", "last", "pre_close", "data_time", "ma20", "low_20d", "industry", "down_limit", "source"))
        result.attrs["universe_size"] = baseline.universe_size
        return result
    if not _valid_rt_k_payload(payload):
        raise RealtimeDataUnavailable("rt_k payload is missing required fields")
    live = _normalize_rt_k(payload, as_of_time)
    if live.empty:
        result = pd.DataFrame(columns=("ts_code", "last", "pre_close", "data_time", "ma20", "low_20d", "industry", "down_limit", "source"))
        result.attrs["universe_size"] = baseline.universe_size
        return result
    baseline_frame = _refresh_limits(pro, baseline).rename(
        columns={"pre_close": "baseline_pre_close"}
    )
    result = live.merge(baseline_frame, on="ts_code", how="inner")
    result["pre_close"] = pd.to_numeric(result["pre_close"], errors="coerce").fillna(
        result["baseline_pre_close"]
    )
    result["source"] = "tushare_rt_k"
    result = result[
        [
            "ts_code",
            "last",
            "pre_close",
            "data_time",
            "ma20",
            "low_20d",
            "industry",
            "down_limit",
            "source",
        ]
    ].copy()
    result.attrs["universe_size"] = baseline.universe_size
    return result
