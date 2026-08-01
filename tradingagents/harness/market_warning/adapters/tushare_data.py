"""Point-in-time normalized Tushare adapter for A-share market data."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, replace
from datetime import date, datetime, time, timedelta
from math import isfinite
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

from tradingagents.harness.market_warning.adapters.data_cache import RawDataCache
from tradingagents.harness.market_warning.domain import Market, MarketDataPoint, RawMarketSnapshot
from tradingagents.harness.market_warning.quality import QUALITY_POLICY_V1, evaluate_data_quality


_SHANGHAI = ZoneInfo("Asia/Shanghai")
_BENCHMARKS = ("000001.SH", "399001.SZ", "000300.SH", "000905.SH", "399006.SZ", "000688.SH")
_DAILY_ROW_LIMIT = 6000
FALLBACK_CALENDAR_VERSION = "weekday-fallback-v1"
TUSHARE_DISCLOSURE_POLICY_VERSION = "tushare-disclosure-v2"


def _number(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if isfinite(result) else None


def _row_date(value: Any) -> date | None:
    if value is None or pd.isna(value):
        return None
    parsed = pd.to_datetime(str(value), format="%Y%m%d", errors="coerce")
    if pd.isna(parsed):
        parsed = pd.to_datetime(value, errors="coerce")
    return None if pd.isna(parsed) else parsed.date()


def _next_weekday(on_date: date) -> date:
    candidate = on_date + timedelta(days=1)
    while candidate.weekday() >= 5:
        candidate += timedelta(days=1)
    return candidate


def _previous_weekday(on_date: date) -> date:
    candidate = on_date - timedelta(days=1)
    while candidate.weekday() >= 5:
        candidate -= timedelta(days=1)
    return candidate


def _is_close_slot(session_slot: str) -> bool:
    normalized = session_slot.strip().lower()
    return normalized in {
        "daily", "close", "closing", "eod", "end_of_day", "post_market", "postmarket",
    } or "close" in normalized or "post_market" in normalized or "postmarket" in normalized


def _data_time(on_date: date, hour: int = 15) -> datetime:
    return datetime.combine(on_date, time(hour), tzinfo=_SHANGHAI)


def _available_at(source: str, on_date: date, next_trading_day: Callable[[date], date]) -> datetime:
    if source == "tushare_shibor":
        return datetime.combine(on_date, time(12), tzinfo=_SHANGHAI)
    if source == "tushare_margin":
        return datetime.combine(next_trading_day(on_date), time(9), tzinfo=_SHANGHAI)
    if source == "tushare_daily_basic":
        return datetime.combine(on_date + timedelta(days=1), time.min, tzinfo=_SHANGHAI)
    if source == "tushare_moneyflow":
        return datetime.combine(on_date, time(19), tzinfo=_SHANGHAI)
    return datetime.combine(on_date, time(18), tzinfo=_SHANGHAI)


@dataclass(frozen=True)
class _FetchResult:
    frame: pd.DataFrame
    complete: bool


@dataclass(frozen=True)
class _LoadResult:
    snapshot: RawMarketSnapshot
    complete: bool
    fetched_at: datetime


class TushareAShareDataAdapter:
    def __init__(
        self,
        *,
        pro: Any,
        cache: RawDataCache | None = None,
        clock: Callable[[], datetime] | None = None,
        next_trading_day: Callable[[date], date] = _next_weekday,
        previous_session: Callable[[date], date] = _previous_weekday,
        calendar_version: str = FALLBACK_CALENDAR_VERSION,
        disclosure_policy_version: str = TUSHARE_DISCLOSURE_POLICY_VERSION,
    ) -> None:
        self.pro = pro
        self.cache = cache
        self.clock = clock
        self.next_trading_day = next_trading_day
        self.previous_session = previous_session
        self.calendar_version = calendar_version
        self.disclosure_policy_version = disclosure_policy_version

    def _fetch(self, method_name: str, **kwargs: Any) -> _FetchResult:
        method = getattr(self.pro, method_name, None)
        if method is None and method_name == "limit_list_d":
            method = getattr(self.pro, "limit_list", None)
        if method is None:
            return _FetchResult(pd.DataFrame(), False)
        try:
            result = method(**kwargs)
        except Exception:
            return _FetchResult(pd.DataFrame(), False)
        if not isinstance(result, pd.DataFrame):
            return _FetchResult(pd.DataFrame(), False)
        return _FetchResult(result.copy(), True)

    def _daily_history(self, start_date: date, end_date: date) -> tuple[pd.DataFrame, bool]:
        frames: list[pd.DataFrame] = []
        complete = True
        current = start_date
        while current <= end_date:
            result = self._fetch("daily", trade_date=current.strftime("%Y%m%d"))
            complete = complete and result.complete
            if len(result.frame) >= _DAILY_ROW_LIMIT:
                complete = False
            elif not result.frame.empty:
                frames.append(result.frame)
            current += timedelta(days=1)
        return (pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(), complete)

    def _latest_rows(
        self, frame: pd.DataFrame, date_column: str, as_of_time: datetime, source: str
    ) -> tuple[pd.DataFrame, date] | None:
        if frame.empty or date_column not in frame:
            return None
        dated = frame.copy()
        dated["_date"] = dated[date_column].map(_row_date)
        dated = dated[dated["_date"].notna()]
        dated = dated[
            dated["_date"].map(
                lambda value: _available_at(source, value, self.next_trading_day) <= as_of_time
            )
        ]
        if dated.empty:
            return None
        latest = max(dated["_date"])
        return dated[dated["_date"] == latest].copy(), latest

    def _expected_observation_date(self, as_of_time: datetime, session_slot: str) -> date:
        local_date = as_of_time.astimezone(_SHANGHAI).date()
        return local_date if _is_close_slot(session_slot) else self.previous_session(local_date)

    def _snapshot(
        self, as_of_time: datetime, session_slot: str, points: tuple[MarketDataPoint, ...]
    ) -> RawMarketSnapshot:
        source_times = {
            source: max(point.data_time for point in points if point.source == source)
            for source in sorted({point.source for point in points})
        }
        raw = RawMarketSnapshot(
            market=Market.A_SHARE,
            as_of_time=as_of_time,
            session_slot=session_slot,
            points=points,
            source_times=source_times,
        )
        return RawMarketSnapshot(
            market=raw.market,
            as_of_time=raw.as_of_time,
            session_slot=raw.session_slot,
            points=raw.points,
            data_status=evaluate_data_quality(
                raw,
                replace(
                    QUALITY_POLICY_V1,
                    previous_session=lambda market, current: self.previous_session(current),
                ),
            ).status,
            source_times=raw.source_times,
        )

    def _load_snapshot_result(
        self,
        market: Market,
        as_of_time: datetime,
        session_slot: str,
        *,
        allow_current_industry: bool = True,
    ) -> _LoadResult:
        if Market(market) != Market.A_SHARE:
            raise ValueError("TushareAShareDataAdapter only supports a_share")
        fetched_at = self.clock() if self.clock is not None else datetime.now(_SHANGHAI)
        local_as_of = as_of_time.astimezone(_SHANGHAI)
        expected_observation_date = self._expected_observation_date(as_of_time, session_slot)
        end_date = expected_observation_date
        start_date = end_date - timedelta(days=45)
        range_query = {"start_date": start_date.strftime("%Y%m%d"), "end_date": end_date.strftime("%Y%m%d")}

        index_results = [self._fetch("index_daily", ts_code=symbol, **range_query) for symbol in _BENCHMARKS]
        daily, daily_complete = self._daily_history(start_date, end_date)
        use_current_industry = (
            allow_current_industry
            and fetched_at.astimezone(_SHANGHAI).date() == local_as_of.date()
        )
        stock_basic = (
            self._fetch(
                "stock_basic",
                exchange="",
                list_status="L",
                fields="ts_code,symbol,name,area,industry,list_status",
            )
            if use_current_industry
            else _FetchResult(pd.DataFrame(), True)
        )
        index_complete = all(result.complete for result in index_results) and all(
            (selected := self._latest_rows(
                result.frame, "trade_date", as_of_time, "tushare_index_daily"
            )) is not None
            and selected[1] == expected_observation_date
            for result in index_results
        )
        complete = daily_complete and stock_basic.complete and index_complete

        market_dates: list[date] = []
        selected_daily = self._latest_rows(daily, "trade_date", as_of_time, "tushare_daily")
        if selected_daily is not None:
            market_dates.append(selected_daily[1])
        for result in index_results:
            selected = self._latest_rows(result.frame, "trade_date", as_of_time, "tushare_index_daily")
            if selected is not None:
                market_dates.append(selected[1])
        latest_market_date = max(market_dates, default=end_date)
        trade_date_query = latest_market_date.strftime("%Y%m%d")
        auxiliary = {
            "daily_basic": self._fetch("daily_basic", trade_date=trade_date_query),
            "margin": self._fetch("margin", trade_date=trade_date_query),
            "shibor": self._fetch("shibor", **range_query),
            "limit": self._fetch("limit_list_d", trade_date=trade_date_query),
            "moneyflow": self._fetch("moneyflow", trade_date=trade_date_query),
        }
        complete = complete and all(result.complete for result in auxiliary.values())

        points = list(
            self._index_points(
                [result.frame for result in index_results],
                fetched_at,
                as_of_time,
                expected_observation_date,
            )
        )
        if daily_complete:
            points.extend(
                self._breadth_points(
                    daily,
                    stock_basic.frame,
                    fetched_at,
                    as_of_time,
                    expected_observation_date,
                )
            )
        points.extend(self._daily_basic_points(auxiliary["daily_basic"].frame, fetched_at, as_of_time))
        points.extend(self._margin_points(auxiliary["margin"].frame, fetched_at, as_of_time))
        points.extend(self._shibor_points(auxiliary["shibor"].frame, fetched_at, as_of_time))
        points.extend(self._limit_points(auxiliary["limit"].frame, daily, fetched_at, as_of_time))
        points.extend(self._moneyflow_points(auxiliary["moneyflow"].frame, fetched_at, as_of_time))
        return _LoadResult(self._snapshot(as_of_time, session_slot, tuple(points)), complete, fetched_at)

    def load_snapshot(self, market: Market, as_of_time: datetime, session_slot: str) -> RawMarketSnapshot:
        return self._load_snapshot_result(market, as_of_time, session_slot).snapshot

    def _index_points(
        self,
        frames: list[pd.DataFrame],
        fetched_at: datetime,
        as_of_time: datetime,
        expected_observation_date: date,
    ) -> Iterator[MarketDataPoint]:
        for frame in frames:
            selected = self._latest_rows(frame, "trade_date", as_of_time, "tushare_index_daily")
            if selected is None or selected[1] != expected_observation_date:
                continue
            rows, trade_date = selected
            row = rows.iloc[0]
            symbol = str(row.get("ts_code") or "")
            close = _number(row.get("close"))
            pre_close = _number(row.get("pre_close"))
            pct_chg = _number(row.get("pct_chg"))
            if pct_chg is None and close is not None and pre_close not in (None, 0):
                pct_chg = (close / pre_close - 1.0) * 100.0
            values = {
                "open": _number(row.get("open")), "high": _number(row.get("high")),
                "low": _number(row.get("low")), "index_price": close,
                "index_change_pct": pct_chg, "volume": _number(row.get("vol")),
                "amount": _number(row.get("amount")),
            }
            for field, value in values.items():
                if value is not None:
                    yield self._point(symbol, field, value, trade_date, fetched_at, "tushare_index_daily")

    def _breadth_points(
        self,
        frame: pd.DataFrame,
        stock_basic: pd.DataFrame,
        fetched_at: datetime,
        as_of_time: datetime,
        expected_observation_date: date,
    ) -> Iterator[MarketDataPoint]:
        selected = self._latest_rows(frame, "trade_date", as_of_time, "tushare_daily")
        if selected is None or selected[1] != expected_observation_date:
            return
        current, trade_date = selected
        current = current.copy()
        current["pct_chg"] = pd.to_numeric(current.get("pct_chg"), errors="coerce")
        current["close"] = pd.to_numeric(current.get("close"), errors="coerce")
        current = current[current["close"].notna()]
        if current.empty:
            return
        values: dict[str, float] = {"breadth_up_pct": float((current["pct_chg"] > 0).mean() * 100.0)}
        history = frame.copy()
        history["_date"] = history["trade_date"].map(_row_date)
        history["close"] = pd.to_numeric(history.get("close"), errors="coerce")
        history = history[
            history["_date"].notna() & history["close"].notna() & (history["_date"] <= trade_date)
        ]
        above: list[bool] = []
        new_low: list[bool] = []
        if "ts_code" in history:
            for _, rows in history.sort_values("_date").groupby("ts_code"):
                closes = rows["close"].tail(20)
                if len(closes) == 20:
                    above.append(bool(closes.iloc[-1] > closes.mean()))
                    new_low.append(bool(closes.iloc[-1] <= closes.min()))
        if above:
            values["breadth_above_ma20_pct"] = float(sum(above) / len(above) * 100.0)
            values["new_low_20d_pct"] = float(sum(new_low) / len(new_low) * 100.0)
        if {"ts_code", "industry"}.issubset(stock_basic.columns):
            industries = stock_basic[["ts_code", "industry"]].dropna().drop_duplicates("ts_code")
            industry_rows = current.merge(industries, on="ts_code", how="left").dropna(subset=["industry"])
            industry_returns = industry_rows.groupby("industry")["pct_chg"].mean()
            if not industry_returns.empty:
                values["industry_decline_pct"] = float((industry_returns < 0).mean() * 100.0)
        for field, value in values.items():
            yield self._point("MARKET", field, value, trade_date, fetched_at, "tushare_daily")

    def _daily_basic_points(
        self, frame: pd.DataFrame, fetched_at: datetime, as_of_time: datetime
    ) -> Iterator[MarketDataPoint]:
        selected = self._latest_rows(frame, "trade_date", as_of_time, "tushare_daily_basic")
        if selected is None:
            return
        rows, trade_date = selected
        for field, column in (("valuation", "pe_ttm"), ("turnover_pct", "turnover_rate")):
            if column in rows:
                values = pd.to_numeric(rows[column], errors="coerce").dropna()
                if not values.empty:
                    yield self._point("MARKET", field, float(values.median()), trade_date, fetched_at, "tushare_daily_basic")

    def _margin_points(
        self, frame: pd.DataFrame, fetched_at: datetime, as_of_time: datetime
    ) -> Iterator[MarketDataPoint]:
        selected = self._latest_rows(frame, "trade_date", as_of_time, "tushare_margin")
        if selected is None:
            return
        rows, trade_date = selected
        for field, column in (("margin_balance", "rzye"), ("margin_buying", "rzmre")):
            if column in rows:
                values = pd.to_numeric(rows[column], errors="coerce").dropna()
                if not values.empty:
                    yield self._point("MARKET", field, float(values.sum()), trade_date, fetched_at, "tushare_margin")

    def _shibor_points(
        self, frame: pd.DataFrame, fetched_at: datetime, as_of_time: datetime
    ) -> Iterator[MarketDataPoint]:
        selected = self._latest_rows(frame, "date", as_of_time, "tushare_shibor")
        if selected is None:
            return
        rows, trade_date = selected
        value = _number(rows.iloc[0].get("3m"))
        if value is not None:
            yield self._point("SHIBOR", "shibor_3m", value, trade_date, fetched_at, "tushare_shibor", hour=11)

    def _limit_points(
        self, frame: pd.DataFrame, daily: pd.DataFrame, fetched_at: datetime, as_of_time: datetime
    ) -> Iterator[MarketDataPoint]:
        selected = self._latest_rows(frame, "trade_date", as_of_time, "tushare_limit_list")
        current_daily = self._latest_rows(daily, "trade_date", as_of_time, "tushare_daily")
        if selected is None or current_daily is None:
            return
        rows, trade_date = selected
        universe, universe_date = current_daily
        if trade_date != universe_date or universe.empty:
            return
        column = "limit" if "limit" in rows else "limit_type" if "limit_type" in rows else None
        if column is None:
            return
        down_count = int(rows[column].astype(str).str.upper().isin({"D", "DOWN"}).sum())
        yield self._point(
            "MARKET", "limit_down_pct", down_count / len(universe) * 100.0,
            trade_date, fetched_at, "tushare_limit_list",
        )

    def _moneyflow_points(
        self, frame: pd.DataFrame, fetched_at: datetime, as_of_time: datetime
    ) -> Iterator[MarketDataPoint]:
        selected = self._latest_rows(frame, "trade_date", as_of_time, "tushare_moneyflow")
        if selected is None:
            return
        rows, trade_date = selected
        if "net_mf_amount" not in rows:
            return
        values = pd.to_numeric(rows["net_mf_amount"], errors="coerce").dropna()
        if not values.empty:
            yield self._point("MARKET", "money_flow_net", float(values.sum()), trade_date, fetched_at, "tushare_moneyflow")

    def _point(
        self,
        symbol: str,
        field: str,
        value: float,
        trade_date: date,
        fetched_at: datetime,
        source: str,
        *,
        hour: int = 15,
    ) -> MarketDataPoint:
        return MarketDataPoint(
            market=Market.A_SHARE,
            symbol=symbol,
            field=field,
            value=value,
            data_time=_data_time(trade_date, hour),
            fetched_at=fetched_at,
            source=source,
            available_at=_available_at(source, trade_date, self.next_trading_day),
        )

    def _cache_query(
        self, current: date, as_of_time: datetime, expected_observation_date: date
    ) -> dict[str, str]:
        return {
            "trade_date": current.strftime("%Y%m%d"),
            "as_of_time": as_of_time.isoformat(),
            "expected_observation_date": expected_observation_date.isoformat(),
            "calendar_version": self.calendar_version,
            "disclosure_policy_version": self.disclosure_policy_version,
        }

    def _read_cached(self, current: date, cache_key: str, query: Mapping[str, Any]) -> RawMarketSnapshot | None:
        if self.cache is None:
            return None
        for year in (current.year, current.year - 1):
            snapshot = self.cache.read_snapshot(
                market=Market.A_SHARE,
                dataset="daily_snapshot",
                year=year,
                cache_key=cache_key,
                query=query,
            )
            if snapshot is not None:
                return snapshot
        return None

    def backfill(self, start_date: date, end_date: date) -> Iterator[RawMarketSnapshot]:
        if end_date < start_date:
            raise ValueError("end_date must not be before start_date")
        current = start_date
        while current <= end_date:
            cache_key = current.isoformat()
            as_of_time = datetime.combine(self.next_trading_day(current), time(9), tzinfo=_SHANGHAI)
            expected_observation_date = self._expected_observation_date(as_of_time, "premarket")
            query = self._cache_query(current, as_of_time, expected_observation_date)
            snapshot = self._read_cached(current, cache_key, query)
            if snapshot is None:
                result = self._load_snapshot_result(
                    Market.A_SHARE,
                    as_of_time,
                    "premarket",
                    allow_current_industry=False,
                )
                snapshot = result.snapshot
                if self.cache is not None and result.complete:
                    self.cache.write_snapshot(
                        dataset="daily_snapshot",
                        cache_key=cache_key,
                        snapshot=snapshot,
                        query=query,
                        source="tushare",
                        fetched_at=result.fetched_at,
                    )
            if snapshot.points and all(
                point.data_time <= (point.available_at or point.fetched_at) <= snapshot.as_of_time
                for point in snapshot.points
            ):
                yield snapshot
            current += timedelta(days=1)
