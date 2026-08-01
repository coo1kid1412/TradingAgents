"""Yahoo Finance adapter for point-in-time normalized US market proxies."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from math import isfinite
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

from tradingagents.harness.market_warning.adapters.data_cache import RawDataCache
from tradingagents.harness.market_warning.domain import Market, MarketDataPoint, RawMarketSnapshot
from tradingagents.harness.market_warning.quality import evaluate_data_quality


_NEW_YORK = ZoneInfo("America/New_York")

YAHOO_TICKERS = {
    "SPX": "^GSPC",
    "NDX": "^NDX",
    "RUT": "^RUT",
    "SOXX": "SOXX",
    "VIX": "^VIX",
    "VIX3M": "^VIX3M",
    "HYG": "HYG",
    "LQD": "LQD",
    "TREASURY": "TLT",
    "USD": "DX-Y.NYB",
}


def _default_download(**kwargs: Any) -> pd.DataFrame:
    import yfinance as yf

    return yf.download(**kwargs)


def _number(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if isfinite(result) else None


@dataclass(frozen=True)
class _LoadResult:
    snapshot: RawMarketSnapshot
    complete: bool
    fetched_at: datetime


class YahooUSDataAdapter:
    def __init__(
        self,
        *,
        download: Callable[..., pd.DataFrame | None] = _default_download,
        ticker_map: Mapping[str, str] | None = None,
        cache: RawDataCache | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.download = download
        self.ticker_map = dict(ticker_map or YAHOO_TICKERS)
        self.cache = cache
        self.clock = clock

    def _column(self, frame: pd.DataFrame, ticker: str, field: str) -> pd.Series:
        if isinstance(frame.columns, pd.MultiIndex):
            for key in ((field, ticker), (ticker, field)):
                if key in frame.columns:
                    return pd.to_numeric(frame[key], errors="coerce")
            return pd.Series(index=frame.index, dtype="float64")
        if len(self.ticker_map) == 1 and field in frame.columns:
            return pd.to_numeric(frame[field], errors="coerce")
        return pd.Series(index=frame.index, dtype="float64")

    @staticmethod
    def _timestamp(value: Any, *, daily: bool) -> datetime:
        timestamp = pd.Timestamp(value)
        if timestamp.tzinfo is None:
            timestamp = timestamp.tz_localize(_NEW_YORK)
        else:
            timestamp = timestamp.tz_convert(_NEW_YORK)
        if daily and timestamp.time() == time.min:
            timestamp = timestamp.replace(hour=16)
        return timestamp.to_pydatetime()

    def _query(self, as_of_time: datetime, interval: str) -> dict[str, Any]:
        local_date = as_of_time.astimezone(_NEW_YORK).date()
        start = local_date - timedelta(days=10) if interval == "1d" else local_date
        return {
            "tickers": " ".join(self.ticker_map.values()),
            "start": start.isoformat(),
            "end": (local_date + timedelta(days=1)).isoformat(),
            "interval": interval,
            "auto_adjust": False,
            "progress": False,
            "threads": False,
        }

    def _download_frame(self, query: Mapping[str, Any]) -> tuple[pd.DataFrame, bool]:
        try:
            frame = self.download(**dict(query))
        except Exception:
            return pd.DataFrame(), False
        if not isinstance(frame, pd.DataFrame):
            return pd.DataFrame(), False
        return frame.copy(), True

    def _visible_close(
        self, frame: pd.DataFrame, ticker: str, as_of_time: datetime, *, daily: bool
    ) -> pd.Series:
        close = self._column(frame, ticker, "Close").dropna()
        if close.empty:
            return close
        mask = [self._timestamp(index, daily=daily) <= as_of_time for index in close.index]
        return close[mask]

    def _has_complete_batch(
        self, frame: pd.DataFrame, as_of_time: datetime, *, daily: bool
    ) -> bool:
        if frame.empty:
            return True
        minimum_rows = 2 if daily else 1
        return all(
            len(self._visible_close(frame, ticker, as_of_time, daily=daily)) >= minimum_rows
            for ticker in self.ticker_map.values()
        )

    def _points(
        self,
        daily_frame: pd.DataFrame,
        quote_frame: pd.DataFrame | None,
        as_of_time: datetime,
        fetched_at: datetime,
    ) -> tuple[MarketDataPoint, ...]:
        points: list[MarketDataPoint] = []
        intraday = quote_frame is not None
        for symbol, ticker in self.ticker_map.items():
            daily_close = self._visible_close(daily_frame, ticker, as_of_time, daily=True)
            if len(daily_close) < 2:
                continue
            if intraday:
                quote_close = self._visible_close(quote_frame, ticker, as_of_time, daily=False)
                if quote_close.empty:
                    continue
                row_index = quote_close.index[-1]
                data_time = self._timestamp(row_index, daily=False)
                current = float(quote_close.iloc[-1])
                previous_close = float(daily_close.iloc[-1])
                value_frame = quote_frame
            else:
                row_index = daily_close.index[-1]
                data_time = self._timestamp(row_index, daily=True)
                current = float(daily_close.iloc[-1])
                previous_close = float(daily_close.iloc[-2])
                value_frame = daily_frame
            field = "vix" if symbol == "VIX" else "vix3m" if symbol == "VIX3M" else "index_price"
            values: dict[str, float | None] = {
                field: current,
                "index_change_pct": None if previous_close == 0 else (current / previous_close - 1.0) * 100.0,
            }
            for normalized, yahoo_field in (
                ("open", "Open"), ("high", "High"), ("low", "Low"), ("volume", "Volume")
            ):
                series = self._column(value_frame, ticker, yahoo_field)
                values[normalized] = _number(series.get(row_index))
            for normalized, value in values.items():
                if value is None:
                    continue
                points.append(
                    MarketDataPoint(
                        market=Market.US,
                        symbol=symbol,
                        field=normalized,
                        value=value,
                        data_time=data_time,
                        fetched_at=fetched_at,
                        source="yahoo_finance",
                        available_at=data_time,
                    )
                )
        return tuple(points)

    @staticmethod
    def _snapshot(as_of_time: datetime, session_slot: str, points: tuple[MarketDataPoint, ...]) -> RawMarketSnapshot:
        source_times = {"yahoo_finance": max(point.data_time for point in points)} if points else {}
        raw = RawMarketSnapshot(
            market=Market.US,
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
            data_status=evaluate_data_quality(raw).status,
            source_times=raw.source_times,
        )

    def _load_snapshot_result(
        self, market: Market, as_of_time: datetime, session_slot: str
    ) -> _LoadResult:
        if Market(market) != Market.US:
            raise ValueError("YahooUSDataAdapter only supports us")
        fetched_at = self.clock() if self.clock is not None else datetime.now(_NEW_YORK)
        daily_frame, daily_complete = self._download_frame(self._query(as_of_time, "1d"))
        if not daily_complete:
            return _LoadResult(self._snapshot(as_of_time, session_slot, ()), False, fetched_at)
        complete = self._has_complete_batch(daily_frame, as_of_time, daily=True)
        quote_frame: pd.DataFrame | None = None
        if "intraday" in session_slot.strip().lower():
            quote_frame, quote_fetched = self._download_frame(self._query(as_of_time, "5m"))
            if not quote_fetched:
                return _LoadResult(self._snapshot(as_of_time, session_slot, ()), False, fetched_at)
            complete = complete and self._has_complete_batch(quote_frame, as_of_time, daily=False)
        points = self._points(daily_frame, quote_frame, as_of_time, fetched_at)
        return _LoadResult(self._snapshot(as_of_time, session_slot, points), complete, fetched_at)

    def load_snapshot(self, market: Market, as_of_time: datetime, session_slot: str) -> RawMarketSnapshot:
        return self._load_snapshot_result(market, as_of_time, session_slot).snapshot

    def _cache_query(self, current: date) -> dict[str, Any]:
        return {
            "trade_date": current.isoformat(),
            "interval": "1d",
            "tickers": list(self.ticker_map.values()),
        }

    def _read_cached(self, current: date, cache_key: str, query: Mapping[str, Any]) -> RawMarketSnapshot | None:
        if self.cache is None:
            return None
        for year in (current.year, current.year - 1):
            snapshot = self.cache.read_snapshot(
                market=Market.US,
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
            query = self._cache_query(current)
            snapshot = self._read_cached(current, cache_key, query)
            if snapshot is None:
                as_of_time = datetime.combine(current, time(16, 0), tzinfo=_NEW_YORK)
                result = self._load_snapshot_result(Market.US, as_of_time, "close")
                snapshot = result.snapshot
                if self.cache is not None and result.complete:
                    self.cache.write_snapshot(
                        dataset="daily_snapshot",
                        cache_key=cache_key,
                        snapshot=snapshot,
                        query=query,
                        source="yahoo_finance",
                        fetched_at=result.fetched_at,
                    )
            if snapshot.points and all(
                point.data_time <= (point.available_at or point.fetched_at) <= snapshot.as_of_time
                for point in snapshot.points
            ):
                yield snapshot
            current += timedelta(days=1)
