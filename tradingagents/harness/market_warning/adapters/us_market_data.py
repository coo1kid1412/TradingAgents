"""Yahoo Finance adapter for normalized US market proxies."""

from __future__ import annotations

from collections.abc import Callable, Iterator
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
    "NDX": "^IXIC",
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


class YahooUSDataAdapter:
    def __init__(
        self,
        *,
        download: Callable[..., pd.DataFrame] = _default_download,
        cache: RawDataCache | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.download = download
        self.cache = cache
        self.clock = clock

    @staticmethod
    def _column(frame: pd.DataFrame, ticker: str, field: str) -> pd.Series:
        if isinstance(frame.columns, pd.MultiIndex):
            for key in ((field, ticker), (ticker, field)):
                if key in frame.columns:
                    return pd.to_numeric(frame[key], errors="coerce")
            return pd.Series(index=frame.index, dtype="float64")
        if field in frame.columns and len(YAHOO_TICKERS) == 1:
            return pd.to_numeric(frame[field], errors="coerce")
        return pd.Series(index=frame.index, dtype="float64")

    @staticmethod
    def _timestamp(value: Any, session_slot: str) -> datetime:
        timestamp = pd.Timestamp(value)
        if timestamp.tzinfo is None:
            timestamp = timestamp.tz_localize(_NEW_YORK)
        else:
            timestamp = timestamp.tz_convert(_NEW_YORK)
        if "close" in session_slot.lower() and timestamp.time() == time.min:
            timestamp = timestamp.replace(hour=16)
        return timestamp.to_pydatetime()

    def load_snapshot(self, market: Market, as_of_time: datetime, session_slot: str) -> RawMarketSnapshot:
        if Market(market) != Market.US:
            raise ValueError("YahooUSDataAdapter only supports us")
        local_as_of = as_of_time.astimezone(_NEW_YORK)
        interval = "5m" if "intraday" in session_slot.lower() else "1d"
        query = {
            "tickers": " ".join(YAHOO_TICKERS.values()),
            "start": local_as_of.date().isoformat(),
            "end": (local_as_of.date() + timedelta(days=1)).isoformat(),
            "interval": interval,
            "auto_adjust": False,
            "progress": False,
            "threads": False,
        }
        frame = self.download(**query)
        fetched_at = self.clock() if self.clock is not None else datetime.now(_NEW_YORK)
        points: list[MarketDataPoint] = []
        for symbol, ticker in YAHOO_TICKERS.items():
            close = self._column(frame, ticker, "Close")
            valid_close = close.dropna()
            valid_close = valid_close[
                [self._timestamp(index, session_slot) <= as_of_time for index in valid_close.index]
            ]
            if valid_close.empty:
                continue
            row_index = valid_close.index[-1]
            data_time = self._timestamp(row_index, session_slot)
            field = "vix" if symbol == "VIX" else "vix3m" if symbol == "VIX3M" else "index_price"
            values: dict[str, float | None] = {field: _number(valid_close.iloc[-1])}
            for normalized, yahoo_field in (("open", "Open"), ("high", "High"), ("low", "Low"), ("volume", "Volume")):
                series = self._column(frame, ticker, yahoo_field)
                values[normalized] = _number(series.get(row_index))
            if len(valid_close) >= 2 and valid_close.iloc[-2] != 0:
                values["index_change_pct"] = float((valid_close.iloc[-1] / valid_close.iloc[-2] - 1.0) * 100.0)
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
        source_times = {"yahoo_finance": max(point.data_time for point in points)} if points else {}
        raw = RawMarketSnapshot(
            market=Market.US,
            as_of_time=as_of_time,
            session_slot=session_slot,
            points=tuple(points),
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

    def backfill(self, start_date: date, end_date: date) -> Iterator[RawMarketSnapshot]:
        if end_date < start_date:
            raise ValueError("end_date must not be before start_date")
        current = start_date
        while current <= end_date:
            cache_key = current.isoformat()
            as_of_time = datetime.combine(current, time(16, 0), tzinfo=_NEW_YORK)
            snapshot = (
                self.cache.read_snapshot(
                    market=Market.US, dataset="daily_snapshot", year=current.year, cache_key=cache_key
                )
                if self.cache is not None
                else None
            )
            if snapshot is None:
                snapshot = self.load_snapshot(Market.US, as_of_time, "close")
                if self.cache is not None:
                    self.cache.write_snapshot(
                        dataset="daily_snapshot",
                        cache_key=cache_key,
                        snapshot=snapshot,
                        query={"trade_date": current.isoformat(), "interval": "1d"},
                        source="yahoo_finance",
                        fetched_at=self.clock() if self.clock is not None else datetime.now(_NEW_YORK),
                    )
            if snapshot.points and all(
                point.data_time <= (point.available_at or point.fetched_at) <= snapshot.as_of_time
                for point in snapshot.points
            ):
                yield snapshot
            current += timedelta(days=1)
