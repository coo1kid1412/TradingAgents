"""Realtime A-share adapter built on the existing validated quote service."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

from tradingagents.dataflows import intraday_quote
from tradingagents.dataflows.intraday_quote import IntradayQuote
from tradingagents.harness.market_warning.domain import Market, MarketDataPoint, RawMarketSnapshot
from tradingagents.harness.market_warning.quality import evaluate_data_quality


_SHANGHAI = ZoneInfo("Asia/Shanghai")


def _quote_points(quote: IntradayQuote, symbol: str, fetched_at: datetime) -> tuple[MarketDataPoint, ...]:
    change_pct = (quote.last / quote.pre_close - 1.0) * 100.0
    values = {
        "open": quote.open,
        "high": quote.high,
        "low": quote.low,
        "index_price": quote.last,
        "index_change_pct": change_pct,
        "volume": quote.volume,
        "amount": quote.amount,
    }
    return tuple(
        MarketDataPoint(
            market=Market.A_SHARE,
            symbol=symbol,
            field=field,
            value=value,
            data_time=quote.quote_time,
            fetched_at=fetched_at,
            source=quote.source,
            available_at=quote.quote_time,
        )
        for field, value in values.items()
    )


def _cross_section_points(frame: pd.DataFrame | None, as_of_time: datetime, fetched_at: datetime) -> tuple[MarketDataPoint, ...]:
    if frame is None or frame.empty:
        return ()
    required = {"last", "pre_close", "data_time"}
    if not required.issubset(frame.columns):
        return ()
    rows = frame.copy()

    def normalize_time(value: object) -> datetime | None:
        try:
            parsed = pd.Timestamp(value)
        except (TypeError, ValueError):
            return None
        if pd.isna(parsed):
            return None
        return (
            parsed.tz_localize(_SHANGHAI).to_pydatetime()
            if parsed.tzinfo is None
            else parsed.tz_convert(_SHANGHAI).to_pydatetime()
        )

    rows["_data_time"] = rows["data_time"].map(normalize_time)
    rows = rows[rows["_data_time"].notna() & (rows["_data_time"] <= as_of_time)].copy()
    if rows.empty:
        return ()
    rows["last"] = pd.to_numeric(rows["last"], errors="coerce")
    rows["pre_close"] = pd.to_numeric(rows["pre_close"], errors="coerce")
    rows = rows[(rows["last"] > 0) & (rows["pre_close"] > 0)].copy()
    if rows.empty:
        return ()
    data_time = max(rows["_data_time"])
    source = "realtime_cross_section"
    if "source" in rows:
        sources = sorted(
            {str(value).strip() for value in rows["source"].dropna() if str(value).strip()}
        )
        if sources:
            source = "+".join(sources)
    rows["return"] = rows["last"] / rows["pre_close"] - 1.0
    values: dict[str, float] = {"breadth_up_pct": float((rows["return"] > 0).mean() * 100.0)}
    if "ma20" in rows:
        ma20 = pd.to_numeric(rows["ma20"], errors="coerce")
        valid = ma20.notna() & (ma20 > 0)
        if valid.any():
            values["breadth_above_ma20_pct"] = float((rows.loc[valid, "last"] > ma20[valid]).mean() * 100.0)
    if "low_20d" in rows:
        low_20d = pd.to_numeric(rows["low_20d"], errors="coerce")
        valid = low_20d.notna() & (low_20d > 0)
        if valid.any():
            values["new_low_20d_pct"] = float((rows.loc[valid, "last"] <= low_20d[valid]).mean() * 100.0)
    if "industry" in rows and rows["industry"].notna().any():
        industry_returns = rows.dropna(subset=["industry"]).groupby("industry")["return"].mean()
        if not industry_returns.empty:
            values["industry_decline_pct"] = float((industry_returns < 0).mean() * 100.0)
    return tuple(
        MarketDataPoint(
            market=Market.A_SHARE,
            symbol="MARKET",
            field=field,
            value=value,
            data_time=data_time,
            fetched_at=fetched_at,
            source=source,
            available_at=data_time,
        )
        for field, value in values.items()
    )


class RealtimeAShareDataAdapter:
    def __init__(
        self,
        *,
        pro: Any = None,
        symbol: str = "000001.SH",
        quote_loader: Callable[..., IntradayQuote | None] | None = None,
        secondary_quote_loader: Callable[..., IntradayQuote | None] | None = None,
        cross_section_loader: Callable[[datetime], pd.DataFrame | None] | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.pro = pro
        self.symbol = symbol
        self.quote_loader = quote_loader
        self.secondary_quote_loader = secondary_quote_loader
        self.cross_section_loader = cross_section_loader
        self.clock = clock

    def load_snapshot(self, market: Market, as_of_time: datetime, session_slot: str) -> RawMarketSnapshot:
        if Market(market) != Market.A_SHARE:
            raise ValueError("RealtimeAShareDataAdapter only supports a_share")
        fetched_at = self.clock() if self.clock is not None else datetime.now(_SHANGHAI)
        loader = self.quote_loader or intraday_quote.fetch_intraday_quote
        loader_kwargs = {
            "symbol": self.symbol,
            "analysis_date": as_of_time.astimezone(_SHANGHAI).date().isoformat(),
            "tushare_api": self.pro,
            "now": as_of_time,
        }
        quotes = [loader(**loader_kwargs)]
        if self.secondary_quote_loader is not None:
            quotes.append(self.secondary_quote_loader(**loader_kwargs))
        valid_quotes = [quote for quote in quotes if isinstance(quote, IntradayQuote)]
        points = tuple(
            point
            for quote in valid_quotes
            for point in _quote_points(quote, self.symbol, fetched_at)
        )
        if self.cross_section_loader is not None:
            try:
                cross_section = self.cross_section_loader(as_of_time)
            except Exception:
                cross_section = None
            points += _cross_section_points(cross_section, as_of_time, fetched_at)
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
            data_status=evaluate_data_quality(raw).status,
            source_times=raw.source_times,
        )
