"""Point-in-time normalized Tushare adapter for A-share market data."""

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


_SHANGHAI = ZoneInfo("Asia/Shanghai")
_BENCHMARKS = ("000001.SH", "399001.SZ", "000300.SH", "000905.SH", "399006.SZ", "000688.SH")


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


def _data_time(on_date: date, hour: int = 15) -> datetime:
    return datetime.combine(on_date, time(hour, 0), tzinfo=_SHANGHAI)


def _available_at(on_date: date) -> datetime:
    return datetime.combine(on_date + timedelta(days=1), time.min, tzinfo=_SHANGHAI)


class TushareAShareDataAdapter:
    def __init__(
        self,
        *,
        pro: Any,
        cache: RawDataCache | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.pro = pro
        self.cache = cache
        self.clock = clock

    def _fetch(self, method_name: str, **kwargs: Any) -> pd.DataFrame:
        method = getattr(self.pro, method_name, None)
        if method is None and method_name == "limit_list_d":
            method = getattr(self.pro, "limit_list", None)
        if method is None:
            return pd.DataFrame()
        try:
            result = method(**kwargs)
        except Exception:
            return pd.DataFrame()
        return result.copy() if isinstance(result, pd.DataFrame) else pd.DataFrame()

    def load_snapshot(self, market: Market, as_of_time: datetime, session_slot: str) -> RawMarketSnapshot:
        if Market(market) != Market.A_SHARE:
            raise ValueError("TushareAShareDataAdapter only supports a_share")
        local_as_of = as_of_time.astimezone(_SHANGHAI)
        end_date = local_as_of.date() - timedelta(days=1)
        start_date = end_date - timedelta(days=45)
        query = {"start_date": start_date.strftime("%Y%m%d"), "end_date": end_date.strftime("%Y%m%d")}
        fetched_at = self.clock() if self.clock is not None else datetime.now(_SHANGHAI)
        index_frames = [self._fetch("index_daily", ts_code=symbol, **query) for symbol in _BENCHMARKS]
        daily = self._fetch("daily", **query)
        selected_daily = self._latest_rows(daily, "trade_date", as_of_time)
        market_dates = [selected_daily[1]] if selected_daily is not None else []
        market_dates.extend(
            selected[1]
            for frame in index_frames
            if (selected := self._latest_rows(frame, "trade_date", as_of_time)) is not None
        )
        latest_market_date = max(market_dates, default=end_date)
        trade_date_query = latest_market_date.strftime("%Y%m%d")
        frames = {
            "index": index_frames,
            "daily": daily,
            "daily_basic": self._fetch("daily_basic", trade_date=trade_date_query),
            "margin": self._fetch("margin", trade_date=trade_date_query),
            "shibor": self._fetch("shibor", **query),
            "limit": self._fetch("limit_list_d", trade_date=trade_date_query),
            "moneyflow": self._fetch("moneyflow", trade_date=trade_date_query),
        }
        points = list(self._index_points(frames["index"], fetched_at, as_of_time))
        daily = frames["daily"]
        points.extend(self._breadth_points(daily, fetched_at, as_of_time))
        points.extend(self._daily_basic_points(frames["daily_basic"], fetched_at, as_of_time))
        points.extend(self._margin_points(frames["margin"], fetched_at, as_of_time))
        points.extend(self._shibor_points(frames["shibor"], fetched_at, as_of_time))
        points.extend(self._limit_points(frames["limit"], daily, fetched_at, as_of_time))
        points.extend(self._moneyflow_points(frames["moneyflow"], fetched_at, as_of_time))
        source_times = {
            source: max(point.data_time for point in points if point.source == source)
            for source in sorted({point.source for point in points})
        }
        raw = RawMarketSnapshot(
            market=Market.A_SHARE,
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

    @staticmethod
    def _latest_rows(frame: pd.DataFrame, date_column: str, as_of_time: datetime) -> tuple[pd.DataFrame, date] | None:
        if frame is None or frame.empty or date_column not in frame:
            return None
        dated = frame.copy()
        dated["_date"] = dated[date_column].map(_row_date)
        dated = dated[dated["_date"].notna()]
        if dated.empty:
            return None
        latest = max(dated["_date"])
        if _available_at(latest) > as_of_time:
            return None
        return dated[dated["_date"] == latest].copy(), latest

    def _index_points(
        self, frames: list[pd.DataFrame], fetched_at: datetime, as_of_time: datetime
    ) -> Iterator[MarketDataPoint]:
        for frame in frames:
            selected = self._latest_rows(frame, "trade_date", as_of_time)
            if selected is None:
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
                "open": _number(row.get("open")),
                "high": _number(row.get("high")),
                "low": _number(row.get("low")),
                "index_price": close,
                "index_change_pct": pct_chg,
                "volume": _number(row.get("vol")),
                "amount": _number(row.get("amount")),
            }
            for field, value in values.items():
                if value is not None:
                    yield self._point(symbol, field, value, trade_date, fetched_at, "tushare_index_daily")

    def _breadth_points(
        self, frame: pd.DataFrame, fetched_at: datetime, as_of_time: datetime
    ) -> Iterator[MarketDataPoint]:
        selected = self._latest_rows(frame, "trade_date", as_of_time)
        if selected is None:
            return
        current, trade_date = selected
        current = current.copy()
        current["pct_chg"] = pd.to_numeric(current.get("pct_chg"), errors="coerce")
        current["close"] = pd.to_numeric(current.get("close"), errors="coerce")
        current = current[current["close"].notna()]
        if current.empty:
            return
        values: dict[str, float] = {
            "breadth_up_pct": float((current["pct_chg"] > 0).mean() * 100.0),
        }
        history = frame.copy()
        history["_date"] = history["trade_date"].map(_row_date)
        history["close"] = pd.to_numeric(history.get("close"), errors="coerce")
        history = history[history["_date"].notna() & history["close"].notna() & (history["_date"] <= trade_date)]
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
        if "industry" in current and current["industry"].notna().any():
            industry_returns = current.dropna(subset=["industry"]).groupby("industry")["pct_chg"].mean()
            if not industry_returns.empty:
                values["industry_decline_pct"] = float((industry_returns < 0).mean() * 100.0)
        for field, value in values.items():
            yield self._point("MARKET", field, value, trade_date, fetched_at, "tushare_daily")

    def _daily_basic_points(
        self, frame: pd.DataFrame, fetched_at: datetime, as_of_time: datetime
    ) -> Iterator[MarketDataPoint]:
        selected = self._latest_rows(frame, "trade_date", as_of_time)
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
        selected = self._latest_rows(frame, "trade_date", as_of_time)
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
        selected = self._latest_rows(frame, "date", as_of_time)
        if selected is None:
            return
        rows, trade_date = selected
        value = _number(rows.iloc[0].get("3m"))
        if value is not None:
            yield self._point("SHIBOR", "shibor_3m", value, trade_date, fetched_at, "tushare_shibor", hour=11)

    def _limit_points(
        self, frame: pd.DataFrame, daily: pd.DataFrame, fetched_at: datetime, as_of_time: datetime
    ) -> Iterator[MarketDataPoint]:
        selected = self._latest_rows(frame, "trade_date", as_of_time)
        current_daily = self._latest_rows(daily, "trade_date", as_of_time)
        if selected is None or current_daily is None:
            return
        rows, trade_date = selected
        universe, universe_date = current_daily
        if trade_date != universe_date or universe.empty:
            return
        if "limit" in rows:
            down_count = int(rows["limit"].astype(str).str.upper().isin({"D", "DOWN"}).sum())
        elif "limit_type" in rows:
            down_count = int(rows["limit_type"].astype(str).str.upper().isin({"D", "DOWN"}).sum())
        else:
            return
        yield self._point(
            "MARKET", "limit_down_pct", down_count / len(universe) * 100.0, trade_date, fetched_at, "tushare_limit_list"
        )

    def _moneyflow_points(
        self, frame: pd.DataFrame, fetched_at: datetime, as_of_time: datetime
    ) -> Iterator[MarketDataPoint]:
        selected = self._latest_rows(frame, "trade_date", as_of_time)
        if selected is None:
            return
        rows, trade_date = selected
        if "net_mf_amount" not in rows:
            return
        values = pd.to_numeric(rows["net_mf_amount"], errors="coerce").dropna()
        if not values.empty:
            yield self._point("MARKET", "money_flow_net", float(values.sum()), trade_date, fetched_at, "tushare_moneyflow")

    @staticmethod
    def _point(
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
            available_at=_available_at(trade_date),
        )

    def backfill(self, start_date: date, end_date: date) -> Iterator[RawMarketSnapshot]:
        if end_date < start_date:
            raise ValueError("end_date must not be before start_date")
        current = start_date
        while current <= end_date:
            cache_key = current.isoformat()
            as_of_time = datetime.combine(current + timedelta(days=1), time.min, tzinfo=_SHANGHAI)
            snapshot = (
                self.cache.read_snapshot(
                    market=Market.A_SHARE, dataset="daily_snapshot", year=current.year, cache_key=cache_key
                )
                if self.cache is not None
                else None
            )
            if snapshot is None:
                snapshot = self.load_snapshot(Market.A_SHARE, as_of_time, "premarket")
                if self.cache is not None:
                    self.cache.write_snapshot(
                        dataset="daily_snapshot",
                        cache_key=cache_key,
                        snapshot=snapshot,
                        query={"trade_date": current.strftime("%Y%m%d")},
                        source="tushare",
                        fetched_at=self.clock() if self.clock is not None else datetime.now(_SHANGHAI),
                    )
            if snapshot.points and all(
                point.data_time <= (point.available_at or point.fetched_at) <= snapshot.as_of_time
                for point in snapshot.points
            ):
                yield snapshot
            current += timedelta(days=1)
