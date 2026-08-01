"""Deterministic, point-in-time features for A-share and US market warnings."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime
from math import isfinite
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

from .domain import (
    DataStatus,
    Evidence,
    FeatureSnapshot,
    Market,
    MarketDataPoint,
    MarketPhase,
    RawMarketSnapshot,
)
from .ohlc import (
    A_SHARE_BROAD_BENCHMARKS,
    US_BROAD_BENCHMARKS,
    assess_ohlc_invariants,
    select_benchmark_symbol,
)
from .quality import QUALITY_POLICY_V1, evaluate_data_quality, select_point_in_time


FEATURE_VERSION = "market-warning-v2"

_USABLE_POINT_STATUSES = frozenset(
    {DataStatus.FRESH, DataStatus.PARTIAL, DataStatus.SHADOW}
)


def _metadata(source: str, availability: str, direction: str, unit: str) -> dict[str, str]:
    return {
        "source": source,
        "availability": availability,
        "missing": "preserve as None; emit unavailable evidence",
        "direction": direction,
        "unit": unit,
        "version": FEATURE_VERSION,
    }


_COMMON_METADATA = {
    "return_1d": _metadata("broad index close", "2 observations visible by as_of", "negative is risk-off", "ratio"),
    "audited_ohlc_return_1d": _metadata(
        "two source-coherent broad-index OHLC observations",
        "current and immediately prior market observations each have complete, aligned, quality-usable OHLC",
        "negative is risk-off",
        "ratio",
    ),
    "return_5d": _metadata("broad index close", "6 observations visible by as_of", "negative is risk-off", "ratio"),
    "return_20d": _metadata("broad index close", "21 observations visible by as_of", "negative is risk-off", "ratio"),
    "return_60d": _metadata("broad index close", "61 observations visible by as_of", "negative is risk-off", "ratio"),
    "return_120d": _metadata("broad index close", "121 observations visible by as_of", "negative is risk-off", "ratio"),
    "return_252d": _metadata("broad index close", "253 observations visible by as_of", "negative is risk-off", "ratio"),
    "drawdown_20d": _metadata("broad index close", "20 observations visible by as_of", "more negative is risk-off", "ratio"),
    "drawdown_60d": _metadata("broad index close", "60 observations visible by as_of", "more negative is risk-off", "ratio"),
    "drawdown_252d": _metadata("broad index close", "252 observations visible by as_of", "more negative is risk-off", "ratio"),
    "market_phase": _metadata("drawdown_20d", "drawdown_20d available", "CONTINUATION is riskier", "category"),
    "ma20_distance": _metadata("broad index close", "20 observations visible by as_of", "negative is risk-off", "ratio"),
    "ma50_distance": _metadata("broad index close", "50 observations visible by as_of", "negative is risk-off", "ratio"),
    "ma200_distance": _metadata("broad index close", "200 observations visible by as_of", "negative is risk-off", "ratio"),
    "ma20_slope": _metadata("broad index close", "21 observations visible by as_of", "negative is risk-off", "ratio"),
    "ma50_slope": _metadata("broad index close", "51 observations visible by as_of", "negative is risk-off", "ratio"),
    "ma200_slope": _metadata("broad index close", "201 observations visible by as_of", "negative is risk-off", "ratio"),
    "realized_volatility_5d": _metadata("broad index close", "6 observations visible by as_of", "higher is risk-off", "ratio"),
    "realized_volatility_20d": _metadata("broad index close", "21 observations visible by as_of", "higher is risk-off", "ratio"),
    "realized_volatility_60d": _metadata("broad index close", "61 observations visible by as_of", "higher is risk-off", "ratio"),
    "volatility_ratio_5d_20d": _metadata("broad index close", "21 observations visible by as_of", "higher is risk-off", "ratio"),
    "volatility_ratio_20d_60d": _metadata("broad index close", "61 observations visible by as_of", "higher is risk-off", "ratio"),
    "range_pct": _metadata("index OHLC", "same-session OHLC visible by as_of", "higher is risk-off", "ratio"),
    "range_zscore_20d": _metadata("index OHLC", "20 aligned OHLC observations visible by as_of", "higher is stress", "z-score"),
    "close_location": _metadata("index high low close", "same-session OHLC visible by as_of", "lower is risk-off", "ratio"),
    "volume_zscore_20d": _metadata("broad index volume", "20 observations visible by as_of", "higher is stress", "z-score"),
    "abnormal_range_weak_close_transition": _metadata(
        "index OHLC history and audited OHLC return",
        "range z-score, close location, and audited OHLC 1-day return available",
        "true is risk-off",
        "boolean",
    ),
}

_A_SHARE_METADATA = {
    "breadth_up_pct": _metadata("stock cross section", "point-in-time stock universe available", "lower is risk-off", "percent"),
    "breadth_above_ma20_pct": _metadata("stock cross section", "point-in-time stock history available", "lower is risk-off", "percent"),
    "new_low_20d_pct": _metadata("stock cross section", "point-in-time stock history available", "higher is risk-off", "percent"),
    "industry_decline_pct": _metadata("industry cross section", "point-in-time industry returns available", "higher is risk-off", "percent"),
    "margin_balance_growth_20d": _metadata("margin balance", "21 disclosed observations visible by as_of", "higher is crowded risk", "ratio"),
    "margin_buying": _metadata("margin purchases", "disclosed observation visible by as_of", "higher is crowded risk", "native"),
    "margin_balance_contracting_from_high": _metadata("margin balance", "21 disclosed observations visible by as_of", "true is risk-off", "boolean"),
    "valuation_percentile_20d": _metadata("index valuation", "21 disclosed observations visible by as_of", "higher is valuation risk", "percentile"),
    "turnover_percentile_20d": _metadata("index turnover", "21 disclosed observations visible by as_of", "higher is crowding risk", "percentile"),
    "limit_down_pct": _metadata("limit-down cross section", "point-in-time stock universe available", "higher is risk-off", "percent"),
    "shibor_3m": _metadata("Shibor", "disclosed observation visible by as_of", "higher is funding stress", "percent"),
    "shibor_3m_change_20d": _metadata("Shibor", "21 disclosed observations visible by as_of", "higher is funding stress", "percentage points"),
    "breadth_deterioration_transition": _metadata("stock and industry breadth plus broad index", "current breadth and industry decline with 1-day broad-index return", "true is risk-off", "boolean"),
}

_US_METADATA = {
    "hyg_lqd_relative_return_5d": _metadata("HYG and LQD closes", "6 aligned observations visible by as_of", "more negative is risk-off", "ratio"),
    "hyg_lqd_relative_return_20d": _metadata("HYG and LQD closes", "21 aligned observations visible by as_of", "more negative is risk-off", "ratio"),
    "vix": _metadata("VIX", "observation visible by as_of", "higher is risk-off", "index"),
    "vix_change_5d": _metadata("VIX", "5-market-day span with two aligned endpoints visible by as_of", "higher is risk-off", "ratio"),
    "vix_vix3m_ratio": _metadata("VIX and VIX3M", "aligned observations visible by as_of", "higher is risk-off", "ratio"),
    "russell_spx_relative_return_5d": _metadata("Russell 2000 and S&P 500 closes", "6 aligned observations visible by as_of", "more negative is risk-off", "ratio"),
    "nasdaq_spx_relative_return_5d": _metadata("Nasdaq and S&P 500 closes", "6 aligned observations visible by as_of", "more negative is risk-off", "ratio"),
    "soxx_spx_relative_return_5d": _metadata("SOXX and S&P 500 closes", "6 aligned observations visible by as_of", "more negative is risk-off", "ratio"),
    "credit_volatility_transition": _metadata("HYG LQD VIX VIX3M", "aligned HYG/LQD credit weakness AND (aligned 5-day VIX change OR current aligned VIX/VIX3M ratio)", "true is risk-off", "boolean"),
    "equity_dispersion_transition": _metadata("Russell Nasdaq SOXX and S&P closes", "6 aligned observations for all four equity legs", "true is risk-off", "boolean"),
}

FEATURE_METADATA = {**_COMMON_METADATA, **_A_SHARE_METADATA, **_US_METADATA}

_EVIDENCE_INPUTS: dict[str, tuple[str, str | None]] = {
    **{name: ("index_price", None) for name in _COMMON_METADATA},
    "range_pct": ("high", None),
    "volume_zscore_20d": ("volume", None),
    **{name: (name, None) for name in _A_SHARE_METADATA},
    "margin_balance_growth_20d": ("margin_balance", None),
    "margin_balance_contracting_from_high": ("margin_balance", None),
    "valuation_percentile_20d": ("valuation", None),
    "turnover_percentile_20d": ("turnover_pct", None),
    "shibor_3m_change_20d": ("shibor_3m", None),
    "hyg_lqd_relative_return_5d": ("index_price", "HYG"),
    "hyg_lqd_relative_return_20d": ("index_price", "HYG"),
    "vix": ("vix", "VIX"),
    "vix_change_5d": ("vix", "VIX"),
    "vix_vix3m_ratio": ("vix", "VIX"),
    "russell_spx_relative_return_5d": ("index_price", "RUT"),
    "nasdaq_spx_relative_return_5d": ("index_price", "NDX"),
    "soxx_spx_relative_return_5d": ("index_price", "SOXX"),
    "credit_volatility_transition": ("index_price", "HYG"),
}


def derive_market_phase(drawdown_20d: float | None) -> MarketPhase | None:
    """Classify a shallow drawdown as first shock; -5% is continuation."""

    if drawdown_20d is None:
        return None
    return MarketPhase.FIRST_SHOCK if drawdown_20d > -0.05 else MarketPhase.CONTINUATION


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if isfinite(value) else None


def _canonical_field(field: str) -> str:
    aliases = {"close": "index_price", "price": "index_price", "last": "index_price", "volatility": "vix"}
    return aliases.get(field.strip().lower(), field.strip().lower())


def _visible_history(raw: RawMarketSnapshot, prior_history: Iterable[RawMarketSnapshot]) -> tuple[RawMarketSnapshot, ...]:
    items = [
        item
        for item in prior_history
        if isinstance(item, RawMarketSnapshot) and item.market == raw.market and item.as_of_time < raw.as_of_time
    ]
    grouped: dict[datetime, list[RawMarketSnapshot]] = {}
    for item in items:
        grouped.setdefault(item.as_of_time, []).append(item)
    merged = []
    for as_of_time, snapshots in sorted(grouped.items()):
        points = tuple(point for item in snapshots for point in item.points)
        source_times = {source: timestamp for item in snapshots for source, timestamp in item.source_times.items()}
        merged.append(
            RawMarketSnapshot(
                market=raw.market,
                as_of_time=as_of_time,
                session_slot=snapshots[0].session_slot,
                points=points,
                source_times=source_times,
            )
        )
    merged.append(raw)
    latest_by_day: dict[object, RawMarketSnapshot] = {}
    for item in merged:
        observation_date = _observation_date(item, raw.market) or _market_day(item.as_of_time, raw.market)
        latest_by_day[observation_date] = item
    return tuple(sorted(latest_by_day.values(), key=lambda item: item.as_of_time))


def _market_day(value: datetime, market: Market) -> object:
    zone = ZoneInfo("Asia/Shanghai") if market == Market.A_SHARE else ZoneInfo("America/New_York")
    return value.astimezone(zone).date()


def _is_intraday_slot(session_slot: str) -> bool:
    normalized = session_slot.strip().lower()
    return normalized in {"intraday", "open", "regular", "session"} or "intraday" in normalized


def _is_premarket_slot(session_slot: str) -> bool:
    normalized = session_slot.strip().lower()
    return normalized in {"premarket", "pre-market", "before_open"} or "premarket" in normalized


def _benchmark_point(points: Iterable[MarketDataPoint], market: Market) -> MarketDataPoint | None:
    symbols = A_SHARE_BROAD_BENCHMARKS if market == Market.A_SHARE else US_BROAD_BENCHMARKS
    return next((_point_for(points, "index_price", symbol) for symbol in symbols if _point_for(points, "index_price", symbol)), None)


def _observation_date(snapshot: RawMarketSnapshot, market: Market) -> object | None:
    points = _points_at(snapshot)
    benchmark = _benchmark_point(points, market)
    if benchmark is not None:
        return _market_day(benchmark.data_time, market)
    if not points:
        return None
    return max((_market_day(point.data_time, market) for point in points), default=None)


def _current_observation_is_usable(raw: RawMarketSnapshot) -> bool:
    observation_date = _observation_date(raw, raw.market)
    if observation_date is None:
        return False
    return _is_premarket_slot(raw.session_slot) or observation_date == _market_day(raw.as_of_time, raw.market)


def _aligned(
    points: Iterable[MarketDataPoint | None], market: Market, session_slot: str, observation_date: object | None
) -> bool:
    candidates = tuple(points)
    values = tuple(point for point in candidates if point is not None)
    if not values or len(values) != len(candidates):
        return False
    if observation_date is None or any(_market_day(point.data_time, market) != observation_date for point in values):
        return False
    return not _is_intraday_slot(session_slot) or (
        max(point.data_time for point in values) - min(point.data_time for point in values)
        <= QUALITY_POLICY_V1.cross_source_timestamp_skew
    )


def _history_aligned(
    history: tuple[RawMarketSnapshot, ...], market: Market, inputs: tuple[tuple[str, str], ...]
) -> bool:
    return all(
        _aligned(
            tuple(_point_for(_points_at(item), field, symbol) for field, symbol in inputs),
            market,
            item.session_slot,
            _observation_date(item, market),
        )
        for item in history
    )


def _points_at(snapshot: RawMarketSnapshot) -> tuple[MarketDataPoint, ...]:
    return select_point_in_time(snapshot.points, snapshot.as_of_time)


def _ohlc_at(snapshot: RawMarketSnapshot, symbol: str):
    return assess_ohlc_invariants(
        _points_at(snapshot),
        snapshot.market,
        snapshot.session_slot,
        timestamp_skew=QUALITY_POLICY_V1.cross_source_timestamp_skew,
        price_tolerance=QUALITY_POLICY_V1.cross_source_price_tolerance,
        benchmark_symbol=symbol,
    )


def _point_for(points: Iterable[MarketDataPoint], field: str, symbol: str | None = None) -> MarketDataPoint | None:
    candidates = [
        point
        for point in points
        if _canonical_field(point.field) == _canonical_field(field) and (symbol is None or point.symbol.upper() == symbol.upper())
        and _number(point.value) is not None
        and point.quality_status in _USABLE_POINT_STATUSES
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda point: (point.data_time, point.fetched_at, point.source, point.symbol))


def _series(history: tuple[RawMarketSnapshot, ...], field: str, symbol: str | None = None) -> pd.Series:
    values: list[float] = []
    for item in history:
        point = _point_for(_points_at(item), field, symbol)
        values.append(float("nan") if point is None else float(point.value))
    return pd.Series(values, dtype="float64")


def _observation_series(history: tuple[RawMarketSnapshot, ...], field: str, symbol: str) -> pd.Series:
    values: list[float] = []
    for item in history:
        point = _point_for(_points_at(item), field, symbol)
        aligned = point is not None and _market_day(point.data_time, item.market) == _observation_date(item, item.market)
        values.append(float(point.value) if aligned else float("nan"))
    return pd.Series(values, dtype="float64")


def _range_series(history: tuple[RawMarketSnapshot, ...], symbol: str) -> pd.Series:
    values: list[float] = []
    for item in history:
        ohlc = _ohlc_at(item, symbol)
        open_value = _number(ohlc.open_point.value) if ohlc.open_point is not None else None
        high = _number(ohlc.high_point.value) if ohlc.high_point is not None else None
        low = _number(ohlc.low_point.value) if ohlc.low_point is not None else None
        value = None if not ohlc.valid else (high - low) / open_value
        values.append(float("nan") if value is None else value)
    return pd.Series(values, dtype="float64")


def _audited_ohlc_return_inputs(
    history: tuple[RawMarketSnapshot, ...], symbol: str
) -> tuple[tuple[MarketDataPoint, ...], float | None]:
    """Return exact inputs and return for the latest two consecutive OHLC assessments."""

    if len(history) < 2 or not _current_observation_is_usable(history[-1]):
        return (), None
    previous = _ohlc_at(history[-2], symbol)
    current = _ohlc_at(history[-1], symbol)
    if not previous.valid or not current.valid:
        return (), None
    if _observation_date(history[-2], history[-2].market) == _observation_date(
        history[-1], history[-1].market
    ):
        return (), None

    previous_close = _number(previous.close_point.value) if previous.close_point is not None else None
    current_close = _number(current.close_point.value) if current.close_point is not None else None
    if previous_close in (None, 0.0) or current_close is None:
        return (), None
    inputs = (previous.close_point, *current.points)
    return tuple(point for point in inputs if point is not None), current_close / previous_close - 1.0


def _last(series: pd.Series) -> float | None:
    if series.empty or pd.isna(series.iloc[-1]):
        return None
    return float(series.iloc[-1])


def _return(series: pd.Series, horizon: int) -> float | None:
    return _last(series.pct_change(periods=horizon, fill_method=None))


def _rolling_mean(series: pd.Series, horizon: int) -> float | None:
    return _last(series.rolling(horizon, min_periods=horizon).mean())


def _drawdown(series: pd.Series, horizon: int) -> float | None:
    high = _last(series.rolling(horizon, min_periods=horizon).max())
    current = _last(series)
    return None if high is None or current is None or high == 0 else current / high - 1.0


def _ma_distance(series: pd.Series, horizon: int) -> float | None:
    average = _rolling_mean(series, horizon)
    current = _last(series)
    return None if average is None or current is None or average == 0 else current / average - 1.0


def _ma_slope(series: pd.Series, horizon: int) -> float | None:
    average = series.rolling(horizon, min_periods=horizon).mean()
    if len(average) < 2 or pd.isna(average.iloc[-1]) or pd.isna(average.iloc[-2]) or average.iloc[-2] == 0:
        return None
    return float(average.iloc[-1] / average.iloc[-2] - 1.0)


def _realized_volatility(series: pd.Series, horizon: int) -> float | None:
    returns = series.pct_change(fill_method=None)
    return _last(returns.rolling(horizon, min_periods=horizon).std(ddof=0))


def _ratio(numerator: float | None, denominator: float | None) -> float | None:
    return None if numerator is None or denominator is None or denominator == 0 else numerator / denominator


def _percentile(current: float | None, history: pd.Series, horizon: int) -> float | None:
    window = history.iloc[-(horizon + 1) : -1]
    if current is None or len(window) != horizon or window.isna().any():
        return None
    return float((window <= current).mean())


def _source_times(raw: RawMarketSnapshot, points: Iterable[MarketDataPoint]) -> dict[str, datetime]:
    by_source: dict[str, list[datetime]] = {}
    for point in points:
        if point.data_time <= raw.as_of_time:
            by_source.setdefault(point.source, []).append(point.data_time)
    if not by_source:
        return {"snapshot:first": raw.as_of_time, "snapshot:last": raw.as_of_time}
    return {
        key: value
        for source, times in sorted(by_source.items())
        for key, value in ((f"{source}:first", min(times)), (f"{source}:last", max(times)))
    }


class _FeatureStrategy:
    market: Market
    metadata: dict[str, dict[str, str]]

    def build(self, raw: RawMarketSnapshot, prior_history: Iterable[RawMarketSnapshot]) -> FeatureSnapshot:
        if not isinstance(raw, RawMarketSnapshot) or raw.market != self.market:
            raise ValueError(f"raw must be a {self.market.value} RawMarketSnapshot")
        quality = evaluate_data_quality(raw)
        selected = quality.selected_points
        history = _visible_history(raw, prior_history)
        features = self._common_features(history, selected)
        features.update(self._market_features(history, selected))
        provenance = {name: self._provenance_points(name, history, selected, features) for name in features}
        evidence = tuple(
            self._evidence(name, features[name], provenance[name], raw.as_of_time)
            for name in sorted(features)
        )
        return FeatureSnapshot(
            market=raw.market,
            as_of_time=raw.as_of_time,
            session_slot=raw.session_slot,
            feature_version=FEATURE_VERSION,
            features={name: features[name] for name in sorted(features)},
            evidence=evidence,
            data_quality=quality.status,
            reliability_grade=quality.reliability_grade,
            source_times=_source_times(raw, (point for points in provenance.values() for point in points)),
        )

    def _main_symbol(self, selected: tuple[MarketDataPoint, ...]) -> str | None:
        return select_benchmark_symbol(selected, self.market)

    def _common_features(self, history: tuple[RawMarketSnapshot, ...], selected: tuple[MarketDataPoint, ...]) -> dict[str, Any]:
        symbol = self._main_symbol(selected)
        close = _series(history, "index_price", symbol) if symbol is not None else pd.Series([float("nan")] * len(history))
        if not _current_observation_is_usable(history[-1]):
            close.iloc[-1] = float("nan")
        volumes = _series(history, "volume", symbol) if symbol is not None else pd.Series([float("nan")] * len(history))
        ranges = _range_series(history, symbol) if symbol is not None else pd.Series([float("nan")] * len(history))
        _, audited_ohlc_return_1d = (
            _audited_ohlc_return_inputs(history, symbol) if symbol is not None else ((), None)
        )
        ohlc = (
            assess_ohlc_invariants(
                selected,
                self.market,
                history[-1].session_slot,
                timestamp_skew=QUALITY_POLICY_V1.cross_source_timestamp_skew,
                price_tolerance=QUALITY_POLICY_V1.cross_source_price_tolerance,
                benchmark_symbol=symbol,
            )
            if symbol is not None
            else None
        )
        high_point = ohlc.high_point if ohlc is not None else None
        low_point = ohlc.low_point if ohlc is not None else None
        open_point = ohlc.open_point if ohlc is not None else None
        high = _number(high_point.value) if high_point is not None else None
        low = _number(low_point.value) if low_point is not None else None
        open_value = _number(open_point.value) if open_point is not None else None
        candle_close = (
            _number(ohlc.close_point.value)
            if ohlc is not None and ohlc.close_point is not None
            else None
        )
        if (
            ohlc is not None
            and ohlc.valid
            and candle_close is not None
            and _current_observation_is_usable(history[-1])
        ):
            close.iloc[-1] = candle_close
        current = _last(close)
        vol5 = _realized_volatility(close, 5)
        vol20 = _realized_volatility(close, 20)
        vol60 = _realized_volatility(close, 60)
        volume_mean = _rolling_mean(volumes, 20)
        volume_std = _last(volumes.rolling(20, min_periods=20).std(ddof=0))
        range_mean = _rolling_mean(ranges, 20)
        range_std = _last(ranges.rolling(20, min_periods=20).std(ddof=0))
        range_pct = None if high is None or low is None or open_value in (None, 0) else (high - low) / open_value
        close_location = (
            None
            if high is None or low is None or candle_close is None or high == low
            else (candle_close - low) / (high - low)
        )
        phase = derive_market_phase(_drawdown(close, 20))
        ohlc_valid = (
            _current_observation_is_usable(history[-1])
            and ohlc is not None
            and ohlc.valid
        )
        return_1d = _return(close, 1)
        range_zscore = (
            None
            if range_mean is None or range_std in (None, 0) or _last(ranges) is None
            else (_last(ranges) - range_mean) / range_std
        )
        abnormal_range_transition = (
            None
            if range_zscore is None or close_location is None or audited_ohlc_return_1d is None
            else range_zscore >= 2.0 and close_location <= 0.25 and audited_ohlc_return_1d < 0.0
        )
        return {
            "return_1d": return_1d, "return_5d": _return(close, 5), "return_20d": _return(close, 20),
            "audited_ohlc_return_1d": audited_ohlc_return_1d,
            "return_60d": _return(close, 60), "return_120d": _return(close, 120), "return_252d": _return(close, 252),
            "drawdown_20d": _drawdown(close, 20), "drawdown_60d": _drawdown(close, 60), "drawdown_252d": _drawdown(close, 252),
            "market_phase": phase.value if phase is not None else None,
            "ma20_distance": _ma_distance(close, 20), "ma50_distance": _ma_distance(close, 50), "ma200_distance": _ma_distance(close, 200),
            "ma20_slope": _ma_slope(close, 20), "ma50_slope": _ma_slope(close, 50), "ma200_slope": _ma_slope(close, 200),
            "realized_volatility_5d": vol5, "realized_volatility_20d": vol20, "realized_volatility_60d": vol60,
            "volatility_ratio_5d_20d": _ratio(vol5, vol20), "volatility_ratio_20d_60d": _ratio(vol20, vol60),
            "range_pct": range_pct if ohlc_valid else None,
            "range_zscore_20d": range_zscore if ohlc_valid else None,
            "close_location": close_location if ohlc_valid else None,
            "volume_zscore_20d": None if volume_mean is None or volume_std in (None, 0) or _last(volumes) is None else (_last(volumes) - volume_mean) / volume_std,
            "abnormal_range_weak_close_transition": abnormal_range_transition if ohlc_valid else None,
        }

    def _evidence(self, name: str, value: Any, points: tuple[MarketDataPoint, ...], as_of_time: datetime) -> Evidence:
        metadata = self.metadata[name]
        source_time = max((point.data_time for point in points), default=as_of_time)
        source = "+".join(sorted({point.source for point in points})) or "unavailable"
        state = "unavailable" if value is None else f"value={value}"
        audit_inputs = ""
        if name == "audited_ohlc_return_1d" and points:
            rendered = ",".join(
                f"{point.source}/{point.symbol}/{_canonical_field(point.field)}@"
                f"{point.data_time.isoformat()}={point.value}"
                for point in points
            )
            audit_inputs = f"; inputs=[{rendered}]"
        return Evidence(
            evidence_id=f"{self.market.value}:{FEATURE_VERSION}:{name}:{as_of_time.isoformat()}",
            group="feature",
            summary=f"{name}: {state}; direction={metadata['direction']}{audit_inputs}",
            value=value,
            source=source,
            as_of_time=source_time,
        )

    def _provenance_points(
        self,
        name: str,
        history: tuple[RawMarketSnapshot, ...],
        selected: tuple[MarketDataPoint, ...],
        features: dict[str, Any],
    ) -> tuple[MarketDataPoint, ...]:
        symbol = self._main_symbol(selected)
        if name in _COMMON_METADATA and symbol is None:
            return ()
        if name in {
            "range_pct",
            "range_zscore_20d",
            "close_location",
            "abnormal_range_weak_close_transition",
        }:
            window = 1 if name in {"range_pct", "close_location"} else 20
            points = tuple(
                point
                for item in history[-window:]
                for point in _ohlc_at(item, symbol).points
            )
            if name == "abnormal_range_weak_close_transition":
                audited_points, _ = _audited_ohlc_return_inputs(history, symbol)
                seen: set[tuple[str, str, str, datetime, datetime]] = set()
                combined = []
                for point in (*points, *audited_points):
                    key = (
                        point.source,
                        point.symbol,
                        _canonical_field(point.field),
                        point.data_time,
                        point.fetched_at,
                    )
                    if key not in seen:
                        seen.add(key)
                        combined.append(point)
                return tuple(combined)
            return points
        if name == "audited_ohlc_return_1d":
            points, _ = _audited_ohlc_return_inputs(history, symbol)
            return points
        inputs = [("index_price", symbol)]
        if name in {"range_pct", "close_location"}:
            inputs = [("index_price", symbol), ("open", symbol), ("high", symbol), ("low", symbol)]
        elif name == "range_zscore_20d":
            inputs = [("index_price", symbol), ("open", symbol), ("high", symbol), ("low", symbol)]
        elif name == "abnormal_range_weak_close_transition":
            inputs = [("index_price", symbol), ("open", symbol), ("high", symbol), ("low", symbol)]
        elif name == "volume_zscore_20d":
            inputs = [("volume", symbol)]
        elif name == "breadth_deterioration_transition":
            inputs = [("index_price", symbol), ("breadth_up_pct", None), ("industry_decline_pct", None)]
        elif name in _A_SHARE_METADATA:
            field, field_symbol = _EVIDENCE_INPUTS[name]
            inputs = [(field, field_symbol)]
        elif name in {"hyg_lqd_relative_return_5d", "hyg_lqd_relative_return_20d"}:
            inputs = [("index_price", "HYG"), ("index_price", "LQD")]
        elif name == "credit_volatility_transition":
            inputs = [("index_price", "HYG"), ("index_price", "LQD"), ("vix", "VIX"), ("vix3m", "VIX3M")]
        elif name == "vix_vix3m_ratio":
            inputs = [("vix", "VIX"), ("vix3m", "VIX3M")]
        elif name in {"vix", "vix_change_5d"}:
            inputs = [("vix", "VIX")]
        elif name.endswith("spx_relative_return_5d"):
            leg = {"russell_spx_relative_return_5d": "RUT", "nasdaq_spx_relative_return_5d": "NDX", "soxx_spx_relative_return_5d": "SOXX"}[name]
            inputs = [("index_price", leg), ("index_price", "SPX")]
        elif name == "equity_dispersion_transition":
            inputs = [
                ("index_price", "SPX"),
                ("index_price", "RUT"),
                ("index_price", "NDX"),
                ("index_price", "SOXX"),
            ]
        windows = {
            "return_1d": 2, "audited_ohlc_return_1d": 2, "return_5d": 6, "return_20d": 21, "return_60d": 61,
            "return_120d": 121, "return_252d": 253, "drawdown_20d": 20, "drawdown_60d": 60,
            "drawdown_252d": 252, "market_phase": 20, "ma20_distance": 20, "ma50_distance": 50,
            "ma200_distance": 200, "ma20_slope": 21, "ma50_slope": 51, "ma200_slope": 201,
            "realized_volatility_5d": 6, "realized_volatility_20d": 21, "realized_volatility_60d": 61,
            "volatility_ratio_5d_20d": 21, "volatility_ratio_20d_60d": 61, "volume_zscore_20d": 20,
            "range_zscore_20d": 20, "abnormal_range_weak_close_transition": 20,
            "margin_balance_growth_20d": 21, "margin_balance_contracting_from_high": 21,
            "valuation_percentile_20d": 21, "turnover_percentile_20d": 21, "shibor_3m_change_20d": 21,
            "hyg_lqd_relative_return_5d": 6, "hyg_lqd_relative_return_20d": 21, "vix_change_5d": 6,
            "russell_spx_relative_return_5d": 6, "nasdaq_spx_relative_return_5d": 6,
            "soxx_spx_relative_return_5d": 6, "credit_volatility_transition": 6,
            "breadth_deterioration_transition": 2, "equity_dispersion_transition": 6,
        }
        current_only = {
            "breadth_up_pct", "breadth_above_ma20_pct", "new_low_20d_pct", "industry_decline_pct",
            "margin_buying", "limit_down_pct", "shibor_3m", "vix", "vix_vix3m_ratio", "range_pct", "close_location",
        }
        if name == "vix_change_5d":
            result = []
            for item in (history[-6], history[-1]) if len(history) >= 6 else ():
                point = _point_for(_points_at(item), "vix", "VIX")
                if point is not None and _market_day(point.data_time, item.market) == _observation_date(item, item.market):
                    result.append(point)
            return tuple(result)
        if name == "credit_volatility_transition":
            result = []
            for item in history[-6:]:
                visible = _points_at(item)
                for field, input_symbol in (("index_price", "HYG"), ("index_price", "LQD")):
                    point = _point_for(visible, field, input_symbol)
                    if point is not None:
                        result.append(point)
            if features["vix_change_5d"] is not None and features["vix_change_5d"] >= 0.20:
                for item in (history[-6], history[-1]):
                    point = _point_for(_points_at(item), "vix", "VIX")
                    if point is not None:
                        result.append(point)
            elif features["vix_vix3m_ratio"] is not None and features["vix_vix3m_ratio"] >= 1.0:
                for field, input_symbol in (("vix", "VIX"), ("vix3m", "VIX3M")):
                    point = _point_for(_points_at(history[-1]), field, input_symbol)
                    if point is not None:
                        result.append(point)
            return tuple(result)
        result = []
        window = 1 if name in current_only else windows.get(name, 1)
        for item in history[-window:]:
            visible = _points_at(item)
            for field, input_symbol in inputs:
                point = _point_for(visible, field, input_symbol)
                if point is not None:
                    result.append(point)
        return tuple(result)

    def _market_features(self, history: tuple[RawMarketSnapshot, ...], selected: tuple[MarketDataPoint, ...]) -> dict[str, Any]:
        raise NotImplementedError


class AShareFeatureStrategy(_FeatureStrategy):
    market = Market.A_SHARE
    metadata = {**_COMMON_METADATA, **_A_SHARE_METADATA}

    def _market_features(self, history: tuple[RawMarketSnapshot, ...], selected: tuple[MarketDataPoint, ...]) -> dict[str, Any]:
        margin = _series(history, "margin_balance")
        valuation = _series(history, "valuation")
        turnover = _series(history, "turnover_pct")
        shibor = _series(history, "shibor_3m")
        margin_current = _last(margin)
        prior_margin = margin.iloc[-21] if len(margin) >= 21 and not pd.isna(margin.iloc[-21]) else None
        margin_high = _last(margin.iloc[-21:-1].rolling(20, min_periods=20).max())
        current_valuation = _last(valuation)
        current_turnover = _last(turnover)
        breadth_up = _last(_series(history, "breadth_up_pct"))
        industry_decline = _last(_series(history, "industry_decline_pct"))
        symbol = self._main_symbol(selected)
        broad_return = _return(_series(history, "index_price", symbol), 1) if symbol is not None else None
        breadth_transition = (
            None
            if breadth_up is None or industry_decline is None or broad_return is None
            else breadth_up <= 30.0 and industry_decline >= 70.0 and broad_return < 0.0
        )
        return {
            "breadth_up_pct": breadth_up,
            "breadth_above_ma20_pct": _last(_series(history, "breadth_above_ma20_pct")),
            "new_low_20d_pct": _last(_series(history, "new_low_20d_pct")),
            "industry_decline_pct": industry_decline,
            "margin_balance_growth_20d": None if margin_current is None or prior_margin in (None, 0) else margin_current / float(prior_margin) - 1.0,
            "margin_buying": _last(_series(history, "margin_buying")),
            "margin_balance_contracting_from_high": None if margin_current is None or margin_high is None else margin_current < margin_high,
            "valuation_percentile_20d": _percentile(current_valuation, valuation, 20),
            "turnover_percentile_20d": _percentile(current_turnover, turnover, 20),
            "limit_down_pct": _last(_series(history, "limit_down_pct")),
            "shibor_3m": _last(shibor),
            "shibor_3m_change_20d": None if _last(shibor) is None or len(shibor) < 21 or pd.isna(shibor.iloc[-21]) else _last(shibor) - float(shibor.iloc[-21]),
            "breadth_deterioration_transition": breadth_transition,
        }


class USFeatureStrategy(_FeatureStrategy):
    market = Market.US
    metadata = {**_COMMON_METADATA, **_US_METADATA}

    def _market_features(self, history: tuple[RawMarketSnapshot, ...], selected: tuple[MarketDataPoint, ...]) -> dict[str, Any]:
        hyg = _series(history, "index_price", "HYG")
        lqd = _series(history, "index_price", "LQD")
        spx = _series(history, "index_price", "SPX")
        vix = _observation_series(history, "vix", "VIX")
        vix3m = _series(history, "vix3m", "VIX3M")
        hyg_point = _point_for(selected, "index_price", "HYG")
        lqd_point = _point_for(selected, "index_price", "LQD")
        spx_point = _point_for(selected, "index_price", "SPX")
        vix_point = _point_for(selected, "vix", "VIX")
        vix3m_point = _point_for(selected, "vix3m", "VIX3M")
        hyg5, lqd5 = _return(hyg, 5), _return(lqd, 5)
        hyg20, lqd20 = _return(hyg, 20), _return(lqd, 20)
        credit_current_aligned = _current_observation_is_usable(history[-1]) and _aligned(
            (hyg_point, lqd_point), self.market, history[-1].session_slot, _observation_date(history[-1], self.market)
        )
        credit5_aligned = credit_current_aligned and _history_aligned(
            history[-6:], self.market, (("index_price", "HYG"), ("index_price", "LQD"))
        )
        credit20_aligned = credit_current_aligned and _history_aligned(
            history[-21:], self.market, (("index_price", "HYG"), ("index_price", "LQD"))
        )
        relative5 = hyg5 - lqd5 if credit5_aligned and hyg5 is not None and lqd5 is not None else None
        relative20 = hyg20 - lqd20 if credit20_aligned and hyg20 is not None and lqd20 is not None else None
        vix_change = _return(vix, 5)
        vix_ratio = _ratio(_last(vix), _last(vix3m)) if _current_observation_is_usable(history[-1]) and _aligned(
            (vix_point, vix3m_point), self.market, history[-1].session_slot, _observation_date(history[-1], self.market)
        ) else None
        relative = lambda symbol: self._relative_return(
            _series(history, "index_price", symbol), spx, 5,
            _current_observation_is_usable(history[-1]) and _aligned(
                (_point_for(selected, "index_price", symbol), spx_point), self.market, history[-1].session_slot,
                _observation_date(history[-1], self.market),
            )
            and _history_aligned(history[-6:], self.market, (("index_price", symbol), ("index_price", "SPX"))),
        )
        relative_russell = relative("RUT")
        relative_nasdaq = relative("NDX")
        relative_soxx = relative("SOXX")
        broad_return = _return(spx, 1)
        equity_relative_returns = (relative_russell, relative_nasdaq, relative_soxx)
        equity_dispersion_transition = (
            None
            if broad_return is None or any(value is None for value in equity_relative_returns)
            else broad_return < 0.0 and sum(value <= -0.01 for value in equity_relative_returns) >= 2
        )
        return {
            "hyg_lqd_relative_return_5d": relative5,
            "hyg_lqd_relative_return_20d": relative20,
            "vix": _last(vix),
            "vix_change_5d": vix_change,
            "vix_vix3m_ratio": vix_ratio,
            "russell_spx_relative_return_5d": relative_russell,
            "nasdaq_spx_relative_return_5d": relative_nasdaq,
            "soxx_spx_relative_return_5d": relative_soxx,
            "credit_volatility_transition": None if relative5 is None or (vix_change is None and vix_ratio is None) else relative5 <= -0.015 and (vix_change is not None and vix_change >= 0.20 or vix_ratio is not None and vix_ratio >= 1.0),
            "equity_dispersion_transition": equity_dispersion_transition,
        }

    @staticmethod
    def _relative_return(left: pd.Series, right: pd.Series, horizon: int, aligned: bool) -> float | None:
        left_return, right_return = _return(left, horizon), _return(right, horizon)
        return None if not aligned or left_return is None or right_return is None else left_return - right_return
