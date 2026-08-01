"""Deterministic, point-in-time features for A-share and US market warnings."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime
from math import isfinite
from typing import Any

import pandas as pd

from .domain import Evidence, FeatureSnapshot, Market, MarketDataPoint, MarketPhase, RawMarketSnapshot
from .quality import evaluate_data_quality, select_point_in_time


FEATURE_VERSION = "market-warning-v1"


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
    "return_1d": _metadata("broad index close", "close visible by as_of", "negative is risk-off", "ratio"),
    "return_5d": _metadata("broad index close", "five completed observations visible by as_of", "negative is risk-off", "ratio"),
    "return_20d": _metadata("broad index close", "twenty completed observations visible by as_of", "negative is risk-off", "ratio"),
    "return_60d": _metadata("broad index close", "sixty completed observations visible by as_of", "negative is risk-off", "ratio"),
    "return_120d": _metadata("broad index close", "120 completed observations visible by as_of", "negative is risk-off", "ratio"),
    "return_252d": _metadata("broad index close", "252 completed observations visible by as_of", "negative is risk-off", "ratio"),
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
    "realized_volatility_5d": _metadata("broad index close", "5 return observations visible by as_of", "higher is risk-off", "ratio"),
    "realized_volatility_20d": _metadata("broad index close", "20 return observations visible by as_of", "higher is risk-off", "ratio"),
    "realized_volatility_60d": _metadata("broad index close", "60 return observations visible by as_of", "higher is risk-off", "ratio"),
    "volatility_ratio_5d_20d": _metadata("broad index close", "20 return observations visible by as_of", "higher is risk-off", "ratio"),
    "volatility_ratio_20d_60d": _metadata("broad index close", "60 return observations visible by as_of", "higher is risk-off", "ratio"),
    "range_pct": _metadata("index high low open", "same-session OHLC visible by as_of", "higher is risk-off", "ratio"),
    "close_location": _metadata("index high low close", "same-session OHLC visible by as_of", "lower is risk-off", "ratio"),
    "volume_zscore_20d": _metadata("broad index volume", "20 observations visible by as_of", "higher is stress", "z-score"),
}

_A_SHARE_METADATA = {
    "breadth_up_pct": _metadata("stock cross section", "point-in-time stock universe available", "lower is risk-off", "percent"),
    "breadth_above_ma20_pct": _metadata("stock cross section", "point-in-time stock history available", "lower is risk-off", "percent"),
    "new_low_20d_pct": _metadata("stock cross section", "point-in-time stock history available", "higher is risk-off", "percent"),
    "industry_decline_pct": _metadata("industry cross section", "point-in-time industry returns available", "higher is risk-off", "percent"),
    "margin_balance_growth_20d": _metadata("margin balance", "20 disclosed observations visible by as_of", "higher is crowded risk", "ratio"),
    "margin_buying": _metadata("margin purchases", "disclosed observation visible by as_of", "higher is crowded risk", "native"),
    "margin_balance_contracting_from_high": _metadata("margin balance", "20 disclosed observations visible by as_of", "true is risk-off", "boolean"),
    "valuation_percentile_20d": _metadata("index valuation", "20 disclosed observations visible by as_of", "higher is valuation risk", "percentile"),
    "turnover_percentile_20d": _metadata("index turnover", "20 disclosed observations visible by as_of", "higher is crowding risk", "percentile"),
    "limit_down_pct": _metadata("limit-down cross section", "point-in-time stock universe available", "higher is risk-off", "percent"),
    "shibor_3m": _metadata("Shibor", "disclosed observation visible by as_of", "higher is funding stress", "percent"),
    "shibor_3m_change_20d": _metadata("Shibor", "20 disclosed observations visible by as_of", "higher is funding stress", "percentage points"),
}

_US_METADATA = {
    "hyg_lqd_relative_return_5d": _metadata("HYG and LQD closes", "five completed observations visible by as_of", "more negative is risk-off", "ratio"),
    "hyg_lqd_relative_return_20d": _metadata("HYG and LQD closes", "twenty completed observations visible by as_of", "more negative is risk-off", "ratio"),
    "vix": _metadata("VIX", "observation visible by as_of", "higher is risk-off", "index"),
    "vix_change_5d": _metadata("VIX", "five completed observations visible by as_of", "higher is risk-off", "ratio"),
    "vix_vix3m_ratio": _metadata("VIX and VIX3M", "same-time observations visible by as_of", "higher is risk-off", "ratio"),
    "russell_spx_relative_return_5d": _metadata("Russell 2000 and S&P 500 closes", "five completed observations visible by as_of", "more negative is risk-off", "ratio"),
    "nasdaq_spx_relative_return_5d": _metadata("Nasdaq and S&P 500 closes", "five completed observations visible by as_of", "more negative is risk-off", "ratio"),
    "soxx_spx_relative_return_5d": _metadata("SOXX and S&P 500 closes", "five completed observations visible by as_of", "more negative is risk-off", "ratio"),
    "credit_volatility_transition": _metadata("HYG LQD VIX VIX3M", "credit and volatility inputs visible by as_of", "true is risk-off", "boolean"),
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


def derive_market_phase(drawdown_20d: float | None) -> MarketPhase:
    """Classify a shallow drawdown as first shock; -5% is continuation."""

    return MarketPhase.FIRST_SHOCK if drawdown_20d is None or drawdown_20d > -0.05 else MarketPhase.CONTINUATION


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
    return tuple(merged) + (raw,)


def _points_at(snapshot: RawMarketSnapshot) -> tuple[MarketDataPoint, ...]:
    return select_point_in_time(snapshot.points, snapshot.as_of_time)


def _point_for(points: Iterable[MarketDataPoint], field: str, symbol: str | None = None) -> MarketDataPoint | None:
    candidates = [
        point
        for point in points
        if _canonical_field(point.field) == _canonical_field(field) and (symbol is None or point.symbol.upper() == symbol.upper())
        and _number(point.value) is not None
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


def _source_times(raw: RawMarketSnapshot, points: tuple[MarketDataPoint, ...]) -> dict[str, datetime]:
    times = {source: timestamp for source, timestamp in raw.source_times.items() if timestamp <= raw.as_of_time}
    for point in points:
        current = times.get(point.source)
        if current is None or point.data_time > current:
            times[point.source] = point.data_time
    return dict(sorted(times.items())) or {"snapshot": raw.as_of_time}


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
        evidence = tuple(
            self._evidence(name, features[name], selected, raw.as_of_time)
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
            source_times=_source_times(raw, selected),
        )

    def _main_symbol(self, selected: tuple[MarketDataPoint, ...]) -> str | None:
        preferred = "SPX" if self.market == Market.US else "INDEX"
        if _point_for(selected, "index_price", preferred) is not None:
            return preferred
        point = _point_for(selected, "index_price")
        return point.symbol if point is not None else None

    def _common_features(self, history: tuple[RawMarketSnapshot, ...], selected: tuple[MarketDataPoint, ...]) -> dict[str, Any]:
        symbol = self._main_symbol(selected)
        close = _series(history, "index_price", symbol)
        current = _last(close)
        volumes = _series(history, "volume", symbol)
        high_point = _point_for(selected, "high", symbol)
        low_point = _point_for(selected, "low", symbol)
        open_point = _point_for(selected, "open", symbol)
        high = _number(high_point.value) if high_point is not None else None
        low = _number(low_point.value) if low_point is not None else None
        open_value = _number(open_point.value) if open_point is not None else None
        vol5 = _realized_volatility(close, 5)
        vol20 = _realized_volatility(close, 20)
        vol60 = _realized_volatility(close, 60)
        volume_mean = _rolling_mean(volumes, 20)
        volume_std = _last(volumes.rolling(20, min_periods=20).std(ddof=0))
        range_pct = None if high is None or low is None or open_value in (None, 0) else (high - low) / open_value
        close_location = None if high is None or low is None or current is None or high == low else (current - low) / (high - low)
        return {
            "return_1d": _return(close, 1), "return_5d": _return(close, 5), "return_20d": _return(close, 20),
            "return_60d": _return(close, 60), "return_120d": _return(close, 120), "return_252d": _return(close, 252),
            "drawdown_20d": _drawdown(close, 20), "drawdown_60d": _drawdown(close, 60), "drawdown_252d": _drawdown(close, 252),
            "market_phase": derive_market_phase(_drawdown(close, 20)).value,
            "ma20_distance": _ma_distance(close, 20), "ma50_distance": _ma_distance(close, 50), "ma200_distance": _ma_distance(close, 200),
            "ma20_slope": _ma_slope(close, 20), "ma50_slope": _ma_slope(close, 50), "ma200_slope": _ma_slope(close, 200),
            "realized_volatility_5d": vol5, "realized_volatility_20d": vol20, "realized_volatility_60d": vol60,
            "volatility_ratio_5d_20d": _ratio(vol5, vol20), "volatility_ratio_20d_60d": _ratio(vol20, vol60),
            "range_pct": range_pct, "close_location": close_location,
            "volume_zscore_20d": None if volume_mean is None or volume_std in (None, 0) or _last(volumes) is None else (_last(volumes) - volume_mean) / volume_std,
        }

    def _evidence(self, name: str, value: Any, selected: tuple[MarketDataPoint, ...], as_of_time: datetime) -> Evidence:
        field, symbol = _EVIDENCE_INPUTS[name]
        source_point = _point_for(selected, field, symbol)
        metadata = self.metadata[name]
        source_time = source_point.data_time if source_point is not None else as_of_time
        source = source_point.source if source_point is not None else "unavailable"
        state = "unavailable" if value is None else f"value={value}"
        return Evidence(
            evidence_id=f"{self.market.value}:{FEATURE_VERSION}:{name}:{as_of_time.isoformat()}",
            group="feature",
            summary=f"{name}: {state}; direction={metadata['direction']}",
            value=value,
            source=source,
            as_of_time=source_time,
        )

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
        return {
            "breadth_up_pct": _last(_series(history, "breadth_up_pct")),
            "breadth_above_ma20_pct": _last(_series(history, "breadth_above_ma20_pct")),
            "new_low_20d_pct": _last(_series(history, "new_low_20d_pct")),
            "industry_decline_pct": _last(_series(history, "industry_decline_pct")),
            "margin_balance_growth_20d": None if margin_current is None or prior_margin in (None, 0) else margin_current / float(prior_margin) - 1.0,
            "margin_buying": _last(_series(history, "margin_buying")),
            "margin_balance_contracting_from_high": None if margin_current is None or margin_high is None else margin_current < margin_high,
            "valuation_percentile_20d": _percentile(current_valuation, valuation, 20),
            "turnover_percentile_20d": _percentile(current_turnover, turnover, 20),
            "limit_down_pct": _last(_series(history, "limit_down_pct")),
            "shibor_3m": _last(shibor),
            "shibor_3m_change_20d": None if _last(shibor) is None or len(shibor) < 21 or pd.isna(shibor.iloc[-21]) else _last(shibor) - float(shibor.iloc[-21]),
        }


class USFeatureStrategy(_FeatureStrategy):
    market = Market.US
    metadata = {**_COMMON_METADATA, **_US_METADATA}

    def _market_features(self, history: tuple[RawMarketSnapshot, ...], selected: tuple[MarketDataPoint, ...]) -> dict[str, Any]:
        hyg = _series(history, "index_price", "HYG")
        lqd = _series(history, "index_price", "LQD")
        spx = _series(history, "index_price", "SPX")
        vix = _series(history, "vix", "VIX")
        vix3m = _series(history, "vix3m", "VIX3M")
        hyg5, lqd5 = _return(hyg, 5), _return(lqd, 5)
        hyg20, lqd20 = _return(hyg, 20), _return(lqd, 20)
        relative5 = None if hyg5 is None or lqd5 is None else hyg5 - lqd5
        relative20 = None if hyg20 is None or lqd20 is None else hyg20 - lqd20
        vix_change = _return(vix, 5)
        vix_ratio = _ratio(_last(vix), _last(vix3m))
        relative = lambda symbol: self._relative_return(_series(history, "index_price", symbol), spx, 5)
        return {
            "hyg_lqd_relative_return_5d": relative5,
            "hyg_lqd_relative_return_20d": relative20,
            "vix": _last(vix),
            "vix_change_5d": vix_change,
            "vix_vix3m_ratio": vix_ratio,
            "russell_spx_relative_return_5d": relative("RUT"),
            "nasdaq_spx_relative_return_5d": relative("NDX"),
            "soxx_spx_relative_return_5d": relative("SOXX"),
            "credit_volatility_transition": None if relative5 is None or (vix_change is None and vix_ratio is None) else relative5 <= -0.015 and (vix_change is not None and vix_change >= 0.20 or vix_ratio is not None and vix_ratio >= 1.0),
        }

    @staticmethod
    def _relative_return(left: pd.Series, right: pd.Series, horizon: int) -> float | None:
        left_return, right_return = _return(left, horizon), _return(right, horizon)
        return None if left_return is None or right_return is None else left_return - right_return
