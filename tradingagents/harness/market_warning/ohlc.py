"""Shared deterministic OHLC selection and invariant validation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from math import isclose, isfinite
from typing import Any, Iterable
from zoneinfo import ZoneInfo

from .domain import DataStatus, Market, MarketDataPoint


A_SHARE_BROAD_BENCHMARKS = (
    "INDEX",
    "000001.SH",
    "399001.SZ",
    "000300.SH",
    "000300",
    "000905.SH",
    "399006.SZ",
    "000688.SH",
)
US_BROAD_BENCHMARKS = ("SPX", "^GSPC", "INDEX")
DEFAULT_PRICE_TOLERANCE = 0.005

_FIELD_ALIASES = {
    "close": "index_price",
    "last": "index_price",
    "price": "index_price",
    "index_level": "index_price",
}
_OHLC_FIELDS = ("open", "high", "low", "index_price")
_USABLE_QUALITY_STATUSES = frozenset(
    {DataStatus.FRESH, DataStatus.PARTIAL, DataStatus.SHADOW}
)
_EXCHANGE_ZONES = {
    Market.A_SHARE: ZoneInfo("Asia/Shanghai"),
    Market.US: ZoneInfo("America/New_York"),
}


@dataclass(frozen=True)
class OHLCInvariantAssessment:
    """Selected same-benchmark OHLC points and their invariant status."""

    benchmark_symbol: str | None
    open_point: MarketDataPoint | None
    high_point: MarketDataPoint | None
    low_point: MarketDataPoint | None
    close_point: MarketDataPoint | None
    complete: bool
    aligned: bool
    values_valid: bool
    bounds_valid: bool | None
    quality_usable: bool
    conflicted: bool
    stale: bool
    insufficient: bool

    @property
    def valid(self) -> bool:
        return (
            self.complete
            and self.aligned
            and self.values_valid
            and self.bounds_valid is True
            and self.quality_usable
            and not self.conflicted
            and not self.stale
            and not self.insufficient
        )

    @property
    def points(self) -> tuple[MarketDataPoint, ...]:
        return tuple(
            point
            for point in (self.open_point, self.high_point, self.low_point, self.close_point)
            if point is not None
        )


@dataclass(frozen=True)
class _OHLCGroup:
    source: str
    open_point: MarketDataPoint | None
    high_point: MarketDataPoint | None
    low_point: MarketDataPoint | None
    close_point: MarketDataPoint | None
    complete: bool
    aligned: bool
    values_valid: bool
    bounds_valid: bool | None
    quality_usable: bool
    conflicted: bool
    stale: bool
    insufficient: bool

    @property
    def points(self) -> tuple[MarketDataPoint, ...]:
        return tuple(
            point
            for point in (self.open_point, self.high_point, self.low_point, self.close_point)
            if point is not None
        )


def _canonical_field(field: str) -> str:
    normalized = str(field).strip().lower()
    return _FIELD_ALIASES.get(normalized, normalized)


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if isfinite(number) else None


def _is_intraday_slot(session_slot: str) -> bool:
    normalized = str(session_slot).strip().lower()
    return normalized in {"intraday", "open", "regular", "session"} or "intraday" in normalized


def _market_day(point: MarketDataPoint) -> object:
    return point.data_time.astimezone(_EXCHANGE_ZONES[point.market]).date()


def _latest_point(
    points: tuple[MarketDataPoint, ...],
    field: str,
    symbol: str,
    source: str | None = None,
) -> MarketDataPoint | None:
    candidates = tuple(
        point
        for point in points
        if point.symbol.upper() == symbol.upper()
        and _canonical_field(point.field) == field
        and (source is None or point.source == source)
    )
    if not candidates:
        return None
    return max(candidates, key=lambda point: (point.data_time, point.fetched_at, point.source))


def select_benchmark_symbol(
    points: Iterable[MarketDataPoint], market: Market, benchmark_symbol: str | None = None
) -> str | None:
    """Select the configured primary benchmark without filtering bad values."""

    selected_points = tuple(point for point in points if point.market == market)
    allowed = (benchmark_symbol,) if benchmark_symbol is not None else (
        A_SHARE_BROAD_BENCHMARKS if market == Market.A_SHARE else US_BROAD_BENCHMARKS
    )
    return next(
        (
            symbol
            for symbol in allowed
            if symbol is not None and _latest_point(selected_points, "index_price", symbol) is not None
        ),
        None,
    )


def _relative_deviation(left: float, right: float) -> float:
    denominator = max(abs(left), abs(right))
    return 0.0 if denominator == 0.0 else abs(left - right) / denominator


def _assess_group(
    points: tuple[MarketDataPoint, ...],
    symbol: str,
    source: str,
    session_slot: str,
    timestamp_skew: timedelta,
) -> _OHLCGroup:
    by_field = {
        field: _latest_point(points, field, symbol, source)
        for field in _OHLC_FIELDS
    }
    present = tuple(point for point in by_field.values() if point is not None)
    complete = len(present) == len(_OHLC_FIELDS)
    same_day = bool(present) and len({_market_day(point) for point in present}) == 1
    within_skew = not _is_intraday_slot(session_slot) or (
        bool(present)
        and max(point.data_time for point in present) - min(point.data_time for point in present)
        <= timestamp_skew
    )
    aligned = same_day and within_skew
    values = {
        field: _number(point.value) if point is not None else None
        for field, point in by_field.items()
    }
    values_valid = all(
        value is not None and value > 0.0
        for field, value in values.items()
        if by_field[field] is not None
    )
    bounds_valid: bool | None = None
    if complete and aligned and values_valid:
        open_value = values["open"]
        high = values["high"]
        low = values["low"]
        close = values["index_price"]
        assert open_value is not None and high is not None and low is not None and close is not None
        bounds_valid = low <= high and low <= open_value <= high and low <= close <= high

    statuses = {point.quality_status for point in present}
    eligible_group = complete and aligned
    stale = eligible_group and DataStatus.STALE in statuses
    insufficient = eligible_group and DataStatus.INSUFFICIENT in statuses
    status_conflicted = eligible_group and DataStatus.CONFLICTED in statuses
    structural_conflicted = aligned and (
        not values_valid or (complete and bounds_valid is False)
    )
    quality_usable = eligible_group and statuses.issubset(_USABLE_QUALITY_STATUSES)
    return _OHLCGroup(
        source=source,
        open_point=by_field["open"],
        high_point=by_field["high"],
        low_point=by_field["low"],
        close_point=by_field["index_price"],
        complete=complete,
        aligned=aligned,
        values_valid=values_valid,
        bounds_valid=bounds_valid,
        quality_usable=quality_usable,
        conflicted=status_conflicted or structural_conflicted,
        stale=stale,
        insufficient=insufficient,
    )


def _group_rank(group: _OHLCGroup) -> tuple[bool, bool, int, object, object, str]:
    latest_data_time = max(point.data_time for point in group.points)
    latest_fetched_at = max(point.fetched_at for point in group.points)
    return (
        group.complete,
        group.aligned,
        len(group.points),
        latest_data_time,
        latest_fetched_at,
        group.source,
    )


def _complete_groups_disagree(
    groups: tuple[_OHLCGroup, ...],
    *,
    timestamp_skew: timedelta,
    price_tolerance: float,
) -> bool:
    for index, left in enumerate(groups):
        for right in groups[index + 1 :]:
            for left_point, right_point in zip(left.points, right.points):
                if abs(left_point.data_time - right_point.data_time) > timestamp_skew:
                    return True
                left_value = _number(left_point.value)
                right_value = _number(right_point.value)
                if left_value is None or right_value is None:
                    return True
                deviation = _relative_deviation(left_value, right_value)
                if deviation > price_tolerance and not isclose(
                    deviation, price_tolerance, rel_tol=0.0, abs_tol=1e-12
                ):
                    return True
    return False


def assess_ohlc_invariants(
    points: Iterable[MarketDataPoint],
    market: Market,
    session_slot: str,
    *,
    timestamp_skew: timedelta,
    price_tolerance: float = DEFAULT_PRICE_TOLERANCE,
    benchmark_symbol: str | None = None,
) -> OHLCInvariantAssessment:
    """Validate a single selected benchmark's aligned current OHLC observation."""

    selected_points = tuple(point for point in points if point.market == market)
    symbol = select_benchmark_symbol(selected_points, market, benchmark_symbol)
    if symbol is None:
        return OHLCInvariantAssessment(
            None, None, None, None, None, False, False, True, None, False, False, False, False
        )

    symbol_points = tuple(
        point
        for point in selected_points
        if point.symbol.upper() == symbol.upper() and _canonical_field(point.field) in _OHLC_FIELDS
    )
    groups = tuple(
        _assess_group(
            symbol_points,
            symbol,
            source,
            session_slot,
            timestamp_skew,
        )
        for source in sorted({point.source for point in symbol_points})
    )
    selected = max(groups, key=_group_rank)
    complete_groups = tuple(group for group in groups if group.complete and group.aligned)
    conflicted = any(group.conflicted for group in groups) or _complete_groups_disagree(
        complete_groups,
        timestamp_skew=timestamp_skew,
        price_tolerance=price_tolerance,
    )
    stale = any(group.stale for group in complete_groups)
    insufficient = any(group.insufficient for group in complete_groups)
    quality_usable = (
        bool(complete_groups)
        and all(group.quality_usable for group in complete_groups)
        and not conflicted
        and not stale
        and not insufficient
    )

    return OHLCInvariantAssessment(
        benchmark_symbol=symbol,
        open_point=selected.open_point,
        high_point=selected.high_point,
        low_point=selected.low_point,
        close_point=selected.close_point,
        complete=selected.complete,
        aligned=selected.aligned,
        values_valid=selected.values_valid,
        bounds_valid=selected.bounds_valid,
        quality_usable=quality_usable,
        conflicted=conflicted,
        stale=stale,
        insufficient=insufficient,
    )
