"""Pure point-in-time data quality rules for market warning inputs."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from math import isclose, isfinite
from typing import Any, Callable, Mapping
from zoneinfo import ZoneInfo

from .domain import DataStatus, Market, MarketDataPoint, RawMarketSnapshot
from .ohlc import assess_ohlc_invariants


@dataclass(frozen=True)
class QualityPolicy:
    """Versioned thresholds and field contract for a quality evaluation."""

    version: str = "quality-v1"
    intraday_core_quote_max_age: timedelta = timedelta(minutes=10)
    complete_daily_max_age: timedelta = timedelta(days=14)
    previous_session: Callable[[Market, date], date] | None = None
    cross_source_price_tolerance: float = 0.005
    cross_source_timestamp_skew: timedelta = timedelta(seconds=120)
    daily_disclosure_cutoffs: Mapping[Market, time] = field(
        default_factory=lambda: {
            Market.A_SHARE: time(15, 0),
            Market.US: time(16, 0),
        }
    )
    core_fields: tuple[str, ...] = ("index_price", "index_change_pct")
    optional_fields: tuple[str, ...] = (
        "breadth_up_pct",
        "breadth_above_ma20_pct",
        "volatility",
        "volume",
        "advance_decline",
    )
    fresh_core_coverage: float = 1.0
    reliability_a_optional_coverage: float = 0.70

    @property
    def core_quote_max_age(self) -> timedelta:
        return self.intraday_core_quote_max_age

    @property
    def max_price_deviation(self) -> float:
        return self.cross_source_price_tolerance

    @property
    def max_timestamp_skew(self) -> timedelta:
        return self.cross_source_timestamp_skew

    @property
    def daily_disclosure_cutoff(self) -> Mapping[Market, time]:
        return self.daily_disclosure_cutoffs


QUALITY_POLICY_V1 = QualityPolicy()


_FIELD_ALIASES = {
    "price": "index_price",
    "close": "index_price",
    "last": "index_price",
    "index_level": "index_price",
    "change_pct": "index_change_pct",
    "pct_change": "index_change_pct",
    "return_pct": "index_change_pct",
    "index_direction": "index_change_pct",
    "advancing_pct": "breadth_up_pct",
    "advance_pct": "breadth_up_pct",
    "above_ma20_pct": "breadth_above_ma20_pct",
    "ma20_breadth_pct": "breadth_above_ma20_pct",
    "vix": "volatility",
}

_PRICE_FIELDS = frozenset({"index_price", "open", "high", "low", "close", "last", "price"})

_EXCHANGE_ZONES = {
    Market.A_SHARE: ZoneInfo("Asia/Shanghai"),
    Market.US: ZoneInfo("America/New_York"),
}


@dataclass(frozen=True)
class DataQualityAssessment:
    """Immutable result consumed by feature and model layers."""

    market: Market
    as_of_time: datetime
    session_slot: str
    status: DataStatus
    reliability_grade: str
    selected_points: tuple[MarketDataPoint, ...]
    core_fields: tuple[str, ...]
    optional_fields: tuple[str, ...]
    covered_core_fields: frozenset[str]
    covered_optional_fields: frozenset[str]
    core_coverage: float
    optional_coverage: float
    source_count: int
    core_source_count: int
    reasons: tuple[str, ...] = ()
    latest_data_time: datetime | None = None
    latest_fetched_at: datetime | None = None

    @property
    def data_status(self) -> DataStatus:
        return self.status

    @property
    def quality_status(self) -> DataStatus:
        return self.status

    @property
    def points(self) -> tuple[MarketDataPoint, ...]:
        return self.selected_points

    @property
    def usable(self) -> bool:
        return self.status in {DataStatus.FRESH, DataStatus.PARTIAL, DataStatus.SHADOW}


def _require_aware(value: datetime, name: str) -> None:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware")


def _canonical_field(field_name: str) -> str:
    normalized = str(field_name).strip().lower()
    return _FIELD_ALIASES.get(normalized, normalized)


def _availability_time(point: MarketDataPoint) -> datetime:
    return point.available_at or point.fetched_at


def select_point_in_time(
    points: tuple[MarketDataPoint, ...] | list[MarketDataPoint],
    as_of_time: datetime,
) -> tuple[MarketDataPoint, ...]:
    """Return the latest visible record per source/field at ``as_of_time``.

    A missing ``available_at`` means the record became visible when it was
    fetched. Both visibility and the market timestamp must be no later than
    the requested point in time.
    """

    _require_aware(as_of_time, "as_of_time")
    latest: dict[tuple[Market, str, str, str], tuple[int, MarketDataPoint]] = {}
    for index, point in enumerate(points):
        if not isinstance(point, MarketDataPoint):
            raise ValueError("points must contain MarketDataPoint values")
        if point.data_time > as_of_time or _availability_time(point) > as_of_time:
            continue
        key = (point.market, point.symbol, _canonical_field(point.field), point.source)
        current = latest.get(key)
        rank = (_availability_time(point), point.data_time, point.fetched_at, index)
        if current is None:
            latest[key] = (index, point)
            continue
        current_point = current[1]
        current_rank = (
            _availability_time(current_point),
            current_point.data_time,
            current_point.fetched_at,
            current[0],
        )
        if rank > current_rank:
            latest[key] = (index, point)
    return tuple(point for _, point in sorted(latest.values(), key=lambda item: item[0]))


def _numeric(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if isfinite(number) else None


def _relative_deviation(left: float, right: float) -> float:
    denominator = max(abs(left), abs(right))
    if denominator == 0:
        return 0.0
    return abs(left - right) / denominator


def _exceeds_tolerance(value: float, tolerance: float) -> bool:
    return value > tolerance and not isclose(value, tolerance, rel_tol=0.0, abs_tol=1e-12)


def combine_source_quotes(
    primary: MarketDataPoint,
    secondary: MarketDataPoint,
    tolerance: float,
) -> MarketDataPoint:
    """Combine two same-field quotes, retaining primary's value on agreement."""

    if primary.market != secondary.market or primary.symbol != secondary.symbol:
        raise ValueError("source quotes must belong to the same market and symbol")
    if _canonical_field(primary.field) != _canonical_field(secondary.field):
        raise ValueError("source quotes must describe the same field")
    if isinstance(tolerance, bool) or not isinstance(tolerance, (int, float)) or tolerance < 0:
        raise ValueError("tolerance must be a non-negative number")

    left = _numeric(primary.value)
    right = _numeric(secondary.value)
    price_field = _is_price_field(primary.field)
    timestamp_conflicted = abs(primary.data_time - secondary.data_time) > QUALITY_POLICY_V1.cross_source_timestamp_skew
    value_conflicted = price_field and (
        left is None or right is None or _exceeds_tolerance(_relative_deviation(left, right), float(tolerance))
    )
    conflicted = (
        primary.quality_status == DataStatus.CONFLICTED
        or secondary.quality_status == DataStatus.CONFLICTED
        or timestamp_conflicted
        or value_conflicted
    )
    stale = primary.quality_status == DataStatus.STALE or secondary.quality_status == DataStatus.STALE
    available_times = [value for value in (primary.available_at, secondary.available_at) if value is not None]
    return MarketDataPoint(
        market=primary.market,
        symbol=primary.symbol,
        field=primary.field,
        value=primary.value,
        data_time=max(primary.data_time, secondary.data_time),
        fetched_at=max(primary.fetched_at, secondary.fetched_at),
        source=f"{primary.source}+{secondary.source}",
        quality_status=DataStatus.CONFLICTED if conflicted else DataStatus.STALE if stale else DataStatus.FRESH,
        available_at=max(available_times) if available_times else None,
    )


def _is_daily_slot(session_slot: str) -> bool:
    normalized = session_slot.strip().lower()
    return normalized in {
        "daily", "close", "closing", "eod", "end_of_day", "post_market", "postmarket",
    } or "close" in normalized or "post_market" in normalized or "postmarket" in normalized


def _is_premarket_slot(session_slot: str) -> bool:
    normalized = session_slot.strip().lower()
    return normalized in {"premarket", "pre-market", "before_open"} or "premarket" in normalized


def _source_conflict(
    points: tuple[MarketDataPoint, ...],
    policy: QualityPolicy,
) -> bool:
    grouped: dict[tuple[str, str], list[MarketDataPoint]] = {}
    for point in points:
        grouped.setdefault((point.symbol, _canonical_field(point.field)), []).append(point)
    for quotes in grouped.values():
        sources = {quote.source for quote in quotes}
        if len(sources) < 2:
            continue
        for index, left in enumerate(quotes):
            for right in quotes[index + 1 :]:
                if left.source == right.source:
                    continue
                if abs(left.data_time - right.data_time) > policy.cross_source_timestamp_skew:
                    return True
                if not _is_price_field(left.field):
                    continue
                left_value = _numeric(left.value)
                right_value = _numeric(right.value)
                if left_value is None or right_value is None:
                    return True
                if _exceeds_tolerance(
                    _relative_deviation(left_value, right_value), policy.cross_source_price_tolerance
                ):
                    return True
    return False


def _is_price_field(field_name: str) -> bool:
    return _canonical_field(field_name) in _PRICE_FIELDS


def _source_names(source: str) -> frozenset[str]:
    return frozenset(part for part in source.split("+") if part)


def _coverage(fields: tuple[str, ...], points: tuple[MarketDataPoint, ...]) -> tuple[frozenset[str], float]:
    available = {_canonical_field(point.field) for point in points}
    covered = frozenset(field for field in fields if field in available)
    return covered, len(covered) / len(fields) if fields else 1.0


def evaluate_data_quality(
    snapshot: RawMarketSnapshot,
    policy: QualityPolicy = QUALITY_POLICY_V1,
    now: datetime | None = None,
) -> DataQualityAssessment:
    """Evaluate raw points without consulting vendors, storage, or models."""

    if not isinstance(snapshot, RawMarketSnapshot):
        raise ValueError("snapshot must be a RawMarketSnapshot")
    if not isinstance(policy, QualityPolicy):
        raise ValueError("policy must be a QualityPolicy")
    evaluation_time = now or snapshot.as_of_time
    _require_aware(evaluation_time, "now")

    points = select_point_in_time(snapshot.points, snapshot.as_of_time)
    core_fields = tuple(_canonical_field(value) for value in policy.core_fields)
    optional_fields = tuple(_canonical_field(value) for value in policy.optional_fields)
    covered_core, core_coverage = _coverage(core_fields, points)
    covered_optional, optional_coverage = _coverage(optional_fields, points)
    source_count = len(set().union(*(_source_names(point.source) for point in points))) if points else 0
    core_points = tuple(point for point in points if _canonical_field(point.field) in covered_core)
    core_source_count = (
        len(set().union(*(_source_names(point.source) for point in core_points))) if core_points else 0
    )
    core_dual_source = all(
        len(
            set().union(
                *(
                    _source_names(point.source)
                    for point in core_points
                    if _canonical_field(point.field) == field
                )
            )
        )
        >= 2
        for field in covered_core
    )
    latest_data_time = max((point.data_time for point in points), default=None)
    latest_fetched_at = max((point.fetched_at for point in points), default=None)
    reasons: list[str] = []

    ohlc = assess_ohlc_invariants(
        points,
        snapshot.market,
        snapshot.session_slot,
        timestamp_skew=policy.cross_source_timestamp_skew,
        price_tolerance=policy.cross_source_price_tolerance,
    )
    source_conflicted = _source_conflict(points, policy) or any(
        point.quality_status == DataStatus.CONFLICTED for point in points
    )
    conflicted = source_conflicted or ohlc.conflicted
    point_stale = any(
        point.quality_status == DataStatus.STALE
        for point in points
        if _canonical_field(point.field) in covered_core
    )
    if conflicted:
        status = DataStatus.CONFLICTED
        if source_conflicted:
            reasons.append("independent sources exceed price or timestamp tolerance")
        if ohlc.conflicted:
            reasons.append("selected benchmark OHLC violates finite positive price or range invariants")
    else:
        stale = point_stale or ohlc.stale
        if covered_core:
            core_values = tuple(
                point for point in points if _canonical_field(point.field) in covered_core
            )
            if _is_daily_slot(snapshot.session_slot):
                zone = _EXCHANGE_ZONES[snapshot.market]
                evaluation_date = evaluation_time.astimezone(zone).date()
                stale = stale or any(
                    point.data_time > evaluation_time
                    or point.data_time.astimezone(zone).date() != evaluation_date
                    for point in core_values
                )
            elif _is_premarket_slot(snapshot.session_slot):
                zone = _EXCHANGE_ZONES[snapshot.market]
                evaluation_date = evaluation_time.astimezone(zone).date()
                max_days = policy.complete_daily_max_age.days
                if policy.previous_session is not None:
                    expected_date = policy.previous_session(snapshot.market, evaluation_date)
                    stale = stale or any(
                        point.data_time > evaluation_time
                        or point.data_time.astimezone(zone).date() != expected_date
                        for point in core_values
                    )
                else:
                    stale = stale or any(
                        point.data_time > evaluation_time
                        or (evaluation_date - point.data_time.astimezone(zone).date()).days > max_days
                        for point in core_values
                    )
            else:
                stale = stale or any(
                    evaluation_time - point.data_time > policy.intraday_core_quote_max_age
                    or point.data_time > evaluation_time
                    for point in core_values
                )
        if stale:
            status = DataStatus.STALE
            if point_stale:
                reasons.append("core fields are older than the actual data-time policy")
            if ohlc.stale:
                reasons.append("selected benchmark OHLC contains stale source observations")
        elif ohlc.insufficient:
            status = DataStatus.INSUFFICIENT
            reasons.append("selected benchmark OHLC contains insufficient source observations")
        elif not covered_core:
            status = DataStatus.INSUFFICIENT
            reasons.append("no required core field is available at the point in time")
        elif core_coverage < policy.fresh_core_coverage:
            status = DataStatus.PARTIAL
            reasons.append("required core-field coverage is incomplete")
        else:
            status = DataStatus.FRESH

    if status == DataStatus.FRESH and snapshot.market == Market.US and _is_intraday_slot(snapshot.session_slot):
        if core_source_count == 1:
            status = DataStatus.SHADOW
            reasons.append("US intraday data has one independent source")

    if status in {DataStatus.CONFLICTED, DataStatus.STALE, DataStatus.INSUFFICIENT}:
        reliability_grade = "UNAVAILABLE"
    elif status == DataStatus.SHADOW or status == DataStatus.PARTIAL:
        reliability_grade = "C"
    elif core_dual_source and optional_coverage >= policy.reliability_a_optional_coverage:
        reliability_grade = "A"
    else:
        reliability_grade = "B"

    return DataQualityAssessment(
        market=snapshot.market,
        as_of_time=snapshot.as_of_time,
        session_slot=snapshot.session_slot,
        status=status,
        reliability_grade=reliability_grade,
        selected_points=points,
        core_fields=core_fields,
        optional_fields=optional_fields,
        covered_core_fields=covered_core,
        covered_optional_fields=covered_optional,
        core_coverage=core_coverage,
        optional_coverage=optional_coverage,
        source_count=source_count,
        core_source_count=core_source_count,
        reasons=tuple(reasons),
        latest_data_time=latest_data_time,
        latest_fetched_at=latest_fetched_at,
    )


def _is_intraday_slot(session_slot: str) -> bool:
    normalized = session_slot.strip().lower()
    return normalized in {"intraday", "open", "regular", "session"} or "intraday" in normalized
