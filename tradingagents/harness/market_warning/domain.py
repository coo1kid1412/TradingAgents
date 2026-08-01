"""Stable, infrastructure-independent domain objects for market warnings."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from math import isfinite
from numbers import Real
from types import MappingProxyType
from typing import Any, Mapping


class Market(str, Enum):
    A_SHARE = "a_share"
    US = "us"


class DataStatus(str, Enum):
    FRESH = "fresh"
    PARTIAL = "partial"
    CONFLICTED = "conflicted"
    STALE = "stale"
    INSUFFICIENT = "insufficient"
    SHADOW = "shadow"


class RiskLevel(str, Enum):
    GREEN = "GREEN"
    YELLOW = "YELLOW"
    ORANGE = "ORANGE"
    RED = "RED"
    UNKNOWN = "UNKNOWN"


class MarketPhase(str, Enum):
    FIRST_SHOCK = "FIRST_SHOCK"
    CONTINUATION = "CONTINUATION"


def _coerce_enum(value: Any, enum_type: type[Enum], field_name: str) -> Enum:
    try:
        return value if isinstance(value, enum_type) else enum_type(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a valid {enum_type.__name__}") from exc


def _require_aware(value: datetime, field_name: str) -> None:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")


def _require_probability(value: Real, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{field_name} must be a finite number from 0 to 1")
    numeric = float(value)
    if not isfinite(numeric) or not 0 <= numeric <= 1:
        raise ValueError(f"{field_name} must be a finite number from 0 to 1")


def _require_percentage(value: Real, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{field_name} must be a finite percentage from 0 to 100")
    numeric = float(value)
    if not isfinite(numeric) or not 0 <= numeric <= 100:
        raise ValueError(f"{field_name} must be a finite percentage from 0 to 100")


@dataclass(frozen=True)
class MarketDataPoint:
    market: Market
    symbol: str
    field: str
    value: Any
    data_time: datetime
    fetched_at: datetime
    source: str
    quality_status: DataStatus = DataStatus.FRESH
    available_at: datetime | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "market", _coerce_enum(self.market, Market, "market"))
        object.__setattr__(self, "quality_status", _coerce_enum(self.quality_status, DataStatus, "quality_status"))
        _require_aware(self.data_time, "data_time")
        _require_aware(self.fetched_at, "fetched_at")
        if self.available_at is not None:
            _require_aware(self.available_at, "available_at")


@dataclass(frozen=True)
class RawMarketSnapshot:
    market: Market
    as_of_time: datetime
    session_slot: str
    points: tuple[MarketDataPoint, ...] = ()
    data_status: DataStatus = DataStatus.FRESH
    source_times: Mapping[str, datetime] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "market", _coerce_enum(self.market, Market, "market"))
        object.__setattr__(self, "data_status", _coerce_enum(self.data_status, DataStatus, "data_status"))
        _require_aware(self.as_of_time, "as_of_time")
        object.__setattr__(self, "points", tuple(self.points))
        for point in self.points:
            if not isinstance(point, MarketDataPoint):
                raise ValueError("points must contain MarketDataPoint values")
        source_times = dict(self.source_times)
        for timestamp in source_times.values():
            _require_aware(timestamp, "source_times value")
        object.__setattr__(self, "source_times", MappingProxyType(source_times))


@dataclass(frozen=True)
class Evidence:
    evidence_id: str
    group: str
    summary: str
    value: Any = None
    source: str | None = None
    as_of_time: datetime | None = None

    def __post_init__(self) -> None:
        if not self.evidence_id:
            raise ValueError("evidence_id must not be empty")
        if self.as_of_time is not None:
            _require_aware(self.as_of_time, "as_of_time")


@dataclass(frozen=True)
class FeatureSnapshot:
    market: Market
    as_of_time: datetime
    session_slot: str
    feature_version: str
    features: Mapping[str, Any]
    evidence: tuple[Evidence, ...]
    data_quality: DataStatus
    reliability_grade: str = "D"
    source_times: Mapping[str, datetime] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "market", _coerce_enum(self.market, Market, "market"))
        object.__setattr__(self, "data_quality", _coerce_enum(self.data_quality, DataStatus, "data_quality"))
        _require_aware(self.as_of_time, "as_of_time")
        evidence = tuple(self.evidence)
        if any(not isinstance(item, Evidence) for item in evidence):
            raise ValueError("evidence must contain Evidence values")
        evidence_ids = tuple(item.evidence_id for item in evidence)
        if len(evidence_ids) != len(set(evidence_ids)):
            raise ValueError("evidence IDs must be unique")
        if not isinstance(self.reliability_grade, str) or not self.reliability_grade:
            raise ValueError("reliability_grade must not be empty")
        source_times = dict(self.source_times)
        for timestamp in source_times.values():
            _require_aware(timestamp, "source_times value")
        object.__setattr__(self, "features", MappingProxyType(dict(self.features)))
        object.__setattr__(self, "evidence", evidence)
        object.__setattr__(self, "source_times", MappingProxyType(source_times))

    @property
    def evidence_ids(self) -> tuple[str, ...]:
        return tuple(item.evidence_id for item in self.evidence)


@dataclass(frozen=True)
class QuantRiskAssessment:
    crash_1d_probability: float
    crash_3d_probability: float
    market_phase: MarketPhase
    base_rate_1d: float
    base_rate_3d: float
    reliability_grade: str
    model_version: str
    calibration_version: str
    top_contributors: tuple[Any, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "market_phase", _coerce_enum(self.market_phase, MarketPhase, "market_phase"))
        for name in ("crash_1d_probability", "crash_3d_probability", "base_rate_1d", "base_rate_3d"):
            _require_probability(getattr(self, name), name)
        object.__setattr__(self, "top_contributors", tuple(self.top_contributors))


@dataclass(frozen=True)
class LLMContextAssessment:
    market_scenario: str
    causal_chain: tuple[str, ...]
    supporting_evidence_ids: tuple[str, ...]
    conflicting_evidence_ids: tuple[str, ...]
    overlooked_risks: tuple[str, ...]
    recommended_risk_level: RiskLevel
    confidence: float
    action_reason: str
    reasoning_status: str
    error_class: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "recommended_risk_level",
            _coerce_enum(self.recommended_risk_level, RiskLevel, "recommended_risk_level"),
        )
        _require_probability(self.confidence, "confidence")
        object.__setattr__(self, "causal_chain", tuple(self.causal_chain))
        object.__setattr__(self, "supporting_evidence_ids", tuple(self.supporting_evidence_ids))
        object.__setattr__(self, "conflicting_evidence_ids", tuple(self.conflicting_evidence_ids))
        object.__setattr__(self, "overlooked_risks", tuple(self.overlooked_risks))
        if self.error_class is not None and (
            not isinstance(self.error_class, str) or not self.error_class.strip()
        ):
            raise ValueError("error_class must be a non-empty string or None")


@dataclass(frozen=True)
class FinalWarningDecision:
    baseline_level: RiskLevel
    final_level: RiskLevel
    state_transition: str
    entry_gate: str
    new_position_cap_pct: float
    holding_action: str
    push_required: bool
    decision_reasons: tuple[str, ...]
    data_status: DataStatus
    valid_snapshot_count: int = 0
    retained_risk_level: RiskLevel | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "baseline_level", _coerce_enum(self.baseline_level, RiskLevel, "baseline_level"))
        object.__setattr__(self, "final_level", _coerce_enum(self.final_level, RiskLevel, "final_level"))
        object.__setattr__(self, "data_status", _coerce_enum(self.data_status, DataStatus, "data_status"))
        _require_percentage(self.new_position_cap_pct, "new_position_cap_pct")
        if (
            isinstance(self.valid_snapshot_count, bool)
            or not isinstance(self.valid_snapshot_count, int)
            or self.valid_snapshot_count < 0
        ):
            raise ValueError("valid_snapshot_count must be a non-negative integer")
        if self.retained_risk_level is not None:
            retained = _coerce_enum(self.retained_risk_level, RiskLevel, "retained_risk_level")
            if retained not in {RiskLevel.ORANGE, RiskLevel.RED}:
                raise ValueError("retained_risk_level must be ORANGE, RED, or None")
            object.__setattr__(self, "retained_risk_level", retained)
        object.__setattr__(self, "decision_reasons", tuple(self.decision_reasons))


@dataclass(frozen=True)
class RunnerResult:
    market: Market
    as_of_time: datetime
    session_slot: str
    feature_snapshot: FeatureSnapshot | None = None
    quant_assessment: QuantRiskAssessment | None = None
    context_assessment: LLMContextAssessment | None = None
    decision: FinalWarningDecision | None = None
    report_path: str | None = None
    error_class: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "market", _coerce_enum(self.market, Market, "market"))
        _require_aware(self.as_of_time, "as_of_time")
