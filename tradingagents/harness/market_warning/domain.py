"""Stable, infrastructure-independent domain objects for market warnings."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
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


class RuleLayer(str, Enum):
    VULNERABILITY = "VULNERABILITY"
    PRESSURE = "PRESSURE"
    CONTINUATION = "CONTINUATION"
    HARD_TRIGGER = "HARD_TRIGGER"


class DecisionSource(str, Enum):
    MODEL = "model"
    RULE_V1 = "rule_v1"


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
class TriggeredRule:
    rule_id: str
    layer: RuleLayer
    severity_points: int
    observed_value: Any
    threshold_description: str
    evidence_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.rule_id, str) or not self.rule_id.strip():
            raise ValueError("rule_id must not be empty")
        object.__setattr__(self, "layer", _coerce_enum(self.layer, RuleLayer, "layer"))
        if (
            isinstance(self.severity_points, bool)
            or not isinstance(self.severity_points, int)
            or not 0 <= self.severity_points <= 2
        ):
            raise ValueError("severity_points must be an integer from 0 to 2")
        if not isinstance(self.threshold_description, str) or not self.threshold_description.strip():
            raise ValueError("threshold_description must not be empty")
        evidence_ids = tuple(self.evidence_ids)
        if not evidence_ids or any(not isinstance(item, str) or not item.strip() for item in evidence_ids):
            raise ValueError("evidence_ids must contain non-empty strings")
        if len(evidence_ids) != len(set(evidence_ids)):
            raise ValueError("evidence_ids must be unique")
        object.__setattr__(self, "evidence_ids", evidence_ids)


@dataclass(frozen=True)
class RuleRiskAssessment:
    market: Market
    as_of_time: datetime
    engine_version: str
    manifest_sha256: str
    risk_level: RiskLevel
    risk_score: float
    market_phase: MarketPhase
    triggered_rules: tuple[TriggeredRule, ...]
    missing_optional_groups: tuple[str, ...]
    reliability_grade: str
    evaluation_latency_ms: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "market", _coerce_enum(self.market, Market, "market"))
        object.__setattr__(self, "risk_level", _coerce_enum(self.risk_level, RiskLevel, "risk_level"))
        object.__setattr__(self, "market_phase", _coerce_enum(self.market_phase, MarketPhase, "market_phase"))
        _require_aware(self.as_of_time, "as_of_time")
        if not isinstance(self.engine_version, str) or not self.engine_version.strip():
            raise ValueError("engine_version must not be empty")
        if not isinstance(self.manifest_sha256, str) or len(self.manifest_sha256) != 64:
            raise ValueError("manifest_sha256 must be a 64-character digest")
        if isinstance(self.risk_score, bool) or not isinstance(self.risk_score, Real):
            raise ValueError("risk_score must be a finite number from 0 to 10")
        numeric_score = float(self.risk_score)
        if not isfinite(numeric_score) or not 0 <= numeric_score <= 10:
            raise ValueError("risk_score must be a finite number from 0 to 10")
        if isinstance(self.evaluation_latency_ms, bool) or not isinstance(self.evaluation_latency_ms, Real):
            raise ValueError("evaluation_latency_ms must be a non-negative finite number")
        numeric_latency = float(self.evaluation_latency_ms)
        if not isfinite(numeric_latency) or numeric_latency < 0:
            raise ValueError("evaluation_latency_ms must be a non-negative finite number")
        triggered_rules = tuple(self.triggered_rules)
        if any(not isinstance(item, TriggeredRule) for item in triggered_rules):
            raise ValueError("triggered_rules must contain TriggeredRule values")
        missing_groups = tuple(self.missing_optional_groups)
        if any(not isinstance(item, str) or not item.strip() for item in missing_groups):
            raise ValueError("missing_optional_groups must contain non-empty strings")
        if len(missing_groups) != len(set(missing_groups)):
            raise ValueError("missing_optional_groups must be unique")
        if not isinstance(self.reliability_grade, str) or not self.reliability_grade.strip():
            raise ValueError("reliability_grade must not be empty")
        object.__setattr__(self, "risk_score", numeric_score)
        object.__setattr__(self, "evaluation_latency_ms", numeric_latency)
        object.__setattr__(self, "triggered_rules", triggered_rules)
        object.__setattr__(self, "missing_optional_groups", missing_groups)


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
    decision_source: DecisionSource = DecisionSource.MODEL

    def __post_init__(self) -> None:
        object.__setattr__(self, "baseline_level", _coerce_enum(self.baseline_level, RiskLevel, "baseline_level"))
        object.__setattr__(self, "final_level", _coerce_enum(self.final_level, RiskLevel, "final_level"))
        object.__setattr__(self, "data_status", _coerce_enum(self.data_status, DataStatus, "data_status"))
        object.__setattr__(
            self,
            "decision_source",
            _coerce_enum(self.decision_source, DecisionSource, "decision_source"),
        )
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
class SessionRiskSummary:
    trade_date: date
    highest_level: RiskLevel
    state_changes: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.trade_date, date) or isinstance(self.trade_date, datetime):
            raise TypeError("trade_date must be a date")
        object.__setattr__(
            self,
            "highest_level",
            _coerce_enum(self.highest_level, RiskLevel, "highest_level"),
        )
        changes = tuple(self.state_changes)
        if any(not isinstance(item, str) or not item.strip() for item in changes):
            raise ValueError("state_changes must contain non-empty strings")
        object.__setattr__(self, "state_changes", changes)


@dataclass(frozen=True)
class RunnerResult:
    market: Market
    as_of_time: datetime
    session_slot: str
    feature_snapshot: FeatureSnapshot | None = None
    quant_assessment: QuantRiskAssessment | None = None
    rule_assessment: RuleRiskAssessment | None = None
    shadow_quant_assessment: QuantRiskAssessment | None = None
    context_assessment: LLMContextAssessment | None = None
    previous_decision: FinalWarningDecision | None = None
    previous_session_summary: SessionRiskSummary | None = None
    decision: FinalWarningDecision | None = None
    snapshot_id: int | None = None
    decision_id: int | None = None
    report_path: str | None = None
    notification_confirmed: bool = False
    error_class: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "market", _coerce_enum(self.market, Market, "market"))
        _require_aware(self.as_of_time, "as_of_time")
        if not isinstance(self.notification_confirmed, bool):
            raise TypeError("notification_confirmed must be a bool")
