"""Pure deterministic warning policy and state transitions."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from numbers import Real
from typing import Iterable

from .domain import (
    DataStatus,
    FeatureSnapshot,
    FinalWarningDecision,
    LLMContextAssessment,
    Market,
    QuantRiskAssessment,
    RiskLevel,
)


POLICY_VERSION = "market-warning-policy-v1"
HARD_TRIGGER_VERSION = "market-warning-hard-triggers-v1"

_UNUSABLE_DATA = frozenset({DataStatus.CONFLICTED, DataStatus.STALE, DataStatus.INSUFFICIENT})
_ORDERED_LEVELS = (RiskLevel.GREEN, RiskLevel.YELLOW, RiskLevel.ORANGE, RiskLevel.RED)
_LEVEL_INDEX = {level: index for index, level in enumerate(_ORDERED_LEVELS)}
_TRANSITION_SIGNAL_FIELDS = (
    "pressure_transition_signal",
    "volatility_acceleration_transition",
    "abnormal_range_weak_close_transition",
    "breadth_deterioration_transition",
    "credit_volatility_transition",
)


@dataclass(frozen=True)
class StateTransitionResult:
    """The complete state update returned to an external persistence layer."""

    final_level: RiskLevel
    state_transition: str
    push_required: bool
    persist_required: bool
    next_valid_snapshot_count: int
    retained_risk_level: RiskLevel | None = None


def _number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, Real):
        return None
    numeric = float(value)
    return numeric if isfinite(numeric) else None


def _probability_ratio(probability: object, base_rate: object) -> float | None:
    probability_value = _number(probability)
    base_rate_value = _number(base_rate)
    if (
        probability_value is None
        or base_rate_value is None
        or not 0.0 <= probability_value <= 1.0
        or not 0.0 < base_rate_value <= 1.0
    ):
        return None
    return probability_value / base_rate_value


def _feature(snapshot: FeatureSnapshot, name: str) -> float | None:
    return _number(snapshot.features.get(name))


def _has_transition_signal(snapshot: FeatureSnapshot) -> bool:
    return any(snapshot.features.get(name) is True for name in _TRANSITION_SIGNAL_FIELDS)


def _market_range_hard_trigger(snapshot: FeatureSnapshot) -> bool:
    range_zscore = _feature(snapshot, "range_zscore_20d")
    close_location = _feature(snapshot, "close_location")
    daily_return = _feature(snapshot, "return_1d")
    return_threshold = -0.020 if snapshot.market == Market.A_SHARE else -0.015
    return (
        range_zscore is not None
        and close_location is not None
        and daily_return is not None
        and range_zscore >= 3.0
        and close_location <= 0.15
        and daily_return <= return_threshold
    )


def _a_share_breadth_hard_trigger(snapshot: FeatureSnapshot) -> bool:
    if snapshot.market != Market.A_SHARE:
        return False
    daily_return = _feature(snapshot, "return_1d")
    if daily_return is None or daily_return >= 0.0:
        return False

    declining_share = _feature(snapshot, "declining_stock_pct")
    breadth_up = _feature(snapshot, "breadth_up_pct")
    if declining_share is None and breadth_up is not None and 0.0 <= breadth_up <= 100.0:
        declining_share = 100.0 - breadth_up
    limit_down_share = _feature(snapshot, "limit_down_pct")
    return (
        declining_share is not None
        and 0.0 <= declining_share <= 100.0
        and declining_share >= 85.0
    ) or (
        limit_down_share is not None
        and 0.0 <= limit_down_share <= 100.0
        and limit_down_share >= 2.0
    )


def _us_credit_hard_trigger(snapshot: FeatureSnapshot) -> bool:
    if snapshot.market != Market.US:
        return False
    relative_return = _feature(snapshot, "hyg_lqd_relative_return_5d")
    vix_change = _feature(snapshot, "vix_change_5d")
    term_structure = _feature(snapshot, "vix_vix3m_ratio")
    return relative_return is not None and relative_return <= -0.015 and (
        (vix_change is not None and vix_change >= 0.20)
        or (term_structure is not None and term_structure >= 1.0)
    )


def _has_hard_trigger(snapshot: FeatureSnapshot) -> bool:
    return (
        _market_range_hard_trigger(snapshot)
        or _a_share_breadth_hard_trigger(snapshot)
        or _us_credit_hard_trigger(snapshot)
    )


def baseline_level(quant: QuantRiskAssessment, snapshot: FeatureSnapshot) -> RiskLevel:
    """Map calibrated probabilities and deterministic hard evidence to a baseline level."""

    if not isinstance(snapshot, FeatureSnapshot):
        return RiskLevel.UNKNOWN
    if snapshot.data_quality in _UNUSABLE_DATA:
        return RiskLevel.UNKNOWN
    if str(getattr(quant, "reliability_grade", "UNAVAILABLE")).strip().upper() == "UNAVAILABLE":
        return RiskLevel.UNKNOWN
    if snapshot.reliability_grade.strip().upper() == "UNAVAILABLE":
        return RiskLevel.UNKNOWN

    ratio_1d = _probability_ratio(
        getattr(quant, "crash_1d_probability", None),
        getattr(quant, "base_rate_1d", None),
    )
    ratio_3d = _probability_ratio(
        getattr(quant, "crash_3d_probability", None),
        getattr(quant, "base_rate_3d", None),
    )
    if ratio_1d is None or ratio_3d is None:
        return RiskLevel.UNKNOWN
    if _has_hard_trigger(snapshot):
        return RiskLevel.RED

    strictest_ratio = max(ratio_1d, ratio_3d)
    if strictest_ratio >= 8.0:
        return RiskLevel.RED
    if strictest_ratio >= 4.0 and _has_transition_signal(snapshot):
        return RiskLevel.ORANGE
    if strictest_ratio >= 2.0:
        return RiskLevel.YELLOW
    return RiskLevel.GREEN


def apply_llm_adjustment(
    baseline: RiskLevel,
    context: LLMContextAssessment | None,
    snapshot: FeatureSnapshot,
) -> RiskLevel:
    """Accept only a validated, evidence-backed, one-level LLM escalation."""

    try:
        baseline_level_value = RiskLevel(baseline)
    except (TypeError, ValueError):
        return RiskLevel.UNKNOWN
    if baseline_level_value == RiskLevel.UNKNOWN or not isinstance(context, LLMContextAssessment):
        return baseline_level_value
    if context.reasoning_status.strip().lower() != "validated":
        return baseline_level_value
    confidence = _number(context.confidence)
    if confidence is None or confidence < 0.70:
        return baseline_level_value

    supporting_ids = tuple(context.supporting_evidence_ids)
    referenced_ids = supporting_ids + tuple(context.conflicting_evidence_ids)
    valid_ids = frozenset(snapshot.evidence_ids)
    if len(set(supporting_ids)) < 2 or any(item not in valid_ids for item in referenced_ids):
        return baseline_level_value

    recommended = context.recommended_risk_level
    if recommended == RiskLevel.UNKNOWN:
        return baseline_level_value
    baseline_index = _LEVEL_INDEX[baseline_level_value]
    recommended_index = _LEVEL_INDEX[recommended]
    if recommended_index != baseline_index + 1:
        return baseline_level_value
    return recommended


def _coerce_level(value: RiskLevel | str | None, *, allow_none: bool = False) -> RiskLevel | None:
    if value is None and allow_none:
        return None
    try:
        return value if isinstance(value, RiskLevel) else RiskLevel(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("risk level must be a valid RiskLevel") from exc


def _recovery_count(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("valid_snapshot_count must be a non-negative integer")
    return value


def transition(
    previous: RiskLevel | None,
    candidate: RiskLevel,
    valid_snapshot_count: int,
    *,
    retained_risk_level: RiskLevel | None = None,
) -> StateTransitionResult:
    """Apply upgrade and recovery hysteresis without hidden mutable state.

    ``valid_snapshot_count`` includes the current candidate. When an UNKNOWN
    observation interrupts recovery, callers can preserve the last confirmed
    ORANGE/RED state through ``retained_risk_level`` on the next invocation.
    """

    previous_level = _coerce_level(previous, allow_none=True)
    candidate_level = _coerce_level(candidate)
    retained_level = _coerce_level(retained_risk_level, allow_none=True)
    count = _recovery_count(valid_snapshot_count)

    if candidate_level == RiskLevel.UNKNOWN:
        retained = retained_level
        if previous_level in {RiskLevel.ORANGE, RiskLevel.RED}:
            retained = previous_level
        return StateTransitionResult(
            final_level=RiskLevel.UNKNOWN,
            state_transition="UNCHANGED" if previous_level == RiskLevel.UNKNOWN else "TO_UNKNOWN",
            push_required=False,
            persist_required=True,
            next_valid_snapshot_count=0,
            retained_risk_level=retained,
        )

    anchor = retained_level if previous_level == RiskLevel.UNKNOWN and retained_level is not None else previous_level
    if anchor is None or anchor == RiskLevel.UNKNOWN:
        push = candidate_level in {RiskLevel.ORANGE, RiskLevel.RED}
        return StateTransitionResult(
            final_level=candidate_level,
            state_transition=f"INITIAL_{candidate_level.value}",
            push_required=push,
            persist_required=True,
            next_valid_snapshot_count=0,
            retained_risk_level=candidate_level if candidate_level in {RiskLevel.ORANGE, RiskLevel.RED} else None,
        )

    anchor_index = _LEVEL_INDEX[anchor]
    candidate_index = _LEVEL_INDEX[candidate_level]
    if candidate_index > anchor_index:
        return StateTransitionResult(
            final_level=candidate_level,
            state_transition=f"UPGRADE_{anchor.value}_TO_{candidate_level.value}",
            push_required=candidate_level in {RiskLevel.ORANGE, RiskLevel.RED},
            persist_required=True,
            next_valid_snapshot_count=0,
            retained_risk_level=candidate_level if candidate_level in {RiskLevel.ORANGE, RiskLevel.RED} else None,
        )
    if candidate_index == anchor_index:
        return StateTransitionResult(
            final_level=anchor,
            state_transition="UNCHANGED",
            push_required=False,
            persist_required=True,
            next_valid_snapshot_count=0,
            retained_risk_level=anchor if anchor in {RiskLevel.ORANGE, RiskLevel.RED} else None,
        )

    if anchor in {RiskLevel.ORANGE, RiskLevel.RED} and count < 2:
        return StateTransitionResult(
            final_level=anchor,
            state_transition="RECOVERY_PENDING",
            push_required=False,
            persist_required=True,
            next_valid_snapshot_count=count,
            retained_risk_level=anchor,
        )

    state_transition = (
        f"RECOVERY_{anchor.value}_TO_{candidate_level.value}"
        if anchor in {RiskLevel.ORANGE, RiskLevel.RED}
        else f"DOWNGRADE_{anchor.value}_TO_{candidate_level.value}"
    )
    return StateTransitionResult(
        final_level=candidate_level,
        state_transition=state_transition,
        push_required=False,
        persist_required=True,
        next_valid_snapshot_count=0,
        retained_risk_level=candidate_level if candidate_level in {RiskLevel.ORANGE, RiskLevel.RED} else None,
    )


_ACTIONS = {
    RiskLevel.GREEN: ("OPEN", 100.0, "HOLD"),
    RiskLevel.YELLOW: ("OPEN", 100.0, "HOLD"),
    RiskLevel.ORANGE: ("CONDITIONAL", 3.0, "HOLD_OR_REDUCE"),
    RiskLevel.RED: ("WAIT", 0.0, "REDUCE"),
    RiskLevel.UNKNOWN: ("WAIT", 0.0, "HOLD"),
}


def build_final_decision(
    *,
    baseline: RiskLevel,
    candidate: RiskLevel,
    snapshot: FeatureSnapshot,
    previous: RiskLevel | None,
    valid_snapshot_count: int,
    retained_risk_level: RiskLevel | None = None,
    decision_reasons: Iterable[str] = (),
) -> FinalWarningDecision:
    """Build the persisted decision from explicit policy and state inputs."""

    baseline_value = _coerce_level(baseline)
    result = transition(
        previous,
        candidate,
        valid_snapshot_count,
        retained_risk_level=retained_risk_level,
    )
    entry_gate, cap, holding_action = _ACTIONS[result.final_level]
    reasons = tuple(str(reason) for reason in decision_reasons if str(reason).strip())
    reasons += (f"policy={POLICY_VERSION}; hard_triggers={HARD_TRIGGER_VERSION}",)
    if result.final_level == RiskLevel.UNKNOWN:
        reasons += ("UNKNOWN means risk cannot be assessed reliably; it is not RED.",)

    return FinalWarningDecision(
        baseline_level=baseline_value,
        final_level=result.final_level,
        state_transition=result.state_transition,
        entry_gate=entry_gate,
        new_position_cap_pct=cap,
        holding_action=holding_action,
        push_required=result.push_required,
        decision_reasons=reasons,
        data_status=snapshot.data_quality,
    )
