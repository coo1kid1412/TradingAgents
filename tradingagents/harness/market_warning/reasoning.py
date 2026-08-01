"""Strict, infrastructure-free contracts for market-warning reasoning."""

from __future__ import annotations

import json
import threading
from datetime import datetime, timedelta, timezone
from enum import Enum
from math import isfinite
from numbers import Real
from typing import Any, Callable, Iterable, Mapping

from .domain import (
    DataStatus,
    FeatureSnapshot,
    FinalWarningDecision,
    LLMContextAssessment,
    QuantRiskAssessment,
    RiskLevel,
)
from .policy import baseline_level


_REQUIRED_FIELDS = frozenset(
    {
        "market_scenario",
        "causal_chain",
        "supporting_evidence_ids",
        "conflicting_evidence_ids",
        "overlooked_risks",
        "recommended_risk_level",
        "confidence",
        "action_reason",
        "reasoning_status",
    }
)
_OUTPUT_LEVELS = frozenset(
    {RiskLevel.GREEN, RiskLevel.YELLOW, RiskLevel.ORANGE, RiskLevel.RED}
)


class ReasoningValidationError(ValueError):
    """Coarse validation failure that never contains provider output."""

    def __init__(self, error_class: str = "validation_error") -> None:
        self.error_class = error_class
        super().__init__(error_class)


def _string(value: Any, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise ReasoningValidationError()
    normalized = value.strip()
    try:
        normalized.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ReasoningValidationError() from exc
    if not allow_empty and not normalized:
        raise ReasoningValidationError()
    return normalized


def _string_list(value: Any, *, allow_empty: bool) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise ReasoningValidationError()
    result = tuple(_string(item) for item in value)
    if not allow_empty and not result:
        raise ReasoningValidationError()
    if len(result) != len(set(result)):
        raise ReasoningValidationError()
    return result


def validate_context_assessment(
    payload: Mapping[str, Any],
    valid_evidence_ids: Iterable[str],
    *,
    baseline: RiskLevel | None = None,
    data_status: DataStatus | None = None,
) -> LLMContextAssessment:
    """Validate an exact JSON object without coercing malformed LLM fields."""

    if not isinstance(payload, Mapping) or set(payload) != _REQUIRED_FIELDS:
        raise ReasoningValidationError()
    valid_ids = frozenset(valid_evidence_ids)
    if any(not isinstance(item, str) or not item for item in valid_ids):
        raise ReasoningValidationError()

    supporting = _string_list(payload["supporting_evidence_ids"], allow_empty=False)
    conflicting = _string_list(payload["conflicting_evidence_ids"], allow_empty=False)
    if len(supporting) < 2 or any(item not in valid_ids for item in supporting + conflicting):
        raise ReasoningValidationError()

    confidence = payload["confidence"]
    if isinstance(confidence, bool) or not isinstance(confidence, Real):
        raise ReasoningValidationError()
    confidence_value = float(confidence)
    if not isfinite(confidence_value) or not 0.0 <= confidence_value <= 1.0:
        raise ReasoningValidationError()

    try:
        recommended = RiskLevel(payload["recommended_risk_level"])
    except (TypeError, ValueError) as exc:
        raise ReasoningValidationError() from exc
    if recommended not in _OUTPUT_LEVELS:
        raise ReasoningValidationError()
    if baseline is not None or data_status is not None:
        try:
            baseline_value = RiskLevel(baseline)
            status_value = DataStatus(data_status)
        except (TypeError, ValueError) as exc:
            raise ReasoningValidationError("semantic_validation") from exc
        if baseline_value == RiskLevel.UNKNOWN or status_value in {
            DataStatus.CONFLICTED,
            DataStatus.STALE,
            DataStatus.INSUFFICIENT,
        }:
            raise ReasoningValidationError("semantic_validation")
        ordered = (RiskLevel.GREEN, RiskLevel.YELLOW, RiskLevel.ORANGE, RiskLevel.RED)
        baseline_index = ordered.index(baseline_value)
        recommended_index = ordered.index(recommended)
        if recommended_index not in {baseline_index, min(baseline_index + 1, 3)}:
            raise ReasoningValidationError("semantic_validation")
    if payload["reasoning_status"] != "validated":
        raise ReasoningValidationError()

    return LLMContextAssessment(
        market_scenario=_string(payload["market_scenario"]),
        causal_chain=_string_list(payload["causal_chain"], allow_empty=False),
        supporting_evidence_ids=supporting,
        conflicting_evidence_ids=conflicting,
        overlooked_risks=_string_list(payload["overlooked_risks"], allow_empty=True),
        recommended_risk_level=recommended,
        confidence=confidence_value,
        action_reason=_string(payload["action_reason"]),
        reasoning_status="validated",
        error_class=None,
    )


class CircuitBreaker:
    """Thread-safe consecutive-failure breaker with an injectable aware clock."""

    def __init__(
        self,
        failure_threshold: int = 3,
        cooldown: timedelta = timedelta(minutes=30),
        *,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        if isinstance(failure_threshold, bool) or failure_threshold < 1:
            raise ValueError("failure_threshold must be a positive integer")
        if not isinstance(cooldown, timedelta) or cooldown <= timedelta(0):
            raise ValueError("cooldown must be positive")
        self._failure_threshold = failure_threshold
        self._cooldown = cooldown
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._consecutive_failures = 0
        self._open_until: datetime | None = None
        self._lock = threading.Lock()

    def _now(self) -> datetime:
        value = self._clock()
        if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("circuit-breaker clock must be timezone-aware")
        return value

    @property
    def consecutive_failures(self) -> int:
        with self._lock:
            return self._consecutive_failures

    def allow_call(self) -> bool:
        now = self._now()
        with self._lock:
            return self._open_until is None or now >= self._open_until

    def record_success(self) -> None:
        with self._lock:
            self._consecutive_failures = 0
            self._open_until = None

    def record_failure(self) -> None:
        now = self._now()
        with self._lock:
            if self._open_until is not None and now >= self._open_until:
                self._consecutive_failures = self._failure_threshold - 1
            self._consecutive_failures += 1
            if self._consecutive_failures >= self._failure_threshold:
                self._open_until = now + self._cooldown


def should_call_reasoning(
    session_slot: str, candidate: RiskLevel, previous: RiskLevel | None
) -> bool:
    """Return whether the deterministic call policy permits an M3 invocation."""

    normalized_slot = str(session_slot).strip().lower()
    try:
        candidate_level = RiskLevel(candidate)
        previous_level = None if previous is None else RiskLevel(previous)
    except (TypeError, ValueError):
        return False
    if "premarket" in normalized_slot:
        return True
    if candidate_level in {RiskLevel.ORANGE, RiskLevel.RED}:
        return True
    return previous_level == RiskLevel.ORANGE and candidate_level == RiskLevel.RED


def _json_value(value: Any) -> Any:
    if isinstance(value, str):
        return value.encode("utf-8", errors="replace").decode("utf-8")
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Real):
        numeric = float(value)
        return numeric if isfinite(numeric) else None
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    return None


def build_reasoning_prompt(
    snapshot: FeatureSnapshot,
    quant: QuantRiskAssessment,
    previous: FinalWarningDecision | None,
) -> str:
    """Build compact point-in-time context and an exact output schema."""

    baseline = baseline_level(quant, snapshot)
    evidence = [
        {
            "evidence_id": item.evidence_id,
            "summary": _json_value(item.summary[:240]),
            "value": _json_value(item.value),
            "as_of_time": item.as_of_time.isoformat() if item.as_of_time else None,
        }
        for item in snapshot.evidence[:16]
    ]
    context = {
        "task": "Assess whether evidence supports raising the code baseline by at most one level.",
        "strict_json_output": True,
        "market": snapshot.market.value,
        "as_of_time": snapshot.as_of_time.isoformat(),
        "session_slot": snapshot.session_slot,
        "data_quality": snapshot.data_quality.value,
        "reliability_grade": snapshot.reliability_grade,
        "current_baseline": baseline.value,
        "previous_level": previous.final_level.value if previous else None,
        "probabilities": {
            "crash_1d": quant.crash_1d_probability,
            "base_rate_1d": quant.base_rate_1d,
            "crash_3d": quant.crash_3d_probability,
            "base_rate_3d": quant.base_rate_3d,
            "market_phase": quant.market_phase.value,
        },
        "top_contributors": _json_value(quant.top_contributors[:5]),
        "evidence": evidence,
        "constraints": [
            "Use only listed evidence_id values.",
            "Do not lower the code baseline or raise more than one level.",
            "Return JSON only; do not include analysis, markdown, or private reasoning.",
        ],
        "output_schema": {
            "market_scenario": "non-empty string",
            "causal_chain": ["one or more non-empty strings"],
            "supporting_evidence_ids": ["at least two listed evidence_id values"],
            "conflicting_evidence_ids": ["at least one listed evidence_id value"],
            "overlooked_risks": ["zero or more non-empty strings"],
            "recommended_risk_level": "GREEN|YELLOW|ORANGE|RED",
            "confidence": "number from 0 to 1",
            "action_reason": "non-empty string",
            "reasoning_status": "validated",
        },
    }
    return json.dumps(context, ensure_ascii=False, sort_keys=True, separators=(",", ": "))
