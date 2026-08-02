"""Small protocols that keep infrastructure outside the warning domain."""

from __future__ import annotations

from datetime import date, datetime
from typing import Protocol

from .domain import (
    Evidence,
    FeatureSnapshot,
    FinalWarningDecision,
    LLMContextAssessment,
    Market,
    QuantRiskAssessment,
    RawMarketSnapshot,
    RuleRiskAssessment,
    RunnerResult,
)


class MarketDataPort(Protocol):
    def load_snapshot(
        self, market: Market, as_of_time: datetime, session_slot: str
    ) -> RawMarketSnapshot: ...


class MarketContextPort(Protocol):
    def load_context(self, market: Market, as_of_time: datetime) -> tuple[Evidence, ...]: ...


class ProbabilityModelPort(Protocol):
    def predict(self, snapshot: FeatureSnapshot) -> QuantRiskAssessment: ...


class ReasoningPort(Protocol):
    def assess(
        self,
        snapshot: FeatureSnapshot,
        quant: QuantRiskAssessment,
        previous: FinalWarningDecision | None,
    ) -> LLMContextAssessment: ...


class WarningRepository(Protocol):
    def save_feature_snapshot(self, snapshot: FeatureSnapshot) -> int: ...

    def save_predictions(
        self, feature_snapshot_id: int, assessment: QuantRiskAssessment
    ) -> tuple[int, int]: ...

    def save_rule_assessment(
        self, feature_snapshot_id: int, assessment: RuleRiskAssessment
    ) -> int: ...

    def load_previous_rule_assessment(
        self, market: Market, before_time: datetime
    ) -> RuleRiskAssessment | None: ...

    def save_reasoning(
        self, feature_snapshot_id: int, assessment: LLMContextAssessment, model_name: str
    ) -> int: ...

    def save_reasoning_for_decision(
        self,
        decision_id: int,
        assessment: LLMContextAssessment,
        model_name: str,
    ) -> int: ...

    def attach_reasoning(self, decision_id: int, reasoning_id: int) -> None: ...

    def load_latest_reasoning(
        self, market: Market, before_time: datetime, model_name: str = "MiniMax-M3"
    ) -> LLMContextAssessment | None: ...

    def save_decision(
        self,
        feature_snapshot_id: int,
        prediction_ids: tuple[int, ...],
        reasoning_id: int | None,
        decision: FinalWarningDecision,
        *,
        rule_assessment_id: int | None = None,
        shadow_prediction_ids: tuple[int, ...] = (),
    ) -> int: ...

    def register_rule_engine(self, record: dict[str, object]) -> None: ...

    def activate_rule_engine(self, engine_version: str, mode: str) -> dict[str, object]: ...

    def load_active_rule_engine(self, market: Market, mode: str) -> dict[str, object] | None: ...

    def load_latest_decision(
        self, market: Market, as_of_time: datetime | None = None
    ) -> FinalWarningDecision | None: ...

    def load_previous_decision(
        self, market: Market, before_time: datetime
    ) -> FinalWarningDecision | None: ...


class WarningNotifier(Protocol):
    def notify(self, result: RunnerResult) -> None: ...


class ClockPort(Protocol):
    def now(self) -> datetime: ...

    def is_trading_day(self, market: Market, on_date: date) -> bool: ...
