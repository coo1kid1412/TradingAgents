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

    def save_prediction(self, feature_snapshot_id: int, assessment: QuantRiskAssessment) -> int: ...

    def save_reasoning(
        self, feature_snapshot_id: int, assessment: LLMContextAssessment, model_name: str
    ) -> int: ...

    def save_decision(
        self,
        feature_snapshot_id: int,
        prediction_ids: tuple[int, ...],
        reasoning_id: int | None,
        decision: FinalWarningDecision,
    ) -> int: ...

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
