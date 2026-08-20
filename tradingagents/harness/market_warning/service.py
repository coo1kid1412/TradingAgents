"""Application service for one point-in-time market warning evaluation."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable, Mapping

from .domain import (
    DataStatus,
    Evidence,
    FeatureSnapshot,
    FinalWarningDecision,
    LLMContextAssessment,
    Market,
    MarketPhase,
    QuantRiskAssessment,
    RawMarketSnapshot,
    RiskLevel,
    RunnerResult,
)
from .policy import apply_llm_adjustment, baseline_level, build_final_decision
from .quality import DataQualityAssessment, evaluate_data_quality
from .reasoning import should_call_reasoning
from .reporting import write_report


HistoryLoader = Callable[[Market, datetime], Iterable[RawMarketSnapshot]]
QualityAssessor = Callable[[RawMarketSnapshot], DataQualityAssessment]


def _unavailable_quant(snapshot: FeatureSnapshot, reason: str) -> QuantRiskAssessment:
    phase_value = snapshot.features.get("market_phase")
    try:
        phase = MarketPhase(phase_value)
    except (TypeError, ValueError):
        phase = MarketPhase.CONTINUATION
    return QuantRiskAssessment(
        crash_1d_probability=0.0,
        crash_3d_probability=0.0,
        market_phase=phase,
        base_rate_1d=0.0,
        base_rate_3d=0.0,
        reliability_grade="UNAVAILABLE",
        model_version="unavailable",
        calibration_version="unavailable",
        top_contributors=({"unavailable_reason": reason},),
    )


def _reasoning_fallback(error_class: str) -> LLMContextAssessment:
    return LLMContextAssessment(
        market_scenario="LLM context unavailable; retain deterministic baseline.",
        causal_chain=(),
        supporting_evidence_ids=(),
        conflicting_evidence_ids=(),
        overlooked_risks=(),
        recommended_risk_level=RiskLevel.UNKNOWN,
        confidence=0.0,
        action_reason="Use the deterministic market-warning baseline.",
        reasoning_status="fallback",
        error_class=error_class,
    )


def _synthetic_raw(market: Market, as_of_time, session_slot: str) -> RawMarketSnapshot:
    return RawMarketSnapshot(
        market=market,
        as_of_time=as_of_time,
        session_slot=session_slot,
        points=(),
        data_status=DataStatus.INSUFFICIENT,
        source_times={"service:error": as_of_time},
    )


def _synthetic_feature(raw: RawMarketSnapshot, reason: str) -> FeatureSnapshot:
    return FeatureSnapshot(
        market=raw.market,
        as_of_time=raw.as_of_time,
        session_slot=raw.session_slot,
        feature_version="market-warning-v2",
        features={"market_phase": None},
        evidence=(
            Evidence(
                evidence_id=f"{raw.market.value}:market-warning-v2:unavailable:{raw.as_of_time.isoformat()}",
                group="data_quality",
                summary="Market warning inputs are unavailable.",
                value=None,
                source=reason,
                as_of_time=raw.as_of_time,
            ),
        ),
        data_quality=DataStatus.INSUFFICIENT,
        reliability_grade="UNAVAILABLE",
        source_times={"service:error": raw.as_of_time},
    )


class MarketWarningService:
    """Coordinate ports while keeping every external failure fail-closed."""

    def __init__(
        self,
        *,
        data_port,
        feature_strategies: Mapping[Market, object],
        probability_model,
        repository,
        reasoning=None,
        notifier=None,
        report_root: Path | str = Path("reports/market_warning"),
        quality_assessor: QualityAssessor = evaluate_data_quality,
        history_loader: HistoryLoader | None = None,
    ) -> None:
        self.data_port = data_port
        self.feature_strategies = {Market(key): value for key, value in feature_strategies.items()}
        self.probability_model = probability_model
        self.repository = repository
        self.reasoning = reasoning
        self.notifier = notifier
        self.report_root = Path(report_root)
        self.quality_assessor = quality_assessor
        self.history_loader = history_loader or (lambda _market, _as_of: ())

    def evaluate(self, market: Market, as_of_time, session_slot: str) -> RunnerResult:
        market_value = Market(market)
        error_class: str | None = None

        load_evaluation = getattr(self.repository, "load_evaluation", None)
        if callable(load_evaluation):
            try:
                existing = load_evaluation(market_value, as_of_time)
            except Exception:
                existing = None
                error_class = "repository_error"
            if isinstance(existing, RunnerResult):
                try:
                    previous = self.repository.load_previous_decision(market_value, as_of_time)
                except Exception:
                    previous = None
                result = existing
                try:
                    path = write_report(result, previous, self.report_root)
                    result = replace(result, report_path=str(path))
                except Exception:
                    result = replace(result, error_class="report_error")
                if self.notifier is not None:
                    try:
                        self.notifier.notify(result)
                    except Exception:
                        result = replace(result, error_class="notifier_error")
                return result

        try:
            raw = self.data_port.load_snapshot(market_value, as_of_time, session_slot)
            if not isinstance(raw, RawMarketSnapshot):
                raise TypeError("data port must return RawMarketSnapshot")
        except Exception:
            raw = _synthetic_raw(market_value, as_of_time, session_slot)
            error_class = "data_unavailable"

        try:
            quality = self.quality_assessor(raw)
            if not isinstance(quality, DataQualityAssessment):
                raise TypeError("quality assessor must return DataQualityAssessment")
        except Exception:
            quality = None
            if error_class is None:
                error_class = "quality_error"

        strategy = self.feature_strategies.get(market_value)
        if strategy is None:
            snapshot = _synthetic_feature(raw, "feature_strategy_unavailable")
            if error_class is None:
                error_class = "feature_error"
        else:
            try:
                history = tuple(self.history_loader(market_value, as_of_time))
                snapshot = strategy.build(raw, history)
                if not isinstance(snapshot, FeatureSnapshot):
                    raise TypeError("feature strategy must return FeatureSnapshot")
            except Exception:
                snapshot = _synthetic_feature(raw, "feature_build_error")
                if error_class is None:
                    error_class = "feature_error"
        if quality is None:
            snapshot = replace(
                snapshot,
                data_quality=DataStatus.INSUFFICIENT,
                reliability_grade="UNAVAILABLE",
            )
        elif (
            snapshot.data_quality != quality.status
            or snapshot.reliability_grade != quality.reliability_grade
        ):
            snapshot = replace(
                snapshot,
                data_quality=quality.status,
                reliability_grade=quality.reliability_grade,
            )

        feature_snapshot_id: int | None = None
        prediction_ids: tuple[int, ...] = ()
        reasoning_id: int | None = None
        decision_id: int | None = None
        try:
            feature_snapshot_id = self.repository.save_feature_snapshot(snapshot)
        except Exception:
            if error_class is None:
                error_class = "repository_error"

        try:
            quant = self.probability_model.predict(snapshot)
            if not isinstance(quant, QuantRiskAssessment):
                raise TypeError("probability model must return QuantRiskAssessment")
        except Exception:
            quant = _unavailable_quant(snapshot, "model_error")
            if error_class is None:
                error_class = "model_error"
        if snapshot.data_quality in {
            DataStatus.CONFLICTED,
            DataStatus.STALE,
            DataStatus.INSUFFICIENT,
        } or snapshot.reliability_grade == "UNAVAILABLE":
            quant = _unavailable_quant(snapshot, "snapshot_data_unusable")
        if quant.reliability_grade == "UNAVAILABLE" and error_class is None:
            error_class = "model_unavailable"

        if feature_snapshot_id is not None:
            try:
                prediction_ids = self.repository.save_predictions(feature_snapshot_id, quant)
            except Exception:
                if error_class is None:
                    error_class = "repository_error"

        previous_state_available = True
        try:
            previous = self.repository.load_previous_decision(market_value, as_of_time)
            if previous is not None and not isinstance(previous, FinalWarningDecision):
                raise TypeError("repository must return FinalWarningDecision or None")
        except Exception:
            previous = None
            previous_state_available = False
            if error_class is None:
                error_class = "repository_error"

        baseline = (
            baseline_level(quant, snapshot)
            if previous_state_available
            else RiskLevel.UNKNOWN
        )
        context: LLMContextAssessment | None = None
        if previous_state_available and self.reasoning is not None and should_call_reasoning(
            session_slot,
            baseline,
            previous.final_level if previous is not None else None,
        ):
            try:
                context = self.reasoning.assess(snapshot, quant, previous)
                if not isinstance(context, LLMContextAssessment):
                    raise TypeError("reasoning port must return LLMContextAssessment")
            except Exception:
                context = _reasoning_fallback("invoke_error")
                if error_class is None:
                    error_class = "reasoning_error"
            if context.reasoning_status != "validated" and error_class is None:
                error_class = "reasoning_fallback"
            if feature_snapshot_id is not None:
                try:
                    reasoning_id = self.repository.save_reasoning(
                        feature_snapshot_id,
                        context,
                        getattr(self.reasoning, "model_name", "deepseek-v4-pro"),
                    )
                except Exception:
                    context = _reasoning_fallback("persistence_error")
                    if error_class is None:
                        error_class = "repository_error"

            if feature_snapshot_id is None and context.reasoning_status == "validated":
                context = _reasoning_fallback("persistence_error")

        policy_context = context if reasoning_id is not None else None
        candidate = apply_llm_adjustment(baseline, policy_context, snapshot)
        valid_snapshot_count = (
            0
            if candidate == RiskLevel.UNKNOWN
            else (previous.valid_snapshot_count if previous is not None else 0) + 1
        )
        reasons = [
            "Calibrated probabilities are estimates, not certainties.",
            f"quant_baseline={baseline.value}",
        ]
        if not previous_state_available:
            reasons.append("Previous warning state is unavailable; recovery state cannot be assessed.")
        if context is not None and context.reasoning_status == "validated":
            reasons.append("DeepSeek supplied a validated evidence-bounded context assessment.")
        decision = build_final_decision(
            baseline=baseline,
            candidate=candidate,
            snapshot=snapshot,
            previous=previous.final_level if previous is not None else None,
            valid_snapshot_count=valid_snapshot_count,
            retained_risk_level=previous.retained_risk_level if previous is not None else None,
            decision_reasons=reasons,
        )

        if feature_snapshot_id is not None and len(prediction_ids) == 2:
            try:
                decision_id = self.repository.save_decision(
                    feature_snapshot_id,
                    prediction_ids,
                    reasoning_id,
                    decision,
                )
            except Exception:
                if error_class is None:
                    error_class = "repository_error"

        result = RunnerResult(
            market=market_value,
            as_of_time=as_of_time,
            session_slot=session_slot,
            feature_snapshot=snapshot,
            quant_assessment=quant,
            context_assessment=context,
            decision=decision,
            decision_id=decision_id,
            error_class=error_class,
        )
        try:
            path = write_report(result, previous, self.report_root)
            result = replace(result, report_path=str(path))
        except Exception:
            if result.error_class is None:
                result = replace(result, error_class="report_error")

        should_notify = "premarket" in session_slot.lower() or decision.push_required
        if self.notifier is not None and should_notify and decision_id is not None:
            try:
                self.notifier.notify(result)
            except Exception:
                if result.error_class is None:
                    result = replace(result, error_class="notifier_error")
                elif result.error_class not in {"data_unavailable", "model_error", "model_unavailable"}:
                    result = replace(result, error_class="notifier_error")
        return result
