"""Fast-path application service for deterministic A-share warnings."""

from __future__ import annotations

import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable

from .domain import (
    DataStatus,
    Evidence,
    FeatureSnapshot,
    FinalWarningDecision,
    LLMContextAssessment,
    Market,
    QuantRiskAssessment,
    RawMarketSnapshot,
    RiskLevel,
    RuleRiskAssessment,
    RunnerResult,
    SessionRiskSummary,
)
from .policy import build_rule_decision
from .quality import DataQualityAssessment, evaluate_data_quality
from .reporting import write_report


HistoryLoader = Callable[[Market, datetime], Iterable[RawMarketSnapshot]]
QualityAssessor = Callable[[RawMarketSnapshot], DataQualityAssessment]


def _synthetic_raw(market: Market, as_of_time: datetime, session_slot: str) -> RawMarketSnapshot:
    return RawMarketSnapshot(
        market=market,
        as_of_time=as_of_time,
        session_slot=session_slot,
        points=(),
        data_status=DataStatus.INSUFFICIENT,
        source_times={"rule_service:error": as_of_time},
    )


def _synthetic_feature(raw: RawMarketSnapshot, reason: str) -> FeatureSnapshot:
    return FeatureSnapshot(
        market=raw.market,
        as_of_time=raw.as_of_time,
        session_slot=raw.session_slot,
        feature_version="market-warning-v2",
        features={"market_phase": "CONTINUATION"},
        evidence=(
            Evidence(
                evidence_id=f"rule-service:{reason}:{raw.as_of_time.isoformat()}",
                group="data_quality",
                summary="Rule warning inputs are unavailable.",
                source=reason,
                as_of_time=raw.as_of_time,
            ),
        ),
        data_quality=DataStatus.INSUFFICIENT,
        reliability_grade="UNAVAILABLE",
        source_times={"rule_service:error": raw.as_of_time},
    )


def _unknown_assessment(
    snapshot: FeatureSnapshot,
    *,
    engine_version: str,
    manifest_sha256: str,
) -> RuleRiskAssessment:
    return RuleRiskAssessment(
        market=snapshot.market,
        as_of_time=snapshot.as_of_time,
        engine_version=engine_version,
        manifest_sha256=manifest_sha256,
        risk_level=RiskLevel.UNKNOWN,
        risk_score=0.0,
        market_phase=snapshot.features.get("market_phase", "CONTINUATION"),
        triggered_rules=(),
        missing_optional_groups=(),
        reliability_grade="UNAVAILABLE",
        evaluation_latency_ms=0.0,
    )


class RuleMarketWarningService:
    """Persist and notify the deterministic result before optional slow work."""

    def __init__(
        self,
        *,
        data_port,
        feature_strategy,
        rule_evaluator,
        repository,
        notifier=None,
        shadow_model=None,
        post_alert_reasoning=None,
        report_root: Path | str = Path("reports/market_warning"),
        reporter=write_report,
        quality_assessor: QualityAssessor = evaluate_data_quality,
        history_loader: HistoryLoader | None = None,
        monotonic_ns: Callable[[], int] = time.monotonic_ns,
        engine_version: str = "rule-v1.0.0",
        manifest_sha256: str = "0" * 64,
    ) -> None:
        self.data_port = data_port
        self.feature_strategy = feature_strategy
        self.rule_evaluator = rule_evaluator
        self.repository = repository
        self.notifier = notifier
        self.shadow_model = shadow_model
        self.post_alert_reasoning = post_alert_reasoning
        self.report_root = Path(report_root)
        self.reporter = reporter
        self.quality_assessor = quality_assessor
        self.history_loader = history_loader or (lambda _market, _time: ())
        self.monotonic_ns = monotonic_ns
        self.engine_version = engine_version
        self.manifest_sha256 = manifest_sha256

    def evaluate(self, market: Market, as_of_time: datetime, session_slot: str) -> RunnerResult:
        """Run both phases for direct callers that do not own a scan coordinator."""

        return self.complete_after_alert(
            self.evaluate_fast(market, as_of_time, session_slot)
        )

    def evaluate_fast(
        self,
        market: Market,
        as_of_time: datetime,
        session_slot: str,
    ) -> RunnerResult:
        """Complete deterministic persistence and delivery without optional LLM work."""

        market_value = Market(market)
        if market_value != Market.A_SHARE:
            raise ValueError("rule V1 service only supports A shares")
        error_class: str | None = None
        existing_loader = getattr(self.repository, "load_evaluation", None)
        if callable(existing_loader):
            try:
                existing = existing_loader(market_value, as_of_time)
            except Exception:
                existing = None
                error_class = "repository_error"
            if isinstance(existing, RunnerResult) and existing.rule_assessment is not None:
                return self._resume_existing(existing)

        data_available = True
        try:
            raw = self.data_port.load_snapshot(market_value, as_of_time, session_slot)
            if not isinstance(raw, RawMarketSnapshot):
                raise TypeError("data port must return RawMarketSnapshot")
        except Exception:
            raw = _synthetic_raw(market_value, as_of_time, session_slot)
            data_available = False
            error_class = "data_unavailable"

        try:
            quality = self.quality_assessor(raw)
            if not isinstance(quality, DataQualityAssessment):
                raise TypeError("quality assessor must return DataQualityAssessment")
        except Exception:
            quality = None
            if error_class is None:
                error_class = "quality_error"

        try:
            history = tuple(self.history_loader(market_value, as_of_time))
            snapshot = self.feature_strategy.build(raw, history)
            if not isinstance(snapshot, FeatureSnapshot):
                raise TypeError("feature strategy must return FeatureSnapshot")
        except Exception:
            snapshot = _synthetic_feature(raw, "feature_build_error")
            if error_class is None:
                error_class = "feature_error"
        if not data_available or quality is None:
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

        partial_loader = getattr(self.repository, "load_feature_snapshot", None)
        if callable(partial_loader):
            try:
                persisted_snapshot = partial_loader(
                    market_value,
                    as_of_time,
                    snapshot.feature_version,
                )
                if isinstance(persisted_snapshot, FeatureSnapshot):
                    snapshot = persisted_snapshot
                    if error_class in {
                        "data_unavailable",
                        "quality_error",
                        "feature_error",
                    }:
                        error_class = None
            except Exception:
                if error_class is None:
                    error_class = "repository_error"

        snapshot_id = None
        rule_assessment_id = None
        decision_id = None
        try:
            snapshot_id = self.repository.save_feature_snapshot(snapshot)
        except Exception:
            error_class = "repository_error"

        previous_rule = None
        if snapshot_id is not None:
            try:
                loader = getattr(self.repository, "load_previous_rule_assessment", None)
                if callable(loader):
                    previous_rule = loader(market_value, as_of_time)
                    if previous_rule is not None and not isinstance(previous_rule, RuleRiskAssessment):
                        raise TypeError("invalid previous rule assessment")
            except Exception:
                previous_rule = None
                error_class = "repository_error"

        start = self.monotonic_ns()
        if snapshot.reliability_grade == "UNAVAILABLE" or snapshot_id is None:
            assessment = _unknown_assessment(
                snapshot,
                engine_version=self.engine_version,
                manifest_sha256=self.manifest_sha256,
            )
        else:
            try:
                assessment = self.rule_evaluator(
                    snapshot,
                    previous_assessment=previous_rule,
                )
                if not isinstance(assessment, RuleRiskAssessment):
                    raise TypeError("rule evaluator must return RuleRiskAssessment")
            except Exception:
                assessment = _unknown_assessment(
                    snapshot,
                    engine_version=self.engine_version,
                    manifest_sha256=self.manifest_sha256,
                )
                if error_class is None:
                    error_class = "rule_error"
        latency_ms = max(0.0, (self.monotonic_ns() - start) / 1_000_000.0)
        assessment = replace(assessment, evaluation_latency_ms=latency_ms)

        if snapshot_id is not None:
            try:
                rule_assessment_id = self.repository.save_rule_assessment(snapshot_id, assessment)
            except Exception:
                assessment = replace(
                    assessment,
                    risk_level=RiskLevel.UNKNOWN,
                    risk_score=0.0,
                    reliability_grade="UNAVAILABLE",
                    triggered_rules=(),
                )
                error_class = "repository_error"

        previous = None
        previous_state_available = True
        try:
            previous = self.repository.load_previous_decision(market_value, as_of_time)
            if previous is not None and not isinstance(previous, FinalWarningDecision):
                raise TypeError("invalid previous decision")
        except Exception:
            previous_state_available = False
            error_class = "repository_error"
        if not previous_state_available:
            assessment = replace(
                assessment,
                risk_level=RiskLevel.UNKNOWN,
                risk_score=0.0,
                reliability_grade="UNAVAILABLE",
                triggered_rules=(),
            )
        decision = build_rule_decision(assessment, snapshot, previous=previous)

        if snapshot_id is not None and rule_assessment_id is not None:
            try:
                decision_id = self.repository.save_decision(
                    snapshot_id,
                    (),
                    None,
                    decision,
                    rule_assessment_id=rule_assessment_id,
                    shadow_prediction_ids=(),
                )
            except Exception:
                error_class = "repository_error"
                assessment = replace(
                    assessment,
                    risk_level=RiskLevel.UNKNOWN,
                    risk_score=0.0,
                    reliability_grade="UNAVAILABLE",
                    triggered_rules=(),
                )
                decision = build_rule_decision(assessment, snapshot, previous=previous)

        prior_context = None
        previous_session_summary = None
        if "premarket" in session_slot.lower():
            try:
                loader = getattr(self.repository, "load_latest_reasoning", None)
                if callable(loader):
                    prior_context = loader(market_value, as_of_time)
                    if prior_context is not None and not isinstance(
                        prior_context, LLMContextAssessment
                    ):
                        prior_context = None
            except Exception:
                prior_context = None
            try:
                summary_loader = getattr(
                    self.repository, "load_previous_intraday_summary", None
                )
                if callable(summary_loader):
                    previous_session_summary = summary_loader(
                        market_value, as_of_time
                    )
                    if previous_session_summary is not None and not isinstance(
                        previous_session_summary, SessionRiskSummary
                    ):
                        previous_session_summary = None
            except Exception:
                previous_session_summary = None

        result = RunnerResult(
            market=market_value,
            as_of_time=as_of_time,
            session_slot=session_slot,
            feature_snapshot=snapshot,
            rule_assessment=assessment,
            context_assessment=prior_context,
            previous_decision=previous,
            previous_session_summary=previous_session_summary,
            decision=decision,
            snapshot_id=snapshot_id,
            decision_id=decision_id,
            error_class=error_class,
        )
        try:
            path = self.reporter(result, previous, self.report_root)
            result = replace(result, report_path=str(path))
        except Exception:
            result = replace(result, error_class="report_error")

        should_notify = "premarket" in session_slot.lower() or (
            decision.push_required and decision.final_level in {RiskLevel.ORANGE, RiskLevel.RED}
        )
        notification_confirmed = False
        if self.notifier is not None and should_notify and decision_id is not None:
            try:
                notification_confirmed = bool(self.notifier.notify(result))
                if not notification_confirmed:
                    was_sent = getattr(self.notifier, "was_sent", None)
                    notification_confirmed = callable(was_sent) and bool(was_sent(result))
            except Exception:
                result = replace(result, error_class="notifier_error")
        return replace(
            result,
            previous_decision=previous,
            notification_confirmed=notification_confirmed,
        )

    def _resume_existing(self, result: RunnerResult) -> RunnerResult:
        try:
            previous = self.repository.load_previous_decision(result.market, result.as_of_time)
        except Exception:
            previous = None
        previous_session_summary = result.previous_session_summary
        if "premarket" in result.session_slot.lower():
            try:
                loader = getattr(
                    self.repository, "load_previous_intraday_summary", None
                )
                if callable(loader):
                    loaded_summary = loader(result.market, result.as_of_time)
                    if isinstance(loaded_summary, SessionRiskSummary):
                        previous_session_summary = loaded_summary
            except Exception:
                pass
        result = replace(
            result,
            previous_decision=previous,
            previous_session_summary=previous_session_summary,
        )
        try:
            path = self.reporter(result, previous, self.report_root)
            result = replace(result, report_path=str(path))
        except Exception:
            return replace(result, error_class="report_error")
        should_notify = "premarket" in result.session_slot.lower() or (
            result.decision is not None and result.decision.push_required
        )
        notification_confirmed = False
        if self.notifier is not None and should_notify:
            try:
                notification_confirmed = bool(self.notifier.notify(result))
                if not notification_confirmed:
                    was_sent = getattr(self.notifier, "was_sent", None)
                    notification_confirmed = callable(was_sent) and bool(was_sent(result))
            except Exception:
                result = replace(result, error_class="notifier_error")
        return replace(
            result,
            previous_decision=previous,
            notification_confirmed=notification_confirmed,
        )

    def complete_after_alert(self, result: RunnerResult) -> RunnerResult:
        """Run optional shadow and M3 work after the fast-scan lease is released."""

        if (
            not isinstance(result, RunnerResult)
            or not result.notification_confirmed
            or result.decision is None
            or not result.decision.push_required
            or result.context_assessment is not None
        ):
            return result
        return self._run_slow_paths(
            result,
            result.previous_decision,
            result.snapshot_id,
        )

    def _run_slow_paths(
        self,
        result: RunnerResult,
        previous: FinalWarningDecision | None,
        snapshot_id: int | None,
    ) -> RunnerResult:
        if self.shadow_model is not None and result.feature_snapshot is not None:
            try:
                shadow = self.shadow_model.predict(result.feature_snapshot)
                if isinstance(shadow, QuantRiskAssessment):
                    result = replace(result, shadow_quant_assessment=shadow)
            except Exception:
                pass
        if self.post_alert_reasoning is not None:
            try:
                context = self.post_alert_reasoning.assess_rule_alert(result, previous)
                if isinstance(context, LLMContextAssessment):
                    if result.decision_id is not None and snapshot_id is None:
                        saver = getattr(
                            self.repository,
                            "save_reasoning_for_decision",
                            None,
                        )
                        if callable(saver):
                            saver(
                                result.decision_id,
                                context,
                                getattr(
                                    self.post_alert_reasoning,
                                    "model_name",
                                    "MiniMax-M3",
                                ),
                            )
                    elif snapshot_id is not None and result.decision_id is not None:
                        reasoning_id = self.repository.save_reasoning(
                            snapshot_id,
                            context,
                            getattr(self.post_alert_reasoning, "model_name", "MiniMax-M3"),
                        )
                        self.repository.attach_reasoning(result.decision_id, reasoning_id)
                    result = replace(result, context_assessment=context)
            except Exception:
                pass
        return result
