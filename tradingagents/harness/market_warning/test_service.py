from __future__ import annotations

import tempfile
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from unittest import TestCase, main

from tradingagents.harness.market_warning.domain import (
    DataStatus,
    Evidence,
    FeatureSnapshot,
    FinalWarningDecision,
    LLMContextAssessment,
    Market,
    MarketDataPoint,
    MarketPhase,
    QuantRiskAssessment,
    RawMarketSnapshot,
    RiskLevel,
)
from tradingagents.harness.market_warning.quality import DataQualityAssessment
from tradingagents.harness.market_warning.service import MarketWarningService


NOW = datetime(2026, 8, 3, 0, 30, tzinfo=timezone.utc)


class DataPort:
    def __init__(self, events: list[str], error: Exception | None = None) -> None:
        self.events = events
        self.error = error

    def load_snapshot(self, market, as_of_time, session_slot):
        self.events.append("load_data")
        if self.error:
            raise self.error
        point = MarketDataPoint(
            market=market,
            symbol="000001.SH",
            field="index_price",
            value=3000.0,
            data_time=as_of_time,
            fetched_at=as_of_time,
            source="fixture",
        )
        return RawMarketSnapshot(
            market=market,
            as_of_time=as_of_time,
            session_slot=session_slot,
            points=(point,),
            source_times={"fixture": as_of_time},
        )


class FeatureStrategy:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    def build(self, raw, prior_history):
        self.events.append("build_features")
        evidence = tuple(
            Evidence(
                evidence_id=f"ev-{index}",
                group="feature",
                summary=f"signal {index}",
                value=index,
                source="fixture",
                as_of_time=raw.as_of_time,
            )
            for index in range(1, 4)
        )
        return FeatureSnapshot(
            market=raw.market,
            as_of_time=raw.as_of_time,
            session_slot=raw.session_slot,
            feature_version="market-warning-v2",
            features={"market_phase": "FIRST_SHOCK", "pressure_transition_signal": True},
            evidence=evidence,
            data_quality=DataStatus.FRESH,
            reliability_grade="A",
            source_times={"fixture:last": raw.as_of_time},
        )


class Model:
    def __init__(self, events: list[str], error: Exception | None = None, unavailable: bool = False) -> None:
        self.events = events
        self.error = error
        self.unavailable = unavailable

    def predict(self, snapshot):
        self.events.append("predict")
        if self.error:
            raise self.error
        reliability = "UNAVAILABLE" if self.unavailable else "A"
        return QuantRiskAssessment(
            crash_1d_probability=0.0 if self.unavailable else 0.05,
            crash_3d_probability=0.0 if self.unavailable else 0.10,
            market_phase=MarketPhase.FIRST_SHOCK,
            base_rate_1d=0.0 if self.unavailable else 0.01,
            base_rate_3d=0.0 if self.unavailable else 0.02,
            reliability_grade=reliability,
            model_version="unavailable" if self.unavailable else "model-v2",
            calibration_version="unavailable" if self.unavailable else "platt-v2",
            top_contributors=(),
        )


class Reasoner:
    model_name = "MiniMax-M3"

    def __init__(self, events: list[str], error: Exception | None = None) -> None:
        self.events = events
        self.error = error

    def assess(self, snapshot, quant, previous):
        self.events.append("reason")
        if self.error:
            raise self.error
        return LLMContextAssessment(
            market_scenario="transition evidence is aligned",
            causal_chain=("pressure", "fragility"),
            supporting_evidence_ids=("ev-1", "ev-2"),
            conflicting_evidence_ids=("ev-3",),
            overlooked_risks=(),
            recommended_risk_level=RiskLevel.RED,
            confidence=0.80,
            action_reason="raise one level",
            reasoning_status="validated",
        )


class Repository:
    def __init__(self, events: list[str], fail_stage: str | None = None) -> None:
        self.events = events
        self.fail_stage = fail_stage
        self.previous = FinalWarningDecision(
            baseline_level=RiskLevel.YELLOW,
            final_level=RiskLevel.YELLOW,
            state_transition="UNCHANGED",
            entry_gate="OPEN",
            new_position_cap_pct=100.0,
            holding_action="HOLD",
            push_required=False,
            decision_reasons=(),
            data_status=DataStatus.FRESH,
        )
        self.saved_decision = None

    def _stage(self, name: str, value):
        self.events.append(name)
        if self.fail_stage == name:
            raise RuntimeError(f"private repository detail at {name}")
        return value

    def save_feature_snapshot(self, snapshot):
        return self._stage("save_snapshot", 11)

    def save_predictions(self, feature_snapshot_id, assessment):
        return self._stage("save_predictions", (21, 22))

    def save_reasoning(self, feature_snapshot_id, assessment, model_name):
        return self._stage("save_reasoning", 31)

    def save_decision(self, feature_snapshot_id, prediction_ids, reasoning_id, decision):
        self.saved_decision = decision
        return self._stage("save_decision", 41)

    def load_previous_decision(self, market, before_time):
        self.events.append("load_previous")
        return self.previous


class Notifier:
    def __init__(self, events: list[str], error: Exception | None = None) -> None:
        self.events = events
        self.error = error

    def notify(self, result):
        self.events.append("notify")
        if self.error:
            raise self.error


def quality_assessor(events: list[str], status: DataStatus = DataStatus.FRESH):
    def assess(raw):
        events.append("assess_quality")
        return DataQualityAssessment(
            market=raw.market,
            as_of_time=raw.as_of_time,
            session_slot=raw.session_slot,
            status=status,
            reliability_grade="A" if status == DataStatus.FRESH else "UNAVAILABLE",
            selected_points=raw.points,
            core_fields=("index_price",),
            optional_fields=(),
            covered_core_fields=frozenset({"index_price"}) if raw.points else frozenset(),
            covered_optional_fields=frozenset(),
            core_coverage=1.0 if raw.points else 0.0,
            optional_coverage=1.0,
            source_count=1 if raw.points else 0,
            core_source_count=1 if raw.points else 0,
        )
    return assess


class MarketWarningServiceTests(TestCase):
    def _service(self, events, directory, **changes):
        repository = changes.pop("repository", Repository(events))
        values = {
            "data_port": DataPort(events),
            "feature_strategies": {Market.A_SHARE: FeatureStrategy(events)},
            "probability_model": Model(events),
            "repository": repository,
            "reasoning": Reasoner(events),
            "notifier": Notifier(events),
            "report_root": Path(directory),
            "quality_assessor": quality_assessor(events),
            "history_loader": lambda _market, _as_of: (),
        }
        values.update(changes)
        return MarketWarningService(**values), repository

    def test_orchestration_order_and_one_level_llm_escalation(self) -> None:
        events: list[str] = []
        with tempfile.TemporaryDirectory() as directory:
            service, repository = self._service(events, directory)

            result = service.evaluate(Market.A_SHARE, NOW, "premarket")
            self.assertEqual(
                events,
                [
                    "load_data", "assess_quality", "build_features", "save_snapshot",
                    "predict", "save_predictions", "load_previous", "reason",
                    "save_reasoning", "save_decision", "notify",
                ],
            )
            self.assertEqual(result.decision.baseline_level, RiskLevel.ORANGE)
            self.assertEqual(result.decision.final_level, RiskLevel.RED)
            self.assertEqual(result.decision_id, 41)
            self.assertTrue(Path(result.report_path).is_file())
            self.assertEqual(repository.saved_decision, result.decision)

    def test_data_adapter_exception_fails_closed_to_unknown_and_still_writes_report(self) -> None:
        events: list[str] = []
        with tempfile.TemporaryDirectory() as directory:
            service, _ = self._service(
                events,
                directory,
                data_port=DataPort(events, RuntimeError("secret upstream detail")),
                quality_assessor=quality_assessor(events, DataStatus.INSUFFICIENT),
            )

            result = service.evaluate(Market.A_SHARE, NOW, "premarket")

            self.assertEqual(result.decision.final_level, RiskLevel.UNKNOWN)
            self.assertEqual(result.error_class, "data_unavailable")
            self.assertEqual(result.quant_assessment.reliability_grade, "UNAVAILABLE")
            self.assertTrue(Path(result.report_path).is_file())
            self.assertNotIn("secret upstream detail", Path(result.report_path).read_text())

    def test_missing_or_failed_model_produces_unknown(self) -> None:
        for model in (Model([], unavailable=True), Model([], error=RuntimeError("model internals"))):
            events: list[str] = []
            model.events = events
            with self.subTest(model=model), tempfile.TemporaryDirectory() as directory:
                service, _ = self._service(events, directory, probability_model=model)
                result = service.evaluate(Market.A_SHARE, NOW, "premarket")

                self.assertEqual(result.decision.final_level, RiskLevel.UNKNOWN)
                self.assertIn(result.error_class, {"model_unavailable", "model_error"})
                self.assertTrue(Path(result.report_path).is_file())

    def test_llm_failure_keeps_quant_decision_and_saved_report(self) -> None:
        events: list[str] = []
        with tempfile.TemporaryDirectory() as directory:
            service, repository = self._service(
                events,
                directory,
                reasoning=Reasoner(events, RuntimeError("provider private failure")),
            )

            result = service.evaluate(Market.A_SHARE, NOW, "premarket")

            self.assertEqual(result.decision.final_level, RiskLevel.ORANGE)
            self.assertEqual(repository.saved_decision.final_level, RiskLevel.ORANGE)
            self.assertEqual(result.context_assessment.reasoning_status, "fallback")
            self.assertEqual(result.error_class, "reasoning_error")
            self.assertTrue(Path(result.report_path).is_file())
            self.assertNotIn("provider private failure", Path(result.report_path).read_text())

    def test_repository_stage_failure_is_coarse_and_does_not_raise(self) -> None:
        for stage in ("save_snapshot", "save_predictions", "save_reasoning", "save_decision"):
            events: list[str] = []
            repository = Repository(events, fail_stage=stage)
            with self.subTest(stage=stage), tempfile.TemporaryDirectory() as directory:
                service, _ = self._service(events, directory, repository=repository)
                result = service.evaluate(Market.A_SHARE, NOW, "premarket")

                self.assertEqual(result.error_class, "repository_error")
                self.assertTrue(Path(result.report_path).is_file())
                self.assertNotIn("private repository detail", Path(result.report_path).read_text())

    def test_unpersisted_llm_assessment_cannot_raise_quant_baseline(self) -> None:
        events: list[str] = []
        repository = Repository(events, fail_stage="save_reasoning")
        with tempfile.TemporaryDirectory() as directory:
            service, _ = self._service(events, directory, repository=repository)

            result = service.evaluate(Market.A_SHARE, NOW, "premarket")

            self.assertEqual(result.decision.baseline_level, RiskLevel.ORANGE)
            self.assertEqual(result.decision.final_level, RiskLevel.ORANGE)
            self.assertEqual(result.error_class, "repository_error")

    def test_notifier_failure_keeps_persisted_decision_and_report(self) -> None:
        events: list[str] = []
        with tempfile.TemporaryDirectory() as directory:
            service, repository = self._service(
                events,
                directory,
                notifier=Notifier(events, RuntimeError("webhook secret")),
            )

            result = service.evaluate(Market.A_SHARE, NOW, "premarket")

            self.assertEqual(result.error_class, "notifier_error")
            self.assertIsNotNone(repository.saved_decision)
            self.assertTrue(Path(result.report_path).is_file())
            self.assertNotIn("webhook secret", Path(result.report_path).read_text())

    def test_ordinary_intraday_yellow_skips_reasoning_and_notification(self) -> None:
        events: list[str] = []
        with tempfile.TemporaryDirectory() as directory:
            service, _ = self._service(events, directory)
            service.probability_model = Model(events)
            service.probability_model.predict = lambda snapshot: replace(
                Model(events).predict(snapshot),
                crash_1d_probability=0.025,
                crash_3d_probability=0.04,
            )

            result = service.evaluate(Market.A_SHARE, NOW, "intraday-0935")

        self.assertEqual(result.decision.final_level, RiskLevel.YELLOW)
        self.assertNotIn("reason", events)
        self.assertNotIn("notify", events)


if __name__ == "__main__":
    main()
