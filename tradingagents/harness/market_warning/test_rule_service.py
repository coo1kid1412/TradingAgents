"""Application-service tests for the deterministic rule fast path."""

from __future__ import annotations

import tempfile
from datetime import datetime, timezone
from pathlib import Path
from unittest import TestCase, main

from tradingagents.harness.market_warning.domain import (
    DataStatus,
    DecisionSource,
    Evidence,
    FeatureSnapshot,
    FinalWarningDecision,
    Market,
    MarketDataPoint,
    MarketPhase,
    QuantRiskAssessment,
    RawMarketSnapshot,
    RiskLevel,
    RuleRiskAssessment,
    TriggeredRule,
)
from tradingagents.harness.market_warning.quality import DataQualityAssessment
from tradingagents.harness.market_warning.rule_service import RuleMarketWarningService


NOW = datetime(2026, 8, 3, 1, 35, tzinfo=timezone.utc)


def rule_assessment(level: RiskLevel, latency: float = 0.0) -> RuleRiskAssessment:
    return RuleRiskAssessment(
        market=Market.A_SHARE,
        as_of_time=NOW,
        engine_version="rule-v1.0.0",
        manifest_sha256="a" * 64,
        risk_level=level,
        risk_score=4.0 if level in {RiskLevel.ORANGE, RiskLevel.RED} else 0.0,
        market_phase=MarketPhase.FIRST_SHOCK,
        triggered_rules=(
            TriggeredRule(
                rule_id="pressure.volatility_acceleration",
                layer="PRESSURE",
                severity_points=2,
                observed_value=1.6,
                threshold_description=">= 1.50",
                evidence_ids=("ev-rule",),
            ),
        ) if level in {RiskLevel.ORANGE, RiskLevel.RED} else (),
        missing_optional_groups=(),
        reliability_grade="A" if level != RiskLevel.UNKNOWN else "UNAVAILABLE",
        evaluation_latency_ms=latency,
    )


class DataPort:
    def __init__(self, events, error=None):
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
    def __init__(self, events):
        self.events = events

    def build(self, raw, _history):
        self.events.append("build_features")
        return FeatureSnapshot(
            market=raw.market,
            as_of_time=raw.as_of_time,
            session_slot=raw.session_slot,
            feature_version="market-warning-v2",
            features={"market_phase": "FIRST_SHOCK"},
            evidence=(
                Evidence(
                    evidence_id="ev-rule",
                    group="rule",
                    summary="rule evidence",
                    source="fixture",
                    as_of_time=raw.as_of_time,
                ),
            ),
            data_quality=DataStatus.FRESH,
            reliability_grade="A",
            source_times={"fixture": raw.as_of_time},
        )


class RuleEvaluator:
    def __init__(self, events, level=RiskLevel.ORANGE, error=None):
        self.events = events
        self.level = level
        self.error = error

    def __call__(self, snapshot, previous_assessment=None):
        self.events.append("evaluate_rules")
        if self.error:
            raise self.error
        return rule_assessment(self.level)


class Repository:
    def __init__(self, events, fail_stage=None, previous=None):
        self.events = events
        self.fail_stage = fail_stage
        self.previous = previous
        self.decision = None

    def _stage(self, name, value):
        self.events.append(name)
        if self.fail_stage == name:
            raise RuntimeError(name)
        return value

    def load_evaluation(self, *_):
        return None

    def save_feature_snapshot(self, _snapshot):
        return self._stage("save_snapshot", 11)

    def load_previous_rule_assessment(self, *_):
        return self._stage("load_previous_rule", None)

    def save_rule_assessment(self, _snapshot_id, _assessment):
        return self._stage("save_rule", 21)

    def load_previous_decision(self, *_):
        return self._stage("load_previous", self.previous)

    def save_decision(self, _snapshot_id, prediction_ids, _reasoning_id, decision, **kwargs):
        self.decision = decision
        self.saved_prediction_ids = prediction_ids
        self.saved_kwargs = kwargs
        return self._stage("save_decision", 31)


class Notifier:
    def __init__(self, events, error=None):
        self.events = events
        self.error = error

    def notify(self, _result):
        self.events.append("notify")
        if self.error:
            raise self.error
        return True


class ShadowModel:
    def __init__(self, events, error=None):
        self.events = events
        self.error = error

    def predict(self, _snapshot):
        self.events.append("shadow_model")
        if self.error:
            raise self.error
        return QuantRiskAssessment(
            crash_1d_probability=0.02,
            crash_3d_probability=0.03,
            market_phase="FIRST_SHOCK",
            base_rate_1d=0.01,
            base_rate_3d=0.02,
            reliability_grade="B",
            model_version="shadow-v1",
            calibration_version="shadow-cal-v1",
            top_contributors=(),
        )


class PostAlertReasoning:
    def __init__(self, events, error=None):
        self.events = events
        self.error = error

    def assess_rule_alert(self, *_):
        self.events.append("post_alert_reasoning")
        if self.error:
            raise self.error
        return None


def quality_assessor(events):
    def assess(raw):
        events.append("assess_quality")
        return DataQualityAssessment(
            market=raw.market,
            as_of_time=raw.as_of_time,
            session_slot=raw.session_slot,
            status=DataStatus.FRESH,
            reliability_grade="A",
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


def reporter(events):
    def write(result, _previous, root):
        events.append("write_report")
        path = Path(root) / "rule-report.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(result.decision.final_level.value, encoding="utf-8")
        return path

    return write


class RuleMarketWarningServiceTests(TestCase):
    def _service(self, events, directory, **changes):
        repository = changes.pop("repository", Repository(events))
        values = {
            "data_port": DataPort(events),
            "feature_strategy": FeatureStrategy(events),
            "rule_evaluator": RuleEvaluator(events),
            "repository": repository,
            "quality_assessor": quality_assessor(events),
            "history_loader": lambda _market, _time: (),
            "report_root": Path(directory),
            "reporter": reporter(events),
            "notifier": Notifier(events),
            "shadow_model": ShadowModel(events),
            "post_alert_reasoning": PostAlertReasoning(events),
            "monotonic_ns": iter((1_000_000, 9_500_000)).__next__,
        }
        values.update(changes)
        return RuleMarketWarningService(**values), repository

    def test_fast_path_persists_and_notifies_before_shadow_and_reasoning(self):
        events = []
        with tempfile.TemporaryDirectory() as directory:
            service, repository = self._service(events, directory)

            result = service.evaluate(Market.A_SHARE, NOW, "intraday-0935")

        self.assertEqual(
            events,
            [
                "load_data",
                "assess_quality",
                "build_features",
                "save_snapshot",
                "load_previous_rule",
                "evaluate_rules",
                "save_rule",
                "load_previous",
                "save_decision",
                "write_report",
                "notify",
                "shadow_model",
                "post_alert_reasoning",
            ],
        )
        self.assertEqual(result.decision.final_level, RiskLevel.ORANGE)
        self.assertEqual(result.decision.decision_source, DecisionSource.RULE_V1)
        self.assertEqual(result.rule_assessment.evaluation_latency_ms, 8.5)
        self.assertIsNotNone(result.shadow_quant_assessment)
        self.assertEqual(repository.saved_prediction_ids, ())
        self.assertEqual(repository.saved_kwargs["rule_assessment_id"], 21)

    def test_intraday_yellow_is_silent_and_skips_all_slow_paths(self):
        events = []
        with tempfile.TemporaryDirectory() as directory:
            service, _ = self._service(
                events,
                directory,
                rule_evaluator=RuleEvaluator(events, RiskLevel.YELLOW),
            )
            result = service.evaluate(Market.A_SHARE, NOW, "intraday-0935")

        self.assertEqual(result.decision.final_level, RiskLevel.YELLOW)
        self.assertNotIn("notify", events)
        self.assertNotIn("shadow_model", events)
        self.assertNotIn("post_alert_reasoning", events)

    def test_direct_red_is_pushed_on_the_first_valid_scan(self):
        events = []
        with tempfile.TemporaryDirectory() as directory:
            service, _ = self._service(
                events,
                directory,
                rule_evaluator=RuleEvaluator(events, RiskLevel.RED),
                shadow_model=None,
                post_alert_reasoning=None,
            )
            result = service.evaluate(Market.A_SHARE, NOW, "intraday-0935")

        self.assertEqual(result.decision.state_transition, "INITIAL_RED")
        self.assertTrue(result.decision.push_required)
        self.assertIn("notify", events)

    def test_data_rule_or_repository_failure_fails_closed_without_notification(self):
        cases = {
            "data": {"data_port": DataPort([], RuntimeError("data"))},
            "rule": {"rule_evaluator": RuleEvaluator([], error=RuntimeError("rule"))},
            "repository": {"repository": Repository([], fail_stage="save_rule")},
        }
        for name, changes in cases.items():
            events = []
            for value in changes.values():
                if hasattr(value, "events"):
                    value.events = events
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                service, _ = self._service(events, directory, **changes)
                result = service.evaluate(Market.A_SHARE, NOW, "intraday-0935")

            self.assertEqual(result.decision.final_level, RiskLevel.UNKNOWN)
            self.assertNotIn("notify", events)

    def test_shadow_and_post_alert_failures_never_change_persisted_decision(self):
        events = []
        with tempfile.TemporaryDirectory() as directory:
            service, repository = self._service(
                events,
                directory,
                shadow_model=ShadowModel(events, RuntimeError("shadow")),
                post_alert_reasoning=PostAlertReasoning(events, RuntimeError("reasoning")),
            )
            result = service.evaluate(Market.A_SHARE, NOW, "intraday-0935")

        self.assertEqual(repository.decision.final_level, RiskLevel.ORANGE)
        self.assertEqual(result.decision, repository.decision)
        self.assertIsNone(result.shadow_quant_assessment)
        self.assertIn("shadow_model", events)
        self.assertIn("post_alert_reasoning", events)


if __name__ == "__main__":
    main()
