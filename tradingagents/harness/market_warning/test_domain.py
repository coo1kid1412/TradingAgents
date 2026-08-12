"""Domain model contract tests for the market warning package."""

from datetime import datetime, timezone
from pathlib import Path
import sys
from unittest import TestCase, main

# Keep the brief's direct-script command rooted at this worktree.
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from tradingagents.harness.market_warning import domain as domain_models
from tradingagents.harness.market_warning.domain import (
    Evidence,
    FeatureSnapshot,
    FinalWarningDecision,
    LLMContextAssessment,
    Market,
    QuantRiskAssessment,
    RawMarketSnapshot,
    RiskLevel,
)


AS_OF = datetime(2026, 8, 1, 9, 35, tzinfo=timezone.utc)


def make_quant(**changes):
    values = {
        "crash_1d_probability": 0.25,
        "crash_3d_probability": 0.35,
        "market_phase": "FIRST_SHOCK",
        "base_rate_1d": 0.10,
        "base_rate_3d": 0.20,
        "reliability_grade": "A",
        "model_version": "test-model",
        "calibration_version": "test-calibration",
        "top_contributors": (),
    }
    values.update(changes)
    return QuantRiskAssessment(**values)


def make_context(**changes):
    values = {
        "market_scenario": "stable",
        "causal_chain": ("breadth is stable",),
        "supporting_evidence_ids": ("e1",),
        "conflicting_evidence_ids": ("e2",),
        "overlooked_risks": (),
        "recommended_risk_level": "GREEN",
        "confidence": 0.80,
        "action_reason": "No escalation.",
        "reasoning_status": "validated",
    }
    values.update(changes)
    return LLMContextAssessment(**values)


class DomainValidationTests(TestCase):
    def test_triggered_rule_validates_layer_points_and_evidence_ids(self):
        rule = domain_models.TriggeredRule(
            rule_id="pressure.volatility_acceleration",
            layer="PRESSURE",
            severity_points=2,
            observed_value=1.5,
            threshold_description=">= 1.50",
            evidence_ids=("volatility-ratio",),
        )

        self.assertEqual(rule.layer, domain_models.RuleLayer.PRESSURE)
        self.assertEqual(rule.evidence_ids, ("volatility-ratio",))
        for invalid_layer in ("MODEL", "", None):
            with self.subTest(layer=invalid_layer):
                with self.assertRaises(ValueError):
                    domain_models.TriggeredRule(
                        rule_id="invalid",
                        layer=invalid_layer,
                        severity_points=1,
                        observed_value=None,
                        threshold_description="invalid",
                        evidence_ids=("e1",),
                    )
        for invalid_points in (-1, 3, 1.5, True):
            with self.subTest(points=invalid_points):
                with self.assertRaises(ValueError):
                    domain_models.TriggeredRule(
                        rule_id="invalid",
                        layer="VULNERABILITY",
                        severity_points=invalid_points,
                        observed_value=None,
                        threshold_description="invalid",
                        evidence_ids=("e1",),
                    )
        for invalid_ids in ((), ("e1", "e1")):
            with self.subTest(evidence_ids=invalid_ids):
                with self.assertRaises(ValueError):
                    domain_models.TriggeredRule(
                        rule_id="invalid",
                        layer="VULNERABILITY",
                        severity_points=1,
                        observed_value=None,
                        threshold_description="invalid",
                        evidence_ids=invalid_ids,
                    )

    def test_rule_assessment_is_a_score_not_a_probability(self):
        rule = domain_models.TriggeredRule(
            rule_id="pressure.volatility_acceleration",
            layer="PRESSURE",
            severity_points=2,
            observed_value=1.5,
            threshold_description=">= 1.50",
            evidence_ids=("volatility-ratio",),
        )
        assessment = domain_models.RuleRiskAssessment(
            market="a_share",
            as_of_time=AS_OF,
            engine_version="rule-v1.0.0",
            manifest_sha256="a" * 64,
            risk_level="ORANGE",
            risk_score=4.0,
            market_phase="FIRST_SHOCK",
            triggered_rules=(rule,),
            missing_optional_groups=("margin",),
            reliability_grade="B",
            evaluation_latency_ms=8.5,
        )

        self.assertEqual(assessment.risk_level, RiskLevel.ORANGE)
        self.assertEqual(assessment.risk_score, 4.0)
        self.assertFalse(hasattr(assessment, "probability"))
        for invalid_score in (-0.1, 10.1, float("nan"), True):
            with self.subTest(score=invalid_score):
                with self.assertRaises(ValueError):
                    domain_models.RuleRiskAssessment(
                        market="a_share",
                        as_of_time=AS_OF,
                        engine_version="rule-v1.0.0",
                        manifest_sha256="a" * 64,
                        risk_level="ORANGE",
                        risk_score=invalid_score,
                        market_phase="FIRST_SHOCK",
                        triggered_rules=(rule,),
                        missing_optional_groups=(),
                        reliability_grade="B",
                        evaluation_latency_ms=0.0,
                    )

    def test_decision_source_and_runner_keep_rule_and_shadow_model_separate(self):
        decision = FinalWarningDecision(
            baseline_level="ORANGE",
            final_level="ORANGE",
            state_transition="GREEN_TO_ORANGE",
            entry_gate="LIMITED",
            new_position_cap_pct=5.0,
            holding_action="reduce",
            push_required=True,
            decision_reasons=("pressure confirmed",),
            data_status="fresh",
            decision_source="rule_v1",
        )
        assessment = domain_models.RuleRiskAssessment(
            market="a_share",
            as_of_time=AS_OF,
            engine_version="rule-v1.0.0",
            manifest_sha256="a" * 64,
            risk_level="ORANGE",
            risk_score=4.0,
            market_phase="FIRST_SHOCK",
            triggered_rules=(),
            missing_optional_groups=(),
            reliability_grade="A",
            evaluation_latency_ms=1.0,
        )
        runner = domain_models.RunnerResult(
            market="a_share",
            as_of_time=AS_OF,
            session_slot="intraday-0935",
            rule_assessment=assessment,
            shadow_quant_assessment=make_quant(),
            decision=decision,
        )

        self.assertEqual(decision.decision_source, domain_models.DecisionSource.RULE_V1)
        self.assertIs(runner.rule_assessment, assessment)
        self.assertIsNotNone(runner.shadow_quant_assessment)
        self.assertIsNone(runner.quant_assessment)
        with self.assertRaises(ValueError):
            FinalWarningDecision(
                baseline_level="GREEN",
                final_level="GREEN",
                state_transition="UNCHANGED",
                entry_gate="OPEN",
                new_position_cap_pct=100.0,
                holding_action="HOLD",
                push_required=False,
                decision_reasons=(),
                data_status="fresh",
                decision_source="rules-masquerading-as-model",
            )

    def test_final_decision_recovery_state_has_safe_validated_defaults(self):
        decision = FinalWarningDecision(
            baseline_level="GREEN",
            final_level="GREEN",
            state_transition="UNCHANGED",
            entry_gate="OPEN",
            new_position_cap_pct=100.0,
            holding_action="HOLD",
            push_required=False,
            decision_reasons=(),
            data_status="fresh",
        )

        self.assertEqual(decision.valid_snapshot_count, 0)
        self.assertIsNone(decision.retained_risk_level)

    def test_final_decision_validates_persisted_recovery_state(self):
        valid = FinalWarningDecision(
            baseline_level="UNKNOWN",
            final_level="UNKNOWN",
            state_transition="TO_UNKNOWN",
            entry_gate="WAIT",
            new_position_cap_pct=0.0,
            holding_action="HOLD",
            push_required=False,
            decision_reasons=(),
            data_status="stale",
            valid_snapshot_count=1,
            retained_risk_level="RED",
        )
        self.assertEqual(valid.retained_risk_level, RiskLevel.RED)

        for count in (True, -1, 1.0, "1", None):
            with self.subTest(count=count):
                with self.assertRaises(ValueError):
                    FinalWarningDecision(
                        baseline_level="GREEN",
                        final_level="GREEN",
                        state_transition="UNCHANGED",
                        entry_gate="OPEN",
                        new_position_cap_pct=100.0,
                        holding_action="HOLD",
                        push_required=False,
                        decision_reasons=(),
                        data_status="fresh",
                        valid_snapshot_count=count,
                    )
        for retained in ("GREEN", "YELLOW", "UNKNOWN", "BLUE"):
            with self.subTest(retained=retained):
                with self.assertRaises(ValueError):
                    FinalWarningDecision(
                        baseline_level="GREEN",
                        final_level="GREEN",
                        state_transition="UNCHANGED",
                        entry_gate="OPEN",
                        new_position_cap_pct=100.0,
                        holding_action="HOLD",
                        push_required=False,
                        decision_reasons=(),
                        data_status="fresh",
                        retained_risk_level=retained,
                    )

    def test_quant_assessment_rejects_probability_outside_unit_interval(self):
        with self.assertRaises(ValueError):
            make_quant(crash_1d_probability=1.01)

    def test_context_assessment_rejects_confidence_outside_unit_interval(self):
        with self.assertRaises(ValueError):
            make_context(confidence=-0.01)

    def test_context_assessment_rejects_unsupported_risk_level(self):
        with self.assertRaises(ValueError):
            make_context(recommended_risk_level="BLUE")

    def test_feature_snapshot_rejects_timezone_naive_as_of_time(self):
        with self.assertRaises(ValueError):
            FeatureSnapshot(
                market=Market.A_SHARE,
                as_of_time=datetime(2026, 8, 1, 9, 35),
                session_slot="premarket",
                feature_version="test-v1",
                features={},
                evidence=(),
                data_quality="fresh",
            )

    def test_feature_snapshot_exposes_stable_evidence_ids(self):
        evidence = (
            Evidence(evidence_id="e1", group="breadth", summary="Broad weakness"),
            Evidence(evidence_id="e2", group="volatility", summary="Volatility rising"),
        )
        snapshot = FeatureSnapshot(
            market=Market.A_SHARE,
            as_of_time=AS_OF,
            session_slot="premarket",
            feature_version="test-v1",
            features={"breadth": 0.25},
            evidence=evidence,
            data_quality="fresh",
        )

        self.assertEqual(snapshot.evidence_ids, ("e1", "e2"))
        self.assertEqual(snapshot.evidence_ids, snapshot.evidence_ids)

    def test_feature_snapshot_rejects_duplicate_evidence_ids(self):
        evidence = (
            Evidence(evidence_id="e1", group="breadth", summary="First"),
            Evidence(evidence_id="e1", group="volatility", summary="Duplicate"),
        )
        with self.assertRaises(ValueError):
            FeatureSnapshot(
                market=Market.US,
                as_of_time=AS_OF,
                session_slot="premarket",
                feature_version="test-v1",
                features={},
                evidence=evidence,
                data_quality="fresh",
            )

    def test_raw_snapshot_copies_source_times_from_caller(self):
        source_times = {"primary": AS_OF}
        snapshot = RawMarketSnapshot(
            market=Market.A_SHARE,
            as_of_time=AS_OF,
            session_slot="premarket",
            source_times=source_times,
        )

        source_times["late"] = datetime(2026, 8, 1, 9, 36)

        self.assertEqual(snapshot.source_times, {"primary": AS_OF})

    def test_raw_snapshot_source_times_are_read_only(self):
        snapshot = RawMarketSnapshot(
            market=Market.A_SHARE,
            as_of_time=AS_OF,
            session_slot="premarket",
            source_times={"primary": AS_OF},
        )

        with self.assertRaises(TypeError):
            snapshot.source_times["late"] = AS_OF

    def test_feature_snapshot_copies_features_from_caller(self):
        features = {"breadth": 0.25}
        snapshot = FeatureSnapshot(
            market=Market.A_SHARE,
            as_of_time=AS_OF,
            session_slot="premarket",
            feature_version="test-v1",
            features=features,
            evidence=(),
            data_quality="fresh",
        )

        features["breadth"] = 0.90
        features["volatility"] = 0.80

        self.assertEqual(snapshot.features, {"breadth": 0.25})

    def test_feature_snapshot_features_are_read_only(self):
        snapshot = FeatureSnapshot(
            market=Market.A_SHARE,
            as_of_time=AS_OF,
            session_slot="premarket",
            feature_version="test-v1",
            features={"breadth": 0.25},
            evidence=(),
            data_quality="fresh",
        )

        with self.assertRaises(TypeError):
            snapshot.features["breadth"] = 0.90

    def test_feature_snapshot_defaults_to_conservative_reliability_grade(self):
        snapshot = FeatureSnapshot(
            market=Market.A_SHARE,
            as_of_time=AS_OF,
            session_slot="premarket",
            feature_version="test-v1",
            features={},
            evidence=(),
            data_quality="fresh",
        )

        self.assertEqual(snapshot.reliability_grade, "D")

    def test_feature_snapshot_freezes_timezone_aware_source_times(self):
        source_times = {"exchange": AS_OF}
        snapshot = FeatureSnapshot(
            market=Market.A_SHARE,
            as_of_time=AS_OF,
            session_slot="premarket",
            feature_version="test-v1",
            features={},
            evidence=(),
            data_quality="fresh",
            source_times=source_times,
        )

        source_times["late"] = AS_OF

        self.assertEqual(snapshot.source_times, {"exchange": AS_OF})
        with self.assertRaises(TypeError):
            snapshot.source_times["late"] = AS_OF
        with self.assertRaises(ValueError):
            FeatureSnapshot(
                market=Market.A_SHARE,
                as_of_time=AS_OF,
                session_slot="premarket",
                feature_version="test-v1",
                features={},
                evidence=(),
                data_quality="fresh",
                source_times={"naive": datetime(2026, 8, 1, 9, 35)},
            )


if __name__ == "__main__":
    main()
