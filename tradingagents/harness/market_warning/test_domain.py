"""Domain model contract tests for the market warning package."""

from datetime import datetime, timezone
from pathlib import Path
import sys
from unittest import TestCase, main

# Keep the brief's direct-script command rooted at this worktree.
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from tradingagents.harness.market_warning.domain import (
    Evidence,
    FeatureSnapshot,
    LLMContextAssessment,
    Market,
    QuantRiskAssessment,
    RawMarketSnapshot,
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
