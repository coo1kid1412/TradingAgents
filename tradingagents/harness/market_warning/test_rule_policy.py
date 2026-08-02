"""Boundary and safety tests for the deterministic A-share rule engine."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import tempfile
from unittest import TestCase, main

from tradingagents.harness.market_warning.domain import (
    Evidence,
    FeatureSnapshot,
    Market,
    RiskLevel,
)
from tradingagents.harness.market_warning.rule_policy import (
    evaluate_a_share_rules,
    load_rule_manifest,
    manifest_sha256,
)


AS_OF = datetime(2026, 8, 3, 1, 35, tzinfo=timezone.utc)
MANIFEST_PATH = Path(__file__).with_name("rule_manifest_v1.json")


def make_snapshot(**feature_changes) -> FeatureSnapshot:
    features = {
        "return_1d": 0.01,
        "audited_ohlc_return_1d": 0.01,
        "return_5d": 0.02,
        "return_60d": 0.05,
        "drawdown_20d": -0.01,
        "ma20_distance": 0.01,
        "volatility_ratio_5d_20d": 1.0,
        "range_zscore_20d": 0.0,
        "close_location": 0.50,
        "abnormal_range_weak_close_transition": False,
        "breadth_deterioration_transition": False,
        "breadth_up_pct": 55.0,
        "breadth_above_ma20_pct": 60.0,
        "new_low_20d_pct": 5.0,
        "industry_decline_pct": 40.0,
        "margin_balance_growth_20d": 0.02,
        "margin_balance_contracting_from_high": False,
        "turnover_percentile_20d": 0.50,
        "limit_down_pct": 0.1,
        "realtime_breadth_coverage_pct": 100.0,
        "realtime_breadth_staleness_minutes": 0.0,
        "market_phase": "FIRST_SHOCK",
    }
    features.update(feature_changes)
    evidence = tuple(
        Evidence(
            evidence_id=f"ev-{name}",
            group="rule-input",
            summary=name,
            value=value,
            source="fixture",
            as_of_time=AS_OF,
        )
        for name, value in features.items()
    )
    return FeatureSnapshot(
        market=Market.A_SHARE,
        as_of_time=AS_OF,
        session_slot="intraday-0935",
        feature_version="market-warning-v2",
        features=features,
        evidence=evidence,
        data_quality="fresh",
        reliability_grade="A",
        source_times={"fixture": AS_OF},
    )


def triggered_ids(snapshot: FeatureSnapshot) -> set[str]:
    result = evaluate_a_share_rules(snapshot, load_rule_manifest(MANIFEST_PATH))
    return {item.rule_id for item in result.triggered_rules}


class RuleManifestTests(TestCase):
    def test_manifest_is_versioned_and_checksum_is_deterministic(self):
        manifest = load_rule_manifest(MANIFEST_PATH)

        self.assertEqual(manifest.engine_version, "rule-v1.0.0")
        self.assertEqual(manifest.market, Market.A_SHARE)
        self.assertEqual(len(manifest_sha256(MANIFEST_PATH)), 64)
        self.assertEqual(manifest_sha256(MANIFEST_PATH), manifest_sha256(MANIFEST_PATH))

    def test_assessment_uses_checksum_of_the_manifest_that_was_loaded(self):
        with tempfile.TemporaryDirectory() as directory:
            candidate = Path(directory) / "candidate.json"
            candidate.write_text(
                MANIFEST_PATH.read_text(encoding="utf-8") + "\n",
                encoding="utf-8",
            )
            manifest = load_rule_manifest(candidate)

            assessment = evaluate_a_share_rules(make_snapshot(), manifest)

            self.assertEqual(assessment.manifest_sha256, manifest_sha256(candidate))
            self.assertNotEqual(assessment.manifest_sha256, manifest_sha256(MANIFEST_PATH))


class AShareRuleEvaluatorTests(TestCase):
    def setUp(self):
        self.manifest = load_rule_manifest(MANIFEST_PATH)

    def test_rule_thresholds_use_frozen_inclusive_boundaries(self):
        cases = (
            (
                "vulnerability.return_60d",
                {"return_60d": 0.15},
                {"return_60d": 0.149999},
            ),
            (
                "vulnerability.margin_growth_20d",
                {"margin_balance_growth_20d": 0.08},
                {"margin_balance_growth_20d": 0.079999},
            ),
            (
                "vulnerability.turnover_narrow_breadth",
                {"turnover_percentile_20d": 0.80, "breadth_above_ma20_pct": 50.0},
                {"turnover_percentile_20d": 0.799999, "breadth_above_ma20_pct": 50.0},
            ),
            (
                "vulnerability.index_breadth_divergence",
                {"return_5d": 0.0, "breadth_up_pct": 40.0},
                {"return_5d": -0.000001, "breadth_up_pct": 40.0},
            ),
            (
                "pressure.volatility_acceleration",
                {"volatility_ratio_5d_20d": 1.50},
                {"volatility_ratio_5d_20d": 1.499999},
            ),
            (
                "continuation.drawdown_below_ma20",
                {"drawdown_20d": -0.05, "ma20_distance": -0.000001},
                {"drawdown_20d": -0.049999, "ma20_distance": -0.000001},
            ),
            (
                "continuation.negative_return_volatility",
                {"return_5d": -0.000001, "volatility_ratio_5d_20d": 1.20},
                {"return_5d": 0.0, "volatility_ratio_5d_20d": 1.20},
            ),
        )
        for rule_id, passing, failing in cases:
            with self.subTest(rule_id=rule_id, side="passing"):
                self.assertIn(rule_id, triggered_ids(make_snapshot(**passing)))
            with self.subTest(rule_id=rule_id, side="failing"):
                self.assertNotIn(rule_id, triggered_ids(make_snapshot(**failing)))

    def test_boolean_pressure_and_continuation_rules_require_exact_confirmation(self):
        pressure = triggered_ids(make_snapshot(abnormal_range_weak_close_transition=True))
        breadth = triggered_ids(make_snapshot(breadth_deterioration_transition=True))
        continuation = triggered_ids(
            make_snapshot(
                margin_balance_contracting_from_high=True,
                breadth_deterioration_transition=True,
            )
        )

        self.assertIn("pressure.abnormal_range_weak_close", pressure)
        self.assertIn("pressure.breadth_deterioration", breadth)
        self.assertIn("continuation.margin_contracting_breadth", continuation)

    def test_new_low_is_display_only_and_does_not_change_score_or_level(self):
        low = evaluate_a_share_rules(make_snapshot(new_low_20d_pct=0.0), self.manifest)
        high = evaluate_a_share_rules(make_snapshot(new_low_20d_pct=100.0), self.manifest)

        self.assertEqual(low.risk_score, high.risk_score)
        self.assertEqual(low.risk_level, high.risk_level)
        self.assertFalse(any("new_low" in item.rule_id for item in high.triggered_rules))

    def test_missing_optional_groups_reduce_reliability_without_unknown(self):
        snapshot = make_snapshot(
            margin_balance_growth_20d=None,
            margin_balance_contracting_from_high=None,
            turnover_percentile_20d=None,
        )

        result = evaluate_a_share_rules(snapshot, self.manifest)

        self.assertNotEqual(result.risk_level, RiskLevel.UNKNOWN)
        self.assertEqual(result.reliability_grade, "B")
        self.assertEqual(result.missing_optional_groups, ("margin", "turnover"))

    def test_missing_core_or_unusable_snapshot_returns_unknown(self):
        missing = evaluate_a_share_rules(make_snapshot(return_1d=None), self.manifest)
        stale_snapshot = make_snapshot()
        stale_snapshot = FeatureSnapshot(
            market=stale_snapshot.market,
            as_of_time=stale_snapshot.as_of_time,
            session_slot=stale_snapshot.session_slot,
            feature_version=stale_snapshot.feature_version,
            features=stale_snapshot.features,
            evidence=stale_snapshot.evidence,
            data_quality="stale",
            reliability_grade="UNAVAILABLE",
            source_times=stale_snapshot.source_times,
        )
        stale = evaluate_a_share_rules(stale_snapshot, self.manifest)

        self.assertEqual(missing.risk_level, RiskLevel.UNKNOWN)
        self.assertEqual(stale.risk_level, RiskLevel.UNKNOWN)

    def test_invalid_intraday_cross_section_blocks_breadth_red_rules(self):
        snapshot = make_snapshot(
            return_1d=-0.01,
            breadth_up_pct=10.0,
            limit_down_pct=4.0,
            breadth_deterioration_transition=True,
            realtime_breadth_coverage_pct=79.99,
        )

        result = evaluate_a_share_rules(snapshot, self.manifest)
        ids = {item.rule_id for item in result.triggered_rules}

        self.assertNotIn("hard.declining_breadth", ids)
        self.assertNotIn("hard.limit_down", ids)
        self.assertNotIn("pressure.breadth_deterioration", ids)
        self.assertNotEqual(result.risk_level, RiskLevel.RED)
        self.assertEqual(result.reliability_grade, "C")

    def test_each_hard_trigger_produces_red_at_the_frozen_boundary(self):
        cases = (
            (
                "hard.range_weak_close",
                {
                    "range_zscore_20d": 3.0,
                    "close_location": 0.15,
                    "audited_ohlc_return_1d": -0.02,
                },
            ),
            (
                "hard.declining_breadth",
                {"return_1d": -0.000001, "breadth_up_pct": 15.0},
            ),
            (
                "hard.limit_down",
                {"return_1d": -0.000001, "limit_down_pct": 2.0},
            ),
        )
        for rule_id, changes in cases:
            with self.subTest(rule_id=rule_id):
                result = evaluate_a_share_rules(make_snapshot(**changes), self.manifest)
                self.assertEqual(result.risk_level, RiskLevel.RED)
                self.assertIn(rule_id, {item.rule_id for item in result.triggered_rules})

    def test_weak_vulnerability_signals_cannot_mechanically_create_red(self):
        result = evaluate_a_share_rules(
            make_snapshot(
                return_60d=0.30,
                margin_balance_growth_20d=0.20,
                turnover_percentile_20d=0.95,
                breadth_above_ma20_pct=30.0,
                return_5d=0.10,
                breadth_up_pct=35.0,
            ),
            self.manifest,
        )

        self.assertEqual(result.risk_score, 2.0)
        self.assertEqual(result.risk_level, RiskLevel.YELLOW)

    def test_two_pressures_and_continuation_require_negative_current_return_for_red(self):
        changes = {
            "volatility_ratio_5d_20d": 1.50,
            "breadth_deterioration_transition": True,
            "drawdown_20d": -0.05,
            "ma20_distance": -0.01,
        }
        positive = evaluate_a_share_rules(make_snapshot(return_1d=0.001, **changes), self.manifest)
        negative = evaluate_a_share_rules(make_snapshot(return_1d=-0.001, **changes), self.manifest)

        self.assertEqual(positive.risk_level, RiskLevel.ORANGE)
        self.assertEqual(negative.risk_level, RiskLevel.RED)

    def test_orange_with_new_pressure_and_continuation_upgrades_to_red(self):
        previous = evaluate_a_share_rules(
            make_snapshot(abnormal_range_weak_close_transition=True),
            self.manifest,
        )
        current = evaluate_a_share_rules(
            make_snapshot(
                abnormal_range_weak_close_transition=True,
                volatility_ratio_5d_20d=1.50,
                drawdown_20d=-0.05,
                ma20_distance=-0.01,
            ),
            self.manifest,
            previous_assessment=previous,
        )

        self.assertEqual(previous.risk_level, RiskLevel.ORANGE)
        self.assertEqual(current.risk_level, RiskLevel.RED)

    def test_evaluator_rejects_non_a_share_snapshot(self):
        a_share = make_snapshot()
        us = FeatureSnapshot(
            market=Market.US,
            as_of_time=a_share.as_of_time,
            session_slot=a_share.session_slot,
            feature_version=a_share.feature_version,
            features=a_share.features,
            evidence=a_share.evidence,
            data_quality=a_share.data_quality,
            reliability_grade=a_share.reliability_grade,
            source_times=a_share.source_times,
        )

        with self.assertRaises(ValueError):
            evaluate_a_share_rules(us, self.manifest)


if __name__ == "__main__":
    main()
