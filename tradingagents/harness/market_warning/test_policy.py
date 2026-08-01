"""Contract tests for the deterministic market-warning policy."""

from datetime import datetime, timezone
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest import TestCase, main

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from tradingagents.harness.market_warning.domain import (
    DataStatus,
    Evidence,
    FeatureSnapshot,
    LLMContextAssessment,
    Market,
    MarketPhase,
    QuantRiskAssessment,
    RiskLevel,
)
from tradingagents.harness.market_warning.policy import (
    POLICY_VERSION,
    apply_llm_adjustment,
    baseline_level,
    build_final_decision,
    transition,
)


AS_OF = datetime(2026, 8, 1, 9, 35, tzinfo=timezone.utc)


def make_snapshot(*, market=Market.A_SHARE, features=None, data_quality=DataStatus.FRESH, reliability="A"):
    return FeatureSnapshot(
        market=market,
        as_of_time=AS_OF,
        session_slot="intraday",
        feature_version="test-features-v1",
        features=features or {},
        evidence=(
            Evidence("e1", "transition", "First evidence"),
            Evidence("e2", "breadth", "Second evidence"),
            Evidence("e3", "credit", "Third evidence"),
        ),
        data_quality=data_quality,
        reliability_grade=reliability,
    )


def make_quant(*, probability_1d=0.01, probability_3d=0.02, base_rate_1d=0.01, base_rate_3d=0.02, reliability="A"):
    return QuantRiskAssessment(
        crash_1d_probability=probability_1d,
        crash_3d_probability=probability_3d,
        market_phase=MarketPhase.FIRST_SHOCK,
        base_rate_1d=base_rate_1d,
        base_rate_3d=base_rate_3d,
        reliability_grade=reliability,
        model_version="test-model",
        calibration_version="test-calibration",
        top_contributors=(),
    )


def make_context(*, recommended=RiskLevel.YELLOW, confidence=0.80, supporting=("e1", "e2"), conflicting=(), status="validated"):
    return LLMContextAssessment(
        market_scenario="Risk is accumulating.",
        causal_chain=("Evidence changed before price.",),
        supporting_evidence_ids=supporting,
        conflicting_evidence_ids=conflicting,
        overlooked_risks=("liquidity",),
        recommended_risk_level=recommended,
        confidence=confidence,
        action_reason="Escalate only when evidence is valid.",
        reasoning_status=status,
    )


class BaselinePolicyTests(TestCase):
    def test_policy_version_is_frozen(self):
        self.assertEqual(POLICY_VERSION, "market-warning-policy-v1")

    def test_probability_multiple_thresholds_are_exact(self):
        cases = (
            (0.019999, False, RiskLevel.GREEN),
            (0.020000, False, RiskLevel.YELLOW),
            (0.040000, False, RiskLevel.YELLOW),
            (0.040000, True, RiskLevel.ORANGE),
            (0.080000, False, RiskLevel.RED),
        )
        for probability, signal, expected in cases:
            with self.subTest(probability=probability, signal=signal):
                snapshot = make_snapshot(features={"pressure_transition_signal": signal})
                quant = make_quant(probability_1d=probability, probability_3d=0.01, base_rate_3d=0.01)
                self.assertEqual(baseline_level(quant, snapshot), expected)

    def test_stricter_horizon_wins_in_both_directions(self):
        cases = (
            (0.081, 0.01, RiskLevel.RED),
            (0.01, 0.081, RiskLevel.RED),
            (0.041, 0.01, RiskLevel.ORANGE),
            (0.01, 0.041, RiskLevel.ORANGE),
        )
        snapshot = make_snapshot(features={"pressure_transition_signal": True})
        for probability_1d, probability_3d, expected in cases:
            with self.subTest(probability_1d=probability_1d, probability_3d=probability_3d):
                self.assertEqual(
                    baseline_level(
                        make_quant(
                            probability_1d=probability_1d,
                            probability_3d=probability_3d,
                            base_rate_1d=0.01,
                            base_rate_3d=0.01,
                        ),
                        snapshot,
                    ),
                    expected,
                )

    def test_unusable_data_or_quant_reliability_is_unknown(self):
        for status in (DataStatus.CONFLICTED, DataStatus.STALE, DataStatus.INSUFFICIENT):
            with self.subTest(status=status):
                self.assertEqual(
                    baseline_level(make_quant(probability_1d=0.50), make_snapshot(data_quality=status)),
                    RiskLevel.UNKNOWN,
                )
        self.assertEqual(
            baseline_level(make_quant(probability_1d=0.50, reliability="UNAVAILABLE"), make_snapshot()),
            RiskLevel.UNKNOWN,
        )

    def test_zero_or_invalid_base_rates_fail_closed_as_unknown(self):
        values = (0.0, -0.01, float("nan"), float("inf"), True, "0.01", None)
        for value in values:
            with self.subTest(value=value):
                quant = SimpleNamespace(
                    crash_1d_probability=0.08,
                    crash_3d_probability=0.08,
                    base_rate_1d=value,
                    base_rate_3d=0.01,
                    reliability_grade="A",
                )
                self.assertEqual(baseline_level(quant, make_snapshot()), RiskLevel.UNKNOWN)

    def test_invalid_probability_values_fail_closed_as_unknown(self):
        for value in (float("nan"), float("inf"), True, "0.08", None, -0.01, 1.01):
            with self.subTest(value=value):
                quant = SimpleNamespace(
                    crash_1d_probability=value,
                    crash_3d_probability=0.01,
                    base_rate_1d=0.01,
                    base_rate_3d=0.01,
                    reliability_grade="A",
                )
                self.assertEqual(baseline_level(quant, make_snapshot()), RiskLevel.UNKNOWN)

    def test_market_range_hard_trigger_uses_exact_market_thresholds(self):
        cases = (
            (Market.A_SHARE, -0.0200),
            (Market.US, -0.0150),
        )
        for market, daily_return in cases:
            with self.subTest(market=market):
                snapshot = make_snapshot(
                    market=market,
                    features={"range_zscore_20d": 3.0, "close_location": 0.15, "return_1d": daily_return},
                )
                self.assertEqual(baseline_level(make_quant(), snapshot), RiskLevel.RED)

    def test_a_share_breadth_hard_triggers_require_negative_index_confirmation(self):
        cases = (
            ({"breadth_up_pct": 15.0, "return_1d": -0.001}, RiskLevel.RED),
            ({"limit_down_pct": 2.0, "return_1d": -0.001}, RiskLevel.RED),
            ({"breadth_up_pct": 15.0, "return_1d": 0.0}, RiskLevel.GREEN),
            ({"limit_down_pct": 2.0, "return_1d": 0.001}, RiskLevel.GREEN),
        )
        for features, expected in cases:
            with self.subTest(features=features):
                self.assertEqual(baseline_level(make_quant(), make_snapshot(features=features)), expected)

    def test_us_credit_hard_trigger_requires_credit_and_volatility_confirmation(self):
        cases = (
            ({"hyg_lqd_relative_return_5d": -0.015, "vix_change_5d": 0.20}, RiskLevel.RED),
            ({"hyg_lqd_relative_return_5d": -0.015, "vix_vix3m_ratio": 1.0}, RiskLevel.RED),
            ({"hyg_lqd_relative_return_5d": -0.0149, "vix_change_5d": 0.50}, RiskLevel.GREEN),
            ({"hyg_lqd_relative_return_5d": -0.02, "vix_change_5d": 0.19, "vix_vix3m_ratio": 0.99}, RiskLevel.GREEN),
        )
        for features, expected in cases:
            with self.subTest(features=features):
                self.assertEqual(
                    baseline_level(make_quant(), make_snapshot(market=Market.US, features=features)),
                    expected,
                )

    def test_source_failure_never_becomes_a_hard_market_trigger(self):
        hard = {"range_zscore_20d": 9.0, "close_location": 0.0, "return_1d": -0.20}
        for status in (DataStatus.CONFLICTED, DataStatus.STALE, DataStatus.INSUFFICIENT):
            with self.subTest(status=status):
                self.assertEqual(
                    baseline_level(make_quant(), make_snapshot(features=hard, data_quality=status)),
                    RiskLevel.UNKNOWN,
                )

    def test_malformed_hard_trigger_values_do_not_create_red(self):
        bad_values = (None, True, "3.0", float("nan"), float("inf"))
        for value in bad_values:
            with self.subTest(value=value):
                features = {"range_zscore_20d": value, "close_location": 0.10, "return_1d": -0.10}
                self.assertEqual(baseline_level(make_quant(), make_snapshot(features=features)), RiskLevel.GREEN)


class LLMAdjustmentTests(TestCase):
    def test_valid_adjustment_can_raise_exactly_one_level(self):
        self.assertEqual(
            apply_llm_adjustment(RiskLevel.YELLOW, make_context(recommended=RiskLevel.ORANGE), make_snapshot()),
            RiskLevel.ORANGE,
        )

    def test_invalid_adjustments_are_rejected(self):
        cases = (
            (RiskLevel.GREEN, make_context(confidence=0.6999), RiskLevel.GREEN),
            (RiskLevel.GREEN, make_context(supporting=("e1",)), RiskLevel.GREEN),
            (RiskLevel.GREEN, make_context(supporting=("e1", "invented")), RiskLevel.GREEN),
            (RiskLevel.GREEN, make_context(conflicting=("invented",)), RiskLevel.GREEN),
            (RiskLevel.ORANGE, make_context(recommended=RiskLevel.YELLOW), RiskLevel.ORANGE),
            (RiskLevel.RED, make_context(recommended=RiskLevel.GREEN), RiskLevel.RED),
            (RiskLevel.GREEN, make_context(recommended=RiskLevel.ORANGE), RiskLevel.GREEN),
            (RiskLevel.YELLOW, make_context(recommended=RiskLevel.ORANGE, status="unavailable"), RiskLevel.YELLOW),
            (RiskLevel.UNKNOWN, make_context(recommended=RiskLevel.RED), RiskLevel.UNKNOWN),
        )
        snapshot = make_snapshot()
        for baseline, context, expected in cases:
            with self.subTest(baseline=baseline, context=context):
                self.assertEqual(apply_llm_adjustment(baseline, context, snapshot), expected)

    def test_duplicate_supporting_ids_do_not_satisfy_two_evidence_requirement(self):
        context = make_context(supporting=("e1", "e1"))
        self.assertEqual(apply_llm_adjustment(RiskLevel.GREEN, context, make_snapshot()), RiskLevel.GREEN)


class StateMachineTests(TestCase):
    def test_upgrades_are_immediate_and_actionable_upgrades_push(self):
        yellow = transition(RiskLevel.GREEN, RiskLevel.YELLOW, 0)
        orange = transition(RiskLevel.YELLOW, RiskLevel.ORANGE, 0)
        red = transition(RiskLevel.ORANGE, RiskLevel.RED, 1)

        self.assertEqual(yellow.final_level, RiskLevel.YELLOW)
        self.assertFalse(yellow.push_required)
        self.assertEqual(orange.final_level, RiskLevel.ORANGE)
        self.assertTrue(orange.push_required)
        self.assertEqual(red.final_level, RiskLevel.RED)
        self.assertTrue(red.push_required)
        self.assertEqual(red.next_valid_snapshot_count, 0)

    def test_equal_level_does_not_push_and_resets_recovery_count(self):
        result = transition(RiskLevel.ORANGE, RiskLevel.ORANGE, 1)
        self.assertEqual(result.final_level, RiskLevel.ORANGE)
        self.assertEqual(result.state_transition, "UNCHANGED")
        self.assertFalse(result.push_required)
        self.assertEqual(result.next_valid_snapshot_count, 0)

    def test_orange_and_red_recovery_need_two_consecutive_valid_snapshots(self):
        for previous in (RiskLevel.ORANGE, RiskLevel.RED):
            with self.subTest(previous=previous):
                pending = transition(previous, RiskLevel.GREEN, 1)
                recovered = transition(previous, RiskLevel.GREEN, 2)
                self.assertEqual(pending.final_level, previous)
                self.assertEqual(pending.state_transition, "RECOVERY_PENDING")
                self.assertEqual(pending.next_valid_snapshot_count, 1)
                self.assertEqual(recovered.final_level, RiskLevel.GREEN)
                self.assertTrue(recovered.persist_required)
                self.assertFalse(recovered.push_required)
                self.assertEqual(recovered.next_valid_snapshot_count, 0)

    def test_unknown_never_counts_as_a_valid_recovery_snapshot(self):
        result = transition(RiskLevel.RED, RiskLevel.UNKNOWN, 2)
        self.assertEqual(result.final_level, RiskLevel.UNKNOWN)
        self.assertEqual(result.retained_risk_level, RiskLevel.RED)
        self.assertEqual(result.next_valid_snapshot_count, 0)
        self.assertFalse(result.push_required)

    def test_unknown_previous_can_preserve_retained_risk_explicitly(self):
        pending = transition(
            RiskLevel.UNKNOWN,
            RiskLevel.GREEN,
            1,
            retained_risk_level=RiskLevel.RED,
        )
        recovered = transition(
            RiskLevel.UNKNOWN,
            RiskLevel.GREEN,
            2,
            retained_risk_level=RiskLevel.RED,
        )
        self.assertEqual(pending.final_level, RiskLevel.RED)
        self.assertEqual(recovered.final_level, RiskLevel.GREEN)

    def test_recovery_counter_rejects_bool_nan_and_wrong_types(self):
        for value in (True, -1, 1.0, float("nan"), "1", None):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    transition(RiskLevel.RED, RiskLevel.GREEN, value)


class FinalDecisionTests(TestCase):
    def test_action_mapping_is_exact(self):
        cases = (
            (RiskLevel.GREEN, "OPEN", 100.0, "HOLD"),
            (RiskLevel.YELLOW, "OPEN", 100.0, "HOLD"),
            (RiskLevel.ORANGE, "CONDITIONAL", 3.0, "HOLD_OR_REDUCE"),
            (RiskLevel.RED, "WAIT", 0.0, "REDUCE"),
            (RiskLevel.UNKNOWN, "WAIT", 0.0, "HOLD"),
        )
        for level, gate, cap, holding in cases:
            with self.subTest(level=level):
                decision = build_final_decision(
                    baseline=level,
                    candidate=level,
                    snapshot=make_snapshot(
                        data_quality=DataStatus.INSUFFICIENT if level == RiskLevel.UNKNOWN else DataStatus.FRESH
                    ),
                    previous=level,
                    valid_snapshot_count=0,
                )
                self.assertEqual(decision.final_level, level)
                self.assertEqual(decision.entry_gate, gate)
                self.assertEqual(decision.new_position_cap_pct, cap)
                self.assertEqual(decision.holding_action, holding)

    def test_unknown_reason_explicitly_says_it_is_not_red(self):
        decision = build_final_decision(
            baseline=RiskLevel.UNKNOWN,
            candidate=RiskLevel.UNKNOWN,
            snapshot=make_snapshot(data_quality=DataStatus.STALE),
            previous=RiskLevel.GREEN,
            valid_snapshot_count=0,
        )
        self.assertTrue(any("not RED" in reason for reason in decision.decision_reasons))


if __name__ == "__main__":
    main()
