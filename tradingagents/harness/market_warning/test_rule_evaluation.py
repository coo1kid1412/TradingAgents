"""Leakage, entry-budget, and phase tests for rule history evaluation."""

from __future__ import annotations

from pathlib import Path
from unittest import TestCase, main

import pandas as pd

from tradingagents.harness.market_warning.rule_evaluation import evaluate_rule_frame
from tradingagents.harness.market_warning.rule_policy import load_rule_manifest


MANIFEST_PATH = Path(__file__).with_name("rule_manifest_v1.json")


def make_frame(times: list[str]) -> pd.DataFrame:
    as_of = pd.to_datetime(times, utc=True)
    rows = len(times)
    frame = pd.DataFrame(
        {
            "as_of_time": as_of,
            "feature_available_at": as_of,
            "return_1d": [0.01] * rows,
            "audited_ohlc_return_1d": [0.01] * rows,
            "return_5d": [0.02] * rows,
            "return_60d": [0.05] * rows,
            "drawdown_20d": [-0.01] * rows,
            "ma20_distance": [0.01] * rows,
            "volatility_ratio_5d_20d": [1.0] * rows,
            "range_zscore_20d": [0.0] * rows,
            "close_location": [0.5] * rows,
            "abnormal_range_weak_close_transition": [False] * rows,
            "breadth_deterioration_transition": [False] * rows,
            "breadth_up_pct": [55.0] * rows,
            "breadth_above_ma20_pct": [60.0] * rows,
            "new_low_20d_pct": [5.0] * rows,
            "industry_decline_pct": [40.0] * rows,
            "margin_balance_growth_20d": [0.02] * rows,
            "margin_balance_contracting_from_high": [False] * rows,
            "turnover_percentile_20d": [0.5] * rows,
            "limit_down_pct": [0.1] * rows,
            "market_phase": ["FIRST_SHOCK"] * rows,
            "label_1d": [False] * rows,
            "label_3d": [False] * rows,
            "crisis_period": [None] * rows,
            "old_market_risk_alert": [False] * rows,
        }
    )
    frame.attrs["availability_proof"] = {"*": "feature_available_at"}
    return frame


class RuleEvaluationTests(TestCase):
    def setUp(self):
        self.manifest = load_rule_manifest(MANIFEST_PATH)

    def test_frozen_calendar_partitions_are_reported_separately(self):
        frame = make_frame(["2012-06-01", "2015-06-01", "2021-06-01"])

        result = evaluate_rule_frame(frame, self.manifest)

        self.assertEqual(result["partitions"]["dev"]["observations"], 1)
        self.assertEqual(result["partitions"]["validation"]["observations"], 1)
        self.assertEqual(result["partitions"]["test"]["observations"], 1)
        self.assertTrue(result["previously_observed_holdout"])

    def test_future_feature_availability_is_rejected_before_evaluation(self):
        frame = make_frame(["2021-06-01", "2021-06-02"])
        frame.loc[1, "feature_available_at"] = (
            frame.loc[1, "as_of_time"] + pd.Timedelta(seconds=1)
        )

        with self.assertRaisesRegex(ValueError, "feature_available_at"):
            evaluate_rule_frame(frame, self.manifest)

    def test_only_new_orange_red_entries_consume_alert_budget(self):
        frame = make_frame(
            [
                "2021-01-04",
                "2021-01-05",
                "2021-01-06",
                "2021-01-07",
                "2021-01-08",
                "2021-01-11",
            ]
        )
        frame.loc[1:2, "abnormal_range_weak_close_transition"] = True
        frame.loc[4, ["range_zscore_20d", "close_location", "audited_ohlc_return_1d"]] = [
            3.0,
            0.15,
            -0.02,
        ]
        frame.loc[1:2, ["label_1d", "label_3d"]] = True
        frame.loc[1, "crisis_period"] = "crisis-a"
        frame.loc[4, "crisis_period"] = "crisis-b"
        frame.loc[1, "old_market_risk_alert"] = True

        result = evaluate_rule_frame(frame, self.manifest)
        test = result["partitions"]["test"]

        self.assertEqual(test["alert_entries"], 2)
        self.assertEqual(test["true_positives"], 1)
        self.assertAlmostEqual(test["precision"], 0.5)
        self.assertEqual(test["crisis_contribution"], {"crisis-a": 1.0})
        self.assertEqual(test["max_crisis_contribution"], 1.0)

    def test_orange_to_red_upgrade_is_counted_as_an_actual_push_event(self):
        frame = make_frame(["2021-01-04", "2021-01-05", "2021-01-06"])
        frame.loc[1, "abnormal_range_weak_close_transition"] = True
        frame.loc[2, ["range_zscore_20d", "close_location", "audited_ohlc_return_1d"]] = [
            3.0,
            0.15,
            -0.02,
        ]

        test = evaluate_rule_frame(frame, self.manifest)["partitions"]["test"]

        self.assertEqual(test["alert_entries"], 1)
        self.assertEqual(test["push_events"], 2)
        self.assertGreater(test["push_events_per_month"], test["alerts_per_month"])

    def test_first_shock_and_continuation_metrics_are_not_blended(self):
        frame = make_frame(["2021-02-01", "2021-02-02", "2021-02-03", "2021-02-04"])
        frame.loc[:, "abnormal_range_weak_close_transition"] = [False, True, False, True]
        frame.loc[:, "market_phase"] = [
            "FIRST_SHOCK",
            "FIRST_SHOCK",
            "CONTINUATION",
            "CONTINUATION",
        ]
        frame.loc[[1, 3], "label_3d"] = True

        test = evaluate_rule_frame(frame, self.manifest)["partitions"]["test"]

        self.assertEqual(test["phase_breakdown"]["FIRST_SHOCK"]["alert_entries"], 1)
        self.assertEqual(test["phase_breakdown"]["CONTINUATION"]["alert_entries"], 1)
        self.assertEqual(test["phase_breakdown"]["FIRST_SHOCK"]["true_positives"], 1)
        self.assertEqual(test["phase_breakdown"]["CONTINUATION"]["true_positives"], 1)

    def test_production_gates_come_only_from_frozen_test_partition(self):
        frame = make_frame(["2015-01-05", "2015-01-06", "2021-01-05", "2021-01-06"])
        frame.loc[[1, 3], "abnormal_range_weak_close_transition"] = True
        frame.loc[[1, 3], "label_3d"] = True

        result = evaluate_rule_frame(frame, self.manifest)

        self.assertEqual(
            result["production_gates"]["lift"],
            result["partitions"]["test"]["lift"],
        )
        self.assertEqual(
            result["production_gates"]["alerts_per_month"],
            result["partitions"]["test"]["alerts_per_month"],
        )
        self.assertEqual(
            result["production_gates"]["push_events_per_month"],
            result["partitions"]["test"]["push_events_per_month"],
        )


if __name__ == "__main__":
    main()
