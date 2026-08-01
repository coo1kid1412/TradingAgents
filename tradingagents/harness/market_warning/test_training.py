"""Contract tests for leakage-safe market-warning model training."""

from __future__ import annotations

import inspect
import sys
from datetime import datetime, timezone
from pathlib import Path
from unittest import TestCase, main

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from tradingagents.harness.market_warning.domain import Market
from tradingagents.harness.market_warning.training import (
    EvaluationReport,
    build_labels,
    evaluate_model,
    fit_model,
    time_partitions,
)
import tradingagents.harness.market_warning.training as training_module


UTC = timezone.utc


def _dated_frame(start: str, rows: int, *, feature_version: str = "market-warning-v1") -> pd.DataFrame:
    dates = pd.bdate_range(start, periods=rows, tz="UTC")
    x = np.linspace(-2.5, 2.5, rows)
    labels = ((np.arange(rows) % 13) == 0) | (x > 2.1)
    frame = pd.DataFrame(
        {
            "as_of_time": dates,
            "signal": x,
            "secondary": np.cos(np.arange(rows) / 4),
            "occasionally_missing": np.where(np.arange(rows) % 7 == 0, np.nan, x / 2),
            "market_phase": np.where(np.arange(rows) % 4 == 0, "CONTINUATION", "FIRST_SHOCK"),
            "label_1d": labels,
            "label_3d": labels | ((np.arange(rows) % 17) == 0),
            "old_market_risk_alert": np.arange(rows) % 11 == 0,
        }
    )
    frame.attrs["feature_version"] = feature_version
    return frame


class LabelTests(TestCase):
    def test_a_share_thresholds_are_inclusive_at_exact_boundaries(self):
        frame = pd.DataFrame(
            {
                "as_of_time": pd.bdate_range("2026-01-05", periods=5, tz="Asia/Shanghai"),
                "close": [100.0, 96.0, 94.0001, 94.0, 100.0],
                "feature_available_at": pd.bdate_range("2026-01-05", periods=5, tz="Asia/Shanghai"),
                "known_signal": [1, 2, 3, 4, 5],
            }
        )

        result = build_labels(frame, Market.A_SHARE)

        self.assertTrue(bool(result.loc[0, "label_1d"]))
        self.assertAlmostEqual(result.loc[0, "future_return_1d"], -0.04)
        self.assertTrue(bool(result.loc[0, "label_3d"]))
        self.assertAlmostEqual(result.loc[0, "future_worst_return_3d"], -0.06)
        self.assertEqual(result.loc[0, "known_signal"], 1)
        self.assertTrue(pd.isna(result.loc[4, "label_1d"]))
        self.assertTrue(pd.isna(result.loc[4, "label_3d"]))

    def test_us_thresholds_do_not_round_near_boundary_values(self):
        frame = pd.DataFrame(
            {
                "as_of_time": pd.bdate_range("2026-02-02", periods=6, tz="America/New_York"),
                "close": [100.0, 97.0001, 95.0001, 95.0001, 92.150095, 101.0],
            }
        )

        result = build_labels(frame, Market.US)

        self.assertFalse(bool(result.loc[0, "label_1d"]))
        self.assertFalse(bool(result.loc[0, "label_3d"]))
        self.assertTrue(bool(result.loc[1, "label_3d"]))

    def test_labels_change_only_with_future_prices_not_feature_values(self):
        frame = pd.DataFrame(
            {
                "as_of_time": pd.bdate_range("2026-03-02", periods=5, tz="UTC"),
                "close": [100.0, 101.0, 102.0, 103.0, 104.0],
                "known_signal": [10, 20, 30, 40, 50],
            }
        )
        feature_mutation = frame.copy()
        feature_mutation.loc[1:, "known_signal"] = -999
        price_mutation = frame.copy()
        price_mutation.loc[1, "close"] = 90.0

        original = build_labels(frame, Market.A_SHARE)
        same_prices = build_labels(feature_mutation, Market.A_SHARE)
        changed_future = build_labels(price_mutation, Market.A_SHARE)

        pd.testing.assert_series_equal(original["label_1d"], same_prices["label_1d"])
        self.assertFalse(bool(original.loc[0, "label_1d"]))
        self.assertTrue(bool(changed_future.loc[0, "label_1d"]))
        self.assertEqual(changed_future.loc[0, "known_signal"], 10)

    def test_future_available_feature_is_rejected_before_labeling(self):
        times = pd.bdate_range("2026-04-01", periods=4, tz="UTC")
        frame = pd.DataFrame(
            {
                "as_of_time": times,
                "close": [100.0, 99.0, 98.0, 97.0],
                "feature_available_at": [times[0], times[1], times[2] + pd.Timedelta(minutes=1), times[3]],
            }
        )

        with self.assertRaisesRegex(ValueError, "point-in-time"):
            build_labels(frame, Market.A_SHARE)

    def test_named_feature_availability_columns_are_audited_individually(self):
        times = pd.bdate_range("2026-04-01", periods=4, tz="UTC")
        frame = pd.DataFrame(
            {
                "as_of_time": times,
                "close": [100.0, 99.0, 98.0, 97.0],
                "breadth": [60.0, 55.0, 50.0, 45.0],
                "breadth_available_at": [times[0], times[1] + pd.Timedelta(seconds=1), times[2], times[3]],
            }
        )

        with self.assertRaisesRegex(ValueError, "breadth_available_at"):
            build_labels(frame, Market.A_SHARE)


class PartitionTests(TestCase):
    def test_partitions_use_frozen_dates_and_purge_three_rows_before_each_boundary(self):
        dates = pd.DatetimeIndex(
            list(pd.bdate_range("2012-12-24", "2013-01-08", tz="UTC"))
            + list(pd.bdate_range("2019-12-20", "2020-01-08", tz="UTC"))
            + list(pd.bdate_range("2026-07-29", "2026-08-04", tz="UTC"))
        )
        frame = pd.DataFrame({"as_of_time": dates, "value": np.arange(len(dates))})

        dev, validation, test = time_partitions(frame)

        self.assertEqual(dev["as_of_time"].max(), pd.Timestamp("2012-12-26", tz="UTC"))
        self.assertEqual(validation["as_of_time"].min(), pd.Timestamp("2013-01-01", tz="UTC"))
        self.assertEqual(validation["as_of_time"].max(), pd.Timestamp("2019-12-26", tz="UTC"))
        self.assertEqual(test["as_of_time"].min(), pd.Timestamp("2020-01-01", tz="UTC"))
        self.assertEqual(test["as_of_time"].max(), pd.Timestamp("2026-07-31", tz="UTC"))
        self.assertTrue((dev["partition"] == "dev").all())
        self.assertTrue((validation["partition"] == "validation").all())
        self.assertTrue((test["partition"] == "test").all())

    def test_explicit_label_end_cannot_cross_partition_boundary(self):
        frame = pd.DataFrame(
            {
                "as_of_time": pd.to_datetime(["2012-12-27", "2012-12-28", "2013-01-02"], utc=True),
                "label_end_3d": pd.to_datetime(["2013-01-03", "2013-01-04", "2013-01-07"], utc=True),
            }
        )

        dev, validation, _ = time_partitions(frame)

        self.assertTrue(dev.empty)
        self.assertEqual(len(validation), 1)

    def test_partition_outputs_are_chronological_even_if_input_rows_are_not(self):
        dates = pd.to_datetime(
            ["2013-01-08", "2013-01-02", "2013-01-04", "2013-01-01", "2013-01-09", "2013-01-03", "2013-01-07"],
            utc=True,
        )
        frame = pd.DataFrame({"as_of_time": dates, "value": [6, 2, 4, 1, 7, 3, 5]})

        _, validation, _ = time_partitions(frame)

        self.assertTrue(validation["as_of_time"].is_monotonic_increasing)
        self.assertEqual(validation["value"].tolist(), [1, 2, 3, 4])

    def test_training_module_has_no_random_split_or_synthetic_oversampling(self):
        source = inspect.getsource(training_module).lower().replace(" ", "")

        self.assertNotIn("train_test_split", source)
        self.assertNotIn("shuffle=true", source)
        self.assertNotIn("smote", source)


class ModelTrainingTests(TestCase):
    def test_fit_model_uses_required_pipeline_and_later_platt_calibration(self):
        train = _dated_frame("2008-01-01", 120)
        calibration = _dated_frame("2013-01-01", 50)
        train["signal_available_at"] = train["as_of_time"]
        calibration["signal_available_at"] = calibration["as_of_time"]

        bundle = fit_model(train, calibration, Market.A_SHARE, "1d")
        probabilities = bundle.predict_proba(calibration)

        self.assertIsInstance(bundle.pipeline, Pipeline)
        self.assertIsInstance(bundle.pipeline.named_steps["imputer"], SimpleImputer)
        self.assertTrue(bundle.pipeline.named_steps["imputer"].add_indicator)
        self.assertIsInstance(bundle.pipeline.named_steps["scaler"], StandardScaler)
        self.assertIsInstance(bundle.pipeline.named_steps["classifier"], LogisticRegression)
        self.assertEqual(bundle.calibration_method, "platt")
        self.assertNotIn("signal_available_at", bundle.feature_names)
        self.assertLess(bundle.training_end, bundle.calibration_start)
        self.assertTrue(np.all((probabilities >= 0) & (probabilities <= 1)))

    def test_fit_model_rejects_overlapping_or_reverse_calibration_window(self):
        train = _dated_frame("2013-01-01", 40)
        calibration = _dated_frame("2013-02-01", 40)

        with self.assertRaisesRegex(ValueError, "later"):
            fit_model(train, calibration, Market.US, "3d")

    def test_fit_model_rejects_calibration_without_three_session_embargo(self):
        train = _dated_frame("2012-10-01", 65)
        calibration = _dated_frame("2013-01-02", 40)

        with self.assertRaisesRegex(ValueError, "embargo"):
            fit_model(train, calibration, Market.A_SHARE, "1d")

    def test_four_market_horizon_bundles_are_independent(self):
        train = _dated_frame("2005-01-03", 100)
        calibration = _dated_frame("2014-01-02", 45)

        bundles = {
            (market, horizon): fit_model(train, calibration, market, horizon)
            for market in (Market.A_SHARE, Market.US)
            for horizon in ("1d", "3d")
        }

        self.assertEqual(len({id(bundle.pipeline) for bundle in bundles.values()}), 4)
        for (market, horizon), bundle in bundles.items():
            self.assertEqual(bundle.market, market)
            self.assertEqual(bundle.horizon, horizon)

    def test_evaluation_reports_calibration_phase_crisis_budget_and_baselines(self):
        train = _dated_frame("2007-01-01", 120)
        calibration = _dated_frame("2014-01-01", 60)
        test = _dated_frame("2020-01-02", 80)
        test.loc[test.index[:20], "crisis_period"] = "2020"
        bundle = fit_model(train, calibration, Market.US, "1d")

        report = evaluate_model(bundle, test)
        expected_probabilities = bundle.predict_proba(test)
        expected_labels = test["label_1d"].astype(int).to_numpy()

        self.assertIsInstance(report, EvaluationReport)
        self.assertAlmostEqual(report.brier_score, brier_score_loss(expected_labels, expected_probabilities))
        self.assertAlmostEqual(report.average_precision, average_precision_score(expected_labels, expected_probabilities))
        self.assertAlmostEqual(report.prevalence, float(expected_labels.mean()))
        self.assertEqual(len(report.calibration_bins), 10)
        self.assertIn("FIRST_SHOCK", report.phase_breakdown)
        self.assertIn("CONTINUATION", report.phase_breakdown)
        self.assertIn("2020", report.crisis_contribution)
        self.assertGreaterEqual(report.monthly_alert_entries, 0)
        self.assertIsNotNone(report.old_market_risk_recall)
        self.assertIsNotNone(report.model_recall_at_old_budget)
        self.assertGreaterEqual(report.constant_base_rate_brier, 0)
        self.assertGreaterEqual(report.expected_calibration_error, 0)


if __name__ == "__main__":
    main()
