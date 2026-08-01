"""Contract tests for leakage-safe market-warning model training."""

from __future__ import annotations

import inspect
import hashlib
import json
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from unittest import TestCase, main

import exchange_calendars
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from tradingagents.harness.market_warning.domain import Market
from tradingagents.harness.market_warning.adapters.sqlite_repository import SQLiteWarningRepository
from tradingagents.harness.market_warning.training import (
    EvaluationReport,
    build_labels,
    evaluate_model,
    fit_model,
    production_partitions,
    time_partitions,
)
import tradingagents.harness.market_warning.training as training_module


UTC = timezone.utc
PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _with_label_availability(frame: pd.DataFrame) -> pd.DataFrame:
    values = frame.copy()
    values["feature_available_at"] = values["as_of_time"]
    values.attrs["availability_proof"] = {"*": "feature_available_at"}
    return values


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
            "label_end_3d": dates + pd.offsets.BDay(3),
            "old_market_risk_alert": np.arange(rows) % 11 == 0,
            "feature_available_at": dates,
            "label_source_available_at": dates,
        }
    )
    frame.attrs["feature_version"] = feature_version
    frame.attrs["point_in_time_validated"] = True
    frame.attrs["availability_proof"] = {
        "*": "feature_available_at",
        "label_source": "label_source_available_at",
    }
    return frame


class LabelTests(TestCase):
    def test_mixed_new_york_dst_offsets_remain_timezone_aware_and_label_safely(self):
        times = pd.Series(
            [
                datetime.fromisoformat("2025-01-02T16:00:00-05:00"),
                datetime.fromisoformat("2025-03-10T16:00:00-04:00"),
                datetime.fromisoformat("2025-07-01T16:00:00-04:00"),
                datetime.fromisoformat("2025-11-03T16:00:00-05:00"),
            ]
        )
        frame = pd.DataFrame(
            {
                "as_of_time": times,
                "close": [100.0, 99.0, 98.0, 97.0],
                "feature_available_at": times,
            }
        )
        frame.attrs["availability_proof"] = {"*": "feature_available_at"}

        result = build_labels(frame, Market.US)

        self.assertEqual(len(result), 4)
        self.assertTrue(result.attrs["point_in_time_validated"])

    def test_a_share_thresholds_are_inclusive_at_exact_boundaries(self):
        frame = _with_label_availability(pd.DataFrame(
            {
                "as_of_time": pd.bdate_range("2026-01-05", periods=5, tz="Asia/Shanghai"),
                "close": [100.0, 96.0, 94.0001, 94.0, 100.0],
                "known_signal": [1, 2, 3, 4, 5],
            }
        ))

        result = build_labels(frame, Market.A_SHARE)

        self.assertTrue(bool(result.loc[0, "label_1d"]))
        self.assertAlmostEqual(result.loc[0, "future_return_1d"], -0.04)
        self.assertTrue(bool(result.loc[0, "label_3d"]))
        self.assertAlmostEqual(result.loc[0, "future_worst_return_3d"], -0.06)
        self.assertEqual(result.loc[0, "known_signal"], 1)
        self.assertTrue(pd.isna(result.loc[4, "label_1d"]))
        self.assertTrue(pd.isna(result.loc[4, "label_3d"]))

    def test_us_thresholds_do_not_round_near_boundary_values(self):
        frame = _with_label_availability(pd.DataFrame(
            {
                "as_of_time": pd.bdate_range("2026-02-02", periods=6, tz="America/New_York"),
                "close": [100.0, 97.0001, 95.0001, 95.0001, 92.150095, 101.0],
            }
        ))

        result = build_labels(frame, Market.US)

        self.assertFalse(bool(result.loc[0, "label_1d"]))
        self.assertFalse(bool(result.loc[0, "label_3d"]))
        self.assertTrue(bool(result.loc[1, "label_3d"]))

    def test_labels_change_only_with_future_prices_not_feature_values(self):
        frame = _with_label_availability(pd.DataFrame(
            {
                "as_of_time": pd.bdate_range("2026-03-02", periods=5, tz="UTC"),
                "close": [100.0, 101.0, 102.0, 103.0, 104.0],
                "known_signal": [10, 20, 30, 40, 50],
            }
        ))
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
        frame.attrs["availability_proof"] = {"*": "feature_available_at"}

        with self.assertRaisesRegex(ValueError, "point-in-time"):
            build_labels(frame, Market.A_SHARE)

    def test_named_feature_availability_columns_are_audited_individually(self):
        times = pd.bdate_range("2026-04-01", periods=4, tz="UTC")
        frame = pd.DataFrame(
            {
                "as_of_time": times,
                "close": [100.0, 99.0, 98.0, 97.0],
                "close_available_at": times,
                "breadth": [60.0, 55.0, 50.0, 45.0],
                "breadth_available_at": [times[0], times[1] + pd.Timedelta(seconds=1), times[2], times[3]],
            }
        )
        frame.attrs["availability_proof"] = {
            "close": "close_available_at",
            "breadth": "breadth_available_at",
        }

        with self.assertRaisesRegex(ValueError, "breadth_available_at"):
            build_labels(frame, Market.A_SHARE)

    def test_labeling_without_documented_availability_proof_fails_closed(self):
        frame = pd.DataFrame(
            {
                "as_of_time": pd.bdate_range("2026-05-04", periods=4, tz="UTC"),
                "close": [100.0, 99.0, 98.0, 97.0],
                "signal": [1.0, 2.0, 3.0, 4.0],
            }
        )

        with self.assertRaisesRegex(ValueError, "availability proof"):
            build_labels(frame, Market.A_SHARE)

    def test_value_strictly_above_threshold_is_not_rounded_into_positive_label(self):
        frame = _with_label_availability(
            pd.DataFrame(
                {
                    "as_of_time": pd.bdate_range("2026-06-01", periods=5, tz="UTC"),
                    "close": [100.0, 96.00000000005, 100.0, 100.0, 100.0],
                }
            )
        )

        result = build_labels(frame, Market.A_SHARE)

        self.assertFalse(bool(result.loc[0, "label_1d"]))


class PartitionTests(TestCase):
    def test_partitions_use_frozen_dates_and_purge_three_rows_before_each_boundary(self):
        dates = pd.DatetimeIndex(
            list(pd.bdate_range("2012-12-24", "2013-01-08", tz="UTC"))
            + list(pd.bdate_range("2019-12-20", "2020-01-08", tz="UTC"))
            + list(pd.bdate_range("2026-07-24", "2026-08-05", tz="UTC"))
        )
        frame = pd.DataFrame(
            {
                "as_of_time": dates,
                "label_end_3d": dates + pd.offsets.BDay(3),
                "value": np.arange(len(dates)),
            }
        )

        dev, validation, test = time_partitions(frame)

        self.assertEqual(dev["as_of_time"].max(), pd.Timestamp("2012-12-26", tz="UTC"))
        self.assertEqual(validation["as_of_time"].min(), pd.Timestamp("2013-01-01", tz="UTC"))
        self.assertEqual(validation["as_of_time"].max(), pd.Timestamp("2019-12-26", tz="UTC"))
        self.assertEqual(test["as_of_time"].min(), pd.Timestamp("2020-01-01", tz="UTC"))
        self.assertEqual(test["as_of_time"].max(), pd.Timestamp("2026-07-28", tz="UTC"))
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

    def test_production_partitions_refit_through_latest_labeled_window(self):
        frame = pd.concat(
            [
                _dated_frame("2008-01-01", 120),
                _dated_frame("2014-01-01", 60),
                _dated_frame("2025-01-02", 300),
            ],
            ignore_index=True,
        )
        frame.attrs.update(_dated_frame("2008-01-01", 4).attrs)

        train, calibration = production_partitions(frame, Market.A_SHARE)

        self.assertEqual(len(calibration), 96)
        self.assertEqual(
            pd.to_datetime(calibration["as_of_time"], utc=True).max(),
            pd.to_datetime(frame["as_of_time"], utc=True).max(),
        )
        self.assertGreaterEqual(
            training_module._embargo_sessions(
                Market.A_SHARE,
                pd.to_datetime(train["as_of_time"], utc=True).max(),
                pd.to_datetime(calibration["as_of_time"], utc=True).min(),
            ),
            3,
        )
        self.assertLess(
            pd.to_datetime(train["label_end_3d"], utc=True).max(),
            pd.to_datetime(calibration["as_of_time"], utc=True).min(),
        )

    def test_frozen_test_excludes_null_or_august_three_day_label_windows(self):
        frame = pd.DataFrame(
            {
                "as_of_time": pd.to_datetime(
                    ["2026-07-27", "2026-07-28", "2026-07-29", "2026-07-30", "2026-07-31"], utc=True
                ),
                "label_end_3d": pd.to_datetime(
                    ["2026-07-30", "2026-07-31", "2026-08-03", None, "2026-08-05"], utc=True
                ),
            }
        )

        _, _, test = time_partitions(frame)

        self.assertEqual(
            test["as_of_time"].tolist(),
            [pd.Timestamp("2026-07-27", tz="UTC"), pd.Timestamp("2026-07-28", tz="UTC")],
        )

    def test_partitions_require_explicit_three_day_label_end_proof(self):
        frame = pd.DataFrame(
            {"as_of_time": pd.to_datetime(["2012-12-20", "2013-01-02", "2020-01-02"], utc=True)}
        )

        with self.assertRaisesRegex(ValueError, "label_end_3d"):
            time_partitions(frame)

    def test_partition_outputs_are_chronological_even_if_input_rows_are_not(self):
        dates = pd.to_datetime(
            ["2013-01-08", "2013-01-02", "2013-01-04", "2013-01-01", "2013-01-09", "2013-01-03", "2013-01-07"],
            utc=True,
        )
        frame = pd.DataFrame(
            {
                "as_of_time": dates,
                "label_end_3d": dates + pd.offsets.BDay(3),
                "value": [6, 2, 4, 1, 7, 3, 5],
            }
        )

        _, validation, _ = time_partitions(frame)

        self.assertTrue(validation["as_of_time"].is_monotonic_increasing)
        self.assertEqual(validation["value"].tolist(), [1, 2, 3, 4, 5, 6, 7])

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

    def test_fit_model_requires_explicit_point_in_time_validation_marker(self):
        train = _dated_frame("2008-01-01", 100)
        calibration = _dated_frame("2014-01-01", 45)
        train.attrs.pop("point_in_time_validated")

        with self.assertRaisesRegex(ValueError, "point-in-time validated"):
            fit_model(train, calibration, Market.A_SHARE, "1d")

    def test_fit_model_requires_proof_for_every_feature_and_label_source(self):
        train = _dated_frame("2008-01-01", 100)
        calibration = _dated_frame("2014-01-01", 45)
        train.attrs["availability_proof"] = {"signal": "feature_available_at"}

        with self.assertRaisesRegex(ValueError, "availability proof"):
            fit_model(train, calibration, Market.A_SHARE, "1d")

    def test_fit_model_rejects_future_feature_or_label_source_availability(self):
        train = _dated_frame("2008-01-01", 100)
        calibration = _dated_frame("2014-01-01", 45)
        train.loc[3, "feature_available_at"] = train.loc[3, "as_of_time"] + pd.Timedelta(seconds=1)

        with self.assertRaisesRegex(ValueError, "feature_available_at"):
            fit_model(train, calibration, Market.A_SHARE, "1d")

        train = _dated_frame("2008-01-01", 100)
        train.loc[4, "label_source_available_at"] = train.loc[4, "as_of_time"] + pd.Timedelta(seconds=1)
        with self.assertRaisesRegex(ValueError, "label_source_available_at"):
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
        self.assertAlmostEqual(
            report.constant_base_rate_brier,
            brier_score_loss(expected_labels, np.full(len(expected_labels), bundle.base_rate)),
        )
        self.assertEqual(len(report.calibration_bins), 10)
        self.assertIn("FIRST_SHOCK", report.phase_breakdown)
        self.assertIn("CONTINUATION", report.phase_breakdown)
        self.assertIn("2020", report.crisis_contribution)
        self.assertGreaterEqual(report.monthly_alert_entries, 0)
        self.assertIsNotNone(report.old_market_risk_recall)
        self.assertIsNotNone(report.model_recall_at_old_budget)
        self.assertGreaterEqual(report.constant_base_rate_brier, 0)
        self.assertGreaterEqual(report.expected_calibration_error, 0)

    def test_evaluate_model_requires_explicit_point_in_time_validation_marker(self):
        train = _dated_frame("2007-01-01", 120)
        calibration = _dated_frame("2014-01-01", 60)
        test = _dated_frame("2020-01-02", 80)
        bundle = fit_model(train, calibration, Market.US, "1d")
        test.attrs.clear()

        with self.assertRaisesRegex(ValueError, "point-in-time validated"):
            evaluate_model(bundle, test)

    def test_evaluate_model_rejects_future_feature_availability(self):
        train = _dated_frame("2007-01-01", 120)
        calibration = _dated_frame("2014-01-01", 60)
        test = _dated_frame("2020-01-02", 80)
        bundle = fit_model(train, calibration, Market.US, "1d")
        test.loc[3, "feature_available_at"] = test.loc[3, "as_of_time"] + pd.Timedelta(seconds=1)

        with self.assertRaisesRegex(ValueError, "feature_available_at"):
            evaluate_model(bundle, test)


class TrainingCommandTests(TestCase):
    def _run(self, *arguments):
        return subprocess.run(
            [sys.executable, "-m", "tradingagents.harness.market_warning.training", *arguments],
            cwd=PROJECT_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )

    def test_command_help_exposes_all_task6_subcommands(self):
        result = self._run("--help")

        self.assertEqual(result.returncode, 0, result.stderr)
        for command in ("backfill", "train", "evaluate", "promote"):
            self.assertIn(command, result.stdout)

    def test_task12_flags_parse_and_missing_inputs_fail_actionably(self):
        result = self._run(
            "train",
            "--start", "2000-01-01",
            "--test-end", "2026-07-31",
            "--version", "market-warning-v1",
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("--dataset", result.stderr)
        self.assertIn("Task 12", result.stderr)

    def test_backfill_requires_an_explicit_market_and_output(self):
        result = self._run(
            "backfill",
            "--start", "2026-07-01",
            "--test-end", "2026-07-31",
            "--version", "market-warning-v1",
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("--market", result.stderr)
        self.assertIn("--output", result.stderr)

    def test_promote_does_not_fake_success_before_task12_evaluation_exists(self):
        result = self._run(
            "promote",
            "--start", "2000-01-01",
            "--test-end", "2026-07-31",
            "--version", "market-warning-v1",
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Task 12", result.stderr)

    def test_cli_rejects_unsafe_model_versions_before_dispatch(self):
        for version in (".", "..", "../escape", "nested/version", r"nested\version", "/absolute"):
            with self.subTest(version=version):
                result = self._run(
                    "train",
                    "--start", "2000-01-01",
                    "--test-end", "2026-07-31",
                    "--version", version,
                )

                self.assertNotEqual(result.returncode, 0)
                self.assertIn("--version", result.stderr)

    def test_train_and_evaluate_execute_with_validated_local_dataset(self):
        with tempfile.TemporaryDirectory(prefix="warning_cli_") as directory:
            root = Path(directory)
            dataset = pd.concat(
                [
                    _dated_frame("2008-01-01", 120),
                    _dated_frame("2014-01-01", 60),
                    _dated_frame("2020-01-02", 80),
                ],
                ignore_index=True,
            )
            dataset.attrs.update(_dated_frame("2008-01-01", 4).attrs)
            dataset_path = root / "dataset.joblib"
            joblib.dump(dataset, dataset_path)

            trained = self._run(
                "train",
                "--dataset", str(dataset_path),
                "--market", "us",
                "--horizon", "1d",
                "--artifact-root", str(root / "models"),
                "--db", str(root / "warning.db"),
                "--start", "2000-01-01",
                "--test-end", "2026-07-31",
                "--version", "cli-fixture-v1",
            )
            self.assertEqual(trained.returncode, 0, trained.stderr)
            payload = json.loads(trained.stdout)
            artifact = Path(payload["artifact_path"])
            self.assertTrue(artifact.is_file())

            evaluated = self._run(
                "evaluate",
                "--dataset", str(dataset_path),
                "--artifact", str(artifact),
                "--start", "2000-01-01",
                "--test-end", "2026-07-31",
                "--version", "cli-fixture-v1",
            )
            self.assertEqual(evaluated.returncode, 0, evaluated.stderr)
            self.assertEqual(json.loads(evaluated.stdout)["horizon"], "1d")

    def test_train_without_single_model_flags_builds_four_model_manifest_and_report(self):
        with tempfile.TemporaryDirectory(prefix="warning_train_all_") as directory:
            root = Path(directory)
            dataset = pd.concat(
                [
                    _dated_frame("2008-01-01", 120),
                    _dated_frame("2014-01-01", 60),
                    _dated_frame("2020-01-02", 80),
                ],
                ignore_index=True,
            )
            dataset.attrs.update(_dated_frame("2008-01-01", 4).attrs)
            a_share_path = root / "a-share.joblib"
            us_path = root / "us.joblib"
            joblib.dump(dataset, a_share_path)
            joblib.dump(dataset, us_path)
            report_path = root / "evaluation.md"
            manifest_path = root / "manifest.json"

            trained = self._run(
                "train",
                "--dataset-a-share", str(a_share_path),
                "--dataset-us", str(us_path),
                "--artifact-root", str(root / "models"),
                "--db", str(root / "warning.db"),
                "--report", str(report_path),
                "--manifest", str(manifest_path),
                "--start", "2000-01-01",
                "--test-end", "2026-07-31",
                "--version", "four-model-v1",
            )

            self.assertEqual(trained.returncode, 0, trained.stderr)
            payload = json.loads(trained.stdout)
            self.assertEqual(len(payload["models"]), 4)
            self.assertTrue(all(not model["active"] for model in payload["models"]))
            self.assertTrue(report_path.is_file())
            self.assertTrue(manifest_path.is_file())
            report = report_path.read_text(encoding="utf-8")
            self.assertIn("Brier", report)
            self.assertIn("AUPRC", report)
            self.assertIn("FIRST_SHOCK", report)
            self.assertIn("危机贡献", report)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["model_version"], "four-model-v1")
            self.assertEqual(len(manifest["models"]), 4)
            self.assertIn("calibration_bins", manifest["models"][0]["evaluation"])
            for model in manifest["models"]:
                self.assertGreaterEqual(model["training_cutoff"], "2020-01-01")
                self.assertIn("production_partitions", model)
                self.assertLess(
                    model["evaluation_bundle"]["calibration_end"],
                    model["training_cutoff"],
                )
                bundle = joblib.load(model["artifact_path"])
                self.assertEqual(
                    bundle.calibration_end.isoformat(),
                    model["production_partitions"]["calibration"]["end"],
                )

    def test_promote_verifies_manifest_and_atomically_activates_four_registered_models(self):
        with tempfile.TemporaryDirectory(prefix="warning_promote_") as directory:
            root = Path(directory)
            artifact_root = root / "models"
            version = "promotion-v1"
            version_root = artifact_root / version
            version_root.mkdir(parents=True)
            db_path = root / "warning.db"
            repository = SQLiteWarningRepository(db_path)
            models = []
            for market in Market:
                for horizon in ("1d", "3d"):
                    artifact = version_root / f"{market.value}-{horizon}.joblib"
                    artifact.write_bytes(f"{market.value}/{horizon}".encode())
                    checksum = hashlib.sha256(artifact.read_bytes()).hexdigest()
                    record = {
                        "model_version": version,
                        "market": market,
                        "horizon": horizon,
                        "feature_version": "market-warning-v1",
                        "calibration_version": "platt-v1",
                        "training_cutoff": "2019-12-31",
                        "artifact_path": str(artifact),
                        "artifact_sha256": checksum,
                        "metrics": {},
                        "base_rate": 0.02,
                        "active": False,
                    }
                    repository.register_model(record)
                    models.append(
                        {
                            **{key: value.value if isinstance(value, Market) else value for key, value in record.items()},
                            "evaluation": {
                                "market": market.value,
                                "horizon": horizon,
                                "prevalence": 0.02,
                                "brier_score": 0.015,
                                "constant_base_rate_brier": 0.019,
                                "average_precision": 0.08,
                                "expected_calibration_error": 0.03,
                                "monthly_alert_entries": 4.0,
                                "crisis_contribution": {"2008": 0.3, "2020": 0.3, "non_crisis": 0.4},
                            },
                        }
                    )
            manifest_path = root / "manifest.json"
            manifest_path.write_text(
                json.dumps({"model_version": version, "models": models}),
                encoding="utf-8",
            )

            promoted = self._run(
                "promote",
                "--artifact-root", str(artifact_root),
                "--db", str(db_path),
                "--manifest", str(manifest_path),
                "--start", "2000-01-01",
                "--test-end", "2026-07-31",
                "--version", version,
            )

            self.assertEqual(promoted.returncode, 0, promoted.stderr)
            self.assertEqual(json.loads(promoted.stdout)["activated_models"], 4)
            for market in Market:
                for horizon in ("1d", "3d"):
                    active = SQLiteWarningRepository(db_path).load_active_model(
                        market, horizon
                    )
                    self.assertEqual(active["model_version"], version)

    def test_task6_dependencies_import_with_supported_versions(self):
        sklearn_version = tuple(int(part) for part in sklearn.__version__.split(".")[:2])
        calendars_version = tuple(int(part) for part in exchange_calendars.__version__.split(".")[:2])

        self.assertGreaterEqual(sklearn_version, (1, 7))
        self.assertLess(sklearn_version, (2, 0))
        self.assertGreaterEqual(calendars_version, (4, 11))
        self.assertLess(calendars_version, (5, 0))


if __name__ == "__main__":
    main()
