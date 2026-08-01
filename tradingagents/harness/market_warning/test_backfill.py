"""Point-in-time historical dataset and promotion-gate tests."""

from __future__ import annotations

import tempfile
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import TestCase, main

import joblib

from tradingagents.harness.market_warning.backfill import (
    build_point_in_time_dataset,
    promotion_failures,
    run_backfill,
    write_dataset,
)
from tradingagents.harness.market_warning.domain import (
    DataStatus,
    FeatureSnapshot,
    Market,
    MarketDataPoint,
    RawMarketSnapshot,
)
from tradingagents.harness.market_warning.features import FEATURE_VERSION


UTC = timezone.utc


class _FeatureStrategy:
    def build(self, raw, prior_history):
        prior = tuple(prior_history)
        close = next(point.value for point in raw.points if point.field == "index_price")
        return FeatureSnapshot(
            market=raw.market,
            as_of_time=raw.as_of_time,
            session_slot=raw.session_slot,
            feature_version=FEATURE_VERSION,
            features={
                "signal": float(close) / 100.0,
                "market_phase": "CONTINUATION" if len(prior) >= 2 else "FIRST_SHOCK",
                "realized_volatility_20d": 0.02,
                "breadth_up_pct": 55.0,
            },
            evidence=(),
            data_quality=DataStatus.FRESH,
            reliability_grade="A",
            source_times={"fixture": raw.as_of_time - timedelta(hours=1)},
        )


def _snapshot(day: int, close: float, *, available_offset: timedelta = timedelta(0)):
    as_of = datetime(2026, 7, day, 21, tzinfo=UTC)
    available = as_of - timedelta(hours=1) + available_offset
    point = MarketDataPoint(
        market=Market.US,
        symbol="SPX",
        field="index_price",
        value=close,
        data_time=as_of - timedelta(hours=1),
        fetched_at=as_of + timedelta(days=10),
        source="fixture",
        available_at=available,
    )
    return RawMarketSnapshot(
        market=Market.US,
        as_of_time=as_of,
        session_slot="close",
        points=(point,),
        source_times={"fixture": point.data_time},
    )


class DatasetBuilderTests(TestCase):
    def test_feature_builder_receives_only_the_declared_maximum_lookback(self):
        class RecordingStrategy(_FeatureStrategy):
            def __init__(self):
                self.history_lengths = []

            def build(self, raw, prior_history):
                prior = tuple(prior_history)
                self.history_lengths.append(len(prior))
                return super().build(raw, prior)

        start = datetime(2020, 1, 1, 21, tzinfo=UTC)
        snapshots = []
        for offset in range(300):
            as_of = start + timedelta(days=offset)
            point = MarketDataPoint(
                market=Market.US,
                symbol="SPX",
                field="index_price",
                value=100.0 + offset,
                data_time=as_of - timedelta(hours=1),
                fetched_at=as_of,
                source="fixture",
                available_at=as_of - timedelta(hours=1),
            )
            snapshots.append(
                RawMarketSnapshot(
                    market=Market.US,
                    as_of_time=as_of,
                    session_slot="close",
                    points=(point,),
                    source_times={"fixture": point.data_time},
                )
            )
        strategy = RecordingStrategy()

        result = build_point_in_time_dataset(Market.US, snapshots, strategy)

        self.assertEqual(len(result.frame), 300)
        self.assertLessEqual(max(strategy.history_lengths), 253)

    def test_dataset_has_auditable_availability_labels_and_duplicate_counts(self):
        snapshots = (
            _snapshot(1, 100.0),
            _snapshot(1, 100.0),
            _snapshot(2, 96.0),
            _snapshot(3, 95.0),
            _snapshot(4, 94.0),
            _snapshot(5, 97.0),
        )

        result = build_point_in_time_dataset(Market.US, snapshots, _FeatureStrategy())

        self.assertEqual(len(result.frame), 5)
        self.assertEqual(result.audit.duplicate_snapshot_keys, 1)
        self.assertEqual(result.audit.missing_close_rows, 0)
        self.assertEqual(result.audit.point_in_time_violations, 0)
        self.assertEqual(result.frame.attrs["feature_version"], FEATURE_VERSION)
        self.assertTrue(result.frame.attrs["point_in_time_validated"])
        self.assertEqual(
            result.frame.attrs["availability_proof"],
            {"*": "feature_available_at", "close": "close_available_at"},
        )
        self.assertTrue(bool(result.frame.loc[0, "label_1d"]))
        self.assertTrue(bool(result.frame.loc[0, "label_3d"]))
        self.assertIn("old_market_risk_alert", result.frame)
        self.assertLessEqual(
            result.frame["feature_available_at"].max(),
            result.frame["as_of_time"].max(),
        )

    def test_future_available_point_fails_the_backfill_instead_of_becoming_training_data(self):
        snapshots = (
            _snapshot(1, 100.0, available_offset=timedelta(hours=2)),
            _snapshot(2, 99.0),
            _snapshot(3, 98.0),
            _snapshot(4, 97.0),
        )

        with self.assertRaisesRegex(ValueError, "point-in-time violation"):
            build_point_in_time_dataset(Market.US, snapshots, _FeatureStrategy())

    def test_dataset_write_is_atomic_and_round_trips_dataframe_attributes(self):
        result = build_point_in_time_dataset(
            Market.US,
            tuple(_snapshot(day, 100.0 - day) for day in range(1, 6)),
            _FeatureStrategy(),
        )
        with tempfile.TemporaryDirectory(prefix="warning_dataset_") as directory:
            target = Path(directory) / "nested" / "us.joblib"

            write_dataset(result.frame, target)

            self.assertTrue(target.is_file())
            self.assertFalse(any(target.parent.glob(f".{target.name}.*.tmp")))

    def test_backfill_runner_writes_dataset_and_sidecar_audit(self):
        class Adapter:
            def backfill(self, start_date, end_date):
                self.window = (start_date, end_date)
                return tuple(_snapshot(day, 100.0 - day) for day in range(1, 6))

        adapter = Adapter()
        with tempfile.TemporaryDirectory(prefix="warning_backfill_") as directory:
            output = Path(directory) / "us.joblib"

            payload = run_backfill(
                Market.US,
                "2026-07-01",
                "2026-07-31",
                adapter,
                _FeatureStrategy(),
                output,
            )

            self.assertEqual(adapter.window[0].isoformat(), "2026-07-01")
            self.assertEqual(adapter.window[1].isoformat(), "2026-07-31")
            self.assertEqual(payload["rows"], 5)
            self.assertEqual(payload["point_in_time_violations"], 0)
            self.assertTrue(output.is_file())
            loaded = joblib.load(output)
            self.assertTrue(loaded.attrs["point_in_time_validated"])
            audit_path = output.with_suffix(".audit.json")
            self.assertEqual(json.loads(audit_path.read_text())["output_rows"], 5)


class PromotionGateTests(TestCase):
    @staticmethod
    def _report(market, horizon, **overrides):
        report = {
            "market": market,
            "horizon": horizon,
            "prevalence": 0.02,
            "brier_score": 0.015,
            "constant_base_rate_brier": 0.019,
            "average_precision": 0.08,
            "expected_calibration_error": 0.03,
            "monthly_alert_entries": 4.0,
            "crisis_contribution": {"2008": 0.3, "2020": 0.3, "non_crisis": 0.4},
        }
        report.update(overrides)
        return report

    def test_all_four_models_must_beat_every_frozen_promotion_gate(self):
        reports = [
            self._report(market, horizon)
            for market in ("a_share", "us")
            for horizon in ("1d", "3d")
        ]

        self.assertEqual(promotion_failures(reports), ())

        reports[-1] = self._report(
            "us",
            "3d",
            brier_score=0.019,
            average_precision=0.02,
            expected_calibration_error=0.051,
            monthly_alert_entries=6.1,
            crisis_contribution={"2020": 0.51, "non_crisis": 0.49},
        )
        failures = promotion_failures(reports)
        self.assertTrue(any("Brier" in item for item in failures))
        self.assertTrue(any("AUPRC" in item for item in failures))
        self.assertTrue(any("calibration" in item for item in failures))
        self.assertTrue(any("crisis" in item for item in failures))
        self.assertTrue(any("alert budget" in item for item in failures))

    def test_missing_model_report_blocks_promotion(self):
        failures = promotion_failures([self._report("a_share", "1d")])

        self.assertTrue(any("missing" in item for item in failures))

    def test_non_crisis_true_positives_do_not_trigger_named_crisis_concentration_gate(self):
        reports = [
            self._report(
                market,
                horizon,
                crisis_contribution={"2008": 0.05, "2020": 0.15, "non_crisis": 0.80},
            )
            for market in ("a_share", "us")
            for horizon in ("1d", "3d")
        ]

        self.assertEqual(promotion_failures(reports), ())


if __name__ == "__main__":
    main()
