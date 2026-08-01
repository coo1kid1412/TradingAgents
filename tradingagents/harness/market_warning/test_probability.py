"""Runtime model artifact and prediction contract tests."""

from __future__ import annotations

import hashlib
import shutil
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import TestCase, main

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from tradingagents.harness.market_warning.domain import DataStatus, Evidence, FeatureSnapshot, Market
from tradingagents.harness.market_warning.features import FEATURE_VERSION
from tradingagents.harness.market_warning.probability import SklearnProbabilityModel, save_model_bundle
from tradingagents.harness.market_warning.training import fit_model


AS_OF = datetime(2026, 7, 31, 7, 0, tzinfo=timezone.utc)


class InMemoryModelRegistry:
    def __init__(self):
        self.records = {}

    def register_model(self, record):
        values = dict(record)
        key = (Market(values["market"]), str(values["horizon"]))
        if values.get("active"):
            for item_key, item in self.records.items():
                if item_key == key:
                    item["active"] = False
        self.records[key] = values

    def load_active_model(self, market, horizon):
        record = self.records.get((Market(market), str(horizon)))
        return dict(record) if record and record.get("active") else None


def _training_frame(start: str, rows: int) -> pd.DataFrame:
    dates = pd.bdate_range(start, periods=rows, tz="UTC")
    signal = np.linspace(-3, 3, rows)
    labels = (np.arange(rows) % 10 == 0) | (signal > 2.3)
    frame = pd.DataFrame(
        {
            "as_of_time": dates,
            "signal": signal,
            "missing_signal": np.where(np.arange(rows) % 6 == 0, np.nan, -signal),
            "market_phase": np.where(signal < 0, "FIRST_SHOCK", "CONTINUATION"),
            "label_1d": labels,
            "label_3d": labels | (np.arange(rows) % 14 == 0),
        }
    )
    frame.attrs["feature_version"] = FEATURE_VERSION
    return frame


def _snapshot(market: Market = Market.A_SHARE, **changes) -> FeatureSnapshot:
    values = {
        "market": market,
        "as_of_time": AS_OF,
        "session_slot": "premarket",
        "feature_version": FEATURE_VERSION,
        "features": {"signal": 1.5, "missing_signal": None, "market_phase": "FIRST_SHOCK"},
        "evidence": (
            Evidence(
                evidence_id=f"{market.value}:{FEATURE_VERSION}:signal:{AS_OF.isoformat()}",
                group="feature",
                summary="signal",
                value=1.5,
                as_of_time=AS_OF,
            ),
            Evidence(
                evidence_id=f"{market.value}:{FEATURE_VERSION}:missing_signal:{AS_OF.isoformat()}",
                group="feature",
                summary="missing signal",
                value=None,
                as_of_time=AS_OF,
            ),
        ),
        "data_quality": DataStatus.FRESH,
        "reliability_grade": "A",
        "source_times": {"fixture": AS_OF},
    }
    values.update(changes)
    return FeatureSnapshot(**values)


class ProbabilityArtifactTests(TestCase):
    def setUp(self):
        self.directory = Path(tempfile.mkdtemp(prefix="warning_probability_"))
        self.repository = InMemoryModelRegistry()
        train = _training_frame("2008-01-01", 100)
        calibration = _training_frame("2014-01-01", 45)
        self.bundles = {
            horizon: fit_model(train, calibration, Market.A_SHARE, horizon)
            for horizon in ("1d", "3d")
        }

    def tearDown(self):
        shutil.rmtree(self.directory, ignore_errors=True)

    def _register_both(self):
        return {
            horizon: save_model_bundle(
                bundle,
                self.directory / "models",
                self.repository,
                active=True,
            )
            for horizon, bundle in self.bundles.items()
        }

    def _model(self, **changes):
        values = {
            "repository": self.repository,
            "artifact_root": self.directory / "models",
            "max_model_age": timedelta(days=5000),
            "now": lambda: AS_OF,
        }
        values.update(changes)
        return SklearnProbabilityModel(**values)

    def test_registered_artifacts_are_immutable_checksum_addressed_joblib_files(self):
        records = self._register_both()

        for horizon, record in records.items():
            artifact = Path(record["artifact_path"])
            self.assertTrue(artifact.is_file())
            self.assertEqual(artifact.suffix, ".joblib")
            self.assertEqual(hashlib.sha256(artifact.read_bytes()).hexdigest(), record["artifact_sha256"])
            self.assertEqual(record["horizon"], horizon)
            self.assertTrue(record["active"])

    def test_predict_loads_two_independent_horizons_and_maps_contributors_to_evidence(self):
        self._register_both()

        result = self._model().predict(_snapshot())

        self.assertEqual(result.reliability_grade, "A")
        self.assertGreaterEqual(result.crash_1d_probability, 0)
        self.assertLessEqual(result.crash_3d_probability, 1)
        self.assertTrue(result.top_contributors)
        contributor = result.top_contributors[0]
        self.assertIn("feature", contributor)
        self.assertIn("contribution", contributor)
        self.assertIn("evidence_id", contributor)

    def test_missing_active_model_returns_unavailable_without_creating_artifacts(self):
        model_root = self.directory / "empty-models"

        result = self._model(artifact_root=model_root).predict(_snapshot())

        self.assertEqual(result.reliability_grade, "UNAVAILABLE")
        self.assertFalse(model_root.exists())

    def test_checksum_mismatch_returns_unavailable_without_loading_or_retraining(self):
        records = self._register_both()
        Path(records["1d"]["artifact_path"]).write_bytes(b"tampered")

        result = self._model().predict(_snapshot())

        self.assertEqual(result.reliability_grade, "UNAVAILABLE")
        self.assertEqual(Path(records["1d"]["artifact_path"]).read_bytes(), b"tampered")

    def test_feature_version_mismatch_returns_unavailable(self):
        self._register_both()

        result = self._model().predict(_snapshot(feature_version="future-feature-v2"))

        self.assertEqual(result.reliability_grade, "UNAVAILABLE")

    def test_stale_registry_training_cutoff_returns_unavailable(self):
        records = self._register_both()
        stale_now = lambda: datetime(2035, 1, 1, tzinfo=timezone.utc)

        result = self._model(now=stale_now, max_model_age=timedelta(days=30)).predict(_snapshot())

        self.assertEqual(result.reliability_grade, "UNAVAILABLE")
        self.assertTrue(all(record["active"] for record in records.values()))

    def test_registry_metadata_mismatch_returns_unavailable(self):
        records = self._register_both()
        record = dict(records["3d"])
        record["feature_version"] = "wrong-registry-version"
        self.repository.register_model(record)

        result = self._model().predict(_snapshot())

        self.assertEqual(result.reliability_grade, "UNAVAILABLE")


if __name__ == "__main__":
    main()
