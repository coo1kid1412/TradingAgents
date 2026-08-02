from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path
from unittest import TestCase, main

from tradingagents.harness.market_warning.adapters.sqlite_repository import (
    SQLiteWarningRepository,
)
from tradingagents.harness.market_warning.domain import Market
from tradingagents.harness.market_warning.features import FEATURE_VERSION
from tradingagents.harness.market_warning.readiness import check_production_readiness


class ProductionReadinessTests(TestCase):
    def _register_set(
        self,
        repository: SQLiteWarningRepository,
        artifact_root: Path,
        *,
        version: str = "ready-v1",
        feature_version: str = FEATURE_VERSION,
    ) -> None:
        for market in Market:
            for horizon in ("1d", "3d"):
                artifact = artifact_root / version / market.value / f"{horizon}.joblib"
                artifact.parent.mkdir(parents=True, exist_ok=True)
                artifact.write_bytes(f"{market.value}:{horizon}".encode())
                repository.register_model(
                    {
                        "model_version": version,
                        "market": market,
                        "horizon": horizon,
                        "feature_version": feature_version,
                        "calibration_version": "isotonic-v1",
                        "training_cutoff": "2026-07-31",
                        "artifact_path": str(artifact),
                        "artifact_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
                        "metrics": {},
                        "base_rate": 0.01,
                        "active": True,
                    }
                )

    def test_missing_active_models_block_production(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = check_production_readiness(
                SQLiteWarningRepository(root / "warning.db"),
                root / "models",
            )

        self.assertFalse(result["ready"])
        self.assertEqual(len(result["failures"]), 4)
        self.assertTrue(all("missing active model" in item for item in result["failures"]))

    def test_complete_verified_active_set_is_ready(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repository = SQLiteWarningRepository(root / "warning.db")
            artifact_root = root / "models"
            self._register_set(repository, artifact_root)

            result = check_production_readiness(repository, artifact_root)

        self.assertTrue(result["ready"])
        self.assertEqual(result["model_version"], "ready-v1")
        self.assertEqual(result["failures"], [])

    def test_mixed_versions_and_tampered_artifacts_block_production(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repository = SQLiteWarningRepository(root / "warning.db")
            artifact_root = root / "models"
            self._register_set(repository, artifact_root)
            artifact = artifact_root / "ready-v1" / Market.US.value / "3d.joblib"
            artifact.write_bytes(b"tampered")
            repository.register_model(
                {
                    **repository.load_active_model(Market.A_SHARE, "1d"),
                    "model_version": "other-v2",
                    "active": True,
                }
            )

            result = check_production_readiness(repository, artifact_root)

        self.assertFalse(result["ready"])
        self.assertTrue(any("one model version" in item for item in result["failures"]))
        self.assertTrue(any("checksum mismatch" in item for item in result["failures"]))


if __name__ == "__main__":
    main()
