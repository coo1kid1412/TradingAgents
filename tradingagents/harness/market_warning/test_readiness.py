from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path
from unittest import TestCase, main

from tradingagents.harness.market_warning.adapters.sqlite_repository import (
    SQLiteWarningRepository,
)
from tradingagents.harness.market_warning.domain import Market
from tradingagents.harness.market_warning.features import FEATURE_VERSION
from tradingagents.harness.market_warning.readiness import check_production_readiness
from tradingagents.harness.market_warning.rule_policy import manifest_sha256


RULE_MANIFEST = Path(__file__).with_name("rule_manifest_v1.json")


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

    @staticmethod
    def _rule_artifacts(
        root: Path,
        *,
        lift: float = 2.4,
        alerts_per_month: float = 4.0,
        concentration: float = 0.75,
        data_ready: bool = True,
        p95_seconds: float = 12.0,
        llm_calls: int = 0,
    ) -> tuple[Path, Path, Path]:
        evaluation = root / "evaluation.json"
        smoke = root / "data-smoke.json"
        benchmark = root / "benchmark.json"
        evaluation.write_text(
            json.dumps(
                {
                    "engine_version": "rule-v1.0.0",
                    "manifest_sha256": manifest_sha256(RULE_MANIFEST),
                    "production_gates": {
                        "lift": lift,
                        "alerts_per_month": alerts_per_month,
                        "max_crisis_contribution": concentration,
                    },
                }
            ),
            encoding="utf-8",
        )
        smoke.write_text(json.dumps({"ready": data_ready}), encoding="utf-8")
        benchmark.write_text(
            json.dumps({"p95_seconds": p95_seconds, "llm_calls": llm_calls, "runs": 100}),
            encoding="utf-8",
        )
        return evaluation, smoke, benchmark

    def test_rule_notify_mode_does_not_require_gate_concentration_or_soak(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            evaluation, smoke, benchmark = self._rule_artifacts(root, concentration=0.75)

            result = check_production_readiness(
                SQLiteWarningRepository(root / "warning.db"),
                mode="rule_v1/notify",
                rule_manifest=RULE_MANIFEST,
                rule_evaluation_path=evaluation,
                data_smoke_path=smoke,
                runtime_benchmark_path=benchmark,
                soak_sessions=0,
            )

        self.assertTrue(result["ready"])
        self.assertEqual(result["mode"], "rule_v1/notify")

    def test_rule_notify_mode_enforces_lift_budget_data_and_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            evaluation, smoke, benchmark = self._rule_artifacts(
                root,
                lift=2.0,
                alerts_per_month=6.1,
                data_ready=False,
                p95_seconds=30.0,
                llm_calls=1,
            )

            result = check_production_readiness(
                SQLiteWarningRepository(root / "warning.db"),
                mode="rule_v1/notify",
                rule_manifest=RULE_MANIFEST,
                rule_evaluation_path=evaluation,
                data_smoke_path=smoke,
                runtime_benchmark_path=benchmark,
            )

        self.assertFalse(result["ready"])
        self.assertTrue(any("lift" in item for item in result["failures"]))
        self.assertTrue(any("alert budget" in item for item in result["failures"]))
        self.assertTrue(any("data smoke" in item for item in result["failures"]))
        self.assertTrue(any("P95" in item for item in result["failures"]))
        self.assertTrue(any("LLM" in item for item in result["failures"]))

    def test_rule_gate_adds_crisis_concentration_and_ten_session_soak(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repository = SQLiteWarningRepository(root / "warning.db")
            evaluation, smoke, benchmark = self._rule_artifacts(root, concentration=0.51)
            blocked = check_production_readiness(
                repository,
                mode="rule_v1/gate",
                rule_manifest=RULE_MANIFEST,
                rule_evaluation_path=evaluation,
                data_smoke_path=smoke,
                runtime_benchmark_path=benchmark,
                soak_sessions=9,
            )
            repository.register_rule_engine(
                {
                    "engine_version": "rule-v1.0.0",
                    "market": Market.A_SHARE,
                    "manifest_sha256": manifest_sha256(RULE_MANIFEST),
                    "metrics": {},
                }
            )
            repository.activate_rule_engine("rule-v1.0.0", "notify")
            evaluation, smoke, benchmark = self._rule_artifacts(root, concentration=0.50)
            ready = check_production_readiness(
                repository,
                mode="rule_v1/gate",
                rule_manifest=RULE_MANIFEST,
                rule_evaluation_path=evaluation,
                data_smoke_path=smoke,
                runtime_benchmark_path=benchmark,
                soak_sessions=10,
            )

        self.assertFalse(blocked["ready"])
        self.assertTrue(any("crisis concentration" in item for item in blocked["failures"]))
        self.assertTrue(any("soak" in item for item in blocked["failures"]))
        self.assertTrue(any("active notify" in item for item in blocked["failures"]))
        self.assertTrue(ready["ready"])

    def test_rule_gate_requires_active_notify_checksum_to_match_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repository = SQLiteWarningRepository(root / "warning.db")
            repository.register_rule_engine(
                {
                    "engine_version": "rule-v1.0.0",
                    "market": Market.A_SHARE,
                    "manifest_sha256": "f" * 64,
                    "metrics": {},
                }
            )
            repository.activate_rule_engine("rule-v1.0.0", "notify")
            evaluation, smoke, benchmark = self._rule_artifacts(root, concentration=0.50)

            result = check_production_readiness(
                repository,
                mode="rule_v1/gate",
                rule_manifest=RULE_MANIFEST,
                rule_evaluation_path=evaluation,
                data_smoke_path=smoke,
                runtime_benchmark_path=benchmark,
                soak_sessions=10,
            )

        self.assertFalse(result["ready"])
        self.assertTrue(any("active notify" in item for item in result["failures"]))


if __name__ == "__main__":
    main()
