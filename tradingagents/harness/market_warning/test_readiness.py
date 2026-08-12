from __future__ import annotations

import hashlib
import json
import tempfile
from datetime import datetime, time, timedelta
from pathlib import Path
from unittest import TestCase, main
from zoneinfo import ZoneInfo

from tradingagents.harness import db as _db

from tradingagents.harness.market_warning.adapters.sqlite_repository import (
    SQLiteWarningRepository,
)
from tradingagents.harness.market_warning.domain import Market
from tradingagents.harness.market_warning.features import FEATURE_VERSION
from tradingagents.harness.market_warning.readiness import check_production_readiness
from tradingagents.harness.market_warning.rule_policy import manifest_sha256


RULE_MANIFEST = Path(__file__).with_name("rule_manifest_v1.json")


class ProductionReadinessTests(TestCase):
    @staticmethod
    def _record_complete_rule_sessions(
        repository: SQLiteWarningRepository,
        session_dates,
    ) -> None:
        zone = ZoneInfo("Asia/Shanghai")
        slots = ["premarket"]
        slots.extend(f"intraday-{hour:02d}{minute:02d}" for hour, minute in (
            *((9, minute) for minute in range(35, 60, 10)),
            *((10, minute) for minute in range(5, 60, 10)),
            *((11, minute) for minute in range(5, 26, 10)),
            *((13, minute) for minute in range(5, 60, 10)),
            *((14, minute) for minute in range(5, 56, 10)),
        ))
        for session_date in session_dates:
            for slot in slots:
                slot_time = time(8, 30) if slot == "premarket" else time(
                    int(slot[-4:-2]), int(slot[-2:])
                )
                as_of = datetime.combine(session_date, slot_time, zone)
                repository.record_run(
                    market=Market.A_SHARE,
                    as_of_time=as_of,
                    session_slot=slot,
                    mode="rule_v1",
                    started_at=as_of,
                    finished_at=as_of + timedelta(seconds=1),
                    status="success",
                    error_class=None,
                    overlap_skipped=False,
                    llm_calls=0,
                )

    @staticmethod
    def _backdate_notify_activation(repository: SQLiteWarningRepository) -> None:
        with _db.connect(repository._db_path) as connection:
            connection.execute(
                "UPDATE market_warning_rule_registry SET notification_activated_at = ? "
                "WHERE notification_active = 1",
                ("2026-07-17T00:00:00+00:00",),
            )

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
        push_events_per_month: float | None = None,
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
                        "push_events_per_month": (
                            alerts_per_month
                            if push_events_per_month is None
                            else push_events_per_month
                        ),
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
                as_of_time=datetime.fromisoformat("2026-08-03T08:00:00+08:00"),
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
            self._backdate_notify_activation(repository)
            self._record_complete_rule_sessions(
                repository,
                tuple(datetime(2026, 7, day).date() for day in range(20, 25))
                + tuple(datetime(2026, 7, day).date() for day in range(27, 32)),
            )
            evaluation, smoke, benchmark = self._rule_artifacts(root, concentration=0.50)
            ready = check_production_readiness(
                repository,
                mode="rule_v1/gate",
                rule_manifest=RULE_MANIFEST,
                rule_evaluation_path=evaluation,
                data_smoke_path=smoke,
                runtime_benchmark_path=benchmark,
                as_of_time=datetime.fromisoformat("2026-08-03T08:00:00+08:00"),
            )

        self.assertFalse(blocked["ready"])
        self.assertTrue(any("crisis concentration" in item for item in blocked["failures"]))
        self.assertTrue(any("soak" in item for item in blocked["failures"]))
        self.assertTrue(any("active notify" in item for item in blocked["failures"]))
        self.assertTrue(ready["ready"])
        self.assertEqual(ready["soak_sessions"], 10)
        self.assertEqual(ready["soak_audit"]["scan_success_rate"], 1.0)

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
                as_of_time=datetime.fromisoformat("2026-08-03T08:00:00+08:00"),
            )

        self.assertFalse(result["ready"])
        self.assertTrue(any("active notify" in item for item in result["failures"]))

    def test_rule_gate_fails_closed_on_corrupt_activation_timestamp(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repository = SQLiteWarningRepository(root / "warning.db")
            repository.register_rule_engine(
                {
                    "engine_version": "rule-v1.0.0",
                    "market": Market.A_SHARE,
                    "manifest_sha256": manifest_sha256(RULE_MANIFEST),
                    "metrics": {},
                }
            )
            repository.activate_rule_engine("rule-v1.0.0", "notify")
            with _db.connect(root / "warning.db") as connection:
                connection.execute(
                    "UPDATE market_warning_rule_registry "
                    "SET notification_activated_at = 'not-a-time'"
                )
            evaluation, smoke, benchmark = self._rule_artifacts(
                root, concentration=0.50
            )

            result = check_production_readiness(
                repository,
                mode="rule_v1/gate",
                rule_manifest=RULE_MANIFEST,
                rule_evaluation_path=evaluation,
                data_smoke_path=smoke,
                runtime_benchmark_path=benchmark,
                as_of_time=datetime.fromisoformat("2026-08-03T08:00:00+08:00"),
            )

        self.assertFalse(result["ready"])
        self.assertTrue(any("activation timestamp" in item for item in result["failures"]))


if __name__ == "__main__":
    main()
