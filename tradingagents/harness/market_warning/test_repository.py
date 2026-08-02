"""SQLite repository contract tests for market-warning audit records."""

from __future__ import annotations

import json
import shutil
import sqlite3
import sys
import tempfile
from contextlib import closing, contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import TestCase, main

# Keep the brief's direct-script command rooted at this worktree.
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from tradingagents.harness import db as _db
from tradingagents.harness.market_warning.domain import (
    DecisionSource,
    Evidence,
    FeatureSnapshot,
    FinalWarningDecision,
    LLMContextAssessment,
    Market,
    QuantRiskAssessment,
    RuleRiskAssessment,
    TriggeredRule,
)
from tradingagents.harness.market_warning.adapters.sqlite_repository import (
    SQLiteCircuitBreaker,
    SQLiteWarningRepository,
)
from tradingagents.harness.market_warning.policy import build_final_decision


AS_OF = datetime(2026, 8, 1, 9, 35, tzinfo=timezone.utc)


@contextmanager
def temporary_database():
    directory = Path(tempfile.mkdtemp(prefix="market_warning_"))
    try:
        yield directory / "warning.db"
    finally:
        shutil.rmtree(directory, ignore_errors=True)


def make_snapshot(**changes) -> FeatureSnapshot:
    values = {
        "market": Market.A_SHARE,
        "as_of_time": AS_OF,
        "session_slot": "premarket",
        "feature_version": "features-v1",
        "features": {"breadth_pct": 22.0, "volatility": {"vix": 27.5}},
        "evidence": (
            Evidence(
                evidence_id="breadth-1",
                group="breadth",
                summary="Market breadth is weak.",
                value=22.0,
                source="exchange",
                as_of_time=AS_OF,
            ),
        ),
        "data_quality": "partial",
        "reliability_grade": "B",
        "source_times": {"exchange": AS_OF},
    }
    values.update(changes)
    return FeatureSnapshot(**values)


def make_quant(**changes) -> QuantRiskAssessment:
    values = {
        "crash_1d_probability": 0.35,
        "crash_3d_probability": 0.48,
        "market_phase": "FIRST_SHOCK",
        "base_rate_1d": 0.10,
        "base_rate_3d": 0.16,
        "reliability_grade": "A",
        "model_version": "quant-v1",
        "calibration_version": "cal-v1",
        "top_contributors": ("breadth", {"factor": "volatility", "weight": 0.31}),
    }
    values.update(changes)
    return QuantRiskAssessment(**values)


def make_reasoning(**changes) -> LLMContextAssessment:
    values = {
        "market_scenario": "broad deleveraging",
        "causal_chain": ("breadth weakened", "volatility increased"),
        "supporting_evidence_ids": ("breadth-1",),
        "conflicting_evidence_ids": (),
        "overlooked_risks": ("policy response",),
        "recommended_risk_level": "ORANGE",
        "confidence": 0.78,
        "action_reason": "Reduce new exposure.",
        "reasoning_status": "validated",
    }
    values.update(changes)
    return LLMContextAssessment(**values)


def make_decision(**changes) -> FinalWarningDecision:
    values = {
        "baseline_level": "YELLOW",
        "final_level": "ORANGE",
        "state_transition": "YELLOW_TO_ORANGE",
        "entry_gate": "LIMITED",
        "new_position_cap_pct": 5.0,
        "holding_action": "reduce",
        "push_required": True,
        "decision_reasons": ("quant risk increased", "context agrees"),
        "data_status": "partial",
    }
    values.update(changes)
    return FinalWarningDecision(**values)


def make_rule_assessment(**changes) -> RuleRiskAssessment:
    values = {
        "market": Market.A_SHARE,
        "as_of_time": AS_OF,
        "engine_version": "rule-v1.0.0",
        "manifest_sha256": "a" * 64,
        "risk_level": "ORANGE",
        "risk_score": 4.0,
        "market_phase": "FIRST_SHOCK",
        "triggered_rules": (
            TriggeredRule(
                rule_id="pressure.volatility_acceleration",
                layer="PRESSURE",
                severity_points=2,
                observed_value=1.6,
                threshold_description=">= 1.50",
                evidence_ids=("breadth-1",),
            ),
        ),
        "missing_optional_groups": ("margin",),
        "reliability_grade": "B",
        "evaluation_latency_ms": 7.5,
    }
    values.update(changes)
    return RuleRiskAssessment(**values)


def save_complete_decision(
    repository: SQLiteWarningRepository, snapshot: FeatureSnapshot | None = None
) -> tuple[int, tuple[int, int]]:
    feature_snapshot_id = repository.save_feature_snapshot(snapshot or make_snapshot())
    prediction_ids = repository.save_predictions(feature_snapshot_id, make_quant())
    reasoning_id = repository.save_reasoning(feature_snapshot_id, make_reasoning(), "reasoning-v1")
    return (
        repository.save_decision(
            feature_snapshot_id,
            prediction_ids,
            reasoning_id,
            make_decision(),
        ),
        prediction_ids,
    )


class SQLiteWarningRepositoryTests(TestCase):
    @staticmethod
    def _model_record(version: str, market: Market, horizon: str, *, active: bool):
        return {
            "model_version": version,
            "market": market,
            "horizon": horizon,
            "feature_version": "market-warning-v1",
            "calibration_version": "platt-v1",
            "training_cutoff": "2019-12-31",
            "artifact_path": f"/tmp/{version}-{market.value}-{horizon}.joblib",
            "artifact_sha256": "a" * 64,
            "metrics": {"brier_score": 0.01},
            "base_rate": 0.02,
            "active": active,
        }

    def test_init_db_closes_its_schema_connection(self):
        with temporary_database() as db_path:
            opened = []
            original_connect = _db.sqlite3.connect

            def tracking_connect(*args, **kwargs):
                connection = original_connect(*args, **kwargs)
                opened.append(connection)
                return connection

            _db.sqlite3.connect = tracking_connect
            try:
                _db.init_db(db_path)
            finally:
                _db.sqlite3.connect = original_connect

            self.assertEqual(len(opened), 1)
            try:
                opened[0].execute("SELECT 1")
            except sqlite3.ProgrammingError:
                pass
            else:
                opened[0].close()
                self.fail("init_db left its schema connection open")

    def test_connect_initializes_warning_tables_and_rule_decision_columns(self):
        with temporary_database() as db_path:
            with _db.connect(db_path) as connection:
                tables = {
                    row["name"]
                    for row in connection.execute(
                        "SELECT name FROM sqlite_master WHERE type = 'table'"
                    )
                }
                reliability_not_null = next(
                    row["notnull"]
                    for row in connection.execute("PRAGMA table_info(market_warning_feature_snapshots)")
                    if row["name"] == "reliability_grade"
                )
                decision_columns = {
                    row["name"]: row
                    for row in connection.execute("PRAGMA table_info(market_warning_decisions)")
                }

        self.assertTrue(
            {
                "market_warning_feature_snapshots",
                "market_warning_predictions",
                "market_warning_reasoning",
                "market_warning_decisions",
                "market_warning_alerts",
                "market_warning_model_registry",
                "market_warning_rule_assessments",
                "market_warning_rule_registry",
            }.issubset(tables)
        )
        self.assertEqual(reliability_not_null, 1)
        self.assertEqual(decision_columns["valid_snapshot_count"]["notnull"], 1)
        self.assertEqual(decision_columns["valid_snapshot_count"]["dflt_value"], "0")
        self.assertIn("retained_risk_level", decision_columns)
        self.assertEqual(decision_columns["decision_source"]["dflt_value"], "'model'")
        self.assertIn("rule_assessment_id", decision_columns)
        self.assertEqual(decision_columns["shadow_prediction_ids_json"]["dflt_value"], "'[]'")

    def test_existing_decision_table_migration_is_idempotent(self):
        with temporary_database() as db_path:
            with closing(sqlite3.connect(db_path)) as connection:
                connection.execute(
                    "CREATE TABLE market_warning_decisions ("
                    "id INTEGER PRIMARY KEY AUTOINCREMENT, feature_snapshot_id INTEGER NOT NULL UNIQUE, "
                    "prediction_ids_json TEXT NOT NULL, reasoning_id INTEGER, baseline_level TEXT NOT NULL, "
                    "final_level TEXT NOT NULL, transition TEXT NOT NULL, entry_gate TEXT NOT NULL, "
                    "new_position_cap_pct REAL NOT NULL, holding_action TEXT NOT NULL, "
                    "push_required INTEGER NOT NULL, data_status TEXT NOT NULL, reasons_json TEXT NOT NULL, "
                    "model_version TEXT, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)"
                )

            _db.init_db(db_path)
            _db.init_db(db_path)

            with closing(sqlite3.connect(db_path)) as connection:
                columns = {
                    row[1]: row for row in connection.execute("PRAGMA table_info(market_warning_decisions)")
                }
            self.assertEqual(columns["valid_snapshot_count"][3], 1)
            self.assertEqual(columns["valid_snapshot_count"][4], "0")
            self.assertIn("retained_risk_level", columns)
            self.assertEqual(columns["decision_source"][3], 1)
            self.assertEqual(columns["decision_source"][4], "'model'")
            self.assertIn("rule_assessment_id", columns)
            self.assertEqual(columns["shadow_prediction_ids_json"][3], 1)
            self.assertEqual(columns["shadow_prediction_ids_json"][4], "'[]'")

    def test_rule_decision_round_trip_does_not_create_fake_probabilities(self):
        with temporary_database() as db_path:
            repository = SQLiteWarningRepository(db_path)
            snapshot_id = repository.save_feature_snapshot(make_snapshot())
            assessment = make_rule_assessment()
            assessment_id = repository.save_rule_assessment(snapshot_id, assessment)
            decision = make_decision(
                decision_reasons=("pressure.volatility_acceleration",),
                decision_source="rule_v1",
            )

            decision_id = repository.save_decision(
                snapshot_id,
                (),
                None,
                decision,
                rule_assessment_id=assessment_id,
            )
            loaded = repository.load_evaluation(Market.A_SHARE, AS_OF)

            with _db.connect(db_path) as connection:
                prediction_count = connection.execute(
                    "SELECT COUNT(*) AS count FROM market_warning_predictions"
                ).fetchone()["count"]
                decision_row = connection.execute(
                    "SELECT decision_source, rule_assessment_id, prediction_ids_json, "
                    "shadow_prediction_ids_json FROM market_warning_decisions WHERE id = ?",
                    (decision_id,),
                ).fetchone()

            self.assertEqual(prediction_count, 0)
            self.assertEqual(decision_row["decision_source"], "rule_v1")
            self.assertEqual(decision_row["rule_assessment_id"], assessment_id)
            self.assertEqual(json.loads(decision_row["prediction_ids_json"]), [])
            self.assertEqual(json.loads(decision_row["shadow_prediction_ids_json"]), [])
            self.assertIsNotNone(loaded)
            self.assertEqual(loaded.decision.decision_source, DecisionSource.RULE_V1)
            self.assertEqual(loaded.rule_assessment, assessment)
            self.assertIsNone(loaded.quant_assessment)
            self.assertIsNone(loaded.shadow_quant_assessment)

    def test_rule_registry_switches_notify_and_gate_versions_atomically(self):
        with temporary_database() as db_path:
            repository = SQLiteWarningRepository(db_path)
            for version, digest in (("rule-v1.0.0", "a" * 64), ("rule-v1.0.1", "b" * 64)):
                repository.register_rule_engine(
                    {
                        "engine_version": version,
                        "market": Market.A_SHARE,
                        "manifest_sha256": digest,
                        "metrics": {"lift": 2.4, "alerts_per_month": 4.0},
                    }
                )

            repository.activate_rule_engine("rule-v1.0.0", "notify")
            repository.activate_rule_engine("rule-v1.0.0", "gate")
            repository.activate_rule_engine("rule-v1.0.1", "notify")

            notify = repository.load_active_rule_engine(Market.A_SHARE, "notify")
            gate = repository.load_active_rule_engine(Market.A_SHARE, "gate")
            self.assertEqual(notify["engine_version"], "rule-v1.0.1")
            self.assertEqual(gate["engine_version"], "rule-v1.0.0")
            self.assertTrue(notify["notification_active"])
            self.assertFalse(notify["gate_active"])
            self.assertTrue(gate["gate_active"])

    def test_previous_rule_assessment_is_loaded_with_triggered_evidence(self):
        with temporary_database() as db_path:
            repository = SQLiteWarningRepository(db_path)
            first_snapshot_id = repository.save_feature_snapshot(make_snapshot())
            first = make_rule_assessment()
            repository.save_rule_assessment(first_snapshot_id, first)
            later = AS_OF + timedelta(minutes=10)
            second_snapshot_id = repository.save_feature_snapshot(
                make_snapshot(as_of_time=later, source_times={"exchange": later})
            )
            repository.save_rule_assessment(
                second_snapshot_id,
                make_rule_assessment(as_of_time=later, risk_level="RED"),
            )

            loaded = repository.load_previous_rule_assessment(Market.A_SHARE, later)

            self.assertEqual(loaded, first)
            self.assertEqual(
                loaded.triggered_rules[0].rule_id,
                "pressure.volatility_acceleration",
            )

    def test_activate_model_set_switches_all_four_horizons_in_one_commit(self):
        with temporary_database() as db_path:
            repository = SQLiteWarningRepository(db_path)
            for market in Market:
                for horizon in ("1d", "3d"):
                    repository.register_model(
                        self._model_record("old-v1", market, horizon, active=True)
                    )
                    repository.register_model(
                        self._model_record("new-v1", market, horizon, active=False)
                    )

            activated = repository.activate_model_set("new-v1")

            self.assertEqual(len(activated), 4)
            for market in Market:
                for horizon in ("1d", "3d"):
                    active = repository.load_active_model(market, horizon)
                    self.assertEqual(active["model_version"], "new-v1")

    def test_activate_incomplete_model_set_rolls_back_without_touching_active_models(self):
        with temporary_database() as db_path:
            repository = SQLiteWarningRepository(db_path)
            for market in Market:
                for horizon in ("1d", "3d"):
                    repository.register_model(
                        self._model_record("old-v1", market, horizon, active=True)
                    )
            repository.register_model(
                self._model_record("broken-v1", Market.US, "1d", active=False)
            )

            with self.assertRaisesRegex(ValueError, "four-model"):
                repository.activate_model_set("broken-v1")

            for market in Market:
                for horizon in ("1d", "3d"):
                    active = repository.load_active_model(market, horizon)
                    self.assertEqual(active["model_version"], "old-v1")

    def test_round_trip_persists_snapshot_two_horizons_reasoning_and_decision(self):
        with temporary_database() as db_path:
            repository = SQLiteWarningRepository(db_path)
            decision_id, prediction_ids = save_complete_decision(repository)

            with _db.connect(db_path) as connection:
                snapshot_row = connection.execute(
                    "SELECT reliability_grade, features_json, evidence_json, source_times_json "
                    "FROM market_warning_feature_snapshots WHERE id = ?",
                    (1,),
                ).fetchone()
                prediction_rows = connection.execute(
                    "SELECT id, horizon, probability, base_rate FROM market_warning_predictions "
                    "WHERE feature_snapshot_id = ? ORDER BY horizon",
                    (1,),
                ).fetchall()
                reasoning_row = connection.execute(
                    "SELECT structured_json FROM market_warning_reasoning WHERE id = ?",
                    (1,),
                ).fetchone()
                decision_row = connection.execute(
                    "SELECT model_version, prediction_ids_json FROM market_warning_decisions WHERE id = ?",
                    (decision_id,),
                ).fetchone()

            self.assertGreater(decision_id, 0)
            self.assertEqual(snapshot_row["reliability_grade"], "B")
            self.assertEqual(json.loads(snapshot_row["features_json"])["breadth_pct"], 22.0)
            self.assertEqual(json.loads(snapshot_row["evidence_json"])[0]["evidence_id"], "breadth-1")
            self.assertEqual(json.loads(snapshot_row["source_times_json"]), {"exchange": AS_OF.isoformat()})
            self.assertEqual(
                [(row["id"], row["horizon"], row["probability"], row["base_rate"]) for row in prediction_rows],
                [
                    (prediction_ids[0], "1d", 0.35, 0.10),
                    (prediction_ids[1], "3d", 0.48, 0.16),
                ],
            )
            self.assertEqual(json.loads(reasoning_row["structured_json"])["market_scenario"], "broad deleveraging")
            self.assertEqual(decision_row["model_version"], "quant-v1")
            self.assertEqual(json.loads(decision_row["prediction_ids_json"]), list(prediction_ids))
            self.assertEqual(repository.load_latest_decision(Market.A_SHARE), make_decision())
            self.assertEqual(
                repository.load_previous_decision(Market.A_SHARE, AS_OF + timedelta(minutes=1)),
                make_decision(),
            )

    def test_recovery_state_survives_repository_reloads_and_unknown_interruption(self):
        with temporary_database() as db_path:
            def persist(at, data_status, decision):
                repository = SQLiteWarningRepository(db_path)
                feature_snapshot_id = repository.save_feature_snapshot(
                    make_snapshot(
                        as_of_time=at,
                        data_quality=data_status,
                        reliability_grade="UNAVAILABLE" if data_status != "fresh" else "A",
                        source_times={"exchange": at},
                    )
                )
                prediction_ids = repository.save_predictions(feature_snapshot_id, make_quant())
                repository.save_decision(feature_snapshot_id, prediction_ids, None, decision)
                return SQLiteWarningRepository(db_path).load_latest_decision(Market.A_SHARE)

            red_snapshot = make_snapshot(
                as_of_time=AS_OF,
                data_quality="fresh",
                reliability_grade="A",
                source_times={"exchange": AS_OF},
            )
            red = build_final_decision(
                baseline="RED",
                candidate="RED",
                snapshot=red_snapshot,
                previous=None,
                valid_snapshot_count=0,
            )
            loaded = persist(AS_OF, "fresh", red)
            self.assertEqual(loaded.final_level.value, "RED")
            self.assertEqual(loaded.retained_risk_level.value, "RED")

            for minutes in (5, 10):
                at = AS_OF + timedelta(minutes=minutes)
                stale_snapshot = make_snapshot(
                    as_of_time=at,
                    data_quality="stale",
                    reliability_grade="UNAVAILABLE",
                    source_times={"exchange": at},
                )
                unknown = build_final_decision(
                    baseline="UNKNOWN",
                    candidate="UNKNOWN",
                    snapshot=stale_snapshot,
                    previous=loaded.final_level,
                    valid_snapshot_count=loaded.valid_snapshot_count,
                    retained_risk_level=loaded.retained_risk_level,
                )
                loaded = persist(at, "stale", unknown)
                self.assertEqual(loaded.final_level.value, "UNKNOWN")
                self.assertEqual(loaded.valid_snapshot_count, 0)
                self.assertEqual(loaded.retained_risk_level.value, "RED")

            first_green_at = AS_OF + timedelta(minutes=15)
            first_green_snapshot = make_snapshot(
                as_of_time=first_green_at,
                data_quality="fresh",
                reliability_grade="A",
                source_times={"exchange": first_green_at},
            )
            first_green = build_final_decision(
                baseline="GREEN",
                candidate="GREEN",
                snapshot=first_green_snapshot,
                previous=loaded.final_level,
                valid_snapshot_count=loaded.valid_snapshot_count + 1,
                retained_risk_level=loaded.retained_risk_level,
            )
            loaded = persist(first_green_at, "fresh", first_green)
            self.assertEqual(loaded.final_level.value, "RED")
            self.assertEqual(loaded.state_transition, "RECOVERY_PENDING")
            self.assertEqual(loaded.valid_snapshot_count, 1)
            self.assertEqual(loaded.retained_risk_level.value, "RED")

            second_green_at = AS_OF + timedelta(minutes=20)
            second_green_snapshot = make_snapshot(
                as_of_time=second_green_at,
                data_quality="fresh",
                reliability_grade="A",
                source_times={"exchange": second_green_at},
            )
            second_green = build_final_decision(
                baseline="GREEN",
                candidate="GREEN",
                snapshot=second_green_snapshot,
                previous=loaded.final_level,
                valid_snapshot_count=loaded.valid_snapshot_count + 1,
                retained_risk_level=loaded.retained_risk_level,
            )
            loaded = persist(second_green_at, "fresh", second_green)
            self.assertEqual(loaded.final_level.value, "GREEN")
            self.assertEqual(loaded.state_transition, "RECOVERY_RED_TO_GREEN")
            self.assertEqual(loaded.valid_snapshot_count, 0)
            self.assertIsNone(loaded.retained_risk_level)

    def test_alert_claim_is_atomic_across_repository_instances(self):
        with temporary_database() as db_path:
            first_repository = SQLiteWarningRepository(db_path)
            decision_id, _ = save_complete_decision(first_repository)
            second_repository = SQLiteWarningRepository(db_path)

            self.assertTrue(first_repository.claim_alert("a-share:2026-08-01", decision_id, "payload-sha"))
            self.assertFalse(second_repository.claim_alert("a-share:2026-08-01", decision_id, "payload-sha"))

            first_repository.finish_alert("a-share:2026-08-01", "sent")
            with _db.connect(db_path) as connection:
                alert = connection.execute(
                    "SELECT push_status, sent_at, error_summary FROM market_warning_alerts "
                    "WHERE idempotency_key = ?",
                    ("a-share:2026-08-01",),
                ).fetchone()

            self.assertEqual(alert["push_status"], "sent")
            self.assertIsNotNone(alert["sent_at"])
            self.assertIsNone(alert["error_summary"])

    def test_circuit_breaker_state_survives_new_process_instances(self):
        with temporary_database() as db_path:
            current = [AS_OF]
            clock = lambda: current[0]
            first = SQLiteCircuitBreaker(
                db_path,
                "minimax-m3",
                failure_threshold=3,
                cooldown=timedelta(minutes=30),
                clock=clock,
            )
            for _ in range(3):
                first.record_failure()

            second = SQLiteCircuitBreaker(
                db_path,
                "minimax-m3",
                failure_threshold=3,
                cooldown=timedelta(minutes=30),
                clock=clock,
            )
            self.assertFalse(second.allow_call())
            self.assertEqual(second.consecutive_failures, 3)

            current[0] += timedelta(minutes=31)
            self.assertTrue(second.allow_call())
            second.record_success()
            third = SQLiteCircuitBreaker(db_path, "minimax-m3", clock=clock)
            self.assertEqual(third.consecutive_failures, 0)
            self.assertTrue(third.allow_call())

    def test_save_feature_snapshot_rejects_empty_source_times(self):
        with temporary_database() as db_path:
            repository = SQLiteWarningRepository(db_path)

            with self.assertRaises(ValueError):
                repository.save_feature_snapshot(make_snapshot(source_times={}))

            with _db.connect(db_path) as connection:
                count = connection.execute(
                    "SELECT COUNT(*) AS row_count FROM market_warning_feature_snapshots"
                ).fetchone()["row_count"]
            self.assertEqual(count, 0)

    def test_save_decision_rejects_prediction_from_another_snapshot(self):
        with temporary_database() as db_path:
            repository = SQLiteWarningRepository(db_path)
            feature_snapshot_id = repository.save_feature_snapshot(make_snapshot())
            prediction_ids = repository.save_predictions(feature_snapshot_id, make_quant())
            other_snapshot_id = repository.save_feature_snapshot(
                make_snapshot(as_of_time=AS_OF + timedelta(minutes=1))
            )
            other_prediction_ids = repository.save_predictions(other_snapshot_id, make_quant())

            with self.assertRaises(ValueError):
                repository.save_decision(
                    feature_snapshot_id,
                    (prediction_ids[0], other_prediction_ids[1]),
                    None,
                    make_decision(),
                )

    def test_save_decision_rejects_unknown_prediction_id(self):
        with temporary_database() as db_path:
            repository = SQLiteWarningRepository(db_path)
            feature_snapshot_id = repository.save_feature_snapshot(make_snapshot())
            prediction_ids = repository.save_predictions(feature_snapshot_id, make_quant())

            with self.assertRaises(ValueError):
                repository.save_decision(
                    feature_snapshot_id,
                    (prediction_ids[0], 999999),
                    None,
                    make_decision(),
                )

    def test_alert_claim_reraises_invalid_decision_foreign_key(self):
        with temporary_database() as db_path:
            repository = SQLiteWarningRepository(db_path)

            with self.assertRaises(sqlite3.IntegrityError):
                repository.claim_alert("invalid-decision", 999999, "payload-sha")

    def test_register_model_replaces_active_model_for_market_and_horizon(self):
        with temporary_database() as db_path:
            repository = SQLiteWarningRepository(db_path)
            repository.register_model(
                {
                    "model_version": "quant-v1",
                    "market": Market.A_SHARE,
                    "horizon": "1d",
                    "feature_version": "features-v1",
                    "calibration_version": "cal-v1",
                    "training_cutoff": "2026-07-31",
                    "artifact_path": "models/quant-v1.json",
                    "artifact_sha256": "a" * 64,
                    "metrics": {"auc": 0.72},
                    "base_rate": 0.10,
                    "active": True,
                }
            )

            active = repository.load_active_model(Market.A_SHARE, "1d")

            self.assertIsNotNone(active)
            self.assertEqual(active["model_version"], "quant-v1")
            self.assertEqual(active["metrics"], {"auc": 0.72})
            self.assertTrue(active["active"])


if __name__ == "__main__":
    main()
