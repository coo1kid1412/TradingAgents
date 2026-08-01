"""SQLite repository contract tests for market-warning audit records."""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import TestCase, main

# Keep the brief's direct-script command rooted at this worktree.
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from tradingagents.harness import db as _db
from tradingagents.harness.market_warning.domain import (
    Evidence,
    FeatureSnapshot,
    FinalWarningDecision,
    LLMContextAssessment,
    Market,
    QuantRiskAssessment,
)
from tradingagents.harness.market_warning.adapters.sqlite_repository import SQLiteWarningRepository


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


def save_complete_decision(repository: SQLiteWarningRepository, snapshot: FeatureSnapshot | None = None) -> int:
    feature_snapshot_id = repository.save_feature_snapshot(snapshot or make_snapshot())
    prediction_id = repository.save_prediction(feature_snapshot_id, make_quant())
    reasoning_id = repository.save_reasoning(feature_snapshot_id, make_reasoning(), "reasoning-v1")
    return repository.save_decision(
        feature_snapshot_id,
        (prediction_id,),
        reasoning_id,
        make_decision(),
    )


class SQLiteWarningRepositoryTests(TestCase):
    def test_connect_initializes_all_six_warning_tables(self):
        with temporary_database() as db_path:
            with _db.connect(db_path) as connection:
                tables = {
                    row["name"]
                    for row in connection.execute(
                        "SELECT name FROM sqlite_master WHERE type = 'table'"
                    )
                }

        self.assertTrue(
            {
                "market_warning_feature_snapshots",
                "market_warning_predictions",
                "market_warning_reasoning",
                "market_warning_decisions",
                "market_warning_alerts",
                "market_warning_model_registry",
            }.issubset(tables)
        )

    def test_round_trip_persists_snapshot_two_horizons_reasoning_and_decision(self):
        with temporary_database() as db_path:
            repository = SQLiteWarningRepository(db_path)
            decision_id = save_complete_decision(repository)

            with _db.connect(db_path) as connection:
                snapshot_row = connection.execute(
                    "SELECT features_json, evidence_json, source_times_json "
                    "FROM market_warning_feature_snapshots WHERE id = ?",
                    (1,),
                ).fetchone()
                prediction_rows = connection.execute(
                    "SELECT horizon, probability, base_rate FROM market_warning_predictions "
                    "WHERE feature_snapshot_id = ? ORDER BY horizon",
                    (1,),
                ).fetchall()
                reasoning_row = connection.execute(
                    "SELECT structured_json FROM market_warning_reasoning WHERE id = ?",
                    (1,),
                ).fetchone()
                decision_row = connection.execute(
                    "SELECT model_version FROM market_warning_decisions WHERE id = ?",
                    (decision_id,),
                ).fetchone()

            self.assertGreater(decision_id, 0)
            self.assertEqual(json.loads(snapshot_row["features_json"])["breadth_pct"], 22.0)
            self.assertEqual(json.loads(snapshot_row["evidence_json"])[0]["evidence_id"], "breadth-1")
            self.assertEqual(json.loads(snapshot_row["source_times_json"]), {})
            self.assertEqual(
                [(row["horizon"], row["probability"], row["base_rate"]) for row in prediction_rows],
                [("1d", 0.35, 0.10), ("3d", 0.48, 0.16)],
            )
            self.assertEqual(json.loads(reasoning_row["structured_json"])["market_scenario"], "broad deleveraging")
            self.assertEqual(decision_row["model_version"], "quant-v1")
            self.assertEqual(repository.load_latest_decision(Market.A_SHARE), make_decision())
            self.assertEqual(
                repository.load_previous_decision(Market.A_SHARE, AS_OF + timedelta(minutes=1)),
                make_decision(),
            )

    def test_alert_claim_is_atomic_across_repository_instances(self):
        with temporary_database() as db_path:
            first_repository = SQLiteWarningRepository(db_path)
            decision_id = save_complete_decision(first_repository)
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
