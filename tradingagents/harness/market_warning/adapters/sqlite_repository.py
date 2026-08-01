"""SQLite persistence adapter for immutable market-warning domain records."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Mapping
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from tradingagents.harness import db as _db

from tradingagents.harness.market_warning.domain import (
    Evidence,
    FeatureSnapshot,
    FinalWarningDecision,
    LLMContextAssessment,
    Market,
    QuantRiskAssessment,
)


def _json_value(value: Any) -> Any:
    """Convert immutable domain values into JSON-compatible standard containers."""
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set, frozenset)):
        return [_json_value(item) for item in value]
    return value


def _json_dump(value: Any) -> str:
    return json.dumps(_json_value(value), ensure_ascii=False, separators=(",", ":"))


def _stored_time(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat()


def _evidence_json(evidence: tuple[Evidence, ...]) -> str:
    return _json_dump(
        [
            {
                "evidence_id": item.evidence_id,
                "group": item.group,
                "summary": item.summary,
                "value": item.value,
                "source": item.source,
                "as_of_time": item.as_of_time,
            }
            for item in evidence
        ]
    )


def _reasoning_json(assessment: LLMContextAssessment) -> str:
    return _json_dump(
        {
            "market_scenario": assessment.market_scenario,
            "causal_chain": assessment.causal_chain,
            "supporting_evidence_ids": assessment.supporting_evidence_ids,
            "conflicting_evidence_ids": assessment.conflicting_evidence_ids,
            "overlooked_risks": assessment.overlooked_risks,
            "recommended_risk_level": assessment.recommended_risk_level,
            "confidence": assessment.confidence,
            "action_reason": assessment.action_reason,
        }
    )


class SQLiteWarningRepository:
    """Persist warning inputs and conclusions using the harness SQLite lifecycle."""

    def __init__(self, db_path: Path | str | None = None) -> None:
        self._db_path = db_path

    def save_feature_snapshot(self, snapshot: FeatureSnapshot) -> int:
        if not snapshot.source_times:
            raise ValueError("source_times must not be empty when saving a feature snapshot")
        with _db.connect(self._db_path) as connection:
            cursor = connection.execute(
                "INSERT INTO market_warning_feature_snapshots "
                "(market, as_of_time, session_slot, feature_version, data_status, reliability_grade, "
                "features_json, evidence_json, source_times_json) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    snapshot.market.value,
                    _stored_time(snapshot.as_of_time),
                    snapshot.session_slot,
                    snapshot.feature_version,
                    snapshot.data_quality.value,
                    snapshot.reliability_grade,
                    _json_dump(dict(snapshot.features)),
                    _evidence_json(snapshot.evidence),
                    _json_dump(dict(snapshot.source_times)),
                ),
            )
            return int(cursor.lastrowid)

    def save_predictions(self, feature_snapshot_id: int, assessment: QuantRiskAssessment) -> tuple[int, int]:
        rows = (
            ("1d", assessment.crash_1d_probability, assessment.base_rate_1d),
            ("3d", assessment.crash_3d_probability, assessment.base_rate_3d),
        )
        with _db.connect(self._db_path) as connection:
            prediction_ids: list[int] = []
            for horizon, probability, base_rate in rows:
                cursor = connection.execute(
                    "INSERT INTO market_warning_predictions "
                    "(feature_snapshot_id, horizon, probability, base_rate, market_phase, "
                    "reliability_grade, model_version, calibration_version, top_contributors_json) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        feature_snapshot_id,
                        horizon,
                        probability,
                        base_rate,
                        assessment.market_phase.value,
                        assessment.reliability_grade,
                        assessment.model_version,
                        assessment.calibration_version,
                        _json_dump(list(assessment.top_contributors)),
                    ),
                )
                prediction_ids.append(int(cursor.lastrowid))
            return (prediction_ids[0], prediction_ids[1])

    def save_reasoning(
        self, feature_snapshot_id: int, assessment: LLMContextAssessment, model_name: str
    ) -> int:
        with _db.connect(self._db_path) as connection:
            cursor = connection.execute(
                "INSERT INTO market_warning_reasoning "
                "(feature_snapshot_id, model_name, reasoning_status, structured_json, error_class) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    feature_snapshot_id,
                    model_name,
                    assessment.reasoning_status,
                    _reasoning_json(assessment),
                    None,
                ),
            )
            return int(cursor.lastrowid)

    def save_decision(
        self,
        feature_snapshot_id: int,
        prediction_ids: tuple[int, ...],
        reasoning_id: int | None,
        decision: FinalWarningDecision,
    ) -> int:
        with _db.connect(self._db_path) as connection:
            if len(prediction_ids) != 2 or len(set(prediction_ids)) != 2:
                raise ValueError("prediction_ids must contain distinct 1d and 3d prediction IDs")
            placeholders = ", ".join("?" for _ in prediction_ids)
            predictions = connection.execute(
                "SELECT id, horizon, model_version FROM market_warning_predictions "
                f"WHERE feature_snapshot_id = ? AND id IN ({placeholders})",
                (feature_snapshot_id, *prediction_ids),
            ).fetchall()
            if len(predictions) != len(prediction_ids):
                raise ValueError("prediction_ids must all belong to feature_snapshot_id")
            if {row["horizon"] for row in predictions} != {"1d", "3d"}:
                raise ValueError("prediction_ids must reference one 1d and one 3d prediction")
            model_versions = {row["model_version"] for row in predictions}
            if len(model_versions) != 1:
                raise ValueError("prediction_ids must share one model_version")
            model_version = model_versions.pop()
            cursor = connection.execute(
                "INSERT INTO market_warning_decisions "
                "(feature_snapshot_id, prediction_ids_json, reasoning_id, baseline_level, final_level, "
                "transition, entry_gate, new_position_cap_pct, holding_action, push_required, "
                "data_status, reasons_json, valid_snapshot_count, retained_risk_level, model_version) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    feature_snapshot_id,
                    _json_dump(list(prediction_ids)),
                    reasoning_id,
                    decision.baseline_level.value,
                    decision.final_level.value,
                    decision.state_transition,
                    decision.entry_gate,
                    decision.new_position_cap_pct,
                    decision.holding_action,
                    int(decision.push_required),
                    decision.data_status.value,
                    _json_dump(list(decision.decision_reasons)),
                    decision.valid_snapshot_count,
                    decision.retained_risk_level.value if decision.retained_risk_level is not None else None,
                    model_version,
                ),
            )
            return int(cursor.lastrowid)

    def load_latest_decision(
        self, market: Market, as_of_time: datetime | None = None
    ) -> FinalWarningDecision | None:
        filter_time = _stored_time(as_of_time) if as_of_time is not None else None
        with _db.connect(self._db_path) as connection:
            row = connection.execute(
                "SELECT d.baseline_level, d.final_level, d.transition, d.entry_gate, "
                "d.new_position_cap_pct, d.holding_action, d.push_required, d.data_status, d.reasons_json, "
                "d.valid_snapshot_count, d.retained_risk_level "
                "FROM market_warning_decisions AS d "
                "INNER JOIN market_warning_feature_snapshots AS s ON s.id = d.feature_snapshot_id "
                "WHERE s.market = ? AND (? IS NULL OR s.as_of_time <= ?) "
                "ORDER BY s.as_of_time DESC, d.id DESC LIMIT 1",
                (Market(market).value, filter_time, filter_time),
            ).fetchone()
        return self._decision_from_row(row) if row is not None else None

    def load_previous_decision(self, market: Market, before_time: datetime) -> FinalWarningDecision | None:
        with _db.connect(self._db_path) as connection:
            row = connection.execute(
                "SELECT d.baseline_level, d.final_level, d.transition, d.entry_gate, "
                "d.new_position_cap_pct, d.holding_action, d.push_required, d.data_status, d.reasons_json, "
                "d.valid_snapshot_count, d.retained_risk_level "
                "FROM market_warning_decisions AS d "
                "INNER JOIN market_warning_feature_snapshots AS s ON s.id = d.feature_snapshot_id "
                "WHERE s.market = ? AND s.as_of_time < ? "
                "ORDER BY s.as_of_time DESC, d.id DESC LIMIT 1",
                (Market(market).value, _stored_time(before_time)),
            ).fetchone()
        return self._decision_from_row(row) if row is not None else None

    def claim_alert(self, idempotency_key: str, decision_id: int, payload_hash: str) -> bool:
        try:
            with _db.connect(self._db_path) as connection:
                connection.execute(
                    "INSERT INTO market_warning_alerts "
                    "(idempotency_key, decision_id, payload_hash, push_status, sent_at, error_summary) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (idempotency_key, decision_id, payload_hash, "claimed", None, None),
                )
        except sqlite3.IntegrityError as exc:
            if "UNIQUE constraint failed: market_warning_alerts.idempotency_key" in str(exc):
                return False
            raise
        return True

    def finish_alert(self, idempotency_key: str, status: str, error_summary: str | None = None) -> None:
        with _db.connect(self._db_path) as connection:
            connection.execute(
                "UPDATE market_warning_alerts "
                "SET push_status = ?, sent_at = CASE WHEN ? = 'sent' THEN CURRENT_TIMESTAMP ELSE sent_at END, "
                "error_summary = ? WHERE idempotency_key = ?",
                (status, status, error_summary, idempotency_key),
            )

    def register_model(self, record: Mapping[str, Any]) -> None:
        values = dict(record)
        market = Market(values["market"]).value
        horizon = str(values["horizon"])
        active = int(bool(values.get("active", False)))
        with _db.connect(self._db_path) as connection:
            if active:
                connection.execute(
                    "UPDATE market_warning_model_registry SET active = ? "
                    "WHERE market = ? AND horizon = ?",
                    (0, market, horizon),
                )
            connection.execute(
                "INSERT INTO market_warning_model_registry "
                "(model_version, market, horizon, feature_version, calibration_version, training_cutoff, "
                "artifact_path, artifact_sha256, metrics_json, base_rate, active) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(model_version, market, horizon) DO UPDATE SET "
                "feature_version = excluded.feature_version, calibration_version = excluded.calibration_version, "
                "training_cutoff = excluded.training_cutoff, artifact_path = excluded.artifact_path, "
                "artifact_sha256 = excluded.artifact_sha256, metrics_json = excluded.metrics_json, "
                "base_rate = excluded.base_rate, active = excluded.active, created_at = CURRENT_TIMESTAMP",
                (
                    values["model_version"],
                    market,
                    horizon,
                    values["feature_version"],
                    values["calibration_version"],
                    _json_value(values["training_cutoff"]),
                    values["artifact_path"],
                    values["artifact_sha256"],
                    _json_dump(values["metrics"]),
                    values["base_rate"],
                    active,
                ),
            )

    def load_active_model(self, market: Market, horizon: str) -> dict[str, Any] | None:
        with _db.connect(self._db_path) as connection:
            row = connection.execute(
                "SELECT model_version, market, horizon, feature_version, calibration_version, training_cutoff, "
                "artifact_path, artifact_sha256, metrics_json, base_rate, active, created_at "
                "FROM market_warning_model_registry WHERE market = ? AND horizon = ? AND active = ? "
                "ORDER BY created_at DESC, model_version DESC LIMIT 1",
                (Market(market).value, horizon, 1),
            ).fetchone()
        if row is None:
            return None
        return {
            "model_version": row["model_version"],
            "market": Market(row["market"]),
            "horizon": row["horizon"],
            "feature_version": row["feature_version"],
            "calibration_version": row["calibration_version"],
            "training_cutoff": row["training_cutoff"],
            "artifact_path": row["artifact_path"],
            "artifact_sha256": row["artifact_sha256"],
            "metrics": json.loads(row["metrics_json"]),
            "base_rate": row["base_rate"],
            "active": bool(row["active"]),
            "created_at": row["created_at"],
        }

    @staticmethod
    def _decision_from_row(row: sqlite3.Row) -> FinalWarningDecision:
        return FinalWarningDecision(
            baseline_level=row["baseline_level"],
            final_level=row["final_level"],
            state_transition=row["transition"],
            entry_gate=row["entry_gate"],
            new_position_cap_pct=row["new_position_cap_pct"],
            holding_action=row["holding_action"],
            push_required=bool(row["push_required"]),
            decision_reasons=tuple(json.loads(row["reasons_json"])),
            data_status=row["data_status"],
            valid_snapshot_count=row["valid_snapshot_count"],
            retained_risk_level=row["retained_risk_level"],
        )
