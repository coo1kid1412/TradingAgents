"""SQLite persistence adapter for immutable market-warning domain records."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Mapping
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from tradingagents.harness import db as _db

from tradingagents.harness.market_warning.domain import (
    DataStatus,
    DecisionSource,
    Evidence,
    FeatureSnapshot,
    FinalWarningDecision,
    LLMContextAssessment,
    Market,
    MarketPhase,
    QuantRiskAssessment,
    RiskLevel,
    RuleRiskAssessment,
    RunnerResult,
    TriggeredRule,
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


def _triggered_rules_json(assessment: RuleRiskAssessment) -> str:
    return _json_dump(
        [
            {
                "rule_id": item.rule_id,
                "layer": item.layer,
                "severity_points": item.severity_points,
                "observed_value": item.observed_value,
                "threshold_description": item.threshold_description,
                "evidence_ids": item.evidence_ids,
            }
            for item in assessment.triggered_rules
        ]
    )


def _context_from_row(row) -> LLMContextAssessment:
    payload = json.loads(row["structured_json"])
    return LLMContextAssessment(
        market_scenario=payload["market_scenario"],
        causal_chain=tuple(payload["causal_chain"]),
        supporting_evidence_ids=tuple(payload["supporting_evidence_ids"]),
        conflicting_evidence_ids=tuple(payload["conflicting_evidence_ids"]),
        overlooked_risks=tuple(payload["overlooked_risks"]),
        recommended_risk_level=RiskLevel(payload["recommended_risk_level"]),
        confidence=payload["confidence"],
        action_reason=payload["action_reason"],
        reasoning_status=row["reasoning_status"],
        error_class=row["error_class"],
    )


class SQLiteCircuitBreaker:
    """Small persistent breaker shared by independent cron processes."""

    def __init__(
        self,
        db_path: Path | str | None,
        breaker_key: str,
        failure_threshold: int = 3,
        cooldown: timedelta = timedelta(minutes=30),
        *,
        clock=None,
    ) -> None:
        if not isinstance(breaker_key, str) or not breaker_key.strip():
            raise ValueError("breaker_key must not be empty")
        if isinstance(failure_threshold, bool) or not isinstance(failure_threshold, int) or failure_threshold < 1:
            raise ValueError("failure_threshold must be a positive integer")
        if not isinstance(cooldown, timedelta) or cooldown <= timedelta(0):
            raise ValueError("cooldown must be positive")
        self._db_path = db_path
        self._breaker_key = breaker_key.strip()
        self._failure_threshold = failure_threshold
        self._cooldown = cooldown
        self._clock = clock or (lambda: datetime.now(timezone.utc))

    def _now(self) -> datetime:
        value = self._clock()
        if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("circuit-breaker clock must be timezone-aware")
        return value.astimezone(timezone.utc)

    def _row(self):
        with _db.connect(self._db_path) as connection:
            return connection.execute(
                "SELECT consecutive_failures, open_until FROM market_warning_circuit_breakers "
                "WHERE breaker_key = ?",
                (self._breaker_key,),
            ).fetchone()

    @property
    def consecutive_failures(self) -> int:
        row = self._row()
        return int(row["consecutive_failures"]) if row is not None else 0

    def allow_call(self) -> bool:
        row = self._row()
        if row is None or row["open_until"] is None:
            return True
        return self._now() >= datetime.fromisoformat(row["open_until"])

    def record_success(self) -> None:
        now = self._now().isoformat()
        with _db.connect(self._db_path) as connection:
            connection.execute(
                "INSERT INTO market_warning_circuit_breakers "
                "(breaker_key, consecutive_failures, open_until, updated_at) VALUES (?, 0, NULL, ?) "
                "ON CONFLICT(breaker_key) DO UPDATE SET consecutive_failures = 0, "
                "open_until = NULL, updated_at = excluded.updated_at",
                (self._breaker_key, now),
            )

    def record_failure(self) -> None:
        now = self._now()
        with _db.connect(self._db_path) as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT consecutive_failures, open_until FROM market_warning_circuit_breakers "
                "WHERE breaker_key = ?",
                (self._breaker_key,),
            ).fetchone()
            failures = int(row["consecutive_failures"]) if row is not None else 0
            open_until = (
                datetime.fromisoformat(row["open_until"])
                if row is not None and row["open_until"] is not None
                else None
            )
            if open_until is not None and now >= open_until:
                failures = self._failure_threshold - 1
            failures += 1
            next_open_until = (
                (now + self._cooldown).isoformat()
                if failures >= self._failure_threshold
                else None
            )
            connection.execute(
                "INSERT INTO market_warning_circuit_breakers "
                "(breaker_key, consecutive_failures, open_until, updated_at) VALUES (?, ?, ?, ?) "
                "ON CONFLICT(breaker_key) DO UPDATE SET "
                "consecutive_failures = excluded.consecutive_failures, "
                "open_until = excluded.open_until, updated_at = excluded.updated_at",
                (self._breaker_key, failures, next_open_until, now.isoformat()),
            )


class SQLiteWarningRepository:
    """Persist warning inputs and conclusions using the harness SQLite lifecycle."""

    def __init__(self, db_path: Path | str | None = None) -> None:
        self._db_path = db_path

    def circuit_breaker(
        self,
        breaker_key: str,
        failure_threshold: int = 3,
        cooldown: timedelta = timedelta(minutes=30),
    ) -> SQLiteCircuitBreaker:
        return SQLiteCircuitBreaker(
            self._db_path,
            breaker_key,
            failure_threshold=failure_threshold,
            cooldown=cooldown,
        )

    def acquire_lease(
        self,
        lease_key: str,
        owner_id: str,
        now: datetime,
        duration: timedelta,
    ) -> bool:
        if not lease_key.strip() or not owner_id.strip():
            raise ValueError("lease_key and owner_id must not be empty")
        if now.tzinfo is None or now.utcoffset() is None:
            raise ValueError("now must be timezone-aware")
        if duration <= timedelta(0):
            raise ValueError("duration must be positive")
        acquired_at = _stored_time(now)
        expires_at = _stored_time(now + duration)
        with _db.connect(self._db_path) as connection:
            cursor = connection.execute(
                "INSERT INTO market_warning_leases "
                "(lease_key, owner_id, acquired_at, expires_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?) "
                "ON CONFLICT(lease_key) DO UPDATE SET "
                "owner_id = excluded.owner_id, acquired_at = excluded.acquired_at, "
                "expires_at = excluded.expires_at, updated_at = excluded.updated_at "
                "WHERE market_warning_leases.expires_at <= excluded.acquired_at",
                (lease_key, owner_id, acquired_at, expires_at, acquired_at),
            )
        return cursor.rowcount == 1

    def release_lease(self, lease_key: str, owner_id: str) -> bool:
        with _db.connect(self._db_path) as connection:
            cursor = connection.execute(
                "DELETE FROM market_warning_leases WHERE lease_key = ? AND owner_id = ?",
                (lease_key, owner_id),
            )
        return cursor.rowcount == 1

    def record_run(
        self,
        *,
        market: Market,
        as_of_time: datetime,
        session_slot: str,
        mode: str,
        started_at: datetime,
        finished_at: datetime,
        status: str,
        error_class: str | None,
        overlap_skipped: bool,
        llm_calls: int,
    ) -> int:
        for name, value in (("as_of_time", as_of_time), ("started_at", started_at), ("finished_at", finished_at)):
            if value.tzinfo is None or value.utcoffset() is None:
                raise ValueError(f"{name} must be timezone-aware")
        if finished_at < started_at:
            raise ValueError("finished_at must not precede started_at")
        if isinstance(llm_calls, bool) or not isinstance(llm_calls, int) or llm_calls < 0:
            raise ValueError("llm_calls must be a non-negative integer")
        latency_ms = (finished_at - started_at).total_seconds() * 1000.0
        with _db.connect(self._db_path) as connection:
            cursor = connection.execute(
                "INSERT INTO market_warning_runs "
                "(market, as_of_time, session_slot, mode, started_at, finished_at, latency_ms, "
                "status, error_class, overlap_skipped, llm_calls) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    Market(market).value,
                    _stored_time(as_of_time),
                    session_slot,
                    mode,
                    _stored_time(started_at),
                    _stored_time(finished_at),
                    latency_ms,
                    status,
                    error_class,
                    int(overlap_skipped),
                    llm_calls,
                ),
            )
            return int(cursor.lastrowid)

    def record_schedule_outcome(
        self,
        market: Market,
        mode: str,
        as_of_time: datetime,
        *,
        succeeded: bool,
        failure_threshold: int = 3,
    ) -> dict[str, Any]:
        if as_of_time.tzinfo is None or as_of_time.utcoffset() is None:
            raise ValueError("as_of_time must be timezone-aware")
        if failure_threshold < 1:
            raise ValueError("failure_threshold must be positive")
        market_value = Market(market).value
        outcome_at = _stored_time(as_of_time)
        with _db.connect(self._db_path) as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT consecutive_failures, incident_started_at "
                "FROM market_warning_failure_streaks WHERE market = ? AND mode = ?",
                (market_value, mode),
            ).fetchone()
            if succeeded:
                failures = 0
                incident_started_at = None
            else:
                failures = (int(row["consecutive_failures"]) if row is not None else 0) + 1
                incident_started_at = (
                    row["incident_started_at"]
                    if row is not None and row["incident_started_at"]
                    else outcome_at
                )
            connection.execute(
                "INSERT INTO market_warning_failure_streaks "
                "(market, mode, consecutive_failures, incident_started_at, last_outcome_at) "
                "VALUES (?, ?, ?, ?, ?) "
                "ON CONFLICT(market, mode) DO UPDATE SET "
                "consecutive_failures = excluded.consecutive_failures, "
                "incident_started_at = excluded.incident_started_at, "
                "last_outcome_at = excluded.last_outcome_at",
                (market_value, mode, failures, incident_started_at, outcome_at),
            )
        return {
            "consecutive_failures": failures,
            "incident_started_at": incident_started_at,
            "alert_due": not succeeded and failures >= failure_threshold,
        }

    def claim_system_alert(
        self,
        idempotency_key: str,
        payload_hash: str,
        *,
        retry_failed: bool = False,
    ) -> bool:
        try:
            with _db.connect(self._db_path) as connection:
                connection.execute(
                    "INSERT INTO market_warning_system_alerts "
                    "(idempotency_key, payload_hash, push_status, sent_at, error_summary) "
                    "VALUES (?, ?, 'claimed', NULL, NULL)",
                    (idempotency_key, payload_hash),
                )
        except sqlite3.IntegrityError as exc:
            if "UNIQUE constraint failed: market_warning_system_alerts.idempotency_key" not in str(exc):
                raise
            if not retry_failed:
                return False
            with _db.connect(self._db_path) as connection:
                cursor = connection.execute(
                    "UPDATE market_warning_system_alerts SET payload_hash = ?, "
                    "push_status = 'claimed', sent_at = NULL, error_summary = NULL "
                    "WHERE idempotency_key = ? AND push_status = 'failed'",
                    (payload_hash, idempotency_key),
                )
            return cursor.rowcount == 1
        return True

    def finish_system_alert(
        self,
        idempotency_key: str,
        status: str,
        error_summary: str | None = None,
    ) -> None:
        with _db.connect(self._db_path) as connection:
            connection.execute(
                "UPDATE market_warning_system_alerts SET push_status = ?, "
                "sent_at = CASE WHEN ? = 'sent' THEN CURRENT_TIMESTAMP ELSE sent_at END, "
                "error_summary = ? WHERE idempotency_key = ?",
                (status, status, error_summary, idempotency_key),
            )

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

    def save_rule_assessment(
        self, feature_snapshot_id: int, assessment: RuleRiskAssessment
    ) -> int:
        with _db.connect(self._db_path) as connection:
            snapshot = connection.execute(
                "SELECT market, as_of_time FROM market_warning_feature_snapshots WHERE id = ?",
                (feature_snapshot_id,),
            ).fetchone()
            if snapshot is None:
                raise ValueError("feature_snapshot_id does not exist")
            if snapshot["market"] != assessment.market.value:
                raise ValueError("rule assessment market must match feature snapshot")
            if datetime.fromisoformat(snapshot["as_of_time"]) != assessment.as_of_time.astimezone(timezone.utc):
                raise ValueError("rule assessment as_of_time must match feature snapshot")
            cursor = connection.execute(
                "INSERT INTO market_warning_rule_assessments "
                "(feature_snapshot_id, engine_version, manifest_sha256, risk_level, risk_score, "
                "market_phase, triggered_rules_json, missing_optional_groups_json, reliability_grade, "
                "evaluation_latency_ms) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    feature_snapshot_id,
                    assessment.engine_version,
                    assessment.manifest_sha256,
                    assessment.risk_level.value,
                    assessment.risk_score,
                    assessment.market_phase.value,
                    _triggered_rules_json(assessment),
                    _json_dump(assessment.missing_optional_groups),
                    assessment.reliability_grade,
                    assessment.evaluation_latency_ms,
                ),
            )
            return int(cursor.lastrowid)

    def load_previous_rule_assessment(
        self, market: Market, before_time: datetime
    ) -> RuleRiskAssessment | None:
        with _db.connect(self._db_path) as connection:
            row = connection.execute(
                "SELECT s.market, s.as_of_time, r.engine_version, r.manifest_sha256, "
                "r.risk_level, r.risk_score, r.market_phase, r.triggered_rules_json, "
                "r.missing_optional_groups_json, r.reliability_grade, r.evaluation_latency_ms "
                "FROM market_warning_rule_assessments AS r "
                "INNER JOIN market_warning_feature_snapshots AS s ON s.id = r.feature_snapshot_id "
                "WHERE s.market = ? AND s.as_of_time < ? "
                "ORDER BY s.as_of_time DESC, r.id DESC LIMIT 1",
                (Market(market).value, _stored_time(before_time)),
            ).fetchone()
        return self._rule_assessment_from_row(row) if row is not None else None

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
                    assessment.error_class,
                ),
            )
            return int(cursor.lastrowid)

    def save_reasoning_for_decision(
        self,
        decision_id: int,
        assessment: LLMContextAssessment,
        model_name: str,
    ) -> int:
        with _db.connect(self._db_path) as connection:
            decision = connection.execute(
                "SELECT feature_snapshot_id FROM market_warning_decisions WHERE id = ?",
                (decision_id,),
            ).fetchone()
            if decision is None:
                raise ValueError("decision_id does not exist")
            cursor = connection.execute(
                "INSERT INTO market_warning_reasoning "
                "(feature_snapshot_id, model_name, reasoning_status, structured_json, error_class) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    decision["feature_snapshot_id"],
                    model_name,
                    assessment.reasoning_status,
                    _reasoning_json(assessment),
                    assessment.error_class,
                ),
            )
            reasoning_id = int(cursor.lastrowid)
            connection.execute(
                "UPDATE market_warning_decisions SET reasoning_id = ? WHERE id = ?",
                (reasoning_id, decision_id),
            )
            return reasoning_id

    def attach_reasoning(self, decision_id: int, reasoning_id: int) -> None:
        with _db.connect(self._db_path) as connection:
            row = connection.execute(
                "SELECT d.feature_snapshot_id AS decision_snapshot_id, "
                "r.feature_snapshot_id AS reasoning_snapshot_id "
                "FROM market_warning_decisions AS d "
                "INNER JOIN market_warning_reasoning AS r ON r.id = ? "
                "WHERE d.id = ?",
                (reasoning_id, decision_id),
            ).fetchone()
            if row is None:
                raise ValueError("decision_id and reasoning_id must exist")
            if row["decision_snapshot_id"] != row["reasoning_snapshot_id"]:
                raise ValueError("reasoning must belong to the decision feature snapshot")
            connection.execute(
                "UPDATE market_warning_decisions SET reasoning_id = ? WHERE id = ?",
                (reasoning_id, decision_id),
            )

    def load_latest_reasoning(
        self,
        market: Market,
        before_time: datetime,
        model_name: str = "MiniMax-M3",
    ) -> LLMContextAssessment | None:
        with _db.connect(self._db_path) as connection:
            row = connection.execute(
                "SELECT r.reasoning_status, r.structured_json, r.error_class "
                "FROM market_warning_reasoning AS r "
                "INNER JOIN market_warning_feature_snapshots AS s "
                "ON s.id = r.feature_snapshot_id "
                "WHERE s.market = ? AND s.as_of_time < ? AND r.model_name = ? "
                "ORDER BY s.as_of_time DESC, r.id DESC LIMIT 1",
                (Market(market).value, _stored_time(before_time), model_name),
            ).fetchone()
        return _context_from_row(row) if row is not None else None

    def save_decision(
        self,
        feature_snapshot_id: int,
        prediction_ids: tuple[int, ...],
        reasoning_id: int | None,
        decision: FinalWarningDecision,
        *,
        rule_assessment_id: int | None = None,
        shadow_prediction_ids: tuple[int, ...] = (),
    ) -> int:
        with _db.connect(self._db_path) as connection:
            def validate_predictions(ids: tuple[int, ...], field_name: str):
                if len(ids) != 2 or len(set(ids)) != 2:
                    raise ValueError(f"{field_name} must contain distinct 1d and 3d prediction IDs")
                placeholders = ", ".join("?" for _ in ids)
                rows = connection.execute(
                    "SELECT id, horizon, model_version FROM market_warning_predictions "
                    f"WHERE feature_snapshot_id = ? AND id IN ({placeholders})",
                    (feature_snapshot_id, *ids),
                ).fetchall()
                if len(rows) != len(ids):
                    raise ValueError(f"{field_name} must all belong to feature_snapshot_id")
                if {row["horizon"] for row in rows} != {"1d", "3d"}:
                    raise ValueError(f"{field_name} must reference one 1d and one 3d prediction")
                versions = {row["model_version"] for row in rows}
                if len(versions) != 1:
                    raise ValueError(f"{field_name} must share one model_version")
                return versions.pop()

            model_version = None
            if decision.decision_source == DecisionSource.MODEL:
                if rule_assessment_id is not None or shadow_prediction_ids:
                    raise ValueError("model decisions cannot reference rule or shadow assessments")
                model_version = validate_predictions(tuple(prediction_ids), "prediction_ids")
            else:
                if prediction_ids:
                    raise ValueError("rule decisions cannot store primary model predictions")
                if rule_assessment_id is None:
                    raise ValueError("rule decisions require rule_assessment_id")
                rule_row = connection.execute(
                    "SELECT engine_version FROM market_warning_rule_assessments "
                    "WHERE id = ? AND feature_snapshot_id = ?",
                    (rule_assessment_id, feature_snapshot_id),
                ).fetchone()
                if rule_row is None:
                    raise ValueError("rule_assessment_id must belong to feature_snapshot_id")
                model_version = rule_row["engine_version"]
                if shadow_prediction_ids:
                    validate_predictions(tuple(shadow_prediction_ids), "shadow_prediction_ids")
            cursor = connection.execute(
                "INSERT INTO market_warning_decisions "
                "(feature_snapshot_id, prediction_ids_json, reasoning_id, baseline_level, final_level, "
                "transition, entry_gate, new_position_cap_pct, holding_action, push_required, "
                "data_status, reasons_json, valid_snapshot_count, retained_risk_level, decision_source, "
                "rule_assessment_id, shadow_prediction_ids_json, model_version) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
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
                    decision.decision_source.value,
                    rule_assessment_id,
                    _json_dump(list(shadow_prediction_ids)),
                    model_version,
                ),
            )
            return int(cursor.lastrowid)

    def load_evaluation(self, market: Market, as_of_time: datetime) -> RunnerResult | None:
        """Rebuild one already-persisted evaluation for idempotent retries."""

        with _db.connect(self._db_path) as connection:
            row = connection.execute(
                "SELECT s.id AS snapshot_id, s.market, s.as_of_time, s.session_slot, "
                "s.feature_version, s.data_status AS snapshot_status, s.reliability_grade AS snapshot_grade, "
                "s.features_json, s.evidence_json, s.source_times_json, "
                "d.id AS decision_id, d.prediction_ids_json, d.reasoning_id, d.baseline_level, "
                "d.final_level, d.transition, d.entry_gate, d.new_position_cap_pct, "
                "d.holding_action, d.push_required, d.data_status AS data_status, "
                "d.reasons_json, d.valid_snapshot_count, d.retained_risk_level, "
                "d.decision_source, d.rule_assessment_id, d.shadow_prediction_ids_json "
                "FROM market_warning_feature_snapshots AS s "
                "INNER JOIN market_warning_decisions AS d ON d.feature_snapshot_id = s.id "
                "WHERE s.market = ? AND s.as_of_time = ? "
                "ORDER BY s.id DESC LIMIT 1",
                (Market(market).value, _stored_time(as_of_time)),
            ).fetchone()
            if row is None:
                return None
            prediction_ids = tuple(json.loads(row["prediction_ids_json"]))
            source = DecisionSource(row["decision_source"])

            def load_predictions(ids: tuple[int, ...]):
                if not ids:
                    return ()
                placeholders = ", ".join("?" for _ in ids)
                return connection.execute(
                    "SELECT id, horizon, probability, base_rate, market_phase, reliability_grade, "
                    "model_version, calibration_version, top_contributors_json "
                    f"FROM market_warning_predictions WHERE id IN ({placeholders}) "
                    "ORDER BY horizon",
                    ids,
                ).fetchall()

            predictions = load_predictions(prediction_ids)
            shadow_prediction_ids = tuple(json.loads(row["shadow_prediction_ids_json"]))
            shadow_predictions = load_predictions(shadow_prediction_ids)
            rule_row = None
            if row["rule_assessment_id"] is not None:
                rule_row = connection.execute(
                    "SELECT engine_version, manifest_sha256, risk_level, risk_score, market_phase, "
                    "triggered_rules_json, missing_optional_groups_json, reliability_grade, "
                    "evaluation_latency_ms FROM market_warning_rule_assessments "
                    "WHERE id = ? AND feature_snapshot_id = ?",
                    (row["rule_assessment_id"], row["snapshot_id"]),
                ).fetchone()
            reasoning = None
            if row["reasoning_id"] is not None:
                reasoning = connection.execute(
                    "SELECT reasoning_status, structured_json, error_class "
                    "FROM market_warning_reasoning WHERE id = ?",
                    (row["reasoning_id"],),
                ).fetchone()
        if source == DecisionSource.MODEL:
            if len(predictions) != 2 or {item["horizon"] for item in predictions} != {"1d", "3d"}:
                return None
            if rule_row is not None or shadow_predictions:
                return None
        elif predictions or rule_row is None:
            return None
        if shadow_predictions and (
            len(shadow_predictions) != 2
            or {item["horizon"] for item in shadow_predictions} != {"1d", "3d"}
        ):
            return None

        evidence_rows = json.loads(row["evidence_json"])
        snapshot = FeatureSnapshot(
            market=row["market"],
            as_of_time=datetime.fromisoformat(row["as_of_time"]),
            session_slot=row["session_slot"],
            feature_version=row["feature_version"],
            features=json.loads(row["features_json"]),
            evidence=tuple(
                Evidence(
                    evidence_id=item["evidence_id"],
                    group=item["group"],
                    summary=item["summary"],
                    value=item.get("value"),
                    source=item.get("source"),
                    as_of_time=(
                        datetime.fromisoformat(item["as_of_time"])
                        if item.get("as_of_time")
                        else None
                    ),
                )
                for item in evidence_rows
            ),
            data_quality=DataStatus(row["snapshot_status"]),
            reliability_grade=row["snapshot_grade"],
            source_times={
                key: datetime.fromisoformat(value)
                for key, value in json.loads(row["source_times_json"]).items()
            },
        )
        def assessment_from_predictions(rows) -> QuantRiskAssessment | None:
            if not rows:
                return None
            by_horizon = {item["horizon"]: item for item in rows}
            first = by_horizon["1d"]
            third = by_horizon["3d"]
            if (
                first["market_phase"] != third["market_phase"]
                or first["model_version"] != third["model_version"]
                or first["calibration_version"] != third["calibration_version"]
            ):
                raise ValueError("persisted prediction pair is inconsistent")
            return QuantRiskAssessment(
                crash_1d_probability=first["probability"],
                crash_3d_probability=third["probability"],
                market_phase=MarketPhase(first["market_phase"]),
                base_rate_1d=first["base_rate"],
                base_rate_3d=third["base_rate"],
                reliability_grade=first["reliability_grade"],
                model_version=first["model_version"],
                calibration_version=first["calibration_version"],
                top_contributors=tuple(json.loads(first["top_contributors_json"])),
            )

        try:
            quant = assessment_from_predictions(predictions)
            shadow_quant = assessment_from_predictions(shadow_predictions)
        except ValueError:
            return None
        rule_assessment = None
        if rule_row is not None:
            rule_assessment = RuleRiskAssessment(
                market=snapshot.market,
                as_of_time=snapshot.as_of_time,
                engine_version=rule_row["engine_version"],
                manifest_sha256=rule_row["manifest_sha256"],
                risk_level=rule_row["risk_level"],
                risk_score=rule_row["risk_score"],
                market_phase=rule_row["market_phase"],
                triggered_rules=tuple(
                    TriggeredRule(**item)
                    for item in json.loads(rule_row["triggered_rules_json"])
                ),
                missing_optional_groups=tuple(
                    json.loads(rule_row["missing_optional_groups_json"])
                ),
                reliability_grade=rule_row["reliability_grade"],
                evaluation_latency_ms=rule_row["evaluation_latency_ms"],
            )
        context = None
        if reasoning is not None:
            context = _context_from_row(reasoning)
        decision = self._decision_from_row(row)
        return RunnerResult(
            market=Market(row["market"]),
            as_of_time=datetime.fromisoformat(row["as_of_time"]),
            session_slot=row["session_slot"],
            feature_snapshot=snapshot,
            quant_assessment=quant,
            rule_assessment=rule_assessment,
            shadow_quant_assessment=shadow_quant,
            context_assessment=context,
            decision=decision,
            decision_id=int(row["decision_id"]),
        )

    def load_latest_decision(
        self, market: Market, as_of_time: datetime | None = None
    ) -> FinalWarningDecision | None:
        filter_time = _stored_time(as_of_time) if as_of_time is not None else None
        with _db.connect(self._db_path) as connection:
            row = connection.execute(
                "SELECT d.baseline_level, d.final_level, d.transition, d.entry_gate, "
                "d.new_position_cap_pct, d.holding_action, d.push_required, d.data_status, d.reasons_json, "
                "d.valid_snapshot_count, d.retained_risk_level, d.decision_source "
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
                "d.valid_snapshot_count, d.retained_risk_level, d.decision_source "
                "FROM market_warning_decisions AS d "
                "INNER JOIN market_warning_feature_snapshots AS s ON s.id = d.feature_snapshot_id "
                "WHERE s.market = ? AND s.as_of_time < ? "
                "ORDER BY s.as_of_time DESC, d.id DESC LIMIT 1",
                (Market(market).value, _stored_time(before_time)),
            ).fetchone()
        return self._decision_from_row(row) if row is not None else None

    def claim_alert(
        self,
        idempotency_key: str,
        decision_id: int,
        payload_hash: str,
        *,
        retry_failed: bool = False,
    ) -> bool:
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
                if retry_failed:
                    with _db.connect(self._db_path) as connection:
                        cursor = connection.execute(
                            "UPDATE market_warning_alerts SET decision_id = ?, payload_hash = ?, "
                            "push_status = 'claimed', sent_at = NULL, error_summary = NULL "
                            "WHERE idempotency_key = ? AND push_status = 'failed'",
                            (decision_id, payload_hash, idempotency_key),
                        )
                    return cursor.rowcount == 1
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

    def load_alert_status(self, idempotency_key: str) -> str | None:
        with _db.connect(self._db_path) as connection:
            row = connection.execute(
                "SELECT push_status FROM market_warning_alerts WHERE idempotency_key = ?",
                (idempotency_key,),
            ).fetchone()
        return str(row["push_status"]) if row is not None else None

    def register_rule_engine(self, record: Mapping[str, Any]) -> None:
        values = dict(record)
        market = Market(values["market"]).value
        digest = str(values["manifest_sha256"])
        if len(digest) != 64:
            raise ValueError("manifest_sha256 must be a 64-character digest")
        with _db.connect(self._db_path) as connection:
            connection.execute(
                "INSERT INTO market_warning_rule_registry "
                "(engine_version, market, manifest_sha256, metrics_json) VALUES (?, ?, ?, ?) "
                "ON CONFLICT(engine_version, market) DO UPDATE SET "
                "manifest_sha256 = excluded.manifest_sha256, metrics_json = excluded.metrics_json, "
                "updated_at = CURRENT_TIMESTAMP",
                (
                    values["engine_version"],
                    market,
                    digest,
                    _json_dump(values.get("metrics", {})),
                ),
            )

    def activate_rule_engine(self, engine_version: str, mode: str) -> dict[str, Any]:
        column = self._rule_activation_column(mode)
        with _db.connect(self._db_path) as connection:
            connection.execute("BEGIN IMMEDIATE")
            target = connection.execute(
                "SELECT market FROM market_warning_rule_registry WHERE engine_version = ?",
                (engine_version,),
            ).fetchall()
            if len(target) != 1:
                raise ValueError("rule engine version must identify exactly one registered market")
            market = target[0]["market"]
            connection.execute(
                f"UPDATE market_warning_rule_registry SET {column} = 0, updated_at = CURRENT_TIMESTAMP "
                "WHERE market = ?",
                (market,),
            )
            cursor = connection.execute(
                f"UPDATE market_warning_rule_registry SET {column} = 1, updated_at = CURRENT_TIMESTAMP "
                "WHERE engine_version = ? AND market = ?",
                (engine_version, market),
            )
            if cursor.rowcount != 1:
                raise ValueError("rule engine activation failed")
        record = self.load_active_rule_engine(Market(market), mode)
        if record is None:
            raise ValueError("rule engine activation was not persisted")
        return record

    def deactivate_rule_engine(self, market: Market, mode: str) -> None:
        column = self._rule_activation_column(mode)
        with _db.connect(self._db_path) as connection:
            connection.execute(
                f"UPDATE market_warning_rule_registry SET {column} = 0, "
                "updated_at = CURRENT_TIMESTAMP WHERE market = ?",
                (Market(market).value,),
            )

    def load_active_rule_engine(self, market: Market, mode: str) -> dict[str, Any] | None:
        column = self._rule_activation_column(mode)
        with _db.connect(self._db_path) as connection:
            row = connection.execute(
                "SELECT engine_version, market, manifest_sha256, metrics_json, "
                "notification_active, gate_active, created_at, updated_at "
                f"FROM market_warning_rule_registry WHERE market = ? AND {column} = 1 "
                "ORDER BY updated_at DESC, engine_version DESC LIMIT 1",
                (Market(market).value,),
            ).fetchone()
        return self._rule_record_from_row(row) if row is not None else None

    @staticmethod
    def _rule_activation_column(mode: str) -> str:
        columns = {"notify": "notification_active", "gate": "gate_active"}
        try:
            return columns[mode]
        except KeyError as exc:
            raise ValueError("mode must be notify or gate") from exc

    @staticmethod
    def _rule_record_from_row(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "engine_version": row["engine_version"],
            "market": Market(row["market"]),
            "manifest_sha256": row["manifest_sha256"],
            "metrics": json.loads(row["metrics_json"]),
            "notification_active": bool(row["notification_active"]),
            "gate_active": bool(row["gate_active"]),
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    @staticmethod
    def _rule_assessment_from_row(row: sqlite3.Row) -> RuleRiskAssessment:
        return RuleRiskAssessment(
            market=row["market"],
            as_of_time=datetime.fromisoformat(row["as_of_time"]),
            engine_version=row["engine_version"],
            manifest_sha256=row["manifest_sha256"],
            risk_level=row["risk_level"],
            risk_score=row["risk_score"],
            market_phase=row["market_phase"],
            triggered_rules=tuple(
                TriggeredRule(**item)
                for item in json.loads(row["triggered_rules_json"])
            ),
            missing_optional_groups=tuple(
                json.loads(row["missing_optional_groups_json"])
            ),
            reliability_grade=row["reliability_grade"],
            evaluation_latency_ms=row["evaluation_latency_ms"],
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

    def load_model_set(self, model_version: str) -> tuple[dict[str, Any], ...]:
        with _db.connect(self._db_path) as connection:
            rows = connection.execute(
                "SELECT model_version, market, horizon, feature_version, calibration_version, "
                "training_cutoff, artifact_path, artifact_sha256, metrics_json, base_rate, active, created_at "
                "FROM market_warning_model_registry WHERE model_version = ? "
                "ORDER BY market, horizon",
                (model_version,),
            ).fetchall()
        return tuple(self._model_record_from_row(row) for row in rows)

    def activate_model_set(self, model_version: str) -> tuple[dict[str, Any], ...]:
        """Atomically activate one complete A-share/US, 1d/3d model set."""

        expected = {(market.value, horizon) for market in Market for horizon in ("1d", "3d")}
        with _db.connect(self._db_path) as connection:
            connection.execute("BEGIN IMMEDIATE")
            rows = connection.execute(
                "SELECT model_version, market, horizon, feature_version, calibration_version, "
                "training_cutoff, artifact_path, artifact_sha256, metrics_json, base_rate, active, created_at "
                "FROM market_warning_model_registry WHERE model_version = ? "
                "ORDER BY market, horizon",
                (model_version,),
            ).fetchall()
            actual = {(row["market"], row["horizon"]) for row in rows}
            if actual != expected or len(rows) != len(expected):
                raise ValueError("model version must contain one complete four-model set")
            for market, horizon in sorted(expected):
                connection.execute(
                    "UPDATE market_warning_model_registry SET active = 0 "
                    "WHERE market = ? AND horizon = ?",
                    (market, horizon),
                )
            cursor = connection.execute(
                "UPDATE market_warning_model_registry SET active = 1 WHERE model_version = ?",
                (model_version,),
            )
            if cursor.rowcount != len(expected):
                raise ValueError("four-model activation did not update every expected row")
        return self.load_model_set(model_version)

    @staticmethod
    def _model_record_from_row(row: sqlite3.Row) -> dict[str, Any]:
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
            decision_source=row["decision_source"],
        )
