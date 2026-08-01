from __future__ import annotations

import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from unittest import TestCase, main
from zoneinfo import ZoneInfo

from tradingagents.harness import db as _db
from tradingagents.harness.market_warning.adapters.feishu_notifier import FeishuNotifier
from tradingagents.harness.market_warning.adapters.sqlite_repository import SQLiteWarningRepository
from tradingagents.harness.market_warning.domain import (
    DataStatus,
    FeatureSnapshot,
    FinalWarningDecision,
    Market,
    MarketPhase,
    QuantRiskAssessment,
    RiskLevel,
    RunnerResult,
)
from tradingagents.harness.market_warning.runner import due_evaluations, run_due_evaluations


UTC = timezone.utc


class CalendarSchedulingTests(TestCase):
    def _slots(self, value: str, market: Market):
        return due_evaluations(datetime.fromisoformat(value), markets=(market,))

    def test_a_share_premarket_and_both_intraday_windows(self) -> None:
        cases = (
            ("2026-08-03T08:30:00+08:00", "premarket"),
            ("2026-08-03T09:35:00+08:00", "intraday-0935"),
            ("2026-08-03T11:25:00+08:00", "intraday-1125"),
            ("2026-08-03T13:05:00+08:00", "intraday-1305"),
            ("2026-08-03T14:55:00+08:00", "intraday-1455"),
        )
        for timestamp, expected in cases:
            with self.subTest(timestamp=timestamp):
                slots = self._slots(timestamp, Market.A_SHARE)
                self.assertEqual(tuple(item.session_slot for item in slots), (expected,))
        for timestamp in (
            "2026-08-03T09:30:00+08:00",
            "2026-08-03T11:30:00+08:00",
            "2026-08-03T12:55:00+08:00",
            "2026-08-03T15:00:00+08:00",
        ):
            with self.subTest(timestamp=timestamp):
                self.assertEqual(self._slots(timestamp, Market.A_SHARE), ())

    def test_exchange_holidays_and_weekends_are_not_due(self) -> None:
        cases = (
            ("2026-10-01T08:30:00+08:00", Market.A_SHARE),
            ("2026-07-03T08:30:00-04:00", Market.US),
            ("2026-08-01T08:30:00+08:00", Market.A_SHARE),
            ("2026-08-01T08:30:00-04:00", Market.US),
        )
        for timestamp, market in cases:
            with self.subTest(timestamp=timestamp, market=market):
                self.assertEqual(self._slots(timestamp, market), ())

    def test_us_summer_and_winter_time_use_exchange_local_clock(self) -> None:
        cases = (
            ("2026-07-06T12:30:00+00:00", "premarket"),
            ("2026-07-06T13:35:00+00:00", "intraday-0935"),
            ("2026-12-28T13:30:00+00:00", "premarket"),
            ("2026-12-28T14:35:00+00:00", "intraday-0935"),
        )
        for timestamp, expected in cases:
            with self.subTest(timestamp=timestamp):
                slots = self._slots(timestamp, Market.US)
                self.assertEqual(tuple(item.session_slot for item in slots), (expected,))

    def test_us_early_close_stops_before_actual_close(self) -> None:
        before_close = self._slots("2026-11-27T17:55:00+00:00", Market.US)
        at_close = self._slots("2026-11-27T18:00:00+00:00", Market.US)

        self.assertEqual(tuple(item.session_slot for item in before_close), ("intraday-1255",))
        self.assertEqual(at_close, ())


class FakeService:
    def __init__(self, calls: list[tuple]) -> None:
        self.calls = calls

    def evaluate(self, market, as_of_time, session_slot):
        self.calls.append((market, as_of_time, session_slot))
        return {"market": market.value, "slot": session_slot}


class RunnerTests(TestCase):
    def test_runner_evaluates_each_due_slot_once(self) -> None:
        calls: list[tuple] = []
        factories: list[tuple] = []

        def factory(slot, force):
            factories.append((slot.market, slot.session_slot, force))
            return FakeService(calls)

        results = run_due_evaluations(
            datetime.fromisoformat("2026-08-03T09:35:00+08:00"),
            service_factory=factory,
            markets=(Market.A_SHARE,),
        )

        self.assertEqual(len(results), 1)
        self.assertEqual(factories, [(Market.A_SHARE, "intraday-0935", False)])
        self.assertEqual(calls[0][0], Market.A_SHARE)
        self.assertEqual(calls[0][2], "intraday-0935")

    def test_dry_run_never_builds_or_invokes_service(self) -> None:
        def forbidden(*_args):
            raise AssertionError("service must not be constructed in dry-run")

        results = run_due_evaluations(
            datetime.fromisoformat("2026-08-03T08:30:00+08:00"),
            service_factory=forbidden,
            markets=(Market.A_SHARE,),
            dry_run=True,
        )

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].session_slot, "premarket")


def _decision(level: RiskLevel, transition: str, push: bool) -> FinalWarningDecision:
    actions = {
        RiskLevel.GREEN: ("OPEN", 100.0, "HOLD"),
        RiskLevel.YELLOW: ("OPEN", 100.0, "HOLD"),
        RiskLevel.ORANGE: ("CONDITIONAL", 3.0, "HOLD_OR_REDUCE"),
        RiskLevel.RED: ("WAIT", 0.0, "REDUCE"),
    }
    gate, cap, holding = actions[level]
    return FinalWarningDecision(
        baseline_level=level,
        final_level=level,
        state_transition=transition,
        entry_gate=gate,
        new_position_cap_pct=cap,
        holding_action=holding,
        push_required=push,
        decision_reasons=(),
        data_status=DataStatus.FRESH,
    )


def _result(
    decision_id: int,
    level: RiskLevel,
    *,
    slot: str,
    transition: str = "UNCHANGED",
    push: bool = False,
) -> RunnerResult:
    as_of = datetime(2026, 8, 3, 1, 35, tzinfo=UTC)
    snapshot = FeatureSnapshot(
        market=Market.A_SHARE,
        as_of_time=as_of,
        session_slot=slot,
        feature_version="market-warning-v2",
        features={"market_phase": "FIRST_SHOCK"},
        evidence=(),
        data_quality=DataStatus.FRESH,
        reliability_grade="A",
        source_times={"fixture": as_of},
    )
    quant = QuantRiskAssessment(
        crash_1d_probability=0.05,
        crash_3d_probability=0.10,
        market_phase=MarketPhase.FIRST_SHOCK,
        base_rate_1d=0.01,
        base_rate_3d=0.02,
        reliability_grade="A",
        model_version="model-v2",
        calibration_version="platt-v2",
        top_contributors=(),
    )
    return RunnerResult(
        market=Market.A_SHARE,
        as_of_time=as_of,
        session_slot=slot,
        feature_snapshot=snapshot,
        quant_assessment=quant,
        decision=_decision(level, transition, push),
        decision_id=decision_id,
    )


def _persist_decision(repository: SQLiteWarningRepository) -> int:
    as_of = datetime(2026, 8, 3, 1, 35, tzinfo=UTC)
    snapshot = FeatureSnapshot(
        market=Market.A_SHARE,
        as_of_time=as_of,
        session_slot="premarket",
        feature_version="market-warning-v2",
        features={"market_phase": "FIRST_SHOCK"},
        evidence=(),
        data_quality=DataStatus.FRESH,
        reliability_grade="A",
        source_times={"fixture": as_of},
    )
    quant = QuantRiskAssessment(
        crash_1d_probability=0.05,
        crash_3d_probability=0.10,
        market_phase=MarketPhase.FIRST_SHOCK,
        base_rate_1d=0.01,
        base_rate_3d=0.02,
        reliability_grade="A",
        model_version="model-v2",
        calibration_version="platt-v2",
        top_contributors=(),
    )
    snapshot_id = repository.save_feature_snapshot(snapshot)
    predictions = repository.save_predictions(snapshot_id, quant)
    return repository.save_decision(
        snapshot_id,
        predictions,
        None,
        _decision(RiskLevel.ORANGE, "INITIAL_ORANGE", True),
    )


class FeishuNotifierTests(TestCase):
    def test_notification_policy_is_defensive(self) -> None:
        class Repository:
            def claim_alert(self, *_args, **_kwargs):
                return True

            def finish_alert(self, *_args, **_kwargs):
                pass

        sent: list[str] = []
        notifier = FeishuNotifier(Repository(), sender=sent.append)
        cases = (
            (_result(1, RiskLevel.GREEN, slot="premarket"), True),
            (_result(1, RiskLevel.GREEN, slot="intraday-0935"), False),
            (_result(1, RiskLevel.YELLOW, slot="intraday-0935"), False),
            (_result(1, RiskLevel.ORANGE, slot="intraday-0935", transition="INITIAL_ORANGE", push=True), True),
            (_result(1, RiskLevel.RED, slot="intraday-0940", transition="UPGRADE_ORANGE_TO_RED", push=True), True),
            (_result(1, RiskLevel.YELLOW, slot="intraday-0945", transition="RECOVERY_ORANGE_TO_YELLOW"), False),
        )
        for result, expected in cases:
            with self.subTest(slot=result.session_slot, level=result.decision.final_level):
                before = len(sent)
                self.assertEqual(notifier.notify(result), expected)
                self.assertEqual(len(sent) - before, int(expected))

    def test_two_repository_instances_cannot_send_duplicate_alert(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            db_path = Path(directory) / "warning.db"
            first_repository = SQLiteWarningRepository(db_path)
            decision_id = _persist_decision(first_repository)
            sent: list[str] = []
            result = _result(
                decision_id,
                RiskLevel.ORANGE,
                slot="intraday-0935",
                transition="INITIAL_ORANGE",
                push=True,
            )

            first = FeishuNotifier(first_repository, sender=sent.append)
            second = FeishuNotifier(SQLiteWarningRepository(db_path), sender=sent.append)

            self.assertTrue(first.notify(result))
            self.assertFalse(second.notify(result))
            self.assertEqual(len(sent), 1)

    def test_failed_alert_can_be_explicitly_retried_without_new_row(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            db_path = Path(directory) / "warning.db"
            repository = SQLiteWarningRepository(db_path)
            decision_id = _persist_decision(repository)
            result = _result(decision_id, RiskLevel.GREEN, slot="premarket")

            failing = FeishuNotifier(
                repository,
                sender=lambda _message: (_ for _ in ()).throw(RuntimeError("credential detail")),
            )
            with self.assertRaises(RuntimeError):
                failing.notify(result)

            sent: list[str] = []
            retry = FeishuNotifier(
                SQLiteWarningRepository(db_path),
                sender=sent.append,
                retry_failed=True,
            )
            self.assertTrue(retry.notify(result))

            with _db.connect(db_path) as connection:
                rows = connection.execute(
                    "SELECT push_status, error_summary FROM market_warning_alerts"
                ).fetchall()
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["push_status"], "sent")
            self.assertIsNone(rows[0]["error_summary"])
            self.assertEqual(len(sent), 1)


if __name__ == "__main__":
    main()
