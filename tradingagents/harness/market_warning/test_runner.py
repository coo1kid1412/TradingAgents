from __future__ import annotations

import json
import io
import tempfile
from contextlib import redirect_stdout
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import TestCase, main, mock
from zoneinfo import ZoneInfo

from tradingagents.harness import db as _db
from tradingagents.harness.market_warning.adapters.feishu_notifier import (
    FeishuNotifier,
    _idempotency_key,
)
from tradingagents.harness.market_warning.adapters.sqlite_repository import SQLiteWarningRepository
from tradingagents.harness.market_warning.domain import (
    DataStatus,
    DecisionSource,
    FeatureSnapshot,
    FinalWarningDecision,
    Market,
    MarketPhase,
    QuantRiskAssessment,
    RiskLevel,
    RuleRiskAssessment,
    RunnerResult,
)
from tradingagents.harness.market_warning.runner import (
    EvaluationSlot,
    FastScanCoordinator,
    _load_history_incrementally,
    _reasoning_adapter,
    due_evaluations,
    main as runner_main,
    run_due_evaluations,
)


UTC = timezone.utc


class CalendarSchedulingTests(TestCase):
    def _slots(self, value: str, market: Market):
        return due_evaluations(datetime.fromisoformat(value), markets=(market,))

    def test_a_share_premarket_and_both_intraday_windows(self) -> None:
        cases = (
            ("2026-08-03T08:30:00+08:00", "premarket"),
            ("2026-08-03T09:35:00+08:00", "intraday-0935"),
            ("2026-08-03T09:45:00+08:00", "intraday-0945"),
            ("2026-08-03T11:25:00+08:00", "intraday-1125"),
            ("2026-08-03T13:05:00+08:00", "intraday-1305"),
            ("2026-08-03T13:15:00+08:00", "intraday-1315"),
            ("2026-08-03T14:55:00+08:00", "intraday-1455"),
        )
        for timestamp, expected in cases:
            with self.subTest(timestamp=timestamp):
                slots = self._slots(timestamp, Market.A_SHARE)
                self.assertEqual(tuple(item.session_slot for item in slots), (expected,))
        for timestamp in (
            "2026-08-03T08:35:00+08:00",
            "2026-08-03T09:30:00+08:00",
            "2026-08-03T09:40:00+08:00",
            "2026-08-03T11:30:00+08:00",
            "2026-08-03T12:00:00+08:00",
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

    def test_us_shadow_runs_only_once_premarket_in_exchange_local_clock(self) -> None:
        cases = (
            ("2026-07-06T12:30:00+00:00", "premarket"),
            ("2026-12-28T13:30:00+00:00", "premarket"),
        )
        for timestamp, expected in cases:
            with self.subTest(timestamp=timestamp):
                slots = self._slots(timestamp, Market.US)
                self.assertEqual(tuple(item.session_slot for item in slots), (expected,))
        self.assertEqual(self._slots("2026-07-06T13:35:00+00:00", Market.US), ())
        self.assertEqual(self._slots("2026-12-28T14:35:00+00:00", Market.US), ())


class FakeService:
    def __init__(self, calls: list[tuple]) -> None:
        self.calls = calls

    def evaluate(self, market, as_of_time, session_slot):
        self.calls.append((market, as_of_time, session_slot))
        return {"market": market.value, "slot": session_slot}


class SplitPathService:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    def evaluate_fast(self, market, as_of_time, session_slot):
        self.events.append("fast")
        return {"market": market.value, "slot": session_slot}

    def complete_after_alert(self, result):
        self.events.append("slow")
        return {**result, "completed": True}


class TracingCoordinator:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    def execute(self, _slot, callback):
        self.events.append("lease_acquired")
        result = callback()
        self.events.append("lease_released")
        return result


class RunnerTests(TestCase):
    def test_fast_scan_coordinator_skips_overlap_and_records_zero_llm_calls(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            db_path = Path(directory) / "warning.db"
            repository = SQLiteWarningRepository(db_path)
            now = datetime.fromisoformat("2026-08-03T09:35:00+08:00")
            slot = due_evaluations(now, markets=(Market.A_SHARE,))[0]
            self.assertTrue(
                repository.acquire_lease(
                    "market_warning_fast_scan:a_share",
                    "existing-owner",
                    now,
                    timedelta(minutes=12),
                )
            )
            coordinator = FastScanCoordinator(
                repository,
                mode="rule_v1",
                owner_id="second-owner",
                clock=lambda: now,
            )

            result = coordinator.execute(slot, lambda: self.fail("overlap must skip work"))

            self.assertEqual(result.error_class, "overlap_skipped")
            with _db.connect(db_path) as connection:
                row = connection.execute(
                    "SELECT status, overlap_skipped, llm_calls "
                    "FROM market_warning_runs ORDER BY id DESC LIMIT 1"
                ).fetchone()
            self.assertEqual(tuple(row), ("overlap_skipped", 1, 0))

    def test_default_lease_blocks_the_next_ten_minute_scan(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repository = SQLiteWarningRepository(Path(directory) / "warning.db")
            started = datetime.fromisoformat("2026-08-03T09:35:00+08:00")
            coordinator = FastScanCoordinator(
                repository,
                mode="rule_v1",
                owner_id="first-owner",
            )

            self.assertEqual(coordinator.lease_duration, timedelta(minutes=12))
            self.assertTrue(
                repository.acquire_lease(
                    "market_warning_fast_scan:a_share",
                    coordinator.owner_id,
                    started,
                    coordinator.lease_duration,
                )
            )
            self.assertFalse(
                repository.acquire_lease(
                    "market_warning_fast_scan:a_share",
                    "next-owner",
                    started + timedelta(minutes=10),
                    coordinator.lease_duration,
                )
            )

    def test_fast_scan_coordinator_releases_lease_after_exception(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            db_path = Path(directory) / "warning.db"
            repository = SQLiteWarningRepository(db_path)
            now = datetime.fromisoformat("2026-08-03T09:35:00+08:00")
            slot = due_evaluations(now, markets=(Market.A_SHARE,))[0]
            coordinator = FastScanCoordinator(
                repository,
                mode="rule_v1",
                owner_id="failed-owner",
                clock=lambda: now,
            )

            result = coordinator.execute(
                slot,
                lambda: (_ for _ in ()).throw(RuntimeError("private failure detail")),
            )

            self.assertEqual(result.error_class, "runtime_error")
            self.assertTrue(
                repository.acquire_lease(
                    "market_warning_fast_scan:a_share",
                    "next-owner",
                    now,
                    timedelta(minutes=8),
                )
            )

    def test_three_consecutive_failed_slots_send_one_distinct_system_alert(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repository = SQLiteWarningRepository(Path(directory) / "warning.db")
            now = datetime.fromisoformat("2026-08-03T09:35:00+08:00")
            sent: list[str] = []
            current = [now]
            coordinator = FastScanCoordinator(
                repository,
                mode="rule_v1",
                owner_id="worker",
                clock=lambda: current[0],
                alert_sender=sent.append,
            )

            for offset in range(4):
                current[0] = now + timedelta(minutes=offset * 10)
                slot = EvaluationSlot(
                    Market.A_SHARE,
                    current[0],
                    f"intraday-{current[0]:%H%M}",
                    current[0].date(),
                )
                coordinator.execute(
                    slot,
                    lambda: RunnerResult(
                        market=Market.A_SHARE,
                        as_of_time=current[0],
                        session_slot=slot.session_slot,
                        error_class="data_unavailable",
                    ),
                )

            self.assertEqual(len(sent), 1)
            self.assertIn("预警系统数据故障", sent[0])
            self.assertIn("不代表市场红灯", sent[0])

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

    def test_rule_slow_path_runs_only_after_coordinator_releases_fast_lease(self) -> None:
        events: list[str] = []

        results = run_due_evaluations(
            datetime.fromisoformat("2026-08-03T09:35:00+08:00"),
            service_factory=lambda _slot, _force: SplitPathService(events),
            markets=(Market.A_SHARE,),
            coordinator_factory=lambda _slot: TracingCoordinator(events),
        )

        self.assertEqual(events, ["lease_acquired", "fast", "lease_released", "slow"])
        self.assertTrue(results[0]["completed"])

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

    def test_deepseek_initialization_failure_becomes_a_persistable_fallback_adapter(self) -> None:
        repository = mock.Mock()
        repository.circuit_breaker.return_value = mock.Mock()
        with mock.patch(
            "tradingagents.harness.market_warning.adapters.deepseek_reasoning."
            "DeepSeekReasoningAdapter.from_environment",
            side_effect=RuntimeError("private initialization failure"),
        ):
            reasoning = _reasoning_adapter(repository)

        repository.circuit_breaker.assert_called_once_with("deepseek-v4-pro")
        self.assertEqual(reasoning.model_name, "deepseek-v4-pro")

        result = _result(1, RiskLevel.ORANGE, slot="premarket")
        assessment = reasoning.assess(
            result.feature_snapshot,
            result.quant_assessment,
            None,
        )
        self.assertEqual(assessment.reasoning_status, "fallback")
        self.assertEqual(assessment.error_class, "initialization_error")

    def test_intraday_history_is_cache_only_and_premarket_fetches_one_missing_session(self) -> None:
        cached_snapshot = _result(1, RiskLevel.GREEN, slot="premarket").feature_snapshot
        cached_snapshot = FeatureSnapshot(
            market=cached_snapshot.market,
            as_of_time=datetime(2026, 7, 31, 1, 0, tzinfo=UTC),
            session_slot=cached_snapshot.session_slot,
            feature_version=cached_snapshot.feature_version,
            features=cached_snapshot.features,
            evidence=cached_snapshot.evidence,
            data_quality=cached_snapshot.data_quality,
            reliability_grade=cached_snapshot.reliability_grade,
            source_times={"fixture": datetime(2026, 7, 31, 1, 0, tzinfo=UTC)},
        )
        cached = (cached_snapshot,)

        class Cache:
            def list_snapshots(self, market, dataset, start_date, end_date):
                self.window = (market, dataset, start_date, end_date)
                return cached

        class Daily:
            def __init__(self):
                self.calls = []

            def backfill(self, start_date, end_date):
                self.calls.append((start_date, end_date))
                return ()

        cache = Cache()
        daily = Daily()
        as_of = datetime.fromisoformat("2026-08-03T09:35:00+08:00")
        previous_session = lambda _current: datetime(2026, 7, 31).date()

        intraday = _load_history_incrementally(
            cache,
            daily,
            Market.A_SHARE,
            as_of,
            "intraday-0935",
            previous_session,
        )
        self.assertEqual(intraday, cached)
        self.assertEqual(daily.calls, [])

        _load_history_incrementally(
            cache,
            daily,
            Market.A_SHARE,
            as_of.replace(hour=8, minute=30),
            "premarket",
            previous_session,
        )
        self.assertEqual(daily.calls, [(datetime(2026, 7, 31).date(), datetime(2026, 7, 31).date())])

    def test_cli_returns_nonzero_when_a_due_evaluation_is_degraded(self) -> None:
        degraded = RunnerResult(
            **{
                **_result(1, RiskLevel.GREEN, slot="premarket").__dict__,
                "error_class": "model_unavailable",
            }
        )
        with mock.patch(
            "tradingagents.harness.market_warning.runner.run_due_evaluations",
            return_value=(degraded,),
        ), redirect_stdout(io.StringIO()):
            code = runner_main(
                ["--market", "a_share", "--at", "2026-08-03T08:30:00+08:00"]
            )

        self.assertEqual(code, 1)


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


def _rule_result(
    decision_id: int,
    level: RiskLevel,
    *,
    slot: str,
    transition: str,
    push: bool,
) -> RunnerResult:
    result = _result(
        decision_id,
        level,
        slot=slot,
        transition=transition,
        push=push,
    )
    assessment = RuleRiskAssessment(
        market=Market.A_SHARE,
        as_of_time=result.as_of_time,
        engine_version="rule-v1.0.0",
        manifest_sha256="b" * 64,
        risk_level=level,
        risk_score=5.0,
        market_phase=MarketPhase.FIRST_SHOCK,
        triggered_rules=(),
        missing_optional_groups=(),
        reliability_grade="A",
        evaluation_latency_ms=12.0,
    )
    return replace(
        result,
        quant_assessment=None,
        rule_assessment=assessment,
        decision=replace(result.decision, decision_source=DecisionSource.RULE_V1),
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
    def test_rule_notification_key_uses_engine_and_manifest_without_model(self) -> None:
        result = _rule_result(
            1,
            RiskLevel.ORANGE,
            slot="intraday-0935",
            transition="INITIAL_ORANGE",
            push=True,
        )

        key = _idempotency_key(result)

        self.assertEqual(
            key,
            "a_share|2026-08-03|bucket-0935|ORANGE|INITIAL_ORANGE|"
            f"rule-v1.0.0|{'b' * 64}",
        )

    def test_rule_notification_renders_and_sends_without_quant_assessment(self) -> None:
        class Repository:
            def claim_alert(self, *_args, **_kwargs):
                return True

            def finish_alert(self, *_args, **_kwargs):
                pass

        result = _rule_result(
            1,
            RiskLevel.RED,
            slot="intraday-0935",
            transition="UPGRADE_ORANGE_TO_RED",
            push=True,
        )
        sent: list[str] = []

        self.assertTrue(FeishuNotifier(Repository(), sender=sent.append).notify(result))
        self.assertEqual(len(sent), 1)
        self.assertTrue(sent[0].startswith("# 【红灯：风险确认】"))

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

    def test_notifier_sends_the_persisted_report_instead_of_rerendering_without_previous_state(self) -> None:
        class Repository:
            def claim_alert(self, *_args, **_kwargs):
                return True

            def finish_alert(self, *_args, **_kwargs):
                pass

        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "warning.md"
            report.write_text("persisted ORANGE -> RED report", encoding="utf-8")
            result = _result(
                1,
                RiskLevel.RED,
                slot="intraday-0940",
                transition="UPGRADE_ORANGE_TO_RED",
                push=True,
            )
            result = RunnerResult(**{**result.__dict__, "report_path": str(report)})
            sent: list[str] = []

            self.assertTrue(FeishuNotifier(Repository(), sender=sent.append).notify(result))
            self.assertEqual(sent, ["persisted ORANGE -> RED report"])

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


class InstallerTests(TestCase):
    def _script(self) -> str:
        return (
            Path(__file__).resolve().parents[3] / "scripts/install_market_warning_cron.sh"
        ).read_text(encoding="utf-8")

    def test_cron_installer_creates_log_directory_before_installing_redirect(self) -> None:
        script = self._script()

        self.assertIn('mkdir -p "$LOG_DIR"', script)
        self.assertLess(script.index('mkdir -p "$LOG_DIR"'), script.rindex("| crontab -"))

    def test_cron_installer_checks_production_readiness_before_crontab(self) -> None:
        script = self._script()

        self.assertIn("tradingagents.harness.market_warning.readiness", script)
        self.assertLess(
            script.index("tradingagents.harness.market_warning.readiness"),
            script.rindex("| crontab -"),
        )


if __name__ == "__main__":
    main()
