"""Exchange-aware runner for scheduled dual-market crash warnings."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import uuid
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from functools import partial
from pathlib import Path
from typing import Callable, Iterable, Sequence
from zoneinfo import ZoneInfo

import exchange_calendars as xcals
import pandas as pd

from .domain import Market, RunnerResult


_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_MARKET_ZONES = {
    Market.A_SHARE: ZoneInfo("Asia/Shanghai"),
    Market.US: ZoneInfo("America/New_York"),
}
_CALENDAR_NAMES = {Market.A_SHARE: "XSHG", Market.US: "XNYS"}
_MARKET_NAMES = {Market.A_SHARE: "A股", Market.US: "美股"}


@dataclass(frozen=True)
class EvaluationSlot:
    market: Market
    as_of_time: datetime
    session_slot: str
    local_trade_date: date


def _calendar_row(market: Market, local_date: date):
    calendar = xcals.get_calendar(_CALENDAR_NAMES[market])
    label = pd.Timestamp(local_date)
    if not calendar.is_session(label):
        return None
    return calendar.schedule.loc[label]


def _on_ten_minute_grid(value: datetime, start: time) -> bool:
    current_minutes = value.hour * 60 + value.minute
    start_minutes = start.hour * 60 + start.minute
    return current_minutes >= start_minutes and (current_minutes - start_minutes) % 10 == 0


def _slot_for_market(market: Market, now: datetime) -> EvaluationSlot | None:
    zone = _MARKET_ZONES[market]
    local = now.astimezone(zone).replace(second=0, microsecond=0)
    row = _calendar_row(market, local.date())
    if row is None:
        return None
    if local.time() == time(8, 30):
        return EvaluationSlot(market, local, "premarket", local.date())
    if market == Market.US:
        return None

    in_morning = time(9, 35) <= local.time() <= time(11, 25)
    in_afternoon = time(13, 5) <= local.time() <= time(14, 55)
    on_grid = (
        in_morning and _on_ten_minute_grid(local, time(9, 35))
    ) or (
        in_afternoon and _on_ten_minute_grid(local, time(13, 5))
    )
    if not on_grid:
        return None
    return EvaluationSlot(
        market=market,
        as_of_time=local,
        session_slot=f"intraday-{local:%H%M}",
        local_trade_date=local.date(),
    )


def due_evaluations(
    now: datetime,
    *,
    markets: Iterable[Market] = (Market.A_SHARE, Market.US),
) -> tuple[EvaluationSlot, ...]:
    """Return only evaluations due on the configured exchange-local schedule."""

    if not isinstance(now, datetime) or now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("now must be timezone-aware")
    result = []
    for market in markets:
        slot = _slot_for_market(Market(market), now)
        if slot is not None:
            result.append(slot)
    return tuple(result)


def _error_class(error: Exception) -> str:
    name = re.sub(r"(?<!^)(?=[A-Z])", "_", error.__class__.__name__).lower()
    return name.removesuffix("_error") + "_error"


class FastScanCoordinator:
    """Serialize one market's scans and persist operational outcomes."""

    def __init__(
        self,
        repository,
        *,
        mode: str,
        owner_id: str | None = None,
        lease_duration: timedelta = timedelta(minutes=8),
        failure_threshold: int = 3,
        clock: Callable[[], datetime] | None = None,
        alert_sender: Callable[[str], None] | None = None,
    ) -> None:
        self.repository = repository
        self.mode = mode
        self.owner_id = owner_id or uuid.uuid4().hex
        self.lease_duration = lease_duration
        self.failure_threshold = failure_threshold
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        self.alert_sender = alert_sender

    def execute(self, slot: EvaluationSlot, callback: Callable[[], object]) -> object:
        lease_key = f"market_warning_fast_scan:{slot.market.value}"
        started_at = self.clock()
        if not self.repository.acquire_lease(
            lease_key,
            self.owner_id,
            started_at,
            self.lease_duration,
        ):
            result = RunnerResult(
                market=slot.market,
                as_of_time=slot.as_of_time,
                session_slot=slot.session_slot,
                error_class="overlap_skipped",
            )
            self._record(slot, started_at, self.clock(), result, overlap_skipped=True)
            return result

        try:
            try:
                result = callback()
            except Exception as error:
                result = RunnerResult(
                    market=slot.market,
                    as_of_time=slot.as_of_time,
                    session_slot=slot.session_slot,
                    error_class=_error_class(error),
                )
            finished_at = self.clock()
            self._record(slot, started_at, finished_at, result, overlap_skipped=False)
            return result
        finally:
            self.repository.release_lease(lease_key, self.owner_id)

    def _record(
        self,
        slot: EvaluationSlot,
        started_at: datetime,
        finished_at: datetime,
        result: object,
        *,
        overlap_skipped: bool,
    ) -> None:
        error_class = getattr(result, "error_class", None)
        succeeded = error_class is None
        self.repository.record_run(
            market=slot.market,
            as_of_time=slot.as_of_time,
            session_slot=slot.session_slot,
            mode=self.mode,
            started_at=started_at,
            finished_at=finished_at,
            status="success" if succeeded else ("overlap_skipped" if overlap_skipped else "failed"),
            error_class=error_class,
            overlap_skipped=overlap_skipped,
            llm_calls=0,
        )
        streak = self.repository.record_schedule_outcome(
            slot.market,
            self.mode,
            slot.as_of_time,
            succeeded=succeeded,
            failure_threshold=self.failure_threshold,
        )
        if streak["alert_due"]:
            self._send_system_alert(slot, streak)

    def _send_system_alert(self, slot: EvaluationSlot, streak: dict[str, object]) -> None:
        incident = str(streak["incident_started_at"])
        key = f"market-warning-system|{slot.market.value}|{self.mode}|{incident}"
        message = "\n".join(
            (
                "【预警系统数据故障】",
                f"{_MARKET_NAMES.get(slot.market, slot.market.value)}预警已连续 "
                f"{streak['consecutive_failures']} 个应执行时点未完成有效扫描。",
                f"最近时点：{slot.as_of_time.isoformat(timespec='minutes')}",
                "这是预警系统运行或数据故障，不代表市场红灯。",
            )
        )
        digest = hashlib.sha256(message.encode("utf-8")).hexdigest()
        if not self.repository.claim_system_alert(key, digest, retry_failed=True):
            return
        try:
            sender = self.alert_sender
            if sender is None:
                from tradingagents.harness.market_risk_daily import _send_feishu_message

                sender = _send_feishu_message
            sender(message)
        except Exception:
            self.repository.finish_system_alert(key, "failed", "send_error")
            return
        self.repository.finish_system_alert(key, "sent")


def run_due_evaluations(
    now: datetime,
    *,
    service_factory: Callable[[EvaluationSlot, bool], object],
    markets: Iterable[Market] = (Market.A_SHARE, Market.US),
    dry_run: bool = False,
    force: bool = False,
    coordinator_factory: Callable[[EvaluationSlot], FastScanCoordinator] | None = None,
) -> tuple[object, ...]:
    slots = due_evaluations(now, markets=markets)
    if dry_run:
        return slots
    results = []
    for slot in slots:
        service = service_factory(slot, force)
        callback = lambda service=service, slot=slot: service.evaluate(
            slot.market,
            slot.as_of_time,
            slot.session_slot,
        )
        if coordinator_factory is None:
            results.append(callback())
        else:
            results.append(coordinator_factory(slot).execute(slot, callback))
    return tuple(results)


class _SessionDataPort:
    def __init__(self, daily, intraday=None) -> None:
        self.daily = daily
        self.intraday = intraday or daily

    def load_snapshot(self, market, as_of_time, session_slot):
        adapter = self.intraday if "intraday" in session_slot.lower() else self.daily
        return adapter.load_snapshot(market, as_of_time, session_slot)


def _load_history_incrementally(
    cache,
    daily,
    market: Market,
    as_of_time: datetime,
    session_slot: str,
    previous_session,
):
    """Use cache intraday; premarket may fetch only the latest completed session."""

    zone = _MARKET_ZONES[Market(market)]
    local_date = as_of_time.astimezone(zone).date()
    incremental = ()
    if "premarket" in session_slot.lower():
        latest = previous_session(local_date)
        try:
            incremental = tuple(daily.backfill(latest, latest))
        except Exception:
            incremental = ()
    cached = cache.list_snapshots(
        market,
        "daily_snapshot",
        local_date - timedelta(days=420),
        local_date,
    )
    merged = {
        (item.as_of_time, item.session_slot): item
        for item in (*cached, *incremental)
        if item.as_of_time < as_of_time
    }
    return tuple(item for _, item in sorted(merged.items()))


def _reasoning_adapter(repository):
    from tradingagents.harness.market_warning.adapters.minimax_reasoning import (
        MiniMaxReasoningAdapter,
        UnavailableReasoningAdapter,
    )

    try:
        return MiniMaxReasoningAdapter.from_environment(
            breaker=repository.circuit_breaker("minimax-m3")
        )
    except Exception:
        return UnavailableReasoningAdapter("initialization_error")


def _load_environment() -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv(_PROJECT_ROOT / ".env", override=False)
    except ImportError:
        return


def _resolve_mode(repository, requested_mode: str | None, market: Market) -> str:
    if market == Market.US:
        return "model"
    if requested_mode is not None:
        return requested_mode
    active = repository.load_active_rule_engine(market, "notify")
    return "rule_v1" if active is not None else "model"


def _default_service_factory(
    slot: EvaluationSlot,
    force: bool,
    *,
    requested_mode: str | None = None,
):
    """Compose production adapters lazily so calendar-only dry runs stay offline."""

    _load_environment()
    from tradingagents.harness.market_warning.adapters.data_cache import RawDataCache
    from tradingagents.harness.market_warning.adapters.feishu_notifier import FeishuNotifier
    from tradingagents.harness.market_warning.adapters.sqlite_repository import SQLiteWarningRepository
    from tradingagents.harness.market_warning.adapters.tushare_data import TushareAShareDataAdapter
    from tradingagents.harness.market_warning.adapters.us_market_data import YahooUSDataAdapter
    from tradingagents.harness.market_warning.adapters.realtime_quote import RealtimeAShareDataAdapter
    from tradingagents.harness.market_warning.features import AShareFeatureStrategy, USFeatureStrategy
    from tradingagents.harness.market_warning.probability import SklearnProbabilityModel
    from tradingagents.harness.market_warning.service import MarketWarningService

    repository = SQLiteWarningRepository()
    cache = RawDataCache(_PROJECT_ROOT / "harness_data/market_warning/raw")
    from tradingagents.harness.market_warning.calendars import session_resolvers

    next_session, previous_session, calendar_version = session_resolvers(slot.market)
    if slot.market == Market.A_SHARE:
        from tradingagents.dataflows.tushare_vendor import _get_tushare_api

        pro = _get_tushare_api()
        daily = TushareAShareDataAdapter(
            pro=pro,
            cache=cache,
            next_trading_day=next_session,
            previous_session=previous_session,
            calendar_version=calendar_version,
        )
        data_port = _SessionDataPort(daily, RealtimeAShareDataAdapter(pro=pro))
        strategy = AShareFeatureStrategy()
    else:
        daily = YahooUSDataAdapter(
            cache=cache,
            previous_session=previous_session,
            calendar_version=calendar_version,
        )
        data_port = _SessionDataPort(daily)
        strategy = USFeatureStrategy()

    def load_history(_market: Market, as_of_time: datetime):
        return _load_history_incrementally(
            cache,
            daily,
            slot.market,
            as_of_time,
            slot.session_slot,
            previous_session,
        )

    mode = _resolve_mode(repository, requested_mode, slot.market)
    if mode == "rule_v1":
        if slot.market != Market.A_SHARE:
            raise ValueError("rule_v1 is only available for A shares")
        from tradingagents.harness.market_warning.adapters.tushare_realtime_breadth import (
            build_premarket_baseline,
            load_realtime_cross_section,
        )
        from tradingagents.harness.market_warning.rule_policy import (
            evaluate_a_share_rules,
            load_rule_manifest,
        )
        from tradingagents.harness.market_warning.rule_service import RuleMarketWarningService

        manifest = load_rule_manifest(Path(__file__).with_name("rule_manifest_v1.json"))
        cross_section_loader = None
        if "intraday" in slot.session_slot.lower():
            baseline = build_premarket_baseline(
                pro,
                trade_date=slot.local_trade_date,
                previous_session=previous_session,
                cache_root=_PROJECT_ROOT / "harness_data/market_warning/baselines",
            )
            cross_section_loader = partial(load_realtime_cross_section, pro, baseline)
        data_port = _SessionDataPort(
            daily,
            RealtimeAShareDataAdapter(
                pro=pro,
                cross_section_loader=cross_section_loader,
            ),
        )
        return RuleMarketWarningService(
            data_port=data_port,
            feature_strategy=AShareFeatureStrategy(),
            rule_evaluator=partial(evaluate_a_share_rules, manifest=manifest),
            repository=repository,
            notifier=FeishuNotifier(repository, retry_failed=force),
            post_alert_reasoning=_reasoning_adapter(repository),
            report_root=_PROJECT_ROOT / "reports/market_warning",
            history_loader=load_history,
            engine_version=manifest.engine_version,
            manifest_sha256=manifest.manifest_sha256,
        )

    reasoning = _reasoning_adapter(repository)
    return MarketWarningService(
        data_port=data_port,
        feature_strategies={slot.market: strategy},
        probability_model=SklearnProbabilityModel(
            repository,
            _PROJECT_ROOT / "harness_data/models/market_warning",
        ),
        repository=repository,
        reasoning=reasoning,
        notifier=FeishuNotifier(repository, retry_failed=force),
        report_root=_PROJECT_ROOT / "reports/market_warning",
        history_loader=load_history,
    )


def _default_coordinator_factory(
    slot: EvaluationSlot,
    *,
    requested_mode: str | None = None,
) -> FastScanCoordinator:
    from tradingagents.harness.market_warning.adapters.sqlite_repository import SQLiteWarningRepository

    repository = SQLiteWarningRepository()
    return FastScanCoordinator(
        repository,
        mode=_resolve_mode(repository, requested_mode, slot.market),
    )


def _parse_time(value: str | None) -> datetime:
    if value is None:
        return datetime.now(timezone.utc)
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("--at must include a timezone offset")
    return parsed


def _markets(value: str) -> tuple[Market, ...]:
    if value == "all":
        return (Market.A_SHARE, Market.US)
    return (Market(value),)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run due dual-market crash warnings.")
    parser.add_argument("--market", choices=("all", "a_share", "us"), default="all")
    parser.add_argument("--at", help="Timezone-aware ISO evaluation time")
    parser.add_argument("--mode", choices=("model", "rule_v1"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true", help="Retry a matching failed alert")
    args = parser.parse_args(argv)
    try:
        now = _parse_time(args.at)
        results = run_due_evaluations(
            now,
            service_factory=partial(
                _default_service_factory,
                requested_mode=args.mode,
            ),
            markets=_markets(args.market),
            dry_run=args.dry_run,
            force=args.force,
            coordinator_factory=partial(
                _default_coordinator_factory,
                requested_mode=args.mode,
            ),
        )
    except (ValueError, RuntimeError) as error:
        parser.error(str(error))
    summaries = []
    for item in results:
        if isinstance(item, EvaluationSlot):
            summaries.append(
                {
                    "market": item.market.value,
                    "as_of_time": item.as_of_time.isoformat(),
                    "session_slot": item.session_slot,
                    "status": "due",
                }
            )
        else:
            decision = getattr(item, "decision", None)
            summaries.append(
                {
                    "market": getattr(getattr(item, "market", None), "value", None),
                    "session_slot": getattr(item, "session_slot", None),
                    "level": getattr(getattr(decision, "final_level", None), "value", None),
                    "report_path": getattr(item, "report_path", None),
                    "error_class": getattr(item, "error_class", None),
                }
            )
    print(json.dumps(summaries, ensure_ascii=False, sort_keys=True))
    return 1 if any(getattr(item, "error_class", None) for item in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
