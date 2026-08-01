"""Exchange-aware five-minute runner for dual-market crash warnings."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Callable, Iterable, Sequence
from zoneinfo import ZoneInfo

import exchange_calendars as xcals
import pandas as pd

from .domain import Market


_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_MARKET_ZONES = {
    Market.A_SHARE: ZoneInfo("Asia/Shanghai"),
    Market.US: ZoneInfo("America/New_York"),
}
_CALENDAR_NAMES = {Market.A_SHARE: "XSHG", Market.US: "XNYS"}


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


def _on_five_minute_grid(value: datetime) -> bool:
    return value.minute % 5 == 0


def _slot_for_market(market: Market, now: datetime) -> EvaluationSlot | None:
    zone = _MARKET_ZONES[market]
    local = now.astimezone(zone).replace(second=0, microsecond=0)
    row = _calendar_row(market, local.date())
    if row is None:
        return None
    if local.time() == time(8, 30):
        return EvaluationSlot(market, local, "premarket", local.date())
    if not _on_five_minute_grid(local):
        return None

    if market == Market.A_SHARE:
        in_morning = time(9, 35) <= local.time() <= time(11, 25)
        in_afternoon = time(13, 5) <= local.time() <= time(14, 55)
        if not (in_morning or in_afternoon):
            return None
    else:
        opened = pd.Timestamp(row["open"]).to_pydatetime().astimezone(zone)
        closed = pd.Timestamp(row["close"]).to_pydatetime().astimezone(zone)
        if not opened + timedelta(minutes=5) <= local <= closed - timedelta(minutes=5):
            return None
        if int((local - opened).total_seconds()) % 300 != 0:
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
    """Return only evaluations due in the current five-minute exchange bucket."""

    if not isinstance(now, datetime) or now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("now must be timezone-aware")
    result = []
    for market in markets:
        slot = _slot_for_market(Market(market), now)
        if slot is not None:
            result.append(slot)
    return tuple(result)


def run_due_evaluations(
    now: datetime,
    *,
    service_factory: Callable[[EvaluationSlot, bool], object],
    markets: Iterable[Market] = (Market.A_SHARE, Market.US),
    dry_run: bool = False,
    force: bool = False,
) -> tuple[object, ...]:
    slots = due_evaluations(now, markets=markets)
    if dry_run:
        return slots
    results = []
    for slot in slots:
        service = service_factory(slot, force)
        results.append(service.evaluate(slot.market, slot.as_of_time, slot.session_slot))
    return tuple(results)


class _SessionDataPort:
    def __init__(self, daily, intraday=None) -> None:
        self.daily = daily
        self.intraday = intraday or daily

    def load_snapshot(self, market, as_of_time, session_slot):
        adapter = self.intraday if "intraday" in session_slot.lower() else self.daily
        return adapter.load_snapshot(market, as_of_time, session_slot)


def _load_environment() -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv(_PROJECT_ROOT / ".env", override=False)
    except ImportError:
        return


def _default_service_factory(slot: EvaluationSlot, force: bool):
    """Compose production adapters lazily so calendar-only dry runs stay offline."""

    _load_environment()
    from tradingagents.harness.market_warning.adapters.data_cache import RawDataCache
    from tradingagents.harness.market_warning.adapters.feishu_notifier import FeishuNotifier
    from tradingagents.harness.market_warning.adapters.minimax_reasoning import MiniMaxReasoningAdapter
    from tradingagents.harness.market_warning.adapters.sqlite_repository import SQLiteWarningRepository
    from tradingagents.harness.market_warning.adapters.tushare_data import TushareAShareDataAdapter
    from tradingagents.harness.market_warning.adapters.us_market_data import YahooUSDataAdapter
    from tradingagents.harness.market_warning.adapters.realtime_quote import RealtimeAShareDataAdapter
    from tradingagents.harness.market_warning.features import AShareFeatureStrategy, USFeatureStrategy
    from tradingagents.harness.market_warning.probability import SklearnProbabilityModel
    from tradingagents.harness.market_warning.service import MarketWarningService

    repository = SQLiteWarningRepository()
    cache = RawDataCache(_PROJECT_ROOT / "harness_data/market_warning/raw")
    if slot.market == Market.A_SHARE:
        from tradingagents.dataflows.tushare_vendor import _get_tushare_api

        pro = _get_tushare_api()
        daily = TushareAShareDataAdapter(pro=pro, cache=cache)
        data_port = _SessionDataPort(daily, RealtimeAShareDataAdapter(pro=pro))
        strategy = AShareFeatureStrategy()
    else:
        daily = YahooUSDataAdapter(cache=cache)
        data_port = _SessionDataPort(daily)
        strategy = USFeatureStrategy()

    zone = _MARKET_ZONES[slot.market]

    def load_history(_market: Market, as_of_time: datetime):
        local_date = as_of_time.astimezone(zone).date()
        return tuple(daily.backfill(local_date - timedelta(days=420), local_date - timedelta(days=1)))

    try:
        reasoning = MiniMaxReasoningAdapter.from_environment(
            breaker=repository.circuit_breaker("minimax-m3")
        )
    except Exception:
        reasoning = None
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
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true", help="Retry a matching failed alert")
    args = parser.parse_args(argv)
    try:
        now = _parse_time(args.at)
        results = run_due_evaluations(
            now,
            service_factory=_default_service_factory,
            markets=_markets(args.market),
            dry_run=args.dry_run,
            force=args.force,
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
