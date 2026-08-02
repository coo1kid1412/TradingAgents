#!/usr/bin/env python3
"""Probe point-in-time A-share inputs required by rule-V1 notifications."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Callable, Sequence
from zoneinfo import ZoneInfo

import pandas as pd

from tradingagents.dataflows.intraday_quote import fetch_intraday_quote
from tradingagents.harness.market_warning.adapters.tushare_realtime_breadth import (
    build_premarket_baseline,
    load_realtime_cross_section,
    probe_rt_k_permission,
)
from tradingagents.harness.market_warning.calendars import session_resolvers
from tradingagents.harness.market_warning.domain import Market


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SHANGHAI = ZoneInfo("Asia/Shanghai")


def _load_environment() -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv(PROJECT_ROOT / ".env", override=False)
    except ImportError:
        pass


def _normalized_time(value: object) -> datetime | None:
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(timestamp):
        return None
    return (
        timestamp.tz_localize(SHANGHAI).to_pydatetime()
        if timestamp.tzinfo is None
        else timestamp.tz_convert(SHANGHAI).to_pydatetime()
    )


def run_data_probe(
    pro,
    as_of_time: datetime,
    *,
    quote_loader: Callable = fetch_intraday_quote,
    permission_probe: Callable = probe_rt_k_permission,
    baseline_builder: Callable = build_premarket_baseline,
    cross_section_loader: Callable = load_realtime_cross_section,
    previous_session: Callable | None = None,
    cache_root: Path | str = PROJECT_ROOT / "harness_data/market_warning/baselines",
) -> dict[str, object]:
    if as_of_time.tzinfo is None or as_of_time.utcoffset() is None:
        raise ValueError("as_of_time must be timezone-aware")
    local = as_of_time.astimezone(SHANGHAI)
    if previous_session is None:
        _, previous_session, _ = session_resolvers(Market.A_SHARE)
    failures: list[str] = []
    source_times: dict[str, str] = {}

    try:
        quote = quote_loader(
            symbol="000001.SH",
            analysis_date=local.date().isoformat(),
            tushare_api=pro,
            now=local,
        )
    except Exception:
        quote = None
    quote_time = getattr(quote, "quote_time", None)
    index_staleness = None
    index_ready = False
    if isinstance(quote_time, datetime) and quote_time.tzinfo is not None:
        normalized_quote_time = quote_time.astimezone(SHANGHAI)
        index_staleness = max(0.0, (local - normalized_quote_time).total_seconds() / 60.0)
        index_ready = (
            normalized_quote_time.date() == local.date() and index_staleness <= 5.0
        )
        source_times["index_quote"] = normalized_quote_time.isoformat()
    if not index_ready:
        failures.append("index_realtime")

    permission = permission_probe(pro, local)
    permission_status = str(getattr(permission, "status", "unavailable"))
    if permission_status != "available":
        failures.append("rt_k_permission")

    baseline = None
    try:
        baseline = baseline_builder(
            pro,
            trade_date=local.date(),
            previous_session=previous_session,
            cache_root=cache_root,
        )
    except Exception:
        failures.append("premarket_baseline")

    completed_date = (
        baseline.completed_trade_date.isoformat()
        if baseline is not None
        else previous_session(local.date()).isoformat()
    )
    stk_limit_available = bool(
        baseline is not None
        and "down_limit" in baseline.frame
        and pd.to_numeric(baseline.frame["down_limit"], errors="coerce").notna().any()
    )
    if not stk_limit_available:
        failures.append("stk_limit")

    coverage = 0.0
    breadth_staleness = None
    if permission_status == "available" and baseline is not None:
        try:
            cross_section = cross_section_loader(pro, baseline, local)
        except Exception:
            cross_section = pd.DataFrame()
        if isinstance(cross_section, pd.DataFrame) and not cross_section.empty:
            universe = int(cross_section.attrs.get("universe_size") or baseline.universe_size)
            observed = (
                int(cross_section["ts_code"].nunique())
                if "ts_code" in cross_section
                else len(cross_section)
            )
            coverage = observed / universe * 100.0 if universe > 0 else 0.0
            times = tuple(
                value
                for value in (
                    _normalized_time(item)
                    for item in cross_section.get("data_time", pd.Series(dtype=object))
                )
                if value is not None and value <= local
            )
            if times:
                latest = max(times)
                breadth_staleness = max(0.0, (local - latest).total_seconds() / 60.0)
                source_times["realtime_cross_section"] = latest.isoformat()
    if coverage < 80.0:
        failures.append("realtime_breadth_coverage")
    if breadth_staleness is None or breadth_staleness > 5.0:
        failures.append("realtime_breadth_staleness")

    failures = list(dict.fromkeys(failures))
    return {
        "ready": not failures,
        "market": "a_share",
        "as_of_time": local.isoformat(),
        "index_realtime_ready": index_ready,
        "index_realtime_source": getattr(quote, "source", None),
        "index_realtime_staleness_minutes": index_staleness,
        "rt_k_permission": permission_status,
        "rt_k_probe_rows": int(getattr(permission, "row_count", 0)),
        "realtime_breadth_coverage_pct": round(coverage, 4),
        "realtime_breadth_staleness_minutes": breadth_staleness,
        "stk_limit_available": stk_limit_available,
        "latest_completed_trade_date": completed_date,
        "source_times": source_times,
        "failures": failures,
    }


def _parse_time(value: str | None) -> datetime:
    if value is None:
        return datetime.now(SHANGHAI)
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("--as-of must include a timezone offset")
    return parsed


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Probe A-share warning data readiness.")
    parser.add_argument("--market", choices=("a_share",), default="a_share")
    parser.add_argument("--as-of")
    parser.add_argument("--output")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    _load_environment()
    from tradingagents.dataflows.tushare_vendor import _get_tushare_api

    result = run_data_probe(_get_tushare_api(), _parse_time(args.as_of))
    rendered = json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2)
    if args.output:
        target = Path(args.output)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if result["ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
