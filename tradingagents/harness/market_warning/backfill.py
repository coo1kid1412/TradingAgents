"""Point-in-time dataset construction and frozen model promotion gates."""

from __future__ import annotations

import math
import os
import tempfile
import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Protocol, Sequence

import joblib
import pandas as pd

from tradingagents.harness.market_risk import compute_market_risk_snapshot

from .domain import FeatureSnapshot, Market, MarketDataPoint, RawMarketSnapshot
from .features import FEATURE_VERSION
from .ohlc import select_benchmark_symbol
from .training import build_labels


_FEATURE_HISTORY_LIMIT = 253


class FeatureBuilder(Protocol):
    def build(
        self,
        raw: RawMarketSnapshot,
        prior_history: Iterable[RawMarketSnapshot],
    ) -> FeatureSnapshot: ...


@dataclass(frozen=True)
class BackfillAudit:
    market: Market
    input_snapshots: int
    unique_snapshot_keys: int
    output_rows: int
    duplicate_snapshot_keys: int
    missing_close_rows: int
    point_in_time_violations: int
    source_time_start: str | None
    source_time_end: str | None
    missing_feature_rates: Mapping[str, float]

    def __post_init__(self) -> None:
        object.__setattr__(self, "market", Market(self.market))
        object.__setattr__(
            self,
            "missing_feature_rates",
            MappingProxyType(dict(self.missing_feature_rates)),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "market": self.market.value,
            "input_snapshots": self.input_snapshots,
            "unique_snapshot_keys": self.unique_snapshot_keys,
            "output_rows": self.output_rows,
            "duplicate_snapshot_keys": self.duplicate_snapshot_keys,
            "missing_close_rows": self.missing_close_rows,
            "point_in_time_violations": self.point_in_time_violations,
            "source_time_start": self.source_time_start,
            "source_time_end": self.source_time_end,
            "missing_feature_rates": dict(self.missing_feature_rates),
        }


@dataclass(frozen=True)
class BackfillDataset:
    frame: pd.DataFrame
    audit: BackfillAudit


def _available_at(point: MarketDataPoint):
    if point.available_at is None:
        raise ValueError(
            f"point-in-time violation: {point.source}/{point.symbol}/{point.field} "
            "has no explicit available_at"
        )
    return point.available_at


def _validate_snapshot(snapshot: RawMarketSnapshot) -> None:
    for point in snapshot.points:
        available_at = _available_at(point)
        if point.market != snapshot.market:
            raise ValueError("point-in-time violation: point market differs from snapshot market")
        if point.data_time > available_at:
            raise ValueError("point-in-time violation: available_at precedes data_time")
        if available_at > snapshot.as_of_time:
            raise ValueError("point-in-time violation: available_at is after as_of_time")


def _close_point(snapshot: RawMarketSnapshot) -> MarketDataPoint | None:
    symbol = select_benchmark_symbol(snapshot.points, snapshot.market)
    if symbol is None:
        return None
    candidates = [
        point
        for point in snapshot.points
        if point.symbol.upper() == symbol.upper()
        and point.field.strip().lower() in {"index_price", "close", "price", "last"}
        and isinstance(point.value, (int, float))
        and not isinstance(point.value, bool)
        and math.isfinite(float(point.value))
        and float(point.value) > 0
    ]
    return max(candidates, key=lambda point: (point.data_time, _available_at(point)), default=None)


def _old_market_risk(
    market: Market,
    closes: Sequence[float],
    features: Mapping[str, Any],
    as_of_time,
) -> tuple[float | None, bool]:
    realized = features.get("realized_volatility_20d")
    volatility_pct = None
    if isinstance(realized, (int, float)) and not isinstance(realized, bool):
        volatility_pct = float(realized) * math.sqrt(252.0) * 100.0
    breadth = features.get("breadth_up_pct")
    breadth_pct = (
        float(breadth)
        if isinstance(breadth, (int, float)) and not isinstance(breadth, bool)
        else None
    )
    snapshot = compute_market_risk_snapshot(
        market.value,
        ({"close": close} for close in closes),
        breadth_pct=breadth_pct,
        volatility_pct=volatility_pct,
        as_of_date=as_of_time.date().isoformat(),
        as_of_time=as_of_time.isoformat(),
    )
    score = snapshot.get("risk_score")
    return score, bool(isinstance(score, (int, float)) and score >= 4)


def build_point_in_time_dataset(
    market: Market,
    snapshots: Iterable[RawMarketSnapshot],
    strategy: FeatureBuilder,
) -> BackfillDataset:
    """Build one chronological, leakage-audited daily feature dataset."""

    market = Market(market)
    received = tuple(snapshots)
    selected: dict[tuple[object, str], RawMarketSnapshot] = {}
    for snapshot in sorted(received, key=lambda item: (item.as_of_time, item.session_slot)):
        if snapshot.market != market:
            raise ValueError("snapshot market does not match requested dataset market")
        selected[(snapshot.as_of_time, snapshot.session_slot)] = snapshot
    ordered = tuple(selected.values())

    rows: list[dict[str, Any]] = []
    history: list[RawMarketSnapshot] = []
    closes: list[float] = []
    missing_close_rows = 0
    source_times = []
    for raw in ordered:
        _validate_snapshot(raw)
        source_times.extend(raw.source_times.values())
        close_point = _close_point(raw)
        if close_point is None:
            missing_close_rows += 1
            continue
        feature_snapshot = strategy.build(
            raw,
            tuple(history[-_FEATURE_HISTORY_LIMIT:]),
        )
        if feature_snapshot.feature_version != FEATURE_VERSION:
            raise ValueError("feature strategy returned an incompatible feature version")
        feature_available_at = max(
            (_available_at(point) for point in raw.points),
            default=raw.as_of_time,
        )
        close_available_at = _available_at(close_point)
        close = float(close_point.value)
        closes.append(close)
        old_score, old_alert = _old_market_risk(
            market,
            closes,
            feature_snapshot.features,
            raw.as_of_time,
        )
        row = dict(feature_snapshot.features)
        row.update(
            {
                "as_of_time": raw.as_of_time,
                "close": close,
                "feature_available_at": feature_available_at,
                "close_available_at": close_available_at,
                "old_market_risk_score": old_score,
                "old_market_risk_alert": old_alert,
                "crisis_period": (
                    str(raw.as_of_time.year)
                    if raw.as_of_time.year in {2008, 2015, 2020, 2022}
                    else "non_crisis"
                ),
            }
        )
        rows.append(row)
        history.append(raw)

    if not rows:
        raise ValueError("backfill produced no rows with a point-in-time benchmark close")
    frame = pd.DataFrame(rows).sort_values("as_of_time", kind="stable").reset_index(drop=True)
    frame.attrs["feature_version"] = FEATURE_VERSION
    frame.attrs["availability_proof"] = {
        "*": "feature_available_at",
        "close": "close_available_at",
    }
    labeled = build_labels(frame, market)
    feature_columns = sorted(
        column
        for column in frame.columns
        if column
        not in {
            "as_of_time",
            "close",
            "feature_available_at",
            "close_available_at",
            "old_market_risk_score",
            "old_market_risk_alert",
            "crisis_period",
        }
    )
    missing_rates = {
        column: float(frame[column].isna().mean())
        for column in feature_columns
    }
    audit = BackfillAudit(
        market=market,
        input_snapshots=len(received),
        unique_snapshot_keys=len(ordered),
        output_rows=len(labeled),
        duplicate_snapshot_keys=len(received) - len(ordered),
        missing_close_rows=missing_close_rows,
        point_in_time_violations=0,
        source_time_start=min(source_times).isoformat() if source_times else None,
        source_time_end=max(source_times).isoformat() if source_times else None,
        missing_feature_rates=missing_rates,
    )
    return BackfillDataset(labeled, audit)


def write_dataset(frame: pd.DataFrame, path: Path | str) -> None:
    """Atomically persist a DataFrame and its point-in-time audit attributes."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.",
        suffix=".tmp",
        dir=target.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        joblib.dump(frame, temporary)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def _write_json(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(
            json.dumps(dict(payload), ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def run_backfill(
    market: Market,
    start: str | date,
    end: str | date,
    adapter: Any,
    strategy: FeatureBuilder,
    output: Path | str,
) -> dict[str, Any]:
    """Run a resumable adapter backfill and persist its audited dataset."""

    market = Market(market)
    start_date = date.fromisoformat(start) if isinstance(start, str) else start
    end_date = date.fromisoformat(end) if isinstance(end, str) else end
    if end_date < start_date:
        raise ValueError("backfill end date must not be before start date")
    result = build_point_in_time_dataset(
        market,
        adapter.backfill(start_date, end_date),
        strategy,
    )
    target = Path(output)
    write_dataset(result.frame, target)
    audit_path = target.with_suffix(".audit.json")
    audit_payload = result.audit.as_dict()
    _write_json(audit_payload, audit_path)
    return {
        "market": market.value,
        "start": start_date.isoformat(),
        "end": end_date.isoformat(),
        "dataset": str(target),
        "audit": str(audit_path),
        "rows": result.audit.output_rows,
        "duplicate_snapshot_keys": result.audit.duplicate_snapshot_keys,
        "missing_close_rows": result.audit.missing_close_rows,
        "point_in_time_violations": result.audit.point_in_time_violations,
    }


def promotion_failures(reports: Iterable[Mapping[str, Any]]) -> tuple[str, ...]:
    """Return deterministic reasons why the four frozen model reports cannot activate."""

    expected = {(market.value, horizon) for market in Market for horizon in ("1d", "3d")}
    indexed: dict[tuple[str, str], Mapping[str, Any]] = {}
    failures: list[str] = []
    for report in reports:
        key = (str(report.get("market")), str(report.get("horizon")))
        if key in indexed:
            failures.append(f"duplicate evaluation report: {key[0]}/{key[1]}")
        indexed[key] = report
    for market, horizon in sorted(expected - set(indexed)):
        failures.append(f"missing evaluation report: {market}/{horizon}")

    for key in sorted(expected & set(indexed)):
        report = indexed[key]
        label = f"{key[0]}/{key[1]}"
        try:
            brier = float(report["brier_score"])
            constant_brier = float(report["constant_base_rate_brier"])
            auprc = float(report["average_precision"])
            prevalence = float(report["prevalence"])
            calibration = float(report["expected_calibration_error"])
            budget = float(report["monthly_alert_entries"])
            crisis = dict(report["crisis_contribution"])
        except (KeyError, TypeError, ValueError):
            failures.append(f"malformed evaluation report: {label}")
            continue
        if not all(math.isfinite(value) for value in (brier, constant_brier, auprc, prevalence, calibration, budget)):
            failures.append(f"non-finite evaluation metric: {label}")
            continue
        if brier >= constant_brier:
            failures.append(f"Brier gate failed: {label}")
        if auprc <= prevalence:
            failures.append(f"AUPRC gate failed: {label}")
        if calibration > 0.05:
            failures.append(f"calibration gate failed: {label}")
        named_crises = ("2008", "2015", "2020", "2022")
        try:
            crisis_max = max(
                (float(crisis.get(name, 0.0)) for name in named_crises),
                default=0.0,
            )
        except (TypeError, ValueError):
            failures.append(f"malformed crisis contribution: {label}")
        else:
            if crisis_max > 0.50:
                failures.append(f"crisis concentration gate failed: {label}")
        if budget > 6.0:
            failures.append(f"alert budget gate failed: {label}")
    return tuple(failures)
