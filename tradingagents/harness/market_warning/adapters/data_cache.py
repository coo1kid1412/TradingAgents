"""Atomic raw normalized cache for market-warning adapters."""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Mapping

from tradingagents.harness.market_warning.domain import DataStatus, Market, MarketDataPoint, RawMarketSnapshot


SCHEMA_VERSION = "raw-market-cache-v1"
DEFAULT_RAW_CACHE_ROOT = Path("harness_data/market_warning/raw")


def _json_value(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    return value


def _parse_time(value: str | datetime | None) -> datetime | None:
    if value is None or isinstance(value, datetime):
        return value
    return datetime.fromisoformat(value)


class RawDataCache:
    """Store normalized vendor rows under market/dataset/year partitions."""

    def __init__(self, root: Path | str = DEFAULT_RAW_CACHE_ROOT) -> None:
        self.root = Path(root)

    def data_path(self, market: Market, dataset: str, year: int, cache_key: str) -> Path:
        safe_key = cache_key.replace("/", "_").replace("\\", "_")
        return self.root / Market(market).value / dataset / str(year) / f"{safe_key}.jsonl"

    def write_rows(
        self,
        *,
        market: Market,
        dataset: str,
        year: int,
        cache_key: str,
        rows: Iterable[Mapping[str, Any]],
        query: Mapping[str, Any],
        source: str,
        fetched_at: datetime,
        snapshot: Mapping[str, Any] | None = None,
    ) -> Path:
        materialized = [dict(row) for row in rows]
        data_times = [
            parsed
            for row in materialized
            if (parsed := _parse_time(row.get("data_time"))) is not None
        ]
        data_path = self.data_path(market, dataset, year, cache_key)
        manifest_path = data_path.with_suffix(".manifest.json")
        data_path.parent.mkdir(parents=True, exist_ok=True)
        manifest = {
            "query": _json_value(query),
            "source": source,
            "fetched_at": fetched_at.isoformat(),
            "min_data_time": min(data_times).isoformat() if data_times else None,
            "max_data_time": max(data_times).isoformat() if data_times else None,
            "rows": len(materialized),
            "schema_version": SCHEMA_VERSION,
        }
        if snapshot is not None:
            manifest["snapshot"] = _json_value(snapshot)
        data_text = "".join(
            json.dumps(_json_value(row), ensure_ascii=False, separators=(",", ":")) + "\n"
            for row in materialized
        )
        self._atomic_write(data_path, data_text)
        self._atomic_write(
            manifest_path,
            json.dumps(manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
        )
        return data_path

    def read_rows(
        self, *, market: Market, dataset: str, year: int, cache_key: str
    ) -> tuple[list[dict[str, Any]], dict[str, Any]] | None:
        data_path = self.data_path(market, dataset, year, cache_key)
        manifest_path = data_path.with_suffix(".manifest.json")
        if not data_path.exists() or not manifest_path.exists():
            return None
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema_version") != SCHEMA_VERSION:
            return None
        rows = [json.loads(line) for line in data_path.read_text(encoding="utf-8").splitlines() if line]
        if len(rows) != manifest.get("rows"):
            return None
        return rows, manifest

    def write_snapshot(
        self,
        *,
        dataset: str,
        cache_key: str,
        snapshot: RawMarketSnapshot,
        query: Mapping[str, Any],
        source: str,
        fetched_at: datetime,
    ) -> Path:
        rows = [
            {
                "market": point.market.value,
                "symbol": point.symbol,
                "field": point.field,
                "value": point.value,
                "data_time": point.data_time,
                "fetched_at": point.fetched_at,
                "source": point.source,
                "quality_status": point.quality_status.value,
                "available_at": point.available_at,
            }
            for point in snapshot.points
        ]
        data_year = min((point.data_time.year for point in snapshot.points), default=snapshot.as_of_time.year)
        return self.write_rows(
            market=snapshot.market,
            dataset=dataset,
            year=data_year,
            cache_key=cache_key,
            rows=rows,
            query=query,
            source=source,
            fetched_at=fetched_at,
            snapshot={
                "market": snapshot.market.value,
                "as_of_time": snapshot.as_of_time,
                "session_slot": snapshot.session_slot,
                "data_status": snapshot.data_status.value,
                "source_times": snapshot.source_times,
            },
        )

    def read_snapshot(
        self, *, market: Market, dataset: str, year: int, cache_key: str
    ) -> RawMarketSnapshot | None:
        cached = self.read_rows(market=market, dataset=dataset, year=year, cache_key=cache_key)
        if cached is None:
            return None
        rows, manifest = cached
        metadata = manifest.get("snapshot")
        if not isinstance(metadata, dict):
            return None
        points = tuple(
            MarketDataPoint(
                market=row["market"],
                symbol=row["symbol"],
                field=row["field"],
                value=row.get("value"),
                data_time=datetime.fromisoformat(row["data_time"]),
                fetched_at=datetime.fromisoformat(row["fetched_at"]),
                source=row["source"],
                quality_status=row.get("quality_status", DataStatus.FRESH.value),
                available_at=_parse_time(row.get("available_at")),
            )
            for row in rows
        )
        return RawMarketSnapshot(
            market=metadata["market"],
            as_of_time=datetime.fromisoformat(metadata["as_of_time"]),
            session_slot=metadata["session_slot"],
            points=points,
            data_status=metadata.get("data_status", DataStatus.FRESH.value),
            source_times={
                source_name: datetime.fromisoformat(timestamp)
                for source_name, timestamp in metadata.get("source_times", {}).items()
            },
        )

    @staticmethod
    def _atomic_write(path: Path, content: str) -> None:
        temp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
            ) as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
                temp_path = Path(handle.name)
            os.replace(temp_path, path)
        finally:
            if temp_path is not None and temp_path.exists():
                temp_path.unlink()
