"""Atomic, generation-checked raw cache for market-warning adapters."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import uuid
from datetime import date, datetime
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Mapping

from tradingagents.harness.market_warning.domain import DataStatus, Market, MarketDataPoint, RawMarketSnapshot


SCHEMA_VERSION = "raw-market-cache-v2"
DEFAULT_RAW_CACHE_ROOT = Path("harness_data/market_warning/raw")


def _json_value(value: Any) -> Any:
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    return value


def _canonical(value: Any) -> Any:
    return json.loads(json.dumps(_json_value(value), ensure_ascii=False, sort_keys=True, separators=(",", ":")))


def _canonical_text(value: Any) -> str:
    return json.dumps(_canonical(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _parse_time(value: str | datetime | None) -> datetime | None:
    if value is None or isinstance(value, datetime):
        return value
    return datetime.fromisoformat(value)


class RawDataCache:
    """Store normalized vendor rows under market/dataset/data-year partitions."""

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
        complete: bool = True,
    ) -> Path:
        materialized = [dict(row) for row in rows]
        data_times = [
            parsed for row in materialized if (parsed := _parse_time(row.get("data_time"))) is not None
        ]
        data_path = self.data_path(market, dataset, year, cache_key)
        manifest_path = data_path.with_suffix(".manifest.json")
        data_path.parent.mkdir(parents=True, exist_ok=True)
        generation = uuid.uuid4().hex
        header = {"generation": generation, "schema_version": SCHEMA_VERSION}
        lines = [
            json.dumps(header, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
            *(
                json.dumps(_json_value(row), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                for row in materialized
            ),
        ]
        data_bytes = ("\n".join(lines) + "\n").encode("utf-8")
        manifest: dict[str, Any] = {
            "query": _canonical(query),
            "source": source,
            "fetched_at": fetched_at.isoformat(),
            "min_data_time": min(data_times).isoformat() if data_times else None,
            "max_data_time": max(data_times).isoformat() if data_times else None,
            "rows": len(materialized),
            "schema_version": SCHEMA_VERSION,
            "generation": generation,
            "data_sha256": hashlib.sha256(data_bytes).hexdigest(),
            "complete": bool(complete),
        }
        if snapshot is not None:
            manifest["snapshot"] = _json_value(snapshot)
        manifest_bytes = json.dumps(
            manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        self._atomic_write(data_path, data_bytes)
        self._atomic_write(manifest_path, manifest_bytes)
        return data_path

    def read_rows(
        self,
        *,
        market: Market,
        dataset: str,
        year: int,
        cache_key: str,
        query: Mapping[str, Any],
    ) -> tuple[list[dict[str, Any]], dict[str, Any]] | None:
        data_path = self.data_path(market, dataset, year, cache_key)
        manifest_path = data_path.with_suffix(".manifest.json")
        if not data_path.exists() or not manifest_path.exists():
            return None
        try:
            data_bytes = data_path.read_bytes()
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if not isinstance(manifest, dict):
                return None
            if manifest.get("schema_version") != SCHEMA_VERSION or manifest.get("complete") is not True:
                return None
            if not isinstance(manifest.get("generation"), str) or not manifest["generation"]:
                return None
            if _canonical_text(manifest.get("query")) != _canonical_text(query):
                return None
            if manifest.get("data_sha256") != hashlib.sha256(data_bytes).hexdigest():
                return None
            decoded = data_bytes.decode("utf-8").splitlines()
            if not decoded:
                return None
            header = json.loads(decoded[0])
            if (
                not isinstance(header, dict)
                or header.get("schema_version") != SCHEMA_VERSION
                or header.get("generation") != manifest["generation"]
            ):
                return None
            rows = [json.loads(line) for line in decoded[1:] if line]
            row_count = manifest.get("rows")
            if (
                any(not isinstance(row, dict) for row in rows)
                or isinstance(row_count, bool)
                or not isinstance(row_count, int)
                or row_count < 0
                or len(rows) != row_count
            ):
                return None
        except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError):
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
        complete: bool = True,
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
            complete=complete,
        )

    def read_snapshot(
        self,
        *,
        market: Market,
        dataset: str,
        year: int,
        cache_key: str,
        query: Mapping[str, Any],
    ) -> RawMarketSnapshot | None:
        cached = self.read_rows(
            market=market, dataset=dataset, year=year, cache_key=cache_key, query=query
        )
        if cached is None:
            return None
        rows, manifest = cached
        metadata = manifest.get("snapshot")
        if not isinstance(metadata, dict):
            return None
        try:
            source_times = metadata.get("source_times", {})
            if not isinstance(source_times, Mapping):
                return None
            if "points" in metadata and not isinstance(metadata["points"], list):
                return None
            if not isinstance(metadata.get("session_slot"), str) or not metadata["session_slot"]:
                return None
            snapshot_market = Market(metadata.get("market"))
            snapshot_status = DataStatus(metadata.get("data_status", DataStatus.FRESH.value))
            if snapshot_market != Market(market):
                return None
            for row in rows:
                if any(
                    not isinstance(row.get(field_name), str) or not row[field_name]
                    for field_name in ("market", "symbol", "field", "source", "data_time", "fetched_at")
                ):
                    return None
                if Market(row["market"]) != snapshot_market:
                    return None
                DataStatus(row.get("quality_status", DataStatus.FRESH.value))
                available_at = row.get("available_at")
                if available_at is not None and not isinstance(available_at, str):
                    return None
            parsed_source_times: dict[str, datetime] = {}
            for source_name, timestamp in source_times.items():
                if not isinstance(source_name, str) or not source_name or not isinstance(timestamp, str):
                    return None
                parsed_source_times[source_name] = datetime.fromisoformat(timestamp)
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
                market=snapshot_market,
                as_of_time=datetime.fromisoformat(metadata["as_of_time"]),
                session_slot=metadata["session_slot"],
                points=points,
                data_status=snapshot_status,
                source_times=parsed_source_times,
            )
        except (AttributeError, KeyError, TypeError, ValueError):
            return None

    @staticmethod
    def _atomic_write(path: Path, content: bytes) -> None:
        temp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
            ) as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
                temp_path = Path(handle.name)
            os.replace(temp_path, path)
        finally:
            if temp_path is not None and temp_path.exists():
                temp_path.unlink()
