"""Fail-closed production readiness check for the scheduled warning runner."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Sequence

from .adapters.sqlite_repository import SQLiteWarningRepository
from .domain import Market
from .features import FEATURE_VERSION


_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_ARTIFACT_ROOT = _PROJECT_ROOT / "harness_data/models/market_warning"
_HORIZONS = ("1d", "3d")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_production_readiness(
    repository: SQLiteWarningRepository,
    artifact_root: Path | str = _DEFAULT_ARTIFACT_ROOT,
) -> dict[str, object]:
    """Verify that one coherent, current four-model set is safe to schedule."""

    root = Path(artifact_root).resolve()
    failures: list[str] = []
    records: list[dict[str, object]] = []
    for market in Market:
        for horizon in _HORIZONS:
            record = repository.load_active_model(market, horizon)
            if record is None:
                failures.append(f"missing active model: {market.value}/{horizon}")
                continue
            records.append(record)
            key = f"{market.value}/{horizon}"
            if record["feature_version"] != FEATURE_VERSION:
                failures.append(f"feature version mismatch: {key}")
            artifact = Path(str(record["artifact_path"])).resolve()
            try:
                artifact.relative_to(root)
            except ValueError:
                failures.append(f"artifact outside production root: {key}")
                continue
            if not artifact.is_file():
                failures.append(f"artifact missing: {key}")
                continue
            if _sha256(artifact) != str(record["artifact_sha256"]):
                failures.append(f"artifact checksum mismatch: {key}")

    versions = {str(record["model_version"]) for record in records}
    if len(records) == 4 and len(versions) != 1:
        failures.append("active models must belong to one model version")
    model_version = next(iter(versions)) if len(versions) == 1 else None
    return {
        "ready": not failures and len(records) == 4,
        "model_version": model_version,
        "active_models": len(records),
        "feature_version": FEATURE_VERSION,
        "failures": failures,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check market-warning production readiness.")
    parser.add_argument("--db")
    parser.add_argument("--artifact-root", default=str(_DEFAULT_ARTIFACT_ROOT))
    args = parser.parse_args(argv)
    result = check_production_readiness(
        SQLiteWarningRepository(args.db),
        args.artifact_root,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0 if result["ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
