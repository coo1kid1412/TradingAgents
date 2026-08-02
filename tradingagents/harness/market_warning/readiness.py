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
from .rule_policy import load_rule_manifest


_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_ARTIFACT_ROOT = _PROJECT_ROOT / "harness_data/models/market_warning"
_HORIZONS = ("1d", "3d")
_DEFAULT_RULE_MANIFEST = Path(__file__).with_name("rule_manifest_v1.json")
_DEFAULT_RULE_EVALUATION = _DEFAULT_ARTIFACT_ROOT / "rule-v1-evaluation.json"
_DEFAULT_DATA_SMOKE = _DEFAULT_ARTIFACT_ROOT / "rule-v1-data-smoke.json"
_DEFAULT_RUNTIME_BENCHMARK = _DEFAULT_ARTIFACT_ROOT / "rule-v1-benchmark.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _check_model_readiness(
    repository: SQLiteWarningRepository,
    artifact_root: Path | str,
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
        "mode": "model",
        "model_version": model_version,
        "active_models": len(records),
        "feature_version": FEATURE_VERSION,
        "failures": failures,
    }


def _read_json(path: Path | str, label: str, failures: list[str]) -> dict[str, object] | None:
    candidate = Path(path)
    if not candidate.is_file():
        failures.append(f"missing {label}: {candidate}")
        return None
    try:
        payload = json.loads(candidate.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        failures.append(f"invalid {label}: {candidate}")
        return None
    if not isinstance(payload, dict):
        failures.append(f"invalid {label}: expected JSON object")
        return None
    return payload


def _finite_number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if number == number and number not in (float("inf"), float("-inf")) else None


def _check_rule_readiness(
    repository: SQLiteWarningRepository,
    *,
    mode: str,
    rule_manifest: Path | str,
    rule_evaluation_path: Path | str,
    data_smoke_path: Path | str,
    runtime_benchmark_path: Path | str,
    soak_sessions: int,
) -> dict[str, object]:
    failures: list[str] = []
    try:
        manifest = load_rule_manifest(rule_manifest)
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        manifest = None
        failures.append("invalid rule manifest")
    evaluation = _read_json(rule_evaluation_path, "rule evaluation", failures)
    smoke = _read_json(data_smoke_path, "data smoke", failures)
    benchmark = _read_json(runtime_benchmark_path, "runtime benchmark", failures)

    gates = evaluation.get("production_gates") if evaluation is not None else None
    if not isinstance(gates, dict):
        if evaluation is not None:
            failures.append("rule evaluation is missing production_gates")
        gates = {}
    if manifest is not None and evaluation is not None:
        if evaluation.get("engine_version") != manifest.engine_version:
            failures.append("rule engine version mismatch")
        if evaluation.get("manifest_sha256") != manifest.manifest_sha256:
            failures.append("rule manifest checksum mismatch")

    lift = _finite_number(gates.get("lift"))
    if lift is None or lift <= 2.0:
        failures.append("frozen rule lift must be greater than 2")
    alerts_per_month = _finite_number(gates.get("alerts_per_month"))
    if alerts_per_month is None or alerts_per_month > 6.0:
        failures.append("monthly alert budget must not exceed 6")
    if smoke is not None and smoke.get("ready") is not True:
        failures.append("data smoke is not ready")
    if benchmark is not None:
        p95_seconds = _finite_number(benchmark.get("p95_seconds"))
        if p95_seconds is None or p95_seconds >= 30.0:
            failures.append("fast-path P95 must be less than 30 seconds")
        llm_calls = benchmark.get("llm_calls")
        if isinstance(llm_calls, bool) or not isinstance(llm_calls, int) or llm_calls != 0:
            failures.append("fast-path LLM calls must equal 0")
        runs = benchmark.get("runs")
        if isinstance(runs, bool) or not isinstance(runs, int) or runs < 100:
            failures.append("runtime benchmark must include at least 100 runs")

    concentration = _finite_number(gates.get("max_crisis_contribution"))
    active_notify = None
    if mode == "rule_v1/gate":
        if concentration is None or concentration > 0.50:
            failures.append("crisis concentration must not exceed 50%")
        if isinstance(soak_sessions, bool) or not isinstance(soak_sessions, int) or soak_sessions < 10:
            failures.append("gate mode requires at least 10 soak sessions")
        active_notify = repository.load_active_rule_engine(Market.A_SHARE, "notify")
        if (
            manifest is None
            or active_notify is None
            or active_notify.get("engine_version") != manifest.engine_version
            or active_notify.get("manifest_sha256") != manifest.manifest_sha256
        ):
            failures.append("active notify rule must match the gate manifest")

    return {
        "ready": not failures,
        "mode": mode,
        "engine_version": manifest.engine_version if manifest is not None else None,
        "manifest_sha256": manifest.manifest_sha256 if manifest is not None else None,
        "feature_version": FEATURE_VERSION,
        "soak_sessions": soak_sessions,
        "active_notify_version": (
            active_notify.get("engine_version") if active_notify is not None else None
        ),
        "failures": failures,
    }


def check_production_readiness(
    repository: SQLiteWarningRepository,
    artifact_root: Path | str = _DEFAULT_ARTIFACT_ROOT,
    *,
    mode: str = "model",
    rule_manifest: Path | str = _DEFAULT_RULE_MANIFEST,
    rule_evaluation_path: Path | str = _DEFAULT_RULE_EVALUATION,
    data_smoke_path: Path | str = _DEFAULT_DATA_SMOKE,
    runtime_benchmark_path: Path | str = _DEFAULT_RUNTIME_BENCHMARK,
    soak_sessions: int = 0,
) -> dict[str, object]:
    """Check either the existing model set or the frozen rule V1 artifacts."""

    if mode == "model":
        return _check_model_readiness(repository, artifact_root)
    if mode not in {"rule_v1/notify", "rule_v1/gate"}:
        raise ValueError("mode must be model, rule_v1/notify, or rule_v1/gate")
    return _check_rule_readiness(
        repository,
        mode=mode,
        rule_manifest=rule_manifest,
        rule_evaluation_path=rule_evaluation_path,
        data_smoke_path=data_smoke_path,
        runtime_benchmark_path=runtime_benchmark_path,
        soak_sessions=soak_sessions,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check market-warning production readiness.")
    parser.add_argument("--db")
    parser.add_argument("--artifact-root", default=str(_DEFAULT_ARTIFACT_ROOT))
    parser.add_argument(
        "--mode",
        choices=("model", "rule_v1/notify", "rule_v1/gate"),
        default="model",
    )
    parser.add_argument("--rule-manifest", default=str(_DEFAULT_RULE_MANIFEST))
    parser.add_argument("--rule-evaluation", default=str(_DEFAULT_RULE_EVALUATION))
    parser.add_argument("--data-smoke", default=str(_DEFAULT_DATA_SMOKE))
    parser.add_argument("--runtime-benchmark", default=str(_DEFAULT_RUNTIME_BENCHMARK))
    parser.add_argument("--soak-sessions", type=int, default=0)
    args = parser.parse_args(argv)
    result = check_production_readiness(
        SQLiteWarningRepository(args.db),
        args.artifact_root,
        mode=args.mode,
        rule_manifest=args.rule_manifest,
        rule_evaluation_path=args.rule_evaluation,
        data_smoke_path=args.data_smoke,
        runtime_benchmark_path=args.runtime_benchmark,
        soak_sessions=args.soak_sessions,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0 if result["ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
