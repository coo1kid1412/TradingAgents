#!/usr/bin/env python3
"""Guarded, idempotent cron installation for market-warning rule V1."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import tempfile
from pathlib import Path
from typing import Mapping, Sequence

from tradingagents.harness.market_warning.adapters.sqlite_repository import (
    SQLiteWarningRepository,
)
from tradingagents.harness.market_warning.domain import Market
from tradingagents.harness.market_warning.readiness import check_production_readiness
from tradingagents.harness.market_warning.rule_policy import load_rule_manifest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = PROJECT_ROOT / "harness_data/tradingagents.db"
DEFAULT_MANIFEST = (
    PROJECT_ROOT / "tradingagents/harness/market_warning/rule_manifest_v1.json"
)
MARKER_START = "# BEGIN TradingAgents market-warning rule-v1"
MARKER_END = "# END TradingAgents market-warning rule-v1"


def _load_environment() -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv(PROJECT_ROOT / ".env", override=False)
    except ImportError:
        pass


def _feishu_configured(environ: Mapping[str, str]) -> bool:
    if str(environ.get("FEISHU_MARKET_RISK_WEBHOOK", "")).strip():
        return True
    return all(
        str(environ.get(name, "")).strip()
        for name in ("FEISHU_APP_ID", "FEISHU_APP_SECRET", "FEISHU_USER_OPEN_ID")
    )


def _db_path_writable(path: Path) -> bool:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            return path.is_file() and os.access(path, os.W_OK)
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".warning-db-probe-"):
            return True
    except OSError:
        return False


def validate_install_preconditions(
    mode: str | None,
    readiness: Mapping[str, object],
    db_path: Path,
    environ: Mapping[str, str],
    manifest_path: Path,
) -> list[str]:
    failures: list[str] = []
    if mode != "rule_v1/notify":
        failures.append("mode must be explicitly set to rule_v1/notify")
    if readiness.get("ready") is not True or readiness.get("mode") != "rule_v1/notify":
        failures.append("rule_v1/notify readiness did not pass")
    try:
        manifest = load_rule_manifest(manifest_path)
    except Exception:
        manifest = None
        failures.append("rule manifest is invalid")
    if manifest is not None and (
        readiness.get("engine_version") != manifest.engine_version
        or readiness.get("manifest_sha256") != manifest.manifest_sha256
    ):
        failures.append("readiness checksum or engine version does not match the manifest")
    if not _db_path_writable(Path(db_path)):
        failures.append("database path is not writable")
    if not _feishu_configured(environ):
        failures.append("Feishu delivery is not configured")
    return failures


def validate_gate_activation_preconditions(
    mode: str | None,
    readiness: Mapping[str, object],
    manifest_path: Path,
) -> list[str]:
    failures: list[str] = []
    if mode != "rule_v1/gate":
        failures.append("mode must be explicitly set to rule_v1/gate")
    if readiness.get("ready") is not True or readiness.get("mode") != "rule_v1/gate":
        failures.append("rule_v1/gate readiness did not pass")
    soak_sessions = readiness.get("soak_sessions")
    if (
        isinstance(soak_sessions, bool)
        or not isinstance(soak_sessions, int)
        or soak_sessions < 10
    ):
        failures.append("gate activation requires at least 10 soak sessions")
    audit = readiness.get("soak_audit")
    if not isinstance(audit, Mapping):
        failures.append("gate activation requires a persisted soak audit")
    else:
        success_rate = audit.get("scan_success_rate")
        counts = tuple(
            audit.get(name)
            for name in ("duplicate_runs", "overlap_skipped", "stale_misjudgments")
        )
        valid_rate = (
            not isinstance(success_rate, bool)
            and isinstance(success_rate, (int, float))
            and success_rate >= 0.98
        )
        valid_counts = all(
            not isinstance(value, bool) and isinstance(value, int) and value == 0
            for value in counts
        )
        if not valid_rate or not valid_counts:
            failures.append("persisted soak audit does not meet activation gates")
    try:
        manifest = load_rule_manifest(manifest_path)
    except Exception:
        manifest = None
        failures.append("rule manifest is invalid")
    if manifest is not None and (
        readiness.get("engine_version") != manifest.engine_version
        or readiness.get("manifest_sha256") != manifest.manifest_sha256
    ):
        failures.append("readiness checksum or engine version does not match the manifest")
    return failures


def _cron_block(project_root: Path) -> tuple[str, ...]:
    root = shlex.quote(str(project_root.resolve()))
    command = (
        f"cd {root} && .venv/bin/python -m "
        "tradingagents.harness.market_warning.runner --market all --mode rule_v1 "
        ">> harness_data/logs/market_warning_rule_v1.log 2>&1"
    )
    return (MARKER_START, f"*/5 * * * 1-5 {command}", MARKER_END)


def render_crontab(existing: str, project_root: Path, *, uninstall: bool) -> str:
    output: list[str] = []
    inside = False
    for line in existing.splitlines():
        if line.strip() == MARKER_START:
            if inside:
                raise ValueError("invalid nested market-warning V1 cron marker")
            inside = True
            continue
        if line.strip() == MARKER_END:
            if not inside:
                raise ValueError("unbalanced market-warning V1 cron marker")
            inside = False
            continue
        if not inside:
            output.append(line)
    if inside:
        raise ValueError("unbalanced market-warning V1 cron marker")
    while output and not output[-1].strip():
        output.pop()
    if not uninstall:
        if output:
            output.append("")
        output.extend(_cron_block(project_root))
    return "\n".join(output) + ("\n" if output else "")


def activate_notify_engine(repository, manifest_path: Path, readiness: Mapping[str, object]) -> None:
    manifest = load_rule_manifest(manifest_path)
    repository.register_rule_engine(
        {
            "engine_version": manifest.engine_version,
            "market": manifest.market,
            "manifest_sha256": manifest.manifest_sha256,
            "metrics": {
                "readiness_mode": readiness.get("mode"),
                "feature_version": readiness.get("feature_version"),
            },
        }
    )
    repository.activate_rule_engine(manifest.engine_version, "notify")


def activate_gate_engine(repository, manifest_path: Path, readiness: Mapping[str, object]) -> None:
    failures = validate_gate_activation_preconditions(
        "rule_v1/gate", readiness, manifest_path
    )
    if failures:
        raise ValueError("; ".join(failures))
    manifest = load_rule_manifest(manifest_path)
    repository.activate_rule_engine(manifest.engine_version, "gate")


def deactivate_notify_engine(repository) -> None:
    repository.deactivate_rule_engine(Market.A_SHARE, "notify")
    repository.deactivate_rule_engine(Market.A_SHARE, "gate")


def _read_crontab() -> str:
    try:
        result = subprocess.run(
            ("crontab", "-l"),
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as error:
        raise RuntimeError(f"unable to read crontab: {error}") from error
    if result.returncode == 0:
        return result.stdout
    detail = (result.stderr or result.stdout or "crontab read failed").strip()
    if result.returncode == 1 and "no crontab for" in detail.lower():
        return ""
    raise RuntimeError(f"unable to read crontab: {detail}")


def _write_crontab(content: str) -> None:
    subprocess.run(
        ("crontab", "-"),
        input=content,
        text=True,
        check=True,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Install guarded rule-V1 warning scans.")
    parser.add_argument("--mode", choices=("rule_v1/notify", "rule_v1/gate"))
    parser.add_argument("--db", default=str(DEFAULT_DB))
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--rule-evaluation")
    parser.add_argument("--data-smoke")
    parser.add_argument("--runtime-benchmark")
    parser.add_argument("--yes", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--uninstall", action="store_true")
    parser.add_argument("--activate-gate", action="store_true")
    args = parser.parse_args(argv)

    if args.uninstall and args.activate_gate:
        parser.error("--uninstall and --activate-gate are mutually exclusive")
    existing = _read_crontab()
    if args.uninstall:
        updated = render_crontab(existing, PROJECT_ROOT, uninstall=True)
        if not args.dry_run:
            _write_crontab(updated)
            try:
                deactivate_notify_engine(SQLiteWarningRepository(args.db))
            except Exception:
                print(
                    json.dumps(
                        {
                            "status": "partially_uninstalled",
                            "schedule_removed": True,
                            "notify_deactivation": "failed",
                            "next_action": "rerun --uninstall",
                        },
                        ensure_ascii=False,
                    )
                )
                return 3
        print(json.dumps({"status": "uninstalled" if not args.dry_run else "dry_run"}))
        return 0

    _load_environment()
    repository = SQLiteWarningRepository(args.db)
    readiness_mode = "rule_v1/gate" if args.activate_gate else "rule_v1/notify"
    readiness_options = {
        "mode": readiness_mode,
        "rule_manifest": args.manifest,
    }
    if args.rule_evaluation:
        readiness_options["rule_evaluation_path"] = args.rule_evaluation
    if args.data_smoke:
        readiness_options["data_smoke_path"] = args.data_smoke
    if args.runtime_benchmark:
        readiness_options["runtime_benchmark_path"] = args.runtime_benchmark
    readiness = check_production_readiness(repository, **readiness_options)
    if args.activate_gate:
        failures = validate_gate_activation_preconditions(
            args.mode,
            readiness,
            Path(args.manifest),
        )
        if failures:
            print(
                json.dumps(
                    {"status": "blocked", "failures": failures}, ensure_ascii=False
                )
            )
            return 2
        if not args.yes and not args.dry_run:
            parser.error("--yes is required for gate activation")
        if args.dry_run:
            print(
                json.dumps(
                    {
                        "status": "dry_run",
                        "mode": "rule_v1/gate",
                        "cron_changed": False,
                    },
                    ensure_ascii=False,
                )
            )
            return 0
        activate_gate_engine(repository, Path(args.manifest), readiness)
        print(
            json.dumps(
                {
                    "status": "gate_activated",
                    "mode": "rule_v1/gate",
                    "cron_changed": False,
                },
                ensure_ascii=False,
            )
        )
        return 0
    failures = validate_install_preconditions(
        args.mode,
        readiness,
        Path(args.db),
        os.environ,
        Path(args.manifest),
    )
    if failures:
        print(json.dumps({"status": "blocked", "failures": failures}, ensure_ascii=False))
        return 2
    if not args.yes and not args.dry_run:
        parser.error("--yes is required for installation")

    updated = render_crontab(existing, PROJECT_ROOT, uninstall=False)
    if args.dry_run:
        print(json.dumps({"status": "dry_run", "crontab": updated}, ensure_ascii=False))
        return 0
    (PROJECT_ROOT / "harness_data/logs").mkdir(parents=True, exist_ok=True)
    _write_crontab(updated)
    try:
        activate_notify_engine(repository, Path(args.manifest), readiness)
    except Exception:
        _write_crontab(existing)
        raise
    active_gate = repository.load_active_rule_engine(Market.A_SHARE, "gate")
    print(
        json.dumps(
            {
                "status": "installed",
                "mode": "rule_v1/notify",
                "gate_active": active_gate is not None,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
