"""Bounded command-line entry point; importing does not call APIs or load LLMs."""

import argparse
import json
from pathlib import Path
import re

from dotenv import load_dotenv

from .runner import DEFAULT_ROOT, REPO_ROOT, RunBusy, replay_snapshot, run_daily


def main(argv=None):
    parser = argparse.ArgumentParser(description="A-share small-cap research candidates (no LLM)")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run")
    run.add_argument("--data-date")
    run.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    notification = run.add_mutually_exclusive_group()
    notification.add_argument("--notify", action="store_true", help="send the report to configured Feishu recipient")
    notification.add_argument("--no-notify", action="store_true")
    run.add_argument("--refresh", action="store_true", help="explicitly capture a new immutable snapshot")
    replay = commands.add_parser("replay")
    replay.add_argument("--snapshot", type=Path, required=True)
    replay.add_argument("--as-of")
    replay.add_argument("--output-dir", type=Path)
    check = commands.add_parser("preflight")
    check.add_argument("--data-date")
    args = parser.parse_args(argv)
    load_dotenv(REPO_ROOT / ".env", override=False)
    try:
        if args.command == "replay":
            result = replay_snapshot(args.snapshot, args.as_of, args.output_dir)
        elif args.command == "preflight":
            from .data_source import TushareSource
            from .selector import select_top10
            snapshot = TushareSource().collect(data_date=args.data_date)
            result = snapshot if snapshot.get("status") == "CLOSED" else select_top10(snapshot)
        else:
            result = run_daily(root=args.root, data_date=args.data_date, notify=args.notify, refresh=args.refresh)
        safe = {key: result.get(key) for key in ("status", "data_date", "target_trade_date", "as_of_at",
                "selection_hash", "report_dir", "snapshot_path", "delivery_status", "delivery_error", "warnings")}
        safe["candidate_count"] = len(result.get("top10", []))
        print(json.dumps(safe, ensure_ascii=False))
        return 0 if result.get("status") in {"SUCCESS", "INSUFFICIENT", "CLOSED"} and result.get("delivery_status") not in {"failed", "unknown", "expired"} else 2
    except RunBusy:
        print(json.dumps({"status": "BUSY"}))
        return 3
    except Exception as error:
        code = getattr(error, "safe_code", "SMALL_CAP_RUN_FAILED")
        code = code if isinstance(code, str) and re.fullmatch(r"[A-Z][A-Z0-9_]{0,63}", code) else "SMALL_CAP_RUN_FAILED"
        endpoint = getattr(error, "endpoint", None)
        endpoint = endpoint if endpoint in {"trade_cal", "stock_basic", "bak_basic", "daily", "daily_basic",
                                           "stock_st", "suspend_d", "stk_limit", "income_vip"} else None
        print(json.dumps({"status": "ERROR", "error": code, "endpoint": endpoint}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
