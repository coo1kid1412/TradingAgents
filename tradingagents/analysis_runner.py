"""Observable single-stock jobs for OpenClaw's independent background exec mode."""

import argparse
import datetime
import fcntl
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import traceback
import uuid
from zoneinfo import ZoneInfo


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class AnalysisAlreadyRunning(RuntimeError):
    pass


def _now():
    return datetime.datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds")


def _write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _log_tail(path):
    if not path.exists():
        return ""
    with path.open("rb") as stream:
        stream.seek(0, os.SEEK_END)
        stream.seek(max(0, stream.tell() - 32768))
        return stream.read().decode("utf-8", errors="replace")


def _emit_status(status):
    try:
        print(json.dumps(status, ensure_ascii=False), flush=True)
    except OSError:
        # Prevent Python's shutdown flush from converting a successful job to exit 120.
        try:
            with open(os.devnull, "w") as sink:
                os.dup2(sink.fileno(), sys.stdout.fileno())
        except (OSError, ValueError):
            pass


def _verified_report(path):
    if not path:
        return None
    decision = Path(path) / "5_portfolio" / "decision.md"
    return str(Path(path).resolve()) if decision.is_file() and decision.stat().st_size else None


def _outcome(run_dir, ticker):
    result_path = run_dir / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8")) if result_path.exists() else {}
    log = _log_tail(run_dir / "output.log")
    report = _verified_report(result.get("report_path"))
    if report is None:
        matches = re.findall(rf"^\[{re.escape(ticker)}\] 报告已保存至: (.+)$", log, re.MULTILINE)
        report = _verified_report(matches[-1]) if matches else None
    return result, report, "sent" if re.search(r"^\[.*\] ✓ 飞书已发送", log, re.MULTILINE) else "not_confirmed"


def read_status(ticker, *, project_root=PROJECT_ROOT):
    run_root = Path(project_root) / "logs" / "analysis_runs"
    for path in sorted(run_root.glob("*/status.json"), reverse=True):
        status = json.loads(path.read_text(encoding="utf-8"))
        if status["ticker"] != ticker:
            continue
        if status["status"] == "running":
            with (path.parent / ".running.lock").open("a+") as lock:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    pass
                else:
                    result, report, delivery = _outcome(path.parent, ticker)
                    completed = result.get("success") is True and report is not None
                    status.update({
                        "status": "succeeded" if completed else "partial" if report else "failed" if result else "interrupted",
                        "exit_reason": "worker_result_recovered" if result else "supervisor_and_worker_not_running",
                        "report_path": report, "delivery_status": delivery, "returncode": None,
                    })
        phase = 0
        log_path = Path(status["log_path"])
        if log_path.exists():
            with log_path.open(encoding="utf-8", errors="replace") as stream:
                for line in stream:
                    match = re.match(r"^\[[^\]]+\] (?:✓ )?Phase (\d+)/13", line)
                    if match:
                        phase = max(phase, int(match.group(1)))
        status["last_completed_phase"] = phase
        return status
    return {"ticker": ticker, "status": "not_found"}


def _notify_abnormal_failure(ticker, reason, log_path):
    # The normal pipeline already notifies its own failures. Only cover lost workers here.
    try:
        import main  # Loads the existing project .env without copying credentials.
        from tradingagents.harness.market_risk_daily import _send_feishu_message

        _send_feishu_message(
            f"【TA 分析失败】{ticker}\n"
            f"原因：分析进程异常退出（{reason}），具体杀因尚未确认。\n"
            f"最终完成状态未确认，请核对已有报告与交付记录。\n排查日志：{log_path}"
        )
    except Exception as error:
        print(f"失败通知未确认：{type(error).__name__}", flush=True)


def _worker(project_root, run_dir, ticker, analysis_date, rounds):
    os.chdir(project_root)
    sys.path.insert(0, str(project_root))
    try:
        import main

        config = main._build_config()
        config["max_debate_rounds"] = rounds
        _, success, result = main.analyze_single_stock(ticker, analysis_date, config)
        if success:
            decision = Path(result) / "5_portfolio" / "decision.md"
            if not decision.is_file() or decision.stat().st_size == 0:
                raise RuntimeError("Analysis returned success without a nonempty decision.md")
        _write_json(run_dir / "result.json", {
            "success": success,
            "report_path": str(Path(result).resolve()) if success else None,
        })
        return 0 if success else 1
    except Exception:
        traceback.print_exc()
        return 1


def run_job(ticker, analysis_date, rounds=2, *, project_root=PROJECT_ROOT):
    project_root = Path(project_root).resolve()
    run_root = project_root / "logs" / "analysis_runs"
    run_root.mkdir(parents=True, exist_ok=True)
    with (run_root / ".active.lock").open("a+") as active:
        try:
            fcntl.flock(active, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise AnalysisAlreadyRunning("已有 TA 分析运行中；请先查状态，不要重复启动") from error

        run_id = datetime.datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d_%H%M%S_%f")
        run_id += "_" + uuid.uuid4().hex[:8]
        run_dir = run_root / run_id
        run_dir.mkdir()
        log_path = run_dir / "output.log"
        status_path = run_dir / "status.json"
        status = {
            "run_id": run_id, "ticker": ticker, "analysis_date": analysis_date,
            "bull_bear_rounds": rounds, "started_at": _now(), "status": "running",
            "supervisor_pid": os.getpid(), "log_path": str(log_path),
            "delivery_status": "not_confirmed",
        }
        with (run_dir / ".running.lock").open("a+") as running, log_path.open("ab", buffering=0) as log:
            fcntl.flock(running, fcntl.LOCK_EX | fcntl.LOCK_NB)
            _write_json(status_path, status)
            command = [
                sys.executable, "-u", str(Path(__file__).resolve()), "_worker",
                "--ticker", ticker, "--date", analysis_date, "--rounds", str(rounds),
                "--project-root", str(project_root), "--run-dir", str(run_dir),
            ]
            environment = dict(os.environ, TA_RUN_LOG=str(log_path))
            try:
                # Keep locks in the worker too, so a lost supervisor cannot allow duplicates.
                child = subprocess.Popen(
                    command, cwd=project_root, env=environment,
                    stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                    pass_fds=(active.fileno(), running.fileno()),
                )
                status["worker_pid"] = child.pid
                _write_json(status_path, status)
                _emit_status(status)
                returncode = child.wait()  # No fixed whole-analysis timeout.
                reason = f"signal:{signal.Signals(-returncode).name}" if returncode < 0 else f"exit:{returncode}"
            except Exception:
                traceback.print_exc(file=sys.stderr)
                returncode, reason = 1, "supervisor_error"

            result, report, delivery = _outcome(run_dir, ticker)
            success = returncode == 0 and result.get("success") is True and report is not None
            status.update({
                "status": "succeeded" if success else "partial" if report else "failed", "finished_at": _now(),
                "returncode": returncode, "exit_reason": reason,
                "report_path": report, "delivery_status": delivery,
            })
            _write_json(status_path, status)
            if not success and not result and delivery != "sent":
                _notify_abnormal_failure(ticker, reason, log_path)
        _emit_status(status)
        return 0 if success else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["run", "status", "_worker"])
    parser.add_argument("--ticker", required=True)
    parser.add_argument("--date", default=_now()[:10], type=lambda value: datetime.date.fromisoformat(value).isoformat())
    parser.add_argument("--rounds", type=int, choices=[1, 2, 3], default=2)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT, help=argparse.SUPPRESS)
    parser.add_argument("--run-dir", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if not re.fullmatch(r"[A-Za-z0-9^][A-Za-z0-9.^=\-]*", args.ticker):
        parser.error("股票代码格式无效")
    if args.action == "status":
        print(json.dumps(read_status(args.ticker, project_root=args.project_root), ensure_ascii=False, indent=2))
        return 0
    if args.action == "_worker":
        if args.run_dir is None:
            parser.error("_worker requires --run-dir")
        return _worker(args.project_root, args.run_dir, args.ticker, args.date, args.rounds)
    try:
        return run_job(args.ticker, args.date, args.rounds, project_root=args.project_root)
    except AnalysisAlreadyRunning as error:
        print(json.dumps({"status": "already_running", "message": str(error)}, ensure_ascii=False))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
