"""Small-sample quality review for the existing weekly cron entry point.

Reads archived evidence without fetching quotes or invoking an LLM. Use
--no-notify to preview without sending or advancing the delivered baseline.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import logging
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from zoneinfo import ZoneInfo

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
try:
    from dotenv import load_dotenv
    load_dotenv(_PROJECT_ROOT / ".env")
except ImportError:
    pass

logger = logging.getLogger(__name__)
_FEISHU_SCRIPT = _PROJECT_ROOT / ".claude" / "skills" / "harness-weekly-review" / "send_feishu.sh"
_DAILY_LOG = _PROJECT_ROOT / "harness_data" / "daily_update.log"
_OUTPUT_DIR = _PROJECT_ROOT / "harness_data" / "reviews"


def parse_cron_health(text: str, today: _dt.date) -> tuple[bool, str]:
    """Audit seven calendar days of log markers, not the validity of prices."""
    cutoff = today - _dt.timedelta(days=6)
    lines = text.splitlines()
    markers = []
    for index, line in enumerate(lines):
        match = re.search(r"=== Harness Daily Update @ (\d{4}-\d{2}-\d{2})", line)
        if match:
            try:
                markers.append((index, _dt.date.fromisoformat(match.group(1))))
            except ValueError:
                continue
    run_dates = {day for _, day in markers if cutoff <= day <= today}
    blocks = []
    for index, (start, day) in enumerate(markers):
        if cutoff <= day <= today:
            end = markers[index + 1][0] if index + 1 < len(markers) else len(lines)
            blocks.append("\n".join(lines[start:end]))
    recent_text = "\n".join(blocks)
    total_failed = sum(int(value) for value in re.findall(r"'failed':\s*(\d+)", recent_text))
    issues = []
    if len(run_dates) < 5:
        issues.append(f"7 天仅 {len(run_dates)} 次运行记录（应 ≥5）")
    if total_failed:
        issues.append(f"真值采集失败 {total_failed} 次（daily_update 自报 failed）")
    if "Traceback (most recent call last):" in recent_text:
        issues.append("日志含未捕获 Python Traceback")
    if issues:
        return False, " / ".join(issues)
    return True, f"7 天 {len(run_dates)} 次启动记录，日志未见运行错误；数据另行检查"


def check_cron_health(today=None) -> tuple[bool, str]:
    if not _DAILY_LOG.exists():
        return False, "daily_update.log 不存在，无法确认任务运行"
    try:
        text = _DAILY_LOG.read_text(encoding="utf-8", errors="replace")
        return parse_cron_health(text, today or _dt.datetime.now(ZoneInfo("Asia/Shanghai")).date())
    except OSError as exc:
        return False, f"读 daily_update.log 失败：{exc}"


def extract_summary(db_path=None, today=None) -> dict:
    from tradingagents.harness.quality_review import build_snapshot
    return build_snapshot(db_path, today=today)


def build_feishu_message(cron_healthy: bool, cron_desc: str, summary: dict) -> str:
    from tradingagents.harness.quality_review import render_summary
    return render_summary(summary, cron_healthy, cron_desc)


def send_feishu(msg: str) -> tuple[bool, str]:
    if not _FEISHU_SCRIPT.exists():
        return False, f"send_feishu.sh 不存在：{_FEISHU_SCRIPT}"
    try:
        result = subprocess.run([str(_FEISHU_SCRIPT), msg], capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            return True, result.stdout.strip()
        return False, f"exit={result.returncode}, stdout={result.stdout}, stderr={result.stderr}"
    except Exception as exc:
        return False, f"调 send_feishu.sh 异常: {exc}"


def _write_atomic(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        try:
            handle.write(text)
            handle.flush()
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


def main(argv=None) -> int:
    from tradingagents.harness.quality_review import compare_snapshot, render_markdown

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-notify", action="store_true")
    parser.add_argument("--db-path", type=Path)
    parser.add_argument("--output-dir", type=Path, default=_OUTPUT_DIR)
    parser.add_argument("--date", type=_dt.date.fromisoformat, default=_dt.datetime.now(ZoneInfo("Asia/Shanghai")).date())
    args = parser.parse_args(argv)
    state_path = args.output_dir / "quality_last_delivered.json"
    try:
        try:
            previous = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else None
            if previous is not None and (not isinstance(previous, dict) or previous.get("schema_version") != "quality-v1"):
                previous = None
        except (OSError, ValueError):
            logger.warning("上次推送基线不可读，本次重新建立证据基线")
            previous = None
        summary = extract_summary(db_path=args.db_path, today=args.date)
        summary["delta"] = compare_snapshot(summary, previous)
        healthy, description = check_cron_health(args.date)
        report = render_markdown(summary, healthy, description)
        serialized = json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
        report_path = args.output_dir / f"quality_{args.date.isoformat()}.md"
        _write_atomic(report_path, report)
        _write_atomic(report_path.with_suffix(".json"), serialized)
        print(build_feishu_message(healthy, description, summary))
        print(f"\n本机报告：{report_path}")
        if args.no_notify:
            print("预览完成：未调用 LLM、未推送、未修改已通知基线。")
            return 0
        sent, info = send_feishu(build_feishu_message(healthy, description, summary))
        if not sent:
            logger.error("飞书发送失败，保留上次基线：%s", info)
            return 1
        _write_atomic(state_path, serialized)
        return 0
    except Exception as exc:
        logger.exception("质量复盘失败")
        if not args.no_notify:
            send_feishu(f"TradingAgents 质量复盘失败：{type(exc).__name__}；请检查本机 weekly_review.log。")
        return 1


if __name__ == "__main__":
    sys.exit(main())
