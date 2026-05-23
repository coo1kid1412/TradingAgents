"""周末回测复盘自动化脚本（cron 入口）。

每周六上午 10:00 由 cron 触发，一站式完成：
  1. 扫 daily_update.log 检查 cron 健康度
  2. 跑 backtest（重算切片）
  3. 跑 review（生成 markdown 偏差清单）
  4. 直接 import review.generate_insights() 拿结构化数据
  5. 拼装飞书消息正文（cron 健康 + 回测摘要 + Top 3 偏差）
  6. 调 send_feishu.sh 推送
  7. 终端打印执行摘要

调度（用户 crontab）：
  0 10 * * 6 cd /path/to/TradingAgents && .venv/bin/python -m tradingagents.harness.weekly_review

手动跑：
  .venv/bin/python -m tradingagents.harness.weekly_review

设计原则：
- 所有错误都通过飞书报告，不静默吞错
- cron 环境极简，模块顶部显式 load_dotenv（即使本脚本不需要 token，保持一致）
"""

from __future__ import annotations

import datetime as _dt
import logging
import os
import re
import subprocess
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]

# 显式加载 .env（与 daily_update.py 同样的考量）
try:
    from dotenv import load_dotenv
    load_dotenv(_PROJECT_ROOT / ".env")
except ImportError:
    pass

logger = logging.getLogger(__name__)

# 飞书脚本路径（skill 内的 helper）
_FEISHU_SCRIPT = _PROJECT_ROOT / ".claude" / "skills" / "harness-weekly-review" / "send_feishu.sh"
_DAILY_LOG = _PROJECT_ROOT / "harness_data" / "daily_update.log"

# 样本量门槛（与 review.py 保持一致）
_MIN_SAMPLE_SIZE = 5


# ---------------------------------------------------------------------------
# Step 1: cron 健康度
# ---------------------------------------------------------------------------
def check_cron_health() -> tuple[bool, str]:
    """扫描 daily_update.log 最近 7 天，返回 (is_healthy, 健康描述/异常说明)。"""
    if not _DAILY_LOG.exists():
        return False, "daily_update.log 不存在，cron 可能从未跑过"

    try:
        text = _DAILY_LOG.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        return False, f"读 daily_update.log 失败：{e}"

    # 取最后 500 行（够覆盖 7-10 天的运行）
    tail_lines = text.splitlines()[-500:]
    tail_text = "\n".join(tail_lines)

    today = _dt.date.today()
    cutoff = today - _dt.timedelta(days=7)

    # 1) 统计运行次数：每个运行片段以 "=== Harness Daily Update @ YYYY-MM-DD HH:MM:SS ===" 开始
    run_dates: set = set()
    for m in re.finditer(
        r"=== Harness Daily Update @ (\d{4}-\d{2}-\d{2})", tail_text
    ):
        try:
            d = _dt.date.fromisoformat(m.group(1))
            if d >= cutoff:
                run_dates.add(d)
        except ValueError:
            continue

    n_runs = len(run_dates)

    # 2) 检查 vendor 失败
    vendor_fail_matches = re.findall(
        r"vendor 链全部失败（cache 已落后 (\d+) 天）", tail_text
    )

    # 3) 检查 Traceback
    has_traceback = "Traceback (most recent call last):" in tail_text

    # 综合判定
    issues = []
    if n_runs < 5:
        issues.append(f"7 天仅 {n_runs} 次运行（应 ≥5）")
    if vendor_fail_matches:
        worst_lag = max(int(x) for x in vendor_fail_matches)
        issues.append(f"vendor 链失败 {len(vendor_fail_matches)} 次，最长落后 {worst_lag} 天")
    if has_traceback:
        issues.append("日志含未捕获 Python Traceback")

    if not issues:
        return True, f"7 天 {n_runs} 次运行正常，无异常"
    return False, " / ".join(issues)


# ---------------------------------------------------------------------------
# Step 2-3: 跑 backtest + review
# ---------------------------------------------------------------------------
def run_backtest_and_review() -> tuple[bool, str | None]:
    """跑 backtest + review。返回 (success, 错误信息 or None)。"""
    from tradingagents.harness import backtest, review

    try:
        backtest.compute_snapshot()
    except Exception as e:
        return False, f"backtest 报错: {e}"

    try:
        review.save_review_report()
    except Exception as e:
        return False, f"review 报错: {e}"

    return True, None


# ---------------------------------------------------------------------------
# Step 4: 解析 insights（直接复用 review.generate_insights 的结构化输出）
# ---------------------------------------------------------------------------
def extract_summary() -> dict:
    """直接拿 review.generate_insights() 的结构化结果，不再 regex 解析 markdown。"""
    from tradingagents.harness import review

    insights = review.generate_insights()
    overall = insights.get("overall_by_horizon", {})

    summary = {
        "total_samples": sum(r["total_runs"] for r in overall.values()) if overall else 0,
        "hit_rate_by_horizon": {
            h: overall[h]["direction_hit_rate"] if h in overall else None
            for h in ("T", "T+1", "T+5", "T+30")
        },
        "issues_high": [i for i in insights["issues"] if i["severity"] == "high"],
        "issues_medium": [i for i in insights["issues"] if i["severity"] == "medium"],
        "snapshot_date": insights["snapshot_date"],
    }
    return summary


# ---------------------------------------------------------------------------
# Step 5: 拼装飞书消息
# ---------------------------------------------------------------------------
def build_feishu_message(cron_healthy: bool, cron_desc: str, summary: dict) -> str:
    """拼装多行 text 消息。"""
    date_str = summary["snapshot_date"]
    lines: list[str] = []
    lines.append(f"【TradingAgents 周末 Review】{date_str}")
    lines.append("")
    cron_icon = "✓" if cron_healthy else "⚠"
    lines.append(f"cron 健康度：{cron_icon} {cron_desc}")
    lines.append("")

    total = summary["total_samples"]
    if total < _MIN_SAMPLE_SIZE:
        lines.append(f"⚠ 本周样本量仅 {total} 条（< {_MIN_SAMPLE_SIZE}），统计意义弱")
        lines.append("建议：再积累 2-3 周数据后再做正式 review")
    else:
        lines.append("回测核心统计：")
        lines.append(f"- 总样本数：{total}")
        for h in ("T", "T+1", "T+5", "T+30"):
            rate = summary["hit_rate_by_horizon"].get(h)
            if rate is not None:
                lines.append(f"- {h} 命中率：{rate:.0%}")
            else:
                lines.append(f"- {h} 命中率：（无数据）")

        n_high = len(summary["issues_high"])
        n_medium = len(summary["issues_medium"])
        lines.append(f"- 偏差告警：高严重度 {n_high} 条 / 中严重度 {n_medium} 条")

        # Top 3 偏差（优先 high，不够补 medium）
        top3 = (summary["issues_high"] + summary["issues_medium"])[:3]
        if top3:
            lines.append("")
            lines.append("Top 3 偏差：")
            for i, issue in enumerate(top3, 1):
                # message 太长截断到 ~50 字
                msg = issue["message"]
                if len(msg) > 50:
                    msg = msg[:47] + "..."
                lines.append(f"{i}. {msg}")

    lines.append("")
    lines.append(f"报告：harness_data/reviews/backtest_{date_str}.md")
    lines.append("下一步：用 Claude 详细阅读完整报告，决定本周改哪些 prompt/阈值")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Step 6: 发飞书
# ---------------------------------------------------------------------------
def send_feishu(msg: str) -> tuple[bool, str]:
    """调 send_feishu.sh 发消息。返回 (success, 输出/错误)。"""
    if not _FEISHU_SCRIPT.exists():
        return False, f"send_feishu.sh 不存在：{_FEISHU_SCRIPT}"
    try:
        result = subprocess.run(
            [str(_FEISHU_SCRIPT), msg],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode == 0:
            return True, result.stdout.strip()
        return False, f"exit={result.returncode}, stdout={result.stdout}, stderr={result.stderr}"
    except Exception as e:
        return False, f"调 send_feishu.sh 异常: {e}"


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s %(levelname)s %(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    # 压住 vendor 噪音（同 daily_update）
    for noisy in (
        "tradingagents.dataflows.tushare_vendor",
        "tradingagents.dataflows.akshare_vendor",
        "tradingagents.dataflows.interface",
        "yfinance", "urllib3", "httpx",
    ):
        logging.getLogger(noisy).setLevel(logging.CRITICAL)

    print(f"=== Weekly Review @ {_dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n")

    # Step 1: cron 健康度
    cron_healthy, cron_desc = check_cron_health()
    print(f"[1/6] cron 健康度: {'✓' if cron_healthy else '⚠'} {cron_desc}")

    # Step 2-3: backtest + review
    ok, err = run_backtest_and_review()
    if not ok:
        # 命令报错，直接报警
        msg = (
            f"✗ TradingAgents 周末 Review 执行失败\n\n"
            f"错误：{err}\n\n"
            f"cron 健康度：{cron_desc}\n\n"
            f"请手动跑确认：\n"
            f".venv/bin/python -m tradingagents.harness.backtest"
        )
        print(f"[2/6] backtest+review 报错：{err}")
        sent_ok, sent_info = send_feishu(msg)
        print(f"[6/6] 飞书发送: {'✓' if sent_ok else '✗'} {sent_info}")
        return 1
    print("[2/6] backtest+review 跑完")

    # Step 4: 提取 insights
    try:
        summary = extract_summary()
    except Exception as e:
        msg = (
            f"✗ TradingAgents 周末 Review：提取 insights 失败\n\n"
            f"错误：{e}\n\n"
            f"cron 健康度：{cron_desc}"
        )
        print(f"[4/6] insights 提取失败：{e}")
        sent_ok, sent_info = send_feishu(msg)
        print(f"[6/6] 飞书发送: {'✓' if sent_ok else '✗'} {sent_info}")
        return 1
    print(f"[4/6] insights: 样本数={summary['total_samples']}, "
          f"high={len(summary['issues_high'])}, "
          f"medium={len(summary['issues_medium'])}")

    # Step 5: 拼消息
    msg = build_feishu_message(cron_healthy, cron_desc, summary)
    print(f"[5/6] 消息正文已拼装（{len(msg)} 字符）")

    # Step 6: 发飞书
    sent_ok, sent_info = send_feishu(msg)
    print(f"[6/6] 飞书发送: {'✓' if sent_ok else '✗'} {sent_info}")

    return 0 if sent_ok else 1


if __name__ == "__main__":
    sys.exit(main())
