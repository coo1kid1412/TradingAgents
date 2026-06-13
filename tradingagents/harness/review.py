"""从 backtest_metrics 自动识别系统性偏差，生成 markdown 洞察报告供人工 review。

输出文件：docs/backtest_<snapshot_date>.md
"""

from __future__ import annotations

import datetime as _dt
import logging
from pathlib import Path

from tradingagents.harness import db as _db

logger = logging.getLogger(__name__)

# 触发"系统性偏差"提醒的阈值
_MIN_SAMPLE_SIZE = 5            # 样本量门槛（V1 较低，因为初期数据少）
_LOW_HIT_RATE = 0.45            # 命中率显著偏低
_HIGH_HIT_RATE = 0.65           # 命中率显著偏高（信号好但样本少可能值得放宽触发）
_NEGATIVE_EXPECTATION = -1.0    # 期望收益负 → 该评级"赔钱"
_HIT_RATE_TIE_THRESHOLD = 0.05  # Conviction 高低命中率差异小于 5% → 信号无价值

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_OUTPUT_DIR = _PROJECT_ROOT / "harness_data" / "reviews"


def _query_metrics(snapshot_date: str, db_path=None) -> list[dict]:
    """拉某个 snapshot_date 的所有 metrics。"""
    with _db.connect(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM backtest_metrics WHERE snapshot_date = ? ORDER BY horizon, group_dimension, group_value",
            (snapshot_date,),
        ).fetchall()
    return [dict(r) for r in rows]


def _filter_significant(rows: list[dict]) -> list[dict]:
    """筛出"显著样本量"的 metrics（n ≥ _MIN_SAMPLE_SIZE）。"""
    return [r for r in rows if r["total_runs"] >= _MIN_SAMPLE_SIZE]


def _identify_low_hit_rate_issues(rows: list[dict]) -> list[dict]:
    """识别"命中率显著偏低"的切片。"""
    issues = []
    for r in rows:
        if r["direction_hit_rate"] < _LOW_HIT_RATE:
            issues.append({
                "type": "low_hit_rate",
                "severity": "high" if r["direction_hit_rate"] < 0.35 else "medium",
                "horizon": r["horizon"],
                "dimension": r["group_dimension"],
                "group_value": r["group_value"],
                "metric": r,
                "message": (
                    f"{r['group_dimension']}={r['group_value']} 在 {r['horizon']} "
                    f"命中率仅 {r['direction_hit_rate']:.0%}（{r['direction_hits']}/{r['total_runs']}）"
                ),
            })
    return issues


def _identify_negative_expectation(rows: list[dict]) -> list[dict]:
    """识别"期望收益负"的切片——这类评级实际上在赔钱。"""
    issues = []
    for r in rows:
        if r["expectation"] is not None and r["expectation"] < _NEGATIVE_EXPECTATION:
            issues.append({
                "type": "negative_expectation",
                "severity": "high" if r["expectation"] < -3 else "medium",
                "horizon": r["horizon"],
                "dimension": r["group_dimension"],
                "group_value": r["group_value"],
                "metric": r,
                "message": (
                    f"{r['group_dimension']}={r['group_value']} 在 {r['horizon']} "
                    f"期望收益 {r['expectation']:+.2f}%（赔钱）"
                ),
            })
    return issues


def _identify_conviction_signal_loss(rows: list[dict]) -> list[dict]:
    """识别 Conviction 高低命中率差异不显著的情况——说明 Conviction 信号无价值。"""
    issues = []
    # 按 horizon 分组
    by_horizon: dict = {}
    for r in rows:
        if r["group_dimension"] != "conviction":
            continue
        by_horizon.setdefault(r["horizon"], []).append(r)

    for horizon, conv_rows in by_horizon.items():
        if len(conv_rows) < 2:
            continue
        # 按 conviction stars 排序
        try:
            sorted_rows = sorted(conv_rows, key=lambda x: int(x["group_value"]))
        except (ValueError, TypeError):
            continue
        if len(sorted_rows) < 2:
            continue
        low_conv = sorted_rows[0]
        high_conv = sorted_rows[-1]
        diff = high_conv["direction_hit_rate"] - low_conv["direction_hit_rate"]
        if abs(diff) < _HIT_RATE_TIE_THRESHOLD:
            issues.append({
                "type": "conviction_signal_loss",
                "severity": "medium",
                "horizon": horizon,
                "dimension": "conviction",
                "group_value": "(整体)",
                "metric": None,
                "message": (
                    f"Conviction {low_conv['group_value']}★（{low_conv['direction_hit_rate']:.0%}）"
                    f"vs {high_conv['group_value']}★（{high_conv['direction_hit_rate']:.0%}）"
                    f"在 {horizon} 命中率差异仅 {diff:+.0%}——Conviction 信号区分能力弱"
                ),
            })

    return issues


def generate_insights(snapshot_date: str | None = None, db_path=None) -> dict:
    """生成洞察清单。"""
    snapshot_date = snapshot_date or _dt.date.today().isoformat()
    all_rows = _query_metrics(snapshot_date, db_path)
    significant = _filter_significant(all_rows)

    insights = {
        "snapshot_date": snapshot_date,
        "total_metric_rows": len(all_rows),
        "significant_rows": len(significant),
        "issues": [],
        "overall_by_horizon": {},
    }

    # 提取每个 horizon 的全局命中率（dimension=overall）
    for r in all_rows:
        if r["group_dimension"] == "overall":
            insights["overall_by_horizon"][r["horizon"]] = r

    # 识别 3 类问题
    insights["issues"].extend(_identify_low_hit_rate_issues(significant))
    insights["issues"].extend(_identify_negative_expectation(significant))
    insights["issues"].extend(_identify_conviction_signal_loss(significant))

    return insights


def render_markdown(insights: dict, db_path=None) -> str:
    """把洞察清单渲染成 markdown。"""
    lines: list[str] = []
    snapshot_date = insights["snapshot_date"]
    lines.append(f"# Backtest Snapshot {snapshot_date}")
    lines.append("")
    lines.append(f"- 总 metric 行数：{insights['total_metric_rows']}")
    lines.append(f"- 显著样本量（n ≥ {_MIN_SAMPLE_SIZE}）：{insights['significant_rows']}")
    lines.append(f"- 识别问题：{len(insights['issues'])} 条")
    lines.append("")
    lines.append("> **口径说明**：收益统计用 `signed_pnl_pct`（按预测方向取符号：long/HOLD=+涨跌幅，"
                 "short=−涨跌幅）——成功看空避开的下跌记为正收益。期望收益 = signed PnL 的样本均值。"
                 "命中带按 horizon 缩放（T±2% / T+1±3% / T+5±5% / T+30±10%）。")
    lines.append("")

    # 全局命中率
    lines.append("## 全局命中率（所有样本）")
    lines.append("")
    if not insights["overall_by_horizon"]:
        lines.append("（暂无已采集样本，请先跑真值采集）")
    else:
        lines.append("| Horizon | 样本数 | 命中率 | 判对均PnL | 判错均PnL | 期望PnL |")
        lines.append("|---------|--------|--------|------------|------------|---------|")
        for horizon in ("T", "T+1", "T+5", "T+30"):
            r = insights["overall_by_horizon"].get(horizon)
            if not r:
                lines.append(f"| {horizon} | (无) | (无) | (无) | (无) | (无) |")
                continue
            ac = f"{r['avg_return_correct']:+.2f}%" if r["avg_return_correct"] is not None else "—"
            aw = f"{r['avg_return_wrong']:+.2f}%" if r["avg_return_wrong"] is not None else "—"
            exp = f"{r['expectation']:+.2f}%" if r["expectation"] is not None else "—"
            lines.append(
                f"| {horizon} | {r['total_runs']} | {r['direction_hit_rate']:.0%} | {ac} | {aw} | {exp} |"
            )
    lines.append("")

    # 问题清单
    lines.append("## 系统性偏差（按严重度排序）")
    lines.append("")
    if not insights["issues"]:
        lines.append("✅ **未发现显著偏差**（样本量门槛 n ≥ {} 内未触发任何告警）".format(_MIN_SAMPLE_SIZE))
    else:
        issues_sorted = sorted(
            insights["issues"],
            key=lambda x: (
                0 if x["severity"] == "high" else (1 if x["severity"] == "medium" else 2),
                x["horizon"],
            ),
        )
        for i, issue in enumerate(issues_sorted, 1):
            sev_icon = "⚠️" if issue["severity"] == "high" else "🔶"
            lines.append(f"### {i}. {sev_icon} [{issue['type']}] {issue['message']}")
            if issue["metric"]:
                m = issue["metric"]
                lines.append("")
                lines.append(f"- 样本量: {m['total_runs']} | 命中: {m['direction_hits']} | 命中率: {m['direction_hit_rate']:.0%}")
                if m["avg_return_correct"] is not None:
                    lines.append(f"- 判对均PnL: {m['avg_return_correct']:+.2f}%")
                if m["avg_return_wrong"] is not None:
                    lines.append(f"- 判错均PnL: {m['avg_return_wrong']:+.2f}%")
                if m["expectation"] is not None:
                    lines.append(f"- 期望收益: {m['expectation']:+.2f}%")
            lines.append("")
    lines.append("")

    # 全切片表
    lines.append("## 全切片明细")
    lines.append("")
    by_horizon: dict = {}
    with _db.connect(db_path) as conn:
        rows = conn.execute(
            """SELECT * FROM backtest_metrics
               WHERE snapshot_date = ? AND group_dimension != 'overall'
               ORDER BY horizon, group_dimension, total_runs DESC""",
            (snapshot_date,),
        ).fetchall()
    for r in rows:
        by_horizon.setdefault(r["horizon"], []).append(r)

    for horizon in ("T", "T+1", "T+5", "T+30"):
        lines.append(f"### {horizon}")
        lines.append("")
        h_rows = by_horizon.get(horizon, [])
        if not h_rows:
            lines.append("（无数据）")
            lines.append("")
            continue
        lines.append("| 维度 | 取值 | n | 命中率 | 期望 |")
        lines.append("|------|------|---|--------|------|")
        for r in h_rows:
            exp = f"{r['expectation']:+.2f}%" if r["expectation"] is not None else "—"
            lines.append(
                f"| {r['group_dimension']} | {r['group_value']} | {r['total_runs']} "
                f"| {r['direction_hit_rate']:.0%} | {exp} |"
            )
        lines.append("")

    return "\n".join(lines)


def save_review_report(snapshot_date: str | None = None, db_path=None) -> Path:
    """生成 + 保存 markdown 报告到 harness_data/reviews/。"""
    snapshot_date = snapshot_date or _dt.date.today().isoformat()
    insights = generate_insights(snapshot_date, db_path)
    md = render_markdown(insights, db_path)

    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = _OUTPUT_DIR / f"backtest_{snapshot_date}.md"
    output_path.write_text(md, encoding="utf-8")
    logger.info("review 报告已保存：%s", output_path)
    return output_path


def main():
    """CLI 入口：python -m tradingagents.harness.review"""
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s %(levelname)s %(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    path = save_review_report()
    print(f"\nreview 报告生成于：{path}")


if __name__ == "__main__":
    main()
