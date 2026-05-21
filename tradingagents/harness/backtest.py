"""按 rating/style/theme_stage/window/conviction 维度切片统计命中率。

输入：outcomes 表已采集的真值
输出：写入 backtest_metrics 表（每次跑都生成新 snapshot_date 的一批记录）
"""

from __future__ import annotations

import datetime as _dt
import logging
import statistics as _stats
from collections import defaultdict
from pathlib import Path

from tradingagents.harness import db as _db

logger = logging.getLogger(__name__)

# 切片维度定义：(dimension_name, predictions 表里取哪个字段)
_SLICE_DIMENSIONS = [
    ("rating_rm", "rm_rating"),
    ("rating_pm", "pm_rating"),
    ("style", "style"),
    ("theme_stage", "theme_stage"),
    ("conviction", "pm_conviction_stars"),
]
# window 维度走 runs 表
_RUN_SLICE_DIMENSIONS = [
    ("window", "report_window"),
]

# 所有 horizon
_HORIZONS = ["T", "T+1", "T+5", "T+30"]


def _query_fetched_outcomes(horizon: str, db_path=None) -> list[dict]:
    """拉取一个 horizon 的所有已采集 outcome + 关联的 predictions + runs。"""
    with _db.connect(db_path) as conn:
        rows = conn.execute(
            """SELECT
                o.run_id, o.horizon, o.realized_return_pct, o.direction_hit,
                o.tp1_hit, o.sl_hard_hit,
                p.rm_rating, p.pm_rating, p.style, p.theme_stage,
                p.pm_conviction_stars, p.composite_score, p.momentum_score,
                r.report_window, r.ticker, r.trade_date
               FROM outcomes o
               JOIN predictions p ON p.run_id = o.run_id
               JOIN runs r ON r.id = o.run_id
               WHERE o.horizon = ? AND o.fetch_status = 'fetched'""",
            (horizon,),
        ).fetchall()
    return [dict(r) for r in rows]


def _compute_group_metric(group_rows: list[dict]) -> dict:
    """对一组 outcome rows 计算命中率 + 收益统计。"""
    n = len(group_rows)
    if n == 0:
        return {
            "total_runs": 0,
            "direction_hits": 0,
            "direction_hit_rate": 0.0,
            "avg_return_correct": None,
            "avg_return_wrong": None,
            "expectation": None,
        }

    valid_dir = [r for r in group_rows if r.get("direction_hit") is not None]
    valid_n = len(valid_dir)
    hits = sum(1 for r in valid_dir if r["direction_hit"] == 1)
    hit_rate = hits / valid_n if valid_n > 0 else 0.0

    correct_returns = [r["realized_return_pct"] for r in valid_dir if r["direction_hit"] == 1]
    wrong_returns = [r["realized_return_pct"] for r in valid_dir if r["direction_hit"] == 0]
    avg_correct = _stats.mean(correct_returns) if correct_returns else None
    avg_wrong = _stats.mean(wrong_returns) if wrong_returns else None

    if avg_correct is not None and avg_wrong is not None:
        expectation = hit_rate * avg_correct + (1 - hit_rate) * avg_wrong
    elif avg_correct is not None:
        expectation = avg_correct  # 全对的特殊情况
    elif avg_wrong is not None:
        expectation = avg_wrong
    else:
        expectation = None

    return {
        "total_runs": valid_n,
        "direction_hits": hits,
        "direction_hit_rate": round(hit_rate, 4),
        "avg_return_correct": round(avg_correct, 4) if avg_correct is not None else None,
        "avg_return_wrong": round(avg_wrong, 4) if avg_wrong is not None else None,
        "expectation": round(expectation, 4) if expectation is not None else None,
    }


def compute_snapshot(snapshot_date: str | None = None, db_path=None) -> dict:
    """生成本次 snapshot：清除当日数据 → 重算所有切片 → 写入 backtest_metrics。

    Args:
        snapshot_date: ISO 日期字符串；默认今天

    Returns:
        统计字典：{horizon: {dimension: n_groups}}
    """
    snapshot_date = snapshot_date or _dt.date.today().isoformat()
    summary: dict = {}

    # 先清除今天已有的 snapshot 数据（避免重复）
    with _db.connect(db_path) as conn:
        conn.execute("DELETE FROM backtest_metrics WHERE snapshot_date = ?", (snapshot_date,))

    for horizon in _HORIZONS:
        rows = _query_fetched_outcomes(horizon, db_path)
        if not rows:
            summary[horizon] = {"overall": 0}
            continue

        horizon_summary = {}

        # 切片 1: 全局
        metric = _compute_group_metric(rows)
        _write_metric(snapshot_date, horizon, "overall", "ALL", metric, db_path)
        horizon_summary["overall"] = 1

        # 切片 2-6: predictions 字段维度
        for dim_name, field_name in _SLICE_DIMENSIONS:
            groups: dict = defaultdict(list)
            for r in rows:
                v = r.get(field_name)
                if v is None or v == "":
                    continue
                groups[str(v)].append(r)
            for value, group_rows in groups.items():
                metric = _compute_group_metric(group_rows)
                _write_metric(snapshot_date, horizon, dim_name, value, metric, db_path)
            horizon_summary[dim_name] = len(groups)

        # 切片 7: runs 字段维度（window）
        for dim_name, field_name in _RUN_SLICE_DIMENSIONS:
            groups = defaultdict(list)
            for r in rows:
                v = r.get(field_name)
                if v is None:
                    continue
                groups[str(v)].append(r)
            for value, group_rows in groups.items():
                metric = _compute_group_metric(group_rows)
                _write_metric(snapshot_date, horizon, dim_name, value, metric, db_path)
            horizon_summary[dim_name] = len(groups)

        summary[horizon] = horizon_summary

    return summary


def _write_metric(
    snapshot_date: str,
    horizon: str,
    dimension: str,
    value: str,
    metric: dict,
    db_path=None,
):
    """写入 backtest_metrics 表。"""
    with _db.connect(db_path) as conn:
        conn.execute(
            """INSERT INTO backtest_metrics (
                snapshot_date, horizon, group_dimension, group_value,
                total_runs, direction_hits, direction_hit_rate,
                avg_return_correct, avg_return_wrong, expectation
               ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                snapshot_date, horizon, dimension, value,
                metric["total_runs"], metric["direction_hits"],
                metric["direction_hit_rate"], metric["avg_return_correct"],
                metric["avg_return_wrong"], metric["expectation"],
            ),
        )


def main():
    """CLI 入口：python -m tradingagents.harness.backtest"""
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s %(levelname)s %(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    print("开始生成回测命中率快照...")
    summary = compute_snapshot()
    print(f"\n生成完成统计：")
    for horizon, dims in summary.items():
        print(f"  {horizon}:")
        for dim, n in dims.items():
            print(f"    {dim}: {n} 组")


if __name__ == "__main__":
    main()
