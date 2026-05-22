"""真值采集器：拉 T/T+1/T+5/T+30 收盘价 + 期间 high/low，计算命中。

V1 防穿越：只用报告时点之后的价格数据。
对 post_market 报告，T = 下一交易日；对 pre/morning/afternoon，T = 当天交易日。

数据源：复用 dataflows.interface.route_to_vendor("get_stock_data", ...)。
"""

from __future__ import annotations

import datetime as _dt
import io
import logging
import re
from pathlib import Path

import pandas as pd

from tradingagents.harness import db as _db

logger = logging.getLogger(__name__)

# 方向命中阈值（±2% 内为 neutral 命中；超出为 long/short 方向判定）
_HIT_THRESHOLD_PCT = 2.0

# 每个 horizon 对应的"交易日偏移量"
_HORIZON_OFFSETS = {"T": 0, "T+1": 1, "T+5": 5, "T+30": 30}

# 拉数据时多拉一段缓冲（30 天 + 周末/节假日缓冲）
_FETCH_BUFFER_DAYS = 50


def _fetch_price_df(ticker: str, start_date: str, end_date: str) -> pd.DataFrame | None:
    """从 route_to_vendor 拉 OHLCV CSV → DataFrame。"""
    from tradingagents.dataflows.interface import route_to_vendor

    try:
        csv_str = route_to_vendor("get_stock_data", ticker, start_date, end_date)
    except Exception as e:
        logger.warning("拉价格失败 %s: %s", ticker, e)
        return None
    if not csv_str or "未找到" in csv_str[:200]:
        return None

    # 跳过以 # 开头的 header 行
    lines = [ln for ln in csv_str.splitlines() if not ln.startswith("#") and ln.strip()]
    if not lines:
        return None
    try:
        df = pd.read_csv(io.StringIO("\n".join(lines)))
    except Exception as e:
        logger.warning("解析 CSV 失败 %s: %s", ticker, e)
        return None
    if "Date" not in df.columns or "Close" not in df.columns:
        return None
    # 标准化 Date 字段
    df["Date"] = pd.to_datetime(df["Date"]).dt.date
    df = df.sort_values("Date").reset_index(drop=True)
    return df


def _compute_anchor_date(trade_date: str, report_window: str) -> _dt.date:
    """计算评估起点日（T 对应的日期）。

    pre_market / morning / afternoon：T 是当天（trade_date）
    post_market：T 是 trade_date 的下一日（实际下一交易日由价格数据决定）
    """
    base = _dt.date.fromisoformat(trade_date)
    if report_window == "post_market":
        return base + _dt.timedelta(days=1)
    return base


def _direction_hit(direction_predicted: str | None, realized_return_pct: float) -> int | None:
    """判定方向是否命中。阈值 ±_HIT_THRESHOLD_PCT。"""
    if direction_predicted is None:
        return None
    if direction_predicted == "long":
        return 1 if realized_return_pct > _HIT_THRESHOLD_PCT else 0
    if direction_predicted == "short":
        return 1 if realized_return_pct < -_HIT_THRESHOLD_PCT else 0
    if direction_predicted == "neutral":
        return 1 if abs(realized_return_pct) < _HIT_THRESHOLD_PCT else 0
    return None


def fetch_one_run_outcomes(run_id: int, db_path=None) -> dict:
    """采集单 run 的全部 horizon 真值。

    Returns:
        统计字典：{horizon: status} 例 {'T': 'fetched', 'T+1': 'fetched', 'T+5': 'not_due', 'T+30': 'not_due'}
    """
    stats: dict = {}
    with _db.connect(db_path) as conn:
        run = conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
        pred = conn.execute("SELECT * FROM predictions WHERE run_id = ?", (run_id,)).fetchone()
        outs = conn.execute(
            "SELECT * FROM outcomes WHERE run_id = ? AND fetch_status = 'pending'",
            (run_id,),
        ).fetchall()
        if not run or not pred or not outs:
            return stats

    ticker = run["ticker"]
    ref_price = pred["current_price"]
    if ref_price is None or ref_price <= 0:
        logger.warning("run %d 缺 reference_price，跳过", run_id)
        return stats

    # 拉宽日期范围（trade_date - 5 天 到 trade_date + 60 天），让数据天然包含交易日序列
    base_date = _dt.date.fromisoformat(run["trade_date"])
    start_str = (base_date - _dt.timedelta(days=10)).isoformat()
    end_str = (base_date + _dt.timedelta(days=_FETCH_BUFFER_DAYS + 10)).isoformat()
    df = _fetch_price_df(ticker, start_str, end_str)

    today = _dt.date.today()

    if df is None or len(df) == 0:
        # 拉不到任何数据：区分"未来日期未到"（not_due）vs"数据源故障"（failed）
        is_future = base_date > today
        new_status = "not_due" if is_future else "failed"
        err_msg = None if is_future else "no price data"
        with _db.connect(db_path) as conn:
            for o in outs:
                conn.execute(
                    """UPDATE outcomes
                       SET fetch_status = ?, error_message = ?,
                           fetched_at = CURRENT_TIMESTAMP
                       WHERE run_id = ? AND horizon = ?""",
                    (new_status, err_msg, run_id, o["horizon"]),
                )
                stats[o["horizon"]] = new_status
        return stats

    # 根据 report_window 找 anchor 行（T 对应的实际交易日）
    # pre/morning/afternoon：T = trade_date 当天那行（如果当天有数据）
    # post_market：T = trade_date 之后第一个交易日
    if run["report_window"] == "post_market":
        anchor_rows = df[df["Date"] > base_date]
    else:
        anchor_rows = df[df["Date"] >= base_date]

    df = anchor_rows.reset_index(drop=True)
    # 计算实际 anchor 日期（df 第一行的 Date），后续 horizon 偏移基于此
    anchor = df["Date"].iloc[0] if len(df) > 0 else None

    # 如果没找到 anchor 行（base_date 之后还没有交易日），说明 anchor 还在未来
    if anchor is None:
        with _db.connect(db_path) as conn:
            for o in outs:
                conn.execute(
                    """UPDATE outcomes
                       SET fetch_status = 'not_due', error_message = NULL,
                           fetched_at = CURRENT_TIMESTAMP
                       WHERE run_id = ? AND horizon = ?""",
                    (run_id, o["horizon"]),
                )
                stats[o["horizon"]] = "not_due"
        return stats

    today = _dt.date.today()

    for o in outs:
        horizon = o["horizon"]
        offset = _HORIZON_OFFSETS.get(horizon)
        if offset is None:
            continue

        # 检查数据是否足够
        if offset >= len(df):
            # 数据还没到日子（horizon 还没到）
            with _db.connect(db_path) as conn:
                conn.execute(
                    """UPDATE outcomes SET fetch_status = 'not_due' WHERE run_id = ? AND horizon = ?""",
                    (run_id, horizon),
                )
            stats[horizon] = "not_due"
            continue

        target_row = df.iloc[offset]
        target_date = target_row["Date"]

        # 如果 target_date 还在未来 → not_due
        if target_date > today:
            with _db.connect(db_path) as conn:
                conn.execute(
                    """UPDATE outcomes SET fetch_status = 'not_due', target_date = ?
                       WHERE run_id = ? AND horizon = ?""",
                    (target_date.isoformat(), run_id, horizon),
                )
            stats[horizon] = "not_due"
            continue

        actual_close = float(target_row["Close"])
        # 期间（anchor 到 target）的 high/low
        period = df.iloc[: offset + 1]
        high_during = float(period["High"].max()) if "High" in period.columns else None
        low_during = float(period["Low"].min()) if "Low" in period.columns else None

        realized_return = (actual_close - ref_price) / ref_price * 100.0
        dir_hit = _direction_hit(o["direction_predicted"], realized_return)

        # TP1 / SL_hard 触达
        tp1_hit = None
        sl_hard_hit = None
        if pred["pm_tp1"] is not None and high_during is not None:
            tp1_hit = 1 if high_during >= pred["pm_tp1"] else 0
        if pred["pm_sl_hard"] is not None and low_during is not None:
            sl_hard_hit = 1 if low_during <= pred["pm_sl_hard"] else 0

        with _db.connect(db_path) as conn:
            conn.execute(
                """UPDATE outcomes SET
                    target_date = ?,
                    actual_close_at_horizon = ?,
                    actual_high_during_horizon = ?,
                    actual_low_during_horizon = ?,
                    realized_return_pct = ?,
                    direction_hit = ?,
                    tp1_hit = ?,
                    sl_hard_hit = ?,
                    fetch_status = 'fetched',
                    fetched_at = CURRENT_TIMESTAMP,
                    error_message = NULL
                   WHERE run_id = ? AND horizon = ?""",
                (
                    target_date.isoformat(),
                    actual_close,
                    high_during,
                    low_during,
                    round(realized_return, 4),
                    dir_hit,
                    tp1_hit,
                    sl_hard_hit,
                    run_id,
                    horizon,
                ),
            )
        stats[horizon] = "fetched"
        logger.info(
            "run %d %s: %s ref=%.2f → close=%.2f (return=%+.2f%%) dir_hit=%s",
            run_id, horizon, target_date, ref_price, actual_close, realized_return, dir_hit,
        )

    return stats


def fetch_all_pending(db_path=None) -> dict:
    """扫描所有 fetch_status='pending' 的 run，能算的就算。"""
    summary = {"fetched": 0, "not_due": 0, "failed": 0, "skipped_no_ref_price": 0}

    with _db.connect(db_path) as conn:
        # 找出所有还有 pending outcome 的 run
        rows = conn.execute(
            """SELECT DISTINCT run_id FROM outcomes WHERE fetch_status = 'pending'
               ORDER BY run_id"""
        ).fetchall()
        run_ids = [r["run_id"] for r in rows]

    logger.info("找到 %d 个 run 含 pending outcomes", len(run_ids))

    for run_id in run_ids:
        stats = fetch_one_run_outcomes(run_id, db_path)
        for horizon, status in stats.items():
            summary[status] = summary.get(status, 0) + 1

    return summary


def main():
    """CLI 入口：python -m tradingagents.harness.truth_fetcher"""
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s %(levelname)s %(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    print("开始扫描 pending 真值任务...")
    summary = fetch_all_pending()
    print(f"\n真值采集完成统计:")
    for k, v in summary.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
