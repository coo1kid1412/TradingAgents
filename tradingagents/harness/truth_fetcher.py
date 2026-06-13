"""真值采集器：拉 T/T+1/T+5/T+30 收盘价 + 期间 high/low，计算命中。

V2 改造：
- 用 price_cache 增量拉取（改造 A）
- 算 relative_return = 个股 - benchmark 同期回报（改造 B）
- not_due 自动重试（startup 时 promote target_date ≤ today 的 not_due → pending）
"""

from __future__ import annotations

import datetime as _dt
import logging
from pathlib import Path

import pandas as pd

from tradingagents.harness import db as _db
from tradingagents.harness import price_cache as _pcache

logger = logging.getLogger(__name__)

# 方向命中阈值（±2% 内为 neutral 命中；超出为 long/short 方向判定）
_HIT_THRESHOLD_PCT = 2.0

# 每个 horizon 对应的"交易日偏移量"
_HORIZON_OFFSETS = {"T": 0, "T+1": 1, "T+5": 5, "T+30": 30}

# 拉数据时多拉一段缓冲（覆盖 horizon=T+30 + 周末/节假日缓冲）
_FETCH_BUFFER_DAYS = 60

# Benchmark 列表（A 股 ETF，覆盖大盘/中盘/创业板/科创板/半导体/消费）
# 选用 ETF 而非指数，确保 get_stock_data 兼容
BENCHMARKS = [
    "510300",  # 沪深300 ETF
    "510500",  # 中证500 ETF
    "588000",  # 科创50 ETF
    "159915",  # 创业板 ETF
    "512760",  # 半导体 ETF
    "159928",  # 消费 ETF
]
# 默认 benchmark（个股没有更精细行业映射时用此对照大盘）
DEFAULT_BENCHMARK = "510300"


# 命中带按 horizon 缩放：±2% 对 T 合理，但对 T+5/T+30 太窄——高波动 regime 下
# 5 日内几乎没票停在 ±2%，HOLD 全被判"踩空"（2026-06 周报 HOLD 命中率虚低的根）。
# 波动随时间约按 √t 放大，这里取保守整数带。
_HIT_BAND_BY_HORIZON = {"T": 2.0, "T+1": 3.0, "T+5": 5.0, "T+30": 10.0}


def _hit_band(horizon: str | None) -> float:
    return _HIT_BAND_BY_HORIZON.get(horizon or "", _HIT_THRESHOLD_PCT)


def _direction_hit(direction_predicted: str | None, realized_return_pct: float,
                   horizon: str | None = None) -> int | None:
    """判定方向是否命中。阈值按 horizon 缩放（band(horizon)）。"""
    if direction_predicted is None:
        return None
    band = _hit_band(horizon)
    if direction_predicted == "long":
        return 1 if realized_return_pct > band else 0
    if direction_predicted == "short":
        return 1 if realized_return_pct < -band else 0
    if direction_predicted == "neutral":
        return 1 if abs(realized_return_pct) < band else 0
    return None


def _signed_pnl(direction_predicted: str | None, realized_return_pct: float) -> float | None:
    """按预测方向取符号的策略盈亏（修原报告"看空判对被记成巨亏"的记账 bug）。

    长/平 A 股账本语义：
    - long（BUY/OVERWEIGHT）：持仓 → 拿到个股涨跌幅
    - neutral（HOLD）：继续持有 → 同样拿到涨跌幅（HOLD 与 BUY 的差异在仓位不在方向）
    - short（SELL/UNDERWEIGHT）：离场/反向 → 拿到反向收益（成功看空=避开的下跌记为正）
    direction_hit（方向对不对）与本指标（按方向行动赚没赚）是两个独立问题，分开记。
    """
    if direction_predicted is None or realized_return_pct is None:
        return None
    if direction_predicted in ("long", "neutral"):
        return round(realized_return_pct, 4)
    if direction_predicted == "short":
        return round(-realized_return_pct, 4)
    return None


def update_benchmark_cache(db_path=None) -> dict:
    """在 truth_fetcher 启动时把所有 benchmark 更新到 cache。"""
    today = _dt.date.today()
    # 拉最近 120 天足够覆盖 T+30 horizon 的 anchor 前后
    start = (today - _dt.timedelta(days=120)).isoformat()
    end = today.isoformat()
    stats: dict = {}
    for bench in BENCHMARKS:
        try:
            df = _pcache.fetch_with_cache(bench, start, end, db_path)
            stats[bench] = len(df) if df is not None else 0
        except Exception as e:
            logger.warning("benchmark %s 更新失败: %s", bench, e)
            stats[bench] = 0
    return stats


def _compute_benchmark_return(
    benchmark_ticker: str,
    anchor_date: _dt.date,
    horizon_date: _dt.date,
    db_path=None,
) -> float | None:
    """从 cache 读 benchmark 在 anchor_date 和 horizon_date 的收盘价，算 horizon 期间收益。"""
    # cache 应该已经在 update_benchmark_cache 时填好了
    bench_df = _pcache.fetch_with_cache(
        benchmark_ticker,
        (anchor_date - _dt.timedelta(days=10)).isoformat(),
        (horizon_date + _dt.timedelta(days=5)).isoformat(),
        db_path,
    )
    if bench_df is None or len(bench_df) == 0:
        return None

    # 找 >= anchor_date 的第一行 close（基准锚定价）
    anchor_rows = bench_df[bench_df["Date"] >= anchor_date]
    if len(anchor_rows) == 0:
        return None
    anchor_close = float(anchor_rows.iloc[0]["Close"])

    # 找 horizon_date 那行（找 <= horizon_date 的最后一行）
    horizon_rows = bench_df[bench_df["Date"] <= horizon_date]
    if len(horizon_rows) == 0:
        return None
    # 在 anchor 之后 + horizon_date 之前的最后一行
    valid_rows = horizon_rows[horizon_rows["Date"] >= anchor_date]
    if len(valid_rows) == 0:
        return None
    horizon_close = float(valid_rows.iloc[-1]["Close"])

    if anchor_close <= 0:
        return None
    return round((horizon_close - anchor_close) / anchor_close * 100, 4)


def promote_due_outcomes(db_path=None) -> int:
    """把 fetch_status='not_due' 但 target_date ≤ today（或 NULL）的 outcomes
    重置为 'pending' 让下一轮重试。

    Returns: 提升的行数。
    """
    today_str = _dt.date.today().isoformat()
    with _db.connect(db_path) as conn:
        cur = conn.execute(
            """UPDATE outcomes
               SET fetch_status = 'pending', error_message = NULL
               WHERE fetch_status = 'not_due'
                 AND (target_date IS NULL OR target_date <= ?)""",
            (today_str,),
        )
        n = cur.rowcount
    if n > 0:
        logger.info("promote: %d 个 not_due outcomes → pending", n)
    return n


def fetch_one_run_outcomes(run_id: int, db_path=None) -> dict:
    """采集单 run 的全部 horizon 真值。"""
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

    base_date = _dt.date.fromisoformat(run["trade_date"])
    start_date = base_date - _dt.timedelta(days=10)
    end_date = base_date + _dt.timedelta(days=_FETCH_BUFFER_DAYS + 10)
    df = _pcache.fetch_with_cache(ticker, start_date, end_date, db_path)

    today = _dt.date.today()

    if df is None or len(df) == 0:
        is_future = base_date > today
        new_status = "not_due" if is_future else "failed"
        err_msg = None if is_future else "no price data"
        with _db.connect(db_path) as conn:
            for o in outs:
                conn.execute(
                    """UPDATE outcomes SET fetch_status = ?, error_message = ?,
                       fetched_at = CURRENT_TIMESTAMP WHERE run_id = ? AND horizon = ?""",
                    (new_status, err_msg, run_id, o["horizon"]),
                )
                stats[o["horizon"]] = new_status
        return stats

    # 找 anchor 行
    if run["report_window"] == "post_market":
        anchor_rows = df[df["Date"] > base_date]
    else:
        anchor_rows = df[df["Date"] >= base_date]

    df = anchor_rows.reset_index(drop=True)
    anchor = df["Date"].iloc[0] if len(df) > 0 else None

    if anchor is None:
        with _db.connect(db_path) as conn:
            for o in outs:
                conn.execute(
                    """UPDATE outcomes SET fetch_status = 'not_due', error_message = NULL,
                       fetched_at = CURRENT_TIMESTAMP WHERE run_id = ? AND horizon = ?""",
                    (run_id, o["horizon"]),
                )
                stats[o["horizon"]] = "not_due"
        return stats

    for o in outs:
        horizon = o["horizon"]
        offset = _HORIZON_OFFSETS.get(horizon)
        if offset is None:
            continue

        if offset >= len(df):
            with _db.connect(db_path) as conn:
                conn.execute(
                    """UPDATE outcomes SET fetch_status = 'not_due' WHERE run_id = ? AND horizon = ?""",
                    (run_id, horizon),
                )
            stats[horizon] = "not_due"
            continue

        target_row = df.iloc[offset]
        target_date = target_row["Date"]

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
        period = df.iloc[: offset + 1]
        high_during = float(period["High"].max()) if "High" in period.columns else None
        low_during = float(period["Low"].min()) if "Low" in period.columns else None

        realized_return = (actual_close - ref_price) / ref_price * 100.0
        dir_hit = _direction_hit(o["direction_predicted"], realized_return, horizon)
        signed_pnl = _signed_pnl(o["direction_predicted"], realized_return)

        tp1_hit = None
        sl_hard_hit = None
        if pred["pm_tp1"] is not None and high_during is not None:
            tp1_hit = 1 if high_during >= pred["pm_tp1"] else 0
        if pred["pm_sl_hard"] is not None and low_during is not None:
            sl_hard_hit = 1 if low_during <= pred["pm_sl_hard"] else 0

        # 算 benchmark / relative_return（改造 B）
        # V1 简单策略：全部用 DEFAULT_BENCHMARK（沪深300 ETF）作对照
        benchmark_ticker = DEFAULT_BENCHMARK
        benchmark_return = _compute_benchmark_return(benchmark_ticker, anchor, target_date, db_path)
        relative_return = None
        if benchmark_return is not None:
            relative_return = round(realized_return - benchmark_return, 4)

        with _db.connect(db_path) as conn:
            conn.execute(
                """UPDATE outcomes SET
                    target_date = ?, actual_close_at_horizon = ?,
                    actual_high_during_horizon = ?, actual_low_during_horizon = ?,
                    realized_return_pct = ?, signed_pnl_pct = ?, direction_hit = ?,
                    tp1_hit = ?, sl_hard_hit = ?,
                    benchmark_ticker = ?, benchmark_return_pct = ?, relative_return_pct = ?,
                    fetch_status = 'fetched', fetched_at = CURRENT_TIMESTAMP, error_message = NULL
                   WHERE run_id = ? AND horizon = ?""",
                (
                    target_date.isoformat(), actual_close, high_during, low_during,
                    round(realized_return, 4), signed_pnl, dir_hit, tp1_hit, sl_hard_hit,
                    benchmark_ticker, benchmark_return, relative_return,
                    run_id, horizon,
                ),
            )
        stats[horizon] = "fetched"
        rel_str = f" | rel={relative_return:+.2f}%" if relative_return is not None else ""
        logger.info(
            "run %d %s: %s ref=%.2f → close=%.2f (ret=%+.2f%%%s) dir_hit=%s",
            run_id, horizon, target_date, ref_price, actual_close, realized_return,
            rel_str, dir_hit,
        )

    return stats


def fetch_all_pending(db_path=None, update_benchmarks: bool = True) -> dict:
    """扫描所有 fetch_status='pending' 的 run，能算的就算。

    Args:
        update_benchmarks: 是否先更新 benchmark cache（V1 默认 True；如果短时多次跑可关闭）
    """
    summary = {
        "promoted_from_not_due": 0,
        "fetched": 0,
        "not_due": 0,
        "failed": 0,
        "skipped_no_ref_price": 0,
    }

    # Step 1: 先 promote not_due → pending（让旧的 not_due 有机会重试）
    summary["promoted_from_not_due"] = promote_due_outcomes(db_path)

    # Step 2: 先把 benchmark cache 更新（确保后续 relative_return 算得到）
    if update_benchmarks:
        bench_stats = update_benchmark_cache(db_path)
        logger.info("benchmark cache 更新: %s", bench_stats)

    # Step 3: 处理所有 pending
    with _db.connect(db_path) as conn:
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
    # cache 统计
    stats = _pcache.get_cache_stats()
    print(f"\nprice_cache 状态: tickers={stats.get('n_tickers', 0)}, "
          f"rows={stats.get('n_rows', 0)}, "
          f"日期跨度: {stats.get('d_min')} → {stats.get('d_max')}")


if __name__ == "__main__":
    main()
