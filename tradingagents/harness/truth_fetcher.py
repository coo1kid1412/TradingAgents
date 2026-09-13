"""真值采集器：拉 T/T+1/T+5/T+30 收盘价 + 期间 high/low，计算命中。

V2 改造：
- 用 price_cache 增量拉取（改造 A）
- 算 relative_return = 个股 - benchmark 同期回报（改造 B）
- 独立交易日历到期钟；缺行情与未到期分开，采集失败最多3次、间隔24小时
"""

from __future__ import annotations

import datetime as _dt
import logging
import math

import pandas as pd

from tradingagents.harness import db as _db
from tradingagents.harness import price_cache as _pcache
from tradingagents.harness import outcome_schedule as _schedule

logger = logging.getLogger(__name__)

# 方向命中阈值（±2% 内为 neutral 命中；超出为 long/short 方向判定）
_HIT_THRESHOLD_PCT = 2.0

# Per-outcome automatic collection budget; not a strategy hyperparameter.
_MAX_ATTEMPTS = 3
_RETRY_DELAY = _dt.timedelta(days=1)

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


def _now(now=None):
    now = now or _dt.datetime.now(_dt.timezone.utc)
    if now.tzinfo is None:
        raise ValueError("now must include a timezone")
    return now.astimezone(_dt.timezone.utc)


def prepare_outcomes(db_path=None, *, now=None, run_id=None) -> int:
    """Resolve targets without I/O to vendors; reopen only due, eligible work."""
    now = _now(now)
    promoted = 0
    with _db.connect(db_path) as conn:
        rows = conn.execute(
            "SELECT r.*,o.horizon,o.fetch_status,o.attempt_count,o.next_retry_at "
            "FROM runs r JOIN outcomes o ON r.id=o.run_id "
            "WHERE o.fetch_status IN ('pending','not_due','data_missing','failed') "
            "AND (? IS NULL OR r.id=?)", (run_id, run_id),
        ).fetchall()
        for row in rows:
            try:
                target = _schedule.target_for(row, row["horizon"])
            except (ValueError, KeyError, TypeError, IndexError) as exc:
                conn.execute(
                    "UPDATE outcomes SET fetch_status='schedule_error',error_message=?,"
                    "schedule_version=? WHERE run_id=? AND horizon=?",
                    (str(exc), _schedule.VERSION, row["id"], row["horizon"]),
                )
                continue
            status = row["fetch_status"]
            if target.due_at > now:
                status = "not_due"
            elif row["attempt_count"] >= _MAX_ATTEMPTS:
                status = "retry_exhausted"
            elif not row["next_retry_at"] or _dt.datetime.fromisoformat(row["next_retry_at"]) <= now:
                status = "pending"
            promoted += status == "pending" and row["fetch_status"] != "pending"
            conn.execute(
                "UPDATE outcomes SET target_date=?,due_at=?,schedule_version=?,fetch_status=? "
                "WHERE run_id=? AND horizon=?",
                (target.date.isoformat(), target.due_at.isoformat(), _schedule.VERSION,
                 status, row["id"], row["horizon"]),
            )
    return promoted


def promote_due_outcomes(db_path=None, *, now=None) -> int:
    return prepare_outcomes(db_path, now=now)


def _record_failure(outcome, status, message, db_path, now):
    attempts = outcome["attempt_count"] + 1
    if attempts >= _MAX_ATTEMPTS:
        status = "retry_exhausted"
    retry = (now + _RETRY_DELAY).isoformat() if attempts < _MAX_ATTEMPTS else None
    with _db.connect(db_path) as conn:
        conn.execute(
            "UPDATE outcomes SET fetch_status=?,error_message=?,attempt_count=?,next_retry_at=? "
            "WHERE run_id=? AND horizon=?",
            (status, message, attempts, retry, outcome["run_id"], outcome["horizon"]),
        )
    return status


def fetch_one_run_outcomes(run_id: int, db_path=None, *, now=None) -> dict:
    """采集单 run 的全部 horizon 真值。"""
    stats: dict = {}
    now = _now(now)
    prepare_outcomes(db_path, now=now, run_id=run_id)
    with _db.connect(db_path) as conn:
        run = conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
        pred = conn.execute("SELECT * FROM predictions WHERE run_id = ?", (run_id,)).fetchone()
        outs = conn.execute(
            "SELECT * FROM outcomes WHERE run_id = ? AND fetch_status = 'pending'",
            (run_id,),
        ).fetchall()
        if not run or not outs:
            return stats

    ticker = run["ticker"]
    ref_price = pred["current_price"] if pred else None
    if ref_price is None or not math.isfinite(ref_price) or ref_price <= 0:
        with _db.connect(db_path) as conn:
            conn.execute(
                "UPDATE outcomes SET fetch_status='invalid',error_message='invalid reference price' "
                "WHERE run_id=? AND fetch_status='pending'", (run_id,),
            )
        stats.update({o["horizon"]: "invalid" for o in outs})
        return stats

    targets = {o["horizon"]: _schedule.target_for(run, o["horizon"]) for o in outs}
    start_date = min(t.sessions[0] for t in targets.values())
    # Some routed vendors use an exclusive end date; scoring below still uses exact sessions.
    end_date = max(t.date for t in targets.values()) + _dt.timedelta(days=1)
    try:
        required = sorted({day for target in targets.values() for day in target.sessions})
        df = _pcache.fetch_with_cache(ticker, start_date, end_date, db_path, expected_sessions=required)
        if df is not None and not df.empty:
            df = df.copy()
            df["Date"] = pd.to_datetime(df["Date"]).dt.date
    except Exception as exc:
        logger.warning("run %d price fetch failed: %s", run_id, type(exc).__name__)
        return {o["horizon"]: _record_failure(o, "failed", "price fetch failed: " + type(exc).__name__, db_path, now) for o in outs}

    for o in outs:
        horizon = o["horizon"]
        target = targets[horizon]
        target_date, anchor = target.date, target.sessions[0]
        try:
            if df is None or df.empty:
                raise ValueError("no price data")
            period = df[df["Date"].isin(target.sessions)].sort_values("Date")
            if tuple(period["Date"]) != target.sessions:
                raise ValueError("missing or duplicate session bars")
            for column in ("Close", "High", "Low"):
                values = pd.to_numeric(period[column], errors="coerce")
                if not values.map(lambda v: math.isfinite(v) and v > 0).all():
                    raise ValueError("invalid OHLC values")
                period[column] = values
            if ((period["Low"] > period["Close"]) | (period["Close"] > period["High"])).any():
                raise ValueError("inconsistent OHLC values")
        except (ValueError, KeyError, TypeError) as exc:
            stats[horizon] = _record_failure(o, "data_missing", str(exc), db_path, now)
            continue

        target_row = period.iloc[-1]
        actual_close = float(target_row["Close"])
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
        # Unsupported market benchmarks remain unavailable, never substituted by A shares.
        benchmark_ticker = DEFAULT_BENCHMARK if _schedule.market_calendar(ticker) == "XSHG" else None
        benchmark_return = None
        if benchmark_ticker:
            try:
                benchmark_return = _compute_benchmark_return(benchmark_ticker, anchor, target_date, db_path)
            except Exception as exc:
                logger.warning("run %d benchmark unavailable: %s", run_id, type(exc).__name__)
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
                    fetch_status = 'fetched', fetched_at = CURRENT_TIMESTAMP, error_message = NULL,
                    attempt_count = attempt_count + 1, next_retry_at = NULL
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


def fetch_all_pending(db_path=None, update_benchmarks: bool = True, *, now=None) -> dict:
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
    now = _now(now)
    summary["promoted_from_not_due"] = promote_due_outcomes(db_path, now=now)

    # Step 2: 先把 benchmark cache 更新（确保后续 relative_return 算得到）
    with _db.connect(db_path) as conn:
        has_pending = conn.execute("SELECT 1 FROM outcomes WHERE fetch_status='pending' LIMIT 1").fetchone()
    if update_benchmarks and has_pending:
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
        stats = fetch_one_run_outcomes(run_id, db_path, now=now)
        for horizon, status in stats.items():
            summary[status] = summary.get(status, 0) + 1

    with _db.connect(db_path) as conn:
        summary["remaining"] = dict(conn.execute(
            "SELECT fetch_status,COUNT(*) FROM outcomes WHERE fetch_status!='fetched' GROUP BY fetch_status"
        ).fetchall())

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
