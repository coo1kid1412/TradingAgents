"""每日 21:00 跑一次的独立调度脚本。

逻辑：
1. 从 DB 取过去 7 天有 run 的 distinct tickers
2. 先把这些 tickers 的最新价格更新到 cache（增量）
3. 再更新 6 个 benchmark ETF 的最新价格到 cache
4. 提升所有 target_date ≤ today 的 not_due → pending
5. fetch_all_pending：把能算的 outcome 都算完

设计哲学：
- 跟 main.py 解耦：main.py 不做真值采集（耗时长 + 盘前盘中数据不准）
- 21:00 已盘后，数据稳定
- 周末跑也无害——cache 已是 5/22 最新数据，5/23/24 没新数据，0 API
- 不全量更新：只 distinct 出过去 7 天有跑过的标的，不扩散到陈年标的

调度建议：
- 手动：每晚 21:00 跑 `python -m tradingagents.harness.daily_update`
- 自动：crontab 加 `0 21 * * * cd <project> && .venv/bin/python -m tradingagents.harness.daily_update`
"""

from __future__ import annotations

import datetime as _dt
import logging

from tradingagents.harness import db as _db
from tradingagents.harness import price_cache as _pcache
from tradingagents.harness import truth_fetcher as _truth

logger = logging.getLogger(__name__)

# 取过去 N 天有 run 的标的（distinct），不扩散到陈年标的
_RECENT_DAYS = 7


def get_recent_tickers(days: int = _RECENT_DAYS, db_path=None) -> list[str]:
    """从 runs 表取过去 N 天 distinct tickers。"""
    cutoff = (_dt.date.today() - _dt.timedelta(days=days)).isoformat()
    with _db.connect(db_path) as conn:
        rows = conn.execute(
            "SELECT DISTINCT ticker FROM runs WHERE trade_date >= ? ORDER BY ticker",
            (cutoff,),
        ).fetchall()
    return [r["ticker"] for r in rows]


def update_recent_tickers_cache(tickers: list[str], db_path=None) -> dict:
    """对每只 ticker，确保 cache 拉到 today 的最新数据。"""
    today = _dt.date.today()
    # 拉最近 120 天足够覆盖任何 horizon
    start = today - _dt.timedelta(days=120)
    stats: dict = {}
    for ticker in tickers:
        try:
            df = _pcache.fetch_with_cache(ticker, start.isoformat(), today.isoformat(), db_path)
            stats[ticker] = len(df) if df is not None else 0
        except Exception as e:
            logger.warning("更新 %s cache 失败: %s", ticker, e)
            stats[ticker] = 0
    return stats


def run_daily_update(db_path=None) -> dict:
    """完整流程：从 DB 找近期标的 → 更新 cache → 拉 benchmark → 采真值。"""
    # Step 1: 找过去 7 天的 distinct tickers
    tickers = get_recent_tickers(_RECENT_DAYS, db_path)
    logger.info("近 %d 天有 %d 只 distinct 标的：%s",
                _RECENT_DAYS, len(tickers), ", ".join(tickers[:10]) + ("..." if len(tickers) > 10 else ""))

    # Step 2: 更新这些 tickers 的 cache
    ticker_stats = update_recent_tickers_cache(tickers, db_path)
    total_rows = sum(ticker_stats.values())
    logger.info("ticker cache 更新完成：%d 只标的，cache 内共 %d 行数据", len(tickers), total_rows)

    # Step 3: 更新 benchmark cache（fetch_all_pending 内部也会跑，这里显式做一次）
    bench_stats = _truth.update_benchmark_cache(db_path)
    logger.info("benchmark cache 更新：%s", bench_stats)

    # Step 4 + 5: promote not_due → fetch all pending
    # （fetch_all_pending 内部会调 promote_due_outcomes 和 update_benchmark_cache）
    fetch_summary = _truth.fetch_all_pending(db_path, update_benchmarks=False)
    logger.info("真值采集统计：%s", fetch_summary)

    return {
        "recent_tickers_count": len(tickers),
        "ticker_cache_stats": ticker_stats,
        "benchmark_cache_stats": bench_stats,
        "fetch_summary": fetch_summary,
        "price_cache_stats": _pcache.get_cache_stats(db_path),
    }


def main():
    """CLI 入口：每晚 21:00 跑一次。"""
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s %(levelname)s %(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    print(f"=== Harness Daily Update @ {_dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n")

    result = run_daily_update()

    print(f"\n=== 结果汇总 ===")
    print(f"近 {_RECENT_DAYS} 天 distinct tickers: {result['recent_tickers_count']} 只")
    print(f"price_cache: tickers={result['price_cache_stats'].get('n_tickers', 0)} "
          f"/ rows={result['price_cache_stats'].get('n_rows', 0)} "
          f"/ 跨度 {result['price_cache_stats'].get('d_min')} → {result['price_cache_stats'].get('d_max')}")
    print(f"\n真值采集统计:")
    for k, v in result["fetch_summary"].items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
