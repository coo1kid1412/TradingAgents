"""本地价格缓存：避免每次 truth_fetcher 都重复调 tushare 拉同一段日期。

设计：
- price_cache 表：(ticker, trade_date) → OHLCV
- fetch_with_cache(ticker, start, end)：先查 cache，缺哪几天就增量拉，再统一返回完整 df

简化策略（适合 truth_fetcher 的用法）：
- 只增量补 cache_max + 1 到 effective_end 的尾部
- 不回填历史漏洞（用户场景里 truth_fetcher 总是往前看，不会回头查历史）
"""

from __future__ import annotations

import datetime as _dt
import io
import logging
from pathlib import Path

import pandas as pd

from tradingagents.harness import db as _db

logger = logging.getLogger(__name__)


def _fetch_from_vendor(ticker: str, start_date: str, end_date: str) -> pd.DataFrame | None:
    """直接调 route_to_vendor 拉 OHLCV，返回 DataFrame（Date 列为 date 类型，升序）。"""
    from tradingagents.dataflows.interface import route_to_vendor

    try:
        csv_str = route_to_vendor("get_stock_data", ticker, start_date, end_date)
    except Exception as e:
        logger.warning("vendor 拉 %s [%s, %s] 失败: %s", ticker, start_date, end_date, e)
        return None
    if not csv_str or "未找到" in csv_str[:200]:
        return None
    lines = [ln for ln in csv_str.splitlines() if not ln.startswith("#") and ln.strip()]
    if not lines:
        return None
    try:
        df = pd.read_csv(io.StringIO("\n".join(lines)))
    except Exception as e:
        logger.warning("CSV 解析失败 %s: %s", ticker, e)
        return None
    if "Date" not in df.columns or "Close" not in df.columns:
        return None
    df["Date"] = pd.to_datetime(df["Date"]).dt.date
    return df.sort_values("Date").reset_index(drop=True)


def _get_cache_range(ticker: str, db_path=None) -> tuple[_dt.date | None, _dt.date | None]:
    """返回 cache 里这只 ticker 的 (min_date, max_date)，全空则 (None, None)。"""
    with _db.connect(db_path) as conn:
        row = conn.execute(
            "SELECT MIN(trade_date) AS d_min, MAX(trade_date) AS d_max FROM price_cache WHERE ticker = ?",
            (ticker,),
        ).fetchone()
    if not row or row["d_min"] is None:
        return None, None
    return _dt.date.fromisoformat(str(row["d_min"])), _dt.date.fromisoformat(str(row["d_max"]))


def _write_to_cache(ticker: str, df: pd.DataFrame, db_path=None) -> int:
    """批量写入 cache。已存在的行用 REPLACE 覆盖（容忍 vendor 数据修正）。返回写入行数。"""
    if df is None or len(df) == 0:
        return 0
    rows = []
    for _, r in df.iterrows():
        rows.append((
            ticker,
            r["Date"].isoformat() if hasattr(r["Date"], "isoformat") else str(r["Date"]),
            float(r.get("Open")) if pd.notna(r.get("Open")) else None,
            float(r.get("High")) if pd.notna(r.get("High")) else None,
            float(r.get("Low")) if pd.notna(r.get("Low")) else None,
            float(r.get("Close")) if pd.notna(r.get("Close")) else None,
            float(r.get("Volume")) if pd.notna(r.get("Volume")) else None,
        ))
    with _db.connect(db_path) as conn:
        conn.executemany(
            """INSERT OR REPLACE INTO price_cache
               (ticker, trade_date, open, high, low, close, volume, fetched_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)""",
            rows,
        )
    return len(rows)


def _read_from_cache(ticker: str, start_date: _dt.date, end_date: _dt.date,
                     db_path=None) -> pd.DataFrame | None:
    """从 cache 读 [start_date, end_date] 之间的所有交易日数据，返回 DataFrame。"""
    with _db.connect(db_path) as conn:
        rows = conn.execute(
            """SELECT trade_date AS Date, open AS Open, high AS High,
                      low AS Low, close AS Close, volume AS Volume
               FROM price_cache
               WHERE ticker = ? AND trade_date BETWEEN ? AND ?
               ORDER BY trade_date""",
            (ticker, start_date.isoformat(), end_date.isoformat()),
        ).fetchall()
    if not rows:
        return None
    df = pd.DataFrame([dict(r) for r in rows])
    df["Date"] = pd.to_datetime(df["Date"]).dt.date
    return df


def fetch_with_cache(ticker: str, start_date: str | _dt.date, end_date: str | _dt.date,
                     db_path=None) -> pd.DataFrame | None:
    """主入口：返回 ticker 在 [start_date, end_date] 之间所有可得交易日的 OHLCV。

    逻辑：
    1. 查 cache 当前最大日期 cache_max
    2. effective_end = min(end_date, today)（不拉未来日期）
    3. 如果 cache_max 缺 effective_end 部分 → 增量拉 [cache_max+1, effective_end]
    4. 从 cache 读 [start_date, end_date] 完整范围返回
    """
    start = start_date if isinstance(start_date, _dt.date) else _dt.date.fromisoformat(start_date)
    end = end_date if isinstance(end_date, _dt.date) else _dt.date.fromisoformat(end_date)
    today = _dt.date.today()
    effective_end = min(end, today)

    if start > effective_end:
        # 完全在未来 → 没数据可拉
        return None

    cache_min, cache_max = _get_cache_range(ticker, db_path)

    # 决定要不要增量拉
    need_fetch_start: _dt.date | None = None
    if cache_max is None:
        # cache 空 → 拉 [start, effective_end]
        need_fetch_start = start
    elif cache_max < effective_end:
        # 增量补尾部：从 cache_max+1 拉
        need_fetch_start = cache_max + _dt.timedelta(days=1)

    if need_fetch_start is not None and need_fetch_start <= effective_end:
        new_df = _fetch_from_vendor(
            ticker, need_fetch_start.isoformat(), effective_end.isoformat()
        )
        if new_df is not None and len(new_df) > 0:
            n = _write_to_cache(ticker, new_df, db_path)
            logger.info("cache 增量更新 %s [%s, %s]: %d 行",
                        ticker, need_fetch_start, effective_end, n)
        else:
            logger.debug("cache 增量拉 %s [%s, %s]: vendor 无数据",
                         ticker, need_fetch_start, effective_end)

    # 从 cache 读完整范围（注意是 [start, end] 不是 effective_end）
    return _read_from_cache(ticker, start, end, db_path)


def get_cache_stats(db_path=None) -> dict:
    """统计 cache 当前状态：股票数 / 总行数 / 日期跨度。"""
    with _db.connect(db_path) as conn:
        row = conn.execute(
            """SELECT COUNT(DISTINCT ticker) AS n_tickers,
                      COUNT(*) AS n_rows,
                      MIN(trade_date) AS d_min,
                      MAX(trade_date) AS d_max
               FROM price_cache"""
        ).fetchone()
    return dict(row) if row else {}
