"""确定性市场风险快照。

该模块不调用 LLM：它把可追溯的市场因子压缩为交易前可执行的风险闸门，
供每日任务和单票 PM 共用。
"""

from __future__ import annotations

import datetime as _dt
import json
import re
from typing import Any, Iterable

from tradingagents.harness import db as _db


SNAPSHOT_VERSION = "v1"
_POSITION_CAP = {"低": 20, "中": 6, "高": 3, "极高": 0, "数据不足": 0}
_A_SHARE_CHECKPOINTS = ((9, 35, "09:35"), (11, 15, "11:15"), (14, 30, "14:30"))


def infer_market(ticker: str) -> str:
    """把项目内的标准 ticker 映射为本期支持的市场。"""
    symbol = (ticker or "").strip().upper()
    if re.fullmatch(r"\d{6}", symbol):
        return "a_share"
    if symbol.endswith(".HK"):
        return "hk"
    return "us"


def _close_series(rows: Iterable[dict[str, Any]]) -> list[float]:
    closes: list[float] = []
    for row in rows:
        try:
            value = float(row["close"] if "close" in row else row["Close"])
        except (KeyError, TypeError, ValueError):
            continue
        if value > 0:
            closes.append(value)
    return closes


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def compute_market_risk_snapshot(
    market: str,
    price_rows: Iterable[dict[str, Any]],
    breadth_pct: float | None = None,
    volatility_pct: float | None = None,
    as_of_date: str | None = None,
    as_of_time: str | None = None,
) -> dict[str, Any]:
    """计算市场快照。

    规则保持故意简单、透明：趋势、波动、市场宽度各投 0-2 个风险点。
    首期不把缺失因子伪装成中性；只有价格不足时才标记数据不足。
    """
    closes = _close_series(price_rows)
    now = _dt.datetime.now(_dt.timezone.utc).astimezone()
    as_of_date = as_of_date or now.date().isoformat()
    as_of_time = as_of_time or now.isoformat(timespec="seconds")

    if len(closes) < 20:
        return {
            "market": market,
            "as_of_date": as_of_date,
            "as_of_time": as_of_time,
            "snapshot_version": SNAPSHOT_VERSION,
            "risk_level": "数据不足",
            "risk_score": None,
            "t_plus_1_bias": "不确定",
            "entry_gate": "WAIT",
            "position_cap_pct": _POSITION_CAP["数据不足"],
            "data_status": "missing",
            "factor_breakdown": {"price_history_days": len(closes)},
            "reasons": ["有效价格历史不足 20 个交易日"],
        }

    latest = closes[-1]
    ma20 = _mean(closes[-20:])
    ma50 = _mean(closes[-50:]) if len(closes) >= 50 else None
    trend_points = 0
    if latest < ma20:
        trend_points += 1
    if ma50 is not None and ma20 < ma50:
        trend_points += 1

    vol_points = 0
    if volatility_pct is not None:
        vol_points = 2 if volatility_pct >= 28 else (1 if volatility_pct >= 20 else 0)

    breadth_points = 0
    if breadth_pct is not None:
        breadth_points = 2 if breadth_pct <= 35 else (1 if breadth_pct <= 50 else 0)

    score = trend_points + vol_points + breadth_points
    risk_level = "低" if score <= 1 else "中" if score <= 3 else "高" if score <= 5 else "极高"
    bias = "偏多" if trend_points == 0 and score <= 1 else "偏空" if trend_points >= 1 or score >= 4 else "震荡"
    gate = "OPEN" if risk_level == "低" else "CONDITIONAL" if risk_level == "中" else "WAIT"
    data_status = "fresh" if breadth_pct is not None and volatility_pct is not None else "partial"
    factors = {
        "latest_close": round(latest, 4), "ma20": round(ma20, 4),
        "ma50": round(ma50, 4) if ma50 is not None else None,
        "trend_points": trend_points, "volatility_pct": volatility_pct,
        "volatility_points": vol_points, "breadth_pct": breadth_pct,
        "breadth_points": breadth_points, "price_history_days": len(closes),
    }
    reasons = [f"趋势风险 {trend_points}/2", f"波动风险 {vol_points}/2", f"市场宽度风险 {breadth_points}/2"]
    if data_status == "partial":
        reasons.append("部分因子缺失，风险等级按可得数据计算")
    return {
        "market": market, "as_of_date": as_of_date, "as_of_time": as_of_time,
        "snapshot_version": SNAPSHOT_VERSION, "risk_level": risk_level,
        "risk_score": score, "t_plus_1_bias": bias, "entry_gate": gate,
        "position_cap_pct": _POSITION_CAP[risk_level], "data_status": data_status,
        "factor_breakdown": factors, "reasons": reasons,
    }


def save_market_risk_snapshot(snapshot: dict[str, Any], db_path=None) -> None:
    """按市场和生效日幂等保存快照；同日重跑覆盖旧值。"""
    with _db.connect(db_path) as conn:
        conn.execute(
            """INSERT INTO market_risk_snapshots (
                market, as_of_date, as_of_time, snapshot_version, risk_level, risk_score,
                t_plus_1_bias, entry_gate, position_cap_pct, data_status,
                factor_breakdown_json, reasons_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(market, as_of_date) DO UPDATE SET
                as_of_time=excluded.as_of_time, snapshot_version=excluded.snapshot_version,
                risk_level=excluded.risk_level, risk_score=excluded.risk_score,
                t_plus_1_bias=excluded.t_plus_1_bias, entry_gate=excluded.entry_gate,
                position_cap_pct=excluded.position_cap_pct, data_status=excluded.data_status,
                factor_breakdown_json=excluded.factor_breakdown_json, reasons_json=excluded.reasons_json""",
            (
                snapshot["market"], snapshot["as_of_date"], snapshot["as_of_time"],
                snapshot.get("snapshot_version", SNAPSHOT_VERSION), snapshot["risk_level"],
                snapshot.get("risk_score"), snapshot["t_plus_1_bias"], snapshot["entry_gate"],
                snapshot["position_cap_pct"], snapshot["data_status"],
                json.dumps(snapshot.get("factor_breakdown", {}), ensure_ascii=False),
                json.dumps(snapshot.get("reasons", []), ensure_ascii=False),
            ),
        )


def load_latest_market_risk_snapshot(market: str, as_of_date: str | None = None, db_path=None) -> dict[str, Any] | None:
    """读取给定日期或此前最近一个有效快照。"""
    cutoff = as_of_date or _dt.date.today().isoformat()
    with _db.connect(db_path) as conn:
        row = conn.execute(
            """SELECT * FROM market_risk_snapshots
               WHERE market = ? AND as_of_date <= ?
               ORDER BY as_of_date DESC LIMIT 1""", (market, cutoff),
        ).fetchone()
    if row is None:
        return None
    out = dict(row)
    out["factor_breakdown"] = json.loads(out.pop("factor_breakdown_json"))
    out["reasons"] = json.loads(out.pop("reasons_json"))
    return out


def enforce_snapshot_freshness(
    snapshot: dict[str, Any] | None,
    trade_date: str,
    analysis_time: str | None = None,
) -> dict[str, Any] | None:
    """Fail the A-share entry gate closed when an expected intraday checkpoint is missing."""
    if not snapshot or snapshot.get("market") != "a_share":
        return snapshot
    now = _dt.datetime.fromisoformat(analysis_time) if analysis_time else _dt.datetime.now().astimezone()
    if str(trade_date) != now.date().isoformat():
        return snapshot

    required = None
    for hour, minute, label in _A_SHARE_CHECKPOINTS:
        if (now.hour, now.minute) >= (hour, minute):
            required = (hour, minute, label)
    if required is None:
        return snapshot

    try:
        captured = _dt.datetime.fromisoformat(str(snapshot.get("as_of_time")))
    except (TypeError, ValueError):
        captured = None
    hour, minute, label = required
    is_current = (
        captured is not None
        and captured.date() == now.date()
        and (captured.hour, captured.minute) >= (hour, minute)
    )
    if is_current:
        return snapshot

    stale = dict(snapshot)
    stale["data_status"] = "stale"
    stale["entry_gate"] = "WAIT"
    stale["position_cap_pct"] = 0
    stale["required_checkpoint"] = label
    stale["reasons"] = list(snapshot.get("reasons") or []) + [
        f"盘中风险快照陈旧：当前应至少使用 {label} 检查点，短期动作强制 WAIT"
    ]
    return stale


def load_market_risk_for_ticker(
    ticker: str,
    trade_date: str,
    db_path=None,
    analysis_time: str | None = None,
) -> dict[str, Any] | None:
    """供个股图读取：按 ticker 市场取分析日及之前最近有效快照。"""
    snapshot = load_latest_market_risk_snapshot(infer_market(ticker), trade_date, db_path)
    return enforce_snapshot_freshness(snapshot, trade_date, analysis_time=analysis_time)
