"""确定性市场风险快照。

该模块不调用 LLM：它把可追溯的市场因子压缩为交易前可执行的风险闸门，
供每日任务和单票 PM 共用。
"""

from __future__ import annotations

import datetime as _dt
import json
import re
from typing import Any, Iterable
from zoneinfo import ZoneInfo

from tradingagents.harness import db as _db


SNAPSHOT_VERSION = "v1"
_POSITION_CAP = {"低": 20, "中": 6, "高": 3, "极高": 0, "数据不足": 0}
_A_SHARE_CHECKPOINTS = ((9, 35, "09:35"), (11, 15, "11:15"), (14, 30, "14:30"))
_GATE_SEVERITY = {"OPEN": 0, "CONDITIONAL": 1, "WAIT": 2}
_WARNING_UNUSABLE = {"conflicted", "stale", "insufficient"}


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


def _warning_expected_time(now: _dt.datetime) -> _dt.datetime | None:
    """Latest A-share five-minute checkpoint expected at ``now``."""
    current = now.time()
    if current < _dt.time(8, 30):
        return None
    if current < _dt.time(9, 35):
        return now.replace(hour=8, minute=30, second=0, microsecond=0)
    if current <= _dt.time(11, 25):
        minute = (now.minute // 5) * 5
        return now.replace(minute=minute, second=0, microsecond=0)
    if current < _dt.time(13, 5):
        return now.replace(hour=11, minute=25, second=0, microsecond=0)
    if current <= _dt.time(14, 55):
        minute = (now.minute // 5) * 5
        return now.replace(minute=minute, second=0, microsecond=0)
    return now.replace(hour=14, minute=55, second=0, microsecond=0)


def _enforce_warning_freshness(
    warning: dict[str, Any] | None,
    trade_date: str,
    analysis_time: str | None,
) -> dict[str, Any] | None:
    if not warning or warning.get("market") != "a_share":
        return warning
    if warning.get("data_status") == "shadow":
        return warning
    now = (
        _dt.datetime.fromisoformat(analysis_time)
        if analysis_time
        else _dt.datetime.now(ZoneInfo("Asia/Shanghai"))
    )
    if now.tzinfo is None or now.utcoffset() is None:
        now = now.replace(tzinfo=ZoneInfo("Asia/Shanghai"))
    now = now.astimezone(ZoneInfo("Asia/Shanghai"))
    if str(trade_date) != now.date().isoformat():
        return warning
    try:
        captured = _dt.datetime.fromisoformat(str(warning.get("as_of_time")))
        if captured.tzinfo is None or captured.utcoffset() is None:
            raise ValueError("warning timestamp must be timezone-aware")
        captured = captured.astimezone(ZoneInfo("Asia/Shanghai"))
    except (TypeError, ValueError):
        captured = None
    expected = _warning_expected_time(now)
    stale_by_clock = (
        expected is not None
        and (
            captured is None
            or captured.date() != now.date()
            or captured < expected - _dt.timedelta(minutes=10)
        )
    )
    unusable = str(warning.get("data_status") or "") in _WARNING_UNUSABLE
    if not stale_by_clock and not unusable:
        return warning

    result = dict(warning)
    result["warning_level"] = "UNKNOWN"
    result["entry_gate"] = "WAIT"
    result["position_cap_pct"] = 0.0
    result["holding_action"] = "HOLD"
    result["data_status"] = "stale" if stale_by_clock else warning.get("data_status")
    reasons = list(warning.get("reasons") or [])
    if stale_by_clock:
        reasons.append(
            "市场预警快照陈旧：未达到当前应有的五分钟检查点；仅表示无法可靠判断"
        )
    else:
        reasons.append(
            f"市场预警数据状态为 {warning.get('data_status')}；仅表示无法可靠判断"
        )
    result["reasons"] = reasons
    result["required_checkpoint"] = expected.strftime("%H:%M") if expected else None
    return result


def load_market_warning_for_ticker(
    ticker: str,
    trade_date: str,
    db_path=None,
    analysis_time: str | None = None,
) -> dict[str, Any] | None:
    """Load the latest typed warning and its two calibrated probabilities."""
    market = infer_market(ticker)
    if market not in {"a_share", "us"}:
        return None
    market_zone = ZoneInfo("Asia/Shanghai" if market == "a_share" else "America/New_York")
    if analysis_time:
        cutoff = _dt.datetime.fromisoformat(analysis_time)
        if cutoff.tzinfo is None or cutoff.utcoffset() is None:
            cutoff = cutoff.replace(tzinfo=market_zone)
    else:
        cutoff = _dt.datetime.combine(
            _dt.date.fromisoformat(trade_date),
            _dt.time.max,
            tzinfo=market_zone,
        )
    cutoff_text = cutoff.astimezone(_dt.timezone.utc).isoformat()
    with _db.connect(db_path) as conn:
        rows = conn.execute(
            """SELECT d.baseline_level, d.final_level, d.transition, d.entry_gate,
                      d.new_position_cap_pct, d.holding_action, d.push_required,
                      d.data_status, d.reasons_json, d.model_version,
                      s.market, s.as_of_time, s.session_slot, s.reliability_grade,
                      p.horizon, p.probability, p.base_rate, p.market_phase,
                      p.calibration_version
               FROM market_warning_decisions AS d
               JOIN market_warning_feature_snapshots AS s ON s.id = d.feature_snapshot_id
               JOIN market_warning_predictions AS p ON p.feature_snapshot_id = s.id
               WHERE d.id = (
                   SELECT d2.id
                   FROM market_warning_decisions AS d2
                   JOIN market_warning_feature_snapshots AS s2 ON s2.id = d2.feature_snapshot_id
                   WHERE s2.market = ? AND s2.as_of_time <= ?
                     AND 4 = (
                         SELECT COUNT(*)
                         FROM market_warning_model_registry AS r
                         WHERE r.model_version = d2.model_version
                           AND r.feature_version = s2.feature_version
                           AND r.active = 1
                           AND r.market IN ('a_share', 'us')
                           AND r.horizon IN ('1d', '3d')
                     )
                   ORDER BY s2.as_of_time DESC, d2.id DESC LIMIT 1
               )
               ORDER BY p.horizon""",
            (market, cutoff_text),
        ).fetchall()
    if len(rows) != 2 or {row["horizon"] for row in rows} != {"1d", "3d"}:
        return None
    first = rows[0]
    probabilities = {row["horizon"]: row["probability"] for row in rows}
    base_rates = {row["horizon"]: row["base_rate"] for row in rows}
    warning = {
        "market": first["market"],
        "as_of_time": first["as_of_time"],
        "session_slot": first["session_slot"],
        "warning_level": first["final_level"],
        "baseline_level": first["baseline_level"],
        "entry_gate": first["entry_gate"],
        "position_cap_pct": first["new_position_cap_pct"],
        "holding_action": first["holding_action"],
        "transition": first["transition"],
        "push_required": bool(first["push_required"]),
        "data_status": first["data_status"],
        "reliability_grade": first["reliability_grade"],
        "phase": first["market_phase"],
        "probabilities": probabilities,
        "base_rates": base_rates,
        "model_version": first["model_version"],
        "calibration_version": first["calibration_version"],
        "reasons": list(json.loads(first["reasons_json"])),
    }
    return _enforce_warning_freshness(warning, trade_date, analysis_time)


def compose_effective_market_gate(
    legacy_snapshot: dict[str, Any] | None,
    warning_decision: dict[str, Any] | None,
    market: str,
) -> dict[str, Any] | None:
    """Compose legacy and warning gates without changing long-horizon fields."""
    if not legacy_snapshot and not warning_decision:
        return None
    legacy = dict(legacy_snapshot or {})
    warning = dict(warning_decision or {})
    result = dict(legacy)
    result["legacy_market_risk"] = legacy if legacy else None
    result["market_warning"] = warning if warning else None
    result["warning_level"] = warning.get("warning_level")
    result["warning_phase"] = warning.get("phase")
    result["warning_probabilities"] = warning.get("probabilities")

    legacy_gate = str(legacy.get("entry_gate") or "OPEN").upper()
    if legacy_gate not in _GATE_SEVERITY:
        legacy_gate = "WAIT"
    try:
        legacy_cap = float(legacy.get("position_cap_pct", 100.0))
    except (TypeError, ValueError):
        legacy_cap = 0.0
    result["effective_gate_source"] = "legacy_market_risk" if legacy else "none"

    production_warning = bool(warning) and not (
        market == "us" and warning.get("data_status") == "shadow"
    )
    if production_warning:
        warning_gate = str(warning.get("entry_gate") or "WAIT").upper()
        if warning_gate not in _GATE_SEVERITY:
            warning_gate = "WAIT"
        try:
            warning_cap = float(warning.get("position_cap_pct", 0.0))
        except (TypeError, ValueError):
            warning_cap = 0.0
        warning_stricter = (
            _GATE_SEVERITY[warning_gate] > _GATE_SEVERITY[legacy_gate]
            or (
                _GATE_SEVERITY[warning_gate] == _GATE_SEVERITY[legacy_gate]
                and warning_cap < legacy_cap
            )
        )
        if warning_stricter or not legacy:
            result["entry_gate"] = warning_gate
            result["position_cap_pct"] = min(legacy_cap, warning_cap) if legacy else warning_cap
            result["effective_gate_source"] = "market_warning"
            if warning.get("data_status") in _WARNING_UNUSABLE:
                result["data_status"] = warning.get("data_status")
                result["required_checkpoint"] = warning.get("required_checkpoint")
        else:
            result["entry_gate"] = legacy_gate
            result["position_cap_pct"] = legacy_cap
    elif legacy:
        result["entry_gate"] = legacy_gate
        result["position_cap_pct"] = legacy_cap

    reasons = list(legacy.get("reasons") or [])
    if warning:
        reasons.extend(f"market_warning: {reason}" for reason in warning.get("reasons") or [])
    result["reasons"] = reasons
    return result


def load_market_risk_for_ticker(
    ticker: str,
    trade_date: str,
    db_path=None,
    analysis_time: str | None = None,
) -> dict[str, Any] | None:
    """Compatibility boundary returning the strictest effective market gate."""
    market = infer_market(ticker)
    legacy = load_latest_market_risk_snapshot(market, trade_date, db_path)
    legacy = enforce_snapshot_freshness(legacy, trade_date, analysis_time=analysis_time)
    warning = load_market_warning_for_ticker(
        ticker,
        trade_date,
        db_path,
        analysis_time=analysis_time,
    )
    return compose_effective_market_gate(legacy, warning, market)
