"""Deterministic technical evidence with bounded display and full local audit."""

import hashlib
import io
import json
import math
from pathlib import Path
import uuid

import pandas as pd
from stockstats import wrap

from .intraday_quote import parse_price_metadata


INDICATORS = {
    "close_50_sma": ("close_50_sma", 50),
    "close_200_sma": ("close_200_sma", 200),
    "close_10_ema": ("close_10_ema", 10),
    "rsi": ("rsi_14", 14),
    "macd": ("macd", 26), "macds": ("macds", 35), "macdh": ("macdh", 35),
    "boll": ("boll", 20), "boll_ub": ("boll_ub", 20), "boll_lb": ("boll_lb", 20),
    "atr": ("atr", 14), "vwma": ("vwma", 14),
}
_COLUMNS = ["Date", "Open", "High", "Low", "Close", "Volume"]


def _number(value):
    return round(float(value), 6) if value is not None and math.isfinite(float(value)) else None


def _records(frame):
    return json.loads(frame.to_json(orient="records", double_precision=15))


def _read_prices(raw: str, as_of: str):
    try:
        frame = pd.read_csv(io.StringIO(raw), comment="#")
        if not set(_COLUMNS).issubset(frame):
            raise ValueError("OHLCV columns missing")
        frame = frame[_COLUMNS].copy()
        dates = pd.to_datetime(frame["Date"], errors="raise").dt.normalize()
        frame["Date"] = dates.dt.strftime("%Y-%m-%d")
        frame = frame.loc[dates <= pd.Timestamp(as_of)].sort_values("Date").reset_index(drop=True)
        if frame.empty or frame["Date"].duplicated().any():
            raise ValueError("OHLCV empty or duplicate dates")
        for column in _COLUMNS[1:]:
            frame[column] = pd.to_numeric(frame[column], errors="raise")
            if not frame[column].map(math.isfinite).all():
                raise ValueError("OHLCV contains non-finite values")
        if (frame[["Open", "High", "Low", "Close"]] <= 0).any().any() or (frame.Volume < 0).any():
            raise ValueError("OHLCV contains invalid price or volume")
        if (frame.High < frame[["Open", "Close", "Low"]].max(axis=1)).any() or (
            frame.Low > frame[["Open", "Close", "High"]].min(axis=1)
        ).any():
            raise ValueError("OHLCV high/low inconsistent")
        return frame
    except (ValueError, KeyError, TypeError, pd.errors.ParserError) as error:
        raise ValueError(f"Invalid technical OHLCV: {error}") from error


def build_technical_context(raw: str, *, symbol: str, as_of: str, audit_dir: Path) -> dict:
    """Calculate with full history; only the display windows are shortened."""
    cutoff = pd.Timestamp(as_of)
    frame = _read_prices(raw, as_of)
    price = parse_price_metadata(raw)
    if price.get("date") and pd.Timestamp(price["date"]) > cutoff:
        raise ValueError("Price metadata is after analysis date")
    actual_date = str(frame.Date.iloc[-1])
    if price.get("date") and price["date"] != actual_date:
        raise ValueError("Price metadata does not match last available bar")
    price.update(date=actual_date, current_price=float(frame.Close.iloc[-1]))
    if actual_date < as_of and price["status"] == "official_daily":
        price["status"] = "t_minus_1"

    stats = wrap(frame.copy())
    in_year = pd.to_datetime(frame.Date) >= cutoff - pd.Timedelta(days=365)
    indicators, full_series = {}, {}
    for name, (column, minimum) in INDICATORS.items():
        series = stats[column].copy()
        series.iloc[:minimum - 1] = math.nan
        series = series.where(series.map(math.isfinite))
        full_series[name] = [None if not math.isfinite(float(value)) else float(value) for value in series]
        year_values = series.loc[in_year].dropna()
        current = _number(series.iloc[-1])
        rank = None
        if current is not None and len(year_values) >= 200:
            rank = float((year_values <= float(series.iloc[-1])).mean() * 100)
        indicators[name] = {
            "value": current, "date": actual_date,
            "percentile_1y": _number(rank), "samples_1y": len(year_values),
            "min_1y": _number(year_values.min()) if len(year_values) else None,
            "max_1y": _number(year_values.max()) if len(year_values) else None,
            "change_5d": _number(series.iloc[-1] - series.iloc[-6]) if len(series) >= 6 else None,
        }

    weekly = frame.copy().set_index(pd.to_datetime(frame.Date)).resample("W-FRI").agg({
        "Date": "last", "Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum",
    }).dropna(subset=["Close"])
    week_end = pd.Timestamp(actual_date).to_period("W-FRI").end_time.normalize()
    partial = week_end > cutoff or (
        week_end == cutoff and (actual_date < as_of or price["status"] == "intraday_provisional")
    )
    completed = weekly.iloc[:-1] if partial else weekly
    year = frame.loc[in_year]
    warnings = []
    if len(year) < 200:
        warnings.append("不足200个一年窗口交易样本，历史分位不可用")
    if len(frame) < 200:
        warnings.append("历史不足200个交易日，长期均线不可用")
    if price["status"] in {"t_minus_1", "unknown", "intraday_provisional"}:
        warnings.append(f"价格状态={price['status']}，不得表述为正式当日收盘")
    last = frame.iloc[-1]
    volume_ratio = None
    if len(frame) >= 20 and frame.Volume.tail(20).mean() > 0:
        volume_ratio = frame.Volume.tail(5).mean() / frame.Volume.tail(20).mean()
    returns = {
        str(days): _number((last.Close / frame.Close.iloc[-days - 1] - 1) * 100)
        if len(frame) > days else None for days in (1, 5, 20, 60, 252)
    }
    packet = {
        "schema": "compact-technical-v1", "symbol": symbol, "analysis_date": as_of,
        "price": price,
        "coverage": {
            "start": str(frame.Date.iloc[0]), "end": actual_date, "bars": len(frame),
            "samples_1y": len(year), "warnings": warnings,
            "adjustment": "as_provided_by_vendor", "percentile_method": "count(value <= latest) / valid_samples * 100; minimum 200",
        },
        "price_statistics": {
            "returns_pct": returns,
            "high_1y": _number(year.High.max()), "low_1y": _number(year.Low.min()),
            "volume_ratio_5d_20d": _number(volume_ratio),
            "prior_20d_high": _number(frame.High.iloc[-21:-1].max()) if len(frame) >= 21 else None,
            "prior_20d_low": _number(frame.Low.iloc[-21:-1].min()) if len(frame) >= 21 else None,
            "atr_pct": _number(indicators["atr"]["value"] / last.Close * 100)
            if indicators["atr"]["value"] is not None else None,
        },
        "weekly": {
            "latest_partial": partial, "ma_basis": "completed_weeks_only",
            "ma20": _number(completed.Close.tail(20).mean()) if len(completed) >= 20 else None,
            "ma52": _number(completed.Close.tail(52).mean()) if len(completed) >= 52 else None,
        },
        "indicators": indicators,
        "recent_daily": _records(frame.tail(20)), "recent_weekly": _records(weekly.tail(12)),
    }
    artifact = {"schema": packet["schema"], "symbol": symbol, "analysis_date": as_of,
                "raw_vendor_response": raw, "ohlcv": _records(frame), "indicators": full_series}
    content = json.dumps(artifact, ensure_ascii=False, allow_nan=False).encode("utf-8")
    digest = hashlib.sha256(content).hexdigest()
    directory = Path(audit_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{digest}.json"
    if path.exists():
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError("Technical audit artifact checksum mismatch")
    else:
        temporary = directory / f"{digest}.{uuid.uuid4().hex}.tmp"
        try:
            temporary.write_bytes(content)
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
    packet["audit"] = {"path": str(path.resolve()), "sha256": digest}
    return packet
