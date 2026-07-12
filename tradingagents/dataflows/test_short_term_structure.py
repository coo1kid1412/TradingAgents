"""Deterministic short-term structure classifier tests.

Run: .venv/bin/python tradingagents/dataflows/test_short_term_structure.py
"""

from __future__ import annotations

import math
import sys

import pandas as pd

from tradingagents.dataflows.profile_calc import compute_short_term_structure
from tradingagents.agents.utils.stock_profile_node import _format_short_term_structure_line


def _frame(closes, volumes, *, high_pad=0.01, low_pad=0.01):
    closes = [float(v) for v in closes]
    return pd.DataFrame({
        "date": pd.date_range("2026-01-01", periods=len(closes), freq="B"),
        "Open": closes,
        "High": [v * (1 + high_pad) for v in closes],
        "Low": [v * (1 - low_pad) for v in closes],
        "Close": closes,
        "Volume": [float(v) for v in volumes],
    })


def test_insufficient_data_fails_closed():
    result = compute_short_term_structure(_frame(range(1, 20), [100] * 19))
    assert result["structure_class"] == "insufficient_data", result
    assert result["reasons"], result


def test_trend_pullback_requires_rising_ma_and_shrinking_volume():
    closes = list(range(80, 110)) + [110, 111, 112, 113, 114, 115, 116, 114, 113, 113]
    volumes = [1000] * 30 + [1100] * 5 + [500] * 5
    result = compute_short_term_structure(_frame(closes, volumes))
    assert result["structure_class"] == "trend_pullback", result
    assert result["ma10_slope_5d_pct"] >= 1.0, result
    assert result["volume_ratio_5d_20d"] <= 0.85, result


def test_breakout_ready_is_below_prior_high_on_contracting_volume():
    closes = list(range(80, 100)) + [100, 101, 102, 103, 104, 105, 104, 105, 104, 105,
                                      104.5, 105, 104.8, 105.2, 104.9, 105.1, 104.8, 105.0, 104.9, 105.0]
    volumes = [1000] * 30 + [550] * 10
    result = compute_short_term_structure(_frame(closes, volumes))
    assert result["structure_class"] == "breakout_ready", result
    assert result["distance_from_prior_20d_high_pct"] <= 0, result
    assert result["volume_ratio_5d_20d"] <= 0.85, result


def test_healthy_trend_when_price_is_above_rising_averages():
    closes = [100 + i * 0.5 for i in range(40)]
    result = compute_short_term_structure(_frame(closes, [1000] * 40))
    assert result["structure_class"] == "healthy_trend", result
    assert result["ma10"] > result["ma20"], result


def test_exhaustion_on_low_volume_new_high():
    closes = [100 + i * 0.3 for i in range(39)] + [115]
    volumes = [1000] * 39 + [500]
    result = compute_short_term_structure(_frame(closes, volumes))
    assert result["structure_class"] == "exhaustion", result
    assert result["breakout_confirmed"] is False, result


def test_broken_wins_when_price_breaks_ma20_on_volume():
    closes = [100 + i * 0.4 for i in range(39)] + [93]
    volumes = [1000] * 39 + [1800]
    result = compute_short_term_structure(_frame(closes, volumes))
    assert result["structure_class"] == "broken", result
    assert result["price_vs_ma20_pct"] < 0, result


def test_zero_recent_volume_cannot_create_positive_structure():
    closes = [100 + i * 0.4 for i in range(40)]
    result = compute_short_term_structure(_frame(closes, [1000] * 35 + [0] * 5))
    assert result["structure_class"] not in {"trend_pullback", "breakout_ready"}, result
    assert result["volume_ratio_5d_20d"] is None, result


def test_output_is_finite_and_repeatable():
    frame = _frame([100 + math.sin(i / 4) + i * 0.2 for i in range(60)], [1000] * 60)
    first = compute_short_term_structure(frame)
    second = compute_short_term_structure(frame.copy())
    assert first == second
    for key in ("ma10", "ma20", "ma10_slope_5d_pct", "price_vs_ma10_pct"):
        assert first[key] is None or math.isfinite(first[key]), (key, first)


def test_truth_line_has_stable_fields_and_boolean_format():
    line = _format_short_term_structure_line({
        "structure_class": "trend_pullback",
        "ma10_slope_5d_pct": 2.3456,
        "price_vs_ma10_pct": None,
        "volume_ratio_5d_20d": 0.7123,
        "breakout_confirmed": False,
    })
    assert line == (
        "SYS_SHORT_TERM_STRUCTURE: class=trend_pullback | "
        "ma10_slope_5d_pct=2.35 | price_vs_ma10_pct=N/A | "
        "volume_ratio_5d_20d=0.71 | breakout_confirmed=false"
    ), line


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for test in tests:
        try:
            test()
            print(f"  PASS {test.__name__}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"  FAIL {test.__name__}: [{type(exc).__name__}] {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
