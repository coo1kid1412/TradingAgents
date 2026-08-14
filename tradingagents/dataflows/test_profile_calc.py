import pandas as pd

from tradingagents.dataflows.profile_calc import compute_short_term_structure


def _frame(closes, volumes=None):
    volumes = volumes or [1_000_000] * len(closes)
    return pd.DataFrame({
        "Date": pd.date_range("2026-01-01", periods=len(closes), freq="B"),
        "Open": closes,
        "High": [value * 1.02 for value in closes],
        "Low": [value * 0.98 for value in closes],
        "Close": closes,
        "Volume": volumes,
    })


def test_short_rebound_inside_medium_downtrend_is_not_healthy_trend():
    closes = list(range(200, 150, -1)) + list(range(151, 166))
    result = compute_short_term_structure(_frame(closes))

    assert result["ma10_slope_5d_pct"] > 0
    assert result["drawdown_60d_pct"] < -15
    assert result["structure_class"] == "weak_rebound_in_downtrend"


if __name__ == "__main__":
    test_short_rebound_inside_medium_downtrend_is_not_healthy_trend()
    print("1/1 passed")
