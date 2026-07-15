"""Quality-aware quant factor regression tests.

Run: .venv/bin/python tradingagents/dataflows/test_factor_quality_gates.py
"""

from tradingagents.dataflows.factor_calc import anticrowding_score, growth_score
from tradingagents.dataflows.capital_flow_utils import compute_capital_flow_score


def test_recurring_loss_caps_headline_growth_score():
    score, detail = growth_score(
        13.88, 111.78, recurring_loss=True, deducted_profit_yoy_pct=-148.14,
    )
    assert score <= 35, (score, detail)
    assert detail["quality_gate"] == "recurring_loss"


def test_anticrowding_penalizes_distributed_trapped_stock():
    score, detail = anticrowding_score(
        -23.92, 1.08, holder_num_qoq_pct=22.08, winner_rate_pct=21.03,
    )
    assert score <= 40, (score, detail)
    assert detail["distribution_penalty"] is True


def test_recent_outflow_and_holder_spread_cap_stale_capital_flow_strength():
    metrics = {
        "ddx_like_5d_pct_1y": 75.5,
        "ddz_like_20d_pct": 1.26,
        "net_inflow_streak_days": 1,
        "main_force_net_inflow_5d_yi": -0.33,
        "main_force_net_inflow_20d_yi": 2.24,
        "holder_num_qoq_pct": 22.08,
        "northbound_data_status": "missing",
    }
    score, detail = compute_capital_flow_score(metrics, "中性")
    assert score <= 55, (score, detail)
    assert detail["recent_contradiction_cap"] == "5d流出+股东户数扩散 → ≤55"


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for test in tests:
        test()
        print(f"  PASS {test.__name__}")
    print(f"\n{len(tests)}/{len(tests)} passed")
