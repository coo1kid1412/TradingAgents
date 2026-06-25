"""PM 市场风险硬闸门的回归测试。"""

from tradingagents.agents.managers.pm_tools import apply_market_risk_gate


def test_high_risk_forces_wait_and_caps_position():
    result = apply_market_risk_gate.invoke({
        "entry_gate": "WAIT", "position_cap_pct": 3,
        "proposed_action": "BUY_NOW", "proposed_size_low_pct": 4, "proposed_size_high_pct": 8,
    })

    assert result["effective_action"] == "WAIT"
    assert result["effective_size_low_pct"] == 3
    assert result["effective_size_high_pct"] == 3


def test_medium_risk_keeps_conditional_entry_but_caps_size():
    result = apply_market_risk_gate.invoke({
        "entry_gate": "CONDITIONAL", "position_cap_pct": 6,
        "proposed_action": "BUY_NOW", "proposed_size_low_pct": 4, "proposed_size_high_pct": 12,
    })

    assert result["effective_action"] == "CONDITIONAL"
    assert result["effective_size_low_pct"] == 4
    assert result["effective_size_high_pct"] == 6
