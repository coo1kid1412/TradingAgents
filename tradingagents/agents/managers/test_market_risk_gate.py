"""PM 市场风险硬闸门的回归测试。"""

from tradingagents.agents.managers.pm_tools import apply_market_risk_gate


def test_high_risk_forces_wait_and_caps_position():
    result = apply_market_risk_gate.invoke({
        "entry_gate": "WAIT", "position_cap_pct": 3,
        "proposed_action": "BUY_NOW", "proposed_size_low_pct": 4, "proposed_size_high_pct": 8,
    })

    assert result["effective_action"] == "WAIT"
    assert result["effective_size_low_pct"] == 0
    assert result["effective_size_high_pct"] == 0
    assert result["position_cap_pct"] == 3


def test_medium_risk_waits_now_but_preserves_future_conditional_cap():
    result = apply_market_risk_gate.invoke({
        "entry_gate": "CONDITIONAL", "position_cap_pct": 6,
        "proposed_action": "BUY_NOW", "proposed_size_low_pct": 4, "proposed_size_high_pct": 12,
    })

    assert result["effective_action"] == "WAIT"
    assert result["effective_size_low_pct"] == 0
    assert result["effective_size_high_pct"] == 0
    assert result["position_cap_pct"] == 6


def test_warning_orange_uses_existing_conditional_vocabulary_and_three_percent_cap():
    result = apply_market_risk_gate.invoke({
        "entry_gate": "CONDITIONAL", "position_cap_pct": 3,
        "proposed_action": "BUY_NOW", "proposed_size_low_pct": 2, "proposed_size_high_pct": 8,
    })

    assert result["effective_action"] == "WAIT"
    assert result["effective_size_low_pct"] == 0
    assert result["effective_size_high_pct"] == 0
    assert result["position_cap_pct"] == 3


def test_warning_unknown_uses_existing_wait_vocabulary():
    result = apply_market_risk_gate.invoke({
        "entry_gate": "WAIT", "position_cap_pct": 0,
        "proposed_action": "BUY_NOW", "proposed_size_low_pct": 2, "proposed_size_high_pct": 8,
    })

    assert result["effective_action"] == "WAIT"
    assert result["effective_size_low_pct"] == 0
    assert result["effective_size_high_pct"] == 0


if __name__ == "__main__":
    import sys
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"  ✓ {fn.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"  ✗ {fn.__name__}: {e}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"  ✗ {fn.__name__}: [{type(e).__name__}] {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
