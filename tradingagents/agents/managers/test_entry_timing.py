"""Deterministic entry-timing gate tests.

Run: .venv/bin/python tradingagents/agents/managers/test_entry_timing.py
"""

from __future__ import annotations

import sys

from tradingagents.agents.managers.rm_tools import compute_entry_timing


def _timing(structure_class="trend_pullback", market_mode="risk_on", **kwargs):
    return compute_entry_timing(
        structure_class=structure_class,
        market_mode=market_mode,
        recurring_loss=kwargs.get("recurring_loss", False),
        earnings_revision=kwargs.get("earnings_revision", "停修"),
        valuation_regime=kwargs.get("valuation_regime", "ride"),
        has_peak_signal=kwargs.get("has_peak_signal", False),
        retail_concentration_signal=kwargs.get("retail_concentration_signal", "中性"),
        rsi_percentile_1y=kwargs.get("rsi_percentile_1y", 60),
        capital_flow_regime=kwargs.get("capital_flow_regime", "中性"),
        main_force_streak_days=kwargs.get("main_force_streak_days", 0),
    )


def test_structure_to_base_action_mapping():
    expected = {
        "trend_pullback": "分批介入",
        "breakout_ready": "等放量突破",
        "healthy_trend": "等回踩",
        "exhaustion": "暂不介入",
        "broken": "退出观察",
        "neutral": "继续观察",
        "insufficient_data": "数据不足",
    }
    for structure_class, action in expected.items():
        result = _timing(structure_class)
        assert result["base_action"] == action, result
        assert result["effective_action"] == action, result


def test_conditional_only_downgrades_active_entry():
    assert _timing("trend_pullback", "conditional")["effective_action"] == "小仓试探"
    assert _timing("breakout_ready", "conditional")["effective_action"] == "等放量突破"
    assert _timing("healthy_trend", "conditional")["effective_action"] == "等回踩"


def test_risk_off_vetoes_positive_actions_but_preserves_broken():
    for structure_class in ("trend_pullback", "breakout_ready", "healthy_trend"):
        result = _timing(structure_class, "risk_off")
        assert result["effective_action"] == "暂不介入", result
        assert result["market_mode"] == "risk_off", result
    assert _timing("broken", "risk_off")["effective_action"] == "退出观察"


def test_unknown_market_mode_fails_closed():
    result = _timing("trend_pullback", "unexpected")
    assert result["market_mode"] == "risk_off", result
    assert result["effective_action"] == "暂不介入", result


def test_fundamental_vetoes_block_active_entry():
    cases = [
        {"recurring_loss": True},
        {"earnings_revision": "下修"},
        {"valuation_regime": "discipline"},
        {"has_peak_signal": True},
        {"retail_concentration_signal": "散户高接盘", "rsi_percentile_1y": 88},
        {"capital_flow_regime": "恶化"},
        {"main_force_streak_days": -3},
    ]
    for case in cases:
        result = _timing(**case)
        assert result["effective_action"] == "暂不介入", (case, result)
        assert result["vetoed"] is True, (case, result)
        assert result["reasons"], (case, result)


def test_outflow_is_not_veto_when_earnings_are_revised_up():
    result = _timing(capital_flow_regime="恶化", main_force_streak_days=-5,
                     earnings_revision="上修")
    assert result["effective_action"] == "分批介入", result
    assert result["vetoed"] is False, result


def test_unknown_structure_returns_data_insufficient():
    result = _timing("new_unrecognized_state")
    assert result["base_action"] == "数据不足", result
    assert result["effective_action"] == "数据不足", result


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
