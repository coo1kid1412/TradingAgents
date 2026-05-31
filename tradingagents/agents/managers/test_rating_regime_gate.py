"""compute_step6_rating_mapping 的 regime 闸门回归测试。

投研做法：估值偏离只定倾向，BUY/SELL 极端由基本面动能(regime)把关。
- ride（基本面强）→ 托底 HOLD（贵不看空，防误杀天孚式趋势票）
- discipline（基本面弱）→ 封顶 HOLD，SELL 保留（贵+恶化=真 Sell，便宜+恶化=价值陷阱）
- neutral → 估值单独不触发极端，收敛 OW/HOLD/UW
- 空 regime → 旧 5 档行为（向后兼容）

运行：python tradingagents/agents/managers/test_rating_regime_gate.py
"""
from tradingagents.agents.managers.rm_tools import compute_step6_rating_mapping as _T

_f = _T.func  # 底层函数（绕过 @tool wrapper）


def _rate(price, mid, reg, dn=27, up=63):
    return _f(price, mid, dn, up, "test", reg)


# 深度高估：100/58 = +72%（>+63 上沿）；深度低估：30/100 = -70%（<-63 上沿）
def test_ride_floors_sell_to_hold():
    """天孚式：ride 趋势票贵到深度高估，本会 SELL → 托底 HOLD（不误杀）。"""
    r = _rate(100, 58, "ride")
    assert r["rating_raw"] == "SELL" and r["rating"] == "HOLD", r


def test_ride_floors_underweight_to_hold():
    r = _rate(100, 72, "ride")  # +38% 高估带 → UNDERWEIGHT
    assert r["rating_raw"] == "UNDERWEIGHT" and r["rating"] == "HOLD", r


def test_ride_keeps_buy():
    """ride + 深度低估 → BUY 保留（便宜 + 强基本面 = 真 Buy）。"""
    r = _rate(30, 100, "ride")
    assert r["rating_raw"] == "BUY" and r["rating"] == "BUY", r


def test_discipline_keeps_sell():
    """澜起式：discipline + 深度高估 → SELL 保留（贵 + 恶化 = 真 Sell）。"""
    r = _rate(100, 58, "discipline")
    assert r["rating_raw"] == "SELL" and r["rating"] == "SELL", r


def test_discipline_caps_buy_to_hold():
    """discipline + 深度低估 → 封顶 HOLD（基本面恶化，便宜也是价值陷阱）。"""
    r = _rate(30, 100, "discipline")
    assert r["rating_raw"] == "BUY" and r["rating"] == "HOLD", r


def test_discipline_caps_overweight_to_hold():
    r = _rate(60, 100, "discipline")  # -40% 低估带 → OVERWEIGHT
    assert r["rating_raw"] == "OVERWEIGHT" and r["rating"] == "HOLD", r


def test_neutral_collapses_extremes():
    """neutral：估值单独不触发极端，SELL→UW, BUY→OW。"""
    hi = _rate(100, 58, "neutral")
    assert hi["rating_raw"] == "SELL" and hi["rating"] == "UNDERWEIGHT", hi
    lo = _rate(30, 100, "neutral")
    assert lo["rating_raw"] == "BUY" and lo["rating"] == "OVERWEIGHT", lo


def test_empty_regime_backward_compat():
    """空 regime → 旧 5 档行为不变（向后兼容）。"""
    assert _rate(100, 58, "")["rating"] == "SELL"
    assert _rate(30, 100, "")["rating"] == "BUY"
    assert _rate(100, 58, None)["rating"] == "SELL"


def test_hold_band_unaffected_by_regime():
    """合理区(±阈值内)→ HOLD，任何 regime 都不动（闸门只管极端侧）。"""
    for reg in ("ride", "neutral", "discipline", ""):
        r = _rate(100, 100, reg)
        assert r["rating"] == "HOLD", (reg, r)


def test_audit_fields_present():
    r = _rate(100, 58, "ride")
    assert r["valuation_regime"] == "ride"
    assert "regime 闸门" in r["explanation"]
    assert r["rating_raw"] == "SELL" and r["rating"] == "HOLD"


if __name__ == "__main__":
    import sys
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn(); print(f"  ✓ {fn.__name__}")
        except AssertionError as e:
            failed += 1; print(f"  ✗ {fn.__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
