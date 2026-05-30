"""valuation_regime 回归测试（Phase 1）——五路合成 + cap regime 条件化。

中际旭创式(主升浪)→ride→cap 放松；澜起式(派发)→discipline→cap 收紧。

运行：python tradingagents/dataflows/test_valuation_regime.py
"""
from tradingagents.dataflows.profile_calc import (
    compute_valuation_regime,
    parse_growth_deceleration,
    parse_distribution_signals,
)
from tradingagents.agents.utils.stock_profile_node import (
    _parse_capital_flow_signals,
    _enforce_target_pe_cap,
)


# ---------------------------------------------------------------------------
# compute_valuation_regime 五路合成
# ---------------------------------------------------------------------------
def test_ride_uptrend():
    """中际旭创式主升浪：强动量+机构净买+高增速+acceleration → ride。"""
    r = compute_valuation_regime(
        momentum_score=80, rsi_percentile_1y=70, has_peak_signal=False,
        capital_flow_regime="强势", main_force_streak_days=5, lhb_inst_direction=1,
        net_profit_growth=0.9, retail_concentration_signal="中性",
        theme_stage_inferred="acceleration", quant_anticrowding=55)
    assert r["valuation_regime"] == "ride", r


def test_discipline_distribution():
    """澜起式派发：主力流出+机构净卖+拥挤+顶部 → discipline（即使增速仍高）。"""
    r = compute_valuation_regime(
        momentum_score=45, rsi_percentile_1y=88, has_peak_signal=True,
        capital_flow_regime="恶化", main_force_streak_days=-6, lhb_inst_direction=-1,
        net_profit_growth=0.5, retail_concentration_signal="散户高接盘",
        theme_stage_inferred="peak", quant_anticrowding=37)
    assert r["valuation_regime"] == "discipline", r
    assert r["legs"]["earnings"] == 1  # 盈利仍正，但被其余四路压成 discipline


def test_neutral_mixed():
    r = compute_valuation_regime(
        momentum_score=55, rsi_percentile_1y=60, has_peak_signal=False,
        capital_flow_regime="中性", main_force_streak_days=1, lhb_inst_direction=0,
        net_profit_growth=0.2, retail_concentration_signal="中性",
        theme_stage_inferred="none", quant_anticrowding=50)
    assert r["valuation_regime"] == "neutral", r


def test_insufficient_data_neutral():
    r = compute_valuation_regime(momentum_score=80)  # 仅 1 路
    assert r["valuation_regime"] == "neutral", r


def test_peak_signal_blocks_ride():
    """peak 信号触发时即使其余偏多，也不许 ride。"""
    r = compute_valuation_regime(
        momentum_score=90, rsi_percentile_1y=95, has_peak_signal=True,
        capital_flow_regime="强势", main_force_streak_days=5, lhb_inst_direction=1,
        net_profit_growth=0.9, theme_stage_inferred="acceleration", quant_anticrowding=60)
    assert r["valuation_regime"] != "ride", r


# ---------------------------------------------------------------------------
# capital_flow_yaml 解析
# ---------------------------------------------------------------------------
def test_parse_capital_flow_signals():
    y = ('CAPITAL_FLOW:\n  capital_flow_regime: "恶化"\n'
         '  capital_flow_regime_reasoning: "主力连续净流出6日"\n'
         '  net_inflow_streak_days: -6\n  lhb_inst_direction: -1\n'
         '  retail_concentration_signal: "散户高接盘"')
    s = _parse_capital_flow_signals(y)
    assert s == {"regime": "恶化", "streak": -6, "lhb_inst_dir": -1, "retail_signal": "散户高接盘"}, s
    # null / 空 容错
    assert _parse_capital_flow_signals("")["regime"] is None
    assert _parse_capital_flow_signals("  retail_concentration_signal: null")["retail_signal"] is None


# ---------------------------------------------------------------------------
# cap regime 条件化（核心修复：主升浪不被低 cap 压死）
# ---------------------------------------------------------------------------
_LLM_OUT = "VALUATION_METHOD:\n  target_pe_range: [89.6, 116.5]\n  primary_method: peg\n"


def test_discipline_clamps():
    """澜起式 discipline：cap=PE_TTM×0.6=72.4 → [89.6,116.5] 被钉回 ≤72.4。"""
    out = _enforce_target_pe_cap(_LLM_OUT, 72.4)
    assert "72.4" in out and "116.5" not in out, out


def test_ride_relaxes():
    """中际旭创式 ride：cap=PE_TTM(≈120) → [89.6,116.5] 不被压低（保留趋势倍数）。"""
    out = _enforce_target_pe_cap(_LLM_OUT, 120.0)
    assert "[89.6, 116.5]" in out, out  # 未触顶，原样保留


# ---------------------------------------------------------------------------
# Phase 3：减速 earnings 腿 + 派发腿
# ---------------------------------------------------------------------------
def test_earnings_decelerating_votes_negative():
    """减速：即使增速水平高(58%)，growth_direction=decelerating → earnings 投 -1。"""
    r = compute_valuation_regime(
        momentum_score=55, net_profit_growth=0.58, growth_direction="decelerating",
        capital_flow_regime="中性", theme_stage_inferred="none")
    assert r["legs"]["earnings"] == -1, r


def test_earnings_accelerating_votes_positive():
    r = compute_valuation_regime(
        momentum_score=55, net_profit_growth=0.58, growth_direction="accelerating",
        capital_flow_regime="中性", theme_stage_inferred="none")
    assert r["legs"]["earnings"] == 1, r


def test_distribution_leg():
    r = compute_valuation_regime(
        momentum_score=55, net_profit_growth=0.2, capital_flow_regime="中性",
        theme_stage_inferred="none", distribution_detected=True)
    assert r["legs"]["distribution"] == -1, r


def test_lanqi_full_discipline():
    """澜起式：减速 + 主力流出 + 减持 → discipline（之前误判 ride/neutral 的根因都补上）。"""
    r = compute_valuation_regime(
        momentum_score=95, rsi_percentile_1y=88, has_peak_signal=False,
        capital_flow_regime="中性", main_force_streak_days=-4, lhb_inst_direction=None,
        net_profit_growth=0.588, growth_direction="decelerating",
        retail_concentration_signal="中性", theme_stage_inferred="none_or_acceleration",
        quant_anticrowding=37.0, distribution_detected=True)
    assert r["valuation_regime"] == "discipline", r
    assert r["legs"]["earnings"] == -1 and r["legs"]["distribution"] == -1, r


def test_ride_threshold_needs_plus3():
    """六路阈值：净 +2 仍 neutral（保守），需 +3 才 ride。"""
    r = compute_valuation_regime(
        momentum_score=80, net_profit_growth=0.5, growth_direction="accelerating",
        capital_flow_regime="中性", theme_stage_inferred="none")  # tech+1, earnings+1 = +2
    assert r["score"] == 2 and r["valuation_regime"] == "neutral", r


# ---------------------------------------------------------------------------
# 新 parser
# ---------------------------------------------------------------------------
def test_parse_growth_deceleration():
    decel = "| **营收同比增速** | +19.51% | **+49.94%** | +57.83% | Q1放缓 |"
    assert parse_growth_deceleration(decel) == "decelerating"
    accel = "| 营收同比增速 | +52% | +49% | |"
    assert parse_growth_deceleration(accel) == "accelerating"
    assert parse_growth_deceleration("无相关行") is None
    # 单季格式（无年度基线）：低单季增速 → 弱/减速（澜起 005034 真实格式）
    assert parse_growth_deceleration("| 营收同比增速(Q1单季) | 4.58% | — |") == "decelerating"
    assert parse_growth_deceleration("Q1营收仅同比+4.58%，显著低于") == "decelerating"
    assert parse_growth_deceleration("Q1营收同比+58%") == "accelerating"


def test_parse_distribution_signals():
    news = '第五大股东通过询价转让方式"折价8%"出让，套现约30.58亿元；170余家机构在Q1已披露减持'
    d = parse_distribution_signals(news)
    assert d["detected"] and len(d["reasons"]) >= 2, d
    # 否定语境不误报
    assert parse_distribution_signals("未发现明显治理红旗（无高管密集减持）")["detected"] is False


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
