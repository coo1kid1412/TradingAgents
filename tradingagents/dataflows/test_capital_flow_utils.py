"""capital_flow_utils 回归测试 —— 锁 P0~P2 迭代（散户死值修复 / 龙虎榜机构方向 / cf_score 重平衡）。

运行：pytest tradingagents/dataflows/test_capital_flow_utils.py
或    python tradingagents/dataflows/test_capital_flow_utils.py
"""
import pandas as pd

from tradingagents.dataflows.capital_flow_utils import (
    compute_retail_concentration_signal,
    compute_lhb_metrics,
    compute_capital_flow_regime,
    compute_capital_flow_score,
    assemble_capital_flow_metrics,
)


# ---------------------------------------------------------------------------
# P0：散户死值修复
# ---------------------------------------------------------------------------
def test_p0_retail_signal_varies():
    """散户接盘信号随输入变化（替代恒等 1.0 的旧 retail_takeover_ratio）。"""
    assert compute_retail_concentration_signal(70, -4) == "散户高接盘"  # 高占比+持续派发
    assert compute_retail_concentration_signal(70, -1) == "中性"        # 仅流出一天，不算派发
    assert compute_retail_concentration_signal(55, -4) == "中性"        # 占比不够高
    assert compute_retail_concentration_signal(None, -4) is None        # 缺失


def test_p0_no_spurious_retail_vote_on_streak_minus1():
    """旧 bug：streak=-1（中性）却投 retail '-'。修复后应为 '0'。"""
    m = {"net_inflow_streak_days": -1, "retail_buy_amount_rate_5d_pct": 70,
         "ddx_like_5d_pct_1y": 50, "northbound_data_status": "missing"}
    votes = compute_capital_flow_regime(m)["capital_flow_votes"]
    assert votes["streak"] == "0"
    assert votes["retail_takeover"] == "0"  # 不再凭空投 '-'
    # 真派发（streak≤-3 + 高占比）才投 '-'
    m["net_inflow_streak_days"] = -4
    assert compute_capital_flow_regime(m)["capital_flow_votes"]["retail_takeover"] == "-"


def test_p0_retail_subscore_not_constant():
    """cf_score 散户子分随买占比变化（旧恒等 50）。"""
    def sub(rate):
        _, bd = compute_capital_flow_score(
            {"ddx_like_5d_pct_1y": 70, "net_inflow_streak_days": -4,
             "retail_buy_amount_rate_5d_pct": rate}, regime="中性")
        return bd["retail_sub_score"]
    assert sub(50) == 100.0 and sub(75) == 0.0 and sub(50) != sub(70)


def test_p0_dead_field_removed():
    """assemble 输出不再含恒等 1.0 的 retail_takeover_ratio。"""
    df = pd.DataFrame({
        "trade_date": [f"2026052{i}" for i in range(5)],
        "main_force_net_amount_yi": [-1, -1, -2, -1, -1],
        "extra_large_net_amount_yi": [-1] * 5,
        "large_net_amount_yi": [0] * 5,
        "small_buy_amount_rate_pct": [40] * 5,
        "medium_buy_amount_rate_pct": [35] * 5,
    })
    m = assemble_capital_flow_metrics(moneyflow_df=df, circulating_market_value_yi=100.0)
    assert "retail_takeover_ratio" not in m
    assert "retail_concentration_signal" in m


# ---------------------------------------------------------------------------
# P1：龙虎榜机构席位方向
# ---------------------------------------------------------------------------
def test_p1_lhb_inst_direction():
    assert compute_lhb_metrics(5, 0.5)["lhb_inst_direction"] == 1     # 机构净买
    assert compute_lhb_metrics(5, -0.8)["lhb_inst_direction"] == -1   # 机构净卖
    assert compute_lhb_metrics(5, 0.02)["lhb_inst_direction"] == 0    # 阈值内持平
    assert compute_lhb_metrics(5, None)["lhb_inst_direction"] is None # 缺失
    # 上榜次数仅展示，不决定方向
    assert compute_lhb_metrics(9, None)["lhb_count_30d"] == 9


def test_p1_lhb_vote_follows_inst_direction():
    """lhb 投票按机构方向，不再 count≥2→'+'。"""
    def vote(net):
        m = {"net_inflow_streak_days": 1, "ddx_like_5d_pct_1y": 50,
             "northbound_data_status": "missing"}
        m.update(compute_lhb_metrics(5, net))
        return compute_capital_flow_regime(m)["capital_flow_votes"]["lhb"]
    assert vote(0.5) == "+" and vote(-0.8) == "-" and vote(0.0) == "0" and vote(None) == "X"


# ---------------------------------------------------------------------------
# P2：cf_score 重平衡
# ---------------------------------------------------------------------------
_FULL = {
    "ddx_like_5d_pct_1y": 70, "ddz_like_20d_pct": 1.0, "net_inflow_streak_days": 4,
    "retail_buy_amount_rate_5d_pct": 55, "northbound_5d_direction": 1,
    "northbound_data_status": "fresh", "lhb_inst_direction": 1,
}


def test_p2_weights_sum_to_one():
    _, bd = compute_capital_flow_score(_FULL, regime="中性")
    assert abs(bd["effective_weight_sum"] - 1.0) < 1e-9
    assert "northbound_sub_score" in bd and "lhb_inst_sub_score" in bd


def test_p2_northbound_stale_reweights():
    m = dict(_FULL, northbound_data_status="stale")
    _, bd = compute_capital_flow_score(m, regime="中性")
    assert abs(bd["effective_weight_sum"] - 0.80) < 1e-9
    assert "northbound_sub_score" not in bd


def test_p2_institutional_direction_moves_score():
    up = compute_capital_flow_score(dict(_FULL, lhb_inst_direction=1, northbound_5d_direction=1), "中性")[0]
    dn = compute_capital_flow_score(dict(_FULL, lhb_inst_direction=-1, northbound_5d_direction=-1), "中性")[0]
    assert up - dn > 20  # 机构方向反转带来显著差异


def test_p2_regime_clamp_intact():
    assert compute_capital_flow_score(_FULL, "恶化")[0] <= 40
    assert compute_capital_flow_score(_FULL, "强势")[0] >= 60
    assert compute_capital_flow_score(_FULL, "数据不足")[0] is None


# ---------------------------------------------------------------------------
# 正交性：单天流出不应被多维重复计票推到恶化
# ---------------------------------------------------------------------------
def test_orthogonality_single_outflow_day_not_deteriorating():
    """主力仅流出一天(streak=-1) + 高散户占比，旧逻辑 retail 重复投 '-' 易误判。
    修复后 retail/streak 都不投 '-'，不应判恶化。"""
    m = {"net_inflow_streak_days": -1, "retail_buy_amount_rate_5d_pct": 70,
         "ddx_like_5d_pct_1y": 50, "northbound_data_status": "missing",
         "lhb_inst_direction": None}
    regime = compute_capital_flow_regime(m)["capital_flow_regime"]
    assert regime != "恶化"


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
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
