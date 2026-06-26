"""capital_flow_utils 回归测试 —— 锁 P0~P2 迭代（散户死值修复 / 龙虎榜机构方向 / cf_score 重平衡）。

运行：pytest tradingagents/dataflows/test_capital_flow_utils.py
或    python tradingagents/dataflows/test_capital_flow_utils.py
"""
import pandas as pd

from tradingagents.dataflows.capital_flow_utils import (
    compute_retail_concentration_signal,
    compute_retail_amount_rate,
    compute_lhb_metrics,
    compute_capital_flow_regime,
    compute_capital_flow_score,
    compute_distribution_into_retail,
    compute_insider_distribution,
    assemble_capital_flow_metrics,
)


# ---------------------------------------------------------------------------
# 散户净流入占比口径（akshare）—— 修"净占比当毛买占比"语义 bug
# ---------------------------------------------------------------------------
def test_retail_net_inflow_kogei_separated():
    """akshare 净占比字段 → retail_net_inflow_rate_5d_pct（净），不污染毛买占比字段。"""
    df = pd.DataFrame({
        "medium_net_inflow_rate_pct": [2.0, 3.0, 1.0, 2.5, 1.5],
        "small_net_inflow_rate_pct":  [1.0, 1.5, 0.5, 1.0, 0.5],
    })
    out = compute_retail_amount_rate(df)
    assert out["retail_buy_amount_rate_5d_pct"] is None          # 毛口径无数据
    assert out["retail_net_inflow_rate_5d_pct"] is not None       # 净口径有
    # 毛口径（tushare buy rate）反过来
    df2 = pd.DataFrame({"small_buy_amount_rate_pct": [60.0]*5, "medium_buy_amount_rate_pct": [5.0]*5})
    out2 = compute_retail_amount_rate(df2)
    assert out2["retail_buy_amount_rate_5d_pct"] == 65.0 and out2["retail_net_inflow_rate_5d_pct"] is None


def test_retail_concentration_net_branch():
    """净流入占比口径：≥+8% 且主力派发 → 散户高接盘；澜起式 +3.17% → 中性（小幅净买，不算高接盘）。"""
    # 净流入 +12% + 连续派发 → 高接盘
    assert compute_retail_concentration_signal(None, -4, retail_net_inflow_rate_5d_pct=12.0) == "散户高接盘"
    # 澜起式：净流入仅 +3.17%（散户小幅净买，不是"只占3%"）→ 中性，不误判高接盘
    assert compute_retail_concentration_signal(None, -4, retail_net_inflow_rate_5d_pct=3.17) == "中性"
    # 散户净流出（踩踏）→ 中性
    assert compute_retail_concentration_signal(None, -4, retail_net_inflow_rate_5d_pct=-5.0) == "中性"
    # 净口径但 streak 缺失 → None
    assert compute_retail_concentration_signal(None, None, retail_net_inflow_rate_5d_pct=12.0) is None


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


def test_distribution_top_variant_confirms():
    """步骤2：机构派发给散户——获利盘高位+户数增+主力流出≥2 路共振=确认。"""
    r = compute_distribution_into_retail(
        winner_rate_pct=90, holder_num_qoq_pct=8, net_inflow_streak_days=-4)
    assert r["confirmed"] and r["score"] == 3 and r["retail_takeover"] == "散户高接盘"
    # 加舆情狂热 → 满分 strong
    r2 = compute_distribution_into_retail(
        sentiment_euphoric=True, winner_rate_pct=90, net_inflow_streak_days=-4,
        holder_num_4q_trend="持续上升")
    assert r2["score"] == 4 and r2["strength"] == "strong"


def test_distribution_single_signal_not_confirm():
    """单路（趋势中获利盘高，但主力仍在流入）不算派发。"""
    r = compute_distribution_into_retail(winner_rate_pct=90, net_inflow_streak_days=3)
    assert not r["confirmed"] and r["score"] == 1 and r["retail_takeover"] == "中性"


def test_distribution_block_trade_leg():
    """步骤2扩展：折价大宗作第五路硬数据——折价大宗+户数增=确认。"""
    r = compute_distribution_into_retail(holder_num_qoq_pct=8, block_trade_distribution=True)
    assert r["confirmed"] and "折价大宗（机构让利出货，硬数据）" in r["drivers"]
    # 折价大宗单路（主力仍流入）不足确认
    r2 = compute_distribution_into_retail(block_trade_distribution=True, net_inflow_streak_days=2)
    assert not r2["confirmed"] and r2["score"] == 1
    # 五路满分 → strong
    r3 = compute_distribution_into_retail(
        sentiment_euphoric=True, winner_rate_pct=90, holder_num_4q_trend="持续上升",
        net_inflow_streak_days=-4, block_trade_distribution=True)
    assert r3["score"] == 5 and r3["strength"] == "strong"


def test_insider_distribution():
    """item1：stk_holdertrade → 确定性内部人派发信号（净减持/清仓式/recency）。"""
    cd = "2026-06-26"
    df = pd.DataFrame({
        "in_de": ["DE", "DE"],
        "change_vol": [3000000, 1000000],
        "change_ratio": [0.8, 0.3],          # 占总股本%
        "after_share": [1000000, 5000000],   # 第1笔：3M/(3M+1M)=75%≥50% 清仓式
        "ann_date": ["20260610", "20260601"],
    })
    r = compute_insider_distribution(df, current_date=cd)
    assert r["insider_net_selling"] is True and r["clearing_style"] is True, r
    assert r["net_sell_ratio_pct"] == 1.1 and r["n_sells"] == 2 and r["n_buys"] == 0, r
    # recency：陈旧减持(>120天)被过滤 → None
    assert compute_insider_distribution(df.assign(ann_date=["20250101", "20250101"]),
                                        current_date=cd) is None
    # 小额净减持(<0.3%)且非清仓 → 不投
    small = pd.DataFrame({"in_de": ["DE"], "change_vol": [1000], "change_ratio": [0.1],
                          "after_share": [9_000_000], "ann_date": ["20260610"]})
    assert compute_insider_distribution(small, current_date=cd)["insider_net_selling"] is False
    # 净增持 → 不投
    buy = pd.DataFrame({"in_de": ["IN"], "change_vol": [1000000], "change_ratio": [0.5],
                        "after_share": [9000000], "ann_date": ["20260610"]})
    assert compute_insider_distribution(buy, current_date=cd)["insider_net_selling"] is False
    # 空/None → None
    assert compute_insider_distribution(None) is None
    assert compute_insider_distribution(pd.DataFrame()) is None


def test_distribution_insider_leg():
    """第六路：内部人净减持 + 户数增 = 确认；六路满分 score=6。"""
    r = compute_distribution_into_retail(holder_num_qoq_pct=8, insider_net_selling=True)
    assert r["confirmed"] and "内部人近期净减持（大股东/高管，硬数据）" in r["drivers"], r
    r2 = compute_distribution_into_retail(insider_net_selling=True, net_inflow_streak_days=2)
    assert not r2["confirmed"] and r2["score"] == 1, r2
    r3 = compute_distribution_into_retail(
        sentiment_euphoric=True, winner_rate_pct=90, holder_num_4q_trend="持续上升",
        net_inflow_streak_days=-4, block_trade_distribution=True, insider_net_selling=True)
    assert r3["score"] == 6 and r3["strength"] == "strong", r3


def test_distribution_enriches_retail_signal():
    """assemble：顶部派发(获利盘高+主力流出)即便 winner_rate>50 也判散户高接盘。
    旧套牢口径(winner_rate≤50)抓不到顶部进行中的派发，合成口径补上。"""
    mf = pd.DataFrame({
        "trade_date": [f"2026010{i}" for i in range(1, 6)],
        "main_force_net_amount_yi": [-1.0, -1.2, -0.8, -1.5, -2.0],  # streak -5
    })
    m = assemble_capital_flow_metrics(
        moneyflow_df=mf,
        chip_metrics={"winner_rate_pct": 92},   # 获利盘高位（套牢口径会判中性）
    )
    assert m["distribution_into_retail"]["confirmed"]
    assert m["retail_concentration_signal"] == "散户高接盘"


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
