"""锁 T1：compute_step6_trend_overlay（合并工具）必须与「顺序调 4 个原工具」逐位一致。

合并工具把 Step 6 第六步的 4 次工具调用（style→vote→catalyst→synthesis）压成 1 次，
目的是减少 RM 工具循环轮数（11-15 → 8-11，远离 15 轮上限）。本测试保证这个合并
**不改任何数学结果**——final_rating / final_adjustment / raw_sum / 三路 components 全等。

运行：python tradingagents/agents/managers/test_step6_trend_overlay.py
"""
from tradingagents.agents.managers.rm_tools import (
    compute_step6_style_adjustment as S,
    compute_step6_report_weighted_vote_adjustment as V,
    compute_step6_catalyst_momentum_adjustment as C,
    compute_step6_adjustment_synthesis as SY,
    compute_step6_trend_overlay as M,
)


def _manual_chain(p: dict):
    """精确复刻历史 4 工具链（style→vote→catalyst→synthesis）。"""
    r0 = p["rating_after_symmetric"]
    sr = S.invoke({"rating_after_mechanical": r0, "style": p["style"],
                   "composite_score": p.get("composite_score"), "momentum_score": p.get("momentum_score")})
    r1 = sr.get("new_rating", r0)
    vr = V.invoke({"rating_after_style_adj": r1,
                   "market_weight": p.get("market_weight", 0), "news_weight": p.get("news_weight", 0),
                   "sentiment_weight": p.get("sentiment_weight", 0),
                   "market_direction_vote": p.get("market_direction_vote", 0),
                   "news_direction_vote": p.get("news_direction_vote", 0),
                   "sentiment_direction_vote": p.get("sentiment_direction_vote", 0)})
    r2 = vr.get("new_rating", r1)
    cr = C.invoke({"rating_after_vote_adj": r2,
                   "sell_side_target_change_pct": p.get("sell_side_target_change_pct"),
                   "institutional_holding_change_pct": p.get("institutional_holding_change_pct"),
                   "northbound_flow_5d_direction": p.get("northbound_flow_5d_direction"),
                   "kol_bullish_ratio_trend_pct": p.get("kol_bullish_ratio_trend_pct")})
    sy = SY.invoke({"rating_after_symmetric": r0, "style_adjustment": sr.get("adjustment", 0),
                    "vote_adjustment": vr.get("adjustment", 0), "catalyst_adjustment": cr.get("adjustment", 0)})
    return (sy.get("new_rating"), sy.get("final_adjustment"), sy.get("raw_sum"),
            (sr.get("adjustment", 0), vr.get("adjustment", 0), cr.get("adjustment", 0)))


_CASES = [
    # 普通：高成长加速，三信号偏多
    dict(rating_after_symmetric="HOLD", style="high_beta_growth", composite_score=75, momentum_score=80,
         market_weight=40, news_weight=30, sentiment_weight=30, market_direction_vote=1,
         news_direction_vote=0.5, sentiment_direction_vote=1, sell_side_target_change_pct=20,
         institutional_holding_change_pct=12, northbound_flow_5d_direction=1, kol_bullish_ratio_trend_pct=15),
    # 边界：style 先把 OW 顶到 BUY（极值），后续同向应被归零
    dict(rating_after_symmetric="OVERWEIGHT", style="theme_speculation", composite_score=60, momentum_score=70,
         market_weight=33, news_weight=33, sentiment_weight=34, market_direction_vote=1,
         news_direction_vote=1, sentiment_direction_vote=1, sell_side_target_change_pct=20,
         institutional_holding_change_pct=12, northbound_flow_5d_direction=1, kol_bullish_ratio_trend_pct=15),
    # 混合方向：style+1 顶到 BUY，vote-1 拉回，catalyst-1
    dict(rating_after_symmetric="OVERWEIGHT", style="high_beta_growth", composite_score=70, momentum_score=75,
         market_weight=50, news_weight=30, sentiment_weight=20, market_direction_vote=-1,
         news_direction_vote=-1, sentiment_direction_vote=-0.5, sell_side_target_change_pct=-20,
         institutional_holding_change_pct=-15, northbound_flow_5d_direction=-1, kol_bullish_ratio_trend_pct=-15),
    # 全空：SELL 底部，向下应被归零
    dict(rating_after_symmetric="SELL", style="theme_speculation", composite_score=20, momentum_score=20,
         market_weight=40, news_weight=30, sentiment_weight=30, market_direction_vote=-1,
         news_direction_vote=-1, sentiment_direction_vote=-1, sell_side_target_change_pct=-20,
         institutional_holding_change_pct=-15, northbound_flow_5d_direction=-1, kol_bullish_ratio_trend_pct=-15),
    # blue_chip 永不 style 调，vote 中性，catalyst 仅 1 项 → skip
    dict(rating_after_symmetric="HOLD", style="blue_chip", composite_score=90, momentum_score=90,
         market_weight=60, news_weight=30, sentiment_weight=10, market_direction_vote=0,
         news_direction_vote=0, sentiment_direction_vote=0, sell_side_target_change_pct=5),
    # catalyst 仅 1 项 → skip；style cyclical 高分 +1
    dict(rating_after_symmetric="UNDERWEIGHT", style="cyclical", composite_score=80, momentum_score=80,
         market_weight=50, news_weight=50, sentiment_weight=0, market_direction_vote=1,
         news_direction_vote=1, sentiment_direction_vote=0, northbound_flow_5d_direction=1),
]


def test_merged_equals_chain():
    for i, p in enumerate(_CASES, 1):
        expected = _manual_chain(p)
        m = M.invoke(p)
        got = (m["final_rating"], m["final_adjustment"], m["raw_sum"],
               (m["components"]["style"], m["components"]["vote"], m["components"]["catalyst"]))
        assert got == expected, f"case{i} 不一致：链={expected} 合并={got}"


if __name__ == "__main__":
    import sys
    try:
        test_merged_equals_chain()
        print(f"  ✓ test_merged_equals_chain（{len(_CASES)} 个 case 逐位一致）")
        print(f"\n1/1 passed")
    except AssertionError as e:
        print(f"  ✗ {e}")
        sys.exit(1)
