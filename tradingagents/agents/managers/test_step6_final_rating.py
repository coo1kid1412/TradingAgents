"""锁 P0：compute_step6_final_rating（评级终段一次合议）的边界与不变量。

背景（两类真实事故）：
  ① 天孚式：regime=ride 把 SELL 托底成 HOLD（语义=贵+强趋势，不看空也不看多），
     下游趋势叠加不知情又 +1 → OVERWEIGHT——偏离 +133% 的票评级看多。
  ② 拥挤绕道：第四步"拥挤多头禁 BUY"降成 OVERWEIGHT 后，第六步叠加 +1 又回 BUY。

本测试锁：
  - 不变量 A（评级方向与隐含收益同号）双向生效
  - 不变量 B（闸门边界对下游持续生效）：ride 地板 / discipline 天花板 / 拥挤边界
  - 对称升降档的升档全条件 / 降档叠加封顶 -2
  - 动态阈值公式与 fading 上沿锁
  - 极端背离防御及其"拐点刚确认"例外
  - 中性输入下与旧链（mapping → overlay）逐位等价（合并不改数学）

运行：python tradingagents/agents/managers/test_step6_final_rating.py
"""
from tradingagents.agents.managers.rm_tools import (
    compute_step6_final_rating as F,
    compute_step6_rating_mapping as MAP,
    compute_step6_trend_overlay as OV,
    _classify_inflection,
)


def test_inflection_classifier_compound_label():
    """inflection 复合标签子串误匹配 bug（协创『加速期顶部』同时踩升档与降档触发器）。"""
    assert _classify_inflection("加速期") == "accel"
    assert _classify_inflection("底部反转") == "accel"
    assert _classify_inflection("顶部") == "top"
    assert _classify_inflection("衰退") == "top"
    assert _classify_inflection("加速期顶部") == "neutral"   # 矛盾复合标签 → 不升不降
    assert _classify_inflection("顶部加速期") == "neutral"
    assert _classify_inflection("拐点期") == "neutral"
    assert _classify_inflection("") == "neutral"
    # 协创回放：低估 -25% + 复合标签 → 顶部降档不再误触发（只剩 bear anchor -1）
    r = F.invoke({"current_price": 223.7, "target_price_mid": 300.65, "style": "high_beta_growth",
                  "valuation_regime": "ride", "inflection_stage": "加速期顶部",
                  "data_completeness": "L2", "bear_anchor_strong": True,
                  "earnings_sustainability": "待验证"})
    assert r["stages"]["symmetric"]["downgrade"] == -1     # 不再 -2（顶部子串不再误触发）
    assert r["final_rating"] == "HOLD"                      # 干净路径到 HOLD（非靠不变量救过冲）
    # 纯顶部仍正常降档
    r2 = F.invoke({"current_price": 280, "target_price_mid": 248, "style": "high_beta_growth",
                   "valuation_regime": "neutral", "inflection_stage": "顶部", "data_completeness": "L2"})
    assert r2["stages"]["symmetric"]["downgrade"] == -1
    print("✓ inflection 复合标签归 neutral，纯顶部正常降档（协创 bug 修复）")


def _base(**kw):
    """缺省入参：中性环境（无拥挤/无升降档触发/无叠加信号）。"""
    p = {
        "current_price": 100.0,
        "target_price_mid": 100.0,
        "style": "blue_chip",
        "theme_premium_pct": 0.0,
        "theme_stage": "none",
        "valuation_regime": "",
        "peg_confidence": "",
        "consensus_crowded": False,
        "consensus_direction": "",
        "inflection_stage": "",
        "data_completeness": "L1",
        "red_flags_count": 0,
        "earnings_sustainability": "持续",
        "bear_anchor_strong": False,
        "decision_style": "value_anchor",
        "composite_score": 50.0,
        "momentum_score": 50.0,
    }
    p.update(kw)
    return p


def test_tianfu_case_ride_floor_not_upgraded():
    """天孚案例：偏离 +134%、ride 托底、强动量成长股 → 必须 HOLD，不得 OVERWEIGHT。"""
    r = F.invoke(_base(
        current_price=459.0, target_price_mid=196.0,
        style="high_beta_growth", theme_premium_pct=50.0, theme_stage="acceleration",
        valuation_regime="ride",
        composite_score=72.0, momentum_score=80.0,
        market_weight=0.4, news_weight=0.3, sentiment_weight=0.3,
        market_direction_vote=1.0, news_direction_vote=0.5, sentiment_direction_vote=0.5,
    ))
    assert r["rating_raw"] == "SELL", r["rating_raw"]
    assert r["rating_after_gate"] == "HOLD"
    # 叠加路确实想升（说明本测试覆盖了事故路径），但被不变量拦下
    assert r["stages"]["overlay"]["rating_overlay_raw"] in ("OVERWEIGHT", "HOLD")
    assert r["final_rating"] == "HOLD", r["explanation"]
    assert "不变量" in r["explanation"] or r["stages"]["overlay"]["rating_overlay_raw"] == "HOLD"
    print("✓ 天孚案例：ride 托底 + 强动量 → HOLD（不再 OVERWEIGHT）")


def test_e_sign_invariant_blocks_bullish_negative_e():
    """不变量 A：评级看多但目标价中位 ≤ 现价 → 收敛 HOLD。"""
    # 构造：价高于目标 10%（HOLD 区内），升档条件全满足也不该升——升档需偏离<0
    r = F.invoke(_base(
        current_price=110.0, target_price_mid=100.0,
        inflection_stage="加速期", data_completeness="L0", red_flags_count=0,
    ))
    assert r["final_rating"] in ("HOLD", "UNDERWEIGHT"), r["final_rating"]
    assert r["final_rating"] != "OVERWEIGHT"
    print("✓ 不变量 A：隐含收益为负时升不到看多档")


def test_e_sign_invariant_blocks_bearish_positive_e():
    """不变量 A 反向：评级看空但目标价中位 ≥ 现价 → 收敛 HOLD。"""
    # 低估区（偏离 -10%，HOLD 区内），降档条件堆满想压到 SELL
    r = F.invoke(_base(
        current_price=90.0, target_price_mid=100.0,
        data_completeness="L3", red_flags_count=4, inflection_stage="顶部",
    ))
    # 降档最多 -2：HOLD → SELL；但偏离 -10% 为正收益 → 不变量收敛 HOLD
    assert r["final_rating"] == "HOLD", (r["final_rating"], r["explanation"])
    assert "违反" in r["stages"]["e_sign_invariant"]["note"]
    print("✓ 不变量 A 反向：隐含收益为正时降不到看空档")


def test_crowded_long_ceiling_survives_overlay():
    """拥挤绕道封堵：拥挤多头（含硬确认）天花板 OVERWEIGHT 对趋势叠加持续生效。"""
    # 深度低估（题材股阈值 ±30/70，偏离 -75% 落 BUY 区）+ 拥挤多头 → 降为 OW；
    # 强动量题材股叠加 +1 想回 BUY → 被天花板钳住
    r = F.invoke(_base(
        current_price=25.0, target_price_mid=100.0,
        style="theme_speculation", theme_premium_pct=0.0,
        consensus_crowded=True, consensus_direction="偏多",
        quant_anticrowding=25.0,   # 硬确认：反拥挤分≤30
        composite_score=60.0, momentum_score=70.0,
    ))
    assert r["rating_raw"] == "BUY"
    assert r["stages"]["crowding"]["rating_after"] == "OVERWEIGHT"
    assert r["final_rating"] != "BUY", r["explanation"]
    assert any("拥挤多头" in s for s in r["bounds"]["sources"])
    print("✓ 拥挤多头禁 BUY：趋势叠加无法绕回 BUY")


def test_crowded_without_hard_confirm_not_triggered():
    """共识官标拥挤但无硬数据确认 → 拥挤闸不触发（软标志单独不可靠）。"""
    r = F.invoke(_base(
        current_price=25.0, target_price_mid=100.0,
        style="theme_speculation", theme_premium_pct=0.0,
        consensus_crowded=True, consensus_direction="偏多",
        quant_anticrowding=55.0,                 # 反拥挤分健康
        retail_concentration_signal="中性",       # 无散户高接盘
    ))
    assert r["rating_raw"] == "BUY"
    assert r["final_rating"] == "BUY", r["explanation"]   # 闸不触发，BUY 保留
    assert "不触发" in r["stages"]["crowding"]["note"]
    assert not any("拥挤" in s for s in r["bounds"]["sources"])
    print("✓ 软拥挤标志无硬确认 → 不动评级")


def test_crowded_confirmed_by_retail_signal():
    """硬确认第二路：散户高接盘单独也能确认拥挤多头。"""
    r = F.invoke(_base(
        current_price=25.0, target_price_mid=100.0,
        style="theme_speculation", theme_premium_pct=0.0,
        consensus_crowded=True, consensus_direction="偏多",
        quant_anticrowding=55.0,
        retail_concentration_signal="散户高接盘",
    ))
    assert r["final_rating"] != "BUY"
    assert any("散户高接盘" in s for s in r["bounds"]["sources"])
    print("✓ 散户高接盘单独确认拥挤")


def test_crowded_short_floor_survives():
    """拥挤空头（含硬确认）地板 UNDERWEIGHT：SELL 收敛且下游不得再压回。"""
    r = F.invoke(_base(
        current_price=200.0, target_price_mid=100.0,
        consensus_crowded=True, consensus_direction="偏空",
        quant_anticrowding=20.0,
    ))
    assert r["rating_raw"] == "SELL"
    assert r["final_rating"] == "UNDERWEIGHT", r["explanation"]
    print("✓ 拥挤空头禁 SELL：收敛 UNDERWEIGHT")


def test_discipline_ceiling_not_reupgraded():
    """discipline 封顶的 HOLD 不被叠加升回看多。"""
    # 深度低估（成长股阈值 ±22.5/52.5，偏离 -60% 落 BUY 区）+ discipline → 封顶 HOLD；
    # 强动量想 +1 → 天花板钳回
    r = F.invoke(_base(
        current_price=40.0, target_price_mid=100.0,
        style="high_beta_growth", valuation_regime="discipline",
        composite_score=70.0, momentum_score=80.0,
    ))
    assert r["rating_raw"] == "BUY"
    assert r["rating_after_gate"] == "HOLD"
    assert r["final_rating"] == "HOLD", r["explanation"]
    print("✓ discipline 封顶：弱基本面优质价不被叠加抬回看多")


def test_symmetric_upgrade_path():
    """对称升档：拐点加速 + L0 + 红旗≤1 + 低估区 + 非 momentum → HOLD 升 OVERWEIGHT。"""
    r = F.invoke(_base(
        current_price=90.0, target_price_mid=100.0,   # 偏离 -10%，HOLD 区且 <0
        inflection_stage="加速期", data_completeness="L0", red_flags_count=0,
    ))
    assert r["stages"]["symmetric"]["upgrade"] == 1
    assert r["final_rating"] == "OVERWEIGHT", r["explanation"]
    print("✓ 对称升档：低估+拐点加速+数据可信 → OVERWEIGHT")


def test_symmetric_upgrade_blocked_for_momentum_style():
    """momentum 决策风格不靠低估升档。"""
    r = F.invoke(_base(
        current_price=90.0, target_price_mid=100.0,
        inflection_stage="加速期", data_completeness="L0", red_flags_count=0,
        decision_style="momentum",
    ))
    assert r["stages"]["symmetric"]["upgrade"] == 0
    assert r["final_rating"] == "HOLD"
    print("✓ momentum 风格不触发低估升档")


def test_symmetric_downgrade_capped_at_2():
    """降档四条全踩也最多 -2。"""
    r = F.invoke(_base(
        current_price=120.0, target_price_mid=100.0,   # +20%：UNDERWEIGHT 区
        data_completeness="L3", red_flags_count=5, inflection_stage="衰退",
        bear_anchor_strong=True, earnings_sustainability="待验证",
    ))
    assert r["stages"]["symmetric"]["downgrade"] == -2
    assert r["final_rating"] == "SELL", r["explanation"]   # UW -2 钳到 SELL（已到边界）
    print("✓ 降档叠加封顶 -2")


def test_threshold_formula_and_fading_lock():
    """阈值 = 15/35 × style × theme；fading 上沿锁 30。"""
    r = F.invoke(_base(style="high_beta_growth", theme_premium_pct=50.0))
    assert abs(r["threshold_dn_pct"] - 15 * 1.5 * 1.5) < 0.01, r["threshold_dn_pct"]
    assert abs(r["threshold_up_pct"] - 35 * 1.5 * 1.5) < 0.01, r["threshold_up_pct"]
    r2 = F.invoke(_base(style="high_beta_growth", theme_premium_pct=-20.0, theme_stage="fading"))
    assert r2["threshold_up_pct"] == 30.0, r2["threshold_up_pct"]
    print("✓ 动态阈值公式 + fading 上沿锁")


def test_extreme_defense_and_exception():
    """composite≤20 压看多 → HOLD；拐点刚确认则跳过。"""
    p = _base(current_price=60.0, target_price_mid=100.0, composite_score=15.0)
    r = F.invoke(p)
    assert r["final_rating"] == "HOLD", r["explanation"]
    assert "≤20" in r["stages"]["extreme_defense"]["note"]
    r2 = F.invoke({**p, "inflection_confirmed_recent": True})
    assert r2["final_rating"] in ("BUY", "OVERWEIGHT"), r2["explanation"]
    print("✓ 极端背离防御 + 拐点刚确认例外")


def test_neutral_passthrough_equals_legacy_chain():
    """中性输入下与旧链（mapping → overlay）逐位等价：合并不改数学。"""
    cases = [
        dict(current_price=100.0, target_price_mid=120.0, style="blue_chip",
             theme_premium_pct=0.0, valuation_regime="neutral"),
        dict(current_price=150.0, target_price_mid=100.0, style="high_beta_growth",
             theme_premium_pct=30.0, valuation_regime="", composite_score=55.0,
             momentum_score=60.0),
        dict(current_price=80.0, target_price_mid=100.0, style="cyclical",
             theme_premium_pct=0.0, valuation_regime="neutral", composite_score=80.0,
             momentum_score=80.0),
    ]
    style_coef = {"blue_chip": 1.0, "cyclical": 1.0, "high_beta_growth": 1.5}
    for c in cases:
        merged = F.invoke(_base(**c))
        thr_dn = max(15 * style_coef[c["style"]] * (1 + c["theme_premium_pct"] / 100), 5.0)
        thr_up = max(35 * style_coef[c["style"]] * (1 + c["theme_premium_pct"] / 100), thr_dn + 5.0)
        legacy_map = MAP.invoke({
            "current_price": c["current_price"], "target_price_mid": c["target_price_mid"],
            "threshold_dn_pct": round(thr_dn, 2), "threshold_up_pct": round(thr_up, 2),
            "valuation_regime": c["valuation_regime"], "peg_confidence": "",
        })
        legacy_ov = OV.invoke({
            "rating_after_symmetric": legacy_map["rating"], "style": c["style"],
            "composite_score": c.get("composite_score", 50.0),
            "momentum_score": c.get("momentum_score", 50.0),
        })
        legacy_final = legacy_ov["final_rating"]
        # 旧链没有不变量终检——比对前手工套用同一规则，验证除不变量外逐位一致
        dev = legacy_map["deviation_pct"]
        if legacy_final in ("BUY", "OVERWEIGHT") and dev >= 0:
            legacy_final = "HOLD"
        elif legacy_final in ("UNDERWEIGHT", "SELL") and dev <= 0:
            legacy_final = "HOLD"
        assert merged["deviation_pct"] == legacy_map["deviation_pct"], c
        assert merged["rating_raw"] == legacy_map["rating_raw"], c
        assert merged["final_rating"] == legacy_final, (c, merged["explanation"])
    print("✓ 中性输入下与旧链逐位等价（3 组）")


def test_cyclical_top_blocks_upgrade_and_overlay():
    """强周期顶部：对称升档禁用 + 趋势叠加正向钳零（顶部要下车不是骑）。"""
    # 低估区 + 拐点加速 + 数据可信（非周期股会升 OW 再被叠加抬 BUY 的组合）
    p = _base(
        current_price=90.0, target_price_mid=100.0,
        style="high_beta_growth",
        inflection_stage="加速期", data_completeness="L0", red_flags_count=0,
        composite_score=70.0, momentum_score=80.0,   # 成长股叠加 +1 阈值满足
    )
    no_cyc = F.invoke(p)
    assert no_cyc["final_rating"] in ("OVERWEIGHT", "BUY")   # 非周期：升档/叠加生效

    top = F.invoke({**p, "cyclical_class": "strong", "cycle_position": "top"})
    assert top["stages"]["symmetric"]["upgrade"] == 0, top["stages"]["symmetric"]
    assert top["final_rating"] == "HOLD", top["explanation"]
    assert "顶部" in str(top["stages"]["symmetric"]["notes"])
    print("✓ 强周期顶部：升档禁用 + 叠加正向钳零 → HOLD")


def test_cyclical_trough_mutes_inflection_downgrade():
    """强周期谷底：『拐点衰退』降档静音（谷底盈利差是常态，不在底部追杀）。"""
    p = _base(
        current_price=110.0, target_price_mid=100.0,   # UW 区边内(HOLD 区上沿内)
        inflection_stage="衰退", data_completeness="L1", red_flags_count=0,
    )
    no_cyc = F.invoke(p)
    assert no_cyc["stages"]["symmetric"]["downgrade"] == -1   # 非周期：衰退降档

    trough = F.invoke({**p, "cyclical_class": "strong", "cycle_position": "trough"})
    assert trough["stages"]["symmetric"]["downgrade"] == 0, trough["stages"]["symmetric"]
    assert "静音" in str(trough["stages"]["symmetric"]["notes"])
    # 数据质量降档不受周期豁免
    bad_data = F.invoke({**p, "cyclical_class": "strong", "cycle_position": "trough",
                         "data_completeness": "L3"})
    assert bad_data["stages"]["symmetric"]["downgrade"] == -1
    print("✓ 强周期谷底：拐点衰退静音，数据质量降档保留")


def test_semi_cyclical_not_affected():
    """半周期不触发顶部/谷底修正（保留成长β语义）。"""
    p = _base(
        current_price=90.0, target_price_mid=100.0,
        inflection_stage="加速期", data_completeness="L0", red_flags_count=0,
        cyclical_class="semi", cycle_position="top",
    )
    r = F.invoke(p)
    assert r["stages"]["symmetric"]["upgrade"] == 1, r["stages"]["symmetric"]
    print("✓ 半周期不触发周期修正")


def test_mapping_error_propagates():
    """target_price_mid ≤ 0 时返回错误而非崩溃。"""
    r = F.invoke(_base(target_price_mid=0.0))
    assert "error" in r
    print("✓ 非法输入返回 error")


if __name__ == "__main__":
    test_tianfu_case_ride_floor_not_upgraded()
    test_e_sign_invariant_blocks_bullish_negative_e()
    test_e_sign_invariant_blocks_bearish_positive_e()
    test_crowded_long_ceiling_survives_overlay()
    test_crowded_without_hard_confirm_not_triggered()
    test_crowded_confirmed_by_retail_signal()
    test_crowded_short_floor_survives()
    test_discipline_ceiling_not_reupgraded()
    test_symmetric_upgrade_path()
    test_symmetric_upgrade_blocked_for_momentum_style()
    test_symmetric_downgrade_capped_at_2()
    test_threshold_formula_and_fading_lock()
    test_extreme_defense_and_exception()
    test_neutral_passthrough_equals_legacy_chain()
    test_cyclical_top_blocks_upgrade_and_overlay()
    test_cyclical_trough_mutes_inflection_downgrade()
    test_semi_cyclical_not_affected()
    test_inflection_classifier_compound_label()
    test_mapping_error_propagates()
    print("\n全部 19 项通过 ✅")
