"""valuation_regime 回归测试（Phase 1）——五路合成 + cap regime 条件化。

中际旭创式(主升浪)→ride→cap 放松；澜起式(派发)→discipline→cap 收紧。

运行：python tradingagents/dataflows/test_valuation_regime.py
"""
from tradingagents.dataflows.profile_calc import (
    compute_valuation_regime,
    parse_growth_deceleration,
    parse_net_profit_growth,
    parse_distribution_signals,
    recommend_growth_primary_method,
    parse_growth_quality,
    gate_premium_by_regime,
    parse_sys_net_growth_components,
    compute_deterministic_peg_inputs,
    compute_peg_band,
    detect_paradigm_growth,
    parse_sys_paradigm,
)


def test_peg_band_by_regime():
    """确定性 PEG 倍数带——治 RM 自拍倍数致目标价摆动（澜起 267↔401 根）。"""
    assert compute_peg_band("ride") == (1.0, 1.5)        # 强基本面+主题可给溢价
    assert compute_peg_band("neutral") == (0.9, 1.2)     # 中性围绕合理估值
    assert compute_peg_band("discipline") == (0.8, 1.0)  # 弱基本面折价
    assert compute_peg_band(None) == (0.9, 1.2)          # 缺 regime → 中性档
    assert compute_peg_band("ride", "low") == (1.0, 1.1)  # 低置信前瞻压上沿
    assert compute_peg_band("neutral", "low") == (0.9, 1.1)
    # 下限 ≤ 上限恒成立
    for reg in ("ride", "neutral", "discipline", None):
        lo, hi = compute_peg_band(reg)
        assert lo <= hi
    print("✓ PEG 带按 regime 派生，低置信压上沿，下限≤上限")


def test_deterministic_peg_inputs():
    """确定性 PEG 输入：钉死前瞻增速/EPS + 低基数护栏（协创式 320↔180 → OW↔UW 摆动根）。"""
    sysline = ("【SYS_GROWTH_YOY｜tushare】 营收YoY 单季=+192.9% 年度=+65.1% | "
               "归母净利YoY 单季=+343.45% 年度=+68.51% | 扣非净利YoY 年度=+353%")
    comp = parse_sys_net_growth_components(sysline)
    assert abs(comp["annual"] - 0.6851) < 1e-6 and abs(comp["quarter"] - 3.4345) < 1e-6

    # 协创：年度 68.51% 分段衰减 → 40 + (68.51-40)/2 ≈ 54%（不用单季尖峰 343%）；
    # 前瞻 EPS = 4.86×1.54255 ≈ 7.50；低基数 → low
    d = compute_deterministic_peg_inputs(4.86, comp["annual"], comp["quarter"])
    assert d["peg_growth_pct"] == 54          # 40 + (68.51-40)×0.5
    assert d["forward_eps"] == 7.5
    assert d["confidence"] == "low"           # 单季 343 >> 年度 68（>2× 且 >100%）
    assert d["low_base_spike"] is True and d["capped"] is True

    # 可持续区间（年度 30% ≤ 40%）→ 全采信不打折，confidence normal
    d2 = compute_deterministic_peg_inputs(2.0, 0.30, 0.25)
    assert d2["peg_growth_pct"] == 30 and d2["confidence"] == "normal" and d2["capped"] is False

    # 分段边界：40% 恰好不打折；80% 触顶 60%；120% 仍封顶 60%
    assert compute_deterministic_peg_inputs(1.0, 0.40)["peg_growth_pct"] == 40
    assert compute_deterministic_peg_inputs(1.0, 0.80)["peg_growth_pct"] == 60
    assert compute_deterministic_peg_inputs(1.0, 1.20)["peg_growth_pct"] == 60

    # 缺确定性年度增速 / EPS / 衰退（年度≤0）→ None（RM 走原路径，不更差）
    assert compute_deterministic_peg_inputs(None, 0.5, 0.5) is None
    assert compute_deterministic_peg_inputs(4.0, None, 0.5) is None
    assert compute_deterministic_peg_inputs(4.0, -0.1, None) is None


def test_premium_regime_gate_cuts_both_ways():
    """主题溢价 regime 闸门：ride 满/neutral 半/discipline 零——必须两头都切才不是结果倒推。"""
    # acceleration 主题默认 +50%
    assert gate_premium_by_regime(50, "ride")[0] == 50        # ride 全给（天孚/中际旭创受益）
    assert gate_premium_by_regime(50, "neutral")[0] == 25      # neutral 减半
    assert gate_premium_by_regime(50, "discipline")[0] == 0    # discipline 归零（澜起/淳中收紧）
    # 负溢价(fading/宏观收紧)不被放松——收紧保留
    assert gate_premium_by_regime(-20, "discipline")[0] == -20
    assert gate_premium_by_regime(-20, "ride")[0] == -20
    # 正负混合：只压正部分
    assert gate_premium_by_regime(30, "neutral")[0] == 15
    # regime 未知 → 不闸（向后兼容）
    assert gate_premium_by_regime(50, None)[0] == 50
    assert gate_premium_by_regime(None, "discipline")[0] is None
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


def test_detect_paradigm_growth():
    """范式识别：硬科技 secular 命中；周期股(存储)让位；非赛道 None。"""
    assert detect_paradigm_growth(None, "中际旭创") == "paradigm"      # CPO 龙头
    assert detect_paradigm_growth("半导体", "某芯片股") == "paradigm"   # 行业关键词
    assert detect_paradigm_growth(None, "沪电股份") == "paradigm"      # PCB
    # 周期优先：兆易创新是 strong 周期(存储) → 让位周期轨，不抢范式
    assert detect_paradigm_growth(None, "兆易创新") is None
    assert detect_paradigm_growth("钢铁", "某钢铁股") is None          # 传统周期
    assert detect_paradigm_growth("白酒", "贵州茅台") is None          # 非赛道
    # SYS_PARADIGM 解析往返
    assert parse_sys_paradigm("【SYS_PARADIGM｜tushare】 class=paradigm | sector=CPO") is True
    assert parse_sys_paradigm("无此行") is False


def test_paradigm_ride_in_acceleration():
    """范式股加速期：原本 neutral 的边际组合(拥挤拖累)，反转后 → ride。"""
    base = dict(
        momentum_score=70, rsi_percentile_1y=78, has_peak_signal=False,
        capital_flow_regime="中性", main_force_streak_days=1, lhb_inst_direction=0,
        net_profit_growth=0.5, retail_concentration_signal="中性",
        theme_stage_inferred="acceleration", quant_anticrowding=25)  # 拥挤(anticrowding≤30)→crowding -1
    # 非范式：earnings+1 / theme+1 / crowding-1 / tech+1(动量70未超买) → 净≈+2? 用拥挤压一下看 baseline
    non_para = compute_valuation_regime(**base)
    para = compute_valuation_regime(**base, is_paradigm=True)
    # 范式反转：crowding -1 抬 0 + 门槛降 → ride；且不低于非范式
    assert para["valuation_regime"] == "ride", para
    assert para["legs"]["crowding"] == 0      # 拥挤腿被抬 0


def test_paradigm_blowoff_guard_no_ride():
    """范式股但 blowoff（peak/派发/破位）→ 反转失效，不骑顶（天孚式）。"""
    # 天孚式：破位(动量弱) + peak + 派发
    r = compute_valuation_regime(
        momentum_score=30, rsi_percentile_1y=30, has_peak_signal=True,
        capital_flow_regime="中性", main_force_streak_days=-1, lhb_inst_direction=0,
        net_profit_growth=0.5, retail_concentration_signal="中性",
        theme_stage_inferred="peak", quant_anticrowding=50, is_paradigm=True)
    assert r["valuation_regime"] != "ride", r   # 破位+peak → 护栏生效
    # 散户高接盘也触发护栏
    r2 = compute_valuation_regime(
        momentum_score=70, rsi_percentile_1y=80, has_peak_signal=False,
        capital_flow_regime="中性", main_force_streak_days=1,
        net_profit_growth=0.5, retail_concentration_signal="散户高接盘",
        theme_stage_inferred="acceleration", quant_anticrowding=25, is_paradigm=True)
    # 散户高接盘 → blowoff → 不降门槛；crowding 因散户高接盘本就 -1
    assert r2["legs"]["crowding"] == -1, r2


def test_paradigm_stale_soft_distribution_not_blocks_ride():
    """中际旭创实测回归：陈旧减持新闻(软派发 distribution_detected)不该否决范式 ride——
    硬数据(retail=中性/户数减少吸筹/大宗无折价)说无派发时，5个月前的减持新闻不是 blowoff 证据。"""
    r = compute_valuation_regime(
        momentum_score=72, rsi_percentile_1y=75, has_peak_signal=False,
        main_force_streak_days=2, net_profit_growth=0.9, growth_direction="accelerating",
        retail_concentration_signal="中性",          # 硬：非散户高接盘
        theme_stage_inferred="acceleration", quant_anticrowding=25,  # 拥挤
        distribution_detected=True,                    # 软：陈旧减持新闻
        is_paradigm=True)
    assert r["valuation_regime"] == "ride", r          # 软派发不否决
    assert r["legs"]["crowding"] == 0                   # 拥挤腿抬 0


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
    assert (s["regime"] == "恶化" and s["streak"] == -6 and s["lhb_inst_dir"] == -1
            and s["retail_signal"] == "散户高接盘"), s
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


def test_ride_threshold_symmetric_plus2():
    """对称阈值：净 +2 → ride（无方向先验）。"""
    r = compute_valuation_regime(
        momentum_score=80, net_profit_growth=0.5, growth_direction="accelerating",
        capital_flow_regime="中性", theme_stage_inferred="none")  # tech+1, earnings+1 = +2
    assert r["score"] == 2 and r["valuation_regime"] == "ride", r


def test_capital_leg_uses_score():
    """资金面用连续 score：68→+1（即使标签是中性），26→-1。"""
    up = compute_valuation_regime(momentum_score=55, capital_flow_regime="中性",
        capital_flow_score=68.0, net_profit_growth=0.2, theme_stage_inferred="none")
    assert up["legs"]["capital"] == 1, up
    dn = compute_valuation_regime(momentum_score=55, capital_flow_regime="中性",
        capital_flow_score=26.0, net_profit_growth=0.2, theme_stage_inferred="none")
    assert dn["legs"]["capital"] == -1, dn


def test_earnings_high_stable_positive():
    """高位稳定增长(45%, stable)→ +1（不只 accelerating）。"""
    r = compute_valuation_regime(momentum_score=55, net_profit_growth=0.4579,
        growth_direction="stable", capital_flow_regime="中性", theme_stage_inferred="none")
    assert r["legs"]["earnings"] == 1, r


def test_distribution_gated_by_inflow():
    """天孚式：舆情有旧减仓 + 当下主力强流入(score68) → 派发腿不投（已被吸收）。"""
    r = compute_valuation_regime(
        momentum_score=80, rsi_percentile_1y=82, capital_flow_regime="中性",
        capital_flow_score=68.0, main_force_streak_days=4, net_profit_growth=0.4579,
        growth_direction="stable", theme_stage_inferred="none_or_acceleration",
        quant_anticrowding=37, distribution_detected=True)
    assert "distribution" not in r["legs"], r          # 被强流入 gate 掉
    assert r["valuation_regime"] == "ride", r           # 真趋势票不被误杀
    # 对照：澜起式无强流入 → 派发腿正常投负
    r2 = compute_valuation_regime(
        momentum_score=95, capital_flow_score=26.4, main_force_streak_days=-4,
        growth_direction="decelerating", net_profit_growth=0.588,
        theme_stage_inferred="none", distribution_detected=True)
    assert r2["legs"]["distribution"] == -1 and r2["valuation_regime"] == "discipline", r2


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


def test_strict_earnings_signal_only_trusts_sys():
    """earnings 腿确定性闸：strict 模式只认 SYS_GROWTH_YOY，散文一律 None。

    澜起 SELL↔HOLD 摆动根源——同股不同跑，散文增速读出 减速 vs 加速 → earnings 腿翻
    → regime 在 discipline/neutral 间翻 → 评级翻。strict 让 SYS 缺失时 earnings 腿落 0，
    不再被散文带飘。
    """
    sys_line = ("【SYS_GROWTH_YOY｜tushare】 营收YoY 单季=+19.51% 年度=+49.94% | "
                "归母净利YoY 年度=+58.84%")
    prose_decel = "| 营收同比增速 | +4.58% | +49.94% | Q1放缓 |\n归母净利润增速 | +58.84%"

    # 减速方向：SYS 在 → 确定判 decelerating；散文-only strict → None（不猜）
    assert parse_growth_deceleration(sys_line, strict=True) == "decelerating"
    assert parse_growth_deceleration(prose_decel, strict=True) is None
    assert parse_growth_deceleration(prose_decel, strict=False) == "decelerating"  # 非strict向后兼容

    # 净利增速：SYS 在 → 0.5884；散文-only strict → None
    assert parse_net_profit_growth(sys_line, strict=True) == 0.5884
    assert parse_net_profit_growth(prose_decel, strict=True) is None
    assert abs(parse_net_profit_growth(prose_decel, strict=False) - 0.5884) < 1e-9

    # 关键：SYS 缺失 + strict 双 None → earnings 腿落 0（除非 recurring_loss），regime 不被散文翻
    assert compute_valuation_regime(
        net_profit_growth=None, growth_direction=None, recurring_loss=False,
    )["legs"].get("earnings", 0) == 0


def test_growth_primary_routing():
    """成长股前瞻路由：high_beta_growth+正增速+非discipline → peg；其余不介入。"""
    off = {"force_valuation": False}
    # 中际旭创/天孚式 → peg
    assert recommend_growth_primary_method("high_beta_growth", 1.0, off, "ride")["recommend"] == "peg"
    assert recommend_growth_primary_method("high_beta_growth", 0.4579, off, "ride")["recommend"] == "peg"
    assert recommend_growth_primary_method("high_beta_growth", 0.30, off, "neutral")["recommend"] == "peg"
    # 负增速/缺失/低增速 → 不介入（防 PEG 失真）
    assert recommend_growth_primary_method("high_beta_growth", -0.05, off, "ride")["recommend"] is None
    assert recommend_growth_primary_method("high_beta_growth", None, off, "ride")["recommend"] is None
    assert recommend_growth_primary_method("high_beta_growth", 0.10, off, "ride")["recommend"] is None
    # discipline（基本面恶化）→ 不前瞻主导
    assert recommend_growth_primary_method("high_beta_growth", 0.6, off, "discipline")["recommend"] is None
    # 非成长 style → 不介入
    assert recommend_growth_primary_method("blue_chip", 0.3, off, "ride")["recommend"] is None
    # forced_valuation（亏损/银行）→ 不介入
    assert recommend_growth_primary_method("high_beta_growth", 0.3, {"force_valuation": True}, "ride")["recommend"] is None


def test_growth_quality_gate():
    """成长质量闸：扣非亏损 / 基数效应增速 → 不走前瞻 PEG（防淳中式价值陷阱）。"""
    off = {"force_valuation": False}
    # 扣非亏损 → 即使归母增速 226% 也不走 peg
    r = recommend_growth_primary_method("high_beta_growth", 2.26, off, "neutral", recurring_loss=True)
    assert r["recommend"] is None and "扣非" in r["reason"], r
    # 基数效应：归母 +80% 但扣非仅 +5% → 不走 peg
    r2 = recommend_growth_primary_method("high_beta_growth", 0.80, off, "neutral",
                                         recurring_loss=False, deducted_yoy=0.05)
    assert r2["recommend"] is None and "基数效应" in r2["reason"], r2
    # 健康成长：扣非不亏 + 扣非增速也高 → 正常走 peg
    r3 = recommend_growth_primary_method("high_beta_growth", 0.50, off, "neutral",
                                         recurring_loss=False, deducted_yoy=0.48)
    assert r3["recommend"] == "peg", r3


def test_recurring_loss_kills_earnings_leg():
    """扣非亏损 → regime 盈利腿投 -1（不被归母假高增抬成 +1）。"""
    r = compute_valuation_regime(momentum_score=55, net_profit_growth=2.26,
        capital_flow_regime="中性", theme_stage_inferred="none", recurring_loss=True)
    assert r["legs"]["earnings"] == -1, r


def test_growth_quality_uses_latest_period_dedt():
    """_format_growth_indicators：年报扣非为正但最新一期扣非转亏 → recurring_loss=yes（捕捉近期恶化）。"""
    import pandas as pd
    from tradingagents.dataflows.tushare_vendor import _format_growth_indicators
    # 淳中式：FY2024 扣非 +2.8亿 / Q1 2026 最新一期扣非 -0.35亿
    df = pd.DataFrame([
        {"end_date": "20241231", "q_sales_yoy": 10, "q_netprofit_yoy": 195, "or_yoy": 20,
         "netprofit_yoy": 235, "dt_netprofit_yoy": 195, "profit_dedt": 2.8e8},
        {"end_date": "20260331", "q_sales_yoy": 14, "q_netprofit_yoy": -50, "or_yoy": 14,
         "netprofit_yoy": -50, "dt_netprofit_yoy": -120, "profit_dedt": -0.35e8},
    ])
    assert "recurring_loss=yes" in _format_growth_indicators(df)
    # 健康公司：年报+最新均正 → no
    df2 = pd.DataFrame([
        {"end_date": "20241231", "q_sales_yoy": 40, "q_netprofit_yoy": 50, "or_yoy": 40,
         "netprofit_yoy": 50, "dt_netprofit_yoy": 48, "profit_dedt": 20e8},
        {"end_date": "20260331", "q_sales_yoy": 45, "q_netprofit_yoy": 55, "or_yoy": 45,
         "netprofit_yoy": 55, "dt_netprofit_yoy": 52, "profit_dedt": 6e8},
    ])
    assert "recurring_loss=no" in _format_growth_indicators(df2)


def test_parse_growth_quality():
    sysline = ("【SYS_GROWTH_QUALITY｜扣非口径成长质量，下游前瞻路由/盈利腿直读】 "
               "扣非净利=-0.35亿(负) | recurring_loss=yes | 扣非净利YoY年度=-12.00%\n")
    q = parse_growth_quality(sysline)
    assert q["recurring_loss"] is True and abs(q["deducted_yoy"] + 0.12) < 1e-9, q
    ok = parse_growth_quality("【SYS_GROWTH_QUALITY】 recurring_loss=no | 扣非净利YoY年度=48.00%")
    assert ok["recurring_loss"] is False and abs(ok["deducted_yoy"] - 0.48) < 1e-9, ok
    # 散文兜底
    assert parse_growth_quality("公司扣非净利润亏损3528万元")["recurring_loss"] is True
    assert parse_growth_quality("无相关")["recurring_loss"] is None


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
