"""Research Manager 计算工具集（方案 B：tool calling）

把 RM 之前让 LLM 心算的数值计算（加权平均、目标价区间、PEG、概率加权 E、
Conviction 校准等）封装为 langchain @tool，由 RM 在 8 步 COT 中显式调用。

设计原则：
- 每个工具职责单一，输入/输出 schema 清晰
- 工具内部纯函数，无副作用
- 返回 dict（langchain tool 自动序列化为 JSON），便于 LLM 续写时引用
"""

from typing import Optional

from langchain_core.tools import tool


# ============================================================================
# 多空辩论评分工具
# ============================================================================

@tool
def compute_bull_bear_score(arguments: list[dict]) -> dict:
    """计算 Bull 或 Bear 论据的加权平均得分。

    使用公式：Score = Σ(final_score × weight) / Σ(weight)
    （注意：分母是权重之和，不是论据条数；常见错误是用论据条数当分母）

    Args:
        arguments: 论据列表，每条必须含 final_score (float, 0-10) 和 weight (float) 字段

    Returns:
        dict: {
            "score": 加权平均分（保留 2 位小数）,
            "total_weighted": Σ(score×weight) 求和,
            "weight_sum": Σ weight 权重之和,
            "n_arguments": 论据条数,
            "formula": "Σ(score×weight) / Σweight"
        }
    """
    if not arguments:
        return {"score": 0.0, "total_weighted": 0.0, "weight_sum": 0.0,
                "n_arguments": 0, "formula": "Σ(score×weight) / Σweight",
                "error": "空论据列表"}

    total = 0.0
    w_sum = 0.0
    for a in arguments:
        s = float(a.get("final_score", 0))
        w = float(a.get("weight", 0))
        total += s * w
        w_sum += w

    if w_sum == 0:
        return {"score": 0.0, "total_weighted": round(total, 2), "weight_sum": 0.0,
                "n_arguments": len(arguments), "formula": "Σ(score×weight) / Σweight",
                "error": "权重总和为 0，无法计算"}

    return {
        "score": round(total / w_sum, 2),
        "total_weighted": round(total, 2),
        "weight_sum": round(w_sum, 2),
        "n_arguments": len(arguments),
        "formula": "Σ(score×weight) / Σweight",
    }


@tool
def compute_score_difference(bull_score: float, bear_score: float) -> dict:
    """计算多空得分差 d = Bull Score - Bear Score。

    Args:
        bull_score: Bull 加权平均得分
        bear_score: Bear 加权平均得分

    Returns:
        dict: {"d": 得分差, "abs_d": 绝对值, "direction": 偏多/偏空/均衡}
    """
    d = round(bull_score - bear_score, 2)
    abs_d = abs(d)
    if d > 0.5:
        direction = "偏多"
    elif d < -0.5:
        direction = "偏空"
    else:
        direction = "均衡"
    return {"d": d, "abs_d": round(abs_d, 2), "direction": direction}


# ============================================================================
# 估值工具
# ============================================================================

@tool
def compute_pe_eps_target_price(eps: float, target_pe_low: float,
                                  target_pe_high: float) -> dict:
    """PE × EPS 估值法（**TTM 口径**）：目标 PE(TTM) × EPS_TTM → 目标价区间。

    Args:
        eps: **必须用 EPS_TTM**（元/股；stock_profile 已给值，直接用）。
             ⚠️ 不得用前瞻/2026E EPS——因为 target_pe 来自 stock_profile（锚自同业 TTM 中位 /
             PE_TTM×0.x）是 **TTM 倍数**；TTM 倍数 × 前瞻 EPS 会**双重计入成长**、目标价虚高 ~50%
             （高估值股被错抬成强买，澜起历史 bug 根因）。前瞻增长由 compute_peg_target_price 单独体现。
        target_pe_low: 目标 PE 区间下沿（TTM 口径）
        target_pe_high: 目标 PE 区间上沿（TTM 口径）

    Returns:
        dict: {"low", "mid", "high", "method", "eps_basis": "TTM"}
    """
    low = round(eps * target_pe_low, 2)
    high = round(eps * target_pe_high, 2)
    mid = round(eps * (target_pe_low + target_pe_high) / 2, 2)
    return {
        "low": low,
        "mid": mid,
        "high": high,
        "method": "PE × EPS",
        "eps_basis": "TTM",  # 口径标记：本法用 EPS_TTM，前瞻增长归 PEG 法
        "inputs": {"eps": eps, "pe_range": [target_pe_low, target_pe_high]},
    }


@tool
def compute_peg_target_price(eps: float, growth_rate_pct: float,
                              target_peg_low: float = 1.0,
                              target_peg_high: float = 1.5) -> dict:
    """PEG 估值法：目标价 = EPS × (PEG × 增速)，其中增速以百分比数值表达（如 50 表示 50%）。

    Args:
        eps: 预期 EPS（元/股）
        growth_rate_pct: 净利润预期增速百分比（如 50 表示 50%）
        target_peg_low: 目标 PEG 下限（默认 1.0，合理估值）
        target_peg_high: 目标 PEG 上限（默认 1.5，略高于合理）

    Returns:
        dict: {"low": ..., "high": ..., "implied_pe_range": [low_pe, high_pe]}
    """
    implied_pe_low = target_peg_low * growth_rate_pct
    implied_pe_high = target_peg_high * growth_rate_pct
    low = round(eps * implied_pe_low, 2)
    high = round(eps * implied_pe_high, 2)
    mid = round((low + high) / 2, 2)
    return {
        "low": low,
        "mid": mid,
        "high": high,
        "method": "PEG",
        "implied_pe_range": [round(implied_pe_low, 1), round(implied_pe_high, 1)],
        "inputs": {"eps": eps, "growth_pct": growth_rate_pct,
                   "peg_range": [target_peg_low, target_peg_high]},
    }


@tool
def compute_overlap_target_price(methods: list[dict]) -> dict:
    """多种估值方法的"真实重叠区间"——三个方法都覆盖到的价位区间。

    严格重叠 = max(所有方法的下沿), min(所有方法的上沿)。
    如果 max(low) > min(high) 则不存在重叠区，返回 is_valid=False。
    此时建议改用 compute_weighted_target_price。

    Args:
        methods: 估值方法列表，每条含 name / low / high 字段。
                例：[{"name": "PEG", "low": 160, "high": 200}, ...]

    Returns:
        dict: {"overlap_low": ..., "overlap_high": ...,
               "is_valid": 是否存在真实重叠, "n_methods": 方法数}
    """
    if not methods:
        return {"overlap_low": None, "overlap_high": None,
                "is_valid": False, "n_methods": 0, "error": "无估值方法"}

    overlap_low = max(float(m["low"]) for m in methods)
    overlap_high = min(float(m["high"]) for m in methods)

    return {
        "overlap_low": round(overlap_low, 2),
        "overlap_high": round(overlap_high, 2),
        "is_valid": overlap_low <= overlap_high,
        "n_methods": len(methods),
        "method_names": [m.get("name", f"method_{i+1}") for i, m in enumerate(methods)],
    }


@tool
def compute_weighted_target_price(methods: list[dict]) -> dict:
    """多种估值方法的加权平均区间——按各方法权重加权得到综合目标价。

    与 compute_overlap_target_price 不同：本方法不要求严格重叠，
    适合方法间分歧较大、需取加权折中的场景。

    Args:
        methods: 估值方法列表，每条含 name / low / high / weight 字段。
                weight 应该是 0-100 的百分比（如 35 表示 35%）。

    Returns:
        dict: {"weighted_low": ..., "weighted_mid": ..., "weighted_high": ...,
               "weight_sum": ...}
    """
    if not methods:
        return {"weighted_low": None, "weighted_mid": None, "weighted_high": None,
                "weight_sum": 0, "error": "无估值方法"}

    total_low = total_high = w_sum = 0.0
    for m in methods:
        w = float(m.get("weight", 0))
        total_low += float(m["low"]) * w
        total_high += float(m["high"]) * w
        w_sum += w

    if w_sum == 0:
        return {"weighted_low": None, "weighted_mid": None, "weighted_high": None,
                "weight_sum": 0, "error": "权重总和为 0"}

    weighted_low = total_low / w_sum
    weighted_high = total_high / w_sum
    return {
        "weighted_low": round(weighted_low, 2),
        "weighted_mid": round((weighted_low + weighted_high) / 2, 2),
        "weighted_high": round(weighted_high, 2),
        "weight_sum": round(w_sum, 2),
        "n_methods": len(methods),
    }


# ============================================================================
# 情景分析与赔率工具
# ============================================================================

@tool
def compute_scenario_weighted_e(scenarios: list[dict], p_0: float) -> dict:
    """三情景概率加权期望收益 E。

    E = Σ(prob_i × (target_price_i − p_0) / p_0)，其中 prob 以百分比表达。
    会校验概率加总是否为 100%，不等于则返回校正后版本。

    Args:
        scenarios: 情景列表，每条含 name / probability / target_price 字段。
                  probability 是 0-100 百分比（如 25 表示 25%）。
                  例：[{"name": "Bull", "probability": 25, "target_price": 277},
                       {"name": "Base", "probability": 50, "target_price": 220},
                       {"name": "Bear", "probability": 25, "target_price": 145}]
        p_0: 当前价（用于计算每个情景的收益率）

    Returns:
        dict: {"expected_return_pct": 概率加权 E（百分比）,
               "scenario_returns": 每个情景的收益率,
               "prob_sum_check": 概率加总是否 = 100}
    """
    if not scenarios:
        return {"expected_return_pct": 0, "error": "无情景"}
    if p_0 <= 0:
        return {"expected_return_pct": 0, "error": "p_0 必须 > 0"}

    prob_sum = sum(float(s.get("probability", 0)) for s in scenarios)

    scenario_returns = []
    weighted_return = 0.0
    for s in scenarios:
        prob = float(s.get("probability", 0))
        tp = float(s["target_price"])
        ret_pct = (tp - p_0) / p_0 * 100
        scenario_returns.append({
            "name": s.get("name", "?"),
            "probability_pct": prob,
            "target_price": tp,
            "return_pct": round(ret_pct, 2),
        })
        weighted_return += (prob / 100) * ret_pct

    return {
        "expected_return_pct": round(weighted_return, 2),
        "scenario_returns": scenario_returns,
        "prob_sum_check": round(prob_sum, 2),
        "prob_sum_valid": abs(prob_sum - 100) < 0.5,
        "p_0": p_0,
    }


@tool
def compute_odds_and_expected_return(p_0: float, p_up: float, p_dn: float,
                                       win_prob: float = 0.5) -> dict:
    """计算赔率 R 和单一胜率的期望收益 E（已被 compute_scenario_weighted_e 优化，
    仅作为简化兜底）。

    Args:
        p_0: 当前价
        p_up: 上行目标价
        p_dn: 下行风险价
        win_prob: 主观胜率（0-1）

    Returns:
        dict: {"R": 赔率, "U_pct": 上行幅度, "D_pct": 下行幅度,
               "E_pct": 期望收益（百分比）}
    """
    if p_0 <= 0:
        return {"R": None, "error": "p_0 必须 > 0"}

    u_pct = (p_up - p_0) / p_0 * 100
    d_pct = (p_0 - p_dn) / p_0 * 100

    r = None
    if d_pct > 0:
        r = round(u_pct / d_pct, 2)
    elif u_pct > 0:
        r = float("inf")

    e_pct = win_prob * u_pct - (1 - win_prob) * d_pct
    return {
        "R": r,
        "U_pct": round(u_pct, 2),
        "D_pct": round(d_pct, 2),
        "win_prob": win_prob,
        "E_pct": round(e_pct, 2),
    }


# ============================================================================
# Conviction 校准工具
# ============================================================================

@tool
def compute_conviction_calibration(abs_d: float,
                                     bull_anchor_refuted: bool = False,
                                     bear_anchor_refuted: bool = False,
                                     rating: str = "") -> dict:
    """Conviction 校准规则。

    基础规则：
    - d > 1.5 → Conviction +1 档
    - d < -1.5 → Conviction -1 档
    - 其他 → 不变

    额外约束：
    - 若多头侧 anchor 被有效反驳 且原评级偏多（BUY/OVERWEIGHT）→ -1
    - 若空头侧 anchor 被有效反驳 且原评级偏空（SELL/UNDERWEIGHT）→ -1

    Args:
        abs_d: |d| 绝对值
        bull_anchor_refuted: 多头 anchor 是否被反驳
        bear_anchor_refuted: 空头 anchor 是否被反驳
        rating: 当前评级（BUY / OVERWEIGHT / HOLD / UNDERWEIGHT / SELL）

    Returns:
        dict: {"adjustment": 校准方向（+1 / 0 / -1）, "reason": 触发理由}
    """
    adjustment = 0
    reasons = []

    if abs_d > 1.5:
        adjustment += 1
        reasons.append(f"|d|={abs_d:.2f} > 1.5 → +1 档")

    rating_upper = (rating or "").upper()
    is_bull_rating = rating_upper in ("BUY", "OVERWEIGHT")
    is_bear_rating = rating_upper in ("SELL", "UNDERWEIGHT")

    if bull_anchor_refuted and is_bull_rating:
        adjustment -= 1
        reasons.append("多头 anchor 被反驳且评级偏多 → -1 档")
    if bear_anchor_refuted and is_bear_rating:
        adjustment -= 1
        reasons.append("空头 anchor 被反驳且评级偏空 → -1 档")

    return {
        "adjustment": adjustment,
        "reason": "; ".join(reasons) if reasons else "无校准",
        "inputs": {"abs_d": abs_d, "bull_anchor_refuted": bull_anchor_refuted,
                   "bear_anchor_refuted": bear_anchor_refuted, "rating": rating},
    }


@tool
def compute_step6_rating_mapping(
    current_price: float,
    target_price_mid: float,
    threshold_dn_pct: float,
    threshold_up_pct: float,
    target_price_source: str = "",
    valuation_regime: str = "",
) -> dict:
    """Step 6 评级机械映射工具（估值定倾向 + regime 闸门控极端）。

    把"动态阈值 → 评级"这个机械计算从 LLM 端移到 Python 端，杜绝 LLM 通过
    "重新调整估值参数" 私改 target_price_mid 从而规避机械映射的漏洞。

    强制约束：
    - target_price_mid **必须**等于 Step 4 中 compute_weighted_target_price 或
      compute_overlap_target_price 工具输出的中位数。若 LLM 想换 target_price_mid，
      必须重新调用 Step 4 工具（这会暴露在工具调用日志里）
    - target_price_source 字段填写产生该中位数的工具调用 ID 或一句话来源（如
      "compute_weighted_target_price 工具结果"），便于审计

    ── 两段式（对标头部投研台做法）──────────────────────────────
    第一段「估值定倾向」（按偏离度的原始 5 档，与第一步动态阈值匹配）：
      偏离度 < -threshold_up   → BUY（深度低估）
      [-threshold_up, -threshold_dn]  → OVERWEIGHT
      [-threshold_dn, +threshold_dn]  → HOLD
      (+threshold_dn, +threshold_up]  → UNDERWEIGHT
      偏离度 > +threshold_up   → SELL（明显高估）

    第二段「regime 闸门控极端」（valuation_regime 来自 stock_profile 五路合成）：
      真台子不会仅凭"贵"就 Sell 优质成长股，也不会仅凭"便宜"就 Buy 基本面恶化股。
      估值偏离只决定倾向，要不要升级到 BUY/SELL 极端，由基本面动能(regime)把关：
      - ride（基本面强：流入/增速/趋势）→ 评级托底 HOLD：UNDERWEIGHT/SELL → HOLD
        （强趋势票贵了也只是 Hold/骑，不因估值看空——防误杀；深度低估时 BUY 保留）
      - discipline（基本面弱：减速/派发/流出）→ 评级封顶 HOLD：OVERWEIGHT/BUY → HOLD
        （恶化票即便optically便宜也是价值陷阱，不追多；深度高估时 SELL 保留=真 Sell 场景）
      - neutral（混合）→ 估值单独不触发极端：SELL→UNDERWEIGHT, BUY→OVERWEIGHT
        （收敛到 OW/HOLD/UW 三档，极端留给有基本面背书的场景）
      - 空串/未知 → 不做闸门，保持原始 5 档（向后兼容）
    注：本闸门只设"regime 约束的基线"，看多侧的 +1「骑」升档仍由 Step 6 第六步
        compute_step6_style_adjustment 负责（discipline 时其 composite/momentum 本就低，
        不会误升；ride 时托底 HOLD 后正好交给它升 OW，无重复计分）。

    Args:
        current_price: 当前价 P_0
        target_price_mid: Step 4 综合目标价中位（必须来自 Step 4 工具输出）
        threshold_dn_pct: 动态阈值下沿百分比（如 27 表示 27%）
        threshold_up_pct: 动态阈值上沿百分比（如 63 表示 63%）
        target_price_source: 目标价来源说明（审计字段）
        valuation_regime: ride / neutral / discipline（来自 stock_profile
            valuation_regime 字段）；留空则不做 regime 闸门

    Returns:
        dict: {"deviation_pct", "rating", "rating_raw", "valuation_regime",
               "target_price_mid", "target_price_source", "explanation"}
    """
    if target_price_mid <= 0:
        return {"error": f"target_price_mid 必须 > 0，当前={target_price_mid}"}
    if threshold_dn_pct <= 0 or threshold_up_pct <= 0:
        return {"error": "动态阈值必须 > 0"}
    if threshold_dn_pct >= threshold_up_pct:
        return {"error": f"threshold_dn_pct({threshold_dn_pct}) 必须小于 threshold_up_pct({threshold_up_pct})"}

    deviation_pct = (current_price - target_price_mid) / target_price_mid * 100

    # 第一段：估值定倾向（原始 5 档）
    if deviation_pct < -threshold_up_pct:
        rating_raw = "BUY"
    elif deviation_pct < -threshold_dn_pct:
        rating_raw = "OVERWEIGHT"
    elif deviation_pct <= threshold_dn_pct:
        rating_raw = "HOLD"
    elif deviation_pct <= threshold_up_pct:
        rating_raw = "UNDERWEIGHT"
    else:
        rating_raw = "SELL"

    # 第二段：regime 闸门控极端
    reg = (valuation_regime or "").strip().lower()
    if reg == "ride":          # 强基本面：托底 HOLD（不因贵看空），BUY 保留
        rating = {"SELL": "HOLD", "UNDERWEIGHT": "HOLD"}.get(rating_raw, rating_raw)
    elif reg == "discipline":  # 弱基本面：封顶 HOLD（防价值陷阱），SELL 保留
        rating = {"BUY": "HOLD", "OVERWEIGHT": "HOLD"}.get(rating_raw, rating_raw)
    elif reg == "neutral":     # 混合：估值单独不触发极端
        rating = {"SELL": "UNDERWEIGHT", "BUY": "OVERWEIGHT"}.get(rating_raw, rating_raw)
    else:                      # 未提供 regime → 保持旧行为
        reg = ""
        rating = rating_raw

    gate_note = ""
    if reg and rating != rating_raw:
        _why = {"ride": "ride 强基本面托底 HOLD（不因贵看空）",
                "discipline": "discipline 弱基本面封顶 HOLD（防价值陷阱）",
                "neutral": "neutral 估值单独不触发极端，收敛三档"}[reg]
        gate_note = f" | regime 闸门：{rating_raw} → {rating}（{_why}）"
    elif reg:
        gate_note = f" | regime={reg}，闸门无调整（{rating_raw} 在 regime 允许区内）"

    explanation = (
        f"偏离度 = (当前价 {current_price} - 目标价中位 {target_price_mid}) / {target_price_mid} = "
        f"{deviation_pct:+.2f}% | 动态阈值 ±{threshold_dn_pct}%/±{threshold_up_pct}% | "
        f"估值倾向 → {rating_raw}{gate_note} | 最终 → {rating}"
    )

    return {
        "deviation_pct": round(deviation_pct, 2),
        "rating": rating,
        "rating_raw": rating_raw,
        "valuation_regime": reg or "（未提供，未做闸门）",
        "target_price_mid": target_price_mid,
        "target_price_source": target_price_source or "（未填写来源，需补充）",
        "explanation": explanation,
    }


@tool
def compute_scenario_consistency_check(
    step4_target_low: float,
    step4_target_high: float,
    bull_target: float,
    base_target: float,
    bear_target: float,
) -> dict:
    """Step 5 三情景 vs Step 4 综合目标价区间 一致性检查。

    Step 4 综合目标价区间是"加权折中估值"，Base case 应在此区间内
    （Base 代表"维持估值方法的中性路径"）。Bull/Bear case 可以超出区间，
    但偏离过大需要充分论证。

    本工具检查：
    - Base 是否在 Step 4 区间内（强约束）
    - Bull 是否超过 Step 4 上限 +50%（提示警告）
    - Bear 是否低于 Step 4 下限 -40%（提示警告）

    LLM 必须显式查看返回的 warnings，并在 Step 5 末尾对每条 warning 给出回应。

    Args:
        step4_target_low: Step 4 综合目标价区间下沿
        step4_target_high: Step 4 综合目标价区间上沿
        bull_target: Bull case 目标价
        base_target: Base case 目标价
        bear_target: Bear case 目标价

    Returns:
        dict: {"ok": bool, "warnings": [...], "details": {...}}
    """
    if step4_target_low >= step4_target_high:
        return {"error": f"step4_target_low({step4_target_low}) 必须 < step4_target_high({step4_target_high})"}

    warnings: list[str] = []
    details: dict = {
        "step4_range": [step4_target_low, step4_target_high],
        "step4_mid": (step4_target_low + step4_target_high) / 2,
    }

    # Base case 必须 within Step 4 范围（强约束）
    if base_target < step4_target_low:
        warnings.append(
            f"❌ Base case {base_target} 低于 Step 4 区间下沿 {step4_target_low}——"
            f"Base 应反映'维持估值方法的中性路径'，必须在 [{step4_target_low}, {step4_target_high}] 内。"
            "请回头修正 Base case 或修改 Step 4 估值方法。"
        )
    elif base_target > step4_target_high:
        warnings.append(
            f"❌ Base case {base_target} 高于 Step 4 区间上沿 {step4_target_high}——"
            f"Base 应反映'维持估值方法的中性路径'，必须在 [{step4_target_low}, {step4_target_high}] 内。"
            "请回头修正 Base case 或修改 Step 4 估值方法。"
        )

    # Bull case 上限保护（弱约束）
    bull_threshold = step4_target_high * 1.5
    if bull_target > bull_threshold:
        warnings.append(
            f"⚠ Bull case {bull_target} 超出 Step 4 上限 {step4_target_high} 的 50%"
            f"（阈值 {bull_threshold:.1f}）——必须在 Step 5 末尾给出充分理由："
            "Step 1 行业景气度是否支持估值整体上移？哪个具体催化兑现？"
        )

    # Bear case 下限保护（弱约束）
    bear_threshold = step4_target_low * 0.6
    if bear_target < bear_threshold:
        warnings.append(
            f"⚠ Bear case {bear_target} 低于 Step 4 下沿 {step4_target_low} 的 40%"
            f"（阈值 {bear_threshold:.1f}）——必须在 Step 5 末尾给出充分理由："
            "哪个 anchor 失效会触发深度估值压缩？时间窗口？"
        )

    details.update({
        "bull_target": bull_target,
        "base_target": base_target,
        "bear_target": bear_target,
        "bull_threshold_upper": bull_threshold,
        "bear_threshold_lower": bear_threshold,
    })

    return {
        "ok": len(warnings) == 0,
        "warnings": warnings,
        "details": details,
    }


_RATINGS_ORDER = ["SELL", "UNDERWEIGHT", "HOLD", "OVERWEIGHT", "BUY"]


def _shift_rating(rating: str, delta: int, no_cross_hold: bool = True) -> tuple[str, str]:
    """按 delta（+1 上调 / -1 下调）调整评级，返回 (new_rating, cap_reason)。

    no_cross_hold=True 时禁止单次跨过 HOLD：
      UNDERWEIGHT → 最多上调到 HOLD（不能直接到 OVERWEIGHT）
      HOLD → 上调到 OVERWEIGHT 允许
      OVERWEIGHT → 上调到 BUY 允许
      反之亦然
    """
    if rating not in _RATINGS_ORDER:
        return rating, f"unknown rating: {rating}"
    idx = _RATINGS_ORDER.index(rating)
    new_idx = max(0, min(len(_RATINGS_ORDER) - 1, idx + delta))

    if no_cross_hold:
        hold_idx = _RATINGS_ORDER.index("HOLD")
        # 上调：从 SELL/UNDERWEIGHT 上调最多到 HOLD
        if delta > 0 and idx < hold_idx and new_idx > hold_idx:
            new_idx = hold_idx
        # 下调：从 BUY/OVERWEIGHT 下调最多到 HOLD
        if delta < 0 and idx > hold_idx and new_idx < hold_idx:
            new_idx = hold_idx

    if new_idx == idx:
        return rating, "已到边界，无调整空间"
    return _RATINGS_ORDER[new_idx], ""


# Style-conditional 调整规则表
# 设计哲学：不同 style 对量化趋势信号的敏感度不同
#   blue_chip   永不调整（估值绝对主导）
#   cyclical    极端时才调（周期股估值锚定强）
#   high_beta_growth  中度敏感（成长股看趋势）
#   theme_speculation 最敏感（题材股情绪/动量主导）
#   illiquid    谨慎调整
#   etf         看动量（技术面主导）
_STYLE_RULES = {
    "blue_chip": {
        "upgrade": lambda c, m: False,
        "downgrade": lambda c, m: False,
        "rationale": "blue_chip 估值绝对主导，趋势信号不参与评级调整",
    },
    "cyclical": {
        "upgrade": lambda c, m: c is not None and m is not None and c >= 75 and m >= 75,
        "downgrade": lambda c, m: c is not None and m is not None and c <= 25 and m <= 30,
        "rationale": "周期股估值锚定较强，composite≥75 + momentum≥75 才调整",
    },
    "high_beta_growth": {
        "upgrade": lambda c, m: c is not None and m is not None and c >= 60 and m >= 70,
        "downgrade": lambda c, m: c is not None and m is not None and c <= 40 and m <= 35,
        "rationale": "成长股趋势信号有显著话语权，composite≥60 + momentum≥70 上调",
    },
    "theme_speculation": {
        "upgrade": lambda c, m: c is not None and m is not None and c >= 50 and m >= 65,
        "downgrade": lambda c, m: c is not None and m is not None and c <= 50 and m <= 45,
        "rationale": "题材股情绪+动量主导，触发阈值最敏感（防止纯估值锁死趋势机会）",
    },
    "illiquid": {
        "upgrade": lambda c, m: c is not None and m is not None and c >= 65 and m >= 65,
        "downgrade": lambda c, m: c is not None and m is not None and c <= 35 and m <= 35,
        "rationale": "流动性差谨慎调整，避免被短期信号误导",
    },
    "etf": {
        "upgrade": lambda c, m: m is not None and m >= 65,
        "downgrade": lambda c, m: m is not None and m <= 35,
        "rationale": "ETF 技术面主导，仅看 momentum（composite 对 ETF 意义有限）",
    },
}


@tool
def compute_step6_style_adjustment(
    rating_after_mechanical: str,
    style: str,
    composite_score: float | None = None,
    momentum_score: float | None = None,
) -> dict:
    """Step 6 Style-Conditional 趋势叠加调整。

    在机械映射（compute_step6_rating_mapping）+ 拥挤度调整 + 对称升降档之后，
    根据 stock_profile.style 用量化趋势信号（composite + momentum）做最后 ±1 档调整。

    设计动机：纯估值主导评级在题材股 / 高 beta 成长股加速期容易"过早 SELL"，
    错过趋势机会。头部投研团队对不同股性用不同框架——blue_chip 估值主导，
    theme_speculation 情绪/动量主导。本工具按 style 差异化用规则化方式注入这种判断，
    避免 LLM 主观介入。

    规则表（同时满足 upgrade / downgrade 条件才触发，已按改造 A 降低阈值）：
      blue_chip:         永不调整
      cyclical:          composite≥75 + momentum≥75 → +1；c≤25 + m≤30 → -1
      high_beta_growth:  composite≥60 + momentum≥70 → +1；c≤40 + m≤35 → -1
      theme_speculation: composite≥50 + momentum≥65 → +1；c≤50 + m≤45 → -1
      illiquid:          composite≥65 + momentum≥65 → +1；c≤35 + m≤35 → -1
      etf:               momentum≥65 → +1；momentum≤35 → -1（不看 composite）

    保护规则：
      - 单次调整 ≤ ±1 档
      - **禁止跨 HOLD 单次调整**——UNDERWEIGHT 上调最多到 HOLD，不允许直接到 OVERWEIGHT；反之亦然
      - 缺数据（style/composite/momentum 任一缺失）→ 不调整，返回 skipped

    Args:
        rating_after_mechanical: 第五步对称升降档后的评级（BUY/OVERWEIGHT/HOLD/UNDERWEIGHT/SELL）
        style: stock_profile.style
        composite_score: QUANT_SCORE.composite（0-100）
        momentum_score: QUANT_SCORE.factor_scores.momentum（0-100）

    Returns:
        dict: {
            "adjustment": -1 / 0 / +1,
            "new_rating": 调整后评级,
            "rule_applied": upgrade / downgrade / no_change / skipped,
            "trigger": 触发条件描述,
            "rationale": style-specific 设计理由,
            "skip_reason": 仅当 skipped 时有值,
        }
    """
    if rating_after_mechanical not in _RATINGS_ORDER:
        return {
            "adjustment": 0,
            "new_rating": rating_after_mechanical,
            "rule_applied": "error",
            "skip_reason": f"未知评级：{rating_after_mechanical}",
        }

    if not style:
        return {
            "adjustment": 0,
            "new_rating": rating_after_mechanical,
            "rule_applied": "skipped",
            "skip_reason": "style 缺失，不做调整",
        }

    rule = _STYLE_RULES.get(style)
    if rule is None:
        return {
            "adjustment": 0,
            "new_rating": rating_after_mechanical,
            "rule_applied": "skipped",
            "skip_reason": f"未知 style：{style}（合法值：blue_chip/cyclical/high_beta_growth/theme_speculation/illiquid/etf）",
        }

    if rule["upgrade"](composite_score, momentum_score):
        new_rating, cap_reason = _shift_rating(rating_after_mechanical, +1, no_cross_hold=True)
        result_type = "upgrade_capped" if cap_reason else "upgrade"
        return {
            "adjustment": 1 if new_rating != rating_after_mechanical else 0,
            "new_rating": new_rating,
            "rule_applied": result_type,
            "trigger": (
                f"style={style}: composite={composite_score} + momentum={momentum_score} "
                f"满足 upgrade 条件"
            ),
            "rationale": rule["rationale"],
            "cap_reason": cap_reason or None,
        }

    if rule["downgrade"](composite_score, momentum_score):
        new_rating, cap_reason = _shift_rating(rating_after_mechanical, -1, no_cross_hold=True)
        result_type = "downgrade_capped" if cap_reason else "downgrade"
        return {
            "adjustment": -1 if new_rating != rating_after_mechanical else 0,
            "new_rating": new_rating,
            "rule_applied": result_type,
            "trigger": (
                f"style={style}: composite={composite_score} + momentum={momentum_score} "
                f"满足 downgrade 条件"
            ),
            "rationale": rule["rationale"],
            "cap_reason": cap_reason or None,
        }

    return {
        "adjustment": 0,
        "new_rating": rating_after_mechanical,
        "rule_applied": "no_change",
        "trigger": (
            f"style={style}: composite={composite_score} + momentum={momentum_score} "
            f"未触发任何调整条件"
        ),
        "rationale": rule["rationale"],
    }


@tool
def compute_step6_report_weighted_vote_adjustment(
    rating_after_style_adj: str,
    market_weight: float,
    news_weight: float,
    sentiment_weight: float,
    market_direction_vote: float = 0.0,
    news_direction_vote: float = 0.0,
    sentiment_direction_vote: float = 0.0,
) -> dict:
    """非估值方向票加权调整（改造 B）。

    把 stock_profile.REPORT_WEIGHTS 真正接入评级——市场/新闻/情绪 三个非估值维度
    的方向投票（LLM 给定 -1~+1）按权重加权，超过阈值则触发 ±1 档调整。

    设计动机：之前 REPORT_WEIGHTS 只影响 Bull/Bear 写论据来源，不影响评级生成。
    现在题材股情绪权重 30%，能在情绪强烈看多时真正把评级抬一档。

    Args:
        rating_after_style_adj: Step 6 第六步 style 调整后的评级
        market_weight: stock_profile.REPORT_WEIGHTS.market（0-100 整数）
        news_weight: stock_profile.REPORT_WEIGHTS.news（0-100 整数）
        sentiment_weight: stock_profile.REPORT_WEIGHTS.sentiment（0-100 整数）
        market_direction_vote: LLM 读 market 报告后给的方向票（-1 全看空 ~ +1 全看多）
        news_direction_vote: 读 news 报告后给的方向票
        sentiment_direction_vote: 读 sentiment 报告后给的方向票

    Returns:
        dict: {
            "adjustment": -1 / 0 / +1,
            "new_rating": str,
            "weighted_vote": float (-1 ~ +1),
            "trigger_threshold": 0.3,
            "rule_applied": upgrade / downgrade / no_change,
        }
    """
    if rating_after_style_adj not in _RATINGS_ORDER:
        return {"error": f"未知评级: {rating_after_style_adj}"}

    total_weight = market_weight + news_weight + sentiment_weight
    if total_weight <= 0:
        return {
            "adjustment": 0,
            "new_rating": rating_after_style_adj,
            "rule_applied": "skipped",
            "skip_reason": f"非估值权重总和={total_weight} 无效",
        }

    # 每个 vote clamp 到 [-1, +1]
    mv = max(-1.0, min(1.0, market_direction_vote))
    nv = max(-1.0, min(1.0, news_direction_vote))
    sv = max(-1.0, min(1.0, sentiment_direction_vote))

    weighted_vote = (market_weight * mv + news_weight * nv + sentiment_weight * sv) / total_weight
    weighted_vote = round(weighted_vote, 3)

    THRESHOLD = 0.3

    if weighted_vote >= THRESHOLD:
        new_rating, cap_reason = _shift_rating(rating_after_style_adj, +1, no_cross_hold=True)
        return {
            "adjustment": 1 if new_rating != rating_after_style_adj else 0,
            "new_rating": new_rating,
            "weighted_vote": weighted_vote,
            "trigger_threshold": THRESHOLD,
            "rule_applied": "upgrade" if new_rating != rating_after_style_adj else "upgrade_capped",
            "explanation": (
                f"加权方向票 {weighted_vote:+.3f} ≥ +{THRESHOLD} 触发上调 → "
                f"{rating_after_style_adj} → {new_rating}"
                + (f"（{cap_reason}）" if cap_reason else "")
            ),
            "votes": {"market": mv, "news": nv, "sentiment": sv},
            "weights": {"market": market_weight, "news": news_weight, "sentiment": sentiment_weight},
        }

    if weighted_vote <= -THRESHOLD:
        new_rating, cap_reason = _shift_rating(rating_after_style_adj, -1, no_cross_hold=True)
        return {
            "adjustment": -1 if new_rating != rating_after_style_adj else 0,
            "new_rating": new_rating,
            "weighted_vote": weighted_vote,
            "trigger_threshold": THRESHOLD,
            "rule_applied": "downgrade" if new_rating != rating_after_style_adj else "downgrade_capped",
            "explanation": (
                f"加权方向票 {weighted_vote:+.3f} ≤ -{THRESHOLD} 触发下调 → "
                f"{rating_after_style_adj} → {new_rating}"
                + (f"（{cap_reason}）" if cap_reason else "")
            ),
            "votes": {"market": mv, "news": nv, "sentiment": sv},
            "weights": {"market": market_weight, "news": news_weight, "sentiment": sentiment_weight},
        }

    return {
        "adjustment": 0,
        "new_rating": rating_after_style_adj,
        "weighted_vote": weighted_vote,
        "trigger_threshold": THRESHOLD,
        "rule_applied": "no_change",
        "explanation": (
            f"加权方向票 {weighted_vote:+.3f} 落在 [-{THRESHOLD}, +{THRESHOLD}] 区间，"
            "非估值信号方向不够强，不触发调整"
        ),
        "votes": {"market": mv, "news": nv, "sentiment": sv},
        "weights": {"market": market_weight, "news": news_weight, "sentiment": sentiment_weight},
    }


@tool
def compute_step6_catalyst_momentum_adjustment(
    rating_after_vote_adj: str,
    sell_side_target_change_pct: float | None = None,
    institutional_holding_change_pct: float | None = None,
    northbound_flow_5d_direction: int | None = None,
    kol_bullish_ratio_trend_pct: float | None = None,
) -> dict:
    """催化动量硬数据调整（改造 C）。

    把 4 个"硬催化信号"打分聚合为 0-100 分，并根据分数给出 ±1 档调整建议。
    这是 Citadel/Tiger 类机构常用的"催化动量"层级，专门捕捉短期机构动量。

    输入 4 个信号（任一可为 None，缺失则跳过该项；至少需 2 项有效才计分）：
      1. sell_side_target_change_pct: 近 30 日卖方目标价中位变化百分比
                                       （如高盛上调 5%, 中信上调 10% → 取中位 7.5）
      2. institutional_holding_change_pct: 近 1 季机构持仓变化百分比
                                            （增仓 +5% / 减仓 -10%）
      3. northbound_flow_5d_direction: 北向资金近 5 日方向（-1 流出 / 0 中性 / +1 流入）
      4. kol_bullish_ratio_trend_pct: KOL 多头率相对 30 日均的变化（百分点）

    打分细则（各信号独立打分后求和，再 offset 到 0-100）：
      sell_side: >+15% → +30 / +5~+15% → +15 / -5~+5% → 0 / -15~-5% → -15 / <-15% → -30
      inst:      >+10% → +20 / 0~+10% → +10 / -10~0% → -10 / <-10% → -20
      north:     +1 → +15 / 0 → 0 / -1 → -15
      kol:       >+10pp → +15 / -10~+10pp → 0 / <-10pp → -15

      composite_offset_50 = 50 + sum(scores) → clamp [0, 100]

    调整规则：
      composite ≥ 70 → +1（催化动量强）
      composite ≤ 30 → -1（催化动量弱）
      其他 → 0

    Returns:
        dict: 含 composite / breakdown / adjustment / new_rating / coverage 等
    """
    if rating_after_vote_adj not in _RATINGS_ORDER:
        return {"error": f"未知评级: {rating_after_vote_adj}"}

    breakdown: dict = {}
    available_count = 0
    score_sum = 0

    # 1. sell-side
    if sell_side_target_change_pct is not None:
        v = sell_side_target_change_pct
        if v > 15:
            s = 30
        elif v >= 5:
            s = 15
        elif v >= -5:
            s = 0
        elif v >= -15:
            s = -15
        else:
            s = -30
        breakdown["sell_side"] = {"value_pct": v, "subscore": s}
        score_sum += s
        available_count += 1

    # 2. institutional holding
    if institutional_holding_change_pct is not None:
        v = institutional_holding_change_pct
        if v > 10:
            s = 20
        elif v >= 0:
            s = 10
        elif v >= -10:
            s = -10
        else:
            s = -20
        breakdown["institutional"] = {"value_pct": v, "subscore": s}
        score_sum += s
        available_count += 1

    # 3. northbound flow
    if northbound_flow_5d_direction is not None:
        d = int(northbound_flow_5d_direction)
        if d > 0:
            s = 15
        elif d == 0:
            s = 0
        else:
            s = -15
        breakdown["northbound"] = {"direction": d, "subscore": s}
        score_sum += s
        available_count += 1

    # 4. KOL trend
    if kol_bullish_ratio_trend_pct is not None:
        v = kol_bullish_ratio_trend_pct
        if v > 10:
            s = 15
        elif v >= -10:
            s = 0
        else:
            s = -15
        breakdown["kol"] = {"value_pct": v, "subscore": s}
        score_sum += s
        available_count += 1

    # 至少需要 2 项有效
    if available_count < 2:
        return {
            "adjustment": 0,
            "new_rating": rating_after_vote_adj,
            "composite": None,
            "breakdown": breakdown,
            "available_count": available_count,
            "rule_applied": "skipped",
            "skip_reason": f"催化动量数据覆盖不足（{available_count}/4），需 ≥2 项才计分",
        }

    composite = max(0, min(100, 50 + score_sum))

    if composite >= 70:
        new_rating, cap_reason = _shift_rating(rating_after_vote_adj, +1, no_cross_hold=True)
        return {
            "adjustment": 1 if new_rating != rating_after_vote_adj else 0,
            "new_rating": new_rating,
            "composite": composite,
            "breakdown": breakdown,
            "available_count": available_count,
            "rule_applied": "upgrade" if new_rating != rating_after_vote_adj else "upgrade_capped",
            "explanation": (
                f"催化动量 composite={composite} ≥ 70 → 触发 +1 档 → "
                f"{rating_after_vote_adj} → {new_rating}"
                + (f"（{cap_reason}）" if cap_reason else "")
            ),
        }

    if composite <= 30:
        new_rating, cap_reason = _shift_rating(rating_after_vote_adj, -1, no_cross_hold=True)
        return {
            "adjustment": -1 if new_rating != rating_after_vote_adj else 0,
            "new_rating": new_rating,
            "composite": composite,
            "breakdown": breakdown,
            "available_count": available_count,
            "rule_applied": "downgrade" if new_rating != rating_after_vote_adj else "downgrade_capped",
            "explanation": (
                f"催化动量 composite={composite} ≤ 30 → 触发 -1 档 → "
                f"{rating_after_vote_adj} → {new_rating}"
                + (f"（{cap_reason}）" if cap_reason else "")
            ),
        }

    return {
        "adjustment": 0,
        "new_rating": rating_after_vote_adj,
        "composite": composite,
        "breakdown": breakdown,
        "available_count": available_count,
        "rule_applied": "no_change",
        "explanation": f"催化动量 composite={composite} 落在 (30, 70) 中性区间，不触发调整",
    }


@tool
def compute_step6_adjustment_synthesis(
    rating_after_symmetric: str,
    style_adjustment: int,
    vote_adjustment: int,
    catalyst_adjustment: int,
) -> dict:
    """三类 ±1 信号最终合成（合成工具）。

    Step 6 三个"趋势叠加"子步骤（style / vote / catalyst）各自给出 -1/0/+1
    建议，本工具合成为**最终单一调整**——总幅度 capped 至 ±1，应用 no-cross-HOLD。

    合成规则：
      raw_sum = style_adj + vote_adj + catalyst_adj   # range: -3 ~ +3
      if raw_sum > 0:  final_adjustment = +1
      if raw_sum < 0:  final_adjustment = -1
      if raw_sum == 0: final_adjustment = 0

    设计动机：
    - 三类信号独立建议，避免单一信号过度主导
    - 累加方向后取符号，避免 +2/-1=+1 这种小偏差被放大
    - 总幅度 ±1 保护：评级单次最多移动 1 档（杠杆 1+A+2 体系的稳定性约定）

    Args:
        rating_after_symmetric: Step 6 第五步对称升降档后的评级
        style_adjustment: style 工具返回的 adjustment 字段（-1/0/+1）
        vote_adjustment: 非估值方向票工具返回的 adjustment 字段（-1/0/+1）
        catalyst_adjustment: 催化动量工具返回的 adjustment 字段（-1/0/+1）

    Returns:
        dict: {
            "raw_sum": int,
            "final_adjustment": -1 / 0 / +1,
            "new_rating": str,
            "components": {...},
            "explanation": str,
        }
    """
    if rating_after_symmetric not in _RATINGS_ORDER:
        return {"error": f"未知评级: {rating_after_symmetric}"}

    # 确保每个输入都是 -1/0/+1
    def _clamp(v: int) -> int:
        if v > 0:
            return 1
        if v < 0:
            return -1
        return 0

    s = _clamp(int(style_adjustment))
    v = _clamp(int(vote_adjustment))
    c = _clamp(int(catalyst_adjustment))
    raw_sum = s + v + c

    if raw_sum > 0:
        final_adj = 1
    elif raw_sum < 0:
        final_adj = -1
    else:
        final_adj = 0

    new_rating, cap_reason = _shift_rating(rating_after_symmetric, final_adj, no_cross_hold=True)
    actual_adj = 0 if new_rating == rating_after_symmetric else final_adj

    return {
        "raw_sum": raw_sum,
        "final_adjustment": actual_adj,
        "new_rating": new_rating,
        "components": {"style": s, "vote": v, "catalyst": c},
        "cap_reason": cap_reason or None,
        "explanation": (
            f"三类信号: style={s:+d} + vote={v:+d} + catalyst={c:+d} = raw_sum={raw_sum:+d} "
            f"→ 取符号得 final_adjustment={actual_adj:+d} "
            f"→ {rating_after_symmetric} → {new_rating}"
            + (f"（{cap_reason}）" if cap_reason else "")
        ),
    }


@tool
def compute_step6_trend_overlay(
    rating_after_symmetric: str,
    style: str,
    composite_score: float | None = None,
    momentum_score: float | None = None,
    market_weight: float = 0.0,
    news_weight: float = 0.0,
    sentiment_weight: float = 0.0,
    market_direction_vote: float = 0.0,
    news_direction_vote: float = 0.0,
    sentiment_direction_vote: float = 0.0,
    sell_side_target_change_pct: float | None = None,
    institutional_holding_change_pct: float | None = None,
    northbound_flow_5d_direction: int | None = None,
    kol_bullish_ratio_trend_pct: float | None = None,
) -> dict:
    """Step 6 第六步「趋势叠加」一次性合成（style + 方向票 + 催化动量 → 最终评级）。

    **本工具把原本需要 4 次顺序调用的 6.1/6.2/6.3/6.4 合并成 1 次**，内部按
    完全相同的顺序串调三类调整 + 合成，返回值与"分 4 次调用"逐位一致——
    纯粹减少 LLM ↔ 工具往返轮数，不改任何打分/阈值/合成逻辑。

    内部执行顺序（与历史 4 工具链严格一致）：
      6.1 style_adjustment(R0)        → adj_style, R1
      6.2 vote_adjustment(R1)         → adj_vote,  R2
      6.3 catalyst_adjustment(R2)     → adj_catalyst
      6.4 synthesis(R0, adj_style, adj_vote, adj_catalyst) → final_rating
    （注意 6.4 用的是 R0=rating_after_symmetric，不是链式 R3——与原设计一致）

    Args:
        rating_after_symmetric: 第五步对称升降档后的评级（R0）
        style / composite_score / momentum_score: 喂给 6.1
        market_weight / news_weight / sentiment_weight + 三个 *_direction_vote: 喂给 6.2
        sell_side_target_change_pct / institutional_holding_change_pct /
            northbound_flow_5d_direction / kol_bullish_ratio_trend_pct: 喂给 6.3（缺失填 None）

    Returns:
        dict: {
            "final_rating": 最终评级,
            "final_adjustment": -1/0/+1,
            "raw_sum": int,
            "components": {"style": __, "vote": __, "catalyst": __},
            "style_detail" / "vote_detail" / "catalyst_detail" / "synthesis_detail": 各步完整返回（留痕）,
            "explanation": 串联说明,
        }
    """
    r0 = rating_after_symmetric

    style_res = compute_step6_style_adjustment.invoke({
        "rating_after_mechanical": r0,
        "style": style,
        "composite_score": composite_score,
        "momentum_score": momentum_score,
    })
    r1 = style_res.get("new_rating", r0)
    adj_style = style_res.get("adjustment", 0)

    vote_res = compute_step6_report_weighted_vote_adjustment.invoke({
        "rating_after_style_adj": r1,
        "market_weight": market_weight,
        "news_weight": news_weight,
        "sentiment_weight": sentiment_weight,
        "market_direction_vote": market_direction_vote,
        "news_direction_vote": news_direction_vote,
        "sentiment_direction_vote": sentiment_direction_vote,
    })
    r2 = vote_res.get("new_rating", r1)
    adj_vote = vote_res.get("adjustment", 0)

    catalyst_res = compute_step6_catalyst_momentum_adjustment.invoke({
        "rating_after_vote_adj": r2,
        "sell_side_target_change_pct": sell_side_target_change_pct,
        "institutional_holding_change_pct": institutional_holding_change_pct,
        "northbound_flow_5d_direction": northbound_flow_5d_direction,
        "kol_bullish_ratio_trend_pct": kol_bullish_ratio_trend_pct,
    })
    adj_catalyst = catalyst_res.get("adjustment", 0)

    synth = compute_step6_adjustment_synthesis.invoke({
        "rating_after_symmetric": r0,
        "style_adjustment": adj_style,
        "vote_adjustment": adj_vote,
        "catalyst_adjustment": adj_catalyst,
    })

    return {
        "final_rating": synth.get("new_rating", r0),
        "final_adjustment": synth.get("final_adjustment", 0),
        "raw_sum": synth.get("raw_sum", 0),
        "components": {"style": adj_style, "vote": adj_vote, "catalyst": adj_catalyst},
        "style_detail": style_res,
        "vote_detail": vote_res,
        "catalyst_detail": catalyst_res,
        "synthesis_detail": synth,
        "explanation": (
            f"趋势叠加合成：style={adj_style:+d}（{style_res.get('rule_applied','-')}） "
            f"+ vote={adj_vote:+d}（{vote_res.get('rule_applied','-')}） "
            f"+ catalyst={adj_catalyst:+d}（{catalyst_res.get('rule_applied','-')}） "
            f"→ {synth.get('explanation','')}"
        ),
    }


# ============================================================================
# 工具集合（供 research_manager.py 一次性绑定）
# ============================================================================
# 注：Step 6 第六步的 4 个子工具（style/vote/catalyst/synthesis）仍保留定义供合并
# 工具内部复用，但**不再单独绑定给 RM**——RM 只调 compute_step6_trend_overlay 一次
# （4 轮 → 1 轮）。这样工具调用轮数从 ~11-15 压到 ~8-11，远离 15 轮上限。

RM_TOOLS = [
    compute_bull_bear_score,
    compute_score_difference,
    compute_pe_eps_target_price,
    compute_peg_target_price,
    compute_overlap_target_price,
    compute_weighted_target_price,
    compute_scenario_weighted_e,
    compute_odds_and_expected_return,
    compute_conviction_calibration,
    compute_step6_rating_mapping,
    compute_scenario_consistency_check,
    compute_step6_trend_overlay,
]


# 工具按名称索引，便于 invoke 链路里查找
RM_TOOLS_BY_NAME = {t.name: t for t in RM_TOOLS}
