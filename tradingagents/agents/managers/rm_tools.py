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
    """PE × EPS 估值法：给定 EPS 和目标 PE 区间，输出目标价区间。

    Args:
        eps: 预期 EPS（元/股），通常用 2026E EPS 中位数
        target_pe_low: 目标 PE 区间下沿
        target_pe_high: 目标 PE 区间上沿

    Returns:
        dict: {"low": 区间下沿目标价, "mid": 区间中位数, "high": 区间上沿,
               "method": "PE × EPS"}
    """
    low = round(eps * target_pe_low, 2)
    high = round(eps * target_pe_high, 2)
    mid = round(eps * (target_pe_low + target_pe_high) / 2, 2)
    return {
        "low": low,
        "mid": mid,
        "high": high,
        "method": "PE × EPS",
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
) -> dict:
    """Step 6 评级机械映射工具。

    把"动态阈值 → 评级"这个机械计算从 LLM 端移到 Python 端，杜绝 LLM 通过
    "重新调整估值参数" 私改 target_price_mid 从而规避机械映射的漏洞。

    强制约束：
    - target_price_mid **必须**等于 Step 4 中 compute_weighted_target_price 或
      compute_overlap_target_price 工具输出的中位数。若 LLM 想换 target_price_mid，
      必须重新调用 Step 4 工具（这会暴露在工具调用日志里）
    - target_price_source 字段填写产生该中位数的工具调用 ID 或一句话来源（如
      "compute_weighted_target_price 工具结果"），便于审计

    评级映射规则（与 Step 6 第一步动态阈值匹配）：
      偏离度 < -threshold_up   → BUY（深度低估）
      [-threshold_up, -threshold_dn]  → OVERWEIGHT
      [-threshold_dn, +threshold_dn]  → HOLD
      (+threshold_dn, +threshold_up]  → UNDERWEIGHT
      偏离度 > +threshold_up   → SELL（明显高估）

    Args:
        current_price: 当前价 P_0
        target_price_mid: Step 4 综合目标价中位（必须来自 Step 4 工具输出）
        threshold_dn_pct: 动态阈值下沿百分比（如 27 表示 27%）
        threshold_up_pct: 动态阈值上沿百分比（如 63 表示 63%）
        target_price_source: 目标价来源说明（审计字段）

    Returns:
        dict: {"deviation_pct", "rating", "target_price_mid",
               "target_price_source", "explanation"}
    """
    if target_price_mid <= 0:
        return {"error": f"target_price_mid 必须 > 0，当前={target_price_mid}"}
    if threshold_dn_pct <= 0 or threshold_up_pct <= 0:
        return {"error": "动态阈值必须 > 0"}
    if threshold_dn_pct >= threshold_up_pct:
        return {"error": f"threshold_dn_pct({threshold_dn_pct}) 必须小于 threshold_up_pct({threshold_up_pct})"}

    deviation_pct = (current_price - target_price_mid) / target_price_mid * 100

    if deviation_pct < -threshold_up_pct:
        rating = "BUY"
    elif deviation_pct < -threshold_dn_pct:
        rating = "OVERWEIGHT"
    elif deviation_pct <= threshold_dn_pct:
        rating = "HOLD"
    elif deviation_pct <= threshold_up_pct:
        rating = "UNDERWEIGHT"
    else:
        rating = "SELL"

    explanation = (
        f"偏离度 = (当前价 {current_price} - 目标价中位 {target_price_mid}) / {target_price_mid} = "
        f"{deviation_pct:+.2f}% | 动态阈值 ±{threshold_dn_pct}%/±{threshold_up_pct}% | "
        f"机械映射 → {rating}"
    )

    return {
        "deviation_pct": round(deviation_pct, 2),
        "rating": rating,
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
        "upgrade": lambda c, m: c is not None and m is not None and c >= 80 and m >= 80,
        "downgrade": lambda c, m: c is not None and m is not None and c <= 20 and m <= 30,
        "rationale": "周期股估值锚定强，仅在量化分极端（≥80 或 ≤20）时调整",
    },
    "high_beta_growth": {
        "upgrade": lambda c, m: c is not None and m is not None and c >= 65 and m >= 75,
        "downgrade": lambda c, m: c is not None and m is not None and c <= 35 and m <= 30,
        "rationale": "成长股趋势信号有显著话语权，composite≥65 + momentum≥75 上调",
    },
    "theme_speculation": {
        "upgrade": lambda c, m: c is not None and m is not None and c >= 55 and m >= 70,
        "downgrade": lambda c, m: c is not None and m is not None and c <= 45 and m <= 40,
        "rationale": "题材股情绪+动量主导，触发阈值最敏感（防止纯估值锁死趋势机会）",
    },
    "illiquid": {
        "upgrade": lambda c, m: c is not None and m is not None and c >= 70 and m >= 70,
        "downgrade": lambda c, m: c is not None and m is not None and c <= 30 and m <= 30,
        "rationale": "流动性差谨慎调整，避免被短期信号误导",
    },
    "etf": {
        "upgrade": lambda c, m: m is not None and m >= 70,
        "downgrade": lambda c, m: m is not None and m <= 30,
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

    规则表（同时满足 upgrade / downgrade 条件才触发）：
      blue_chip:         永不调整
      cyclical:          composite≥80 + momentum≥80 → +1；c≤20 + m≤30 → -1
      high_beta_growth:  composite≥65 + momentum≥75 → +1；c≤35 + m≤30 → -1
      theme_speculation: composite≥55 + momentum≥70 → +1；c≤45 + m≤40 → -1
      illiquid:          composite≥70 + momentum≥70 → +1；c≤30 + m≤30 → -1
      etf:               momentum≥70 → +1；momentum≤30 → -1（不看 composite）

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


# ============================================================================
# 工具集合（供 research_manager.py 一次性绑定）
# ============================================================================

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
    compute_step6_style_adjustment,
]


# 工具按名称索引，便于 invoke 链路里查找
RM_TOOLS_BY_NAME = {t.name: t for t in RM_TOOLS}
