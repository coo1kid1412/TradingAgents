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
]


# 工具按名称索引，便于 invoke 链路里查找
RM_TOOLS_BY_NAME = {t.name: t for t in RM_TOOLS}
