"""Portfolio Manager 计算工具集（方案 B：tool calling）

PM 需要的数值计算工具：
- R-multiple 体系（1R / TP1/2/3 / SL_soft / SL_hard）
- Conviction 五星档 → 仓位映射
- 4 情景概率加权 E（PM 比 RM 多一档"黑天鹅"）
"""

from langchain_core.tools import tool


# ============================================================================
# R-multiple 体系
# ============================================================================

@tool
def compute_r_multiple_levels(entry_price: float, sl_hard_price: float) -> dict:
    """计算完整的 R-multiple 价位体系。

    定义 1R = Entry − SL_hard（每股承担的最大风险）
    TP1 = Entry + 1R（赚 1R 减 1/3 仓位）
    TP2 = Entry + 2R（赚 2R 再减 1/3）
    TP3 = Entry + 3R（赚 3R 清仓）
    SL_soft = Entry − 0.6R（软止损，减半仓位预警）
    SL_hard 由用户输入

    Args:
        entry_price: 建仓价
        sl_hard_price: 硬止损价（必须 < entry_price）

    Returns:
        dict: 含 1R 数值 + TP1-3 + SL_soft + SL_hard
    """
    if entry_price <= 0 or sl_hard_price <= 0:
        return {"error": "价格必须 > 0"}
    if sl_hard_price >= entry_price:
        return {"error": "硬止损价必须 < 建仓价"}

    one_r = entry_price - sl_hard_price
    tp1 = entry_price + one_r
    tp2 = entry_price + 2 * one_r
    tp3 = entry_price + 3 * one_r
    sl_soft = entry_price - 0.6 * one_r

    return {
        "entry_price": round(entry_price, 2),
        "one_r": round(one_r, 2),
        "tp1": round(tp1, 2),
        "tp2": round(tp2, 2),
        "tp3": round(tp3, 2),
        "sl_soft": round(sl_soft, 2),
        "sl_hard": round(sl_hard_price, 2),
        "tp1_action": "减仓 1/3",
        "tp2_action": "再减仓 1/3",
        "tp3_action": "清仓",
        "sl_soft_action": "减半 + 预警",
        "sl_hard_action": "全部清仓",
    }


# ============================================================================
# Conviction → 仓位映射
# ============================================================================

_STAR_POSITION = {
    5: ("Very High", 15, 20),
    4: ("High", 8, 12),
    3: ("Medium", 4, 6),
    2: ("Low", 2, 3),
    1: ("Very Low", 0, 1),
}


@tool
def compute_conviction_position_map(rm_conviction: str, odds_r: float = 1.0,
                                    abs_d: float = 0.0,
                                    anchor_sensitive: bool = False) -> dict:
    """Conviction 五星制 + 仓位上限映射（主输入 = RM Conviction + 赔率 R）。

    为什么不再以 \|d\|（多空辩论比分差）为主：仓位是执行端最重要的参数，而辩论
    比分是全链最软的信号（LLM 给 LLM 打分的 prose 产物），且系统设计本就规定
    "辩论不影响方向只影响置信"——让它主导仓位自相矛盾。对标真实台子：仓位由
    信念强度（证据质量：数据完整度/估值方法收敛度/拐点确认度——RM Conviction
    正是按这些硬条件校准的）+ 赔率 + 风险预算决定，辩论比分顶多是微调。

    映射规则（确定性）：
      基础星 = RM Conviction：高 → 4★ / 中 → 3★ / 低 → 2★
      赔率调整：R ≥ 2.0 → +1★；R < 1.0 → −1★
      辩论微调（降级为仅减分）：\|d\| < 0.5（多空胶着、thesis 有real争议）→ −1★
      anchor 敏感（单一 anchor 失效即跨档）→ 封顶 4★
      5★ 门槛：必须同时 RM Conviction=高 且 R ≥ 2.0 且 anchor 不敏感
      结果钳制在 1-5★

    仓位上限（不变）：5★ 15-20% / 4★ 8-12% / 3★ 4-6% / 2★ 2-3% / 1★ ≤1%

    Args:
        rm_conviction: RM 评级置信度（高 / 中 / 低，照抄 RM thesis）
        odds_r: 赔率 R（U/D）
        abs_d: \|d\| 多空辩论比分差绝对值（仅作减分微调；缺省 0 会按"胶着"
               保守扣一档，应如实填 RM 辅助分析的 d 值）
        anchor_sensitive: anchor 是否敏感（True 表示 anchor 失效会跨档）

    Returns:
        dict: {"conviction_stars", "conviction_label", "position_low_pct",
               "position_high_pct", "reason"}
    """
    conv = (rm_conviction or "").strip()
    base = {"高": 4, "中": 3, "低": 2}.get(conv)
    if base is None:
        return {"error": f"rm_conviction 必须是 高/中/低，当前={rm_conviction!r}"}

    stars = base
    notes = [f"RM Conviction={conv} → 基础 {base}★"]

    if odds_r >= 2.0:
        stars += 1
        notes.append(f"R={odds_r:.2f}≥2.0 赔率优 → +1★")
    elif odds_r < 1.0:
        stars -= 1
        notes.append(f"R={odds_r:.2f}<1.0 赔率劣 → -1★")

    if abs_d < 0.5:
        stars -= 1
        notes.append(f"|d|={abs_d:.2f}<0.5 多空胶着 → -1★（辩论仅作减分微调）")

    if stars >= 5 and not (conv == "高" and odds_r >= 2.0 and not anchor_sensitive):
        stars = 4
        notes.append("5★ 门槛未全满足（需 RM=高 且 R≥2.0 且 anchor 不敏感）→ 封 4★")
    if anchor_sensitive and stars > 4:
        stars = 4
        notes.append("anchor 敏感 → 封顶 4★")

    stars = max(1, min(5, stars))
    label, lo, hi = _STAR_POSITION[stars]
    return {"conviction_stars": stars, "conviction_label": label,
            "position_low_pct": lo, "position_high_pct": hi,
            "reason": "；".join(notes)}


# ============================================================================
# PM 4 情景概率加权 E（含黑天鹅）
# ============================================================================

@tool
def compute_pm_scenario_e(scenarios: list[dict], p_0: float) -> dict:
    """PM 4 情景（乐观 / 基础 / 悲观 / 黑天鹅）概率加权期望收益 E。

    与 RM 3 情景不同：PM 必须加一档"黑天鹅"（5-15%），覆盖尾部风险。
    会校验概率加总 = 100，黑天鹅概率在 5-15% 范围。

    Args:
        scenarios: 4 个情景列表，每条含 name / probability (0-100) / target_price
                  必须包含 "黑天鹅" 或 "tail" 之一作为情景名
        p_0: 当前价

    Returns:
        dict: 含每情景收益率 + 概率加权 E + 校验信息
    """
    if not scenarios or p_0 <= 0:
        return {"error": "scenarios 或 p_0 无效"}

    prob_sum = sum(float(s.get("probability", 0)) for s in scenarios)
    scenario_returns = []
    weighted = 0.0
    has_tail = False
    tail_prob = 0.0

    for s in scenarios:
        name = s.get("name", "?")
        prob = float(s.get("probability", 0))
        tp = float(s["target_price"])
        ret = (tp - p_0) / p_0 * 100
        scenario_returns.append({
            "name": name, "probability_pct": prob,
            "target_price": tp, "return_pct": round(ret, 2),
        })
        weighted += (prob / 100) * ret
        if "黑天鹅" in name or "tail" in name.lower():
            has_tail = True
            tail_prob = prob

    return {
        "expected_return_pct": round(weighted, 2),
        "scenario_returns": scenario_returns,
        "prob_sum_check": round(prob_sum, 2),
        "prob_sum_valid": abs(prob_sum - 100) < 0.5,
        "has_tail_scenario": has_tail,
        "tail_probability_pct": tail_prob,
        "tail_valid": 5 <= tail_prob <= 15 if has_tail else False,
        "p_0": p_0,
    }


# ============================================================================
# 工具集合
# ============================================================================

PM_TOOLS = [
    compute_r_multiple_levels,
    compute_conviction_position_map,
    compute_pm_scenario_e,
]


PM_TOOLS_BY_NAME = {t.name: t for t in PM_TOOLS}
