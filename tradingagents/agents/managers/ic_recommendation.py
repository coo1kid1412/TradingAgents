"""Deterministic four-pillar investment-committee recommendation engine."""

from __future__ import annotations

from math import isfinite
from typing import Optional

from langchain_core.tools import tool


RATINGS_ORDER = ["SELL", "UNDERWEIGHT", "HOLD", "OVERWEIGHT", "BUY"]
THRESHOLD_STYLE_COEF = {
    "blue_chip": 1.0,
    "cyclical": 1.0,
    "illiquid": 0.7,
    "etf": 1.0,
    "high_beta_growth": 1.5,
    "theme_speculation": 2.0,
}
THRESHOLD_BASE_CONFIGURATION = 15.0
THRESHOLD_BASE_HIGH = 35.0
FADING_HIGH_LOCK = 30.0

_ENUMS = {
    "thesis_state": {"strong", "adequate", "mixed", "weak", "invalid"},
    "thesis_direction": {"bullish", "bearish", "neutral", "mixed"},
    "valuation_state": {"attractive", "fair", "stretched", "invalid"},
    "catalyst_state": {"strong", "visible", "weak", "absent", "adverse"},
    "priced_in": {"low", "partial", "high", "unknown"},
    "durability_state": {"resilient", "acceptable", "fragile", "broken"},
    "evidence_quality": {"complete", "partial", "insufficient"},
}


def _thresholds(style: str, theme_premium_pct: float, theme_stage: str) -> dict:
    style_key = (style or "").strip().lower()
    coefficient = THRESHOLD_STYLE_COEF.get(style_key, 1.0)
    theme_factor = max(0.0, 1.0 + float(theme_premium_pct or 0.0) / 100.0)
    configuration = max(
        5.0,
        THRESHOLD_BASE_CONFIGURATION * coefficient * theme_factor,
    )
    high = THRESHOLD_BASE_HIGH * coefficient * theme_factor
    if "fading" in (theme_stage or "").strip().lower():
        high = min(high, FADING_HIGH_LOCK)
    high = max(high, configuration + 5.0)
    return {
        "configuration_pct": round(configuration, 2),
        "high_pct": round(high, 2),
        "style": style_key or "default",
        "style_coefficient": coefficient,
        "theme_factor": round(theme_factor, 4),
    }


def _validate_enum(name: str, value: str) -> Optional[dict]:
    normalized = (value or "").strip().lower()
    if normalized not in _ENUMS[name]:
        return {"error": f"{name} 非法：{value}"}
    return None


def _validate_evidence(
    eligible_evidence_ids: list[str],
    evidence_groups: list[list[str]],
) -> Optional[dict]:
    eligible = {str(item).strip() for item in eligible_evidence_ids if str(item).strip()}
    referenced = {
        str(item).strip()
        for group in evidence_groups
        for item in group
        if str(item).strip()
    }
    invalid = sorted(referenced - eligible)
    if invalid:
        return {"error": f"不合格证据 ID：{', '.join(invalid)}"}
    return None


@tool
def compute_ic_recommendation(
    scenario_expected_return_pct: float,
    downside_pct: float,
    payoff_ratio: float,
    thesis_state: str,
    thesis_direction: str,
    valuation_state: str,
    catalyst_state: str,
    priced_in: str,
    durability_state: str,
    hard_veto: bool,
    evidence_quality: str,
    style: str = "",
    theme_premium_pct: float = 0.0,
    theme_stage: str = "",
    crowded_long: bool = False,
    eligible_evidence_ids: Optional[list[str]] = None,
    thesis_evidence_ids: Optional[list[str]] = None,
    valuation_evidence_ids: Optional[list[str]] = None,
    catalyst_evidence_ids: Optional[list[str]] = None,
    durability_evidence_ids: Optional[list[str]] = None,
    thesis_breaker_evidence_ids: Optional[list[str]] = None,
) -> dict:
    """Map four audited research pillars to one 12-month rating.

    Short-term timing, market entry gates, positions and trade actions are
    intentionally absent. Those belong to the Portfolio Manager layer.
    """

    values = {
        "thesis_state": thesis_state,
        "thesis_direction": thesis_direction,
        "valuation_state": valuation_state,
        "catalyst_state": catalyst_state,
        "priced_in": priced_in,
        "durability_state": durability_state,
        "evidence_quality": evidence_quality,
    }
    for name, value in values.items():
        error = _validate_enum(name, value)
        if error:
            return error
        values[name] = value.strip().lower()

    numeric_values = {
        "scenario_expected_return_pct": scenario_expected_return_pct,
        "downside_pct": downside_pct,
        "payoff_ratio": payoff_ratio,
        "theme_premium_pct": theme_premium_pct,
    }
    for name, value in numeric_values.items():
        if isinstance(value, bool):
            return {"error": f"{name} 必须是有限数值"}
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return {"error": f"{name} 必须是有限数值"}
        if not isfinite(numeric):
            return {"error": f"{name} 必须是有限数值"}
        numeric_values[name] = numeric
    if numeric_values["payoff_ratio"] < 0:
        return {"error": "payoff_ratio 必须 >= 0"}

    evidence_groups = [
        list(thesis_evidence_ids or []),
        list(valuation_evidence_ids or []),
        list(catalyst_evidence_ids or []),
        list(durability_evidence_ids or []),
        list(thesis_breaker_evidence_ids or []),
    ]
    evidence_error = _validate_evidence(
        list(eligible_evidence_ids or []), evidence_groups,
    )
    if evidence_error:
        return evidence_error

    breaker_ids = list(thesis_breaker_evidence_ids or [])
    if hard_veto and not breaker_ids:
        return {"error": "hard_veto=true 但缺少合格 thesis breaker 证据"}

    expected_return = numeric_values["scenario_expected_return_pct"]
    thresholds = _thresholds(
        style,
        numeric_values["theme_premium_pct"],
        theme_stage,
    )
    configuration = thresholds["configuration_pct"]
    high = thresholds["high_pct"]
    reason_codes: list[str] = []
    bound_sources: list[str] = []

    thesis = values["thesis_state"]
    thesis_direction_value = values["thesis_direction"]
    valuation = values["valuation_state"]
    catalyst = values["catalyst_state"]
    durability = values["durability_state"]
    quality = values["evidence_quality"]

    if hard_veto:
        rating = "SELL"
        reason_codes.extend(["THESIS_BREAK_OVERRIDE", "RESEARCH_HARD_VETO"])
        bound_sources.append("正式证据验证 thesis breaker，研究评级直接 SELL")
    else:
        if expected_return >= high:
            rating = "BUY"
            reason_codes.append("EXPECTED_RETURN_HIGH")
        elif expected_return >= configuration:
            rating = "OVERWEIGHT"
            reason_codes.append("EXPECTED_RETURN_POSITIVE")
        elif expected_return > -configuration:
            rating = "HOLD"
            reason_codes.append("EXPECTED_RETURN_NEUTRAL")
        elif expected_return > -high:
            rating = "UNDERWEIGHT"
            reason_codes.append("EXPECTED_RETURN_NEGATIVE")
        else:
            rating = "SELL"
            reason_codes.append("EXPECTED_RETURN_DEEPLY_NEGATIVE")

        if rating in {"BUY", "OVERWEIGHT"}:
            if thesis == "strong":
                reason_codes.append("THESIS_STRONG")
            elif thesis == "adequate":
                reason_codes.append("THESIS_ADEQUATE")

            if thesis in {"weak", "invalid"}:
                rating = "HOLD"
                reason_codes.append("WEAK_THESIS_CAP")
                bound_sources.append("经营 thesis weak/invalid，正向评级封顶 HOLD")
            elif durability == "broken":
                rating = "HOLD"
                reason_codes.append("BROKEN_DURABILITY_CAP")
                bound_sources.append("持续性 broken 且未形成正式 breaker，正向评级封顶 HOLD")
            elif valuation == "invalid":
                rating = "HOLD"
                reason_codes.append("INVALID_VALUATION_CAP")
                bound_sources.append("估值情景 invalid，正向评级封顶 HOLD")
            elif valuation == "stretched":
                rating = "HOLD"
                reason_codes.append("VALUATION_CONFLICT_CAP")
                bound_sources.append("正预期收益与 stretched 估值状态冲突，封顶 HOLD")
            elif rating == "BUY" and (
                thesis not in {"strong", "adequate"}
                or catalyst not in {"strong", "visible"}
                or durability == "fragile"
            ):
                rating = "OVERWEIGHT"
                reason_codes.append("BUY_PILLAR_GATE_CAP")
                bound_sources.append("BUY 支柱门槛未全部满足，封顶 OVERWEIGHT")

        elif rating in {"SELL", "UNDERWEIGHT"}:
            non_valuation_bear_case = (
                thesis == "weak"
                or thesis_direction_value == "bearish"
                or catalyst == "adverse"
                or durability in {"fragile", "broken"}
            )
            if not non_valuation_bear_case:
                rating = "HOLD"
                reason_codes.append("NO_NON_VALUATION_BEAR_CASE")
                bound_sources.append("只有估值负收益，缺少非估值看空支柱，收敛 HOLD")
            elif rating == "SELL" and not (
                thesis == "weak" or durability == "broken"
            ):
                rating = "UNDERWEIGHT"
                reason_codes.append("SELL_PILLAR_GATE_CAP")
                bound_sources.append("SELL 缺少 weak thesis/broken durability，封顶 UNDERWEIGHT")

        if quality == "insufficient" or thesis == "invalid" or valuation == "invalid":
            if rating != "HOLD":
                rating = "HOLD"
            reason_codes.append("INSUFFICIENT_EVIDENCE_CAP")
            bound_sources.append("关键证据不足或无效，方向性评级封顶 HOLD")
        elif quality == "partial" and rating in {"BUY", "SELL"}:
            rating = "OVERWEIGHT" if rating == "BUY" else "UNDERWEIGHT"
            reason_codes.append("PARTIAL_EVIDENCE_CAP")
            bound_sources.append("证据 partial，极端评级收敛一档")

        if crowded_long and rating == "BUY":
            rating = "OVERWEIGHT"
            reason_codes.append("CROWDED_LONG_CAP")
            bound_sources.append("拥挤多头，BUY 封顶 OVERWEIGHT")
        elif crowded_long:
            reason_codes.append("CROWDED_LONG_CAP")

    pillar_effects = {
        "thesis": {
            "state": thesis,
            "direction": thesis_direction_value,
            "effect": "foundation" if thesis in {"strong", "adequate"} else "cap",
        },
        "valuation": {
            "state": valuation,
            "effect": "foundation",
        },
        "catalyst": {
            "state": catalyst,
            "priced_in": values["priced_in"],
            "effect": "support" if catalyst in {"strong", "visible"} else "oppose",
        },
        "durability": {
            "state": durability,
            "effect": "research_veto" if hard_veto else (
                "cap" if durability in {"fragile", "broken"} else "support"
            ),
        },
    }
    reason_codes = list(dict.fromkeys(reason_codes))
    explanation = (
        f"情景概率加权收益 {expected_return:+.2f}%（配置/高档阈值 ±{configuration:.2f}%/±{high:.2f}%）"
        f"；thesis={thesis}，valuation={valuation}，catalyst={catalyst}，"
        f"durability={durability}，evidence={quality}；最终 {rating}。"
    )
    if bound_sources:
        explanation += " 约束：" + "；".join(bound_sources)

    return {
        "research_rating": rating,
        "rating_reason_codes": reason_codes,
        "scenario_expected_return_pct": round(expected_return, 2),
        "downside_pct": round(numeric_values["downside_pct"], 2),
        "payoff_ratio": round(numeric_values["payoff_ratio"], 4),
        "thresholds": thresholds,
        "pillar_effects": pillar_effects,
        "bounds": {"sources": bound_sources},
        "evidence_ids": {
            "thesis": list(thesis_evidence_ids or []),
            "valuation": list(valuation_evidence_ids or []),
            "catalyst": list(catalyst_evidence_ids or []),
            "durability": list(durability_evidence_ids or []),
            "thesis_breaker": breaker_ids,
        },
        "explanation": explanation,
    }

