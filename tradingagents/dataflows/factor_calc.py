"""量化因子计算（pure deterministic Python，无 I/O）

设计原则：
- 每个因子用"绝对阈值分段映射"打分到 0-100
- 阈值参考 A 股长期统计分布（不依赖跨标的横截面数据）
- 缺失输入返回 None，不猜测
- 复合分数 = 6 因子加权平均（GARP 风格默认权重）

因子组与权重（默认）：
- Momentum 动量      12%   R3M + R6M + R12M
- Value 价值          22%   PE(TTM) + PB + 行业相对 PE
- Quality 质量        22%   ROE + 毛利率 + 净利率
- Growth 成长         18%   营收 YoY + 净利 YoY
- LowVol 低波动       5%    30 日年化波动率
- AntiCrowding 反拥挤  9%   60 日累计收益 + 换手率加速度
- Capital Flow 资金流  12%  capital_flow_score（由 Capital Flow Officer 预计算）

复合分数解读：
- 0-30   显著负面，quant 信号建议规避
- 30-50  偏弱
- 50-65  中性
- 65-80  偏强
- 80-100 显著正面，quant 信号强烈支持
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional


# ---------------------------------------------------------------------------
# 辅助：阈值分段映射
# ---------------------------------------------------------------------------
def _stepwise(value: float, breakpoints: list[tuple[float, float]]) -> float:
    """根据有序断点列表 [(阈值, 分数), ...] 把 value 映射到分数。

    断点按 value 从小到大排列。value <= 阈值的最小档命中。
    """
    for bp, score in breakpoints:
        if value <= bp:
            return score
    return breakpoints[-1][1]  # 超出最大档，给最差分


# ---------------------------------------------------------------------------
# Factor 1: Momentum
# ---------------------------------------------------------------------------
def _score_return(ret_pct: float, scale: str) -> float:
    """根据持有期收益率（百分比，如 +25 表示 +25%）映射到 0-100 分。

    scale 控制阈值：3m / 6m / 12m。
    """
    if scale == "3m":
        bps = [(-20, 5), (-10, 20), (0, 35), (10, 50), (20, 65), (40, 80), (1e9, 95)]
    elif scale == "6m":
        bps = [(-25, 5), (-10, 20), (0, 35), (10, 50), (25, 65), (50, 80), (1e9, 95)]
    elif scale == "12m":
        bps = [(-30, 5), (-10, 20), (0, 35), (10, 50), (30, 65), (60, 80), (1e9, 95)]
    else:
        raise ValueError(f"未知 scale: {scale}")
    return _stepwise(ret_pct, bps)


def momentum_score(
    r3m_pct: Optional[float],
    r6m_pct: Optional[float],
    r12m_pct: Optional[float],
) -> tuple[Optional[float], dict]:
    """动量因子：R3M + R6M + R12M 加权平均（30/35/35）。

    Returns:
        (composite_score, breakdown_dict)
        若三段收益均缺失，composite 返回 None。
    """
    parts: list[tuple[float, float]] = []  # (子分数, 权重)
    breakdown: dict = {}

    if r3m_pct is not None:
        s = _score_return(r3m_pct, "3m")
        breakdown["r3m_pct"] = round(r3m_pct, 2)
        breakdown["r3m_score"] = round(s, 1)
        parts.append((s, 0.30))
    if r6m_pct is not None:
        s = _score_return(r6m_pct, "6m")
        breakdown["r6m_pct"] = round(r6m_pct, 2)
        breakdown["r6m_score"] = round(s, 1)
        parts.append((s, 0.35))
    if r12m_pct is not None:
        s = _score_return(r12m_pct, "12m")
        breakdown["r12m_pct"] = round(r12m_pct, 2)
        breakdown["r12m_score"] = round(s, 1)
        parts.append((s, 0.35))

    if not parts:
        return None, breakdown

    total_w = sum(w for _, w in parts)
    composite = sum(s * w for s, w in parts) / total_w
    return round(composite, 1), breakdown


# ---------------------------------------------------------------------------
# Factor 2: Value
# ---------------------------------------------------------------------------
def value_score(
    pe_ttm: Optional[float],
    pb: Optional[float],
    pe_industry_median: Optional[float] = None,
) -> tuple[Optional[float], dict]:
    """价值因子：PE TTM + PB + 行业相对 PE。

    - PE TTM 绝对阈值（A 股长期分布）
    - PB 绝对阈值
    - 行业相对 PE：若可得，作为加减分项（±25 分）
    """
    breakdown: dict = {}
    parts: list[tuple[float, float]] = []

    # PE TTM
    if pe_ttm is not None and pe_ttm > 0:
        bps = [(15, 100), (25, 85), (40, 70), (60, 50), (100, 30), (150, 15), (1e9, 5)]
        s_pe = _stepwise(pe_ttm, bps)
        breakdown["pe_ttm"] = round(pe_ttm, 2)
        breakdown["pe_score"] = round(s_pe, 1)
        parts.append((s_pe, 0.60))
    elif pe_ttm is not None and pe_ttm <= 0:
        # 负 PE（亏损）：直接给极低分
        breakdown["pe_ttm"] = round(pe_ttm, 2)
        breakdown["pe_score"] = 0
        parts.append((0, 0.60))

    # PB
    if pb is not None and pb > 0:
        bps = [(1, 100), (2, 85), (4, 65), (7, 45), (15, 25), (1e9, 5)]
        s_pb = _stepwise(pb, bps)
        breakdown["pb"] = round(pb, 2)
        breakdown["pb_score"] = round(s_pb, 1)
        parts.append((s_pb, 0.30))

    # 行业相对 PE（加减分项）
    if pe_ttm is not None and pe_industry_median is not None and pe_industry_median > 0:
        rel = pe_ttm / pe_industry_median
        breakdown["pe_relative"] = round(rel, 2)
        # rel < 0.5: +20 分; 0.5-0.8: +10; 0.8-1.2: 0; 1.2-2: -10; >2: -25
        if rel <= 0.5:
            adj = 20
        elif rel <= 0.8:
            adj = 10
        elif rel <= 1.2:
            adj = 0
        elif rel <= 2.0:
            adj = -10
        else:
            adj = -25
        breakdown["pe_relative_adj"] = adj
        parts.append((50 + adj, 0.10))  # 基础 50 + 调整

    if not parts:
        return None, breakdown

    total_w = sum(w for _, w in parts)
    composite = sum(s * w for s, w in parts) / total_w
    composite = max(0, min(100, composite))
    return round(composite, 1), breakdown


# ---------------------------------------------------------------------------
# Factor 3: Quality
# ---------------------------------------------------------------------------
def quality_score(
    roe_ttm_pct: Optional[float],
    gross_margin_pct: Optional[float],
    net_margin_pct: Optional[float],
) -> tuple[Optional[float], dict]:
    """质量因子：ROE + 毛利率 + 净利率。"""
    breakdown: dict = {}
    parts: list[tuple[float, float]] = []

    if roe_ttm_pct is not None:
        if roe_ttm_pct < 0:
            s = 0
        else:
            bps = [(5, 15), (10, 35), (15, 55), (20, 70), (25, 85), (1e9, 100)]
            s = _stepwise(roe_ttm_pct, bps)
        breakdown["roe_ttm_pct"] = round(roe_ttm_pct, 2)
        breakdown["roe_score"] = round(s, 1)
        parts.append((s, 0.50))

    if gross_margin_pct is not None:
        bps = [(10, 10), (20, 25), (30, 45), (45, 65), (60, 85), (1e9, 100)]
        s = _stepwise(gross_margin_pct, bps)
        breakdown["gross_margin_pct"] = round(gross_margin_pct, 2)
        breakdown["gross_margin_score"] = round(s, 1)
        parts.append((s, 0.25))

    if net_margin_pct is not None:
        if net_margin_pct < 0:
            s = 0
        else:
            bps = [(5, 25), (10, 45), (15, 65), (25, 85), (1e9, 100)]
            s = _stepwise(net_margin_pct, bps)
        breakdown["net_margin_pct"] = round(net_margin_pct, 2)
        breakdown["net_margin_score"] = round(s, 1)
        parts.append((s, 0.25))

    if not parts:
        return None, breakdown

    total_w = sum(w for _, w in parts)
    composite = sum(s * w for s, w in parts) / total_w
    return round(composite, 1), breakdown


# ---------------------------------------------------------------------------
# Factor 4: Growth
# ---------------------------------------------------------------------------
def growth_score(
    revenue_yoy_pct: Optional[float],
    net_profit_yoy_pct: Optional[float],
) -> tuple[Optional[float], dict]:
    """成长因子：营收 YoY + 净利 YoY。"""
    breakdown: dict = {}
    parts: list[tuple[float, float]] = []

    def _score_yoy(yoy: float) -> float:
        bps = [(-10, 5), (0, 15), (5, 30), (15, 50), (30, 70), (50, 85), (1e9, 100)]
        return _stepwise(yoy, bps)

    if revenue_yoy_pct is not None:
        s = _score_yoy(revenue_yoy_pct)
        breakdown["revenue_yoy_pct"] = round(revenue_yoy_pct, 2)
        breakdown["revenue_yoy_score"] = round(s, 1)
        parts.append((s, 0.35))

    if net_profit_yoy_pct is not None:
        s = _score_yoy(net_profit_yoy_pct)
        breakdown["net_profit_yoy_pct"] = round(net_profit_yoy_pct, 2)
        breakdown["net_profit_yoy_score"] = round(s, 1)
        parts.append((s, 0.65))

    if not parts:
        return None, breakdown

    total_w = sum(w for _, w in parts)
    composite = sum(s * w for s, w in parts) / total_w
    return round(composite, 1), breakdown


# ---------------------------------------------------------------------------
# Factor 5: Low Volatility
# ---------------------------------------------------------------------------
def lowvol_score(realized_vol_annualized_pct: Optional[float]) -> tuple[Optional[float], dict]:
    """低波因子：30 日年化波动率，越低越好。"""
    breakdown: dict = {}
    if realized_vol_annualized_pct is None:
        return None, breakdown
    bps = [(15, 100), (25, 80), (40, 60), (60, 40), (80, 20), (1e9, 5)]
    s = _stepwise(realized_vol_annualized_pct, bps)
    breakdown["vol_30d_annualized_pct"] = round(realized_vol_annualized_pct, 2)
    breakdown["vol_score"] = round(s, 1)
    return round(s, 1), breakdown


# ---------------------------------------------------------------------------
# Factor 6: Anti-Crowding
# ---------------------------------------------------------------------------
def anticrowding_score(
    r60d_pct: Optional[float],
    turnover_ratio_30d_to_90d: Optional[float],
) -> tuple[Optional[float], dict]:
    """反拥挤因子：近 60 日累计收益 + 换手率加速度（30d 均值/90d 均值）。

    越冷门 = 分越高（防追高）。
    """
    breakdown: dict = {}
    parts: list[tuple[float, float]] = []

    if r60d_pct is not None:
        bps = [(5, 95), (15, 80), (30, 60), (60, 35), (100, 15), (1e9, 0)]
        s = _stepwise(r60d_pct, bps)
        breakdown["r60d_pct"] = round(r60d_pct, 2)
        breakdown["r60d_score"] = round(s, 1)
        parts.append((s, 0.60))

    if turnover_ratio_30d_to_90d is not None:
        bps = [(0.8, 90), (1.2, 70), (2.0, 40), (3.0, 20), (1e9, 5)]
        s = _stepwise(turnover_ratio_30d_to_90d, bps)
        breakdown["turnover_accel"] = round(turnover_ratio_30d_to_90d, 2)
        breakdown["turnover_accel_score"] = round(s, 1)
        parts.append((s, 0.40))

    if not parts:
        return None, breakdown

    total_w = sum(w for _, w in parts)
    composite = sum(s * w for s, w in parts) / total_w
    return round(composite, 1), breakdown


# ---------------------------------------------------------------------------
# Composite
# ---------------------------------------------------------------------------
DEFAULT_FACTOR_WEIGHTS = {
    "momentum": 0.12,       # was 0.15, -3% → 给 capital_flow 腾权重
    "value": 0.22,          # was 0.25, 等比缩
    "quality": 0.22,        # was 0.25, 等比缩
    "growth": 0.18,         # was 0.20, 等比缩
    "lowvol": 0.05,         # 不动（已是最小）
    "anticrowding": 0.09,   # was 0.10, 等比缩
    "capital_flow": 0.12,   # 新：Capital Flow Officer 的 capital_flow_score (0-100)
}


@dataclass
class QuantScoreResult:
    composite: Optional[float]
    factor_scores: dict
    factor_breakdowns: dict
    weights_used: dict
    interpretation: str
    coverage: dict  # 哪些因子有数据 / 哪些缺失


def _interpret(score: Optional[float]) -> str:
    if score is None:
        return "数据不足，无法给出量化判断"
    if score < 30:
        return "显著负面（强烈建议规避）"
    if score < 50:
        return "偏弱"
    if score < 65:
        return "中性"
    if score < 80:
        return "偏强"
    return "显著正面"


def compute_quant_score(
    *,
    # Momentum inputs
    r3m_pct: Optional[float] = None,
    r6m_pct: Optional[float] = None,
    r12m_pct: Optional[float] = None,
    # Value inputs
    pe_ttm: Optional[float] = None,
    pb: Optional[float] = None,
    pe_industry_median: Optional[float] = None,
    # Quality inputs
    roe_ttm_pct: Optional[float] = None,
    gross_margin_pct: Optional[float] = None,
    net_margin_pct: Optional[float] = None,
    # Growth inputs
    revenue_yoy_pct: Optional[float] = None,
    net_profit_yoy_pct: Optional[float] = None,
    # LowVol input
    realized_vol_annualized_pct: Optional[float] = None,
    # AntiCrowding inputs
    r60d_pct: Optional[float] = None,
    turnover_ratio_30d_to_90d: Optional[float] = None,
    # Capital Flow (第 7 因子，由 Capital Flow Officer 预计算，直接传入 0-100 分)
    capital_flow_score_input: Optional[float] = None,
    # Optional weight override
    weights: Optional[dict] = None,
) -> QuantScoreResult:
    """主入口：聚合 6 类因子，按权重加权得到 0-100 综合分。

    缺失的因子自动从权重分母中扣除（归一化），不强制要求所有输入都齐全。
    """
    weights = weights or DEFAULT_FACTOR_WEIGHTS

    factor_scores: dict = {}
    factor_breakdowns: dict = {}

    s, bd = momentum_score(r3m_pct, r6m_pct, r12m_pct)
    factor_scores["momentum"] = s
    factor_breakdowns["momentum"] = bd

    s, bd = value_score(pe_ttm, pb, pe_industry_median)
    factor_scores["value"] = s
    factor_breakdowns["value"] = bd

    s, bd = quality_score(roe_ttm_pct, gross_margin_pct, net_margin_pct)
    factor_scores["quality"] = s
    factor_breakdowns["quality"] = bd

    s, bd = growth_score(revenue_yoy_pct, net_profit_yoy_pct)
    factor_scores["growth"] = s
    factor_breakdowns["growth"] = bd

    s, bd = lowvol_score(realized_vol_annualized_pct)
    factor_scores["lowvol"] = s
    factor_breakdowns["lowvol"] = bd

    s, bd = anticrowding_score(r60d_pct, turnover_ratio_30d_to_90d)
    factor_scores["anticrowding"] = s
    factor_breakdowns["anticrowding"] = bd

    # Capital Flow（第 7 因子）：分数由 capital_flow_utils 已计算好，此处直接注册
    if capital_flow_score_input is not None:
        # clamp 到 [0, 100] 保护
        cf_clamped = max(0.0, min(100.0, capital_flow_score_input))
        factor_scores["capital_flow"] = round(cf_clamped, 1)
        factor_breakdowns["capital_flow"] = {
            "capital_flow_score_raw": round(capital_flow_score_input, 2),
            "note": "由 Capital Flow Officer 预计算（5 维投票 + regime 硬约束）",
        }
    else:
        factor_scores["capital_flow"] = None
        factor_breakdowns["capital_flow"] = {"note": "capital_flow_score 不可用（非 A 股或数据缺失）"}

    # 复合分数（缺失因子从权重分母扣除）
    available = [(k, v) for k, v in factor_scores.items() if v is not None]
    total_w = sum(weights.get(k, 0) for k, _ in available)

    if total_w <= 0:
        composite = None
    else:
        composite = sum(weights.get(k, 0) * v for k, v in available) / total_w
        composite = round(composite, 1)

    coverage = {
        "available": [k for k, v in factor_scores.items() if v is not None],
        "missing": [k for k, v in factor_scores.items() if v is None],
        "total_weight_used": round(total_w, 3),
    }

    return QuantScoreResult(
        composite=composite,
        factor_scores=factor_scores,
        factor_breakdowns=factor_breakdowns,
        weights_used=weights,
        interpretation=_interpret(composite),
        coverage=coverage,
    )


# ---------------------------------------------------------------------------
# 价格因子辅助：从 OHLCV DataFrame 计算
# ---------------------------------------------------------------------------
def compute_price_factors(price_df) -> dict:
    """从 OHLCV DataFrame 提取 momentum / lowvol / anticrowding 的原始输入。

    Args:
        price_df: 含 Date / Close / Volume 列的 DataFrame，按日期升序，至少 250 个交易日。

    Returns:
        dict 含 r3m_pct / r6m_pct / r12m_pct / r60d_pct
              / realized_vol_annualized_pct / turnover_ratio_30d_to_90d
        缺失项返回 None。
    """
    import pandas as pd

    result = {
        "r3m_pct": None,
        "r6m_pct": None,
        "r12m_pct": None,
        "r60d_pct": None,
        "realized_vol_annualized_pct": None,
        "turnover_ratio_30d_to_90d": None,
    }

    if price_df is None or len(price_df) == 0 or "Close" not in price_df.columns:
        return result

    closes = pd.to_numeric(price_df["Close"], errors="coerce").dropna().reset_index(drop=True)
    if len(closes) < 22:  # 至少一个月
        return result

    latest = float(closes.iloc[-1])

    def _ret_pct(n_days: int) -> Optional[float]:
        if len(closes) <= n_days:
            return None
        prev = float(closes.iloc[-n_days - 1])
        if prev <= 0:
            return None
        return (latest / prev - 1) * 100

    result["r3m_pct"] = _ret_pct(63)   # ~3 个月 (252/4)
    result["r6m_pct"] = _ret_pct(126)  # ~6 个月
    result["r12m_pct"] = _ret_pct(252) # ~12 个月
    result["r60d_pct"] = _ret_pct(60)  # 60 个交易日

    # 30 日年化波动率：日收益标准差 × sqrt(252)
    if len(closes) >= 31:
        daily_ret = closes.pct_change().dropna()
        recent_30 = daily_ret.tail(30)
        if len(recent_30) >= 20 and recent_30.std() > 0:
            vol_annual = float(recent_30.std() * math.sqrt(252) * 100)
            result["realized_vol_annualized_pct"] = vol_annual

    # 换手率加速度：30 日均成交量 / 90 日均成交量
    if "Volume" in price_df.columns and len(price_df) >= 90:
        volumes = pd.to_numeric(price_df["Volume"], errors="coerce").dropna()
        if len(volumes) >= 90:
            v30 = volumes.tail(30).mean()
            v90 = volumes.tail(90).mean()
            if v90 > 0:
                result["turnover_ratio_30d_to_90d"] = float(v30 / v90)

    return result
