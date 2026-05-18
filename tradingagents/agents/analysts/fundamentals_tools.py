"""Fundamentals Analyst 计算工具集（方案 B：tool calling）

把基本面分析里需要"自算"的数值（FCF、同比增速、应收增速差、扣非净利、
TTM ROE、经营现金流/净利润比 等）封装为 langchain @tool，
避免 LLM 心算偷懒填 null（之前多次发现的 bug）。

设计原则：
- 每个工具职责单一，输入/输出 schema 清晰
- 工具内部纯函数，无副作用
- 数据缺失时显式返回 null + 原因，不静默失败
"""

from typing import Optional

from langchain_core.tools import tool


# ============================================================================
# 自由现金流（FCF）
# ============================================================================

@tool
def compute_fcf(operating_cash_flow: float, capex: float,
                  unit: str = "亿元") -> dict:
    """计算自由现金流：FCF = OCF - CapEx。

    Args:
        operating_cash_flow: 经营活动现金流净额
        capex: 资本开支（购建固定资产、无形资产和其他长期资产支付的现金）
        unit: 单位（默认"亿元"，可填"万元"/"元"）

    Returns:
        dict: {"fcf": __, "ocf": __, "capex": __, "unit": __, "formula": "FCF = OCF − CapEx"}
    """
    fcf = operating_cash_flow - capex
    return {
        "fcf": round(fcf, 2),
        "ocf": round(operating_cash_flow, 2),
        "capex": round(capex, 2),
        "unit": unit,
        "formula": "FCF = OCF − CapEx",
    }


# ============================================================================
# 同比增速
# ============================================================================

@tool
def compute_yoy_growth(current_value: float, prior_year_value: float,
                         metric_name: str = "指标") -> dict:
    """计算同比增长率：(本期 − 去年同期) / |去年同期| × 100%。

    Args:
        current_value: 本期数值
        prior_year_value: 去年同期数值（同口径，如 Q1 对 Q1、年度对年度）
        metric_name: 指标名（如"营业收入"、"归母净利润"、"扣非净利润"）

    Returns:
        dict: {"yoy_growth_pct": __, "current": __, "prior": __, "metric": __, "formula": ...}
    """
    if prior_year_value == 0:
        return {
            "yoy_growth_pct": None,
            "current": current_value,
            "prior": prior_year_value,
            "metric": metric_name,
            "error": "去年同期为 0，无法计算同比",
        }

    yoy = (current_value - prior_year_value) / abs(prior_year_value) * 100
    return {
        "yoy_growth_pct": round(yoy, 2),
        "current": round(current_value, 2),
        "prior": round(prior_year_value, 2),
        "metric": metric_name,
        "formula": "(本期 − 去年同期) / |去年同期| × 100%",
    }


@tool
def compute_recurring_profit_yoy(current_recurring_eps: float, prior_recurring_eps: float,
                                    total_shares: float) -> dict:
    """计算扣非净利润同比增速（先从扣非 EPS × 总股本算出扣非净利，再算同比）。

    Args:
        current_recurring_eps: 本期扣非每股收益（元）
        prior_recurring_eps: 去年同期扣非每股收益（元）
        total_shares: 总股本（股，例如 12.22e8 表示 12.22 亿股）

    Returns:
        dict: 含本期/去年同期扣非净利绝对值（亿元）+ 同比增速
    """
    current_profit = current_recurring_eps * total_shares
    prior_profit = prior_recurring_eps * total_shares

    if prior_profit == 0:
        return {
            "yoy_growth_pct": None,
            "current_recurring_profit_yi": round(current_profit / 1e8, 2),
            "prior_recurring_profit_yi": round(prior_profit / 1e8, 2),
            "error": "去年同期扣非净利为 0",
        }

    yoy = (current_profit - prior_profit) / abs(prior_profit) * 100
    return {
        "yoy_growth_pct": round(yoy, 2),
        "current_recurring_profit_yi": round(current_profit / 1e8, 2),
        "prior_recurring_profit_yi": round(prior_profit / 1e8, 2),
        "total_shares_yi": round(total_shares / 1e8, 2),
        "formula": "扣非净利 = 扣非 EPS × 总股本；同比 = (本-去)/|去|",
    }


# ============================================================================
# 盈利质量
# ============================================================================

@tool
def compute_receivable_revenue_gap(receivable_growth_pct: float,
                                       revenue_growth_pct: float) -> dict:
    """计算应收账款增速与营收增速的差值（百分点）。

    判断标准：
    - < 0: 健康（应收增长慢于营收，回款好）
    - 0-5 pp: 正常
    - > 5 pp: 警惕（应收增长过快可能虚增收入）

    Args:
        receivable_growth_pct: 应收账款同比增速（百分比，如 30 表示 30%）
        revenue_growth_pct: 营业收入同比增速

    Returns:
        dict: {"gap_pp": __, "rating": 健康/正常/警惕, "interpretation": __}
    """
    gap = receivable_growth_pct - revenue_growth_pct
    if gap < 0:
        rating = "健康"
        interp = "应收增长慢于营收，回款情况良好"
    elif gap <= 5:
        rating = "正常"
        interp = "应收与营收增速基本匹配"
    else:
        rating = "警惕"
        interp = f"应收增速比营收快 {gap:.1f}pp，可能虚增收入或回款恶化"
    return {
        "gap_pp": round(gap, 2),
        "receivable_growth_pct": round(receivable_growth_pct, 2),
        "revenue_growth_pct": round(revenue_growth_pct, 2),
        "rating": rating,
        "interpretation": interp,
    }


@tool
def compute_ocf_to_profit_ratio(operating_cash_flow: float, net_profit: float) -> dict:
    """计算经营现金流/净利润比，判断盈利质量。

    判断标准：
    - > 0.8: 健康（净利润有现金支撑）
    - 0.5-0.8: 警惕
    - < 0.5: 红旗（盈利质量低，可能存在应收/存货大量挤压）

    Args:
        operating_cash_flow: 经营活动现金流净额
        net_profit: 归母净利润

    Returns:
        dict: {"ratio": __, "rating": 高/中/低, "interpretation": __}
    """
    if net_profit == 0:
        return {"ratio": None, "rating": "无法判断", "error": "净利润为 0"}

    ratio = operating_cash_flow / net_profit
    if ratio > 0.8:
        rating = "高"
        interp = "净利润有充足现金支撑，盈利质量高"
    elif ratio >= 0.5:
        rating = "中"
        interp = "盈利质量一般，需关注应收/存货变化"
    else:
        rating = "低"
        interp = "现金流远低于净利润，可能存在应收挤压或会计调整，红旗"
    return {
        "ratio": round(ratio, 2),
        "ocf": round(operating_cash_flow, 2),
        "net_profit": round(net_profit, 2),
        "rating": rating,
        "interpretation": interp,
    }


@tool
def compute_payable_receivable_ratio(accounts_payable: float,
                                       accounts_receivable: float) -> dict:
    """应付/应收比，反映对上下游议价能力。

    > 1: 议价能力强（占用上游资金多于被下游占用）
    < 1: 现金流压力（被下游占用多于占用上游）

    Args:
        accounts_payable: 应付账款
        accounts_receivable: 应收账款

    Returns:
        dict: {"ratio": __, "interpretation": __}
    """
    if accounts_receivable == 0:
        return {"ratio": None, "error": "应收账款为 0，比值无意义"}

    ratio = accounts_payable / accounts_receivable
    if ratio > 1:
        interp = "议价能力强：占用上游资金多于被下游占用"
    else:
        interp = "现金流偏紧：被下游占用资金多于占用上游"
    return {
        "ratio": round(ratio, 2),
        "accounts_payable": round(accounts_payable, 2),
        "accounts_receivable": round(accounts_receivable, 2),
        "interpretation": interp,
    }


# ============================================================================
# TTM ROE
# ============================================================================

@tool
def compute_ttm_roe(quarterly_net_profits: list[float],
                      current_equity: float) -> dict:
    """计算 TTM 年化 ROE = 滚动 4 季度净利润 ÷ 当期净资产 × 100%。

    用于避免单季 ROE 被误读为年度（季节性强的公司 Q1 ROE 远低于年化）。

    Args:
        quarterly_net_profits: 最近 4 个季度的归母净利润（同单位，如亿元），按时间正序
        current_equity: 当期净资产（同单位）

    Returns:
        dict: {"ttm_roe_pct": __, "ttm_net_profit": __, "current_equity": __, "basis": "ttm"}
    """
    if len(quarterly_net_profits) != 4:
        return {
            "ttm_roe_pct": None,
            "error": f"需要 4 季度数据，实际 {len(quarterly_net_profits)} 季",
        }
    if current_equity <= 0:
        return {"ttm_roe_pct": None, "error": "净资产应 > 0"}

    ttm_profit = sum(quarterly_net_profits)
    roe = ttm_profit / current_equity * 100
    return {
        "ttm_roe_pct": round(roe, 2),
        "ttm_net_profit": round(ttm_profit, 2),
        "current_equity": round(current_equity, 2),
        "basis": "ttm",
        "formula": "Σ最近4季净利 / 当期净资产 × 100%",
    }


# ============================================================================
# 工具集合
# ============================================================================

FUNDAMENTALS_TOOLS = [
    compute_fcf,
    compute_yoy_growth,
    compute_recurring_profit_yoy,
    compute_receivable_revenue_gap,
    compute_ocf_to_profit_ratio,
    compute_payable_receivable_ratio,
    compute_ttm_roe,
]


FUNDAMENTALS_TOOLS_BY_NAME = {t.name: t for t in FUNDAMENTALS_TOOLS}
