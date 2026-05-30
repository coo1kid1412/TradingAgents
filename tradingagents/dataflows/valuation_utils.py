"""估值指标公共计算工具。

从原始财务数据 + 收盘价自行计算 PE/PB/PS/市值等估值指标，
供 Tushare 层和 AKShare 层复用，避免依赖可能限流的估值接口。
"""

import pandas as pd


def compute_ttm_eps(
    fina_df: pd.DataFrame,
    eps_col: str = "eps",
    date_col: str = "end_date",
) -> float | None:
    """从累计财报数据计算 TTM（滚动12个月）每股收益。

    TTM_EPS = 最新年报EPS - 上年同期Q1_EPS + 今年Q1_EPS

    中国财报披露规则:
      Q1(单季) → H1(累计) → 9M(累计) → Annual(累计)
      fina_indicator 的 eps 字段为累计值，故:
      TTM = Annual_EPS - prev_Q1_EPS + curr_Q1_EPS

    Args:
        fina_df: 包含 EPS 和报告期的 DataFrame
        eps_col: EPS 列名（Tushare: "eps", AKShare利润表: "BASIC_EPS"）
        date_col: 日期列名（Tushare: "end_date", AKShare: "REPORT_DATE"）
    """
    if fina_df is None or fina_df.empty or eps_col not in fina_df.columns:
        return None
    if date_col not in fina_df.columns:
        return None

    df = fina_df.sort_values(date_col).copy()
    # 统一日期格式：去除连字符，兼容 "2023-12-31" / "20231231" / datetime.date
    # AKShare REPORT_DATE 可能带时间部分如 "2025-12-31 00:00:00"，先截取前10字符
    df["_date_norm"] = df[date_col].astype(str).str[:10].str.replace("-", "", regex=False)

    # 最新年报 EPS（日期以 1231 结尾）
    annual_mask = df["_date_norm"].str.endswith("1231")
    annual_rows = df[annual_mask]
    if annual_rows.empty:
        return None
    latest_annual = annual_rows.iloc[-1]
    annual_eps = latest_annual.get(eps_col, None)
    if annual_eps is None or pd.isna(annual_eps) or float(annual_eps) <= 0:
        return None

    ttm_eps = float(annual_eps)

    # + 最新 Q1 EPS（日期以 0331 结尾）
    q1_mask = df["_date_norm"].str.endswith("0331")
    if q1_mask.any():
        latest_q1 = df[q1_mask].iloc[-1]
        latest_q1_eps = latest_q1.get(eps_col, None)
        if latest_q1_eps is not None and not pd.isna(latest_q1_eps):
            ttm_eps = ttm_eps + float(latest_q1_eps)

    # - 上年同期 Q1 EPS
    if q1_mask.sum() >= 2:
        prev_q1 = df[q1_mask].iloc[-2]
        prev_q1_eps = prev_q1.get(eps_col, None)
        if prev_q1_eps is not None and not pd.isna(prev_q1_eps):
            ttm_eps = ttm_eps - float(prev_q1_eps)

    return ttm_eps if ttm_eps > 0 else None


def compute_ttm_revenue_per_share(
    income_df: pd.DataFrame,
    revenue_col: str = "OPERATE_INCOME",
    date_col: str = "REPORT_DATE",
    total_shares: float | None = None,
) -> float | None:
    """从利润表累计数据计算 TTM 每股营业收入。

    逻辑与 compute_ttm_eps 相同：Annual - prev_Q1 + curr_Q1，再除以总股本。
    """
    if income_df is None or income_df.empty or revenue_col not in income_df.columns:
        return None
    if date_col not in income_df.columns:
        return None
    if total_shares is None or total_shares <= 0:
        return None

    df = income_df.sort_values(date_col).copy()
    # 统一日期格式：去除连字符，兼容 "2023-12-31" / "20231231" / datetime.date
    # AKShare REPORT_DATE 可能带时间部分如 "2025-12-31 00:00:00"，先截取前10字符
    df["_date_norm"] = df[date_col].astype(str).str[:10].str.replace("-", "", regex=False)

    annual_mask = df["_date_norm"].str.endswith("1231")
    annual_rows = df[annual_mask]
    if annual_rows.empty:
        return None
    latest_annual = annual_rows.iloc[-1]
    annual_rev = latest_annual.get(revenue_col, None)
    if annual_rev is None or pd.isna(annual_rev):
        return None
    ttm_rev = float(annual_rev)

    q1_mask = df["_date_norm"].str.endswith("0331")
    if q1_mask.any():
        latest_q1 = df[q1_mask].iloc[-1]
        q1_rev = latest_q1.get(revenue_col, None)
        if q1_rev is not None and not pd.isna(q1_rev):
            ttm_rev = ttm_rev + float(q1_rev)

    if q1_mask.sum() >= 2:
        prev_q1 = df[q1_mask].iloc[-2]
        prev_q1_rev = prev_q1.get(revenue_col, None)
        if prev_q1_rev is not None and not pd.isna(prev_q1_rev):
            ttm_rev = ttm_rev - float(prev_q1_rev)

    if ttm_rev <= 0:
        return None

    return ttm_rev / total_shares


def compute_valuation_metrics(
    close_price: float | None,
    ttm_eps: float | None = None,
    annual_eps: float | None = None,
    bps: float | None = None,
    total_shares: float | None = None,
    outstanding_shares: float | None = None,
    ttm_revenue_per_share: float | None = None,
) -> dict:
    """从原始数据计算估值指标。

    Returns:
        dict with keys: dynamic_pe, static_pe, pb, ps_ttm, total_mv_yi, circ_mv_yi
        市值单位为亿元，None 表示无法计算
    """
    result = {
        "dynamic_pe": None,
        "static_pe": None,
        "pb": None,
        "ps_ttm": None,
        "total_mv_yi": None,
        "circ_mv_yi": None,
    }

    if close_price is None or close_price <= 0:
        return result

    # 动态 PE(TTM)
    if ttm_eps is not None and ttm_eps > 0:
        result["dynamic_pe"] = round(close_price / ttm_eps, 2)

    # 静态 PE
    if annual_eps is not None and annual_eps > 0:
        result["static_pe"] = round(close_price / annual_eps, 2)

    # PB
    if bps is not None and bps > 0:
        result["pb"] = round(close_price / bps, 2)

    # PS(TTM)
    if ttm_revenue_per_share is not None and ttm_revenue_per_share > 0:
        result["ps_ttm"] = round(close_price / ttm_revenue_per_share, 2)

    # 总市值(亿元)
    if total_shares is not None and total_shares > 0:
        result["total_mv_yi"] = round(close_price * total_shares / 1e8, 2)

    # 流通市值(亿元)
    if outstanding_shares is not None and outstanding_shares > 0:
        result["circ_mv_yi"] = round(close_price * outstanding_shares / 1e8, 2)

    return result
