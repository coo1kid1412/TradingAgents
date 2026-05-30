"""Financial data field mappings and formatting helpers.

Replaces raw DataFrame.to_csv() dumps with curated field extraction,
ensuring LLM receives clear, unambiguous financial data with Chinese labels.

Each mapping: list of (raw_column_name, chinese_label) tuples.
The period/identifier column is listed first in each mapping.
"""

import logging
import re
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Number formatting
# ---------------------------------------------------------------------------

def _fmt_number(val: Any) -> str:
    """Format a numeric value for display; return 'N/A' for missing data."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return "N/A"
    try:
        n = float(val)
        if abs(n) >= 1:
            # Integer-like or large number: comma separators, no decimals for whole numbers
            if n == int(n) and abs(n) < 1e12:
                return f"{int(n):,}"
            # Keep up to 2 decimal places for non-integers
            return f"{n:,.2f}"
        # Small numbers (ratios, EPS, etc.): up to 4 decimal places
        return f"{n:.4f}"
    except (ValueError, TypeError):
        return str(val)


# ---------------------------------------------------------------------------
# AKShare fuzzy column matching
# ---------------------------------------------------------------------------

def _normalize_col(name: str) -> str:
    """Normalize column name for fuzzy matching (strip whitespace, normalize brackets)."""
    s = str(name).strip()
    # Normalize full-width brackets to half-width
    s = s.replace("（", "(").replace("）", ")")
    # Collapse multiple spaces
    s = re.sub(r"\s+", " ", s)
    return s


def _find_col(df: pd.DataFrame, target: str) -> str | None:
    """Find a column in DataFrame by exact then fuzzy match.

    Returns the actual column name from the DataFrame, or None if not found.
    """
    if target in df.columns:
        return target
    # Fuzzy: normalize both sides
    norm_target = _normalize_col(target)
    for col in df.columns:
        if _normalize_col(col) == norm_target:
            return col
    return None


# ---------------------------------------------------------------------------
# Core extraction helper
# ---------------------------------------------------------------------------

def extract_and_format(
    df: pd.DataFrame,
    field_map: list[tuple[str, str]],
    period_col: str = "end_date",
    limit: int = 4,
) -> str:
    """Extract curated fields from a financial DataFrame and format as Markdown table.

    Args:
        df: Raw DataFrame from vendor API.
        field_map: List of (raw_column_name, chinese_label) tuples.
        period_col: Column name that identifies the reporting period.
        limit: Max number of periods (rows) to include.

    Returns:
        Markdown table string with Chinese row labels and periods as columns.
    """
    if df is None or df.empty:
        return "（数据为空）"

    df = df.head(limit)

    # Resolve period column (fuzzy match for AKShare)
    actual_period_col = _find_col(df, period_col)

    # Collect periods (column headers)
    periods = []
    if actual_period_col and actual_period_col in df.columns:
        for v in df[actual_period_col]:
            periods.append(str(v))
    else:
        periods = [f"第{i+1}期" for i in range(len(df))]

    # Build rows: each row = (label, [values...])
    rows: list[tuple[str, list[str]]] = []
    for raw_col, label in field_map:
        if raw_col == period_col:
            continue  # Period column is used as header, not a data row

        actual_col = _find_col(df, raw_col)
        if actual_col is None:
            logger.debug("字段 '%s' 不存在于 DataFrame 中，已跳过", raw_col)
            continue

        values = []
        for v in df[actual_col]:
            values.append(_fmt_number(v))
        rows.append((label, values))

    if not rows:
        available = list(df.columns)[:20]
        return f"（精选字段均未匹配，原始列前20项：{available}）"

    # Build Markdown table
    # Header
    header = "| 指标 | " + " | ".join(periods) + " |"
    separator = "|" + "------|" * (len(periods) + 1)

    # Data rows
    data_lines = []
    for label, values in rows:
        data_lines.append(f"| {label} | " + " | ".join(values) + " |")

    return header + "\n" + separator + "\n" + "\n".join(data_lines)


# ---------------------------------------------------------------------------
# Tushare field mappings
# ---------------------------------------------------------------------------

TUSHARE_FUNDAMENTALS_MAP: list[tuple[str, str]] = [
    ("end_date", "报告期"),
    ("eps", "基本每股收益(EPS)"),
    ("bps", "每股净资产(BPS)"),
    ("cfps", "每股经营现金流(元)"),
    ("roe_waa", "加权净资产收益率ROE(%)"),
    ("npta", "总资产净利润率ROA(%)"),
    ("grossprofit_margin", "销售毛利率(%)"),
    ("netprofit_margin", "销售净利率(%)"),
    ("debt_to_assets", "资产负债率(%)"),
    ("current_ratio", "流动比率"),
    ("quick_ratio", "速动比率"),
    ("or_yoy", "营业收入同比增长率(%)"),
    ("netprofit_yoy", "归母净利润同比增长率(%)"),
    ("dt_netprofit_yoy", "扣非净利润同比增长率(%)"),
    ("ocf_to_profit", "经营现金流/净利润(%)"),
    ("q_sales_yoy", "单季度营收同比增长率(%)"),
    ("q_netprofit_yoy", "单季度归母净利润同比增长率(%)"),
    ("q_gsprofit_margin", "单季度销售毛利率(%)"),
]

TUSHARE_BALANCE_SHEET_MAP: list[tuple[str, str]] = [
    ("end_date", "报告期"),
    ("money_cap", "货币资金"),
    ("accounts_receiv", "应收账款"),
    ("prepayment", "预付款项"),
    ("inventories", "存货"),
    ("total_cur_assets", "流动资产合计"),
    ("fix_assets_total", "固定资产"),
    ("total_nca", "非流动资产合计"),
    ("total_assets", "资产总计"),
    ("st_borr", "短期借款"),
    ("accounts_payable", "应付账款"),
    ("total_cur_liab", "流动负债合计"),
    ("lt_borr", "长期借款"),
    ("total_ncl", "非流动负债合计"),
    ("total_liab", "负债合计"),
    ("total_hldr_eqy_exc_min_int", "归属母公司股东权益"),
    ("minority_int", "少数股东权益"),
    ("total_hldr_eqy_inc_min_int", "股东权益合计"),
]

TUSHARE_CASHFLOW_MAP: list[tuple[str, str]] = [
    ("end_date", "报告期"),
    ("n_cashflow_act", "经营活动现金流量净额"),
    ("n_cashflow_inv_act", "投资活动现金流量净额"),
    ("n_cash_flows_fnc_act", "筹资活动现金流量净额"),
    ("c_fr_sale_sg", "销售商品收到现金"),
    ("c_pay_acq_const_fiolta", "购建固定资产支付现金"),
    ("c_pay_dist_dpcp_int_exp", "分配股利支付现金"),
    ("c_cash_equ_end_period", "期末现金及等价物余额"),
]

TUSHARE_INCOME_MAP: list[tuple[str, str]] = [
    ("end_date", "报告期"),
    ("revenue", "营业收入"),
    ("oper_cost", "营业成本"),
    ("sell_exp", "销售费用"),
    ("admin_exp", "管理费用"),
    ("fin_exp", "财务费用"),
    ("rd_exp", "研发费用"),
    ("operate_profit", "营业利润"),
    ("total_profit", "利润总额"),
    ("n_income", "净利润"),
    ("n_income_attr_p", "归属母公司净利润"),
    ("minority_gain", "少数股东损益"),
    ("basic_eps", "基本每股收益"),
    ("diluted_eps", "稀释每股收益"),
    ("ebit", "息税前利润(EBIT)"),
]


# ---------------------------------------------------------------------------
# AKShare field mappings
# ---------------------------------------------------------------------------

AKSHARE_FUNDAMENTALS_MAP: list[tuple[str, str]] = [
    ("日期", "报告期"),
    ("摊薄每股收益(元)", "基本每股收益(EPS)"),
    ("加权每股收益(元)", "加权每股收益(元)"),
    ("扣除非经常性损益后的每股收益(元)", "扣非每股收益(元)"),
    ("每股净资产_调整后(元)", "每股净资产(BPS)"),
    ("每股经营性现金流(元)", "每股经营现金流(CFPS)"),
    ("每股未分配利润(元)", "每股未分配利润(元)"),
    ("加权净资产收益率(%)", "加权净资产收益率ROE(%)"),
    ("总资产净利润率(%)", "总资产净利润率ROA(%)"),
    ("销售毛利率(%)", "销售毛利率(%)"),
    ("销售净利率(%)", "销售净利率(%)"),
    ("资产负债率(%)", "资产负债率(%)"),
    ("流动比率", "流动比率"),
    ("速动比率", "速动比率"),
    ("净利润增长率(%)", "净利润同比增长率(%)"),
    ("主营业务收入增长率(%)", "营业收入同比增长率(%)"),
    ("营业利润增长率(%)", "营业利润同比增长率(%)"),
    ("总资产周转率(次)", "总资产周转率(次)"),
]

AKSHARE_BALANCE_SHEET_MAP: list[tuple[str, str]] = [
    ("REPORT_DATE", "报告期"),
    ("MONETARYFUNDS", "货币资金"),
    ("ACCOUNTS_RECE", "应收账款"),
    ("PREPAYMENT", "预付款项"),
    ("INVENTORY", "存货"),
    ("TOTAL_CURRENT_ASSETS", "流动资产合计"),
    ("FIXED_ASSET", "固定资产"),
    ("TOTAL_NONCURRENT_ASSETS", "非流动资产合计"),
    ("TOTAL_ASSETS", "资产总计"),
    ("SHORT_LOAN", "短期借款"),
    ("ACCOUNTS_PAYABLE", "应付账款"),
    ("TOTAL_CURRENT_LIAB", "流动负债合计"),
    ("LONG_LOAN", "长期借款"),
    ("TOTAL_NONCURRENT_LIAB", "非流动负债合计"),
    ("TOTAL_LIABILITIES", "负债合计"),
    ("TOTAL_PARENT_EQUITY", "归属母公司股东权益"),
    ("MINORITY_EQUITY", "少数股东权益"),
    ("TOTAL_EQUITY", "股东权益合计"),
]

AKSHARE_CASHFLOW_MAP: list[tuple[str, str]] = [
    ("REPORT_DATE", "报告期"),
    ("NETCASH_OPERATE", "经营活动现金流量净额"),
    ("NETCASH_INVEST", "投资活动现金流量净额"),
    ("NETCASH_FINANCE", "筹资活动现金流量净额"),
    ("SALES_SERVICES", "销售商品收到现金"),
    ("CONSTRUCT_LONG_ASSET", "购建固定资产支付现金"),
    ("END_CCE", "期末现金及等价物余额"),
]

AKSHARE_INCOME_MAP: list[tuple[str, str]] = [
    ("REPORT_DATE", "报告期"),
    ("TOTAL_OPERATE_INCOME", "营业总收入"),
    ("OPERATE_INCOME", "营业收入"),
    ("TOTAL_OPERATE_COST", "营业总成本"),
    ("OPERATE_COST", "营业成本"),
    ("SALE_EXPENSE", "销售费用"),
    ("MANAGE_EXPENSE", "管理费用"),
    ("RESEARCH_EXPENSE", "研发费用"),
    ("FINANCE_EXPENSE", "财务费用"),
    ("OPERATE_PROFIT", "营业利润"),
    ("TOTAL_PROFIT", "利润总额"),
    ("NETPROFIT", "净利润"),
    ("PARENT_NETPROFIT", "归属母公司净利润"),
    ("MINORITY_INTEREST", "少数股东损益"),
    ("BASIC_EPS", "基本每股收益"),
    ("DILUTED_EPS", "稀释每股收益"),
]


# ---------------------------------------------------------------------------
# YFinance field mappings
# ---------------------------------------------------------------------------

YFINANCE_BALANCE_SHEET_MAP: list[tuple[str, str]] = [
    ("Total Assets", "资产总计"),
    ("Current Assets", "流动资产合计"),
    ("Cash And Cash Equivalents", "货币资金"),
    ("Receivables", "应收账款"),
    ("Inventory", "存货"),
    ("Total Liabilities Net Minority Interest", "负债合计"),
    ("Current Liabilities", "流动负债合计"),
    ("Long Term Debt", "长期借款"),
    ("Total Equity Gross Minority Interest", "股东权益合计"),
    ("Common Stock Equity", "归属母公司股东权益"),
]

YFINANCE_CASHFLOW_MAP: list[tuple[str, str]] = [
    ("Operating Cash Flow", "经营活动现金流量净额"),
    ("Investing Cash Flow", "投资活动现金流量净额"),
    ("Financing Cash Flow", "筹资活动现金流量净额"),
    ("End Cash Position", "期末现金及等价物余额"),
    ("Capital Expenditure", "资本支出"),
    ("Free Cash Flow", "自由现金流"),
]

YFINANCE_INCOME_MAP: list[tuple[str, str]] = [
    ("Total Revenue", "营业收入"),
    ("Cost Of Revenue", "营业成本"),
    ("Gross Profit", "毛利润"),
    ("Operating Income", "营业利润"),
    ("Net Income", "净利润"),
    ("Net Income Common Stockholders", "归属母公司净利润"),
    ("EBIT", "息税前利润(EBIT)"),
    ("EBITDA", "EBITDA"),
    ("Basic EPS", "基本每股收益"),
    ("Diluted EPS", "稀释每股收益"),
    ("Selling General And Administration", "销售及管理费用"),
    ("Research And Development", "研发费用"),
]


# ---------------------------------------------------------------------------
# YFinance helper (transposed DataFrame)
# ---------------------------------------------------------------------------

def extract_yfinance_table(
    df: pd.DataFrame,
    field_map: list[tuple[str, str]],
) -> str:
    """Extract curated fields from a YFinance financial DataFrame.

    YFinance returns DataFrames with index=indicator names, columns=dates.
    This function selects desired rows, transposes, then formats as Markdown table.

    Args:
        df: Raw YFinance DataFrame (rows=indicators, cols=dates).
        field_map: List of (yfinance_row_label, chinese_label) tuples.

    Returns:
        Markdown table string.
    """
    if df is None or df.empty:
        return "（数据为空）"

    # Collect periods from column names (dates)
    periods = [str(col.date()) if hasattr(col, "date") else str(col) for col in df.columns]

    # Build rows
    rows: list[tuple[str, list[str]]] = []
    for raw_label, cn_label in field_map:
        if raw_label in df.index:
            values = [_fmt_number(v) for v in df.loc[raw_label]]
            rows.append((cn_label, values))

    if not rows:
        available = list(df.index)[:20]
        return f"（精选字段均未匹配，原始行前20项：{available}）"

    # Build Markdown table
    header = "| 指标 | " + " | ".join(periods) + " |"
    separator = "|" + "------|" * (len(periods) + 1)
    data_lines = [f"| {label} | " + " | ".join(values) + " |" for label, values in rows]

    return header + "\n" + separator + "\n" + "\n".join(data_lines)


# ---------------------------------------------------------------------------
# Alpha Vantage field mappings
# ---------------------------------------------------------------------------

ALPHAVANTAGE_FUNDAMENTALS_MAP: list[tuple[str, str]] = [
    ("MarketCapitalization", "总市值"),
    ("PERatio", "PE(TTM)"),
    ("ForwardPE", "Forward PE"),
    ("PEGRatio", "PEG Ratio"),
    ("PriceToBookRatio", "PB"),
    ("PriceToSalesRatioTTM", "PS(TTM)"),
    ("EVToRevenue", "EV/Revenue"),
    ("EVToEBITDA", "EV/EBITDA"),
    ("EPS", "每股收益(EPS)"),
    ("BookValue", "每股净资产(BPS)"),
    ("DividendPerShare", "每股股利"),
    ("DividendYield", "股息率(%)"),
    ("ProfitMargin", "净利润率(%)"),
    ("OperatingMarginTTM", "营业利润率(%)"),
    ("ReturnOnAssetsTTM", "ROA(%)"),
    ("ReturnOnEquityTTM", "ROE(%)"),
    ("RevenueTTM", "营业收入(TTM)"),
    ("GrossProfitTTM", "毛利润(TTM)"),
    ("EBITDA", "EBITDA"),
    ("QuarterlyEarningsGrowthYOY", "季度盈利同比增长率(%)"),
    ("QuarterlyRevenueGrowthYOY", "季度营收同比增长率(%)"),
    ("Beta", "Beta"),
    ("52WeekHigh", "52周最高价"),
    ("52WeekLow", "52周最低价"),
    ("50DayMovingAverage", "50日均线"),
    ("200DayMovingAverage", "200日均线"),
    ("SharesOutstanding", "流通股数"),
    ("AnalystTargetPrice", "分析师目标价"),
]

ALPHAVANTAGE_BALANCE_SHEET_MAP: list[tuple[str, str]] = [
    ("fiscalDateEnding", "报告期"),
    ("totalAssets", "资产总计"),
    ("totalCurrentAssets", "流动资产合计"),
    ("cashAndCashEquivalentsAtCarryingValue", "货币资金"),
    ("cashAndShortTermInvestments", "现金及短期投资"),
    ("currentNetReceivables", "应收账款"),
    ("inventory", "存货"),
    ("totalNonCurrentAssets", "非流动资产合计"),
    ("propertyPlantEquipment", "固定资产"),
    ("totalLiabilities", "负债合计"),
    ("totalCurrentLiabilities", "流动负债合计"),
    ("currentAccountsPayable", "应付账款"),
    ("longTermDebt", "长期借款"),
    ("totalShareholderEquity", "股东权益合计"),
    ("commonStock", "普通股"),
    ("retainedEarnings", "留存收益"),
]

ALPHAVANTAGE_CASHFLOW_MAP: list[tuple[str, str]] = [
    ("fiscalDateEnding", "报告期"),
    ("operatingCashflow", "经营活动现金流量净额"),
    ("cashflowFromInvestment", "投资活动现金流量净额"),
    ("cashflowFromFinancing", "筹资活动现金流量净额"),
    ("capitalExpenditures", "资本支出"),
    ("dividendPayout", "股利支付"),
    ("netIncome", "净利润"),
    ("depreciationDepletionAndAmortization", "折旧摊销"),
    ("changeInCashAndCashEquivalents", "现金及等价物变动"),
]

ALPHAVANTAGE_INCOME_MAP: list[tuple[str, str]] = [
    ("fiscalDateEnding", "报告期"),
    ("totalRevenue", "营业总收入"),
    ("totalOperatingExpense", "营业总费用"),
    ("costOfRevenue", "营业成本"),
    ("grossProfit", "毛利润"),
    ("operatingIncome", "营业利润"),
    ("incomeBeforeTax", "税前利润"),
    ("incomeTaxExpense", "所得税费用"),
    ("netIncome", "净利润"),
    ("netIncomeFromContinuingOperations", "持续经营净利润"),
    ("ebitda", "EBITDA"),
    ("eps", "每股收益(EPS)"),
    ("researchAndDevelopment", "研发费用"),
    ("sellingGeneralAndAdministrative", "销售及管理费用"),
]


# ---------------------------------------------------------------------------
# Alpha Vantage helper (JSON-based)
# ---------------------------------------------------------------------------

def extract_alphavantage_table(
    data: dict | str,
    field_map: list[tuple[str, str]],
    report_key: str = "quarterlyReports",
    limit: int = 4,
) -> str:
    """Extract curated fields from Alpha Vantage JSON response.

    Args:
        data: Raw API response (JSON string or parsed dict).
        field_map: List of (json_key, chinese_label) tuples.
        report_key: Key for report list ('quarterlyReports' or 'annualReports').
        limit: Max number of reports to include.

    Returns:
        Markdown table string.
    """
    import json as _json

    # Parse JSON string if needed
    if isinstance(data, str):
        try:
            data = _json.loads(data)
        except (_json.JSONDecodeError, TypeError):
            return "（JSON 解析失败）"

    if not isinstance(data, dict):
        return "（数据格式异常）"

    reports = data.get(report_key, [])
    if not reports:
        # Fallback to annual if quarterly not available
        reports = data.get("annualReports", [])

    if not reports:
        return "（无报告数据）"

    reports = reports[:limit]

    # Collect periods
    periods = []
    for r in reports:
        periods.append(r.get("fiscalDateEnding", "N/A"))

    # Build rows
    rows: list[tuple[str, list[str]]] = []
    for json_key, cn_label in field_map:
        if json_key == "fiscalDateEnding":
            continue  # Used as header
        values = []
        for r in reports:
            val = r.get(json_key)
            values.append(_fmt_number(val))
        # Only include row if at least one non-N/A value
        if any(v != "N/A" for v in values):
            rows.append((cn_label, values))

    if not rows:
        return "（精选字段均未匹配）"

    # Build Markdown table
    header = "| 指标 | " + " | ".join(periods) + " |"
    separator = "|" + "------|" * (len(periods) + 1)
    data_lines = [f"| {label} | " + " | ".join(values) + " |" for label, values in rows]

    return header + "\n" + separator + "\n" + "\n".join(data_lines)


def extract_alphavantage_overview(
    data: dict | str,
    field_map: list[tuple[str, str]],
) -> str:
    """Extract curated fields from Alpha Vantage OVERVIEW response.

    Args:
        data: Raw API response (JSON string or parsed dict).
        field_map: List of (json_key, chinese_label) tuples.

    Returns:
        Key-value formatted string.
    """
    import json as _json

    # Parse JSON string if needed
    if isinstance(data, str):
        try:
            data = _json.loads(data)
        except (_json.JSONDecodeError, TypeError):
            return "（JSON 解析失败）"

    if not isinstance(data, dict):
        return "（数据格式异常）"

    lines = []
    for json_key, cn_label in field_map:
        val = data.get(json_key)
        if val is not None and val != "None" and val != "":
            lines.append(f"{cn_label}: {val}")

    if not lines:
        return "（无可用数据）"

    return "\n".join(lines)
