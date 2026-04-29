import json
from datetime import datetime

from .alpha_vantage_common import _make_api_request
from .financial_field_maps import (
    extract_alphavantage_overview,
    extract_alphavantage_table,
    ALPHAVANTAGE_FUNDAMENTALS_MAP,
    ALPHAVANTAGE_BALANCE_SHEET_MAP,
    ALPHAVANTAGE_CASHFLOW_MAP,
    ALPHAVANTAGE_INCOME_MAP,
)


def _filter_reports_by_date(data: dict, curr_date: str) -> dict:
    """Filter annualReports/quarterlyReports to exclude entries after curr_date.

    Prevents look-ahead bias by removing fiscal periods that end after
    the simulation's current date.
    """
    if not curr_date or not isinstance(data, dict):
        return data
    for key in ("annualReports", "quarterlyReports"):
        if key in data:
            data[key] = [
                r for r in data[key]
                if r.get("fiscalDateEnding", "") <= curr_date
            ]
    return data


def _parse_json(data) -> dict | None:
    """Parse JSON string to dict if needed."""
    if isinstance(data, dict):
        return data
    if isinstance(data, str):
        try:
            return json.loads(data)
        except (json.JSONDecodeError, TypeError):
            return None
    return None


def get_fundamentals(ticker: str, curr_date: str = None) -> str:
    """Retrieve curated fundamental data for a given ticker symbol using Alpha Vantage.

    Returns key-value formatted string with Chinese labels for LLM consumption.
    """
    params = {"symbol": ticker}
    raw = _make_api_request("OVERVIEW", params)

    # Extract curated fields with Chinese labels
    result = extract_alphavantage_overview(raw, ALPHAVANTAGE_FUNDAMENTALS_MAP)

    header = (
        f"# Company Fundamentals for {ticker.upper()}\n"
        f"# Source: Alpha Vantage\n"
        f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    )
    return header + result


def get_balance_sheet(ticker: str, freq: str = "quarterly", curr_date: str = None):
    """Retrieve curated balance sheet data using Alpha Vantage."""
    raw = _make_api_request("BALANCE_SHEET", {"symbol": ticker})
    parsed = _parse_json(raw)
    if parsed is None:
        return f"No balance sheet data found for symbol '{ticker}'"

    parsed = _filter_reports_by_date(parsed, curr_date)

    report_key = "quarterlyReports" if freq == "quarterly" else "annualReports"
    limit = 4 if freq == "quarterly" else 2
    table = extract_alphavantage_table(parsed, ALPHAVANTAGE_BALANCE_SHEET_MAP, report_key=report_key, limit=limit)

    header = (
        f"# Balance Sheet for {ticker.upper()} ({freq})\n"
        f"# Source: Alpha Vantage\n"
        f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    )
    return header + table


def get_cashflow(ticker: str, freq: str = "quarterly", curr_date: str = None):
    """Retrieve curated cash flow data using Alpha Vantage."""
    raw = _make_api_request("CASH_FLOW", {"symbol": ticker})
    parsed = _parse_json(raw)
    if parsed is None:
        return f"No cash flow data found for symbol '{ticker}'"

    parsed = _filter_reports_by_date(parsed, curr_date)

    report_key = "quarterlyReports" if freq == "quarterly" else "annualReports"
    limit = 4 if freq == "quarterly" else 2
    table = extract_alphavantage_table(parsed, ALPHAVANTAGE_CASHFLOW_MAP, report_key=report_key, limit=limit)

    header = (
        f"# Cash Flow for {ticker.upper()} ({freq})\n"
        f"# Source: Alpha Vantage\n"
        f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    )
    return header + table


def get_income_statement(ticker: str, freq: str = "quarterly", curr_date: str = None):
    """Retrieve curated income statement data using Alpha Vantage."""
    raw = _make_api_request("INCOME_STATEMENT", {"symbol": ticker})
    parsed = _parse_json(raw)
    if parsed is None:
        return f"No income statement data found for symbol '{ticker}'"

    parsed = _filter_reports_by_date(parsed, curr_date)

    report_key = "quarterlyReports" if freq == "quarterly" else "annualReports"
    limit = 4 if freq == "quarterly" else 2
    table = extract_alphavantage_table(parsed, ALPHAVANTAGE_INCOME_MAP, report_key=report_key, limit=limit)

    header = (
        f"# Income Statement for {ticker.upper()} ({freq})\n"
        f"# Source: Alpha Vantage\n"
        f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    )
    return header + table
