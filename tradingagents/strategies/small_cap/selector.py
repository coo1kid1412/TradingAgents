"""Pure point-in-time screen. Unknowns are never converted into exclusions."""

from datetime import date, datetime, time, timedelta
import math
from zoneinfo import ZoneInfo

from .config import Config, RULE_ASSUMPTIONS, digest

SHANGHAI = ZoneInfo("Asia/Shanghai")


def parse_date(value):
    value = str(value)
    return datetime.strptime(value, "%Y%m%d").date() if len(value) == 8 else date.fromisoformat(value)


def parse_instant(value):
    instant = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if instant.tzinfo is None or instant.utcoffset() is None:
        raise ValueError("timezone is required")
    return instant.astimezone(SHANGHAI)


def number(value):
    if isinstance(value, bool) or value is None:
        return None
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def _publication(row):
    # A date without publication time becomes visible only on the next day.
    values = [row[key] for key in ("ann_date", "f_ann_date") if row.get(key)]
    if not values:
        raise ValueError("missing publication date")
    return max(parse_instant(value) if "T" in str(value) else datetime.combine(
        parse_date(value) + timedelta(days=1), time.min, SHANGHAI) for value in values)


def _annual(rows, cutoff, data_date, config):
    visible, uncertain_periods = [], []
    for row in rows:
        try:
            end = parse_date(row.get("end_date"))
            if (end.month, end.day) != (12, 31) or end > data_date:
                continue
            if str(row.get("report_type")) in {"2", "3", "6", "7", "8", "9", "10", "11", "12"}:
                continue
        except (TypeError, ValueError):
            continue
        try:
            observed = parse_instant(row.get("observed_at"))
            if observed > cutoff:
                continue
            published = _publication(row)
            if published > cutoff:
                continue
            if str(row.get("report_type")) not in {"1", "4", "5"}:
                uncertain_periods.append(end)
                continue
            visible.append((end, published, observed, row))
        except (TypeError, ValueError):
            uncertain_periods.append(end)
    if not visible:
        return "UNKNOWN", "annual_financial_unavailable", None
    newest_period = max(item[0] for item in visible)
    if any(period >= newest_period for period in uncertain_periods):
        return "UNKNOWN", "annual_financial_provenance_unknown", None
    latest = [item for item in visible if item[0] == newest_period]
    published = max(item[1] for item in latest)
    latest = [item for item in latest if item[1] == published]
    # Tushare type 1 is its latest consolidated report; type 5 preserves the
    # pre-adjustment original. HTTP request order is not a financial revision.
    priority = {"1": 3, "4": 2, "5": 1}
    best = max(priority[str(item[3]["report_type"])] for item in latest)
    latest = [item for item in latest if priority[str(item[3]["report_type"])] == best]
    end, published, observed, row = max(latest, key=lambda item: str(item[3].get("revision_id", "")))
    tied = [item[3] for item in latest]
    facts = {(number(item.get("n_income_attr_p")), number(item.get("revenue"))) for item in tied}
    if len(facts) > 1:
        return "UNKNOWN", "conflicting_financial_versions", None
    profit, revenue = number(row.get("n_income_attr_p")), number(row.get("revenue"))
    if (data_date - end).days > config.maximum_annual_age_days:
        return "UNKNOWN", "annual_financial_stale", None
    if profit is None or revenue is None:
        return "UNKNOWN", "annual_financial_incomplete", None
    if profit <= 0 or revenue <= config.minimum_revenue_yuan:
        return "EXCLUDED", "annual_profit_or_revenue_threshold", None
    return "ELIGIBLE", "", {"financial_period": end.isoformat(),
        "financial_available_at": max(published, observed).isoformat(),
        "profit": profit, "revenue": revenue, "financial_revision_id": row.get("revision_id")}


def _complete_primary_census(pages):
    rows = []
    for status in ("L", "D", "P"):
        group = [page for page in pages if page["params"].get("list_status") == status]
        if not group:
            raise ValueError("missing primary status census")
        group.sort(key=lambda page: page["params"]["offset"])
        offset, page_limit = 0, None
        seen = set()
        for index, page in enumerate(group):
            params, items = page["params"], page["rows"]
            if (set(params) - {"list_status", "exchange", "fields", "limit", "offset"}
                    or params.get("exchange", "") != ""
                    or type(params.get("offset")) is not int or params["offset"] != offset
                    or type(params.get("limit")) is not int or not 0 < params["limit"] <= 5000
                    or (page_limit is not None and params["limit"] != page_limit)
                    or not isinstance(items, list) or len(items) > params["limit"]):
                raise ValueError("filtered or noncontiguous primary census")
            page_limit = params["limit"]
            terminal = len(items) < page_limit
            if terminal != (index == len(group) - 1):
                raise ValueError("missing terminal-page evidence")
            for item in items:
                code = item["ts_code"]
                if code in seen or item.get("list_status") != status:
                    raise ValueError("duplicate or inconsistent primary identity")
                seen.add(code)
                rows.append(item)
            offset += page_limit
    return rows


def _outside_official_universe(snapshot, cutoff, data_date):
    """Negative membership needs two frozen censuses, not an empty quote lookup.

    This is current forward identity verification only. The exchange's complete
    A-share master must contain every primary active SZSE CNY identity, including
    ST/suspended names. Any captured positive identity/feed defeats exclusion.
    """
    raw = snapshot.get("raw") or {}
    pages = raw.get("szse_official_a_share_master") or []
    primary = raw.get("stock_basic") or []
    try:
        if len(pages) != 1:
            return set()
        primary_rows = _complete_primary_census(primary)
        page = pages[0]
        if (page.get("source_url") != "https://www.szse.cn/api/report/ShowReport"
                or page.get("params") != {"SHOWTYPE": "xlsx", "CATALOGID": "1110", "TABKEY": "tab1"}
                or page.get("mode") != "forward" or page.get("data_date") != data_date.isoformat()
                or page.get("scope") != "current_as_of_listed_a_shares"
                or page.get("coverage_complete") is not True or page.get("cross_check_complete") is not True):
            return set()
        for observed in [page["observed_at"]] + [item["observed_at"] for item in primary]:
            instant = parse_instant(observed)
            if instant.date() != data_date or instant > cutoff:
                return set()
        rows = page["rows"]
        if page.get("rows_hash") != digest(rows):
            return set()
        codes = {item["ts_code"] for item in rows}
        if not codes or len(codes) != len(rows) or any(
            len(code) != 9 or not code[:6].isdigit() or not code.endswith(".SZ") for code in codes):
            return set()
        if any(item.get("market") not in {"主板", "创业板"} or not parse_date(item["list_date"]) <= data_date
               for item in rows):
            return set()
        known = {item["ts_code"] for item in primary_rows
                 if item.get("exchange") == "SZSE" and item.get("curr_type") == "CNY"
                 and item["list_status"] in {"L", "P"} and parse_date(item["list_date"]) <= data_date}
        if not known or not known <= codes:
            return set()
        positive = {item["ts_code"] for name in ("stock_basic", "bak_basic", "daily", "daily_basic", "stock_st",
                                                  "suspend_d", "stk_limit")
                    for p in raw.get(name, []) for item in p["rows"]}
        return {item["ts_code"] for item in snapshot.get("records", []) if item.get("financials")
                and str(item.get("ts_code", "")).endswith(".SZ")
                and item["ts_code"] not in positive and item["ts_code"] not in codes}
    except (KeyError, TypeError, ValueError):
        return set()


def _evaluate(row, cutoff, data_date, config, outside_verified=False):
    unknown = []
    def missing(reason):
        unknown.append(reason)

    try:
        if parse_instant(row.get("observed_at")) > cutoff:
            return "UNKNOWN", "identity_observed_after_cutoff", None
    except (TypeError, ValueError):
        return "UNKNOWN", "identity_observation_unknown", None
    if outside_verified:
        return "EXCLUDED", "outside_verified_listed_security_universe", None
    for key, allowed in (("exchange", {"SSE", "SZSE"}), ("market", {"主板"}),
                         ("curr_type", {"CNY"})):
        if not row.get(key):
            missing("missing_" + key)
        elif row[key] not in allowed:
            return "EXCLUDED", "ineligible_" + key, None
    code = str(row.get("ts_code", ""))
    if not (code.startswith(("600", "601", "603", "605", "000", "001", "002", "003"))
            and code.endswith((".SH", ".SZ"))):
        return "EXCLUDED", "not_ordinary_mainboard_share", None
    try:
        if row.get("delist_date") and parse_date(row["delist_date"]) <= data_date:
            return "EXCLUDED", "delisted", None
        listing_days = (data_date - parse_date(row.get("list_date"))).days
        if listing_days < config.minimum_listing_days:
            return "EXCLUDED", "listing_age", None
    except (TypeError, ValueError):
        listing_days = None
        missing("listing_dates_unknown")
    try:
        effective = parse_date(row.get("status_effective_date"))
        if effective != data_date:
            missing("security_status_not_current")
        elif row.get("security_status") in {"ST", "DELISTING"}:
            return "EXCLUDED", "special_security_status", None
        elif row.get("security_status") != "NORMAL":
            missing("security_status_unknown")
    except (TypeError, ValueError):
        missing("security_status_date_unknown")
    try:
        quote_current = parse_date(row.get("trade_date")) == data_date
    except (TypeError, ValueError):
        quote_current = False
    try:
        suspension_current = parse_date(row.get("suspension_date", row.get("trade_date"))) == data_date
    except (TypeError, ValueError):
        suspension_current = False
    if ((row.get("suspended") is True and suspension_current)
            or (number(row.get("volume")) == 0 and quote_current)):
        return "EXCLUDED", "suspended_or_zero_volume", None
    if row.get("suspended") is not False:
        missing("suspension_unknown")
    volume = number(row.get("volume"))
    if volume is None or volume < 0:
        missing("volume_unknown")
    try:
        cap_current = parse_date(row.get("valuation_date", row.get("trade_date"))) == data_date
    except (TypeError, ValueError):
        cap_current = False
    if not quote_current:
        missing("quote_not_current")
    if not cap_current:
        missing("valuation_not_current")
    close, cap = number(row.get("close")), number(row.get("total_market_cap_yuan"))
    if cap is None or cap <= 0:
        missing("market_cap_unknown")
    elif cap_current and not config.minimum_cap_yuan <= cap <= config.maximum_cap_yuan:
        return "EXCLUDED", "market_cap_threshold", None
    if close is None or close <= 0:
        missing("close_unknown")
    elif quote_current and close > config.maximum_price:
        return "EXCLUDED", "price_threshold", None
    state, reason, facts = _annual(row.get("financials") or [], cutoff, data_date, config)
    if state == "EXCLUDED":
        return state, reason, None
    if state == "UNKNOWN":
        missing(reason)
    if unknown:
        return "UNKNOWN", ";".join(unknown), None
    selected = {key: row.get(key) for key in ("ts_code", "name", "industry", "industry_basis")}
    selected.update(exchange=row["exchange"], board=row["market"])
    selected.update(total_market_cap_yuan=cap, close=close, listing_days=listing_days,
                    **facts)
    selected["evidence"] = {"quote_trade_date": data_date.isoformat(),
                            "observed_at": row["observed_at"],
                            "financial_revision_id": facts["financial_revision_id"]}
    selected["flags"] = []
    selected["rule_checks"] = {key: "PASS" for key in (
        "universe", "status", "listing_age", "total_cap", "annual_profit", "annual_revenue",
        "annual_freshness", "close", "tradeability", "point_in_time")}
    for key, flag in (("up_limit", "limit_up_at_close"), ("down_limit", "limit_down_at_close")):
        limit = number(row.get(key))
        if limit is not None and abs(close - limit) < 0.005:
            selected["flags"].append(flag)
    return "ELIGIBLE", "all_rules_passed", selected


def select_top10(snapshot, config=Config()):
    result = {"schema_version": "small-cap-result-v1", "rule_version": config.rule_version,
              "config_hash": config.config_hash, "timezone": config.timezone,
              "mode": snapshot.get("mode"), "rule_assumptions": dict(RULE_ASSUMPTIONS),
              "data_date": snapshot.get("data_date"), "as_of_at": snapshot.get("as_of_at"),
              "target_trade_date": snapshot.get("target_trade_date"),
              "status": "BLOCKED", "ranking_complete": False, "top10": [],
              "exclusions": [], "coverage": snapshot.get("coverage", {}),
              "warnings": [dict(w) if isinstance(w, dict) else w for w in snapshot.get("warnings", [])],
              "regime": "UNKNOWN", "strategy_mode": "RESEARCH_CANDIDATES",
              "new_position_policy": "NOT_EVALUATED", "rebalance_due": False,
              "changes": {}, "delivery_status": "not_sent"}
    try:
        data_date, target = parse_date(snapshot["data_date"]), parse_date(snapshot["target_trade_date"])
        cutoff = parse_instant(snapshot["as_of_at"])
        calendar = {parse_date(row["cal_date"]): int(row["is_open"]) for row in snapshot["calendar"]}
        next_days = sorted(day for day, opened in calendar.items() if opened and day > data_date)
        if (snapshot.get("schema_version") != "small-cap-snapshot-v1" or snapshot.get("mode") != "forward"
                or calendar.get(data_date) != 1 or not next_days or next_days[0] != target
                or cutoff.date() < data_date or cutoff >= datetime.combine(target, time.min, SHANGHAI)
                or (cutoff.date() == data_date and cutoff.time() < time(15))):
            raise ValueError("invalid snapshot cutoff, calendar or provenance")
        result["rebalance_due"] = target.weekday() == 1
        if target.month in {1, 4}:
            result.update(strategy_mode="SEASONAL_STOCK_AVOID", new_position_policy="BLOCK")
    except (KeyError, TypeError, ValueError):
        result["warnings"].append("snapshot_calendar_or_point_in_time_invalid")
        result["selection_hash"] = digest({"status": "BLOCKED", "data_date": result["data_date"],
                                            "config": config.config_hash})
        return result

    eligible, unknown, seen = [], [], set()
    outside = _outside_official_universe(snapshot, cutoff, data_date)
    result["identity_verification"] = {"official_szse_resolved_count": len(outside),
                                      "scope": "current_forward_as_of_only"}
    coverage_complete = snapshot.get("coverage", {}).get("complete") is True
    for row in snapshot.get("records", []):
        code = str(row.get("ts_code", ""))
        if code in seen:
            coverage_complete = False
            result["warnings"].append("duplicate_security:" + code)
            continue
        seen.add(code)
        state, reason, selected = _evaluate(row, cutoff, data_date, config, code in outside)
        if state == "ELIGIBLE":
            eligible.append(selected)
        else:
            excluded = {"ts_code": code, "status": state, "reason": reason}
            if code in outside:
                excluded["evidence_ref"] = ["raw.stock_basic", "raw.szse_official_a_share_master"]
                excluded["identity_scope"] = "current_forward_as_of_only_not_historical_or_T_close"
            result["exclusions"].append(excluded)
            if state == "UNKNOWN":
                cap = number(row.get("total_market_cap_yuan"))
                try:
                    if (parse_instant(row.get("observed_at")) > cutoff or
                            parse_date(row.get("valuation_date", row.get("trade_date"))) != data_date):
                        cap = None
                except (ValueError, TypeError):
                    cap = None
                unknown.append((cap, code))
    eligible.sort(key=lambda row: (row["total_market_cap_yuan"], row["ts_code"]))
    top = eligible[:config.top_n]
    boundary = (top[-1]["total_market_cap_yuan"], top[-1]["ts_code"]) if len(top) == config.top_n else None
    blocking_unknowns = [code for cap, code in unknown
                         if cap is None or cap <= 0 or boundary is None or (cap, code) <= boundary]
    if blocking_unknowns or not coverage_complete:
        result["warnings"].append("ranking_incomplete")
        result["blocking_unknowns"] = blocking_unknowns
    else:
        result.update(status="SUCCESS" if len(top) == config.top_n else "INSUFFICIENT",
                      ranking_complete=True, top10=[dict(row, rank=i + 1) for i, row in enumerate(top)])
        if unknown:
            result["warnings"].append("unknown_candidates_proven_behind_top10")
    result["selection_hash"] = digest({key: result[key] for key in
        ("rule_version", "config_hash", "data_date", "target_trade_date", "status", "top10",
         "strategy_mode", "new_position_policy")})
    return result
