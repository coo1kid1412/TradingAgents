"""Bounded, current-day Tushare capture. No cache, dotenv, LLM or delivery calls."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
from collections import defaultdict
from datetime import date, datetime, time, timedelta, timezone
from time import monotonic
from typing import Any
from zoneinfo import ZoneInfo

import requests

from .official_master import OfficialMasterError, fetch_szse_a_share_master


SHANGHAI = ZoneInfo("Asia/Shanghai")
_API_LIMITS = {
    "trade_cal": 6000, "stock_basic": 5000, "bak_basic": 5000,
    "daily": 6000, "daily_basic": 6000, "stock_st": 1000,
    "suspend_d": 5000, "stk_limit": 5000, "income_vip": 5000,
}
_LOGGER = logging.getLogger(__name__)
_LOGGER.setLevel(logging.INFO)
if not _LOGGER.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(levelname)s %(name)s %(message)s"))
    _LOGGER.addHandler(_handler)
_LOGGER.propagate = False

_MESSAGES = {
    "TOKEN_MISSING": "TUSHARE_TOKEN is required; load the environment in the caller.",
    "INVALID_CLOCK": "Clock must return a timezone-aware datetime.",
    "INVALID_DATE": "Expected an ISO or YYYYMMDD date.",
    "FORWARD_DATE_REQUIRED": "Only the current Shanghai date supports forward capture.",
    "BEFORE_CLOSE": "Forward capture requires the Shanghai market close at 15:00.",
    "CUTOFF_EXPIRED": "Capture crossed the next trading day's Shanghai midnight.",
    "CLOCK_REGRESSION": "Capture timestamps are not monotonic.",
    "CALENDAR_INCOMPLETE": "Trading calendars do not confirm today and the next open day.",
    "TRANSPORT_ERROR": "Tushare transport failed after bounded attempts.",
    "COLLECT_TIMEOUT": "The monotonic time budget for this capture was exhausted.",
    "AUTHENTICATION_FAILED": "Tushare authentication failed.",
    "PERMISSION_DENIED": "Tushare denied access to a required dataset.",
    "RATE_LIMIT": "Tushare rate limit reached; capture was not completed.",
    "API_ERROR": "Tushare returned an unsuccessful response.",
    "MALFORMED_RESPONSE": "Tushare returned an invalid structured response.",
    "PAGINATION_REPEATED": "Tushare repeated or overlapped a page.",
    "PAGINATION_TRUNCATED": "Tushare pagination reached its bounded page limit.",
    "DATE_MISMATCH": "A required response does not match its requested effective date.",
    "EMPTY_DATA": "A required market-wide dataset was empty.",
    "AMBIGUOUS_DATA": "Conflicting rows prevent a reliable security snapshot.",
}


class DataSourceError(RuntimeError):
    """Only static, credential-free messages cross the provider boundary."""

    def __init__(self, code: str, message: str | None = None, endpoint: str | None = None):
        self.code = code if code in _MESSAGES else "API_ERROR"
        self.safe_code = self.code
        self.endpoint = endpoint if endpoint in _API_LIMITS else None
        # Do not trust provider exception strings, even when supplied as a message.
        self.message = _MESSAGES[self.code]
        super().__init__(f"{self.code}: {self.message}")


def _date(value: Any) -> date | None:
    if value is None or value == "":
        return None
    try:
        text = str(value).strip()
        if len(text) == 8 and text.isdigit():
            return datetime.strptime(text, "%Y%m%d").date()
        return date.fromisoformat(text)
    except (ValueError, TypeError):
        return None


def _number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (ValueError, TypeError, OverflowError):
        return None


def _hash(value: Any) -> str:
    body = json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False,
                      separators=(",", ":"))
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def _business_code(message: str, code: Any = None) -> str:
    text = message.lower()
    if any(word in text for word in ("token", "authentication", "\u4ee4\u724c")):
        return "AUTHENTICATION_FAILED"
    if any(word in text for word in ("\u6bcf\u5206\u949f", "\u9891\u7387", "\u6b21\u6570", "rate limit")):
        return "RATE_LIMIT"
    if str(code) == "2002" or any(word in text for word in (
            "permission", "\u6743\u9650", "\u79ef\u5206")):
        return "PERMISSION_DENIED"
    return "API_ERROR"


class _HTTPSAPI:
    def __init__(self, token: str):
        self.token = token
        self.deadline = None

    def query(self, name: str, **kwargs: Any) -> dict:
        params = dict(kwargs)
        fields = params.pop("fields", "")
        timeout = (5, 30)
        if self.deadline is not None:
            remaining = self.deadline - monotonic()
            if remaining <= 0:
                raise DataSourceError("COLLECT_TIMEOUT", endpoint=name)
            timeout = (min(5, remaining), min(30, remaining))
        response = requests.post(
            "https://api.tushare.pro",
            json={"api_name": name, "token": self.token, "params": params,
                  "fields": fields},
            timeout=timeout,
            allow_redirects=False,
        )
        if response.status_code in (401, 403, 429):
            code = {401: "AUTHENTICATION_FAILED", 403: "PERMISSION_DENIED",
                    429: "RATE_LIMIT"}[response.status_code]
            raise DataSourceError(code)
        response.raise_for_status()
        try:
            return response.json()
        except ValueError:
            raise DataSourceError("MALFORMED_RESPONSE") from None


class TushareSource:
    """Collect exact-day evidence; leave eligibility and PIT selection to selector.

    ``api`` may expose ``query(name, **params)`` returning rows, a DataFrame, or
    the documented Tushare JSON envelope. ``clock`` is an aware datetime callable.
    Empty successful financial/status responses are not fabricated into facts.
    """

    PAGE_LIMITS = dict(_API_LIMITS)
    MAX_PAGES = 64
    MAX_FINANCIAL_QUERIES = 64
    TRANSPORT_ATTEMPTS = 2
    COLLECT_TIMEOUT_SECONDS = 300

    def __init__(self, api: Any = None, clock: Any = None):
        self.api = api
        self.clock = clock if clock is not None else lambda: datetime.now(timezone.utc)
        self._secrets = []
        self._raw: dict[str, list[dict]] = {}
        self._cutoff: datetime | None = None
        self._last_observed: datetime | None = None
        self._deadline: float | None = None

    def _check_deadline(self, endpoint: str | None = None) -> None:
        if self._deadline is not None and monotonic() >= self._deadline:
            raise DataSourceError("COLLECT_TIMEOUT", endpoint=endpoint)

    def _now(self) -> datetime:
        value = self.clock()
        if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
            raise DataSourceError("INVALID_CLOCK")
        return value.astimezone(SHANGHAI)

    def _clean(self, value: Any) -> Any:
        if isinstance(value, dict):
            return {self._clean(str(k)): self._clean(v) for k, v in value.items()
                    if not any(s in str(k).lower().replace("_", "").replace("-", "") for s in (
                        "token", "secret", "password", "apikey", "accesskey", "authorization"))}
        if isinstance(value, (list, tuple)):
            return [self._clean(v) for v in value]
        if isinstance(value, str):
            for secret in self._secrets:
                value = value.replace(secret, "[REDACTED]")
            return value
        if value is None or isinstance(value, (bool, int)):
            return value
        if isinstance(value, float):
            return value if math.isfinite(value) else None
        if isinstance(value, (datetime, date)):
            return value.isoformat()
        if hasattr(value, "item"):
            try:
                return self._clean(value.item())
            except (ValueError, TypeError):
                pass
        # Includes pandas NA/NaT, which cannot be serialized as JSON facts.
        return None

    def _decode(self, response: Any) -> list[dict]:
        if isinstance(response, dict):
            if "code" not in response:
                raise DataSourceError("MALFORMED_RESPONSE")
            if str(response["code"]) != "0":
                raise DataSourceError(_business_code(self._clean(str(response.get("msg", ""))), response["code"]))
            data = response.get("data")
            if not isinstance(data, dict):
                raise DataSourceError("MALFORMED_RESPONSE")
            fields, items = data.get("fields"), data.get("items")
            if (not isinstance(fields, list) or not isinstance(items, list)
                    or not all(isinstance(f, str) for f in fields)
                    or len(set(fields)) != len(fields)
                    or any(not isinstance(r, (list, tuple)) or len(r) != len(fields) for r in items)):
                raise DataSourceError("MALFORMED_RESPONSE")
            response = [dict(zip(fields, item)) for item in items]
        elif hasattr(response, "to_dict"):
            try:
                response = response.to_dict(orient="records")
            except Exception:
                raise DataSourceError("MALFORMED_RESPONSE") from None
        if not isinstance(response, list) or any(not isinstance(r, dict) for r in response):
            raise DataSourceError("MALFORMED_RESPONSE")
        return self._clean(response)

    def _query(self, name: str, params: dict) -> list[dict]:
        for attempt in range(self.TRANSPORT_ATTEMPTS):
            self._check_deadline(name)
            _LOGGER.info("request endpoint=%s offset=%d limit=%d attempt=%d",
                         name, params["offset"], params["limit"], attempt + 1)
            try:
                response = self.api.query(name, **params)
            except DataSourceError:
                self._check_deadline(name)
                raise
            except requests.RequestException:
                self._check_deadline(name)
                if attempt + 1 == self.TRANSPORT_ATTEMPTS:
                    raise DataSourceError("TRANSPORT_ERROR") from None
                continue
            except Exception as exc:
                self._check_deadline(name)
                raise DataSourceError(_business_code(self._clean(str(exc)))) from None
            self._check_deadline(name)
            rows = self._decode(response)
            self._check_deadline(name)
            return rows
        raise DataSourceError("TRANSPORT_ERROR")

    def _fetch(self, name: str, **kwargs: Any) -> list[dict]:
        if name not in _API_LIMITS:
            raise DataSourceError("API_ERROR")
        _LOGGER.info("fetch_start endpoint=%s period=%s", name, _date(kwargs.get("period")))
        try:
            rows = self._fetch_pages(name, **kwargs)
        except DataSourceError as error:
            error.endpoint = name
            _LOGGER.info("fetch_failed endpoint=%s code=%s", name, error.safe_code)
            raise
        _LOGGER.info("fetch_complete endpoint=%s rows=%d", name, len(rows))
        return rows

    def _fetch_pages(self, name: str, **kwargs: Any) -> list[dict]:
        limit = self.PAGE_LIMITS[name]
        result = []
        seen_rows: set[str] = set()
        for page in range(self.MAX_PAGES):
            params = dict(kwargs, limit=limit, offset=page * limit)
            rows = self._read_page(name, params)
            fingerprints = {_hash({k: v for k, v in row.items() if k != "observed_at"}) for row in rows}
            if fingerprints & seen_rows:
                raise DataSourceError("PAGINATION_REPEATED")
            seen_rows.update(fingerprints)
            result.extend(rows)
            self._check_deadline(name)
            if len(rows) < limit:
                return result
        raise DataSourceError("PAGINATION_TRUNCATED")

    def _read_page(self, name: str, params: dict) -> list[dict]:
        rows = self._query(name, params)
        _LOGGER.info("page_received endpoint=%s offset=%d rows=%d", name, params["offset"], len(rows))
        observed = self._now()
        self._check_deadline(name)
        if self._cutoff is not None and observed >= self._cutoff:
            raise DataSourceError("CUTOFF_EXPIRED", endpoint=name)
        if self._last_observed is not None and observed < self._last_observed:
            raise DataSourceError("CLOCK_REGRESSION", endpoint=name)
        self._last_observed = observed
        if len(rows) > params["limit"]:
            raise DataSourceError("PAGINATION_TRUNCATED", endpoint=name)
        self._raw.setdefault(name, []).append({
            "params": self._clean(params), "rows": rows, "observed_at": observed.isoformat(),
        })
        for row in rows:
            if "trade_date" in params and _date(row.get("trade_date")) != _date(params["trade_date"]):
                raise DataSourceError("DATE_MISMATCH", endpoint=name)
            if "period" in params and _date(row.get("end_date")) != _date(params["period"]):
                raise DataSourceError("DATE_MISMATCH", endpoint=name)
            if "report_type" in params and str(row.get("report_type")) != str(params["report_type"]):
                raise DataSourceError("MALFORMED_RESPONSE", endpoint=name)
        self._check_deadline(name)
        return [dict(row, observed_at=observed.isoformat()) for row in rows]

    def _fetch_financial(self, period: str, report_type: str, day: date) -> list[dict]:
        # Offset pagination can skip tied announcement dates. Only sub-limit,
        # disjoint date leaves prove coverage; full parent responses stay in raw.
        name = "income_vip"
        pending = [(_date(period) + timedelta(days=1), day)]
        result, seen = [], set()
        count = 0
        limit = self.PAGE_LIMITS[name]
        _LOGGER.info("fetch_start endpoint=%s period=%s report_type=%s", name, period, report_type)
        try:
            while pending:
                self._check_deadline(name)
                if count >= self.MAX_FINANCIAL_QUERIES:
                    raise DataSourceError("PAGINATION_TRUNCATED", endpoint=name)
                start, end = pending.pop()
                params = dict(period=period, report_type=report_type,
                              start_date=start.strftime("%Y%m%d"), end_date=end.strftime("%Y%m%d"),
                              fields="ts_code,end_date,ann_date,f_ann_date,report_type,n_income_attr_p,revenue",
                              limit=limit, offset=0)
                rows = self._read_page(name, params)
                count += 1
                page = self._raw[name][-1]
                for row in rows:
                    announced = _date(row.get("ann_date"))
                    if announced is None or not start <= announced <= end:
                        raise DataSourceError("DATE_MISMATCH", endpoint=name)
                if len(rows) == limit:
                    page["selection_disposition"] = "split"
                    if start == end:
                        page["selection_disposition"] = "truncated"
                        raise DataSourceError("PAGINATION_TRUNCATED", endpoint=name)
                    middle = start + timedelta(days=(end - start).days // 2)
                    pending.extend([(middle + timedelta(days=1), end), (start, middle)])
                    _LOGGER.info("financial_split endpoint=%s start_date=%s end_date=%s", name, start, end)
                    continue
                page["selection_disposition"] = "leaf"
                for row in rows:
                    fingerprint = _hash({k: v for k, v in row.items() if k != "observed_at"})
                    if fingerprint not in seen:
                        result.append(row)
                        seen.add(fingerprint)
            self._check_deadline(name)
        except DataSourceError as error:
            error.endpoint = name
            _LOGGER.info("fetch_failed endpoint=%s code=%s", name, error.safe_code)
            raise
        _LOGGER.info("fetch_complete endpoint=%s rows=%d queries=%d", name, len(result), count)
        return result

    def _calendar(self, day: date) -> tuple[list[dict], date | None]:
        rows = []
        decisions = []
        for exchange in ("SSE", "SZSE"):
            calendar = self._fetch("trade_cal", exchange=exchange,
                                   start_date=day.strftime("%Y%m%d"),
                                   end_date=(day + timedelta(days=45)).strftime("%Y%m%d"))
            by_date = {}
            for row in calendar:
                effective = _date(row.get("cal_date"))
                flag = str(row.get("is_open"))
                if effective is None or flag not in ("0", "1"):
                    raise DataSourceError("CALENDAR_INCOMPLETE", endpoint="trade_cal")
                if effective in by_date and by_date[effective] != flag:
                    raise DataSourceError("CALENDAR_INCOMPLETE", endpoint="trade_cal")
                by_date[effective] = flag
                rows.append(dict(row, exchange=exchange))
            if day not in by_date:
                raise DataSourceError("CALENDAR_INCOMPLETE", endpoint="trade_cal")
            if by_date[day] == "0":
                decisions.append(None)
                continue
            next_open = min((d for d, flag in by_date.items() if d > day and flag == "1"), default=None)
            if next_open is None:
                raise DataSourceError("CALENDAR_INCOMPLETE", endpoint="trade_cal")
            cursor = day + timedelta(days=1)
            while cursor <= next_open:
                if cursor not in by_date:
                    raise DataSourceError("CALENDAR_INCOMPLETE", endpoint="trade_cal")
                cursor += timedelta(days=1)
            decisions.append(next_open)
        if decisions[0] != decisions[1]:
            raise DataSourceError("CALENDAR_INCOMPLETE", endpoint="trade_cal")
        return rows, decisions[0]

    @staticmethod
    def _index(rows: list[dict], endpoint: str) -> dict[str, dict]:
        result = {}
        for row in rows:
            code = row.get("ts_code")
            if not isinstance(code, str) or not code.strip():
                raise DataSourceError("MALFORMED_RESPONSE", endpoint=endpoint)
            if code in result:
                old = {k: v for k, v in result[code].items() if k != "observed_at"}
                new = {k: v for k, v in row.items() if k != "observed_at"}
                if old != new:
                    raise DataSourceError("AMBIGUOUS_DATA", endpoint=endpoint)
            result[code] = row
        return result

    @staticmethod
    def _financial(row: dict) -> dict:
        dates = [d for d in (_date(row.get("f_ann_date")), _date(row.get("ann_date"))) if d]
        available = datetime.combine(max(dates) + timedelta(days=1), time.min, SHANGHAI) if dates else None
        original = {k: v for k, v in row.items() if k != "observed_at"}
        return {
            "end_date": row.get("end_date"), "ann_date": row.get("ann_date"),
            "f_ann_date": row.get("f_ann_date"), "report_type": str(row.get("report_type")),
            "n_income_attr_p": _number(row.get("n_income_attr_p")),
            "revenue": _number(row.get("revenue")),
            "available_at": available.isoformat() if available else None,
            "observed_at": row["observed_at"], "first_observed_at": row["observed_at"],
            "source": "tushare.income_vip", "revision_id": _hash(original),
        }

    def collect(self, data_date: str | None = None) -> dict:
        self._deadline = monotonic() + self.COLLECT_TIMEOUT_SECONDS
        started = self._now()
        day = started.date() if data_date is None else _date(data_date)
        if day is None:
            raise DataSourceError("INVALID_DATE")
        if day != started.date():
            raise DataSourceError("FORWARD_DATE_REQUIRED")
        token = os.environ.get("TUSHARE_TOKEN", "").strip()
        if self.api is None:
            if not token:
                raise DataSourceError("TOKEN_MISSING")
            self.api = _HTTPSAPI(token)
        if isinstance(self.api, _HTTPSAPI):
            self.api.deadline = self._deadline
        self._secrets = [s for s in (token, getattr(self.api, "token", None)) if isinstance(s, str) and s]
        self._raw, self._cutoff, self._last_observed = {}, None, None
        calendar, target = self._calendar(day)
        snapshot = {"schema_version": "small-cap-snapshot-v1", "mode": "forward", "data_date": day.isoformat(),
                    "target_trade_date": target.isoformat() if target else None,
                    "timezone": "Asia/Shanghai", "calendar": calendar, "records": [],
                    "raw": self._raw, "warnings": []}
        if target is None:
            now = self._now()
            snapshot.update(status="CLOSED", as_of_at=now.isoformat(),
                            observed_at=self._last_observed.isoformat(),
                            coverage={"complete": True, "unknown_codes": [], "endpoints": self._coverage()})
            self._check_deadline()
            return snapshot
        if started.time() < time(15):
            raise DataSourceError("BEFORE_CLOSE")
        self._cutoff = datetime.combine(target, time.min, SHANGHAI)
        if self._now() >= self._cutoff:
            raise DataSourceError("CUTOFF_EXPIRED")
        compact = day.strftime("%Y%m%d")
        identities = []
        for status in ("L", "D", "P"):
            identities.extend(self._fetch(
                "stock_basic", exchange="", list_status=status,
                fields="ts_code,name,exchange,market,curr_type,list_date,delist_date,industry,list_status"))
        basic = self._index(identities, "stock_basic")
        bak = self._index(self._fetch("bak_basic", trade_date=compact), "bak_basic")
        daily = self._index(self._fetch("daily", trade_date=compact), "daily")
        valuation = self._index(self._fetch("daily_basic", trade_date=compact), "daily_basic")
        st = self._index(self._fetch("stock_st", trade_date=compact,
                                    fields="ts_code,name,trade_date"), "stock_st")
        suspensions = self._fetch("suspend_d", trade_date=compact)
        if any(not isinstance(r.get("ts_code"), str) or not r["ts_code"].strip() for r in suspensions):
            raise DataSourceError("MALFORMED_RESPONSE", endpoint="suspend_d")
        suspension_rows = defaultdict(list)
        for row in suspensions:
            suspension_rows[row["ts_code"]].append(row)
        if not basic or not daily or not valuation:
            endpoint = "stock_basic" if not basic else "daily" if not daily else "daily_basic"
            raise DataSourceError("EMPTY_DATA", endpoint=endpoint)
        try:
            limits = self._index(self._fetch("stk_limit", trade_date=compact), "stk_limit")
        except DataSourceError as exc:
            if exc.code in {"COLLECT_TIMEOUT", "CUTOFF_EXPIRED", "CLOCK_REGRESSION", "INVALID_CLOCK"}:
                raise
            limits = {}
            snapshot["warnings"].append({"endpoint": "stk_limit", "code": exc.code})
        financials = defaultdict(list)
        financial_codes = set()
        for year in (day.year - 1, day.year - 2):
            for report_type in ("1", "4", "5"):
                for row in self._fetch_financial(f"{year}1231", report_type, day):
                    self._check_deadline("income_vip")
                    if not isinstance(row.get("ts_code"), str) or not row["ts_code"].strip():
                        raise DataSourceError("MALFORMED_RESPONSE", endpoint="income_vip")
                    financial_codes.add(row["ts_code"])
                    financials[row["ts_code"]].append(self._financial(row))

        positive_codes = (set(basic) | set(bak) | set(daily) | set(valuation) | set(st)
                          | set(suspension_rows) | set(limits))
        finance_only_szse = {code for code in financial_codes - positive_codes
                             if len(code) == 9 and code[:6].isascii() and code[:6].isdigit() and code.endswith(".SZ")}
        if finance_only_szse:
            endpoint = "szse_official_a_share_master"
            _LOGGER.info("fetch_start endpoint=%s", endpoint)
            try:
                known = set()
                for code, row in basic.items():
                    if (row.get("exchange") == "SZSE" and row.get("curr_type") == "CNY"
                            and row.get("list_status") in {"L", "P"}):
                        listed = _date(row.get("list_date"))
                        if listed is None:
                            raise OfficialMasterError("OFFICIAL_CROSS_CHECK_FAILED")
                        if listed <= day:
                            known.add(code)
                self._check_deadline()
                page = fetch_szse_a_share_master(day.isoformat(), known, clock=self.clock,
                                                timeout_seconds=min(20, self._deadline - monotonic()))
                self._check_deadline()
                observed = datetime.fromisoformat(page["observed_at"]).astimezone(SHANGHAI)
                if observed >= self._cutoff:
                    raise DataSourceError("CUTOFF_EXPIRED")
                if observed < self._last_observed:
                    raise DataSourceError("CLOCK_REGRESSION")
                self._last_observed = observed
                self._raw[endpoint] = [self._clean(page)]
                _LOGGER.info("fetch_complete endpoint=%s rows=%d", endpoint, len(page["rows"]))
            except OfficialMasterError as error:
                self._check_deadline()
                snapshot["warnings"].append({"endpoint": endpoint, "code": error.safe_code})
                _LOGGER.info("fetch_failed endpoint=%s code=%s", endpoint, error.safe_code)

        # Include every quote/status code, even when identity joins fail. Missing
        # values are UNKNOWN evidence, never a reason to discard a low-cap name.
        codes = (set(basic) | set(bak) | set(daily) | set(valuation) | set(st)
                 | set(suspension_rows) | set(limits) | financial_codes)
        unknown_fields = {}
        for code in sorted(codes):
            self._check_deadline()
            identity = basic.get(code, {})
            current = bak.get(code, {})
            quote = daily.get(code, {})
            value = valuation.get(code, {})
            quote_close, valuation_close = _number(quote.get("close")), _number(value.get("close"))
            if quote_close is not None and valuation_close is not None and abs(quote_close - valuation_close) > 1e-8:
                raise DataSourceError("AMBIGUOUS_DATA", endpoint="daily_basic")
            name = st.get(code, {}).get("name") or current.get("name")
            delisted = _date(identity.get("delist_date"))
            effective = day.isoformat()
            if delisted is not None and delisted <= day:
                status, effective = "DELISTING", delisted.isoformat()
            elif code in st:
                status = "ST"
            elif current.get("name"):
                reliable_name = str(current["name"]).upper().replace(" ", "")
                if "\u9000" in reliable_name:
                    status = "DELISTING"
                elif reliable_name.startswith(("ST", "*ST", "SST", "S*ST")):
                    status = "ST"
                else:
                    status = "NORMAL"
            else:
                status, effective = "UNKNOWN", None
            volume_lots = _number(quote.get("vol"))
            events = suspension_rows.get(code, [])
            stops = [r for r in events if r.get("suspend_type") == "S"]
            resumes = [r for r in events if r.get("suspend_type") == "R"]
            intraday = any(r.get("suspend_timing") for r in stops)
            if volume_lots == 0:
                suspended = True
            elif stops and not resumes and not intraday and volume_lots is None:
                suspended = True
            elif stops:
                suspended = None
                snapshot["warnings"].append({"endpoint": "suspend_d", "code": "SUSPENSION_UNRESOLVED",
                                             "ts_code": code})
            else:
                suspended = False if volume_lots is not None and volume_lots > 0 else None
            market_cap = _number(value.get("total_mv"))
            evidence = [identity, current, quote, value, st.get(code, {}), limits.get(code, {})]
            evidence += suspension_rows.get(code, []) + financials.get(code, [])
            observed_times = [r["observed_at"] for r in evidence if r]
            record = {
                "ts_code": code, "name": name, "exchange": identity.get("exchange"),
                "market": identity.get("market"), "curr_type": identity.get("curr_type"),
                "list_date": identity.get("list_date") or current.get("list_date"),
                "delist_date": identity.get("delist_date"),
                "industry": current.get("industry") or identity.get("industry"),
                "industry_basis": "current_snapshot", "security_status": status,
                "status_effective_date": effective,
                "observed_at": max(observed_times, default=self._last_observed.isoformat()),
                "trade_date": day.isoformat() if quote else None,
                "valuation_date": day.isoformat() if value else None,
                "close": _number(quote.get("close")),
                "total_market_cap_yuan": _number(market_cap * 10000) if market_cap is not None else None,
                "volume": _number(volume_lots * 100) if volume_lots is not None else None,
                "volume_unit": "share", "suspended": suspended,
                "suspension_date": day.isoformat() if suspended is True else None,
                "financials": financials.get(code, []),
                "up_limit": _number(limits.get(code, {}).get("up_limit")),
                "down_limit": _number(limits.get(code, {}).get("down_limit")),
                "source": "tushare", "status_basis": "T_day_stock_st_and_bak_basic",
            }
            missing = [field for field in ("exchange", "market", "curr_type", "list_date", "name",
                                          "close", "total_market_cap_yuan", "volume", "suspended")
                       if record[field] is None]
            if status == "UNKNOWN":
                missing.append("security_status")
            if not record["financials"]:
                missing.append("financials")
            else:
                for field in ("n_income_attr_p", "revenue", "available_at"):
                    if any(r[field] is None for r in record["financials"]):
                        missing.append("financials." + field)
            if missing:
                unknown_fields[code] = missing
            snapshot["records"].append(record)
        sealed = self._now()
        if sealed >= self._cutoff:
            raise DataSourceError("CUTOFF_EXPIRED")
        if sealed < self._last_observed:
            raise DataSourceError("CLOCK_REGRESSION")
        snapshot.update(status="OK", as_of_at=sealed.isoformat(),
                        observed_at=self._last_observed.isoformat(), coverage={
                            "complete": True, "scope": "endpoint_acquisition",
                            "security_count": len(codes), "unknown_codes": sorted(unknown_fields),
                            "missing_fields": unknown_fields, "endpoints": self._coverage(),
                        })
        self._check_deadline()
        return snapshot

    def _coverage(self) -> dict:
        return {name: {"pages": len(pages), "rows": sum(len(p["rows"]) for p in pages)}
                for name, pages in self._raw.items()}

    def preflight(self, data_date: str | None = None) -> dict:
        """Read-only capability check; never persists or sends a report."""
        return self.collect(data_date)
