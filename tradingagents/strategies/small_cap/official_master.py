"""Current-day, identity-only SZSE official listed A-share census capture."""

from datetime import datetime, timezone
from hashlib import sha256
from io import BytesIO
from time import monotonic
from urllib.parse import urlsplit
from zipfile import ZipFile
from zoneinfo import ZoneInfo

import pandas as pd
import requests

from .config import digest


URL = "https://www.szse.cn/api/report/ShowReport"
PARAMS = {"SHOWTYPE": "xlsx", "CATALOGID": "1110", "TABKEY": "tab1"}
FIELDS = ("板块", "A股代码", "A股简称", "A股上市日期")
SHANGHAI = ZoneInfo("Asia/Shanghai")
_CODES = {"OFFICIAL_DATE_MISMATCH", "OFFICIAL_CROSS_CHECK_FAILED", "OFFICIAL_INVALID_SCHEMA",
          "OFFICIAL_HTTP_ERROR", "OFFICIAL_TRANSPORT_ERROR", "OFFICIAL_TIMEOUT"}


class OfficialMasterError(RuntimeError):
    def __init__(self, code):
        self.safe_code = code if code in _CODES else "OFFICIAL_INVALID_SCHEMA"
        self.code = self.safe_code
        self.endpoint = "szse_official_a_share_master"
        super().__init__(self.safe_code)


def _date(value):
    try:
        text = str(value).strip()
        if len(text) == 8 and text.isdigit():
            return datetime.strptime(text, "%Y%m%d").date()
        parsed = datetime.fromisoformat(text)
        if parsed.hour or parsed.minute or parsed.second or parsed.microsecond or parsed.tzinfo:
            raise ValueError("not a date")
        return parsed.date()
    except (TypeError, ValueError):
        raise OfficialMasterError("OFFICIAL_INVALID_SCHEMA") from None


def _now(clock):
    value = clock()
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise OfficialMasterError("OFFICIAL_DATE_MISMATCH")
    return value.astimezone(SHANGHAI)


def _parse(payload, day):
    if not isinstance(payload, bytes) or not 0 < len(payload) <= 5 * 1024 * 1024:
        raise OfficialMasterError("OFFICIAL_INVALID_SCHEMA")
    try:
        with ZipFile(BytesIO(payload)) as archive:
            if sum(item.file_size for item in archive.infolist()) > 64 * 1024 * 1024:
                raise OfficialMasterError("OFFICIAL_INVALID_SCHEMA")
        # Reading the header as data avoids pandas silently renaming duplicates.
        table = pd.read_excel(BytesIO(payload), header=None, dtype=str, engine="openpyxl")
    except OfficialMasterError:
        raise
    except Exception:
        raise OfficialMasterError("OFFICIAL_INVALID_SCHEMA") from None
    if len(table) < 2:
        raise OfficialMasterError("OFFICIAL_INVALID_SCHEMA")
    headers = table.iloc[0].tolist()
    if (not all(isinstance(h, str) and h for h in headers) or len(set(headers)) != len(headers)
            or not set(FIELDS) <= set(headers)):
        raise OfficialMasterError("OFFICIAL_INVALID_SCHEMA")
    positions = [headers.index(field) for field in FIELDS]
    rows, codes = [], set()
    for values in table.iloc[1:].itertuples(index=False, name=None):
        board, code, name, listing = [values[position] for position in positions]
        if any(pd.isna(value) for value in (board, code, name, listing)):
            raise OfficialMasterError("OFFICIAL_INVALID_SCHEMA")
        board, code, name = str(board).strip(), str(code).strip(), str(name).strip()
        listed = _date(listing)
        if (board not in {"主板", "创业板"} or not code.isascii() or not code.isdigit()
                or not 1 <= len(code) <= 6 or not name or listed > day):
            raise OfficialMasterError("OFFICIAL_INVALID_SCHEMA")
        code = code.zfill(6) + ".SZ"
        if code in codes:
            raise OfficialMasterError("OFFICIAL_INVALID_SCHEMA")
        codes.add(code)
        rows.append(dict(ts_code=code, name=name, market=board, list_date=listed.isoformat()))
    return sorted(rows, key=lambda row: row["ts_code"]), codes


def fetch_szse_a_share_master(data_date, known_active_codes, *, clock=None, timeout_seconds=20):
    """Return a frozen census page only after schema and full active-set checks.

    It proves current capture-day identity membership, never historical status
    or T-close status. It does not backfill prices, ST status or financial data.
    """
    clock = clock or (lambda: datetime.now(timezone.utc))
    started = _now(clock)
    day = _date(data_date)
    if day != started.date():
        raise OfficialMasterError("OFFICIAL_DATE_MISMATCH")
    expected = set(known_active_codes)
    if not expected or any(not isinstance(code, str) or len(code) != 9
                           or not code[:6].isascii() or not code[:6].isdigit()
                           or not code.endswith(".SZ") for code in expected):
        raise OfficialMasterError("OFFICIAL_CROSS_CHECK_FAILED")
    if not 0 < timeout_seconds <= 20:
        raise OfficialMasterError("OFFICIAL_TIMEOUT")
    deadline = monotonic() + timeout_seconds
    try:
        with requests.get(URL, params=dict(PARAMS), verify=True, allow_redirects=False,
                          timeout=(min(5, timeout_seconds), timeout_seconds)) as response:
            final = urlsplit(response.url)
            if (response.status_code != 200 or final.scheme != "https" or final.netloc != "www.szse.cn"
                    or final.path != "/api/report/ShowReport"):
                raise OfficialMasterError("OFFICIAL_HTTP_ERROR")
            payload = response.content
    except requests.RequestException:
        raise OfficialMasterError("OFFICIAL_TRANSPORT_ERROR") from None
    if monotonic() >= deadline:
        raise OfficialMasterError("OFFICIAL_TIMEOUT")
    rows, codes = _parse(payload, day)
    if not expected <= codes:
        raise OfficialMasterError("OFFICIAL_CROSS_CHECK_FAILED")
    observed = _now(clock)
    if observed.date() != day or observed < started:
        raise OfficialMasterError("OFFICIAL_DATE_MISMATCH")
    if monotonic() >= deadline:
        raise OfficialMasterError("OFFICIAL_TIMEOUT")
    return {"source_url": URL, "params": dict(PARAMS), "data_date": day.isoformat(),
            "scope": "current_as_of_listed_a_shares", "mode": "forward",
            "observed_at": observed.isoformat(), "rows": rows,
            "coverage_complete": True, "cross_check_complete": True,
            "cross_check_codes": sorted(expected), "rows_hash": digest(rows),
            "source_hash": sha256(payload).hexdigest()}
