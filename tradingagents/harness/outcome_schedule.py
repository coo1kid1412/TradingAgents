"""Exchange-session targets independent of quote availability."""

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from functools import lru_cache
import re
from zoneinfo import ZoneInfo

import exchange_calendars as xcals
import pandas as pd

VERSION = "exchange-close-v1"
OFFSETS = {"T": 0, "T+1": 1, "T+5": 5, "T+30": 30}
CLOSE_DELAY = timedelta(hours=1)


def market_calendar(ticker):
    ticker = str(ticker).strip().upper()
    if re.fullmatch(r"\d{6}(?:\.(?:SH|SZ|BJ))?", ticker):
        return "XSHG"
    if re.fullmatch(r"[A-Z]{1,6}(?:[.-][A-Z])?", ticker):
        return "XNYS"
    raise ValueError("unsupported market: " + ticker)


@lru_cache(maxsize=16)
def _calendar(name, year):
    template = xcals.get_calendar(name)
    start = pd.Timestamp(year=year, month=1, day=1)
    end = pd.Timestamp(year=year + 1, month=4, day=30)
    upper = template.bound_max()
    if upper is not None:
        end = min(end, upper)
    return type(template)(start=start, end=end)


@dataclass(frozen=True)
class Target:
    sessions: tuple[date, ...]
    due_at: datetime

    @property
    def date(self):
        return self.sessions[-1]


def target_for(run, horizon):
    """T is the first session closing after report publication; T+n adds n sessions.

    Legacy naive report timestamps are Shanghai time, including US reports.
    Calendar bounds fail explicitly; missing sessions are never invented.
    """
    offset = OFFSETS[horizon]
    report_at = datetime.fromisoformat(run["report_timestamp"])
    if report_at.tzinfo is None:
        report_at = report_at.replace(tzinfo=ZoneInfo("Asia/Shanghai"))
    base = datetime.combine(date.fromisoformat(run["trade_date"]), datetime.min.time(),
                            tzinfo=ZoneInfo("Asia/Shanghai"))
    report_at = max(report_at, base)
    name = market_calendar(run["ticker"])
    local_day = report_at.astimezone(ZoneInfo("Asia/Shanghai" if name == "XSHG" else "America/New_York")).date()
    calendar = _calendar(name, local_day.year)
    anchor = calendar.date_to_session(local_day, direction="next")
    if calendar.session_close(anchor).to_pydatetime() <= report_at:
        anchor = calendar.next_session(anchor)
    index = calendar.sessions.get_loc(anchor)
    sessions = calendar.sessions[index:index + offset + 1]
    if len(sessions) != offset + 1:
        raise ValueError("target exceeds published exchange calendar")
    due_at = calendar.session_close(sessions[-1]).to_pydatetime() + CLOSE_DELAY
    return Target(tuple(s.date() for s in sessions), due_at.astimezone(timezone.utc))
