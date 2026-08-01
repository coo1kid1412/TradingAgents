"""Shared exchange-calendar session resolvers."""

from __future__ import annotations

from datetime import date

import exchange_calendars as xcals
import pandas as pd

from .domain import Market


def calendar_for_range(market: Market, start: date, end: date):
    """Build an uncached exchange calendar covering the requested range."""

    market = Market(market)
    calendar_name = "XSHG" if market == Market.A_SHARE else "XNYS"
    calendar_type = type(xcals.get_calendar(calendar_name))
    return calendar_type(start=start.isoformat(), end=end.isoformat())


def session_resolvers(market: Market):
    """Return strict next/previous session functions and a version marker."""

    market = Market(market)
    calendar_name = "XSHG" if market == Market.A_SHARE else "XNYS"
    calendar = calendar_for_range(
        market,
        date(1999, 1, 1),
        date(2026, 12, 31),
    )

    def next_session(current: date) -> date:
        candidate = pd.Timestamp(current) + pd.Timedelta(days=1)
        return calendar.date_to_session(candidate, direction="next").date()

    def previous_session(current: date) -> date:
        candidate = pd.Timestamp(current) - pd.Timedelta(days=1)
        return calendar.date_to_session(candidate, direction="previous").date()

    return (
        next_session,
        previous_session,
        f"exchange-calendars-{xcals.__version__}-{calendar_name}",
    )
