"""Exchange-session resolver tests used by runtime and backfills."""

from __future__ import annotations

from datetime import date
from unittest import TestCase, main

from tradingagents.harness.market_warning.calendars import session_resolvers
from tradingagents.harness.market_warning.domain import Market


class SessionResolverTests(TestCase):
    def test_shared_resolvers_cover_the_frozen_history_from_2000(self):
        for market in Market:
            with self.subTest(market=market):
                next_session, previous_session, _ = session_resolvers(market)
                self.assertEqual(previous_session(date(2000, 1, 10)), date(2000, 1, 7))
                self.assertEqual(next_session(date(2000, 1, 7)), date(2000, 1, 10))

    def test_us_independence_day_and_a_share_national_day_use_exchange_sessions(self):
        _, us_previous, us_version = session_resolvers(Market.US)
        _, a_previous, a_version = session_resolvers(Market.A_SHARE)

        self.assertEqual(us_previous(date(2026, 7, 6)), date(2026, 7, 2))
        self.assertEqual(a_previous(date(2026, 10, 8)), date(2026, 9, 30))
        self.assertIn("XNYS", us_version)
        self.assertIn("XSHG", a_version)


if __name__ == "__main__":
    main()
