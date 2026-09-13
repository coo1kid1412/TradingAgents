"""Offline regressions for calendar maturity and bounded outcome collection."""

import datetime as dt
import tempfile
import unittest
import types
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from tradingagents.harness import db, truth_fetcher as truth


def stamp(value):
    return dt.datetime.fromisoformat(value)


def prices(*days):
    return pd.DataFrame([
        {"Date": dt.date.fromisoformat(day), "Close": 103.0, "High": 105.0, "Low": 99.0}
        for day in days
    ])


class TruthScheduleTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.path = db.init_db(Path(temp.name) / "test.db")
        self.now = stamp("2026-09-13T04:00:00+00:00")
        vendor = patch.object(truth._pcache, "fetch_with_cache", return_value=pd.DataFrame())
        self.vendor = vendor.start()
        self.addCleanup(vendor.stop)
        benchmark = patch.object(truth, "_compute_benchmark_return", return_value=None)
        self.benchmark = benchmark.start()
        self.addCleanup(benchmark.stop)

    def add_run(self, ident=1, ticker="300308", timestamp="2026-09-11T10:00:00",
                window="morning", horizon="T", status="pending"):
        with db.connect(self.path) as conn:
            conn.execute(
                "INSERT INTO runs(id,ticker,trade_date,report_timestamp,report_window,report_dir) "
                "VALUES (?,?,?,?,?,?)",
                (ident, ticker, timestamp[:10], timestamp, window, f"reports/{ident}"),
            )
            conn.execute("INSERT INTO predictions(run_id,current_price) VALUES (?,100)", (ident,))
            conn.execute(
                "INSERT INTO outcomes(run_id,horizon,reference_price,fetch_status,direction_predicted) "
                "VALUES (?,?,100,?,'long')", (ident, horizon, status),
            )

    def row(self, ident=1):
        with db.connect(self.path) as conn:
            return dict(conn.execute("SELECT * FROM outcomes WHERE run_id=?", (ident,)).fetchone())

    def collect(self, now=None):
        return truth.fetch_all_pending(self.path, update_benchmarks=False, now=now or self.now)

    def test_old_empty_prices_are_missing_not_future(self):
        self.add_run(timestamp="2026-05-28T10:00:00", horizon="T+30")
        truth.fetch_one_run_outcomes(1, self.path)
        row = self.row()
        self.assertEqual(row["fetch_status"], "data_missing")
        self.assertIsNotNone(row["target_date"])

    def test_friday_post_close_targets_monday_without_fetching(self):
        self.add_run(timestamp="2026-09-11T16:00:00", window="post_market")
        self.collect()
        row = self.row()
        self.assertEqual(row["target_date"], "2026-09-14")
        self.assertEqual(row["fetch_status"], "not_due")
        self.vendor.assert_not_called()

    def test_us_during_session_does_not_use_shanghai_window(self):
        self.add_run(ticker="NVDA", timestamp="2026-09-11T23:00:00", window="post_market")
        self.collect(stamp("2026-09-11T15:30:00+00:00"))
        row = self.row()
        self.assertEqual(row["target_date"], "2026-09-11")
        self.assertEqual(row["fetch_status"], "not_due")
        self.vendor.assert_not_called()

    def test_us_early_close_and_holiday(self):
        self.add_run(ticker="NVDA", timestamp="2026-11-28T01:30:00", window="post_market")
        self.collect(stamp("2026-11-27T18:30:00+00:00"))
        self.assertEqual(self.row()["target_date"], "2026-11-27")
        self.assertEqual(self.row()["fetch_status"], "not_due")
        self.collect(stamp("2026-11-27T19:01:00+00:00"))
        self.assertEqual(self.row()["fetch_status"], "data_missing")

    def test_missing_bar_does_not_shift_horizon(self):
        self.add_run(timestamp="2026-09-07T10:00:00", horizon="T+1")
        self.vendor.return_value = prices("2026-09-07", "2026-09-09")
        self.collect()
        self.assertEqual(self.row()["target_date"], "2026-09-08")
        self.assertEqual(self.row()["fetch_status"], "data_missing")

    def test_missing_period_bar_prevents_partial_high_low(self):
        self.add_run(timestamp="2026-09-07T10:00:00", horizon="T+1")
        self.vendor.return_value = prices("2026-09-08")
        self.collect()
        self.assertEqual(self.row()["fetch_status"], "data_missing")

    def test_complete_exact_period_is_collected(self):
        self.add_run(timestamp="2026-09-07T10:00:00", horizon="T+1")
        self.vendor.return_value = prices("2026-09-07", "2026-09-08", "2026-09-09")
        self.collect()
        row = self.row()
        self.assertEqual(row["fetch_status"], "fetched")
        self.assertEqual(row["target_date"], "2026-09-08")
        self.assertEqual(row["realized_return_pct"], 3)
        self.assertEqual(row["actual_high_during_horizon"], 105)

    def test_legacy_failed_retry_cooldown_and_cap(self):
        self.add_run(status="failed")
        self.collect()
        self.assertEqual(self.row()["attempt_count"], 1)
        self.collect()
        self.assertEqual(self.row()["attempt_count"], 1)
        self.collect(self.now + dt.timedelta(days=1))
        self.collect(self.now + dt.timedelta(days=2))
        self.assertEqual(self.row()["fetch_status"], "retry_exhausted")
        self.collect(self.now + dt.timedelta(days=10))
        self.assertEqual(self.row()["attempt_count"], 3)
        self.assertEqual(self.vendor.call_count, 3)

    def test_one_vendor_exception_does_not_abort_other_runs(self):
        self.add_run()
        self.add_run(2)
        self.vendor.side_effect = [RuntimeError("offline"), prices("2026-09-11")]
        self.collect()
        self.assertEqual(self.row()["fetch_status"], "failed")
        self.assertEqual(self.row(2)["fetch_status"], "fetched")

    def test_fetched_history_is_untouched(self):
        self.add_run(status="fetched")
        before = self.row()
        self.collect()
        self.assertEqual(before, self.row())
        self.vendor.assert_not_called()

    def test_end_exclusive_vendor_can_supply_target_close(self):
        self.add_run()
        def exclusive(ticker, start, end, path, **kwargs):
            return prices("2026-09-11") if end > dt.date(2026, 9, 11) else pd.DataFrame()
        self.vendor.side_effect = exclusive
        self.collect()
        self.assertEqual(self.row()["fetch_status"], "fetched")

    def test_schema_upgrade_does_not_change_old_result_evidence(self):
        from tradingagents.harness.quality_review import build_snapshot, compare_snapshot
        self.add_run(status="fetched")
        with db.connect(self.path) as conn:
            conn.execute("UPDATE outcomes SET target_date='2026-09-11'")
            for column in ("due_at", "schedule_version", "attempt_count", "next_retry_at"):
                conn.execute(f"ALTER TABLE outcomes DROP COLUMN {column}")
        before = build_snapshot(self.path, today=self.now.date())
        db.init_db(self.path)
        after = build_snapshot(self.path, today=self.now.date())
        self.assertFalse(compare_snapshot(after, before)["review_required"])

    def test_unsupported_market_is_explicit_and_not_retried(self):
        self.add_run(ticker="00700.HK")
        self.collect()
        self.assertEqual(self.row()["fetch_status"], "schedule_error")
        self.collect()
        self.vendor.assert_not_called()

    def test_invalid_reference_price_is_terminal(self):
        self.add_run()
        with db.connect(self.path) as conn:
            conn.execute("UPDATE predictions SET current_price=0")
        self.collect()
        self.assertEqual(self.row()["fetch_status"], "invalid")
        self.vendor.assert_not_called()

    def test_invalid_bar_is_not_scored(self):
        self.add_run()
        self.vendor.return_value = prices("2026-09-11")
        self.vendor.return_value.loc[0, "Close"] = float("inf")
        self.collect()
        self.assertEqual(self.row()["fetch_status"], "data_missing")

    def test_new_us_outcome_does_not_use_a_share_benchmark(self):
        self.add_run(ticker="MU", timestamp="2026-09-11T23:00:00", window="post_market")
        self.vendor.return_value = prices("2026-09-11")
        self.collect()
        self.assertEqual(self.row()["fetch_status"], "fetched")
        self.assertIsNone(self.row()["benchmark_ticker"])
        self.benchmark.assert_not_called()

    def test_quality_distinguishes_missing_and_exhausted(self):
        from tradingagents.harness.quality_review import build_snapshot, compare_snapshot
        self.add_run()
        self.collect()
        first = build_snapshot(self.path, today=self.now.date())
        self.assertEqual(first["tracking"]["data_missing"], 1)
        self.collect(self.now + dt.timedelta(days=1))
        second = build_snapshot(self.path, today=self.now.date())
        self.assertFalse(compare_snapshot(second, first)["review_required"])
        self.collect(self.now + dt.timedelta(days=2))
        third = build_snapshot(self.path, today=self.now.date())
        self.assertEqual(third["tracking"]["retry_exhausted"], 1)
        self.assertTrue(compare_snapshot(third, second)["review_required"])

    def test_calendar_bound_failure_is_explicit(self):
        self.add_run(timestamp="2026-12-21T10:00:00", horizon="T+30")
        self.collect(stamp("2026-12-22T12:00:00+00:00"))
        self.assertEqual(self.row()["fetch_status"], "schedule_error")
        self.vendor.assert_not_called()

    def test_a_share_holiday_is_not_a_missing_bar(self):
        self.add_run(timestamp="2026-09-30T16:00:00", window="post_market")
        self.collect(stamp("2026-10-01T12:00:00+00:00"))
        self.assertEqual(self.row()["target_date"], "2026-10-08")
        self.assertEqual(self.row()["fetch_status"], "not_due")


class CacheCoverageTests(unittest.TestCase):
    def test_invalid_cached_ohlc_is_refetched(self):
        from tradingagents.harness import price_cache
        with tempfile.TemporaryDirectory() as root:
            path = db.init_db(Path(root) / "test.db")
            frame = prices("2026-09-11")
            frame.loc[0, "High"] = float("nan")
            price_cache._write_to_cache("300308", frame, path)
            required = tuple(frame["Date"])
            with patch.object(price_cache, "_fetch_from_vendor", return_value=prices("2026-09-11")) as vendor:
                result = price_cache.fetch_with_cache("300308", required[0], required[-1], path,
                                                      expected_sessions=required)
            self.assertEqual(result.iloc[0]["High"], 105)
            vendor.assert_called_once()

    def test_inclusive_vendor_boundary_on_today_and_no_future_cache(self):
        from tradingagents.harness import price_cache
        today = dt.date.today()
        tomorrow = today + dt.timedelta(days=1)
        interface = types.ModuleType("tradingagents.dataflows.interface")
        def exclusive(method, ticker, start, end):
            self.assertEqual(end, tomorrow.isoformat())
            # An inclusive vendor could return the extra date too; never retain it.
            return prices(today.isoformat(), tomorrow.isoformat()).to_csv(index=False)
        interface.route_to_vendor = exclusive
        with tempfile.TemporaryDirectory() as root, patch.dict("sys.modules", {interface.__name__: interface}):
            path = db.init_db(Path(root) / "test.db")
            result = price_cache.fetch_with_cache("300308", today, tomorrow, path,
                                                  expected_sessions=(today,))
            self.assertEqual(tuple(result["Date"]), (today,))
            self.assertEqual(price_cache._get_cache_range("300308", path)[1], today)

    def test_requested_historical_gap_is_refetched(self):
        from tradingagents.harness import price_cache
        with tempfile.TemporaryDirectory() as root:
            path = db.init_db(Path(root) / "test.db")
            price_cache._write_to_cache("300308", prices("2026-09-07", "2026-09-09"), path)
            required = (dt.date(2026, 9, 7), dt.date(2026, 9, 8))
            with patch.object(price_cache, "_fetch_from_vendor", return_value=prices("2026-09-07", "2026-09-08")) as vendor:
                frame = price_cache.fetch_with_cache("300308", required[0], required[-1], path,
                                                     expected_sessions=required)
            self.assertEqual(tuple(frame["Date"]), required)
            vendor.assert_called_once_with("300308", "2026-09-07", "2026-09-08")

    def test_complete_cached_sessions_do_not_fetch(self):
        from tradingagents.harness import price_cache
        with tempfile.TemporaryDirectory() as root:
            path = db.init_db(Path(root) / "test.db")
            frame = prices("2026-09-07", "2026-09-08")
            price_cache._write_to_cache("300308", frame, path)
            required = tuple(frame["Date"])
            with patch.object(price_cache, "_fetch_from_vendor") as vendor:
                result = price_cache.fetch_with_cache("300308", required[0], required[-1], path,
                                                      expected_sessions=required)
            self.assertEqual(tuple(result["Date"]), required)
            vendor.assert_not_called()


if __name__ == "__main__":
    unittest.main()
