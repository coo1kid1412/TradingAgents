"""Tests for Tushare realtime permission, baseline, and breadth frames."""

from __future__ import annotations

import tempfile
from datetime import date, datetime, timedelta
from pathlib import Path
from unittest import TestCase, main
from zoneinfo import ZoneInfo

import pandas as pd

from tradingagents.harness.market_warning.adapters.tushare_realtime_breadth import (
    PremarketBreadthBaseline,
    RealtimePermissionUnavailable,
    build_premarket_baseline,
    load_realtime_cross_section,
    probe_rt_k_permission,
)


SHANGHAI = ZoneInfo("Asia/Shanghai")
AS_OF = datetime(2026, 8, 3, 9, 35, tzinfo=SHANGHAI)


class PermissionProbeTests(TestCase):
    def test_probe_classifies_available_permission_empty_invalid_and_missing_api(self):
        class Available:
            def rt_k(self, **_):
                return pd.DataFrame(
                    [
                        {
                            "ts_code": "600000.SH",
                            "pre_close": 10.0,
                            "close": 10.1,
                            "trade_time": "2026-08-03 09:35:00",
                        }
                    ]
                )

        class Denied:
            def rt_k(self, **_):
                raise RuntimeError("permission denied for rt_k")

        class Empty:
            def rt_k(self, **_):
                return pd.DataFrame()

        class Invalid:
            def rt_k(self, **_):
                return pd.DataFrame([{"ts_code": "600000.SH", "close": 10.1}])

        self.assertEqual(probe_rt_k_permission(Available(), AS_OF).status, "available")
        self.assertEqual(probe_rt_k_permission(Denied(), AS_OF).status, "permission_denied")
        self.assertEqual(probe_rt_k_permission(Empty(), AS_OF).status, "unavailable")
        self.assertEqual(probe_rt_k_permission(Invalid(), AS_OF).status, "invalid_payload")
        self.assertEqual(probe_rt_k_permission(object(), AS_OF).status, "unavailable")


class BaselineTests(TestCase):
    def test_baseline_uses_only_twenty_sessions_ending_at_previous_completed_day(self):
        sessions = pd.bdate_range(end="2026-07-31", periods=20)

        class FakePro:
            def stock_basic(self, **_):
                return pd.DataFrame(
                    [
                        {"ts_code": "600000.SH", "industry": "bank"},
                        {"ts_code": "000001.SZ", "industry": "bank"},
                    ]
                )

            def trade_cal(self, **_):
                return pd.DataFrame({"cal_date": sessions.strftime("%Y%m%d"), "is_open": 1})

            def daily(self, **kwargs):
                trade_date = kwargs["trade_date"]
                position = list(sessions.strftime("%Y%m%d")).index(trade_date)
                return pd.DataFrame(
                    [
                        {"ts_code": "600000.SH", "trade_date": trade_date, "close": 10.0 + position},
                        {"ts_code": "000001.SZ", "trade_date": trade_date, "close": 20.0 + position},
                    ]
                )

            def stk_limit(self, **kwargs):
                self.limit_trade_date = kwargs["trade_date"]
                return pd.DataFrame(
                    [
                        {"ts_code": "600000.SH", "down_limit": 26.1},
                        {"ts_code": "000001.SZ", "down_limit": 35.1},
                    ]
                )

        pro = FakePro()
        with tempfile.TemporaryDirectory() as directory:
            baseline = build_premarket_baseline(
                pro,
                trade_date=date(2026, 8, 3),
                previous_session=lambda _: date(2026, 7, 31),
                cache_root=Path(directory),
            )

        first = baseline.frame.set_index("ts_code").loc["600000.SH"]
        self.assertEqual(baseline.completed_trade_date, date(2026, 7, 31))
        self.assertEqual(baseline.universe_size, 2)
        self.assertEqual(first["pre_close"], 29.0)
        self.assertEqual(first["ma20"], 19.5)
        self.assertEqual(first["low_20d"], 10.0)
        self.assertEqual(first["industry"], "bank")
        self.assertEqual(first["down_limit"], 26.1)
        self.assertEqual(pro.limit_trade_date, "20260803")


class RealtimeCrossSectionTests(TestCase):
    @staticmethod
    def baseline() -> PremarketBreadthBaseline:
        return PremarketBreadthBaseline(
            trade_date=date(2026, 8, 3),
            completed_trade_date=date(2026, 7, 31),
            universe_size=3,
            frame=pd.DataFrame(
                [
                    {"ts_code": "600000.SH", "pre_close": 10.0, "ma20": 9.5, "low_20d": 8.0, "industry": "bank", "down_limit": 9.0},
                    {"ts_code": "000001.SZ", "pre_close": 20.0, "ma20": 21.0, "low_20d": 18.0, "industry": "bank", "down_limit": 18.0},
                    {"ts_code": "300001.SZ", "pre_close": 30.0, "ma20": 29.0, "low_20d": 25.0, "industry": "tech", "down_limit": 24.0},
                ]
            ),
        )

    def test_loader_accepts_only_current_visible_rows_and_preserves_baseline(self):
        class FakePro:
            def rt_k(self, **_):
                return pd.DataFrame(
                    [
                        {"ts_code": "600000.SH", "pre_close": 10.0, "close": 10.2, "trade_time": "2026-08-03 09:34:00"},
                        {"ts_code": "000001.SZ", "pre_close": 20.0, "close": 19.8, "trade_time": "2026-08-03 09:35:00"},
                        {"ts_code": "300001.SZ", "pre_close": 30.0, "close": 30.5, "trade_time": "2026-08-01 14:55:00"},
                        {"ts_code": "300001.SZ", "pre_close": 30.0, "close": 29.5, "trade_time": "2026-08-03 09:36:00"},
                    ]
                )

        result = load_realtime_cross_section(FakePro(), self.baseline(), AS_OF)

        self.assertEqual(set(result["ts_code"]), {"600000.SH", "000001.SZ"})
        self.assertEqual(result.attrs["universe_size"], 3)
        self.assertEqual(result.set_index("ts_code").loc["600000.SH", "ma20"], 9.5)
        self.assertTrue((result["source"] == "tushare_rt_k").all())
        self.assertTrue((result["data_time"].dt.date == AS_OF.date()).all())

    def test_permission_error_is_explicit_and_never_falls_back_to_t_minus_one(self):
        class Denied:
            def rt_k(self, **_):
                raise RuntimeError("没有 rt_k 权限")

        with self.assertRaises(RealtimePermissionUnavailable):
            load_realtime_cross_section(Denied(), self.baseline(), AS_OF)

    def test_empty_current_session_is_not_replaced_with_baseline_prices(self):
        class OnlyOld:
            def rt_k(self, **_):
                return pd.DataFrame(
                    [
                        {"ts_code": "600000.SH", "pre_close": 9.8, "close": 10.0, "trade_time": "2026-07-31 15:00:00"}
                    ]
                )

        result = load_realtime_cross_section(OnlyOld(), self.baseline(), AS_OF)

        self.assertTrue(result.empty)

    def test_rows_older_than_five_minutes_are_excluded_individually(self):
        class MixedFreshness:
            def rt_k(self, **_):
                return pd.DataFrame(
                    [
                        {"ts_code": "600000.SH", "pre_close": 10.0, "close": 9.0, "trade_time": "2026-08-03 09:29:59"},
                        {"ts_code": "000001.SZ", "pre_close": 20.0, "close": 20.2, "trade_time": "2026-08-03 09:35:00"},
                        {"ts_code": "300001.SZ", "pre_close": 30.0, "close": 29.0, "trade_time": "2026-08-03 09:20:00"},
                    ]
                )

        result = load_realtime_cross_section(
            MixedFreshness(), self.baseline(), AS_OF
        )

        self.assertEqual(result["ts_code"].tolist(), ["000001.SZ"])
        self.assertTrue(
            ((AS_OF - result["data_time"]).dt.total_seconds() <= 300).all()
        )


if __name__ == "__main__":
    main()
