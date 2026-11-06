import hashlib
import importlib
import importlib.util
import json
import math
from pathlib import Path
import tempfile
import unittest

import pandas as pd
from stockstats import wrap


def history(rows=520, end="2026-11-05"):
    dates = pd.bdate_range(end=end, periods=rows)
    closes = [30 + i * 0.03 + math.sin(i / 8) for i in range(rows)]
    return pd.DataFrame({
        "Date": dates.strftime("%Y-%m-%d"), "Open": [v - 0.1 for v in closes],
        "High": [v + 1 for v in closes], "Low": [v - 1 for v in closes],
        "Close": closes, "Volume": [100000 + (i % 30) * 1000 for i in range(rows)],
    })


def vendor_text(frame, status="official_daily", quote_time=None):
    return (
        f"# Stock data for 301171\n# Actual date range: {frame.Date.iloc[0]} to {frame.Date.iloc[-1]}\n"
        f"# Source: Tushare Pro\n# Price data status: {status}\n"
        f"# Latest bar source: {'sina_realtime' if quote_time else 'tushare_daily'}\n"
        + (f"# Latest quote time: {quote_time}\n" if quote_time else "")
        + frame.to_csv(index=False)
    )


class TechnicalContextTests(unittest.TestCase):
    def build(self, frame=None, raw=None, **kwargs):
        name = "tradingagents.dataflows.technical_context"
        self.assertIsNotNone(importlib.util.find_spec(name), "compact technical packet is missing")
        module = importlib.import_module(name)
        return module.build_technical_context(
            raw if raw is not None else vendor_text(frame if frame is not None else history()),
            symbol="301171", as_of="2026-11-05", audit_dir=Path(self.directory.name), **kwargs,
        )

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)

    def test_bounded_display_and_full_precision_audit(self):
        frame = history()
        packet = self.build(frame)
        self.assertEqual(len(packet["recent_daily"]), 20)
        self.assertEqual(len(packet["recent_weekly"]), 12)
        self.assertEqual(packet["price"]["current_price"], frame.Close.iloc[-1])
        self.assertLess(len(json.dumps(packet, ensure_ascii=False)), 16000)
        artifact = Path(packet["audit"]["path"])
        self.assertEqual(hashlib.sha256(artifact.read_bytes()).hexdigest(), packet["audit"]["sha256"])
        saved = json.loads(artifact.read_text())
        self.assertEqual(len(saved["ohlcv"]), len(frame))
        self.assertEqual(len(saved["indicators"]["rsi"]), len(frame))
        self.assertEqual(saved["raw_vendor_response"], vendor_text(frame))

    def test_values_and_percentiles_use_history_not_display_window(self):
        frame = history()
        packet = self.build(frame)
        expected = wrap(frame.copy())["rsi_14"]
        year = expected[pd.to_datetime(frame.Date) >= pd.Timestamp("2026-11-05") - pd.Timedelta(days=365)]
        rank = float((year <= expected.iloc[-1]).mean() * 100)
        self.assertAlmostEqual(packet["indicators"]["rsi"]["value"], expected.iloc[-1], places=5)
        self.assertAlmostEqual(packet["indicators"]["rsi"]["percentile_1y"], rank, places=5)
        self.assertGreater(packet["indicators"]["rsi"]["samples_1y"], 200)
        self.assertAlmostEqual(packet["indicators"]["close_200_sma"]["value"], frame.Close.tail(200).mean(), places=5)

    def test_future_rows_do_not_enter_calculations(self):
        frame = history()
        raw = vendor_text(frame) + "2026-11-06,99999,99999,99999,99999,99999\n"
        packet = self.build(raw=raw)
        baseline = self.build(frame)
        self.assertEqual(packet["indicators"], baseline["indicators"])
        self.assertEqual(packet["recent_daily"], baseline["recent_daily"])
        self.assertEqual(packet["coverage"]["end"], "2026-11-05")

    def test_provisional_and_t_minus_one_metadata_survive(self):
        for status, end, quote_time in (
            ("intraday_provisional", "2026-11-05", "2026-11-05 11:15:00"),
            ("t_minus_1", "2026-11-04", None),
        ):
            with self.subTest(status=status):
                packet = self.build(raw=vendor_text(history(end=end), status, quote_time))
                self.assertEqual(packet["price"]["status"], status)
                self.assertEqual(packet["price"]["date"], end)
                self.assertEqual(packet["price"]["time"], quote_time)
                self.assertTrue(packet["weekly"]["latest_partial"])

    def test_short_history_does_not_fake_long_averages_or_percentiles(self):
        packet = self.build(history(30))
        self.assertIsNone(packet["indicators"]["close_200_sma"]["value"])
        self.assertIsNone(packet["indicators"]["rsi"]["percentile_1y"])
        self.assertIsNone(packet["weekly"]["ma52"])
        self.assertTrue(packet["coverage"]["warnings"])

    def test_invalid_csv_and_bad_prices_fail_closed(self):
        bad = history()
        bad.loc[len(bad) - 1, "Close"] = float("inf")
        duplicate = pd.concat([history(), history().tail(1)], ignore_index=True)
        for raw in ("获取行情数据失败", vendor_text(bad), vendor_text(duplicate)):
            with self.subTest(raw=raw[:30]):
                with self.assertRaises(ValueError):
                    self.build(raw=raw)

    def test_strict_json_and_weekly_cutoff(self):
        packet = self.build(history(5))
        json.dumps(packet, allow_nan=False)
        self.assertLessEqual(packet["recent_weekly"][-1]["Date"], "2026-11-05")
        self.assertTrue(packet["weekly"]["latest_partial"])

    def test_corrupted_existing_artifact_is_not_silently_reused(self):
        packet = self.build()
        Path(packet["audit"]["path"]).write_text("corrupted")
        with self.assertRaisesRegex(ValueError, "audit|artifact|存档"):
            self.build()

    def test_week_ending_before_cutoff_is_complete_even_if_last_bar_is_thursday(self):
        frame = history(end="2026-10-29")
        packet = self.build(frame)
        self.assertFalse(packet["weekly"]["latest_partial"])
        weeks = frame.set_index(pd.to_datetime(frame.Date)).Close.resample("W-FRI").last().dropna()
        self.assertAlmostEqual(packet["weekly"]["ma52"], weeks.tail(52).mean(), places=5)


if __name__ == "__main__":
    unittest.main()
