from __future__ import annotations

import json
import sys
import tempfile
import unittest
from datetime import date, datetime, timedelta
from pathlib import Path
from unittest.mock import patch
from zoneinfo import ZoneInfo

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from tradingagents.dataflows import intraday_quote
from tradingagents.dataflows.intraday_quote import IntradayQuote
from tradingagents.harness.market_warning.adapters.data_cache import RawDataCache
from tradingagents.harness.market_warning.adapters.realtime_quote import RealtimeAShareDataAdapter
from tradingagents.harness.market_warning.adapters.tushare_data import TushareAShareDataAdapter
from tradingagents.harness.market_warning.adapters.us_market_data import YAHOO_TICKERS, YahooUSDataAdapter
from tradingagents.harness.market_warning.domain import DataStatus, Market, MarketDataPoint, RawMarketSnapshot
from tradingagents.harness.market_warning.quality import evaluate_data_quality


SHANGHAI = ZoneInfo("Asia/Shanghai")
NEW_YORK = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")


def _quote(source: str, last: float, at: datetime) -> IntradayQuote:
    return IntradayQuote(
        symbol="000001",
        name="SSE Composite",
        trade_date=at.date().isoformat(),
        quote_time=at,
        open=100.0,
        high=max(101.0, last),
        low=min(99.0, last),
        last=last,
        pre_close=100.0,
        volume=123456,
        amount=987654.0,
        source=source,
    )


class MockTusharePro:
    def __init__(self, *, include_daily: bool = True) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []
        self.include_daily = include_daily

    def _record(self, name: str, kwargs: dict[str, object]) -> None:
        self.calls.append((name, kwargs))

    def index_daily(self, **kwargs):
        self._record("index_daily", kwargs)
        return pd.DataFrame(
            [
                {
                    "ts_code": kwargs["ts_code"],
                    "trade_date": "20260720",
                    "open": 100.0,
                    "high": 103.0,
                    "low": 99.0,
                    "close": 102.0,
                    "pre_close": 101.0,
                    "pct_chg": 0.9901,
                    "vol": 1000.0,
                    "amount": 2000.0,
                }
            ]
        )

    def daily(self, **kwargs):
        self._record("daily", kwargs)
        if not self.include_daily:
            return pd.DataFrame()
        days = pd.bdate_range("2026-06-23", "2026-07-20")
        rows = []
        for index, day in enumerate(days):
            trade_date = day.strftime("%Y%m%d")
            rows.extend(
                [
                    {
                        "ts_code": "600000.SH",
                        "trade_date": trade_date,
                        "close": 10.0 + index,
                        "pre_close": 9.0 + index,
                        "pct_chg": 1.0,
                        "industry": "bank",
                    },
                    {
                        "ts_code": "000001.SZ",
                        "trade_date": trade_date,
                        "close": 40.0 - index,
                        "pre_close": 41.0 - index,
                        "pct_chg": -1.0,
                        "industry": "bank" if index % 2 else "broker",
                    },
                ]
            )
        return pd.DataFrame(rows)

    def daily_basic(self, **kwargs):
        self._record("daily_basic", kwargs)
        return pd.DataFrame(
            [
                {"ts_code": "600000.SH", "trade_date": "20260720", "turnover_rate": 2.0, "pe_ttm": 12.0},
                {"ts_code": "000001.SZ", "trade_date": "20260720", "turnover_rate": 4.0, "pe_ttm": 18.0},
            ]
        )

    def margin(self, **kwargs):
        self._record("margin", kwargs)
        return pd.DataFrame(
            [
                {"trade_date": "20260720", "exchange_id": "SSE", "rzye": 100.0, "rzmre": 10.0},
                {"trade_date": "20260720", "exchange_id": "SZSE", "rzye": 200.0, "rzmre": 20.0},
            ]
        )

    def shibor(self, **kwargs):
        self._record("shibor", kwargs)
        return pd.DataFrame([{"date": "20260720", "3m": 1.85}])

    def limit_list_d(self, **kwargs):
        self._record("limit_list_d", kwargs)
        return pd.DataFrame([{"trade_date": "20260720", "ts_code": "000001.SZ", "limit": "D"}])

    def moneyflow(self, **kwargs):
        self._record("moneyflow", kwargs)
        return pd.DataFrame(
            [
                {"trade_date": "20260720", "ts_code": "600000.SH", "net_mf_amount": -20.0},
                {"trade_date": "20260720", "ts_code": "000001.SZ", "net_mf_amount": 5.0},
            ]
        )


def _yahoo_frame() -> pd.DataFrame:
    timestamps = pd.DatetimeIndex(
        [datetime(2026, 7, 31, 15, 55, tzinfo=NEW_YORK), datetime(2026, 7, 31, 15, 59, tzinfo=NEW_YORK)]
    )
    columns = pd.MultiIndex.from_product(
        [["Open", "High", "Low", "Close", "Volume"], tuple(YAHOO_TICKERS.values())]
    )
    frame = pd.DataFrame(index=timestamps, columns=columns, dtype="float64")
    for offset, ticker in enumerate(YAHOO_TICKERS.values(), start=1):
        frame[("Open", ticker)] = [99.0 + offset, 100.0 + offset]
        frame[("High", ticker)] = [101.0 + offset, 102.0 + offset]
        frame[("Low", ticker)] = [98.0 + offset, 99.0 + offset]
        frame[("Close", ticker)] = [100.0 + offset, 101.0 + offset]
        frame[("Volume", ticker)] = [1000.0 * offset, 1100.0 * offset]
    frame.loc[timestamps[-1], ("Close", YAHOO_TICKERS["USD"])] = float("nan")
    return frame


class TushareDataAdapterTests(unittest.TestCase):
    def test_normalizes_all_disclosed_a_share_datasets_with_conservative_availability(self):
        pro = MockTusharePro()
        fetched_at = datetime(2026, 8, 1, 9, 0, tzinfo=SHANGHAI)
        as_of = datetime(2026, 7, 21, 8, 0, tzinfo=SHANGHAI)
        adapter = TushareAShareDataAdapter(pro=pro, clock=lambda: fetched_at)

        snapshot = adapter.load_snapshot(Market.A_SHARE, as_of, "premarket")

        fields = {point.field for point in snapshot.points}
        self.assertTrue(
            {
                "index_price",
                "index_change_pct",
                "open",
                "high",
                "low",
                "volume",
                "breadth_up_pct",
                "breadth_above_ma20_pct",
                "new_low_20d_pct",
                "industry_decline_pct",
                "valuation",
                "turnover_pct",
                "margin_balance",
                "margin_buying",
                "shibor_3m",
                "limit_down_pct",
                "money_flow_net",
            }.issubset(fields)
        )
        index_points = [point for point in snapshot.points if point.field == "index_price"]
        self.assertEqual(
            {point.symbol for point in index_points},
            {"000001.SH", "399001.SZ", "000300.SH", "000905.SH", "399006.SZ", "000688.SH"},
        )
        self.assertTrue(all(point.source == "tushare_index_daily" for point in index_points))
        self.assertEqual(index_points[0].data_time, datetime(2026, 7, 20, 15, 0, tzinfo=SHANGHAI))
        self.assertTrue(all(point.available_at == datetime(2026, 7, 21, 0, 0, tzinfo=SHANGHAI) for point in snapshot.points))
        self.assertTrue(all(point.available_at != point.fetched_at for point in snapshot.points))
        self.assertEqual(next(point.value for point in snapshot.points if point.field == "breadth_up_pct"), 50.0)
        self.assertEqual(next(point.value for point in snapshot.points if point.field == "valuation"), 15.0)
        self.assertEqual(next(point.value for point in snapshot.points if point.field == "margin_balance"), 300.0)
        self.assertEqual(next(point.value for point in snapshot.points if point.field == "money_flow_net"), -15.0)
        self.assertEqual(
            {name for name, _ in pro.calls},
            {"index_daily", "daily", "daily_basic", "margin", "shibor", "limit_list_d", "moneyflow"},
        )

    def test_missing_stock_cross_section_leaves_all_breadth_fields_missing(self):
        adapter = TushareAShareDataAdapter(pro=MockTusharePro(include_daily=False))

        snapshot = adapter.load_snapshot(
            Market.A_SHARE,
            datetime(2026, 7, 21, 8, 0, tzinfo=SHANGHAI),
            "premarket",
        )

        fields = {point.field for point in snapshot.points}
        self.assertTrue(
            {"breadth_up_pct", "breadth_above_ma20_pct", "new_low_20d_pct", "industry_decline_pct"}.isdisjoint(fields)
        )

    def test_auxiliary_queries_follow_latest_market_day_across_weekends(self):
        pro = MockTusharePro()
        adapter = TushareAShareDataAdapter(pro=pro)

        adapter.load_snapshot(
            Market.A_SHARE,
            datetime(2026, 7, 27, 8, 0, tzinfo=SHANGHAI),
            "premarket",
        )

        auxiliary = {
            name: kwargs["trade_date"]
            for name, kwargs in pro.calls
            if name in {"daily_basic", "margin", "limit_list_d", "moneyflow"}
        }
        self.assertEqual(
            auxiliary,
            {
                "daily_basic": "20260720",
                "margin": "20260720",
                "limit_list_d": "20260720",
                "moneyflow": "20260720",
            },
        )

    def test_backfill_resumes_from_cache_and_never_yields_future_visibility(self):
        pro = MockTusharePro()
        with tempfile.TemporaryDirectory() as temp_dir:
            cache = RawDataCache(Path(temp_dir))
            adapter = TushareAShareDataAdapter(pro=pro, cache=cache)
            first = tuple(adapter.backfill(date(2026, 7, 20), date(2026, 7, 20)))
            call_count = len(pro.calls)
            second = tuple(adapter.backfill(date(2026, 7, 20), date(2026, 7, 20)))

        self.assertEqual(first, second)
        self.assertEqual(len(pro.calls), call_count)
        self.assertEqual(len(first), 1)
        self.assertTrue(all(point.data_time <= point.available_at <= first[0].as_of_time for point in first[0].points))


class RealtimeAShareAdapterTests(unittest.TestCase):
    def setUp(self) -> None:
        intraday_quote._QUOTE_CACHE.clear()
        intraday_quote._TUSHARE_RT_K_DISABLED = False

    def tearDown(self) -> None:
        intraday_quote._QUOTE_CACHE.clear()
        intraday_quote._TUSHARE_RT_K_DISABLED = False

    def test_forbidden_tushare_rt_k_uses_existing_sina_fallback_and_preserves_quote_time(self):
        class ForbiddenPro:
            def rt_k(self, **kwargs):
                raise RuntimeError("rt_k permission denied")

        now = datetime(2026, 7, 20, 11, 31, tzinfo=SHANGHAI)
        sina_quote = _quote("sina_realtime", 101.0, now - timedelta(minutes=1))
        adapter = RealtimeAShareDataAdapter(pro=ForbiddenPro(), symbol="000001.SH")

        with patch.object(intraday_quote, "_fetch_sina_quote", return_value=sina_quote):
            snapshot = adapter.load_snapshot(Market.A_SHARE, now, "intraday")

        price = next(point for point in snapshot.points if point.field == "index_price")
        self.assertEqual(price.source, "sina_realtime")
        self.assertEqual(price.data_time, sina_quote.quote_time)
        self.assertNotIn("breadth_up_pct", {point.field for point in snapshot.points})

    def test_disagreeing_realtime_sources_are_retained_and_mark_snapshot_conflicted(self):
        now = datetime(2026, 7, 20, 10, 0, tzinfo=SHANGHAI)
        adapter = RealtimeAShareDataAdapter(
            symbol="000001.SH",
            quote_loader=lambda **_: _quote("primary", 100.0, now),
            secondary_quote_loader=lambda **_: _quote("secondary", 102.0, now),
        )

        snapshot = adapter.load_snapshot(Market.A_SHARE, now, "intraday")

        prices = [point for point in snapshot.points if point.field == "index_price"]
        self.assertEqual({point.source for point in prices}, {"primary", "secondary"})
        self.assertEqual(snapshot.data_status, DataStatus.CONFLICTED)
        self.assertEqual(evaluate_data_quality(snapshot).status, DataStatus.CONFLICTED)

    def test_realtime_cross_section_is_normalized_only_when_loader_returns_stock_rows(self):
        now = datetime(2026, 7, 20, 10, 0, tzinfo=SHANGHAI)
        stocks = pd.DataFrame(
            [
                {"ts_code": "600000.SH", "last": 11.0, "pre_close": 10.0, "ma20": 9.0, "low_20d": 8.0, "industry": "bank"},
                {"ts_code": "000001.SZ", "last": 9.0, "pre_close": 10.0, "ma20": 11.0, "low_20d": 9.0, "industry": "broker"},
            ]
        )
        adapter = RealtimeAShareDataAdapter(
            symbol="000001.SH",
            quote_loader=lambda **_: _quote("primary", 100.0, now),
            cross_section_loader=lambda _: stocks,
        )

        snapshot = adapter.load_snapshot(Market.A_SHARE, now, "intraday")

        values = {point.field: point.value for point in snapshot.points if point.symbol == "MARKET"}
        self.assertEqual(values["breadth_up_pct"], 50.0)
        self.assertEqual(values["breadth_above_ma20_pct"], 50.0)
        self.assertEqual(values["new_low_20d_pct"], 50.0)
        self.assertEqual(values["industry_decline_pct"], 50.0)

    def test_realtime_cross_section_excludes_rows_timestamped_after_as_of(self):
        now = datetime(2026, 7, 20, 10, 0, tzinfo=SHANGHAI)
        stocks = pd.DataFrame(
            [
                {"ts_code": "600000.SH", "last": 9.0, "pre_close": 10.0, "data_time": now - timedelta(seconds=1)},
                {"ts_code": "000001.SZ", "last": 11.0, "pre_close": 10.0, "data_time": now + timedelta(seconds=1)},
            ]
        )
        adapter = RealtimeAShareDataAdapter(
            symbol="000001.SH",
            quote_loader=lambda **_: _quote("primary", 100.0, now),
            cross_section_loader=lambda _: stocks,
        )

        snapshot = adapter.load_snapshot(Market.A_SHARE, now, "intraday")

        breadth = next(point for point in snapshot.points if point.field == "breadth_up_pct")
        self.assertEqual(breadth.value, 0.0)
        self.assertEqual(breadth.data_time, now - timedelta(seconds=1))


class YahooDataAdapterTests(unittest.TestCase):
    def test_batch_normalizes_explicit_proxy_map_and_uses_each_tickers_actual_last_timestamp(self):
        frame = _yahoo_frame()
        calls = []

        def download(**kwargs):
            calls.append(kwargs)
            return frame

        as_of = datetime(2026, 7, 31, 16, 1, tzinfo=NEW_YORK)
        fetched_at = as_of + timedelta(minutes=1)
        adapter = YahooUSDataAdapter(download=download, clock=lambda: fetched_at)

        snapshot = adapter.load_snapshot(Market.US, as_of, "intraday")

        self.assertEqual(len(calls), 1)
        self.assertEqual(set(calls[0]["tickers"].split()), set(YAHOO_TICKERS.values()))
        canonical_symbols = {point.symbol for point in snapshot.points}
        self.assertEqual(canonical_symbols, set(YAHOO_TICKERS))
        self.assertEqual(next(point.field for point in snapshot.points if point.symbol == "VIX"), "vix")
        self.assertEqual(next(point.field for point in snapshot.points if point.symbol == "VIX3M"), "vix3m")
        spx = next(point for point in snapshot.points if point.symbol == "SPX" and point.field == "index_price")
        usd = next(point for point in snapshot.points if point.symbol == "USD" and point.field == "index_price")
        self.assertEqual(spx.data_time, frame.index[-1].to_pydatetime())
        self.assertEqual(usd.data_time, frame.index[0].to_pydatetime())
        self.assertEqual(spx.available_at, spx.data_time)
        self.assertNotEqual(spx.data_time, fetched_at)
        self.assertEqual(spx.source, "yahoo_finance")
        self.assertEqual(snapshot.source_times["yahoo_finance"], frame.index[-1].to_pydatetime())

    def test_yahoo_only_us_intraday_snapshot_is_shadow_and_never_reliability_a(self):
        as_of = datetime(2026, 7, 31, 16, 1, tzinfo=NEW_YORK)
        adapter = YahooUSDataAdapter(download=lambda **_: _yahoo_frame())

        snapshot = adapter.load_snapshot(Market.US, as_of, "intraday")
        quality = evaluate_data_quality(snapshot)

        self.assertEqual(snapshot.data_status, DataStatus.SHADOW)
        self.assertEqual(quality.status, DataStatus.SHADOW)
        self.assertNotEqual(quality.reliability_grade, "A")
        self.assertEqual(quality.source_count, 1)

    def test_yahoo_backfill_resumes_from_cache_without_future_rows(self):
        calls = []

        def download(**kwargs):
            calls.append(kwargs)
            return _yahoo_frame()

        with tempfile.TemporaryDirectory() as temp_dir:
            cache = RawDataCache(Path(temp_dir))
            adapter = YahooUSDataAdapter(download=download, cache=cache)
            first = tuple(adapter.backfill(date(2026, 7, 31), date(2026, 7, 31)))
            second = tuple(adapter.backfill(date(2026, 7, 31), date(2026, 7, 31)))

        self.assertEqual(first, second)
        self.assertEqual(len(calls), 1)
        self.assertEqual(len(first), 1)
        self.assertTrue(all(point.data_time <= point.available_at <= first[0].as_of_time for point in first[0].points))


class RawDataCacheTests(unittest.TestCase):
    def test_atomic_write_creates_partitioned_data_and_complete_manifest(self):
        rows = [
            {
                "symbol": "SPX",
                "field": "index_price",
                "value": 100.0,
                "data_time": datetime(2026, 7, 31, 15, 59, tzinfo=NEW_YORK),
                "available_at": datetime(2026, 7, 31, 15, 59, tzinfo=NEW_YORK),
            }
        ]
        fetched_at = datetime(2026, 7, 31, 20, 0, tzinfo=UTC)
        with tempfile.TemporaryDirectory() as temp_dir:
            cache = RawDataCache(Path(temp_dir))
            with patch("os.replace", wraps=__import__("os").replace) as replace:
                data_path = cache.write_rows(
                    market=Market.US,
                    dataset="quotes",
                    year=2026,
                    cache_key="2026-07-31-intraday",
                    rows=rows,
                    query={"tickers": ["^GSPC"], "interval": "5m"},
                    source="yahoo_finance",
                    fetched_at=fetched_at,
                )
            manifest_path = data_path.with_suffix(".manifest.json")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            leftovers = list(data_path.parent.glob("*.tmp"))

        self.assertGreaterEqual(replace.call_count, 2)
        self.assertEqual(data_path.parts[-4:-1], ("us", "quotes", "2026"))
        self.assertEqual(manifest["query"], {"tickers": ["^GSPC"], "interval": "5m"})
        self.assertEqual(manifest["source"], "yahoo_finance")
        self.assertEqual(manifest["fetched_at"], fetched_at.isoformat())
        self.assertEqual(manifest["min_data_time"], rows[0]["data_time"].isoformat())
        self.assertEqual(manifest["max_data_time"], rows[0]["data_time"].isoformat())
        self.assertEqual(manifest["rows"], 1)
        self.assertEqual(manifest["schema_version"], "raw-market-cache-v1")
        self.assertEqual(leftovers, [])

    def test_snapshot_partition_uses_data_year_across_new_year_availability(self):
        point = MarketDataPoint(
            market=Market.A_SHARE,
            symbol="000001.SH",
            field="index_price",
            value=100.0,
            data_time=datetime(2026, 12, 31, 15, 0, tzinfo=SHANGHAI),
            available_at=datetime(2027, 1, 1, 0, 0, tzinfo=SHANGHAI),
            fetched_at=datetime(2027, 2, 1, 0, 0, tzinfo=SHANGHAI),
            source="tushare_index_daily",
        )
        snapshot = RawMarketSnapshot(
            market=Market.A_SHARE,
            as_of_time=datetime(2027, 1, 1, 0, 0, tzinfo=SHANGHAI),
            session_slot="premarket",
            points=(point,),
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            path = RawDataCache(Path(temp_dir)).write_snapshot(
                dataset="daily_snapshot",
                cache_key="2026-12-31",
                snapshot=snapshot,
                query={"trade_date": "20261231"},
                source="tushare",
                fetched_at=point.fetched_at,
            )

        self.assertEqual(path.parts[-2], "2026")


if __name__ == "__main__":
    unittest.main()
