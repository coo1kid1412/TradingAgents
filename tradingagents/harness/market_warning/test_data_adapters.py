from __future__ import annotations

import json
import hashlib
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
    """Tushare-shaped responses: daily is one cross-section per trade_date."""

    def __init__(
        self,
        *,
        latest_trade_date: str = "20260720",
        include_daily: bool = True,
        truncate_latest: bool = False,
        fail_index_once: bool = False,
        all_empty: bool = False,
    ) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []
        self.latest_trade_date = latest_trade_date
        self.include_daily = include_daily
        self.truncate_latest = truncate_latest
        self.fail_index_once = fail_index_once
        self.all_empty = all_empty
        latest = pd.Timestamp(latest_trade_date)
        self.trade_days = {day.strftime("%Y%m%d") for day in pd.bdate_range(end=latest, periods=21)}

    def _record(self, name: str, kwargs: dict[str, object]) -> None:
        self.calls.append((name, kwargs))

    def index_daily(self, **kwargs):
        self._record("index_daily", kwargs)
        if self.fail_index_once:
            self.fail_index_once = False
            raise RuntimeError("temporary index_daily failure")
        if self.all_empty:
            return pd.DataFrame()
        latest = pd.Timestamp(self.latest_trade_date)
        previous = (latest - pd.offsets.BDay()).strftime("%Y%m%d")
        return pd.DataFrame(
            [
                {
                    "ts_code": kwargs["ts_code"], "trade_date": self.latest_trade_date,
                    "open": 100.0, "high": 103.0, "low": 99.0, "close": 102.0,
                    "pre_close": 101.0, "change": 1.0, "pct_chg": 0.9901,
                    "vol": 1000.0, "amount": 2000.0,
                },
                {
                    "ts_code": kwargs["ts_code"], "trade_date": previous,
                    "open": 99.0, "high": 102.0, "low": 98.0, "close": 101.0,
                    "pre_close": 100.0, "change": 1.0, "pct_chg": 1.0,
                    "vol": 900.0, "amount": 1800.0,
                },
            ]
        )

    def daily(self, **kwargs):
        self._record("daily", kwargs)
        if self.all_empty or not self.include_daily or kwargs.get("trade_date") not in self.trade_days:
            return pd.DataFrame()
        trade_date = str(kwargs["trade_date"])
        if self.truncate_latest and trade_date == self.latest_trade_date:
            return pd.DataFrame(
                {
                    "ts_code": [f"{index:06d}.SZ" for index in range(6000)],
                    "trade_date": [trade_date] * 6000,
                    "open": [10.0] * 6000,
                    "high": [11.0] * 6000,
                    "low": [9.0] * 6000,
                    "close": [10.5] * 6000,
                    "pre_close": [10.0] * 6000,
                    "change": [0.5] * 6000,
                    "pct_chg": [5.0] * 6000,
                    "vol": [100.0] * 6000,
                    "amount": [1000.0] * 6000,
                }
            )
        day_index = sorted(self.trade_days).index(trade_date)
        return pd.DataFrame(
            [
                {
                    "ts_code": "600000.SH", "trade_date": trade_date,
                    "open": 9.5 + day_index, "high": 10.5 + day_index, "low": 9.0 + day_index,
                    "close": 10.0 + day_index, "pre_close": 9.0 + day_index,
                    "change": 1.0, "pct_chg": 1.0, "vol": 100.0, "amount": 1000.0,
                },
                {
                    "ts_code": "000001.SZ", "trade_date": trade_date,
                    "open": 40.5 - day_index, "high": 41.0 - day_index, "low": 39.0 - day_index,
                    "close": 40.0 - day_index, "pre_close": 41.0 - day_index,
                    "change": -1.0, "pct_chg": -1.0, "vol": 200.0, "amount": 2000.0,
                },
            ]
        )

    def stock_basic(self, **kwargs):
        self._record("stock_basic", kwargs)
        if self.all_empty:
            return pd.DataFrame()
        return pd.DataFrame(
            [
                {"ts_code": "600000.SH", "symbol": "600000", "name": "PFB", "area": "Shanghai", "industry": "bank", "list_status": "L"},
                {"ts_code": "000001.SZ", "symbol": "000001", "name": "PAB", "area": "Shenzhen", "industry": "broker", "list_status": "L"},
            ]
        )

    def daily_basic(self, **kwargs):
        self._record("daily_basic", kwargs)
        if self.all_empty:
            return pd.DataFrame()
        return pd.DataFrame(
            [
                {"ts_code": "600000.SH", "trade_date": self.latest_trade_date, "turnover_rate": 2.0, "pe_ttm": 12.0},
                {"ts_code": "000001.SZ", "trade_date": self.latest_trade_date, "turnover_rate": 4.0, "pe_ttm": 18.0},
            ]
        )

    def margin(self, **kwargs):
        self._record("margin", kwargs)
        if self.all_empty:
            return pd.DataFrame()
        return pd.DataFrame(
            [
                {"trade_date": self.latest_trade_date, "exchange_id": "SSE", "rzye": 100.0, "rzmre": 10.0},
                {"trade_date": self.latest_trade_date, "exchange_id": "SZSE", "rzye": 200.0, "rzmre": 20.0},
            ]
        )

    def shibor(self, **kwargs):
        self._record("shibor", kwargs)
        if self.all_empty:
            return pd.DataFrame()
        return pd.DataFrame([{"date": self.latest_trade_date, "on": 1.2, "1w": 1.4, "3m": 1.85}])

    def limit_list_d(self, **kwargs):
        self._record("limit_list_d", kwargs)
        if self.all_empty:
            return pd.DataFrame()
        return pd.DataFrame(
            [{"trade_date": self.latest_trade_date, "ts_code": "000001.SZ", "industry": "Bank", "name": "PAB", "close": 10.0, "pct_chg": -10.0, "limit": "D"}]
        )

    def moneyflow(self, **kwargs):
        self._record("moneyflow", kwargs)
        if self.all_empty:
            return pd.DataFrame()
        return pd.DataFrame(
            [
                {"trade_date": self.latest_trade_date, "ts_code": "600000.SH", "buy_sm_amount": 20.0, "sell_sm_amount": 40.0, "net_mf_amount": -20.0},
                {"trade_date": self.latest_trade_date, "ts_code": "000001.SZ", "buy_sm_amount": 15.0, "sell_sm_amount": 10.0, "net_mf_amount": 5.0},
            ]
        )


def _yahoo_frame(
    *, interval: str, ticker_map: dict[str, str] | None = None, order: str = "price_ticker"
) -> pd.DataFrame:
    mapping = ticker_map or YAHOO_TICKERS
    if interval == "1d":
        index = pd.DatetimeIndex(["2026-07-29", "2026-07-30", "2026-07-31"])
        closes = [98.0, 99.0, 100.0]
    else:
        index = pd.DatetimeIndex(
            [datetime(2026, 8, 3, 9, 55, tzinfo=NEW_YORK), datetime(2026, 8, 3, 10, 0, tzinfo=NEW_YORK)]
        )
        closes = [109.0, 110.0]
    fields = ["Open", "High", "Low", "Close", "Volume"]
    if len(mapping) == 1 and order == "flat":
        frame = pd.DataFrame(index=index, columns=fields, dtype="float64")
        frame["Open"] = [value - 1 for value in closes]
        frame["High"] = [value + 1 for value in closes]
        frame["Low"] = [value - 2 for value in closes]
        frame["Close"] = closes
        frame["Volume"] = [1000.0 + index for index in range(len(closes))]
        return frame
    tickers = tuple(mapping.values())
    columns = (
        pd.MultiIndex.from_product([fields, tickers])
        if order == "price_ticker"
        else pd.MultiIndex.from_product([tickers, fields])
    )
    frame = pd.DataFrame(index=index, columns=columns, dtype="float64")
    for offset, ticker in enumerate(tickers):
        ticker_closes = [value + offset for value in closes]
        for field, values in {
            "Open": [value - 1 for value in ticker_closes],
            "High": [value + 1 for value in ticker_closes],
            "Low": [value - 2 for value in ticker_closes],
            "Close": ticker_closes,
            "Volume": [1000.0 * (offset + 1) + index for index in range(len(closes))],
        }.items():
            frame[(field, ticker) if order == "price_ticker" else (ticker, field)] = values
    return frame


def _snapshot_for_cache(*, as_of: datetime, data_time: datetime, value: float = 100.0) -> RawMarketSnapshot:
    point = MarketDataPoint(
        market=Market.US,
        symbol="SPX",
        field="index_price",
        value=value,
        data_time=data_time,
        available_at=data_time,
        fetched_at=as_of,
        source="yahoo_finance",
    )
    return RawMarketSnapshot(
        market=Market.US,
        as_of_time=as_of,
        session_slot="close",
        points=(point,),
        source_times={"yahoo_finance": data_time},
    )


class TushareDataAdapterTests(unittest.TestCase):
    def test_normalizes_realistic_responses_with_per_dataset_availability_and_daily_loops(self):
        pro = MockTusharePro()
        fetched_at = datetime(2026, 7, 21, 9, 5, tzinfo=SHANGHAI)
        as_of = datetime(2026, 7, 21, 9, 0, tzinfo=SHANGHAI)
        adapter = TushareAShareDataAdapter(pro=pro, clock=lambda: fetched_at)

        snapshot = adapter.load_snapshot(Market.A_SHARE, as_of, "premarket")

        fields = {point.field for point in snapshot.points}
        self.assertTrue(
            {
                "index_price", "index_change_pct", "open", "high", "low", "volume",
                "breadth_up_pct", "breadth_above_ma20_pct", "new_low_20d_pct",
                "industry_decline_pct", "valuation", "turnover_pct", "margin_balance",
                "margin_buying", "shibor_3m", "limit_down_pct", "money_flow_net",
            }.issubset(fields)
        )
        daily_calls = [kwargs for name, kwargs in pro.calls if name == "daily"]
        self.assertGreaterEqual(len(daily_calls), 40)
        self.assertTrue(all(set(kwargs) == {"trade_date"} for kwargs in daily_calls))
        self.assertIn("stock_basic", {name for name, _ in pro.calls})
        index_point = next(point for point in snapshot.points if point.field == "index_price" and point.symbol == "000001.SH")
        margin_point = next(point for point in snapshot.points if point.field == "margin_balance")
        shibor_point = next(point for point in snapshot.points if point.field == "shibor_3m")
        self.assertEqual(index_point.data_time, datetime(2026, 7, 20, 15, 0, tzinfo=SHANGHAI))
        self.assertEqual(index_point.available_at, datetime(2026, 7, 20, 18, 0, tzinfo=SHANGHAI))
        self.assertEqual(margin_point.available_at, datetime(2026, 7, 21, 9, 0, tzinfo=SHANGHAI))
        self.assertEqual(shibor_point.available_at, datetime(2026, 7, 20, 12, 0, tzinfo=SHANGHAI))
        expected_availability = {
            "tushare_index_daily": datetime(2026, 7, 20, 18, 0, tzinfo=SHANGHAI),
            "tushare_daily": datetime(2026, 7, 20, 18, 0, tzinfo=SHANGHAI),
            "tushare_daily_basic": datetime(2026, 7, 21, 0, 0, tzinfo=SHANGHAI),
            "tushare_margin": datetime(2026, 7, 21, 9, 0, tzinfo=SHANGHAI),
            "tushare_shibor": datetime(2026, 7, 20, 12, 0, tzinfo=SHANGHAI),
            "tushare_limit_list": datetime(2026, 7, 20, 18, 0, tzinfo=SHANGHAI),
            "tushare_moneyflow": datetime(2026, 7, 20, 19, 0, tzinfo=SHANGHAI),
        }
        for source, available_at in expected_availability.items():
            self.assertEqual(
                {point.available_at for point in snapshot.points if point.source == source},
                {available_at},
            )
        self.assertTrue(all(point.fetched_at == fetched_at for point in snapshot.points))
        self.assertEqual(next(point.value for point in snapshot.points if point.field == "breadth_up_pct"), 50.0)
        self.assertEqual(next(point.value for point in snapshot.points if point.field == "industry_decline_pct"), 50.0)

    def test_margin_is_hidden_monday_0830_and_visible_monday_0900_after_friday(self):
        pro = MockTusharePro(latest_trade_date="20260717")
        adapter = TushareAShareDataAdapter(pro=pro)
        before = adapter.load_snapshot(
            Market.A_SHARE, datetime(2026, 7, 20, 8, 30, tzinfo=SHANGHAI), "premarket"
        )
        at_disclosure = adapter.load_snapshot(
            Market.A_SHARE, datetime(2026, 7, 20, 9, 0, tzinfo=SHANGHAI), "premarket"
        )

        self.assertNotIn("margin_balance", {point.field for point in before.points})
        margin = next(point for point in at_disclosure.points if point.field == "margin_balance")
        self.assertEqual(margin.available_at, datetime(2026, 7, 20, 9, 0, tzinfo=SHANGHAI))

    def test_moneyflow_is_hidden_at_1859_and_visible_at_1900(self):
        adapter = TushareAShareDataAdapter(pro=MockTusharePro())
        before = adapter.load_snapshot(
            Market.A_SHARE, datetime(2026, 7, 20, 18, 59, tzinfo=SHANGHAI), "post_market"
        )
        at_disclosure = adapter.load_snapshot(
            Market.A_SHARE, datetime(2026, 7, 20, 19, 0, tzinfo=SHANGHAI), "post_market"
        )

        self.assertNotIn("money_flow_net", {point.field for point in before.points})
        moneyflow = next(point for point in at_disclosure.points if point.field == "money_flow_net")
        self.assertEqual(moneyflow.available_at, datetime(2026, 7, 20, 19, 0, tzinfo=SHANGHAI))

    def test_truncated_6000_row_cross_section_never_emits_partial_breadth(self):
        snapshot = TushareAShareDataAdapter(pro=MockTusharePro(truncate_latest=True)).load_snapshot(
            Market.A_SHARE, datetime(2026, 7, 21, 9, 0, tzinfo=SHANGHAI), "premarket"
        )

        breadth_fields = {"breadth_up_pct", "breadth_above_ma20_pct", "new_low_20d_pct", "industry_decline_pct"}
        self.assertTrue(breadth_fields.isdisjoint({point.field for point in snapshot.points}))

    def test_missing_stock_cross_section_leaves_breadth_missing(self):
        snapshot = TushareAShareDataAdapter(pro=MockTusharePro(include_daily=False)).load_snapshot(
            Market.A_SHARE, datetime(2026, 7, 21, 9, 0, tzinfo=SHANGHAI), "premarket"
        )

        breadth_fields = {"breadth_up_pct", "breadth_above_ma20_pct", "new_low_20d_pct", "industry_decline_pct"}
        self.assertTrue(breadth_fields.isdisjoint({point.field for point in snapshot.points}))

    def test_backfill_uses_disclosure_time_reuses_fetch_clock_and_resumes_cache(self):
        pro = MockTusharePro()
        clock_calls = []

        def clock():
            clock_calls.append(1)
            if len(clock_calls) > 1:
                raise AssertionError("clock called more than once for one fetch")
            return datetime(2026, 8, 1, 9, 0, tzinfo=SHANGHAI)

        with tempfile.TemporaryDirectory() as temp_dir:
            adapter = TushareAShareDataAdapter(pro=pro, cache=RawDataCache(Path(temp_dir)), clock=clock)
            first = tuple(adapter.backfill(date(2026, 7, 20), date(2026, 7, 20)))
            call_count = len(pro.calls)
            second = tuple(adapter.backfill(date(2026, 7, 20), date(2026, 7, 20)))

        self.assertEqual(first, second)
        self.assertEqual(len(pro.calls), call_count)
        self.assertEqual(first[0].as_of_time, datetime(2026, 7, 21, 9, 0, tzinfo=SHANGHAI))
        self.assertTrue(all(point.available_at <= first[0].as_of_time for point in first[0].points))

    def test_tushare_exception_is_not_cached_and_next_backfill_retries(self):
        pro = MockTusharePro(fail_index_once=True)
        with tempfile.TemporaryDirectory() as temp_dir:
            adapter = TushareAShareDataAdapter(pro=pro, cache=RawDataCache(Path(temp_dir)))
            tuple(adapter.backfill(date(2026, 7, 20), date(2026, 7, 20)))
            first_count = len(pro.calls)
            tuple(adapter.backfill(date(2026, 7, 20), date(2026, 7, 20)))

        self.assertGreater(len(pro.calls), first_count)

    def test_empty_core_tushare_result_is_not_cached_and_retries(self):
        pro = MockTusharePro(all_empty=True)
        with tempfile.TemporaryDirectory() as temp_dir:
            adapter = TushareAShareDataAdapter(pro=pro, cache=RawDataCache(Path(temp_dir)))
            self.assertEqual(tuple(adapter.backfill(date(2026, 7, 20), date(2026, 7, 20))), ())
            first_count = len(pro.calls)
            self.assertEqual(tuple(adapter.backfill(date(2026, 7, 20), date(2026, 7, 20))), ())

        self.assertGreater(len(pro.calls), first_count)

    def test_unusable_nonempty_core_index_rows_are_not_cached_and_retry(self):
        class InvalidCorePro(MockTusharePro):
            def __init__(self, invalid_shape: str) -> None:
                super().__init__()
                self.invalid_shape = invalid_shape

            def index_daily(self, **kwargs):
                frame = super().index_daily(**kwargs)
                if self.invalid_shape == "missing":
                    return frame.drop(columns=["close", "pct_chg"])
                if self.invalid_shape == "null":
                    frame.loc[:, ["close", "pct_chg"]] = None
                elif self.invalid_shape == "non_finite_derived":
                    frame.loc[:, "close"] = 1e308
                    frame.loc[:, "pre_close"] = 1e-308
                    frame.loc[:, "pct_chg"] = float("nan")
                else:
                    frame.loc[:, "close"] = float("inf")
                    frame.loc[:, "pct_chg"] = float("nan")
                return frame

        for invalid_shape in ("missing", "null", "non_finite", "non_finite_derived"):
            with self.subTest(invalid_shape=invalid_shape), tempfile.TemporaryDirectory() as temp_dir:
                pro = InvalidCorePro(invalid_shape)
                root = Path(temp_dir)
                adapter = TushareAShareDataAdapter(pro=pro, cache=RawDataCache(root))

                tuple(adapter.backfill(date(2026, 7, 20), date(2026, 7, 20)))
                first_count = len(pro.calls)
                tuple(adapter.backfill(date(2026, 7, 20), date(2026, 7, 20)))

                self.assertGreater(len(pro.calls), first_count)
                self.assertEqual(list(root.rglob("*.manifest.json")), [])

    def test_historical_backfill_does_not_apply_current_stock_basic_industry(self):
        pro = MockTusharePro()
        adapter = TushareAShareDataAdapter(
            pro=pro,
            clock=lambda: datetime(2026, 8, 1, 9, 0, tzinfo=SHANGHAI),
        )

        result = tuple(adapter.backfill(date(2026, 7, 20), date(2026, 7, 20)))

        self.assertEqual(len(result), 1)
        self.assertNotIn("industry_decline_pct", {point.field for point in result[0].points})

    def test_calendar_resolver_controls_premarket_expected_session(self):
        friday = date(2026, 7, 17)
        pro = MockTusharePro(latest_trade_date="20260717")
        adapter = TushareAShareDataAdapter(
            pro=pro,
            previous_session=lambda current: friday,
            calendar_version="exchange-calendar-test-v1",
        )

        snapshot = adapter.load_snapshot(
            Market.A_SHARE, datetime(2026, 7, 21, 8, 30, tzinfo=SHANGHAI), "premarket"
        )

        self.assertEqual(snapshot.data_status, DataStatus.FRESH)
        index_calls = [kwargs for name, kwargs in pro.calls if name == "index_daily"]
        self.assertTrue(index_calls)
        self.assertEqual({kwargs["end_date"] for kwargs in index_calls}, {"20260717"})

    def test_new_year_backfill_finds_t_minus_one_cache_in_previous_data_year(self):
        as_of = datetime(2027, 1, 1, 9, 0, tzinfo=SHANGHAI)
        expected_date = date(2026, 12, 31)
        point = MarketDataPoint(
            market=Market.A_SHARE, symbol="000001.SH", field="index_price", value=100.0,
            data_time=datetime(2026, 12, 31, 15, 0, tzinfo=SHANGHAI),
            available_at=datetime(2026, 12, 31, 18, 0, tzinfo=SHANGHAI),
            fetched_at=as_of, source="tushare_index_daily",
        )
        cached = RawMarketSnapshot(
            market=Market.A_SHARE, as_of_time=as_of, session_slot="premarket", points=(point,)
        )
        pro = MockTusharePro(all_empty=True)
        with tempfile.TemporaryDirectory() as temp_dir:
            cache = RawDataCache(Path(temp_dir))
            adapter = TushareAShareDataAdapter(
                pro=pro,
                cache=cache,
                next_trading_day=lambda current: current,
                previous_session=lambda current: expected_date,
                calendar_version="exchange-calendar-test-v1",
            )
            query = adapter._cache_query(date(2027, 1, 1), as_of, expected_date)
            cache.write_snapshot(
                dataset="daily_snapshot", cache_key="2027-01-01", snapshot=cached,
                query=query, source="tushare", fetched_at=as_of,
            )
            result = tuple(adapter.backfill(date(2027, 1, 1), date(2027, 1, 1)))

        self.assertEqual(result, (cached,))
        self.assertEqual(pro.calls, [])


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

    def test_disagreeing_realtime_sources_remain_conflicted_with_lineage(self):
        now = datetime(2026, 7, 20, 10, 0, tzinfo=SHANGHAI)
        snapshot = RealtimeAShareDataAdapter(
            symbol="000001.SH",
            quote_loader=lambda **_: _quote("primary", 100.0, now),
            secondary_quote_loader=lambda **_: _quote("secondary", 102.0, now),
        ).load_snapshot(Market.A_SHARE, now, "intraday")

        self.assertEqual({point.source for point in snapshot.points if point.field == "index_price"}, {"primary", "secondary"})
        self.assertEqual(snapshot.data_status, DataStatus.CONFLICTED)

    def test_cross_section_without_parseable_data_time_emits_no_breadth(self):
        now = datetime(2026, 7, 20, 10, 0, tzinfo=SHANGHAI)
        stocks = pd.DataFrame(
            [{"ts_code": "600000.SH", "last": 11.0, "pre_close": 10.0, "data_time": "not-a-time", "source": "vendor_x"}]
        )
        snapshot = RealtimeAShareDataAdapter(
            symbol="000001.SH", quote_loader=lambda **_: _quote("primary", 100.0, now),
            cross_section_loader=lambda _: stocks,
        ).load_snapshot(Market.A_SHARE, now, "intraday")

        self.assertNotIn("breadth_up_pct", {point.field for point in snapshot.points})

    def test_cross_section_preserves_vendor_source_time_and_excludes_future_rows(self):
        now = datetime(2026, 7, 20, 10, 0, tzinfo=SHANGHAI)
        visible_time = now - timedelta(seconds=1)
        stocks = pd.DataFrame(
            [
                {"ts_code": "600000.SH", "last": 9.0, "pre_close": 10.0, "ma20": 11.0, "low_20d": 9.0, "industry": "bank", "data_time": visible_time, "source": "vendor_x"},
                {"ts_code": "000001.SZ", "last": 11.0, "pre_close": 10.0, "ma20": 9.0, "low_20d": 8.0, "industry": "broker", "data_time": now + timedelta(seconds=1), "source": "vendor_x"},
            ]
        )
        snapshot = RealtimeAShareDataAdapter(
            symbol="000001.SH", quote_loader=lambda **_: _quote("primary", 100.0, now),
            cross_section_loader=lambda _: stocks,
        ).load_snapshot(Market.A_SHARE, now, "intraday")

        breadth = next(point for point in snapshot.points if point.field == "breadth_up_pct")
        self.assertEqual(breadth.value, 0.0)
        self.assertEqual(breadth.data_time, visible_time)
        self.assertEqual(breadth.source, "vendor_x")

    def test_cross_section_lineage_ignores_newer_rows_with_invalid_prices(self):
        now = datetime(2026, 7, 20, 10, 0, tzinfo=SHANGHAI)
        valid_time = now - timedelta(seconds=2)
        stocks = pd.DataFrame(
            [
                {"last": 9.0, "pre_close": 10.0, "data_time": valid_time, "source": "vendor_x"},
                {"last": None, "pre_close": 10.0, "data_time": now - timedelta(seconds=1), "source": "bad_row"},
            ]
        )
        snapshot = RealtimeAShareDataAdapter(
            quote_loader=lambda **_: _quote("primary", 100.0, now),
            cross_section_loader=lambda _: stocks,
        ).load_snapshot(Market.A_SHARE, now, "intraday")

        breadth = next(point for point in snapshot.points if point.field == "breadth_up_pct")
        self.assertEqual(breadth.data_time, valid_time)
        self.assertEqual(breadth.source, "vendor_x")


class YahooDataAdapterTests(unittest.TestCase):
    def _download(self, *, order: str = "price_ticker", ticker_map: dict[str, str] | None = None):
        calls = []

        def download(**kwargs):
            calls.append(kwargs)
            return _yahoo_frame(interval=kwargs["interval"], ticker_map=ticker_map, order=order)

        return download, calls

    def test_premarket_fetches_ten_calendar_days_and_emits_t_minus_one_change(self):
        download, calls = self._download()
        as_of = datetime(2026, 8, 3, 8, 30, tzinfo=NEW_YORK)
        snapshot = YahooUSDataAdapter(download=download).load_snapshot(Market.US, as_of, "premarket")

        self.assertEqual(len(calls), 1)
        self.assertLessEqual(date.fromisoformat(calls[0]["start"]), as_of.date() - timedelta(days=10))
        self.assertEqual(calls[0]["end"], (as_of.date() + timedelta(days=1)).isoformat())
        self.assertEqual(YAHOO_TICKERS["NDX"], "^NDX")
        spx = {point.field: point for point in snapshot.points if point.symbol == "SPX"}
        self.assertEqual(spx["index_price"].data_time, datetime(2026, 7, 31, 16, 0, tzinfo=NEW_YORK))
        self.assertAlmostEqual(spx["index_change_pct"].value, (100.0 / 99.0 - 1.0) * 100.0)
        self.assertEqual(snapshot.data_status, DataStatus.FRESH)

    def test_intraday_change_uses_previous_complete_daily_close_not_previous_5m_bar(self):
        download, calls = self._download()
        as_of = datetime(2026, 8, 3, 10, 0, tzinfo=NEW_YORK)
        snapshot = YahooUSDataAdapter(download=download).load_snapshot(Market.US, as_of, "intraday")

        self.assertEqual({call["interval"] for call in calls}, {"1d", "5m"})
        spx = {point.field: point for point in snapshot.points if point.symbol == "SPX"}
        self.assertEqual(spx["index_price"].value, 110.0)
        self.assertAlmostEqual(spx["index_change_pct"].value, 10.0)
        self.assertEqual(snapshot.data_status, DataStatus.SHADOW)

    def test_supports_both_multiindex_orders(self):
        as_of = datetime(2026, 8, 3, 8, 30, tzinfo=NEW_YORK)
        for order in ("price_ticker", "ticker_price"):
            with self.subTest(order=order):
                download, _ = self._download(order=order)
                snapshot = YahooUSDataAdapter(download=download).load_snapshot(Market.US, as_of, "premarket")
                self.assertEqual({point.symbol for point in snapshot.points}, set(YAHOO_TICKERS))

    def test_supports_flat_columns_for_injected_single_ticker_map(self):
        ticker_map = {"SPX": "^GSPC"}
        download, _ = self._download(order="flat", ticker_map=ticker_map)
        snapshot = YahooUSDataAdapter(download=download, ticker_map=ticker_map).load_snapshot(
            Market.US, datetime(2026, 8, 3, 8, 30, tzinfo=NEW_YORK), "premarket"
        )

        self.assertIn("index_price", {point.field for point in snapshot.points})
        self.assertIn("index_change_pct", {point.field for point in snapshot.points})

    def test_daily_symbol_with_fewer_than_two_rows_is_not_normalized(self):
        frame = _yahoo_frame(interval="1d").iloc[-1:]
        snapshot = YahooUSDataAdapter(download=lambda **_: frame).load_snapshot(
            Market.US, datetime(2026, 8, 3, 8, 30, tzinfo=NEW_YORK), "premarket"
        )

        self.assertEqual(snapshot.data_status, DataStatus.INSUFFICIENT)
        self.assertEqual(snapshot.points, ())

    def test_monday_close_without_monday_bar_is_insufficient_and_not_cached(self):
        calls = []

        def friday_only(**kwargs):
            calls.append(kwargs)
            return _yahoo_frame(interval=kwargs["interval"])

        with tempfile.TemporaryDirectory() as temp_dir:
            adapter = YahooUSDataAdapter(download=friday_only, cache=RawDataCache(Path(temp_dir)))
            first = tuple(adapter.backfill(date(2026, 8, 3), date(2026, 8, 3)))
            second = tuple(adapter.backfill(date(2026, 8, 3), date(2026, 8, 3)))

        self.assertEqual(first, ())
        self.assertEqual(second, ())
        self.assertEqual(len(calls), 2)

    def test_empty_yahoo_core_batch_is_not_cached_and_retries(self):
        calls = []

        def empty(**kwargs):
            calls.append(kwargs)
            return pd.DataFrame()

        with tempfile.TemporaryDirectory() as temp_dir:
            adapter = YahooUSDataAdapter(download=empty, cache=RawDataCache(Path(temp_dir)))
            self.assertEqual(tuple(adapter.backfill(date(2026, 7, 31), date(2026, 7, 31))), ())
            self.assertEqual(tuple(adapter.backfill(date(2026, 7, 31), date(2026, 7, 31))), ())

        self.assertEqual(len(calls), 2)

    def test_yahoo_exception_and_none_return_insufficient_without_raising(self):
        def failed(**kwargs):
            raise RuntimeError("temporary Yahoo failure")

        for download in (failed, lambda **_: None):
            with self.subTest(download=download):
                snapshot = YahooUSDataAdapter(download=download).load_snapshot(
                    Market.US, datetime(2026, 8, 3, 8, 30, tzinfo=NEW_YORK), "premarket"
                )
                self.assertEqual(snapshot.data_status, DataStatus.INSUFFICIENT)
                self.assertEqual(snapshot.points, ())

    def test_yahoo_exception_is_not_cached_and_next_backfill_retries(self):
        calls = []

        def flaky(**kwargs):
            calls.append(kwargs)
            if len(calls) == 1:
                raise RuntimeError("temporary Yahoo failure")
            return _yahoo_frame(interval=kwargs["interval"])

        with tempfile.TemporaryDirectory() as temp_dir:
            adapter = YahooUSDataAdapter(download=flaky, cache=RawDataCache(Path(temp_dir)))
            self.assertEqual(tuple(adapter.backfill(date(2026, 7, 31), date(2026, 7, 31))), ())
            result = tuple(adapter.backfill(date(2026, 7, 31), date(2026, 7, 31)))

        self.assertEqual(len(result), 1)
        self.assertGreaterEqual(len(calls), 2)

    def test_yahoo_partial_daily_response_is_not_cached_and_next_backfill_retries(self):
        calls = []

        def partial_once(**kwargs):
            calls.append(kwargs)
            frame = _yahoo_frame(interval=kwargs["interval"])
            return frame.iloc[-1:] if len(calls) == 1 else frame

        with tempfile.TemporaryDirectory() as temp_dir:
            adapter = YahooUSDataAdapter(download=partial_once, cache=RawDataCache(Path(temp_dir)))
            self.assertEqual(tuple(adapter.backfill(date(2026, 7, 31), date(2026, 7, 31))), ())
            result = tuple(adapter.backfill(date(2026, 7, 31), date(2026, 7, 31)))

        self.assertEqual(len(result), 1)
        self.assertEqual(len(calls), 2)

    def test_yahoo_backfill_resumes_complete_cache_without_future_rows(self):
        download, calls = self._download()
        with tempfile.TemporaryDirectory() as temp_dir:
            adapter = YahooUSDataAdapter(download=download, cache=RawDataCache(Path(temp_dir)))
            first = tuple(adapter.backfill(date(2026, 7, 31), date(2026, 7, 31)))
            call_count = len(calls)
            second = tuple(adapter.backfill(date(2026, 7, 31), date(2026, 7, 31)))

        self.assertEqual(first, second)
        self.assertEqual(len(calls), call_count)
        self.assertTrue(all(point.data_time <= point.available_at <= first[0].as_of_time for point in first[0].points))

    def test_yahoo_backfill_manifest_reuses_snapshot_fetch_time(self):
        download, _ = self._download()
        fetched_at = datetime(2026, 8, 1, 12, 0, tzinfo=NEW_YORK)
        clock_calls = []

        def clock():
            clock_calls.append(1)
            if len(clock_calls) > 1:
                raise AssertionError("clock called more than once")
            return fetched_at

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            adapter = YahooUSDataAdapter(download=download, cache=RawDataCache(root), clock=clock)
            tuple(adapter.backfill(date(2026, 7, 31), date(2026, 7, 31)))
            manifest_path = next(root.glob("us/daily_snapshot/2026/*.manifest.json"))
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        self.assertEqual(manifest["fetched_at"], fetched_at.isoformat())

    def test_cache_query_records_policy_calendar_asof_and_expected_date_and_version_change_misses(self):
        first_download, first_calls = self._download()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            cache = RawDataCache(root)
            first = YahooUSDataAdapter(
                download=first_download,
                cache=cache,
                calendar_version="weekday-fallback-v1",
                disclosure_policy_version="yahoo-disclosure-v2",
            )
            tuple(first.backfill(date(2026, 7, 31), date(2026, 7, 31)))
            manifest_path = next(root.glob("us/daily_snapshot/2026/*.manifest.json"))
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

            second_download, second_calls = self._download()
            second = YahooUSDataAdapter(
                download=second_download,
                cache=cache,
                calendar_version="exchange-calendar-v1",
                disclosure_policy_version="yahoo-disclosure-v2",
            )
            tuple(second.backfill(date(2026, 7, 31), date(2026, 7, 31)))

        self.assertEqual(len(first_calls), 1)
        self.assertEqual(len(second_calls), 1)
        self.assertEqual(manifest["query"]["calendar_version"], "weekday-fallback-v1")
        self.assertEqual(manifest["query"]["disclosure_policy_version"], "yahoo-disclosure-v2")
        self.assertEqual(manifest["query"]["as_of_time"], "2026-07-31T16:00:00-04:00")
        self.assertEqual(manifest["query"]["expected_observation_date"], "2026-07-31")


class RawDataCacheTests(unittest.TestCase):
    def test_manifest_contains_generation_hash_and_canonical_query(self):
        query = {
            "tickers": ["^GSPC"],
            "start": date(2026, 7, 31),
            "options": {"interval": "5m", "adjust": False},
        }
        rows = [{"symbol": "SPX", "data_time": datetime(2026, 7, 31, 15, 59, tzinfo=NEW_YORK)}]
        fetched_at = datetime(2026, 7, 31, 20, 0, tzinfo=UTC)
        with tempfile.TemporaryDirectory() as temp_dir:
            cache = RawDataCache(Path(temp_dir))
            with patch("os.replace", wraps=__import__("os").replace) as replace:
                data_path = cache.write_rows(
                    market=Market.US, dataset="quotes", year=2026, cache_key="intraday",
                    rows=rows, query=query, source="yahoo_finance", fetched_at=fetched_at,
                )
            manifest = json.loads(data_path.with_suffix(".manifest.json").read_text(encoding="utf-8"))
            hit = cache.read_rows(
                market=Market.US, dataset="quotes", year=2026, cache_key="intraday",
                query={
                    "options": {"adjust": False, "interval": "5m"},
                    "start": date(2026, 7, 31),
                    "tickers": ["^GSPC"],
                },
            )

        self.assertGreaterEqual(replace.call_count, 2)
        self.assertEqual(manifest["schema_version"], "raw-market-cache-v2")
        self.assertTrue(manifest["generation"])
        self.assertEqual(len(manifest["data_sha256"]), 64)
        self.assertEqual(
            manifest["query"],
            {"tickers": ["^GSPC"], "start": "2026-07-31", "options": {"interval": "5m", "adjust": False}},
        )
        self.assertIsNotNone(hit)

    def test_mixed_generations_damaged_json_query_and_row_count_are_cache_misses(self):
        query = {"trade_date": "20260731"}
        with tempfile.TemporaryDirectory() as temp_dir:
            cache = RawDataCache(Path(temp_dir))
            first = cache.write_rows(
                market=Market.US, dataset="quotes", year=2026, cache_key="daily",
                rows=[{"value": 100.0, "data_time": "2026-07-31T16:00:00-04:00"}],
                query=query, source="yahoo_finance", fetched_at=datetime(2026, 8, 1, tzinfo=UTC),
            )
            manifest_path = first.with_suffix(".manifest.json")
            data_v1, manifest_v1 = first.read_bytes(), manifest_path.read_bytes()
            cache.write_rows(
                market=Market.US, dataset="quotes", year=2026, cache_key="daily",
                rows=[{"value": 101.0, "data_time": "2026-07-31T16:00:00-04:00"}],
                query=query, source="yahoo_finance", fetched_at=datetime(2026, 8, 2, tzinfo=UTC),
            )
            data_v2, manifest_v2 = first.read_bytes(), manifest_path.read_bytes()

            cases = (
                (data_v2, manifest_v1, query),
                (data_v1, manifest_v2, query),
                (b"{damaged json\n", manifest_v2, query),
                (data_v2, b"{damaged manifest", query),
                (data_v2, manifest_v2, {"trade_date": "20260801"}),
            )
            for data_bytes, manifest_bytes, expected_query in cases:
                with self.subTest(data=data_bytes[:20], manifest=manifest_bytes[:20], query=expected_query):
                    first.write_bytes(data_bytes)
                    manifest_path.write_bytes(manifest_bytes)
                    self.assertIsNone(
                        cache.read_rows(
                            market=Market.US, dataset="quotes", year=2026, cache_key="daily", query=expected_query
                        )
                    )

            first.write_bytes(data_v2)
            manifest = json.loads(manifest_v2)
            manifest["rows"] += 1
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            self.assertIsNone(
                cache.read_rows(market=Market.US, dataset="quotes", year=2026, cache_key="daily", query=query)
            )
            manifest["rows"] = True
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            self.assertIsNone(
                cache.read_rows(market=Market.US, dataset="quotes", year=2026, cache_key="daily", query=query)
            )

    def test_incomplete_manifest_is_never_reused(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            cache = RawDataCache(Path(temp_dir))
            cache.write_rows(
                market=Market.US, dataset="quotes", year=2026, cache_key="daily", rows=[],
                query={"trade_date": "20260731"}, source="yahoo_finance",
                fetched_at=datetime(2026, 8, 1, tzinfo=UTC), complete=False,
            )
            result = cache.read_rows(
                market=Market.US, dataset="quotes", year=2026, cache_key="daily",
                query={"trade_date": "20260731"},
            )

        self.assertIsNone(result)

    def test_manifest_complete_accepts_only_exact_json_boolean_true(self):
        query = {"trade_date": "20260731"}
        with tempfile.TemporaryDirectory() as temp_dir:
            cache = RawDataCache(Path(temp_dir))
            data_path = cache.write_rows(
                market=Market.US,
                dataset="quotes",
                year=2026,
                cache_key="daily",
                rows=[],
                query=query,
                source="yahoo_finance",
                fetched_at=datetime(2026, 8, 1, tzinfo=UTC),
            )
            manifest_path = data_path.with_suffix(".manifest.json")
            pristine = json.loads(manifest_path.read_text(encoding="utf-8"))

            self.assertIsNotNone(
                cache.read_rows(
                    market=Market.US,
                    dataset="quotes",
                    year=2026,
                    cache_key="daily",
                    query=query,
                )
            )
            for malformed in ("false", 1, {"complete": True}, []):
                with self.subTest(complete=malformed):
                    manifest = json.loads(json.dumps(pristine))
                    manifest["complete"] = malformed
                    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
                    self.assertIsNone(
                        cache.read_rows(
                            market=Market.US,
                            dataset="quotes",
                            year=2026,
                            cache_key="daily",
                            query=query,
                        )
                    )

    def test_json_valid_but_structurally_invalid_snapshot_is_a_cache_miss(self):
        query = {"trade_date": "20260731"}
        snapshot = _snapshot_for_cache(
            as_of=datetime(2026, 7, 31, 16, 0, tzinfo=NEW_YORK),
            data_time=datetime(2026, 7, 31, 16, 0, tzinfo=NEW_YORK),
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            cache = RawDataCache(Path(temp_dir))
            data_path = cache.write_snapshot(
                dataset="daily_snapshot",
                cache_key="2026-07-31",
                snapshot=snapshot,
                query=query,
                source="yahoo_finance",
                fetched_at=snapshot.as_of_time,
            )
            manifest_path = data_path.with_suffix(".manifest.json")
            pristine_data = data_path.read_bytes()
            pristine_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

            metadata_cases = (
                ("source_times", []),
                ("points", {"not": "a list"}),
                ("data_status", "not-a-status"),
                ("as_of_time", "not-a-time"),
                ("session_slot", ["close"]),
            )
            for key, value in metadata_cases:
                with self.subTest(metadata_key=key):
                    manifest = json.loads(json.dumps(pristine_manifest))
                    manifest["snapshot"][key] = value
                    data_path.write_bytes(pristine_data)
                    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
                    self.assertIsNone(
                        cache.read_snapshot(
                            market=Market.US,
                            dataset="daily_snapshot",
                            year=2026,
                            cache_key="2026-07-31",
                            query=query,
                        )
                    )

            row_cases = (
                ("quality_status", "not-a-status"),
                ("data_time", "not-a-time"),
                ("field", ["index_price"]),
                ("source", {"vendor": "yahoo"}),
            )
            for key, value in row_cases:
                with self.subTest(row_key=key):
                    lines = pristine_data.decode("utf-8").splitlines()
                    row = json.loads(lines[1])
                    row[key] = value
                    changed_data = (lines[0] + "\n" + json.dumps(row, separators=(",", ":")) + "\n").encode("utf-8")
                    manifest = json.loads(json.dumps(pristine_manifest))
                    manifest["data_sha256"] = hashlib.sha256(changed_data).hexdigest()
                    data_path.write_bytes(changed_data)
                    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
                    self.assertIsNone(
                        cache.read_snapshot(
                            market=Market.US,
                            dataset="daily_snapshot",
                            year=2026,
                            cache_key="2026-07-31",
                            query=query,
                        )
                    )


if __name__ == "__main__":
    unittest.main()
