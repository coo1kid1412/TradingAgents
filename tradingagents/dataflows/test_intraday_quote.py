"""Intraday A-share quote overlay regression tests.

Run: .venv/bin/python tradingagents/dataflows/test_intraday_quote.py
"""

from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace
from unittest.mock import patch
from zoneinfo import ZoneInfo

import pandas as pd

import tradingagents.dataflows.tushare_vendor as tushare_vendor
import tradingagents.dataflows.intraday_quote as intraday_quote_module
import tradingagents.agents.analysts.market_analyst as market_analyst_module
import tradingagents.agents.utils.quant_score_node as quant_score_module

try:
    from tradingagents.dataflows.intraday_quote import (
        IntradayQuote,
        _clear_quote_cache_for_tests,
        _parse_sina_response,
        _parse_tushare_quote,
        fetch_intraday_quote,
        is_quote_fresh,
        merge_intraday_quote,
    )
except ModuleNotFoundError:
    def _missing(*args, **kwargs):
        raise AssertionError("intraday quote service is not implemented")

    IntradayQuote = _missing
    _clear_quote_cache_for_tests = _missing
    _parse_sina_response = _missing
    _parse_tushare_quote = _missing
    fetch_intraday_quote = _missing
    is_quote_fresh = _missing
    merge_intraday_quote = _missing


SHANGHAI = ZoneInfo("Asia/Shanghai")


def _now(hour: int, minute: int, second: int = 0) -> datetime:
    return datetime(2026, 7, 17, hour, minute, second, tzinfo=SHANGHAI)


def _quote(
    *,
    trade_time: datetime | None = None,
    trade_date: str = "2026-07-17",
    source: str = "sina_realtime",
) -> "IntradayQuote":
    return IntradayQuote(
        symbol="300308",
        name="中际旭创",
        trade_date=trade_date,
        quote_time=trade_time or _now(11, 29, 58),
        open=1100.0,
        high=1130.0,
        low=1090.0,
        last=1120.5,
        pre_close=1113.0,
        volume=12_345_678,
        amount=13_800_000_000.0,
        source=source,
    )


def _sina_payload(date: str = "2026-07-17", time: str = "11:29:58") -> str:
    fields = [
        "中际旭创", "1100.00", "1113.00", "1120.50", "1130.00", "1090.00",
        "1120.00", "1121.00", "12345678", "13800000000",
        "100", "1120.00", "200", "1119.00", "300", "1118.00", "400", "1117.00", "500", "1116.00",
        "100", "1121.00", "200", "1122.00", "300", "1123.00", "400", "1124.00", "500", "1125.00",
        date, time, "00",
    ]
    return 'var hq_str_sz300308="' + ",".join(fields) + '";'


def test_intraday_quote_module_exists():
    assert IntradayQuote is not None, "intraday quote service is not implemented"


def test_parse_sina_single_symbol_quote():
    quote = _parse_sina_response(_sina_payload(), "300308")
    assert quote is not None
    assert quote.name == "中际旭创"
    assert quote.trade_date == "2026-07-17"
    assert quote.quote_time == _now(11, 29, 58)
    assert quote.last == 1120.5
    assert quote.volume == 12_345_678
    assert quote.source == "sina_realtime"


def test_parse_sina_rejects_empty_or_zero_quote():
    assert _parse_sina_response('var hq_str_sz300308="";', "300308") is None
    payload = _sina_payload().replace(",1120.50,", ",0.00,", 1)
    assert _parse_sina_response(payload, "300308") is None


def test_parse_tushare_rt_k_quote():
    frame = pd.DataFrame([{
        "ts_code": "300308.SZ", "name": "中际旭创", "pre_close": 1113.0,
        "open": 1100.0, "high": 1130.0, "low": 1090.0, "close": 1120.5,
        "vol": 12_345_678, "amount": 13_800_000_000,
        "trade_time": "2026-07-17 11:29:58",
    }])
    quote = _parse_tushare_quote(frame, "300308")
    assert quote is not None
    assert quote.source == "tushare_rt_k"
    assert quote.quote_time == _now(11, 29, 58)
    assert quote.last == 1120.5


def test_continuous_session_requires_quote_within_five_minutes():
    assert is_quote_fresh(_quote(trade_time=_now(11, 29)), "2026-07-17", _now(11, 31))
    assert not is_quote_fresh(_quote(trade_time=_now(11, 20)), "2026-07-17", _now(11, 31))


def test_lunch_break_accepts_only_fresh_morning_close():
    assert is_quote_fresh(_quote(trade_time=_now(11, 30)), "2026-07-17", _now(12, 15))
    assert not is_quote_fresh(_quote(trade_time=_now(11, 20)), "2026-07-17", _now(12, 15))


def test_preopen_wrong_date_and_stale_after_close_are_rejected():
    assert not is_quote_fresh(_quote(trade_time=_now(9, 20)), "2026-07-17", _now(9, 20))
    assert not is_quote_fresh(_quote(trade_date="2026-07-16"), "2026-07-17", _now(11, 31))
    assert not is_quote_fresh(_quote(trade_time=_now(14, 40)), "2026-07-17", _now(15, 10))
    assert is_quote_fresh(_quote(trade_time=_now(15, 0)), "2026-07-17", _now(15, 10))
    assert is_quote_fresh(_quote(trade_time=_now(16, 29)), "2026-07-17", _now(16, 30))


def test_fetch_falls_back_to_sina_when_rt_k_is_denied():
    class DeniedPro:
        def rt_k(self, **kwargs):
            raise RuntimeError("没有接口(rt_k)访问权限")

    response = SimpleNamespace(content=_sina_payload().encode("gbk"))
    response.raise_for_status = lambda: None
    _clear_quote_cache_for_tests()
    with patch("requests.get", return_value=response) as get:
        quote = fetch_intraday_quote(
            "300308", "2026-07-17", tushare_api=DeniedPro(), now=_now(11, 31),
        )
    assert quote is not None
    assert quote.source == "sina_realtime"
    assert get.call_count == 1


def test_merge_appends_provisional_bar_and_converts_shares_to_hands():
    daily = pd.DataFrame([{
        "trade_date": "20260716", "open": 1122.22, "high": 1163.0,
        "low": 1098.0, "close": 1113.0, "vol": 306633.63,
    }])
    merged, meta = merge_intraday_quote(daily, _quote(), "2026-07-17")
    assert list(merged["trade_date"].astype(str)) == ["20260716", "20260717"]
    assert float(merged.iloc[-1]["close"]) == 1120.5
    assert float(merged.iloc[-1]["vol"]) == 123456.78
    assert meta["status"] == "intraday_provisional"
    assert meta["source"] == "sina_realtime"


def test_merge_keeps_official_same_day_bar_over_provisional_quote():
    daily = pd.DataFrame([{
        "trade_date": "20260717", "open": 1101.0, "high": 1140.0,
        "low": 1088.0, "close": 1133.0, "vol": 200000.0,
    }])
    merged, meta = merge_intraday_quote(daily, _quote(), "2026-07-17")
    assert len(merged) == 1
    assert float(merged.iloc[-1]["close"]) == 1133.0
    assert meta["status"] == "official_daily"
    assert meta["source"] == "tushare_daily"


def test_merge_without_quote_labels_t_minus_one():
    daily = pd.DataFrame([{
        "trade_date": "20260716", "open": 1122.22, "high": 1163.0,
        "low": 1098.0, "close": 1113.0, "vol": 306633.63,
    }])
    merged, meta = merge_intraday_quote(daily, None, "2026-07-17")
    assert len(merged) == 1
    assert meta == {
        "status": "t_minus_1",
        "date": "2026-07-16",
        "time": None,
        "source": "tushare_daily",
    }


def _daily_history() -> pd.DataFrame:
    return pd.DataFrame([
        {"trade_date": "20260714", "open": 1108.53, "high": 1198.66, "low": 1098.0, "close": 1184.05, "vol": 400475.72},
        {"trade_date": "20260715", "open": 1190.0, "high": 1207.98, "low": 1160.0, "close": 1169.31, "vol": 247946.35},
        {"trade_date": "20260716", "open": 1122.22, "high": 1163.0, "low": 1098.0, "close": 1113.0, "vol": 306633.63},
    ])


def test_tushare_stock_data_appends_quote_and_discloses_freshness():
    pro = SimpleNamespace(daily=lambda **kwargs: None)
    with (
        patch.object(tushare_vendor, "_get_tushare_api", return_value=pro),
        patch.object(tushare_vendor, "_safe_call", return_value=_daily_history()),
        patch.object(tushare_vendor, "fetch_intraday_quote", return_value=_quote(), create=True) as fetch,
    ):
        output = tushare_vendor.get_stock("300308", "2026-07-14", "2026-07-17")
    assert "# Actual date range: 2026-07-14 to 2026-07-17" in output
    assert "# Price data status: intraday_provisional" in output
    assert "# Latest bar source: sina_realtime" in output
    assert "# Latest quote time: 2026-07-17 11:29:58" in output
    assert "2026-07-17,1100.0,1130.0,1090.0,1120.5,12345678" in output
    fetch.assert_called_once()


def test_tushare_stock_data_labels_t_minus_one_when_quote_missing():
    pro = SimpleNamespace(daily=lambda **kwargs: None)
    with (
        patch.object(tushare_vendor, "_get_tushare_api", return_value=pro),
        patch.object(tushare_vendor, "_safe_call", return_value=_daily_history()),
        patch.object(tushare_vendor, "fetch_intraday_quote", return_value=None, create=True),
    ):
        output = tushare_vendor.get_stock("300308", "2026-07-14", "2026-07-17")
    assert "# Actual date range: 2026-07-14 to 2026-07-16" in output
    assert "# Price data status: t_minus_1" in output
    assert "# Latest bar source: tushare_daily" in output


def test_tushare_indicator_uses_provisional_current_bar():
    pro = SimpleNamespace(daily=lambda **kwargs: None)
    with (
        patch.object(tushare_vendor, "_get_tushare_api", return_value=pro),
        patch.object(tushare_vendor, "_safe_call", return_value=_daily_history()),
        patch.object(tushare_vendor, "fetch_intraday_quote", return_value=_quote(), create=True),
    ):
        output = tushare_vendor.get_indicator("300308", "close_10_ema", "2026-07-17", 5)
    assert "## Price data status: intraday_provisional" in output
    assert "## Latest quote time: 2026-07-17 11:29:58" in output
    current_line = next(line for line in output.splitlines() if line.startswith("2026-07-17:"))
    assert "N/A" not in current_line


def test_stock_header_metadata_parser_is_shared_and_deterministic():
    parser = getattr(intraday_quote_module, "parse_price_metadata", None)
    assert callable(parser), "shared price metadata parser is missing"
    meta = parser(
        "# Actual date range: 2025-07-18 to 2026-07-17 (requested: x to y)\n"
        "# Source: Tushare Pro\n"
        "# Price data status: intraday_provisional\n"
        "# Latest bar source: sina_realtime\n"
        "# Latest quote time: 2026-07-17 11:29:58\n"
    )
    assert meta == {
        "status": "intraday_provisional",
        "date": "2026-07-17",
        "time": "2026-07-17 11:29:58",
        "source": "sina_realtime",
    }


def test_quant_report_propagates_price_freshness_into_yaml():
    formatter = quant_score_module._format_report
    result = SimpleNamespace(
        composite=66.0,
        interpretation="偏强",
        weights_used={"momentum": 1.0},
        factor_scores={"momentum": 66.0},
        factor_breakdowns={},
        coverage={"available": ["momentum"], "missing": [], "total_weight_used": 1.0},
    )
    report = formatter(
        "300308", "中际旭创", "2026-07-17", result, {},
        price_meta={
            "status": "intraday_provisional", "date": "2026-07-17",
            "time": "2026-07-17 11:29:58", "source": "sina_realtime",
        },
    )
    assert 'price_data_status: "intraday_provisional"' in report
    assert 'price_data_date: "2026-07-17"' in report
    assert 'price_data_time: "2026-07-17 11:29:58"' in report
    assert 'price_data_source: "sina_realtime"' in report


def test_market_report_metadata_is_deterministically_visible_and_in_summary():
    enforce = getattr(market_analyst_module, "_enforce_price_metadata", None)
    assert callable(enforce), "market price metadata enforcement is missing"
    report = """# 技术分析\n\n正文\n\n```yaml\nSUMMARY:\n  trend: 上行\n```\n"""
    output = enforce(report, {
        "status": "intraday_provisional", "date": "2026-07-17",
        "time": "2026-07-17 11:29:58", "source": "sina_realtime",
    })
    assert output.startswith("> **价格数据状态：盘中临时K线**")
    assert 'price_data_status: "intraday_provisional"' in output
    assert 'price_data_date: "2026-07-17"' in output
    assert 'price_data_time: "2026-07-17 11:29:58"' in output
    assert 'price_data_source: "sina_realtime"' in output


def test_market_prompt_requires_price_freshness_summary_contract():
    import inspect

    source = inspect.getsource(market_analyst_module.create_market_analyst)
    for field in (
        "price_data_status", "price_data_date", "price_data_time", "price_data_source",
    ):
        assert field in source, field


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for test in tests:
        test()
        print(f"  PASS {test.__name__}")
    print(f"\n{len(tests)}/{len(tests)} passed")
