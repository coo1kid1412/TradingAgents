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
import tradingagents.agents.analysts.fundamentals_analyst as fundamentals_analyst_module
import tradingagents.agents.analysts.market_analyst as market_analyst_module
import tradingagents.agents.utils.macro_context_node as macro_context_module
import tradingagents.agents.utils.quant_score_node as quant_score_module
from tradingagents.dataflows.profile_calc import parse_market_cap_from_fundamentals

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


def _fundamentals_quote() -> "IntradayQuote":
    return IntradayQuote(
        symbol="301217",
        name="铜冠铜箔",
        trade_date="2026-07-20",
        quote_time=datetime(2026, 7, 20, 14, 19, 15, tzinfo=SHANGHAI),
        open=119.0,
        high=119.0,
        low=103.24,
        last=103.24,
        pre_close=129.05,
        volume=24_760_000,
        amount=2_750_000_000.0,
        source="sina_realtime",
    )


def _fundamentals_output(
    quote,
    daily_trade_date: str = "20260717",
    daily_close: float = 129.05,
) -> str:
    stock_basic = pd.DataFrame([{
        "ts_code": "301217.SZ", "symbol": "301217", "name": "铜冠铜箔",
        "area": "安徽", "industry": "元件", "market": "创业板", "list_date": "20220127",
    }])
    fina = pd.DataFrame([
        {"end_date": "20251231", "eps": 0.2, "tob_operate_income": 2.0},
        {"end_date": "20241231", "eps": -0.1, "tob_operate_income": 1.0},
    ])
    daily_basic = pd.DataFrame([{
        "trade_date": daily_trade_date, "close": daily_close, "pe_ttm": 651.37,
        "ps_ttm": 55.0, "pb": 19.45, "total_mv": 10_698_400.0,
        "circ_mv": 2_955_000.0, "total_share": 82_900.0,
    }])
    pro = SimpleNamespace(stock_basic=lambda **kwargs: stock_basic)

    with (
        patch.object(tushare_vendor, "_get_tushare_api", return_value=pro),
        patch.object(tushare_vendor, "_safe_call", return_value=stock_basic),
        patch.object(tushare_vendor, "_fetch_fina_indicator_cached", return_value=fina),
        patch.object(tushare_vendor, "_fetch_cached", return_value=daily_basic),
        patch.object(tushare_vendor, "fetch_intraday_quote", return_value=quote, create=True),
        patch.object(tushare_vendor, "_compute_ttm_eps", return_value=0.2),
        patch.object(tushare_vendor, "_compute_ttm_revenue_per_share_fina", return_value=2.0),
        patch.object(tushare_vendor, "_format_growth_indicators", return_value=""),
        patch.object(tushare_vendor, "_format_landmine_line", return_value=""),
        patch.object(tushare_vendor, "_format_cyclical_line", return_value=""),
        patch.object(tushare_vendor, "_format_paradigm_line", return_value=""),
        patch.object(tushare_vendor, "_format_main_business_line", return_value=""),
    ):
        return tushare_vendor.get_fundamentals("301217", "2026-07-20")


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


def test_fundamentals_valuation_uses_intraday_price_for_all_price_sensitive_metrics():
    output = _fundamentals_output(_fundamentals_quote())
    assert "价格数据状态: intraday_provisional" in output
    assert "估值基准价(元): 103.24" in output
    assert "估值基准时间: 2026-07-20 14:19:15" in output
    assert "估值基准来源: sina_realtime" in output
    assert "前收参考价(元,非当前价): 129.05" in output
    assert "动态PE(系统计算): 516.2倍" in output
    assert "静态PE(系统计算): 516.2倍" in output
    assert "PS(TTM/系统计算): 51.62" in output
    assert "PB: 15.56 (按估值基准价调整)" in output
    assert "总市值(亿元): 855.87 (按估值基准价调整)" in output
    assert parse_market_cap_from_fundamentals(output) == 855.87


def test_fundamentals_valuation_labels_t_minus_one_when_intraday_quote_is_missing():
    output = _fundamentals_output(None)
    assert "价格数据状态: t_minus_1" in output
    assert "估值基准价(元): 129.05" in output
    assert "估值基准日期: 2026-07-17" in output
    assert "估值基准来源: tushare_daily_basic" in output
    assert "动态PE(系统计算): 645.25倍" in output
    assert "静态PE(系统计算): 645.25倍" in output
    assert "PB: 19.45 (按估值基准价调整)" in output
    assert "总市值(亿元): 1069.84 (按估值基准价调整)" in output
    assert parse_market_cap_from_fundamentals(output) == 1069.84


def test_fundamentals_report_has_deterministic_intraday_price_banner():
    enforce = getattr(fundamentals_analyst_module, "_enforce_price_context", None)
    assert callable(enforce), "fundamentals price context enforcement is missing"
    vendor_text = """## 估值价格口径
价格数据状态: intraday_provisional
估值基准价(元): 103.24
估值基准日期: 2026-07-20
估值基准时间: 2026-07-20 14:19:15
估值基准来源: sina_realtime
前收参考价(元,非当前价): 129.05
前收日期: 2026-07-17
"""
    output = enforce("# 基本面分析\n\n正文", vendor_text)
    assert output.startswith("> **估值价格口径：盘中临时价 103.24 元**")
    assert "前收参考：129.05 元（非当前价）" in output


def test_fundamentals_same_day_official_close_beats_realtime_quote():
    daily_basic = pd.DataFrame([{
        "trade_date": "20260722",
        "close": 118.34,
    }])
    quote = IntradayQuote(
        symbol="603629",
        name="利通电子",
        trade_date="2026-07-22",
        quote_time=datetime(2026, 7, 22, 15, 34, 59, tzinfo=SHANGHAI),
        open=114.0,
        high=118.34,
        low=112.8,
        last=118.34,
        pre_close=107.58,
        volume=12_825_794,
        amount=1_500_000_000.0,
        source="sina_realtime",
    )

    with patch.object(tushare_vendor, "fetch_intraday_quote", return_value=quote):
        context = tushare_vendor._resolve_fundamentals_price_context(
            "603629", "2026-07-22", SimpleNamespace(), daily_basic,
        )

    assert context["status"] == "official_daily"
    assert context["price"] == 118.34
    assert context["date"] == "2026-07-22"
    assert context["time"] is None
    assert context["source"] == "tushare_daily_basic"

    vendor_text = """## 估值价格口径
价格数据状态: official_daily
估值基准价(元): 118.34
估值基准日期: 2026-07-22
估值基准时间: N/A
估值基准来源: tushare_daily_basic
前收参考价(元,非当前价): 118.34
前收日期: 2026-07-22
"""
    report = fundamentals_analyst_module._enforce_price_context("# 基本面分析", vendor_text)
    assert report.startswith("> **估值价格口径：正式收盘价 118.34 元**")
    assert "盘中临时价" not in report
    assert "前收参考" not in report


def test_fundamentals_official_daily_output_omits_previous_close_fields():
    output = _fundamentals_output(
        _fundamentals_quote(), daily_trade_date="20260720", daily_close=103.24,
    )
    assert "价格数据状态: official_daily" in output
    assert "估值基准价(元): 103.24" in output
    assert "估值基准时间: N/A" in output
    assert "前收参考价(元,非当前价)" not in output
    assert "前收日期:" not in output


def test_macro_prompt_prioritizes_intraday_price_over_previous_close():
    class CaptureLLM:
        def __init__(self):
            self.prompt = ""

        def invoke(self, prompt):
            self.prompt = prompt
            return SimpleNamespace(content="ok")

    llm = CaptureLLM()
    node = macro_context_module.create_macro_context_node(llm)
    node({
        "company_of_interest": "301217", "company_name": "铜冠铜箔",
        "trade_date": "2026-07-20", "news_report": "", "sentiment_report": "",
        "fundamentals_report": "前收参考价(元,非当前价): 129.05",
        "market_report": "最新价：103.24 元（盘中）",
    })
    assert "盘中临时价是唯一当前价" in llm.prompt
    assert "前收参考价不得写成当前价" in llm.prompt


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for test in tests:
        test()
        print(f"  PASS {test.__name__}")
    print(f"\n{len(tests)}/{len(tests)} passed")
