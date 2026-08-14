import pandas as pd

from tradingagents.dataflows import akshare_vendor


class _FakeAK:
    @staticmethod
    def stock_research_report_em(symbol):
        return pd.DataFrame([
            {"股票代码": symbol, "报告名称": "未来研报", "日期": "2026-08-15"},
            {"股票代码": symbol, "报告名称": "当日研报", "日期": "2026-08-14"},
            {"股票代码": symbol, "报告名称": "历史研报", "日期": "2026-08-10"},
        ])


def test_research_reports_exclude_rows_after_analysis_date():
    original = akshare_vendor._import_akshare
    akshare_vendor._import_akshare = lambda: _FakeAK()
    try:
        report = akshare_vendor.get_research_reports(
            "603629", curr_date="2026-08-14", limit=20,
        )
    finally:
        akshare_vendor._import_akshare = original

    assert "未来研报" not in report
    assert "当日研报" in report
    assert "历史研报" in report


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"  PASS {test.__name__}")
    print(f"\n{len(tests)}/{len(tests)} passed")
