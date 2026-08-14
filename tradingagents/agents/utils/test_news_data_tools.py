from datetime import datetime
from unittest.mock import patch

from tradingagents.agents.utils.news_data_tools import get_announcements


def test_announcements_use_long_official_disclosure_lookback():
    with patch("tradingagents.agents.utils.news_data_tools.route_to_vendor") as route:
        route.return_value = "official disclosures"
        result = get_announcements.invoke({"ticker": "603629", "curr_date": "2026-08-13"})

    assert "official disclosures" in result
    method, ticker, start_date, end_date = route.call_args_list[0].args
    assert method == "get_announcements"
    assert ticker == "603629"
    assert end_date == "2026-08-13"
    assert (datetime.fromisoformat(end_date) - datetime.fromisoformat(start_date)).days == 120


def test_announcements_include_structured_official_forecasts():
    with patch("tradingagents.agents.utils.news_data_tools.route_to_vendor") as route:
        route.side_effect = ["cninfo list", "forecast facts"]
        result = get_announcements.invoke({"ticker": "603629", "curr_date": "2026-08-13"})

    assert "cninfo list" in result
    assert "forecast facts" in result
    assert route.call_args_list[1].args[0] == "get_performance_forecasts"


if __name__ == "__main__":
    test_announcements_use_long_official_disclosure_lookback()
    test_announcements_include_structured_official_forecasts()
    print("2/2 passed")
