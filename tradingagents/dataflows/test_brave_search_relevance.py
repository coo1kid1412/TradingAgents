"""Brave result relevance filtering tests."""

from unittest.mock import MagicMock, patch

from tradingagents.dataflows.brave_search import _is_relevant_result, search_news


def test_result_is_relevant_when_company_or_ticker_is_present():
    anchors = ("绿的谐波", "688017")
    assert _is_relevant_result(
        {"title": "机器人 ETF 调整", "description": "重仓股包括绿的谐波"}, anchors,
    )
    assert _is_relevant_result(
        {"title": "688017 半年报", "description": "毛利率变化"}, anchors,
    )


def test_result_is_rejected_when_it_only_matches_generic_news_query_words():
    anchors = ("绿的谐波", "688017")
    assert not _is_relevant_result(
        {"title": "FBI 解密案件", "description": "与公司和机器人行业无关"}, anchors,
    )
    assert not _is_relevant_result(
        {"title": "新闻 - 搜狐", "description": "娱乐新闻首页"}, anchors,
    )


def test_ascii_ticker_requires_a_token_boundary():
    assert _is_relevant_result(
        {"title": "NOK earnings", "description": "quarterly results"}, ("NOK",),
    )
    assert not _is_relevant_result(
        {"title": "Nokia-like token", "description": "SNOKER event"}, ("NOK",),
    )


def test_search_news_drops_unrelated_results_before_they_reach_the_llm():
    response = MagicMock()
    response.raise_for_status.return_value = None
    response.json.return_value = {
        "web": {
            "results": [
                {
                    "title": "FBI 解密案件",
                    "description": "与公司无关的政治新闻",
                    "url": "https://example.com/politics",
                },
                {
                    "title": "绿的谐波发布半年报",
                    "description": "688017 毛利率变化",
                    "url": "https://example.com/688017",
                },
            ]
        }
    }
    with patch("tradingagents.dataflows.brave_search._get_api_key", return_value="test"), patch(
        "tradingagents.dataflows.brave_search.requests.get", return_value=response,
    ):
        result = search_news(
            "绿的谐波 688017 新闻",
            relevance_terms=("绿的谐波", "688017"),
        )

    assert "绿的谐波发布半年报" in result
    assert "FBI 解密案件" not in result


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"{len(tests)}/{len(tests)} passed")
