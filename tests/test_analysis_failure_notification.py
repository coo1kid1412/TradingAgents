from unittest.mock import MagicMock, patch

from main import _notify_analysis_failure, analyze_single_stock


def test_failure_notification_translates_provider_error_and_sends_text():
    error = RuntimeError("422 input new_sensitive (1026); secret detail")
    with patch("tradingagents.harness.market_risk_daily._send_feishu_message") as send:
        _notify_analysis_failure("688017", error)

    message = send.call_args.args[0]
    assert "TA 分析失败" in message
    assert "688017" in message
    assert "MiniMax 输入审核拒绝（1026）" in message
    assert "secret detail" not in message
    assert "/tmp/ta_688017.log" in message


def test_non_retryable_analysis_failure_notifies_user():
    graph = MagicMock()
    graph.propagate.side_effect = RuntimeError("422 input new_sensitive (1026)")
    with patch("tradingagents.graph.trading_graph.TradingAgentsGraph", return_value=graph), patch(
        "main._notify_analysis_failure",
    ) as notify:
        ticker, success, result = analyze_single_stock("688017", "2026-08-30", {})

    assert ticker == "688017"
    assert success is False
    assert "1026" in result
    notify.assert_called_once()


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"{len(tests)}/{len(tests)} passed")
