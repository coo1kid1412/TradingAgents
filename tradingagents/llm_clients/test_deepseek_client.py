from __future__ import annotations

import os
import sys

from langchain_core.messages import AIMessage

from tradingagents.agents.utils.handoff import pack_agent_context
from tradingagents.llm_clients.base_client import normalize_content
from tradingagents.llm_clients.deepseek_client import DeepSeekClient
from tradingagents.llm_clients.factory import create_llm_client
from tradingagents.llm_clients.role_policy import resolve_role_policy
from tradingagents.default_config import DEFAULT_CONFIG


def _with_key(value: str | None):
    previous = os.environ.get("DEEPSEEK_API_KEY")
    if value is None:
        os.environ.pop("DEEPSEEK_API_KEY", None)
    else:
        os.environ["DEEPSEEK_API_KEY"] = value
    return previous


def _restore_key(previous: str | None) -> None:
    if previous is None:
        os.environ.pop("DEEPSEEK_API_KEY", None)
    else:
        os.environ["DEEPSEEK_API_KEY"] = previous


def test_deepseek_client_enables_think_explicitly():
    previous = _with_key("sk-test-deepseek")
    try:
        llm = DeepSeekClient(
            "deepseek-v4-pro", reasoning_effort="max", max_tokens=16384,
        ).get_llm()
    finally:
        _restore_key(previous)

    assert str(llm.openai_api_base).rstrip("/") == "https://api.deepseek.com"
    assert llm.extra_body["thinking"] == {"type": "enabled"}
    assert llm.extra_body["reasoning_effort"] == "max"
    assert llm.max_tokens == 16384


def test_deepseek_client_fails_fast_without_key():
    previous = _with_key(None)
    try:
        try:
            DeepSeekClient("deepseek-v4-flash").get_llm()
        except ValueError as exc:
            assert "DEEPSEEK_API_KEY" in str(exc)
        else:
            raise AssertionError("missing key should fail before any request")
    finally:
        _restore_key(previous)


def test_factory_and_defaults_select_deepseek_without_minimax_fallback():
    previous = _with_key("sk-test-deepseek")
    try:
        client = create_llm_client("deepseek", "deepseek-v4-flash")
    finally:
        _restore_key(previous)
    assert isinstance(client, DeepSeekClient)
    assert DEFAULT_CONFIG["llm_provider"] == "deepseek"
    assert DEFAULT_CONFIG["deep_think_llm"] == "deepseek-v4-pro"
    assert DEFAULT_CONFIG["quick_think_llm"] == "deepseek-v4-flash"
    assert DEFAULT_CONFIG["backend_url"] == "https://api.deepseek.com"


def test_role_policy_routes_agents_by_judgment_cost():
    config = {
        "deep_think_llm": "deepseek-v4-pro",
        "quick_think_llm": "deepseek-v4-flash",
        "fundamentals_analyst_max_tokens": 16384,
        "research_manager_max_tokens": 16384,
        "portfolio_manager_max_tokens": 16384,
    }
    rm = resolve_role_policy(config, "research_manager")
    pm = resolve_role_policy(config, "portfolio_manager")
    market = resolve_role_policy(config, "market")
    news = resolve_role_policy(config, "news")
    risk = resolve_role_policy(config, "neutral_risk")

    assert (rm.model, rm.reasoning_effort) == ("deepseek-v4-pro", "max")
    assert (pm.model, pm.reasoning_effort) == ("deepseek-v4-pro", "max")
    assert (market.model, market.reasoning_effort) == ("deepseek-v4-pro", "high")
    assert (news.model, news.reasoning_effort) == ("deepseek-v4-flash", "high")
    assert (risk.model, risk.reasoning_effort) == ("deepseek-v4-flash", "high")


def test_graph_role_factory_applies_policy_to_actual_clients():
    from tradingagents.graph.trading_graph import TradingAgentsGraph

    previous = _with_key("sk-test-deepseek")
    graph = TradingAgentsGraph.__new__(TradingAgentsGraph)
    graph.config = DEFAULT_CONFIG.copy()
    graph.callbacks = []
    try:
        market = graph._create_role_llm("market", 0.2)
        news = graph._create_role_llm("news", 0.4)
        rm = graph._create_role_llm("research_manager", 0.3)
    finally:
        _restore_key(previous)

    assert market.model_name == "deepseek-v4-pro"
    assert news.model_name == "deepseek-v4-flash"
    assert rm.model_name == "deepseek-v4-pro"
    assert rm.extra_body["reasoning_effort"] == "max"


def test_reasoning_is_preserved_for_tool_subturn_but_not_visible_content():
    message = AIMessage(
        content="<think>private chain</think>最终正文",
        additional_kwargs={"reasoning_content": "private tool reasoning"},
    )
    normalized = normalize_content(message)
    assert normalized.content == "最终正文"
    assert normalized.additional_kwargs["reasoning_content"] == "private tool reasoning"

    packed = pack_agent_context(
        [{"label": "报告", "content": "reasoning_content: private\n<think>x</think>结论", "priority": "handoff"}],
        budget_chars=300,
    )
    assert "private" not in packed
    assert "<think>" not in packed
    assert "结论" in packed


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for test in tests:
        try:
            test()
            print(f"  PASS {test.__name__}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"  FAIL {test.__name__}: [{type(exc).__name__}] {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
