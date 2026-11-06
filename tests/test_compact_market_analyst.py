import importlib
import importlib.util
import json
import re
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from langchain_core.messages import AIMessage, HumanMessage
import yaml

from test_technical_context import history, vendor_text
from tradingagents.agents.utils.handoff import extract_handoff
from tradingagents.agents.utils.test_handoff import _handoff
from tradingagents.agents.utils.yaml_summary import extract_yaml_mapping


class FakeLLM:
    def __init__(self, content):
        self.content = content
        self.calls = []

    def invoke(self, messages):
        self.calls.append(messages)
        return AIMessage(content=self.content)

    def bind_tools(self, *_args, **_kwargs):
        raise AssertionError("compact market must not bind data tools")


class CompactMarketTests(unittest.TestCase):
    def setUp(self):
        name = "tradingagents.agents.analysts.compact_market_analyst"
        self.assertIsNotNone(importlib.util.find_spec(name), "compact market node is missing")
        self.module = importlib.import_module(name)
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)

    def response(self):
        summary = {key: None for key in self.module.SUMMARY_FIELDS}
        summary.update(trend="震荡", trend_weekly="震荡", trend_daily="震荡", momentum="中", rating="HOLD", data_implied_direction="中性")
        handoff = _handoff("market").replace("2026-08-14", "2026-11-05").replace(
            "value: 100", f"value: {history().Close.iloc[-1]}",
        )
        return "# 技术面\n\n等待量价确认。\n\n```yaml\n" + yaml.safe_dump({"SUMMARY": summary}, allow_unicode=True) + "```\n" + handoff

    def run_node(self, llm, *, cap=24000, capital="", raw=None, old_messages=None):
        config = {"data_cache_dir": self.directory.name, "market_context_max_chars": cap}
        with patch.object(self.module, "get_config", return_value=config), patch.object(
            self.module, "get_stock_data",
        ) as fetch:
            fetch.invoke.return_value = raw or vendor_text(history())
            result = self.module.create_compact_market_analyst(llm)({
                "company_of_interest": "301171", "company_name": "易点天下", "trade_date": "2026-11-05",
                "capital_flow_yaml": capital, "messages": old_messages or [],
            })
        self.assertEqual(fetch.invoke.call_count, 1)
        return result

    def test_one_compact_invocation_no_tools_or_old_conversation(self):
        llm = FakeLLM(self.response())
        result = self.run_node(llm, old_messages=[HumanMessage(content="OLD_HISTORY" * 20000)])
        self.assertEqual(len(llm.calls), 1)
        text = "\n".join(message.content for message in llm.calls[0])
        self.assertLess(len(text), 24000)
        self.assertNotIn("OLD_HISTORY", text)
        self.assertNotIn("N/A：非交易日", text)
        self.assertIn("percentile_1y", text)
        self.assertEqual(result["messages"][0].tool_calls, [])
        self.assertNotEqual(extract_handoff(result["market_report"], "market")["quality"]["status"], "invalid")

    def test_summary_keeps_price_and_computed_numbers(self):
        result = self.run_node(FakeLLM(self.response()))
        summary, _ = extract_yaml_mapping(result["market_report"], "SUMMARY")
        self.assertEqual(summary["current_price"], history().Close.iloc[-1])
        self.assertEqual(summary["price_data_date"], "2026-11-05")
        self.assertEqual(summary["price_data_status"], "official_daily")
        self.assertGreater(summary["rsi_value"], 0)
        self.assertGreater(summary["atr_pct"], 0)
        self.assertIn("SHA256", result["market_report"])

    def test_incomplete_artifact_fails_before_later_agents(self):
        for response in ("", "<think>only private reasoning</think>", "普通正文", self.response().split("HANDOFF:")[0]):
            with self.subTest(response=response[:20]):
                llm = FakeLLM(response)
                with self.assertRaisesRegex(RuntimeError, "技术面"):
                    self.run_node(llm)
                self.assertEqual(len(llm.calls), 1)

    def test_input_cap_rejects_before_any_paid_request(self):
        llm = FakeLLM(self.response())
        with self.assertRaisesRegex(ValueError, "context|上下文"):
            self.run_node(llm, cap=100)
        self.assertEqual(llm.calls, [])

    def test_handoff_rejects_conflicting_current_price_and_future_asof(self):
        wrong_price = self.response().replace(
            f"value: {history().Close.iloc[-1]}", "value: 999999",
        )
        for content in (wrong_price, self.response().replace("as_of: 2026-11-05", "as_of: 2026-11-06")):
            with self.subTest(content=content[-200:]):
                with self.assertRaisesRegex(RuntimeError, "技术面"):
                    self.run_node(FakeLLM(content))

    def test_unknown_summary_direction_is_not_accepted_as_a_valid_report(self):
        response = self.response().replace("trend_daily: 震荡", "trend_daily: GUARANTEED_BUY")
        with self.assertRaisesRegex(RuntimeError, "技术面"):
            self.run_node(FakeLLM(response))

    def test_null_core_judgments_fail_instead_of_becoming_neutral_evidence(self):
        content = self.response()
        for key in ("trend", "trend_daily", "trend_weekly", "momentum", "data_implied_direction"):
            content = re.sub(rf"(?m)^  {key}:.*$", f"  {key}: null", content)
        with self.assertRaisesRegex(RuntimeError, "技术面"):
            self.run_node(FakeLLM(content))

    def test_handoff_calculation_facts_are_generated_by_code_not_model(self):
        content = self.response().replace("metric_or_event: 收盘价", "metric_or_event: RSI").replace(
            f"value: {history().Close.iloc[-1]}", "value: 999",
        ).replace("period: 2026-11-05", "period: 2027-01-01")
        result = self.run_node(FakeLLM(content))
        parsed = extract_handoff(result["market_report"], "market")
        facts = {fact["local_fact_key"]: fact for fact in parsed["facts"]}
        summary, _ = extract_yaml_mapping(result["market_report"], "SUMMARY")
        self.assertEqual(facts["rsi-14"]["value"], summary["rsi_value"])
        self.assertEqual(facts["price-latest"]["value"], summary["current_price"])
        self.assertTrue(all(fact["period"] <= "2026-11-05" for fact in facts.values()))

    def test_capital_flow_is_preserved_without_replaying_large_prose(self):
        llm = FakeLLM(self.response())
        capital = yaml.safe_dump({"CAPITAL_FLOW": {
            "capital_flow_regime": "恶化", "capital_flow_score": 22,
            "holder_num_qoq_pct": 15, "holder_num_latest_report_date": "2026-09-30",
            "irrelevant_raw_table": "RAW_TABLE" * 50000,
        }}, allow_unicode=True)
        result = self.run_node(llm, capital=capital)
        summary, _ = extract_yaml_mapping(result["market_report"], "SUMMARY")
        self.assertEqual(summary["capital_flow_regime"], "恶化")
        self.assertEqual(summary["capital_flow_score"], 22)
        text = "\n".join(message.content for message in llm.calls[0])
        self.assertIn("holder_num_latest_report_date", text)
        self.assertNotIn("RAW_TABLE", text)

    def test_production_factory_selects_compact_and_retains_legacy_rollback(self):
        from tradingagents.agents.analysts import market_analyst

        llm = FakeLLM(self.response())
        with patch.object(market_analyst, "get_config", return_value={"market_context_mode": "compact"}), patch.object(
            self.module, "create_compact_market_analyst", return_value=lambda state: {"marker": "compact"},
        ):
            result = market_analyst.create_market_analyst(llm)({})
        self.assertEqual(result, {"marker": "compact"})
        with patch.object(market_analyst, "get_config", return_value={"market_context_mode": "legacy"}), patch.object(
            self.module, "create_compact_market_analyst", side_effect=AssertionError("legacy should not select compact"),
        ):
            self.assertTrue(callable(market_analyst.create_market_analyst(llm)))


if __name__ == "__main__":
    unittest.main()
