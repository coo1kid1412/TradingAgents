import importlib
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from langchain_core.messages import AIMessage, HumanMessage
import yaml

from test_technical_context import history, vendor_text
from tradingagents.agents.utils.handoff import extract_handoff
from tradingagents.agents.utils.yaml_summary import extract_yaml_mapping


class FakeLLM:
    def __init__(self, content):
        self.content = content
        self.calls = []

    def invoke(self, messages, config=None):
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
        return json.dumps({
            "trend_daily": "震荡", "trend_weekly": "震荡", "momentum": "中",
            "next_3_sessions_bias": "中性", "confidence": 3,
            "rationale": "均线与动量信号尚未形成一致方向。",
            "key_risk": "短期波动可能放大。", "entry_condition": "等待量价共同确认。",
            "invalidation_condition": "价格结构与量价配合进一步走弱。",
            "evidence_keys": ["rsi-14", "sma-50"],
            "support_key": None, "resistance_key": None,
        }, ensure_ascii=False)

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
        self.assertNotIn("```yaml", text)
        self.assertIn("小JSON", text)
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
        for response in ("", "<think>only private reasoning</think>", "普通正文", "{}"):
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

    def test_model_cannot_supply_conflicting_price_or_future_asof(self):
        for key, value in (("current_price", 999999), ("as_of", "2026-11-06")):
            content = json.loads(self.response())
            content[key] = value
            with self.subTest(key=key):
                with self.assertRaisesRegex(RuntimeError, "技术面"):
                    self.run_node(FakeLLM(json.dumps(content)))

    def test_unknown_summary_direction_is_not_accepted_as_a_valid_report(self):
        response = json.loads(self.response())
        response["trend_daily"] = "GUARANTEED_BUY"
        with self.assertRaisesRegex(RuntimeError, "技术面"):
            self.run_node(FakeLLM(json.dumps(response)))

    def test_null_core_judgments_fail_instead_of_becoming_neutral_evidence(self):
        for key in ("trend_daily", "trend_weekly", "momentum", "next_3_sessions_bias"):
            content = json.loads(self.response())
            content[key] = None
            with self.subTest(key=key), self.assertRaisesRegex(RuntimeError, "技术面"):
                self.run_node(FakeLLM(json.dumps(content)))

    def test_handoff_calculation_facts_are_generated_by_code_not_model(self):
        result = self.run_node(FakeLLM(self.response()))
        parsed = extract_handoff(result["market_report"], "market")
        facts = {fact["local_fact_key"]: fact for fact in parsed["facts"]}
        summary, _ = extract_yaml_mapping(result["market_report"], "SUMMARY")
        self.assertEqual(facts["rsi-14"]["value"], summary["rsi_value"])
        self.assertEqual(facts["price-latest"]["value"], summary["current_price"])
        self.assertTrue(all(fact["period"] <= "2026-11-05" for fact in facts.values()))
        self.assertIn("等待量价共同确认", json.dumps(parsed, ensure_ascii=False))
        self.assertIn("价格结构与量价配合进一步走弱", json.dumps(parsed, ensure_ascii=False))
        self.assertIsNone(summary["rating"])

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

    def test_invalid_judgments_fail_once_without_schema_repair(self):
        changes = [
            {"confidence": True}, {"confidence": 0}, {"confidence": 6},
            {"rationale": ""}, {"rationale": "x" * 121},
            {"rationale": "<think>PRIVATE</think>"},
            {"rationale": "price 999999"}, {"key_risk": "```yaml\nHANDOFF: {}"},
            {"evidence_keys": []}, {"evidence_keys": ["invented-fact"]},
            {"evidence_keys": ["sma-50", "sma-50"]},
            {"evidence_keys": [None]}, {"support_key": "rsi-14"},
            {"next_3_sessions_bias": "数据不足"},
        ]
        for change in changes:
            response = json.loads(self.response())
            response.update(change)
            llm = FakeLLM(json.dumps(response, ensure_ascii=False))
            with self.subTest(change=change), self.assertRaisesRegex(RuntimeError, "技术面"):
                self.run_node(llm)
            self.assertEqual(len(llm.calls), 1)

    def test_duplicate_json_fields_and_nonfinite_values_are_rejected(self):
        for content in (
            self.response().replace('"confidence": 3', '"confidence": 3, "confidence": 4'),
            self.response().replace('"confidence": 3', '"confidence": NaN'),
            '```json\n' + self.response() + '\n```',
            self.response() + ' trailing narration',
        ):
            with self.subTest(content=content[:30]), self.assertRaisesRegex(RuntimeError, "技术面"):
                self.run_node(FakeLLM(content))

    def test_selected_support_and_resistance_are_canonical_code_values(self):
        response = json.loads(self.response())
        response.update(support_key="prior-20d-low", resistance_key="prior-20d-high")
        result = self.run_node(FakeLLM(json.dumps(response)))
        summary, _ = extract_yaml_mapping(result["market_report"], "SUMMARY")
        self.assertEqual(summary["key_support"], round(history().Low.iloc[-21:-1].min(), 6))
        self.assertEqual(summary["key_resistance"], round(history().High.iloc[-21:-1].max(), 6))
        response.update(support_key="prior-20d-high")
        with self.assertRaisesRegex(RuntimeError, "技术面"):
            self.run_node(FakeLLM(json.dumps(response)))

    def test_truncated_response_and_tool_calls_do_not_become_reports(self):
        for response in (
            AIMessage(content=self.response(), response_metadata={"finish_reason": "length"}),
            AIMessage(content=self.response(), tool_calls=[{
                "id": "t", "name": "other_tool", "args": {},
            }]),
        ):
            class ReturningLLM:
                def invoke(self, messages, config=None):
                    return response
            with self.subTest(response=response.response_metadata), self.assertRaisesRegex(RuntimeError, "技术面"):
                self.run_node(ReturningLLM())

    def test_production_empty_output_never_triggers_the_large_budget_recovery(self):
        from langchain_openai import ChatOpenAI
        from tradingagents.llm_clients import create_llm_client

        with patch.dict(os.environ, {"MINIMAX_API_KEY": "offline-placeholder"}):
            llm = create_llm_client(provider="minimax", model="MiniMax-M3",
                                    max_retries=0, wall_clock_max_retries=0).get_llm_wrapped()
        with patch.object(ChatOpenAI, "invoke", return_value=AIMessage(
            content="", response_metadata={"finish_reason": "length"},
        )) as provider, patch("tradingagents.llm_clients.openai_client._log_llm_input"), patch(
            "tradingagents.llm_clients.openai_client._log_llm_output",
        ):
            with self.assertRaisesRegex(RuntimeError, "技术面"):
                self.run_node(llm)
        self.assertEqual(provider.call_count, 1)

    def test_missing_capital_and_price_provenance_are_explicit_in_artifacts(self):
        result = self.run_node(FakeLLM(self.response()))
        parsed = extract_handoff(result["market_report"], "market")
        self.assertEqual(parsed["quality"]["status"], "partial")
        self.assertIn("capital_flow", parsed["quality"]["missing_fields"])
        self.assertIn("资金面数据不足", result["market_report"])
        self.assertIn("2026-11-05", result["market_report"])

    def test_unavailable_evidence_is_rejected_instead_of_fabricated(self):
        response = json.loads(self.response())
        response["evidence_keys"] = ["sma-200"]
        with self.assertRaisesRegex(RuntimeError, "技术面"):
            self.run_node(FakeLLM(json.dumps(response)), raw=vendor_text(history(30)))

    def test_empty_or_partial_macd_composite_cannot_support_a_judgment(self):
        for rows in (10, 30):
            response = json.loads(self.response())
            response["evidence_keys"] = ["macd-lines"]
            with self.subTest(rows=rows), self.assertRaisesRegex(RuntimeError, "技术面"):
                self.run_node(FakeLLM(json.dumps(response)), raw=vendor_text(history(rows)))

    def test_every_free_text_field_rejects_markdown_links_and_images(self):
        for key in ("rationale", "key_risk", "entry_condition", "invalidation_condition"):
            for text in ("![技术图](https://attacker.example/pixel)",
                         "[消息](https://attacker.example/news)",
                         "**强烈看多**", "https://attacker.example/pixel"):
                response = json.loads(self.response())
                response[key] = text
                with self.subTest(key=key, text=text), self.assertRaisesRegex(RuntimeError, "技术面"):
                    self.run_node(FakeLLM(json.dumps(response)))

    def test_handoff_transmits_momentum_and_selected_levels_not_only_raw_facts(self):
        from tradingagents.agents.utils.handoff import pack_report_handoffs

        baseline = self.run_node(FakeLLM(self.response()))["market_report"]
        response = json.loads(self.response())
        response.update(momentum="强", support_key="prior-20d-low", resistance_key="prior-20d-high")
        updated = self.run_node(FakeLLM(json.dumps(response)))["market_report"]
        before = extract_handoff(baseline, "market")
        after = extract_handoff(updated, "market")
        self.assertNotEqual(before, after)
        self.assertEqual(after["specialist_view"]["momentum"], "强")
        levels = after["specialist_view"]["reference_levels"]
        self.assertEqual(levels["support_fact_key"], "prior-20d-low")
        self.assertEqual(levels["resistance_fact_key"], "prior-20d-high")
        self.assertNotEqual(pack_report_handoffs({"market": baseline}, budget_chars=10000),
                            pack_report_handoffs({"market": updated}, budget_chars=10000))

    def test_t_minus_one_date_is_not_upgraded_to_analysis_day(self):
        raw = vendor_text(history(end="2026-11-04"), status="t_minus_1")
        result = self.run_node(FakeLLM(self.response()), raw=raw)
        summary, _ = extract_yaml_mapping(result["market_report"], "SUMMARY")
        self.assertEqual(summary["price_data_date"], "2026-11-04")
        self.assertEqual(summary["price_data_status"], "t_minus_1")
        self.assertIn("t_minus_1", result["market_report"])

    def test_intraday_timestamp_and_status_survive_rendering(self):
        raw = vendor_text(history(), status="intraday_provisional", quote_time="2026-11-05 10:15:00")
        result = self.run_node(FakeLLM(self.response()), raw=raw)
        summary, _ = extract_yaml_mapping(result["market_report"], "SUMMARY")
        self.assertEqual(summary["price_data_date"], "2026-11-05")
        self.assertEqual(summary["price_data_time"], "2026-11-05 10:15:00")
        self.assertEqual(summary["price_data_status"], "intraday_provisional")
        self.assertEqual(summary["price_data_source"], "sina_realtime")

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

    def test_unconfigured_factory_does_not_activate_before_live_acceptance(self):
        from tradingagents.agents.analysts import market_analyst
        from tradingagents.default_config import DEFAULT_CONFIG

        self.assertEqual(DEFAULT_CONFIG["market_context_mode"], "legacy")
        with patch.object(market_analyst, "get_config", return_value={}), patch.object(
            self.module, "create_compact_market_analyst", side_effect=AssertionError("compact is opt-in"),
        ):
            self.assertTrue(callable(market_analyst.create_market_analyst(FakeLLM(self.response()))))


if __name__ == "__main__":
    unittest.main()
