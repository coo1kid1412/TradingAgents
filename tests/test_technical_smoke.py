"""Offline contract tests; no provider or market-data calls are permitted."""

import builtins
import contextlib
import io
import json
import os
from pathlib import Path
import socket
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from tradingagents import technical_smoke as smoke


class OfflineTestCase(unittest.TestCase):
    def setUp(self):
        self.enterContext(patch.object(
            socket.socket, "connect", side_effect=AssertionError("network forbidden")
        ))
        self.enterContext(patch.object(
            socket, "create_connection", side_effect=AssertionError("network forbidden")
        ))


class OneRequestTests(OfflineTestCase):
    def test_second_request_is_rejected_after_success(self):
        response = AIMessage(content="done")
        delegate = Mock(invoke=Mock(return_value=response))
        llm = smoke.OneRequest(delegate)
        self.assertIs(llm.invoke("prompt"), response)
        with self.assertRaises(RuntimeError):
            llm.invoke("second prompt")
        self.assertEqual(delegate.invoke.call_count, 1)
        self.assertEqual(llm.request_count, 1)
        self.assertEqual(llm.input_chars, 6)
        self.assertEqual(llm.max_tokens, 8192)

    def test_exception_consumes_budget_and_preserves_usage_without_raw_text(self):
        error = RuntimeError("PRIVATE_REASONING_AND_PROVIDER_BODY")
        error.body = {
            "usage": {"prompt_tokens": 15, "completion_tokens": 8192,
                      "total_tokens": 8207, "reasoning_content": "PRIVATE"},
            "choices": [{"finish_reason": "length", "message": {"content": "PRIVATE"}}],
        }
        delegate = Mock(invoke=Mock(side_effect=error))
        llm = smoke.OneRequest(delegate)
        with self.assertRaises(RuntimeError):
            llm.invoke("prompt")
        with self.assertRaises(RuntimeError):
            llm.invoke("retry")
        self.assertEqual(delegate.invoke.call_count, 1)
        self.assertEqual(llm.usage["completion_tokens"], 8192)
        self.assertEqual(llm.finish_reason, "length")
        self.assertNotIn("PRIVATE", json.dumps(llm.usage))

    def test_forces_suppression_even_if_caller_disables_it(self):
        delegate = Mock(invoke=Mock(return_value=AIMessage(content="done")))
        config = {"metadata": {"market_warning_disable_compliance_retry": False}}
        smoke.OneRequest(delegate).invoke("prompt", config=config)
        actual = delegate.invoke.call_args.kwargs["config"]["metadata"]
        self.assertIs(actual["market_warning_disable_compliance_retry"], True)
        self.assertIs(actual["market_warning_disable_raw_io_logging"], True)
        self.assertIs(config["metadata"]["market_warning_disable_compliance_retry"], False)

    def test_input_ceiling_is_checked_before_delegate_including_unicode(self):
        delegate = Mock()
        llm = smoke.OneRequest(delegate)
        messages = [SystemMessage(content="s"), HumanMessage(content="x" * 24000)]
        with self.assertRaises(ValueError):
            llm.invoke(messages)
        delegate.invoke.assert_not_called()
        self.assertEqual(llm.request_count, 0)
        self.assertEqual(llm.input_chars, 24001)
        delegate.invoke.return_value = AIMessage(content="done")
        llm.invoke([HumanMessage(content="\u4e2d" * 24000)])
        self.assertEqual(llm.input_chars, 24000)

    def test_unknown_inputs_and_invocation_overrides_fail_before_request(self):
        delegate = Mock()
        llm = smoke.OneRequest(delegate)
        with self.assertRaises((TypeError, ValueError)):
            llm.invoke(object())
        for kwargs in ({"max_tokens": 32768}, {"tools": ["tool"]}, {"max_retries": 2}):
            with self.subTest(kwargs=kwargs), self.assertRaises(TypeError):
                llm.invoke("prompt", **kwargs)
        delegate.invoke.assert_not_called()
        self.assertFalse(hasattr(llm, "bind_tools"))

    def test_usage_and_finish_reason_do_not_include_private_reasoning(self):
        response = AIMessage(
            content="already normalized",
            usage_metadata={"input_tokens": 8, "output_tokens": 3, "total_tokens": 11,
                            "output_token_details": {"reasoning": 2}},
            response_metadata={"finish_reason": "stop", "reasoning_content": "PRIVATE"},
            additional_kwargs={"reasoning_content": "PRIVATE"},
        )
        llm = smoke.OneRequest(Mock(invoke=Mock(return_value=response)))
        self.assertIs(llm.invoke([HumanMessage(content="test")]), response)
        self.assertEqual(llm.usage["total_tokens"], 11)
        self.assertEqual(llm.usage["output_token_details"]["reasoning"], 2)
        self.assertEqual(llm.finish_reason, "stop")
        self.assertNotIn("PRIVATE", json.dumps(llm.usage))

    def test_keeps_only_normalized_visible_content_for_failed_artifact_diagnosis(self):
        response = AIMessage(content="<think>PRIVATE_CHAIN</think>可见但尚未校验的正文")
        llm = smoke.OneRequest(Mock(invoke=Mock(return_value=response)))
        llm.invoke("test")
        self.assertEqual(getattr(llm, "visible_content", None), "可见但尚未校验的正文")

    def test_response_metadata_usage_is_supported(self):
        response = SimpleNamespace(response_metadata={
            "token_usage": {"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6},
            "finish_reason": "length",
        })
        llm = smoke.OneRequest(Mock(invoke=Mock(return_value=response)))
        llm.invoke("test")
        self.assertEqual(llm.usage["total_tokens"], 6)
        self.assertEqual(llm.finish_reason, "length")

    def test_real_minimax_stack_never_retries_moderation_timeout_or_empty_output(self):
        from langchain_openai import ChatOpenAI
        from tradingagents.llm_clients import create_llm_client

        for outcome in (
            RuntimeError("422 1026 input new_sensitive"),
            RuntimeError("422 1027 output new_sensitive"),
            TimeoutError("timeout"),
            AIMessage(content="", response_metadata={"finish_reason": "length"}),
        ):
            with self.subTest(outcome=type(outcome).__name__), patch.dict(
                os.environ, {"MINIMAX_API_KEY": "offline-placeholder"}
            ):
                client = create_llm_client(
                    provider="minimax", model="MiniMax-M3", max_tokens=8192,
                    max_retries=0, wall_clock_max_retries=0,
                    empty_output_retry_max_tokens=0, timeout=180,
                )
                delegate = client.get_llm_wrapped()
                self.assertEqual(delegate._max_retries, 0)
                self.assertEqual(delegate._llm.max_retries, 0)
                self.assertEqual(delegate._llm.max_tokens, 8192)
                llm = smoke.OneRequest(delegate)
                kwargs = {"side_effect": outcome} if isinstance(outcome, Exception) else {
                    "return_value": outcome
                }
                with patch.object(ChatOpenAI, "invoke", **kwargs) as provider, patch(
                    "tradingagents.llm_clients.openai_client._log_llm_input"
                ) as input_log, patch(
                    "tradingagents.llm_clients.openai_client._log_llm_output"
                ) as output_log:
                    if isinstance(outcome, Exception):
                        with self.assertRaises(type(outcome)):
                            llm.invoke("prompt")
                    else:
                        self.assertIs(llm.invoke("prompt"), outcome)
                    self.assertEqual(provider.call_count, 1)
                    input_log.assert_not_called()
                    output_log.assert_not_called()


class SmokeCLITests(OfflineTestCase):
    def setUp(self):
        super().setUp()
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.output = self.root / "run"
        self.config = {
            "llm_provider": "minimax", "deep_think_llm": "MiniMax-M3",
            "backend_url": "https://example.invalid/v1", "secret": "PRIVATE_CONFIG",
        }
        self.response = AIMessage(
            content="validated report", response_metadata={"finish_reason": "stop"},
            usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
        )
        self.delegate = Mock(invoke=Mock(return_value=self.response))
        self.client = Mock(get_llm_wrapped=Mock(return_value=self.delegate))
        self.factory = self.enterContext(patch.object(
            smoke, "create_llm_client", return_value=self.client
        ))
        self.set_config = self.enterContext(patch.object(smoke, "set_config"))
        self.build_config = Mock(return_value=self.config)
        self.enterContext(patch.dict(sys.modules, {"main": SimpleNamespace(
            _build_config=self.build_config
        )}))
        self.states = []

        def node_factory(llm):
            def node(state):
                self.states.append(state)
                llm.invoke([HumanMessage(content="bounded prompt")])
                return {"market_report": "validated report", "messages": [self.response]}
            return node

        self.node_factory = self.enterContext(patch.object(
            smoke, "create_compact_market_analyst", side_effect=node_factory
        ))
        self.enterContext(contextlib.redirect_stdout(io.StringIO()))
        self.enterContext(contextlib.redirect_stderr(io.StringIO()))

    def run_cli(self, *extra):
        return smoke.main([
            "--ticker", "301171", "--date", "2026-11-05",
            "--output-dir", str(self.output), *extra,
        ])

    def result(self):
        return json.loads((self.output / "result.json").read_text(encoding="utf-8"))

    def test_success_uses_only_single_node_and_exact_bounded_model_configuration(self):
        self.assertEqual(self.run_cli(), 0)
        self.build_config.assert_called_once_with()
        self.factory.assert_called_once_with(
            provider="minimax", model="MiniMax-M3", base_url=self.config["backend_url"],
            max_tokens=8192, max_retries=0, wall_clock_max_retries=0,
            empty_output_retry_max_tokens=0, timeout=180,
        )
        self.client.get_llm_wrapped.assert_called_once_with()
        self.node_factory.assert_called_once()
        self.assertEqual(self.states, [{
            "company_of_interest": "301171", "company_name": "301171",
            "trade_date": "2026-11-05", "capital_flow_yaml": "", "messages": [],
        }])
        actual_config = self.set_config.call_args.args[0]
        self.assertEqual(actual_config["market_context_max_chars"], 24000)
        self.assertTrue(Path(actual_config["data_cache_dir"]).is_relative_to(self.output.resolve()))
        self.assertEqual((self.output / "report.md").read_text(), "validated report")
        result = self.result()
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["request_count"], 1)
        self.assertEqual(result["input_chars"], len("bounded prompt"))
        self.assertEqual(result["max_tokens"], 8192)
        self.assertEqual(result["usage"]["total_tokens"], 15)
        self.assertIsNone(result["error"])
        self.assertNotIn("PRIVATE_CONFIG", json.dumps(result))

    def test_node_failure_retains_usage_and_never_publishes_report(self):
        def factory(llm):
            def node(state):
                llm.invoke("prompt")
                raise ValueError("invalid SUMMARY PRIVATE_REASONING")
            return node
        self.node_factory.side_effect = factory
        self.assertEqual(self.run_cli(), 1)
        result = self.result()
        self.assertEqual(result["request_count"], 1)
        self.assertEqual(result["usage"]["total_tokens"], 15)
        self.assertEqual(result["error"]["type"], "ValueError")
        self.assertNotIn("PRIVATE_REASONING", json.dumps(result))
        self.assertFalse((self.output / "report.md").exists())
        self.assertEqual((self.output / "model_visible.md").read_text(), "validated report")

    def test_provider_failure_records_exception_usage_once(self):
        error = RuntimeError("PRIVATE_PROVIDER_BODY")
        error.usage_metadata = {"input_tokens": 13, "output_tokens": 8, "total_tokens": 21}
        self.delegate.invoke.side_effect = error
        self.assertEqual(self.run_cli(), 1)
        result = self.result()
        self.assertEqual(result["request_count"], 1)
        self.assertEqual(result["usage"]["total_tokens"], 21)
        self.assertNotIn("PRIVATE_PROVIDER_BODY", json.dumps(result))
        self.assertEqual(self.delegate.invoke.call_count, 1)

    def test_config_failure_is_saved_with_zero_requests(self):
        self.build_config.side_effect = RuntimeError("PRIVATE_CONFIG")
        self.assertEqual(self.run_cli(), 1)
        self.assertEqual(self.result()["request_count"], 0)
        self.factory.assert_not_called()

    def test_non_minimax_is_rejected_without_selecting_another_model(self):
        self.config["llm_provider"] = "deepseek"
        self.assertEqual(self.run_cli(), 1)
        self.factory.assert_not_called()
        self.node_factory.assert_not_called()
        self.assertEqual(self.result()["request_count"], 0)

    def test_prompt_overflow_fails_before_any_request(self):
        self.node_factory.side_effect = lambda llm: lambda state: llm.invoke("x" * 24001)
        self.assertEqual(self.run_cli(), 1)
        self.delegate.invoke.assert_not_called()
        self.assertEqual(self.result()["input_chars"], 24001)
        self.assertEqual(self.result()["request_count"], 0)

    def test_node_cannot_request_a_second_generation(self):
        def factory(llm):
            def node(state):
                llm.invoke("prompt")
                llm.invoke("repair")
            return node
        self.node_factory.side_effect = factory
        self.assertEqual(self.run_cli(), 1)
        self.assertEqual(self.result()["request_count"], 1)
        self.assertEqual(self.delegate.invoke.call_count, 1)
        self.assertFalse((self.output / "report.md").exists())

    def test_missing_or_empty_report_fails_closed(self):
        for result in ({}, {"market_report": ""}, {"market_report": " \n"}, None):
            with self.subTest(result=result):
                self.output = self.root / f"invalid_{len(list(self.root.iterdir()))}"
                self.node_factory.side_effect = lambda llm: lambda state: result
                self.assertEqual(self.run_cli(), 1)
                self.assertFalse((self.output / "report.md").exists())

    def test_artifact_paths_are_recorded_after_node_failure(self):
        def factory(llm):
            def node(state):
                path = Path(self.set_config.call_args.args[0]["data_cache_dir"]) / "full.json"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("{}")
                raise ValueError("invalid SUMMARY")
            return node
        self.node_factory.side_effect = factory
        self.assertEqual(self.run_cli(), 1)
        self.assertEqual(len(self.result()["artifact_paths"]), 1)
        self.assertTrue(Path(self.result()["artifact_paths"][0]).is_file())

    def test_existing_output_is_not_overwritten(self):
        self.output.mkdir()
        report = self.output / "report.md"
        report.write_text("old report")
        self.assertEqual(self.run_cli(), 1)
        self.assertEqual(report.read_text(), "old report")
        self.assertFalse((self.output / "result.json").exists())
        self.build_config.assert_not_called()

    def test_date_validation_happens_before_model_creation(self):
        self.assertEqual(self.run_cli("--date", "2026-02-30"), 1)
        self.assertEqual(self.result()["request_count"], 0)
        self.factory.assert_not_called()

    def test_output_directory_is_required(self):
        with self.assertRaises(SystemExit) as error:
            smoke.main(["--ticker", "301171", "--date", "2026-11-05"])
        self.assertEqual(error.exception.code, 2)
        self.build_config.assert_not_called()

    def test_cli_does_not_import_graph_runner_or_feishu(self):
        original_import = builtins.__import__

        def guarded_import(name, *args, **kwargs):
            if any(fragment in name.lower() for fragment in ("graph", "feishu", "analysis_runner")):
                raise AssertionError(f"forbidden full-analysis import: {name}")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=guarded_import):
            self.assertEqual(self.run_cli(), 0)
        self.assertEqual(self.delegate.invoke.call_count, 1)


if __name__ == "__main__":
    unittest.main()
