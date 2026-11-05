import unittest
import time
import contextlib
import io
import json
import re
from unittest.mock import mock_open, patch

from langchain_core.messages import AIMessage, HumanMessage

from tradingagents.llm_clients.minimax_client import MiniMaxClient
from tradingagents.llm_clients.openai_client import _log_llm_output


class ScriptedModel:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []
        self.bound_tools = None

    def invoke(self, input, config=None, **kwargs):
        self.calls.append((input, config, kwargs))
        return self.responses.pop(0)

    def bind_tools(self, tools, **kwargs):
        self.bound_tools = (tools, kwargs)
        return self


def truncated():
    return AIMessage(content="", response_metadata={"finish_reason": "length"})


def wrapped(model, **kwargs):
    client = MiniMaxClient("MiniMax-M3", max_tokens=8192, **kwargs)
    with patch.object(MiniMaxClient, "get_llm", return_value=model):
        return client.get_llm_wrapped()


class MiniMaxOutputRecoveryTests(unittest.TestCase):
    def test_request_diagnostics_measure_full_context_before_log_truncation(self):
        from langchain_openai import ChatOpenAI
        from tradingagents.llm_clients.openai_client import NormalizedChatOpenAI

        llm = NormalizedChatOpenAI(model="MiniMax-M3", api_key="sk-test", max_tokens=8192)
        bound = llm.bind_tools([{
            "type": "function", "function": {
                "name": "ic", "description": "IC", "parameters": {"type": "object"},
            },
        }])
        stderr = io.StringIO()
        messages = [HumanMessage(content="你是投资研究总监。" + "材料" * 20000)]
        with patch.object(ChatOpenAI, "invoke", return_value=AIMessage(content="done")), \
                patch("builtins.open", mock_open()), \
                patch("os.makedirs"), contextlib.redirect_stderr(stderr):
            bound.invoke(messages, max_tokens=16384)
        match = re.search(r"Request diagnostics: (\{[^\n]+\})", stderr.getvalue())
        self.assertIsNotNone(match, stderr.getvalue())
        diagnostics = json.loads(match.group(1))
        self.assertEqual(diagnostics["message_count"], 1)
        self.assertGreater(diagnostics["message_chars"], 40000)
        self.assertGreater(diagnostics["tool_schema_chars"], 0)
        self.assertEqual(diagnostics["max_completion_tokens"], 16384)
        self.assertEqual(diagnostics["agent"], "RM")

    def test_request_diagnostics_support_responses_payload_without_exposing_text(self):
        from tradingagents.llm_clients.openai_client import _request_diagnostics

        diagnostics = _request_diagnostics({
            "input": [{"role": "user", "content": "PRIVATE_INPUT_SENTINEL"}],
            "max_output_tokens": 8192,
        })
        self.assertEqual(diagnostics["message_count"], 1)
        self.assertGreater(diagnostics["message_chars"], 20)
        self.assertEqual(diagnostics["max_completion_tokens"], 8192)
        self.assertNotIn("PRIVATE_INPUT_SENTINEL", json.dumps(diagnostics))

    def test_empty_length_response_retries_once_with_larger_budget(self):
        model = ScriptedModel([truncated(), AIMessage(content="有效结论")])
        llm = wrapped(model)
        messages = [HumanMessage(content="生成报告")]
        result = llm.invoke(messages)
        self.assertEqual(result.content, "有效结论")
        self.assertEqual(len(model.calls), 2)
        self.assertIs(model.calls[1][0], messages)
        self.assertEqual(model.calls[1][2], {"max_tokens": 16384})

    def test_second_truncation_is_returned_without_an_unbounded_loop(self):
        model = ScriptedModel([truncated(), truncated()])
        self.assertEqual(wrapped(model).invoke("report").content, "")
        self.assertEqual(len(model.calls), 2)

    def test_partial_text_and_tool_calls_are_not_replayed(self):
        responses = [
            AIMessage(content="已有正文", response_metadata={"finish_reason": "length"}),
            AIMessage(content="", tool_calls=[{
                "name": "compute_ic_recommendation", "args": {}, "id": "ic-1",
            }], response_metadata={"finish_reason": "length"}),
            AIMessage(content="", invalid_tool_calls=[{
                "name": "ic", "args": "{", "id": "bad-1", "error": "invalid JSON",
            }], response_metadata={"finish_reason": "length"}),
            AIMessage(content="", response_metadata={"finish_reason": "stop"}),
        ]
        for response in responses:
            with self.subTest(response=response):
                model = ScriptedModel([response])
                self.assertIs(wrapped(model).invoke("report"), response)
                self.assertEqual(len(model.calls), 1)

    def test_budget_ceiling_and_opt_out_are_respected(self):
        for cap in (0, 8192):
            model = ScriptedModel([truncated()])
            wrapped(model, empty_output_retry_max_tokens=cap).invoke("report")
            self.assertEqual(len(model.calls), 1)
        model = ScriptedModel([truncated(), AIMessage(content="done")])
        wrapped(model, empty_output_retry_max_tokens=12000).invoke("report")
        self.assertEqual(model.calls[1][2]["max_tokens"], 12000)

    def test_whitespace_only_length_output_is_empty(self):
        model = ScriptedModel([
            AIMessage(content=" \n ", response_metadata={"finish_reason": "length"}),
            AIMessage(content="done"),
        ])
        self.assertEqual(wrapped(model).invoke("report").content, "done")

    def test_opt_out_preserves_original_wall_clock_wrapper(self):
        from tradingagents.llm_clients.base_client import WallClockTimeoutLLM

        model = ScriptedModel([truncated()])
        self.assertIsInstance(wrapped(model, empty_output_retry_max_tokens=0), WallClockTimeoutLLM)

    def test_graph_routes_role_budget_and_recovery_cap_to_actual_client(self):
        from tradingagents.default_config import DEFAULT_CONFIG
        from tradingagents.graph.trading_graph import TradingAgentsGraph

        graph = TradingAgentsGraph.__new__(TradingAgentsGraph)
        graph.config = {**DEFAULT_CONFIG, "minimax_empty_output_retry_max_tokens": 20000}
        graph.callbacks = []
        model = ScriptedModel([truncated(), AIMessage(content="done")])
        with patch.object(MiniMaxClient, "get_llm", return_value=model):
            llm = graph._create_role_llm("research_manager", 0.3)
        llm.invoke("report")
        self.assertEqual(model.calls[1][2]["max_tokens"], 20000)

    def test_real_langchain_payload_uses_the_increased_limit(self):
        llm = MiniMaxClient("MiniMax-M3", api_key="sk-test").get_llm()
        payload = llm._get_request_payload("report", max_tokens=16384)
        self.assertEqual(payload["max_completion_tokens"], 16384)
        self.assertNotIn("max_tokens", payload)

    def test_real_tool_binding_budget_is_not_lowered_on_retry(self):
        from tradingagents.llm_clients.openai_client import NormalizedChatOpenAI

        for limit, expected in ((20000, 32768), (32768, None)):
            with self.subTest(limit=limit):
                llm = MiniMaxClient("MiniMax-M3", api_key="sk-test").get_llm_wrapped()
                bound = llm.bind_tools([{
                    "type": "function", "function": {
                        "name": "ic", "description": "IC", "parameters": {"type": "object"},
                    },
                }], max_tokens=limit)
                with patch.object(NormalizedChatOpenAI, "invoke", side_effect=[
                    truncated(), AIMessage(content="done"),
                ]) as invoke:
                    bound.invoke("report")
                self.assertEqual(invoke.call_args_list[0].kwargs["max_tokens"], limit)
                self.assertEqual(invoke.call_count, 2 if expected else 1)
                if expected:
                    self.assertEqual(invoke.call_args_list[1].kwargs["max_tokens"], expected)

    def test_structured_output_retains_original_parser_path_without_hidden_retry(self):
        from langchain_core.exceptions import OutputParserException
        from tradingagents.llm_clients.base_client import WallClockTimeoutLLM
        from tradingagents.llm_clients.openai_client import NormalizedChatOpenAI

        for include_raw in (False, True):
            llm = MiniMaxClient("MiniMax-M3", api_key="sk-test").get_llm_wrapped()
            structured = llm.with_structured_output(
                method="json_mode", include_raw=include_raw,
            )
            self.assertIsInstance(structured, WallClockTimeoutLLM)
            with patch.object(NormalizedChatOpenAI, "invoke", return_value=truncated()) as invoke:
                if include_raw:
                    self.assertIsNone(structured.invoke("report")["parsed"])
                else:
                    with self.assertRaises(OutputParserException):
                        structured.invoke("report")
            self.assertEqual(invoke.call_count, 1)

    def test_bound_tools_keep_the_recovery_and_real_tool_result(self):
        final = AIMessage(content="", tool_calls=[{
            "name": "compute_ic_recommendation", "args": {"p_0": 34.47}, "id": "ic-1",
        }])
        model = ScriptedModel([truncated(), final])
        llm = wrapped(model).bind_tools(["ic"], tool_choice="auto")
        self.assertIs(llm.invoke("report"), final)
        self.assertEqual(len(model.calls), 2)
        self.assertEqual(model.bound_tools, (["ic"], {"tool_choice": "auto"}))

    def test_explicit_completion_limit_and_invocation_options_survive_retry(self):
        model = ScriptedModel([truncated(), AIMessage(content="done")])
        config = {"metadata": {"role": "RM"}}
        wrapped(model).invoke("report", config, max_completion_tokens=20000, stop=["END"])
        self.assertEqual(model.calls[1][2], {
            "max_completion_tokens": 32768, "stop": ["END"],
        })
        self.assertIs(model.calls[1][1], config)

    def test_strict_caller_owns_the_retry_budget(self):
        model = ScriptedModel([truncated()])
        wrapped(model).invoke("report", {"metadata": {
            "market_warning_disable_compliance_retry": True,
        }})
        self.assertEqual(len(model.calls), 1)

    def test_output_diagnostics_record_termination_not_private_reasoning(self):
        response = AIMessage(
            content="<think>PRIVATE_REASONING</think>",
            response_metadata={"finish_reason": "length"},
            usage_metadata={"input_tokens": 500, "output_tokens": 8192, "total_tokens": 8692},
        )
        writer = mock_open()
        with patch("builtins.open", writer):
            _log_llm_output(1, "MiniMax-M3", response, 55.0)
        recorded = "".join(call.args[0] for call in writer().write.call_args_list)
        self.assertNotIn("PRIVATE_REASONING", recorded)
        self.assertIn('"finish_reason": "length"', recorded)
        self.assertIn('"output_tokens": 8192', recorded)
        self.assertIn('"visible_chars": 0', recorded)
        self.assertEqual(response.content, "<think>PRIVATE_REASONING</think>")

    def test_each_attempt_gets_its_own_wall_clock_deadline(self):
        class SlowModel(ScriptedModel):
            def invoke(self, *args, **kwargs):
                time.sleep(0.12)
                return super().invoke(*args, **kwargs)

        model = SlowModel([truncated(), AIMessage(content="done")])
        result = wrapped(model, timeout=0.2, wall_clock_max_retries=0).invoke("report")
        self.assertEqual(result.content, "done")
        self.assertEqual(len(model.calls), 2)

    def test_rm_still_requires_a_real_successful_ic_tool_call(self):
        from tradingagents.agents.managers.research_manager import _run_tool_calling_loop
        from tradingagents.agents.managers.test_entry_timing import (
            _complete_rm_summary, _ICTool, _substantive_rm_body,
        )

        tool_request = AIMessage(content="", tool_calls=[{
            "name": "compute_ic_recommendation", "args": {}, "id": "ic-1",
        }])
        model = ScriptedModel([
            truncated(), tool_request,
            AIMessage(content=_substantive_rm_body() + "\n" + _complete_rm_summary()),
        ])
        tool = _ICTool()
        with patch.object(tool, "invoke", wraps=tool.invoke) as invoke_ic:
            result = _run_tool_calling_loop(
                wrapped(model), [HumanMessage(content="生成 RM 报告")],
                tools_by_name={"compute_ic_recommendation": tool},
                authoritative_tool_name="compute_ic_recommendation",
            )
        self.assertEqual(invoke_ic.call_count, 1)
        self.assertIn("HOLD", result.content)
        self.assertEqual(len(model.calls), 3)

    def test_rm_exhaustion_does_not_fabricate_authority(self):
        from tradingagents.agents.managers.research_manager import _run_tool_calling_loop

        model = ScriptedModel([truncated() for _ in range(6)])
        with self.assertRaisesRegex(RuntimeError, "未获得强制工具"):
            _run_tool_calling_loop(
                wrapped(model), [HumanMessage(content="生成 RM 报告")],
                tools_by_name={}, authoritative_tool_name="compute_ic_recommendation",
            )
        self.assertEqual(len(model.calls), 6)


if __name__ == "__main__":
    unittest.main()
