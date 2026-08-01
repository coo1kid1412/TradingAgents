from __future__ import annotations

import json
import os
import sqlite3
import tempfile
import time
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase, main, mock

from langchain_openai import ChatOpenAI

from tradingagents.harness.market_warning.domain import (
    DataStatus,
    Evidence,
    FeatureSnapshot,
    FinalWarningDecision,
    Market,
    MarketPhase,
    QuantRiskAssessment,
    RiskLevel,
)
from tradingagents.harness.market_warning.reasoning import (
    CircuitBreaker,
    ReasoningValidationError,
    build_reasoning_prompt,
    should_call_reasoning,
    validate_context_assessment,
)
from tradingagents.harness.market_warning.adapters.minimax_reasoning import (
    MiniMaxReasoningAdapter,
)
from tradingagents.harness.market_warning.adapters.sqlite_repository import (
    SQLiteWarningRepository,
)
from tradingagents.llm_clients.openai_client import (
    NormalizedChatOpenAI,
    _io_logging_suppressed,
)


NOW = datetime(2026, 8, 1, 0, 30, tzinfo=timezone.utc)


def make_snapshot(*, slot: str = "premarket") -> FeatureSnapshot:
    evidence = tuple(
        Evidence(
            evidence_id=f"ev-{index}",
            group="feature",
            summary=f"signal {index}",
            value=index,
            source="test",
            as_of_time=NOW,
        )
        for index in range(1, 5)
    )
    return FeatureSnapshot(
        market=Market.A_SHARE,
        as_of_time=NOW,
        session_slot=slot,
        feature_version="market-warning-v2",
        features={
            "return_1d": -0.01,
            "breadth_up_pct": 28.0,
            "breadth_deterioration_transition": True,
        },
        evidence=evidence,
        data_quality=DataStatus.FRESH,
        reliability_grade="A",
        source_times={"test:first": NOW, "test:last": NOW},
    )


def make_quant() -> QuantRiskAssessment:
    return QuantRiskAssessment(
        crash_1d_probability=0.04,
        crash_3d_probability=0.08,
        market_phase=MarketPhase.FIRST_SHOCK,
        base_rate_1d=0.01,
        base_rate_3d=0.02,
        reliability_grade="A",
        model_version="model-v1",
        calibration_version="platt-v1",
        top_contributors=(
            {"feature": "return_1d", "contribution": 1.2, "evidence_id": "ev-1"},
            {"feature": "breadth_up_pct", "contribution": 0.8, "evidence_id": "ev-2"},
        ),
    )


def make_previous(level: RiskLevel = RiskLevel.YELLOW) -> FinalWarningDecision:
    return FinalWarningDecision(
        baseline_level=level,
        final_level=level,
        state_transition="UNCHANGED",
        entry_gate="OPEN",
        new_position_cap_pct=100.0,
        holding_action="HOLD",
        push_required=False,
        decision_reasons=("test",),
        data_status=DataStatus.FRESH,
    )


def valid_payload() -> dict[str, object]:
    return {
        "market_scenario": "breadth and volatility are deteriorating",
        "causal_chain": ["weak breadth", "higher fragility"],
        "supporting_evidence_ids": ["ev-1", "ev-2"],
        "conflicting_evidence_ids": ["ev-3"],
        "overlooked_risks": ["policy surprise"],
        "recommended_risk_level": "ORANGE",
        "confidence": 0.78,
        "action_reason": "reduce new risk while confirmation develops",
        "reasoning_status": "validated",
    }


class FakeLLM:
    def __init__(self, responses: list[object]) -> None:
        self.responses = list(responses)
        self.calls: list[tuple[object, object]] = []

    def invoke(self, prompt: object, config: object = None) -> object:
        self.calls.append((prompt, config))
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        if callable(response):
            return response()
        return SimpleNamespace(content=response)


class MutableClock:
    def __init__(self, value: datetime = NOW) -> None:
        self.value = value

    def __call__(self) -> datetime:
        return self.value


class ContextValidationTests(TestCase):
    def test_valid_payload_builds_typed_assessment(self) -> None:
        result = validate_context_assessment(valid_payload(), {"ev-1", "ev-2", "ev-3"})

        self.assertEqual(result.recommended_risk_level, RiskLevel.ORANGE)
        self.assertEqual(result.reasoning_status, "validated")
        self.assertIsNone(result.error_class)

    def test_invalid_payloads_fail_closed(self) -> None:
        cases = {
            "invalid level": {"recommended_risk_level": "PURPLE"},
            "unknown level": {"recommended_risk_level": "UNKNOWN"},
            "confidence high": {"confidence": 1.01},
            "confidence bool": {"confidence": True},
            "missing causal": {"causal_chain": []},
            "missing conflicting": {"conflicting_evidence_ids": []},
            "invented evidence": {"supporting_evidence_ids": ["ev-1", "invented"]},
            "nested evidence": {"supporting_evidence_ids": [["ev-1"], "ev-2"]},
            "extra field": {"raw_think": "must not survive"},
        }
        for name, changes in cases.items():
            with self.subTest(name=name):
                payload = valid_payload()
                payload.update(changes)
                with self.assertRaises(ReasoningValidationError):
                    validate_context_assessment(payload, {"ev-1", "ev-2", "ev-3"})

    def test_missing_required_key_is_rejected(self) -> None:
        for key in valid_payload():
            with self.subTest(key=key):
                payload = valid_payload()
                payload.pop(key)
                with self.assertRaises(ReasoningValidationError):
                    validate_context_assessment(payload, {"ev-1", "ev-2", "ev-3"})


class PromptAndCallPolicyTests(TestCase):
    def test_prompt_is_compact_structured_and_contains_no_think_request(self) -> None:
        prompt = build_reasoning_prompt(make_snapshot(), make_quant(), make_previous())

        self.assertIn('"strict_json_output"', prompt)
        self.assertIn('"ev-1"', prompt)
        self.assertIn('"current_baseline": "ORANGE"', prompt)
        self.assertNotIn("<think>", prompt.lower())
        self.assertLess(len(prompt), 12_000)

    def test_call_policy_matches_session_and_transition_contract(self) -> None:
        self.assertTrue(should_call_reasoning("premarket", RiskLevel.GREEN, None))
        self.assertFalse(should_call_reasoning("intraday", RiskLevel.GREEN, RiskLevel.GREEN))
        self.assertFalse(should_call_reasoning("intraday", RiskLevel.YELLOW, RiskLevel.GREEN))
        self.assertTrue(should_call_reasoning("intraday", RiskLevel.ORANGE, RiskLevel.YELLOW))
        self.assertTrue(should_call_reasoning("intraday", RiskLevel.RED, RiskLevel.ORANGE))


class AdapterTests(TestCase):
    def _adapter(self, llm: FakeLLM, **changes: object) -> MiniMaxReasoningAdapter:
        options = {"timeout": 0.05, **changes}
        return MiniMaxReasoningAdapter(llm, **options)

    def test_fenced_json_and_typed_blocks_discard_reasoning(self) -> None:
        raw = json.dumps(valid_payload())
        llm = FakeLLM(
            [[
                {"type": "reasoning", "text": "PRIVATE THINK CONTENT"},
                {"type": "text", "text": f"```json\n{raw}\n```"},
            ]]
        )

        result = self._adapter(llm).assess(make_snapshot(), make_quant(), make_previous())

        self.assertEqual(result.reasoning_status, "validated")
        self.assertNotIn("PRIVATE", repr(result))
        self.assertEqual(len(llm.calls), 1)
        config = llm.calls[0][1]
        self.assertTrue(_io_logging_suppressed(config))

    def test_empty_after_think_stripping_repairs_once_then_falls_back(self) -> None:
        secret = "RAW_PRIVATE_REASONING_987"
        llm = FakeLLM([f"<think>{secret}</think>", ""])

        result = self._adapter(llm).assess(make_snapshot(), make_quant(), make_previous())

        self.assertEqual(result.reasoning_status, "fallback")
        self.assertEqual(result.error_class, "empty_output")
        self.assertEqual(len(llm.calls), 2)
        self.assertNotIn(secret, str(llm.calls[1][0]))
        self.assertIn("empty_output", str(llm.calls[1][0]))
        self.assertNotIn("<think>", repr(result).lower())

    def test_invalid_json_gets_exactly_one_schema_repair(self) -> None:
        llm = FakeLLM(["not-json-private", json.dumps(valid_payload())])

        result = self._adapter(llm).assess(make_snapshot(), make_quant(), make_previous())

        self.assertEqual(result.reasoning_status, "validated")
        self.assertEqual(len(llm.calls), 2)
        self.assertIn("invalid_json", str(llm.calls[1][0]))
        self.assertIn("ev-1", str(llm.calls[1][0]))
        self.assertNotIn("not-json-private", str(llm.calls[1][0]))

    def test_timeout_repairs_once_without_waiting_for_worker_shutdown(self) -> None:
        def slow() -> object:
            time.sleep(0.15)
            return SimpleNamespace(content=json.dumps(valid_payload()))

        llm = FakeLLM([slow, slow])
        started = time.monotonic()

        result = self._adapter(llm, timeout=0.01).assess(
            make_snapshot(), make_quant(), make_previous()
        )

        self.assertLess(time.monotonic() - started, 0.12)
        self.assertEqual(result.reasoning_status, "fallback")
        self.assertEqual(result.error_class, "timeout")
        self.assertEqual(len(llm.calls), 2)

    def test_blocked_exceptions_are_coarsened_and_never_copied(self) -> None:
        secret = "provider moderation details secret-token"
        llm = FakeLLM([RuntimeError(f"1027 blocked {secret}"), RuntimeError(secret)])

        result = self._adapter(llm).assess(make_snapshot(), make_quant(), make_previous())

        self.assertEqual(result.error_class, "content_blocked")
        self.assertNotIn(secret, str(llm.calls[1][0]))
        self.assertNotIn(secret, repr(result))

    def test_prompt_build_failure_falls_back_without_invoking_llm(self) -> None:
        llm = FakeLLM([json.dumps(valid_payload())])
        adapter = self._adapter(llm)
        with mock.patch(
            "tradingagents.harness.market_warning.adapters.minimax_reasoning.build_reasoning_prompt",
            side_effect=TypeError("private provider-shaped object"),
        ):
            result = adapter.assess(make_snapshot(), make_quant(), make_previous())

        self.assertEqual(result.reasoning_status, "fallback")
        self.assertEqual(result.error_class, "prompt_error")
        self.assertEqual(llm.calls, [])

    def test_unsupported_evidence_values_are_redacted_from_prompt(self) -> None:
        snapshot = make_snapshot()
        evidence = list(snapshot.evidence)
        evidence[0] = replace(evidence[0], value=object())
        snapshot = replace(snapshot, evidence=tuple(evidence))
        llm = FakeLLM([json.dumps(valid_payload())])

        result = self._adapter(llm).assess(snapshot, make_quant(), make_previous())

        self.assertEqual(result.reasoning_status, "validated")
        self.assertIn('"value": null', str(llm.calls[0][0]))

    def test_three_final_failures_open_breaker_then_success_resets_it(self) -> None:
        clock = MutableClock()
        breaker = CircuitBreaker(
            failure_threshold=3,
            cooldown=timedelta(minutes=30),
            clock=clock,
        )
        invalid_pair = ["bad", "still bad"]
        llm = FakeLLM(invalid_pair * 3 + [json.dumps(valid_payload())])
        adapter = self._adapter(llm, breaker=breaker)

        for _ in range(3):
            self.assertEqual(
                adapter.assess(make_snapshot(), make_quant(), make_previous()).reasoning_status,
                "fallback",
            )
        self.assertEqual(len(llm.calls), 6)

        open_result = adapter.assess(make_snapshot(), make_quant(), make_previous())
        self.assertEqual(open_result.error_class, "circuit_open")
        self.assertEqual(len(llm.calls), 6)

        clock.value += timedelta(minutes=31)
        success = adapter.assess(make_snapshot(), make_quant(), make_previous())
        self.assertEqual(success.reasoning_status, "validated")
        self.assertEqual(len(llm.calls), 7)
        self.assertEqual(breaker.consecutive_failures, 0)

    def test_environment_factory_uses_m3_wrapped_client_and_validated_limits(self) -> None:
        wrapped = FakeLLM([json.dumps(valid_payload())])
        client = mock.Mock()
        client.get_llm_wrapped.return_value = wrapped
        env = {
            "MARKET_WARNING_LLM_TIMEOUT": "45",
            "MARKET_WARNING_LLM_MAX_TOKENS": "2048",
            "MINIMAX_BASE_URL": "https://example.invalid/v1",
        }
        with mock.patch.dict(os.environ, env, clear=False), mock.patch(
            "tradingagents.harness.market_warning.adapters.minimax_reasoning.create_llm_client",
            return_value=client,
        ) as factory:
            adapter = MiniMaxReasoningAdapter.from_environment()

        factory.assert_called_once_with(
            "minimax",
            "MiniMax-M3",
            "https://example.invalid/v1",
            timeout=45,
            max_tokens=2048,
            wall_clock_max_retries=0,
        )
        self.assertIs(adapter.llm, wrapped)

    def test_environment_factory_rejects_invalid_limits_without_building_client(self) -> None:
        for name, value in (
            ("MARKET_WARNING_LLM_TIMEOUT", "0"),
            ("MARKET_WARNING_LLM_TIMEOUT", "abc"),
            ("MARKET_WARNING_LLM_MAX_TOKENS", "999999"),
        ):
            with self.subTest(name=name, value=value), mock.patch.dict(
                os.environ, {name: value}, clear=False
            ), self.assertRaises(ValueError):
                MiniMaxReasoningAdapter.from_environment()


class RawLoggingGuardTests(TestCase):
    def test_market_warning_metadata_disables_raw_io_logging_only_for_that_call(self) -> None:
        self.assertTrue(
            _io_logging_suppressed(
                {"metadata": {"market_warning_disable_raw_io_logging": True}}
            )
        )
        self.assertFalse(_io_logging_suppressed(None))
        self.assertFalse(_io_logging_suppressed({"metadata": {}}))

    def test_normalized_client_does_not_persist_suppressed_prompt_or_raw_think(self) -> None:
        llm = NormalizedChatOpenAI.model_construct(model_name="MiniMax-M3")
        response = SimpleNamespace(content="<think>private chain</think>{}")
        config = {"metadata": {"market_warning_disable_raw_io_logging": True}}
        with mock.patch.object(ChatOpenAI, "invoke", return_value=response), mock.patch(
            "tradingagents.llm_clients.openai_client._log_llm_input"
        ) as input_log, mock.patch(
            "tradingagents.llm_clients.openai_client._log_llm_output"
        ) as output_log, mock.patch(
            "tradingagents.profiling.record_llm"
        ) as profiling_log:
            result = llm.invoke("private prompt", config=config)

        self.assertEqual(result.content, "{}")
        input_log.assert_not_called()
        output_log.assert_not_called()
        profiling_log.assert_not_called()


class ReasoningPersistenceTests(TestCase):
    def test_repository_stores_only_structured_fallback_and_coarse_error(self) -> None:
        secret = "provider raw secret and private think"
        adapter = MiniMaxReasoningAdapter(
            FakeLLM([RuntimeError(f"1027 {secret}"), RuntimeError(secret)]),
            timeout=0.05,
        )
        assessment = adapter.assess(make_snapshot(), make_quant(), make_previous())

        with tempfile.TemporaryDirectory() as directory:
            db_path = Path(directory) / "warning.db"
            repository = SQLiteWarningRepository(db_path)
            snapshot_id = repository.save_feature_snapshot(make_snapshot())
            reasoning_id = repository.save_reasoning(
                snapshot_id, assessment, "MiniMax-M3"
            )
            with sqlite3.connect(db_path) as connection:
                row = connection.execute(
                    "SELECT structured_json, error_class FROM market_warning_reasoning WHERE id = ?",
                    (reasoning_id,),
                ).fetchone()

        self.assertIsNotNone(row)
        structured_json, error_class = row
        self.assertEqual(error_class, "content_blocked")
        self.assertNotIn(secret, structured_json)
        self.assertNotIn("think", structured_json.lower())
        self.assertNotIn("error_class", structured_json)


if __name__ == "__main__":
    main()
