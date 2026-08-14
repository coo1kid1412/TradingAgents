"""MiniMax M3 adapter that exposes only validated warning context."""

from __future__ import annotations

import json
import os
import re
from datetime import timedelta
from typing import Any, Mapping

from tradingagents.llm_clients.base_client import WallClockTimeoutLLM, _strip_think_tags
from tradingagents.llm_clients.factory import create_llm_client

from ..domain import (
    FeatureSnapshot,
    FinalWarningDecision,
    LLMContextAssessment,
    QuantRiskAssessment,
    RiskLevel,
    RunnerResult,
)
from ..reasoning import (
    CircuitBreaker,
    ReasoningValidationError,
    build_reasoning_prompt,
    build_rule_reasoning_prompt,
    validate_context_assessment,
    validate_rule_context_assessment,
)
from ..policy import baseline_level


_FENCE_RE = re.compile(r"^```(?:json)?\s*([\s\S]*?)\s*```$", re.IGNORECASE)
_NO_RAW_LOG_CONFIG = {
    "metadata": {
        "market_warning_disable_raw_io_logging": True,
        "market_warning_disable_compliance_retry": True,
    }
}
_REPAIR_SCHEMA = {
    "market_scenario": "non-empty string",
    "causal_chain": ["one or more non-empty strings"],
    "supporting_evidence_ids": ["at least two valid evidence IDs"],
    "conflicting_evidence_ids": ["at least one valid evidence ID"],
    "overlooked_risks": ["zero or more strings"],
    "recommended_risk_level": "GREEN|YELLOW|ORANGE|RED",
    "confidence": "number 0..1",
    "action_reason": "non-empty string",
    "reasoning_status": "validated",
}
_RULE_REPAIR_SCHEMA = {
    "market_scenario": "non-empty string",
    "causal_chain": ["one or more non-empty strings"],
    "conflicting_evidence_ids": ["zero or more valid evidence IDs"],
    "overlooked_risks": ["zero or more strings"],
}


def _env_int(name: str, default: int, minimum: int, maximum: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return value


def _response_text(response: Any) -> str:
    content = getattr(response, "content", response)
    if isinstance(content, list):
        content = "\n".join(
            item.get("text", "")
            for item in content
            if isinstance(item, Mapping) and item.get("type") == "text"
        )
    if not isinstance(content, str):
        raise ReasoningValidationError("empty_output")
    return _strip_think_tags(content).strip()


def _json_payload(text: str) -> Mapping[str, Any]:
    if not text:
        raise ReasoningValidationError("empty_output")
    fenced = _FENCE_RE.fullmatch(text)
    candidate = fenced.group(1).strip() if fenced else text.strip()
    if not candidate.startswith("{") or not candidate.endswith("}"):
        raise ReasoningValidationError("invalid_json")
    try:
        payload = json.loads(candidate)
    except (json.JSONDecodeError, UnicodeError) as exc:
        raise ReasoningValidationError("invalid_json") from exc
    if not isinstance(payload, Mapping):
        raise ReasoningValidationError("invalid_json")
    return payload


def _coarse_exception(error: BaseException) -> str:
    if isinstance(error, TimeoutError):
        return "timeout"
    if isinstance(error, ReasoningValidationError):
        return error.error_class
    text = str(error).lower()
    if any(marker in text for marker in ("1026", "1027", "blocked", "moderation", "sensitive")):
        return "content_blocked"
    return "invoke_error"


def _fallback(error_class: str) -> LLMContextAssessment:
    return LLMContextAssessment(
        market_scenario="LLM context unavailable; retain deterministic baseline.",
        causal_chain=(),
        supporting_evidence_ids=(),
        conflicting_evidence_ids=(),
        overlooked_risks=(),
        recommended_risk_level=RiskLevel.UNKNOWN,
        confidence=0.0,
        action_reason="Use the deterministic market-warning baseline.",
        reasoning_status="fallback",
        error_class=error_class,
    )


class UnavailableReasoningAdapter:
    """ReasoningPort that persists a coarse initialization failure."""

    model_name = "MiniMax-M3"

    def __init__(self, error_class: str = "initialization_error") -> None:
        self.error_class = error_class

    def assess(self, snapshot, quant, previous) -> LLMContextAssessment:
        return _fallback(self.error_class)

    def assess_rule_alert(self, result, previous) -> LLMContextAssessment:
        return _fallback(self.error_class)


def _repair_prompt(
    original_prompt: str,
    error_class: str,
    valid_evidence_ids: tuple[str, ...],
) -> str:
    context = json.loads(original_prompt)
    context["repair_request"] = {
        "task": "Retry once and return one corrected JSON object only.",
        "error_class": error_class,
        "valid_evidence_ids": list(valid_evidence_ids),
        "output_schema": _REPAIR_SCHEMA,
        "constraints": [
            "Use only evidence IDs from the original structured request.",
            "Do not include markdown, analysis, private reasoning, or prior output.",
        ],
    }
    return json.dumps(
        context,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ": "),
    )


def _rule_repair_prompt(
    original_prompt: str,
    error_class: str,
    valid_evidence_ids: tuple[str, ...],
) -> str:
    context = json.loads(original_prompt)
    context["repair_request"] = {
        "task": "Retry once with explanation fields only.",
        "error_class": error_class,
        "valid_evidence_ids": list(valid_evidence_ids),
        "output_schema": _RULE_REPAIR_SCHEMA,
        "constraints": [
            "Do not return a level, trade action, position cap, markdown, or private reasoning.",
            "Return one corrected JSON object only.",
        ],
    }
    return json.dumps(
        context,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ": "),
    )


class MiniMaxReasoningAdapter:
    """ReasoningPort implementation with one repair and a circuit breaker."""

    def __init__(
        self,
        llm: Any,
        model_name: str = "MiniMax-M3",
        timeout: float = 90,
        breaker: CircuitBreaker | None = None,
    ) -> None:
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or not 0 < timeout <= 90
        ):
            raise ValueError("timeout must be between 0 and 90 seconds")
        self.llm = llm
        self.model_name = model_name
        self.timeout = float(timeout)
        self.breaker = breaker or CircuitBreaker(
            failure_threshold=3, cooldown=timedelta(minutes=30)
        )

    @classmethod
    def from_environment(
        cls, breaker: CircuitBreaker | None = None
    ) -> "MiniMaxReasoningAdapter":
        timeout = _env_int("MARKET_WARNING_LLM_TIMEOUT", 90, 1, 90)
        max_tokens = _env_int("MARKET_WARNING_LLM_MAX_TOKENS", 4096, 256, 65536)
        base_url = os.environ.get("MINIMAX_BASE_URL") or None
        client = create_llm_client(
            "minimax",
            "MiniMax-M3",
            base_url,
            timeout=timeout,
            max_tokens=max_tokens,
            max_retries=0,
            wall_clock_max_retries=0,
        )
        return cls(client.get_llm_wrapped(), timeout=timeout, breaker=breaker)

    def assess(
        self,
        snapshot: FeatureSnapshot,
        quant: QuantRiskAssessment,
        previous: FinalWarningDecision | None,
    ) -> LLMContextAssessment:
        if not self.breaker.allow_call():
            return _fallback("circuit_open")

        try:
            prompt = build_reasoning_prompt(snapshot, quant, previous)
            prompt_context = json.loads(prompt)
            valid_ids = tuple(
                str(item["evidence_id"])
                for item in prompt_context.get("evidence", ())
                if isinstance(item, Mapping) and item.get("evidence_id")
            )
        except Exception:
            self.breaker.record_failure()
            return _fallback("prompt_error")
        baseline = baseline_level(quant, snapshot)
        first_error: str | None = None
        request = prompt
        for attempt in range(2):
            try:
                response = self._invoke(request)
                payload = _json_payload(_response_text(response))
                result = validate_context_assessment(
                    payload,
                    valid_ids,
                    baseline=baseline,
                    data_status=snapshot.data_quality,
                )
                self.breaker.record_success()
                return result
            except Exception as error:
                error_class = _coarse_exception(error)
                if first_error is None or error_class == "content_blocked":
                    first_error = error_class
                if attempt == 0:
                    request = _repair_prompt(prompt, error_class, valid_ids)

        self.breaker.record_failure()
        return _fallback(first_error or "reasoning_unavailable")

    def assess_rule_alert(
        self,
        result: RunnerResult,
        previous: FinalWarningDecision | None,
    ) -> LLMContextAssessment:
        if not self.breaker.allow_call():
            return _fallback("circuit_open")
        try:
            prompt = build_rule_reasoning_prompt(result, previous)
            prompt_context = json.loads(prompt)
            valid_ids = tuple(
                str(item["evidence_id"])
                for item in prompt_context.get("evidence", ())
                if isinstance(item, Mapping) and item.get("evidence_id")
            )
            supporting_ids = tuple(
                dict.fromkeys(
                    str(evidence_id)
                    for rule in prompt_context.get("triggered_rules", ())
                    if isinstance(rule, Mapping)
                    for evidence_id in rule.get("evidence_ids", ())
                    if evidence_id in valid_ids
                )
            )
            decision = result.decision
            if decision is None:
                raise ReasoningValidationError("prompt_error")
        except Exception:
            self.breaker.record_failure()
            return _fallback("prompt_error")

        first_error: str | None = None
        request = prompt
        attempt_timeout = self.timeout / 2.0
        for attempt in range(2):
            try:
                response = self._invoke(request, timeout=attempt_timeout)
                payload = _json_payload(_response_text(response))
                assessment = validate_rule_context_assessment(
                    payload,
                    valid_ids,
                    supporting_ids,
                    decision.final_level,
                )
                self.breaker.record_success()
                return assessment
            except Exception as error:
                error_class = _coarse_exception(error)
                if first_error is None or error_class == "content_blocked":
                    first_error = error_class
                if attempt == 0:
                    request = _rule_repair_prompt(prompt, error_class, valid_ids)

        self.breaker.record_failure()
        return _fallback(first_error or "reasoning_unavailable")

    def _invoke(self, prompt: str, *, timeout: float | None = None) -> Any:
        if timeout is None and isinstance(self.llm, WallClockTimeoutLLM):
            llm = self.llm
        else:
            raw_llm = self.llm._llm if isinstance(self.llm, WallClockTimeoutLLM) else self.llm
            llm = WallClockTimeoutLLM(
                raw_llm,
                timeout=self.timeout if timeout is None else timeout,
                max_retries=0,
            )
        return llm.invoke(prompt, config=_NO_RAW_LOG_CONFIG)
