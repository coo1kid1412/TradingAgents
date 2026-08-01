"""MiniMax M3 adapter that exposes only validated warning context."""

from __future__ import annotations

import json
import os
import queue
import re
import threading
from datetime import timedelta
from typing import Any, Mapping

from tradingagents.llm_clients.base_client import _strip_think_tags
from tradingagents.llm_clients.factory import create_llm_client

from ..domain import (
    FeatureSnapshot,
    FinalWarningDecision,
    LLMContextAssessment,
    QuantRiskAssessment,
    RiskLevel,
)
from ..reasoning import (
    CircuitBreaker,
    ReasoningValidationError,
    build_reasoning_prompt,
    validate_context_assessment,
)


_FENCE_RE = re.compile(r"^```(?:json)?\s*([\s\S]*?)\s*```$", re.IGNORECASE)
_NO_RAW_LOG_CONFIG = {
    "metadata": {"market_warning_disable_raw_io_logging": True}
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
        market_scenario="M3 context unavailable; retain deterministic baseline.",
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


def _repair_prompt(error_class: str, valid_evidence_ids: tuple[str, ...]) -> str:
    return json.dumps(
        {
            "task": "Return one corrected JSON object only.",
            "error_class": error_class,
            "valid_evidence_ids": list(valid_evidence_ids),
            "output_schema": _REPAIR_SCHEMA,
            "constraints": [
                "Use only evidence IDs from the original structured request.",
                "Do not include markdown, analysis, private reasoning, or prior output.",
            ],
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
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
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or timeout <= 0:
            raise ValueError("timeout must be positive")
        self.llm = llm
        self.model_name = model_name
        self.timeout = float(timeout)
        self.breaker = breaker or CircuitBreaker(
            failure_threshold=3, cooldown=timedelta(minutes=30)
        )

    @classmethod
    def from_environment(cls) -> "MiniMaxReasoningAdapter":
        timeout = _env_int("MARKET_WARNING_LLM_TIMEOUT", 90, 1, 600)
        max_tokens = _env_int("MARKET_WARNING_LLM_MAX_TOKENS", 4096, 256, 65536)
        base_url = os.environ.get("MINIMAX_BASE_URL") or None
        client = create_llm_client(
            "minimax",
            "MiniMax-M3",
            base_url,
            timeout=timeout,
            max_tokens=max_tokens,
            wall_clock_max_retries=0,
        )
        return cls(client.get_llm_wrapped(), timeout=timeout)

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
        except Exception:
            self.breaker.record_failure()
            return _fallback("prompt_error")
        valid_ids = snapshot.evidence_ids
        first_error: str | None = None
        request = prompt
        for attempt in range(2):
            try:
                response = self._invoke(request)
                payload = _json_payload(_response_text(response))
                result = validate_context_assessment(payload, valid_ids)
                self.breaker.record_success()
                return result
            except Exception as error:
                error_class = _coarse_exception(error)
                if first_error is None or error_class == "content_blocked":
                    first_error = error_class
                if attempt == 0:
                    request = _repair_prompt(error_class, valid_ids)

        self.breaker.record_failure()
        return _fallback(first_error or "reasoning_unavailable")

    def _invoke(self, prompt: str) -> Any:
        result_queue: queue.Queue[tuple[bool, Any]] = queue.Queue(maxsize=1)

        def worker() -> None:
            try:
                result_queue.put(
                    (True, self.llm.invoke(prompt, config=_NO_RAW_LOG_CONFIG)),
                    block=False,
                )
            except Exception as error:
                result_queue.put((False, error), block=False)

        thread = threading.Thread(target=worker, daemon=True, name="market-warning-m3")
        thread.start()
        try:
            ok, value = result_queue.get(timeout=self.timeout)
        except queue.Empty as exc:
            raise TimeoutError("market warning reasoning timed out") from exc
        if not ok:
            raise value
        return value
