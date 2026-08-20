"""DeepSeek V4 Pro adapter for explanation-only market-warning reasoning."""

from __future__ import annotations

import os
from typing import Any

from tradingagents.llm_clients.deepseek_client import DEEPSEEK_BASE_URL
from tradingagents.llm_clients.factory import create_llm_client

from .minimax_reasoning import (
    MiniMaxReasoningAdapter,
    UnavailableReasoningAdapter,
    _env_int,
)
from ..reasoning import CircuitBreaker


MODEL_NAME = "deepseek-v4-pro"


class DeepSeekUnavailableReasoningAdapter(UnavailableReasoningAdapter):
    """Persist a coarse DeepSeek initialization failure without provider details."""

    model_name = MODEL_NAME


class DeepSeekReasoningAdapter(MiniMaxReasoningAdapter):
    """Use DeepSeek Pro/max Think behind the deterministic warning contract."""

    def __init__(
        self,
        llm: Any,
        model_name: str = MODEL_NAME,
        timeout: float = 90,
        breaker: CircuitBreaker | None = None,
    ) -> None:
        super().__init__(llm, model_name=model_name, timeout=timeout, breaker=breaker)

    @classmethod
    def from_environment(
        cls, breaker: CircuitBreaker | None = None
    ) -> "DeepSeekReasoningAdapter":
        timeout = _env_int("MARKET_WARNING_LLM_TIMEOUT", 90, 1, 90)
        max_tokens = _env_int("MARKET_WARNING_LLM_MAX_TOKENS", 4096, 256, 65536)
        base_url = os.environ.get("DEEPSEEK_BASE_URL") or DEEPSEEK_BASE_URL
        client = create_llm_client(
            "deepseek",
            MODEL_NAME,
            base_url,
            timeout=timeout,
            max_tokens=max_tokens,
            max_retries=0,
            wall_clock_max_retries=0,
            reasoning_effort="max",
        )
        return cls(client.get_llm_wrapped(), timeout=timeout, breaker=breaker)
