"""MiniMax LLM client for TradingAgents.

Uses MiniMax's OpenAI-compatible API at https://api.minimaxi.com/v1
via langchain-openai's ChatOpenAI, with max_tokens override for
financial analysis (MiniMax default 256 is too small).
"""

import logging
import os
from typing import Any

from langchain_core.runnables import Runnable

from .base_client import BaseLLMClient
from .validators import validate_model
from .openai_client import NormalizedChatOpenAI, _compliance_retry_suppressed

logger = logging.getLogger(__name__)

_MINIMAX_BASE_URL = "https://api.minimaxi.com/v1"
_MINIMAX_API_KEY_ENV = "MINIMAX_API_KEY"

# Kwargs forwarded from user config to ChatOpenAI
_PASSTHROUGH_KWARGS = (
    "timeout", "max_retries", "api_key", "callbacks",
    "http_client", "http_async_client", "temperature",
)


class MiniMaxOutputRecoveryLLM(Runnable):
    """Retry reasoning-only truncation once, with a separate per-call timeout."""

    def __init__(self, llm, token_limit: int, retry_token_cap: int):
        self._llm = llm
        self._token_limit = token_limit
        self._retry_token_cap = retry_token_cap

    def invoke(self, input, config=None, **kwargs):
        result = self._llm.invoke(input, config, **kwargs)
        metadata = getattr(result, "response_metadata", {}) or {}
        content = getattr(result, "content", "")
        if (
            _compliance_retry_suppressed(config)
            or metadata.get("finish_reason") != "length"
            or (content.strip() if isinstance(content, str) else content)
            or getattr(result, "tool_calls", None)
            or getattr(result, "invalid_tool_calls", None)
        ):
            return result

        limit_key = "max_completion_tokens" if "max_completion_tokens" in kwargs else "max_tokens"
        limit = kwargs.get(limit_key, self._token_limit)
        if not isinstance(limit, int) or limit <= 0:
            return result
        retry_limit = min(limit * 2, self._retry_token_cap)
        if retry_limit <= limit:
            return result

        logger.warning(
            "MiniMax 正文为空且 finish_reason=length；仅重试一次，输出上限 %d → %d",
            limit, retry_limit,
        )
        # Never replay a tool call or partial report; the original input is unchanged.
        return self._llm.invoke(input, config, **{**kwargs, limit_key: retry_limit})

    def bind_tools(self, *args, **kwargs):
        token_limit = kwargs.get(
            "max_completion_tokens", kwargs.get("max_tokens", self._token_limit),
        )
        return type(self)(
            self._llm.bind_tools(*args, **kwargs), token_limit, self._retry_token_cap,
        )

    def with_structured_output(self, *args, **kwargs):
        # Parsed-output callers retain their existing error/repair policy.
        return self._llm.with_structured_output(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._llm, name)


class MiniMaxClient(BaseLLMClient):
    """Client for MiniMax LLM provider.

    Uses MiniMax's OpenAI-compatible API endpoint (api.minimaxi.com/v1).
    Reads MINIMAX_API_KEY from environment and sets max_tokens=8192
    by default (upstream default 256 too small for financial analysis).
    """

    def get_llm_wrapped(self) -> Any:
        llm = super().get_llm_wrapped()
        retry_cap = self.kwargs.get("empty_output_retry_max_tokens", 32768)
        if not retry_cap:
            return llm
        return MiniMaxOutputRecoveryLLM(
            llm,
            self.kwargs.get("max_tokens", 8192),
            retry_cap,
        )

    def get_llm(self) -> Any:
        """Return configured NormalizedChatOpenAI instance for MiniMax."""
        api_key = self.kwargs.get("api_key") or os.environ.get(_MINIMAX_API_KEY_ENV)
        if not api_key:
            raise ValueError("缺少 MINIMAX_API_KEY，无法创建 MiniMax 客户端")

        llm_kwargs = {
            "model": self.model,
            "base_url": self.base_url or _MINIMAX_BASE_URL,
            "api_key": api_key,
            "max_tokens": self.kwargs.get("max_tokens", 8192),
            "timeout": self._get_timeout(),
        }

        # Forward user-provided kwargs (timeout already set, skip it)
        for key in _PASSTHROUGH_KWARGS:
            if key in self.kwargs and key not in llm_kwargs:
                llm_kwargs[key] = self.kwargs[key]

        return NormalizedChatOpenAI(**llm_kwargs)

    def validate_model(self) -> bool:
        """Validate model for MiniMax provider."""
        return validate_model("minimax", self.model)
