"""MiniMax LLM client for TradingAgents.

Uses MiniMax's OpenAI-compatible API at https://api.minimaxi.com/v1
via langchain-openai's ChatOpenAI, with max_tokens override for
financial analysis (MiniMax default 256 is too small).
"""

import os
from typing import Any

from .base_client import BaseLLMClient, normalize_content
from .validators import validate_model
from .openai_client import NormalizedChatOpenAI

_MINIMAX_BASE_URL = "https://api.minimaxi.com/v1"
_MINIMAX_API_KEY_ENV = "MINIMAX_API_KEY"

# Kwargs forwarded from user config to ChatOpenAI
_PASSTHROUGH_KWARGS = (
    "timeout", "max_retries", "api_key", "callbacks",
    "http_client", "http_async_client", "temperature",
)


class MiniMaxClient(BaseLLMClient):
    """Client for MiniMax LLM provider.

    Uses MiniMax's OpenAI-compatible API endpoint (api.minimaxi.com/v1).
    Reads MINIMAX_API_KEY from environment and sets max_tokens=8192
    by default (upstream default 256 too small for financial analysis).
    """

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
