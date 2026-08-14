"""DeepSeek V4 client with explicit Think and safe observability boundaries."""

from __future__ import annotations

import logging
import os
import time
from typing import Any

from .base_client import BaseLLMClient
from .openai_client import NormalizedChatOpenAI
from .validators import validate_model


logger = logging.getLogger(__name__)

DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODELS = frozenset({"deepseek-v4-pro", "deepseek-v4-flash"})
_API_KEY_ENV = "DEEPSEEK_API_KEY"
_PASSTHROUGH_KWARGS = (
    "timeout", "max_retries", "callbacks", "http_client", "http_async_client",
    "temperature",
)


class DeepSeekNormalizedChatOpenAI(NormalizedChatOpenAI):
    """Disable raw I/O logs while retaining aggregate timing and usage logs."""

    def invoke(self, input, config=None, **kwargs):
        safe_config = dict(config or {})
        metadata = dict(safe_config.get("metadata") or {})
        metadata["market_warning_disable_raw_io_logging"] = True
        safe_config["metadata"] = metadata
        started = time.monotonic()
        response = super().invoke(input, safe_config, **kwargs)
        elapsed = time.monotonic() - started
        usage = getattr(response, "usage_metadata", None) or {}
        logger.info(
            "DeepSeek call model=%s duration=%.2fs content_chars=%d tokens=%s",
            self.model,
            elapsed,
            len(str(getattr(response, "content", "") or "")),
            usage,
        )
        return response


class DeepSeekClient(BaseLLMClient):
    """OpenAI-compatible DeepSeek V4 client with explicit reasoning controls."""

    def get_llm(self) -> Any:
        if self.model not in DEEPSEEK_MODELS:
            raise ValueError(
                f"DeepSeek 模型不受支持：{self.model}；"
                f"可用模型：{', '.join(sorted(DEEPSEEK_MODELS))}"
            )
        api_key = self.kwargs.get("api_key") or os.environ.get(_API_KEY_ENV)
        if not api_key:
            raise ValueError("缺少 DEEPSEEK_API_KEY，无法创建 DeepSeek 客户端")

        reasoning_effort = str(self.kwargs.get("reasoning_effort") or "high").lower()
        if reasoning_effort not in {"high", "max"}:
            raise ValueError("DeepSeek reasoning_effort 仅允许 high 或 max")

        extra_body = dict(self.kwargs.get("extra_body") or {})
        extra_body.update({
            "thinking": {"type": "enabled"},
            "reasoning_effort": reasoning_effort,
        })
        llm_kwargs = {
            "model": self.model,
            "base_url": (self.base_url or DEEPSEEK_BASE_URL).rstrip("/"),
            "api_key": api_key,
            "max_tokens": int(self.kwargs.get("max_tokens") or 8192),
            "timeout": self._get_timeout(),
            "extra_body": extra_body,
        }
        for key in _PASSTHROUGH_KWARGS:
            if key in self.kwargs and key not in llm_kwargs:
                llm_kwargs[key] = self.kwargs[key]
        return DeepSeekNormalizedChatOpenAI(**llm_kwargs)

    def validate_model(self) -> bool:
        return validate_model("deepseek", self.model)
