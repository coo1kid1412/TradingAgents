"""Resolve the active LLM provider from one environment switch."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Mapping, MutableMapping


DEFAULT_LLM_PROVIDER = "minimax"


@dataclass(frozen=True)
class LLMProviderSettings:
    provider: str
    backend_url: str
    deep_model: str
    quick_model: str


def resolve_llm_provider_settings(
    environ: Mapping[str, str] | None = None,
) -> LLMProviderSettings:
    """Resolve provider, endpoint and role models from ``LLM_PROVIDER``."""

    env = os.environ if environ is None else environ
    requested = (env.get("LLM_PROVIDER") or DEFAULT_LLM_PROVIDER).strip().lower()
    provider = requested

    if provider == "minimax":
        model = env.get("MINIMAX_MODEL") or "MiniMax-M3"
        return LLMProviderSettings(
            provider="minimax",
            backend_url=(
                env.get("MINIMAX_BASE_URL") or "https://api.minimaxi.com/v1"
            ).rstrip("/"),
            deep_model=env.get("MINIMAX_DEEP_MODEL") or model,
            quick_model=env.get("MINIMAX_QUICK_MODEL") or model,
        )

    if provider == "deepseek":
        return LLMProviderSettings(
            provider="deepseek",
            backend_url=(
                env.get("DEEPSEEK_BASE_URL") or "https://api.deepseek.com"
            ).rstrip("/"),
            deep_model=env.get("DEEPSEEK_DEEP_MODEL") or "deepseek-v4-pro",
            quick_model=env.get("DEEPSEEK_QUICK_MODEL") or "deepseek-v4-flash",
        )

    raise ValueError(
        f"不支持的 LLM_PROVIDER={requested}；统一配置支持 minimax/deepseek"
    )


def apply_llm_provider_settings(
    config: MutableMapping[str, object],
    environ: Mapping[str, str] | None = None,
) -> dict[str, object]:
    """Return a config copy with all provider-dependent fields updated."""

    settings = resolve_llm_provider_settings(environ)
    updated = dict(config)
    updated.update(
        llm_provider=settings.provider,
        backend_url=settings.backend_url,
        deep_think_llm=settings.deep_model,
        quick_think_llm=settings.quick_model,
    )
    return updated
