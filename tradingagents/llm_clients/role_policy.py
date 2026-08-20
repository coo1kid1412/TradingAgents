"""Role-based DeepSeek model and reasoning-budget policy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class RoleLLMPolicy:
    model: str
    reasoning_effort: str
    max_tokens: int


_PRO_MAX = frozenset({"research_manager", "portfolio_manager"})
_PRO_HIGH = frozenset({"market", "fundamentals", "macro", "stock_profile"})

_ROLE_ALIASES = {
    "social": "sentiment",
    "profile": "stock_profile",
    "rm": "research_manager",
    "pm": "portfolio_manager",
    "bull": "bull_researcher",
    "bear": "bear_researcher",
}


def resolve_role_policy(config: Mapping[str, Any], role: str) -> RoleLLMPolicy:
    """Resolve one role without allowing implicit provider/model drift."""
    role_key = _ROLE_ALIASES.get((role or "").strip().lower(), (role or "").strip().lower())
    deep_model = str(config.get("deep_think_llm") or "deepseek-v4-pro")
    quick_model = str(config.get("quick_think_llm") or "deepseek-v4-flash")

    if role_key in _PRO_MAX:
        effort = "max"
        model = deep_model
        default_tokens = 16_384
    elif role_key in _PRO_HIGH:
        effort = "high"
        model = deep_model
        default_tokens = 12_288
    else:
        effort = "high"
        model = quick_model
        default_tokens = 8_192

    token_overrides = {
        "fundamentals": config.get("fundamentals_analyst_max_tokens"),
        "research_manager": config.get("research_manager_max_tokens"),
        "portfolio_manager": config.get("portfolio_manager_max_tokens"),
    }
    max_tokens = int(token_overrides.get(role_key) or default_tokens)
    custom = (config.get("role_llm_policies") or {}).get(role_key, {})
    if custom:
        model = str(custom.get("model") or model)
        effort = str(custom.get("reasoning_effort") or effort).lower()
        max_tokens = int(custom.get("max_tokens") or max_tokens)
    if effort not in {"high", "max"}:
        raise ValueError(f"角色 {role_key} 的 reasoning_effort 非法：{effort}")
    return RoleLLMPolicy(model=model, reasoning_effort=effort, max_tokens=max_tokens)
