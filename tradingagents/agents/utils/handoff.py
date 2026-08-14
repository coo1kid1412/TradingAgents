"""Role-specific handoff validation and bounded downstream context packing."""

from __future__ import annotations

import json
import re
from datetime import date, datetime
from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any

from tradingagents.agents.utils.yaml_summary import extract_yaml_mapping


ROLE_BUDGET_CHARS = {
    "market": 6_000,
    "fundamentals": 8_000,
    "news": 8_000,
    "sentiment": 4_000,
    "macro": 5_000,
    "stock_profile": 8_000,
    "consensus": 5_000,
    "bull": 6_000,
    "bear": 6_000,
    "aggressive_risk": 5_000,
    "neutral_risk": 5_000,
    "conservative_risk": 5_000,
    "research_manager": 12_000,
    "portfolio_manager": 10_000,
}

_ROLE_ALIASES = {
    "social_media": "sentiment",
    "social": "sentiment",
    "profile": "stock_profile",
    "rm": "research_manager",
    "pm": "portfolio_manager",
}

_FORBIDDEN_KEYS = {
    "market": {"target_price", "target_price_12m", "long_term_rating", "research_rating", "position_pct"},
    "fundamentals": {"entry_timing", "trade_action", "market_entry_gate", "position_pct", "short_term_trend"},
    "news": {"research_rating", "final_rating", "trade_action", "position_pct"},
    "sentiment": {"target_price", "revenue", "profit", "research_rating", "final_rating"},
    "macro": {"research_rating", "final_rating", "trade_action", "target_price"},
    "stock_profile": {"research_rating", "final_rating", "trade_action"},
    "consensus": {"research_rating", "final_rating", "trade_action"},
    "bull": {"research_rating", "position_pct", "trade_action"},
    "bear": {"research_rating", "position_pct", "trade_action"},
    "aggressive_risk": {"research_rating", "final_rating"},
    "neutral_risk": {"research_rating", "final_rating"},
    "conservative_risk": {"research_rating", "final_rating"},
}

_REQUIRED_FIELDS = {
    "schema_version", "role", "mandate", "as_of", "horizons", "source_periods",
    "specialist_view", "change", "facts", "interpretations", "counterpoints",
    "uncertainties", "requested_checks", "quality",
}

_PRIORITY = {
    "hard_constraint": 0,
    "evidence": 1,
    "handoff": 2,
    "decision": 2,
    "narrative": 3,
}


def _normalize_role(role: str) -> str:
    normalized = (role or "").strip().lower()
    return _ROLE_ALIASES.get(normalized, normalized)


def _strip_reasoning(text: str) -> str:
    clean = re.sub(r"(?is)<think\b[^>]*>.*?</think\s*>", "", text or "")
    clean = re.sub(r"(?im)^.*\breasoning_content\b.*(?:\n|$)", "", clean)
    clean = re.sub(r"(?im)^.*\bchain[_ -]?of[_ -]?thought\b.*(?:\n|$)", "", clean)
    return clean.strip()


def _sanitize(value: Any) -> Any:
    if isinstance(value, str):
        return _strip_reasoning(value)
    if isinstance(value, list):
        return [_sanitize(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _sanitize(item) for key, item in value.items()}
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    return value


def _all_keys(value: Any) -> set[str]:
    keys: set[str] = set()
    if isinstance(value, Mapping):
        for key, item in value.items():
            keys.add(str(key))
            keys.update(_all_keys(item))
    elif isinstance(value, list):
        for item in value:
            keys.update(_all_keys(item))
    return keys


def _truncate_strings(value: Any, limit: int = 2_000) -> Any:
    if isinstance(value, str):
        return value if len(value) <= limit else value[: limit - 1].rstrip() + "…"
    if isinstance(value, list):
        return [_truncate_strings(item, limit) for item in value]
    if isinstance(value, dict):
        return {key: _truncate_strings(item, limit) for key, item in value.items()}
    return value


def _invalid_handoff(role: str, reason: str) -> dict[str, Any]:
    return {
        "schema_version": "agent-handoff-v1",
        "role": role,
        "specialist_view": {
            "conclusion": "交接单不可用",
            "direction": "not_applicable",
            "conviction": "low",
            "materiality": "low",
        },
        "facts": [],
        "interpretations": [],
        "counterpoints": [],
        "uncertainties": [reason],
        "requested_checks": [],
        "quality": {"status": "invalid", "missing_fields": [], "conflicts": [reason]},
    }


def extract_handoff(report: str, role: str) -> dict[str, Any]:
    """Parse and validate an analyst HANDOFF without propagating malformed data."""
    expected_role = _normalize_role(role)
    handoff, status = extract_yaml_mapping(report or "", "HANDOFF")
    if handoff is None:
        return _invalid_handoff(expected_role, f"HANDOFF 无法解析（{status}）")
    result = _sanitize(deepcopy(handoff))
    actual_role = _normalize_role(str(result.get("role") or ""))
    missing = sorted(_REQUIRED_FIELDS - set(result))
    conflicts: list[str] = []
    if actual_role != expected_role:
        conflicts.append(f"角色不匹配：期望 {expected_role}，实际 {actual_role or 'missing'}")
    forbidden = sorted(_all_keys(result) & _FORBIDDEN_KEYS.get(expected_role, set()))
    if forbidden:
        conflicts.append(f"角色越权字段：{', '.join(forbidden)}")

    quality = result.get("quality")
    if not isinstance(quality, dict):
        quality = {"status": "invalid", "missing_fields": [], "conflicts": []}
        result["quality"] = quality
        conflicts.append("quality 字段无效")
    quality.setdefault("missing_fields", [])
    quality.setdefault("conflicts", [])
    quality["missing_fields"] = list(quality["missing_fields"] or []) + missing
    quality["conflicts"] = list(quality["conflicts"] or []) + conflicts

    serialized = json.dumps(result, ensure_ascii=False, separators=(",", ":"))
    over_budget = len(serialized) > ROLE_BUDGET_CHARS.get(expected_role, 6_000)
    if over_budget:
        result = _truncate_strings(result)
        quality = result["quality"]
        quality["missing_fields"] = list(quality.get("missing_fields") or [])
        quality["missing_fields"].append("超过角色字符预算")

    if conflicts or missing:
        quality["status"] = "invalid"
    elif over_budget or status != "valid" or quality.get("status") != "complete":
        quality["status"] = "partial"
    else:
        quality["status"] = "complete"
    result["role"] = expected_role
    return result


def pack_agent_context(
    items: Sequence[Mapping[str, Any]],
    *,
    budget_chars: int,
) -> str:
    """Pack context by information priority while enforcing a hard char bound."""
    if budget_chars <= 0:
        return ""
    normalized = []
    for index, item in enumerate(items):
        content = item.get("content", "")
        if not isinstance(content, str):
            content = json.dumps(content, ensure_ascii=False, separators=(",", ":"))
        content = _strip_reasoning(content)
        if not content:
            continue
        priority = str(item.get("priority") or "narrative")
        normalized.append((
            _PRIORITY.get(priority, _PRIORITY["narrative"]),
            index,
            str(item.get("label") or f"上下文 {index + 1}"),
            content,
        ))

    chunks: list[str] = []
    used = 0
    for _priority, _index, label, content in sorted(normalized):
        prefix = f"### {label}\n"
        remaining = budget_chars - used
        if remaining <= len(prefix):
            break
        allowed = remaining - len(prefix)
        body = content if len(content) <= allowed else content[: max(0, allowed - 1)].rstrip() + "…"
        chunk = prefix + body
        chunks.append(chunk)
        used += len(chunk) + (2 if len(chunks) > 1 else 0)
        if used >= budget_chars:
            break
    return "\n\n".join(chunks)[:budget_chars]


def analyst_handoff_contract(role: str, mandate: str, specialist_requirements: str) -> str:
    """Return the common analyst envelope plus a role-specific payload contract."""
    normalized_role = _normalize_role(role)
    return f"""

## 强制输出：角色交接单

完整报告与 `SUMMARY` 之后必须再输出一个 `HANDOFF` YAML；这是下游默认读取的限长交接，
不是思维链。不得包含 `<think>`、`reasoning_content` 或正式证据 ID。

本角色任务：{mandate}
专业载荷要求：{specialist_requirements}
角色字符预算：{ROLE_BUDGET_CHARS.get(normalized_role, 6000)}。

```yaml
HANDOFF:
  schema_version: agent-handoff-v1
  role: {normalized_role}
  mandate: "{mandate}"
  as_of: <分析截止时间>
  horizons: [<适用期限>]
  source_periods: [<行情/财务/事件期间>]
  specialist_view:
    conclusion: <一句话专业结论>
    direction: bullish / bearish / neutral / mixed / not_applicable
    conviction: high / medium / low
    materiality: high / medium / low
  change:
    basis: <前一交易日/前一报告期/前一事件/无可比基准>
    delta: <强化/弱化/反转/无变化/不可比较及原因>
  facts:
    - local_fact_key: <角色内临时键，不是正式证据 ID>
      metric_or_event: <事实名称>
      value: <数值或事件>
      period: <数据所属期间>
      observed_at: <观察时间或 null>
      published_at: <公开日期或 null>
      source_ref: <来源引用>
      source_tier: official / regulatory / mainstream / research / social / calculated
  interpretations:
    - inference: <事实到专业结论的解释，不写隐性推理过程>
      fact_keys: [<local_fact_key>]
      assumptions: [<必要假设>]
  counterpoints: [<反证>]
  uncertainties: [<无法确认事项>]
  requested_checks: [<下游需继续核查事项>]
  quality:
    status: complete / partial / invalid
    missing_fields: [<缺失项>]
    conflicts: [<内部冲突>]
```
"""


def decision_handoff_contract(role: str, mandate: str) -> str:
    """Return the no-new-facts contract for post-evidence decision roles."""
    normalized_role = _normalize_role(role)
    return f"""

输出末尾附 `DECISION_HANDOFF`。只能引用 IC 包中已有正式证据 ID，不得新增事实，
不得包含 `<think>` 或 `reasoning_content`。本角色任务：{mandate}

```yaml
DECISION_HANDOFF:
  role: {normalized_role}
  as_of: <分析截止时间>
  accepted_claim_ids: [<正式证据 ID>]
  challenged_claims:
    - claim_id: <正式证据 ID>
      challenge_type: stale / weak_source / causal_gap / conflicting / mispriced
      reason: <挑战理由>
  decision_judgments:
    - judgment: <本角色判断>
      claim_ids: [<正式证据 ID>]
      assumptions: [<必要假设>]
      affected_horizon: 3d / 3m / 12m / position
      decision_dimension: thesis / valuation / catalyst / expectation_gap / timing / crowding / macro / risk
      allowed_effect: foundation / support / oppose / cap / floor / timing_only / conviction_only / veto_request
      direction: bullish / bearish / neutral / mixed
      materiality: high / medium / low
      confidence: high / medium / low
  unresolved_dissent: [<未解决分歧>]
  conditions_to_revisit: [<重新评估条件>]
  quality_status: complete / partial / invalid
```
"""


def pack_report_handoffs(
    reports: Mapping[str, str],
    *,
    budget_chars: int,
    extra_items: Sequence[Mapping[str, Any]] = (),
) -> str:
    """Pack analyst handoffs, falling back to bounded SUMMARY mappings."""
    items: list[dict[str, Any]] = [dict(item) for item in extra_items]
    for role, report in reports.items():
        handoff = extract_handoff(report or "", role)
        quality = str((handoff.get("quality") or {}).get("status") or "invalid")
        payload: Any = handoff
        label = f"{_normalize_role(role)} HANDOFF"
        if quality == "invalid":
            summary, _status = extract_yaml_mapping(report or "", "SUMMARY")
            if summary is not None:
                payload = {
                    "role": _normalize_role(role),
                    "fallback": "SUMMARY",
                    "quality_status": "partial",
                    "summary": _sanitize(summary),
                }
                label = f"{_normalize_role(role)} SUMMARY 降级交接"
        items.append({
            "label": label,
            "content": payload,
            "priority": "handoff",
        })
    return pack_agent_context(items, budget_chars=budget_chars)
