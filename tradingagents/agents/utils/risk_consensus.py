"""Parse data-backed risk views and reconcile one executable cap."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

import yaml


_ROLE_EVIDENCE_PREFIXES = {
    "liquidity": ("TECHNICAL-SUMMARY", "STOCK-PROFILE-TRUTH", "MKT-LEVEL-", "MKT-FLOW-", "MKT-STRUCT-", "RISK-GATE-", "MARKET-RISK-SNAPSHOT"),
    "event": ("NEWS-CAT-",),
    "tail": ("RM-PLAN", "FUND-", "QUANT-", "MKT-", "RISK-GATE-", "MARKET-RISK-SNAPSHOT"),
}


def _last_view(content: str) -> dict[str, Any] | None:
    blocks = re.findall(r"```yaml\s*\n(.*?)\n```", content or "", re.S)
    blocks.append(content or "")
    for block in reversed(blocks):
        try:
            parsed = yaml.safe_load(block)
        except yaml.YAMLError:
            continue
        view = parsed.get("RISK_VIEW") if isinstance(parsed, dict) else None
        if isinstance(view, dict):
            return view
    return None


def build_risk_consensus(
    risk_debate_state: Mapping[str, Any],
    market_risk_snapshot: Mapping[str, Any],
    *,
    allowed_evidence_ids: set[str] | None = None,
    official_event_evidence_ids: set[str] | None = None,
) -> dict[str, Any]:
    gate = str(market_risk_snapshot.get("entry_gate") or "WAIT").upper()
    if gate not in {"OPEN", "CONDITIONAL", "WAIT"}:
        gate = "WAIT"
    try:
        market_cap = float(market_risk_snapshot.get("position_cap_pct", 0))
    except (TypeError, ValueError):
        market_cap = 0.0
    if gate == "WAIT":
        market_cap = 0.0

    accepted: list[dict[str, Any]] = []
    rejected_roles: list[str] = []
    for key, fallback_role in (
        ("aggressive_history", "liquidity"),
        ("conservative_history", "event"),
        ("neutral_history", "tail"),
    ):
        view = _last_view(str(risk_debate_state.get(key) or ""))
        if not view:
            continue
        role = str(view.get("role") or fallback_role)
        try:
            cap = float(view.get("cap_pct"))
        except (TypeError, ValueError):
            cap = -1
        supported = view.get("data_supported") is True
        basis = str(view.get("cap_basis") or "").strip()
        evidence_ids = view.get("evidence_ids")
        if isinstance(evidence_ids, str):
            evidence_ids = [item.strip() for item in evidence_ids.split(",") if item.strip()]
        evidence_ids = [str(item) for item in (evidence_ids or []) if item not in (None, "")]
        references_allowed = bool(evidence_ids) and (
            allowed_evidence_ids is None
            or set(evidence_ids).issubset(allowed_evidence_ids)
        )
        event_reference_valid = (
            role != "event"
            or official_event_evidence_ids is None
            or bool(set(evidence_ids) & official_event_evidence_ids)
        )
        role_reference_valid = any(
            evidence_id.startswith(_ROLE_EVIDENCE_PREFIXES[fallback_role])
            for evidence_id in evidence_ids
        )
        if (
            role != fallback_role
            or not supported
            or not basis
            or not references_allowed
            or not event_reference_valid
            or not role_reference_valid
            or not 0 <= cap <= 100
        ):
            rejected_roles.append(fallback_role)
            continue
        accepted.append({
            "role": role,
            "cap_pct": cap,
            "cap_basis": basis,
            "evidence_ids": evidence_ids,
        })

    effective_cap = min([market_cap, *[view["cap_pct"] for view in accepted]])
    return {
        "entry_gate": gate,
        "market_cap_pct": market_cap,
        "effective_cap_pct": round(max(0.0, effective_cap), 2),
        "accepted_roles": [view["role"] for view in accepted],
        "rejected_roles": rejected_roles,
        "accepted_views": accepted,
    }
