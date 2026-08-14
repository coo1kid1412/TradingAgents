"""Build a compact, traceable data packet for risk debate agents."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

import yaml

from tradingagents.agents.utils.report_calendar import has_traceable_official_reference


_TECHNICAL_FIELDS = (
    "price_data_date",
    "price_data_time",
    "price_data_status",
    "price_data_source",
    "trend_weekly",
    "trend_daily",
    "momentum",
    "key_support",
    "key_resistance",
    "atr_pct",
    "volume_state",
    "volume_price_pattern",
    "capital_flow_regime",
    "capital_flow_score",
)

_RISK_FIELDS = (
    "as_of_date",
    "as_of_time",
    "market",
    "risk_level",
    "entry_gate",
    "position_cap_pct",
    "data_status",
    "t_plus_1_bias",
    "required_checkpoint",
)

_PROFILE_PREFIXES = (
    "| liquidity |",
    "SYS_SHORT_TERM_STRUCTURE:",
    "SYS_SHORT_TERM_STRUCTURE_REASONS:",
    "SYS_SHORT_TERM_STRUCTURE_BLOCKERS:",
    "SYS_ENTRY_RECURRING_LOSS:",
    "SYS_ENTRY_HAS_PEAK_SIGNAL:",
    "SYS_ENTRY_RETAIL_CONCENTRATION:",
    "SYS_ENTRY_RSI_PERCENTILE_1Y:",
    "SYS_ENTRY_CAPITAL_FLOW_REGIME:",
    "SYS_ENTRY_MAIN_FORCE_STREAK_DAYS:",
)


def _extract_summary(report: str) -> dict[str, Any]:
    for block in reversed(re.findall(r"```yaml\s*\n(.*?)\n```", report or "", re.DOTALL)):
        if not re.search(r"(?m)^\s*SUMMARY\s*:", block):
            continue
        try:
            parsed = yaml.safe_load(block)
        except yaml.YAMLError:
            continue
        summary = parsed.get("SUMMARY") if isinstance(parsed, dict) else None
        if isinstance(summary, dict):
            return summary
    return {}


def _compact_evidence(ledger: Mapping[str, Any]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for card in ledger.get("cards") or []:
        if not isinstance(card, Mapping):
            continue
        if card.get("decision_eligible") is not True:
            continue
        if str(card.get("quality_status") or "") == "invalid":
            continue
        result.append({
            key: card.get(key)
            for key in (
                "claim_id", "owner", "decision_variable", "claim", "direction",
                "horizon", "as_of", "quality_status", "source_name", "document_id",
                "source_tier", "source_url", "verification_status", "date_basis", "event_date", "falsifier",
            )
            if card.get(key) not in (None, "")
        })
        if len(result) >= 16:
            break
    return result


def build_risk_data_packet(state: Mapping[str, Any]) -> dict[str, Any]:
    """Return the minimum shared facts needed by all three risk agents."""
    technical_summary = _extract_summary(str(state.get("market_report") or ""))
    technical = {
        key: technical_summary.get(key)
        for key in _TECHNICAL_FIELDS
        if technical_summary.get(key) not in (None, "")
    }
    profile_truth = [
        line.strip()
        for line in str(state.get("stock_profile") or "").splitlines()
        if line.strip().startswith(_PROFILE_PREFIXES)
    ][:16]
    snapshot = state.get("market_risk_snapshot")
    snapshot = snapshot if isinstance(snapshot, Mapping) else {}
    market_risk = {
        key: snapshot.get(key)
        for key in _RISK_FIELDS
        if snapshot.get(key) not in (None, "")
    }
    ledger = state.get("research_evidence_ledger")
    ledger = ledger if isinstance(ledger, Mapping) else {}
    evidence = _compact_evidence(ledger)
    reference_ids: list[str] = []
    if technical:
        reference_ids.append("TECHNICAL-SUMMARY")
    if profile_truth:
        reference_ids.append("STOCK-PROFILE-TRUTH")
    if market_risk:
        reference_ids.append("MARKET-RISK-SNAPSHOT")
    reference_ids.extend(
        str(item["claim_id"])
        for item in evidence
        if item.get("claim_id") not in (None, "")
    )
    official_event_evidence_ids = [
        str(item["claim_id"])
        for item in evidence
        if item.get("claim_id") not in (None, "")
        and has_traceable_official_reference(
            source_tier=item.get("source_tier"),
            verification_status=item.get("verification_status"),
            source_url=item.get("source_url"),
            document_id=item.get("document_id"),
        )
    ]
    return {
        "technical": technical,
        "profile_truth": profile_truth,
        "market_risk": market_risk,
        "evidence_status": ledger.get("analysis_status") or "missing",
        "evidence_warnings": list(ledger.get("warnings") or [])[:8],
        "eligible_evidence": evidence,
        "reference_ids": reference_ids,
        "official_event_evidence_ids": official_event_evidence_ids,
    }
