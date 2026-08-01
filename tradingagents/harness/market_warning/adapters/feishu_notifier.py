"""Idempotent Feishu delivery for persisted market-warning decisions."""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from datetime import datetime
from zoneinfo import ZoneInfo

from tradingagents.harness.market_risk_daily import _send_feishu_message

from ..domain import Market, RiskLevel, RunnerResult
from ..reporting import render_premarket_report, render_upgrade_report


_MARKET_ZONES = {
    Market.A_SHARE: ZoneInfo("Asia/Shanghai"),
    Market.US: ZoneInfo("America/New_York"),
}


def should_notify(result: RunnerResult) -> bool:
    decision = result.decision
    if decision is None:
        return False
    if "premarket" in result.session_slot.lower():
        return True
    return decision.push_required and decision.final_level in {
        RiskLevel.ORANGE,
        RiskLevel.RED,
    }


def _idempotency_key(result: RunnerResult) -> str:
    decision = result.decision
    if decision is None or result.quant_assessment is None:
        raise ValueError("notification requires decision and quant assessment")
    local = result.as_of_time.astimezone(_MARKET_ZONES[result.market])
    slot = (
        "premarket"
        if "premarket" in result.session_slot.lower()
        else f"bucket-{local:%H%M}"
    )
    parts = (
        result.market.value,
        local.date().isoformat(),
        slot,
        decision.final_level.value,
        decision.state_transition,
        result.quant_assessment.model_version,
    )
    return "|".join(parts)


class FeishuNotifier:
    """Claim one alert row before using the project's existing Feishu transport."""

    def __init__(
        self,
        repository,
        *,
        sender: Callable[[str], None] = _send_feishu_message,
        retry_failed: bool = False,
    ) -> None:
        self.repository = repository
        self.sender = sender
        self.retry_failed = retry_failed

    def notify(self, result: RunnerResult) -> bool:
        if not should_notify(result) or result.decision_id is None:
            return False
        message = (
            render_premarket_report(result, None)
            if "premarket" in result.session_slot.lower()
            else render_upgrade_report(result, None)
        )
        payload_hash = hashlib.sha256(message.encode("utf-8")).hexdigest()
        key = _idempotency_key(result)
        if not self.repository.claim_alert(
            key,
            result.decision_id,
            payload_hash,
            retry_failed=self.retry_failed,
        ):
            return False
        try:
            self.sender(message)
        except Exception:
            self.repository.finish_alert(key, "failed", "send_error")
            raise
        self.repository.finish_alert(key, "sent")
        return True
