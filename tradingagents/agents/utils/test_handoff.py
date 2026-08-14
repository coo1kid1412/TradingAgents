from __future__ import annotations

import sys
from pathlib import Path

from tradingagents.agents.utils.handoff import extract_handoff, pack_agent_context


ROOT = Path(__file__).resolve().parents[3]


def _handoff(role: str = "market", extra: str = "") -> str:
    return f"""# report

```yaml
HANDOFF:
  schema_version: agent-handoff-v1
  role: {role}
  mandate: 回答本角色问题
  as_of: 2026-08-14 14:30
  horizons: [3d]
  source_periods: [2026-08-14]
  specialist_view:
    conclusion: 短线结构偏强
    direction: bullish
    conviction: medium
    materiality: high
  change:
    basis: 前一交易日
    delta: 强化
  facts:
    - local_fact_key: price-01
      metric_or_event: 收盘价
      value: 100
      period: 2026-08-14
      observed_at: 2026-08-14 14:30
      published_at: null
      source_ref: tushare
      source_tier: calculated
  interpretations: []
  counterpoints: []
  uncertainties: []
  requested_checks: []
  quality:
    status: complete
    missing_fields: []
    conflicts: []
{extra}
```"""


def test_valid_market_handoff_is_parsed():
    result = extract_handoff(_handoff(), "market")
    assert result["quality"]["status"] == "complete"
    assert result["specialist_view"]["direction"] == "bullish"


def test_market_handoff_rejects_long_term_target_price():
    result = extract_handoff(_handoff(extra="  target_price_12m: 150"), "market")
    assert result["quality"]["status"] == "invalid"
    assert "target_price_12m" in result["quality"]["conflicts"][0]


def test_role_mismatch_and_missing_required_fields_fail_closed():
    mismatch = extract_handoff(_handoff(role="fundamentals"), "market")
    missing = extract_handoff("HANDOFF:\n  role: market", "market")
    assert mismatch["quality"]["status"] == "invalid"
    assert missing["quality"]["status"] == "invalid"


def test_context_packer_keeps_hard_constraints_before_narrative():
    packed = pack_agent_context(
        [
            {"label": "长篇叙述", "content": "叙事" * 2000, "priority": "narrative"},
            {"label": "风险门", "content": "RISK-GATE-01: WAIT", "priority": "hard_constraint"},
            {"label": "证据", "content": "FUND-GROWTH-01: 增长确认", "priority": "evidence"},
        ],
        budget_chars=1200,
    )
    assert len(packed) <= 1200
    assert "RISK-GATE-01" in packed
    assert "FUND-GROWTH-01" in packed
    assert packed.index("RISK-GATE-01") < packed.index("叙事")


def test_context_packer_removes_reasoning_leakage():
    packed = pack_agent_context(
        [{
            "label": "输入",
            "priority": "evidence",
            "content": "结论保留\nreasoning_content: secret\n<think>hidden chain</think>\n证据保留",
        }],
        budget_chars=500,
    )
    assert "结论保留" in packed
    assert "证据保留" in packed
    assert "reasoning_content" not in packed
    assert "hidden chain" not in packed
    assert "<think>" not in packed


def test_partial_handoff_is_bounded_by_role_budget():
    report = _handoff().replace("短线结构偏强", "结构" * 5000)
    result = extract_handoff(report, "market")
    assert len(result["specialist_view"]["conclusion"]) < 6000
    assert result["quality"]["status"] == "partial"
    assert "超过角色字符预算" in result["quality"]["missing_fields"]


def test_active_analysts_emit_role_specific_handoff_contracts():
    for relative in (
        "tradingagents/agents/analysts/market_analyst.py",
        "tradingagents/agents/analysts/fundamentals_analyst.py",
        "tradingagents/agents/analysts/news_analyst.py",
        "tradingagents/agents/analysts/social_media_analyst.py",
        "tradingagents/agents/utils/macro_context_node.py",
        "tradingagents/agents/utils/stock_profile_node.py",
        "tradingagents/agents/utils/consensus_node.py",
    ):
        source = (ROOT / relative).read_text(encoding="utf-8")
        assert "analyst_handoff_contract" in source, relative


def test_downstream_research_and_risk_use_bounded_contexts():
    for relative in (
        "tradingagents/agents/researchers/bull_researcher.py",
        "tradingagents/agents/researchers/bear_researcher.py",
        "tradingagents/agents/managers/research_manager.py",
    ):
        source = (ROOT / relative).read_text(encoding="utf-8")
        assert "pack_report_handoffs" in source, relative
        assert "Company fundamentals report: {fundamentals_report}" not in source, relative
        if "researchers" in relative:
            assert "decision_handoff_contract" in source, relative

    for relative in (
        "tradingagents/agents/risk_mgmt/aggressive_debator.py",
        "tradingagents/agents/risk_mgmt/neutral_debator.py",
        "tradingagents/agents/risk_mgmt/conservative_debator.py",
    ):
        source = (ROOT / relative).read_text(encoding="utf-8")
        assert "pack_agent_context" in source, relative
        assert "decision_handoff_contract" in source, relative
        assert '"history": history + "\\n" + argument' not in source, relative


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for test in tests:
        try:
            test()
            print(f"  PASS {test.__name__}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"  FAIL {test.__name__}: [{type(exc).__name__}] {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
