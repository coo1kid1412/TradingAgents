from tradingagents.agents.utils.risk_consensus import build_risk_consensus


def test_consensus_uses_only_supported_caps_and_takes_the_strictest():
    state = {
        "aggressive_history": """```yaml
RISK_VIEW:
  role: liquidity
  severity: high
  cap_pct: 4
  cap_basis: 近60日日均成交额与ATR
  evidence_ids: [TECHNICAL-SUMMARY, STOCK-PROFILE-TRUTH]
  data_supported: true
```""",
        "conservative_history": """```yaml
RISK_VIEW:
  role: event
  severity: high
  cap_pct: 2
  cap_basis: 假设账户规模一亿元
  evidence_ids: []
  data_supported: false
```""",
        "neutral_history": """```yaml
RISK_VIEW:
  role: tail
  severity: medium
  cap_pct: 5
  cap_basis: RM悲观情景下行幅度
  evidence_ids: [RM-PLAN]
  data_supported: true
```""",
    }
    result = build_risk_consensus(
        state,
        {"entry_gate": "CONDITIONAL", "position_cap_pct": 6},
        allowed_evidence_ids={"TECHNICAL-SUMMARY", "STOCK-PROFILE-TRUTH", "RM-PLAN"},
    )
    assert result["effective_cap_pct"] == 4
    assert result["accepted_roles"] == ["liquidity", "tail"]
    assert result["rejected_roles"] == ["event"]


def test_missing_market_gate_remains_zero_even_with_risk_views():
    result = build_risk_consensus({}, {})
    assert result["effective_cap_pct"] == 0
    assert result["entry_gate"] == "WAIT"


def test_unknown_gate_is_normalized_before_cap_reconciliation():
    result = build_risk_consensus({}, {"entry_gate": "INVALID", "position_cap_pct": 8})
    assert result["entry_gate"] == "WAIT"
    assert result["market_cap_pct"] == 0
    assert result["effective_cap_pct"] == 0


def test_unrecognized_evidence_id_rejects_otherwise_supported_cap():
    state = {
        "neutral_history": """RISK_VIEW:
  role: tail
  severity: high
  cap_pct: 3
  cap_basis: 自称来自压力测试
  evidence_ids: [INVENTED-01]
  data_supported: true
""",
    }
    result = build_risk_consensus(
        state,
        {"entry_gate": "OPEN", "position_cap_pct": 10},
        allowed_evidence_ids={"RM-PLAN"},
    )
    assert result["effective_cap_pct"] == 10
    assert result["accepted_roles"] == []
    assert result["rejected_roles"] == ["tail"]


def test_event_cap_requires_at_least_one_verified_official_event_reference():
    state = {
        "conservative_history": """RISK_VIEW:
  role: event
  severity: high
  cap_pct: 2
  cap_basis: 仅引用 RM 方案推断事件窗口
  evidence_ids: [RM-PLAN]
  data_supported: true
""",
    }
    result = build_risk_consensus(
        state,
        {"entry_gate": "OPEN", "position_cap_pct": 8},
        allowed_evidence_ids={"RM-PLAN", "NEWS-CAT-01"},
        official_event_evidence_ids={"NEWS-CAT-01"},
    )
    assert result["effective_cap_pct"] == 8
    assert result["accepted_roles"] == []
    assert result["rejected_roles"] == ["event"]


def test_liquidity_cap_requires_market_or_liquidity_evidence_not_only_rm_plan():
    state = {
        "aggressive_history": """RISK_VIEW:
  role: liquidity
  severity: high
  cap_pct: 2
  cap_basis: 自称流动性不足
  evidence_ids: [RM-PLAN]
  data_supported: true
""",
    }
    result = build_risk_consensus(
        state,
        {"entry_gate": "OPEN", "position_cap_pct": 8},
        allowed_evidence_ids={"RM-PLAN", "TECHNICAL-SUMMARY"},
    )
    assert result["effective_cap_pct"] == 8
    assert result["rejected_roles"] == ["liquidity"]


def test_risk_agent_cannot_claim_another_role():
    state = {
        "aggressive_history": """RISK_VIEW:
  role: tail
  severity: high
  cap_pct: 2
  cap_basis: 角色错配
  evidence_ids: [RM-PLAN]
  data_supported: true
""",
    }
    result = build_risk_consensus(
        state,
        {"entry_gate": "OPEN", "position_cap_pct": 8},
        allowed_evidence_ids={"RM-PLAN"},
    )
    assert result["effective_cap_pct"] == 8
    assert result["rejected_roles"] == ["liquidity"]


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        test()
        print(f"  PASS {test.__name__}")
    print(f"\n{len(tests)}/{len(tests)} passed")
