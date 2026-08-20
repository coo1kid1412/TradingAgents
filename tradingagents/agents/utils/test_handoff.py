from __future__ import annotations

import sys
from pathlib import Path

from tradingagents.agents.utils import handoff
from tradingagents.agents.utils.decision_response import invoke_decision_response
from tradingagents.agents.utils.handoff import (
    extract_handoff,
    pack_agent_context,
    pack_report_handoffs,
)
from tradingagents.agents.utils.yaml_summary import extract_yaml_mapping


ROOT = Path(__file__).resolve().parents[3]


class _Reply:
    def __init__(self, content: str):
        self.content = content


class _FakeLLM:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def invoke(self, prompt):
        self.calls.append(prompt)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return _Reply(reply)


def _decision_tail(role: str = "bear") -> str:
    return f"""```yaml
DECISION_HANDOFF:
  role: {role}
  as_of: 2026-08-14
  accepted_claim_ids: [FUND-GROWTH-01]
  challenged_claims: []
  decision_judgments:
    - judgment: 增长确认但估值约束仍在
      claim_ids: [FUND-GROWTH-01]
      assumptions: []
      affected_horizon: 12m
      decision_dimension: thesis
      allowed_effect: support
      direction: bullish
      materiality: high
      confidence: medium
  unresolved_dissent: []
  conditions_to_revisit: [下一期业绩]
  quality_status: complete
```"""


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


def test_final_report_artifact_removes_tool_status_preamble():
    content = """财联社工具三次超时，我将改用其他来源并开始撰写报告。

---

# 新易盛新闻研究报告

## 数据覆盖

财联社缺失，已由其他来源部分补足。
"""

    result = handoff.extract_final_report_artifact(content)

    assert result.startswith("# 新易盛新闻研究报告")
    assert "我将改用其他来源" not in result
    assert "财联社缺失" in result


def test_final_report_artifact_preserves_a_report_that_starts_with_h2():
    content = """## 核心事件

这是合法正文，不能因后面出现一级标题而删除。

# 风险清单

风险正文。
"""

    result = handoff.extract_final_report_artifact(content)

    assert result.startswith("## 核心事件")
    assert "这是合法正文" in result


def test_complete_decision_handoff_does_not_retry():
    llm = _FakeLLM([f"空头正文\n\n{_decision_tail()}"])

    response = invoke_decision_response(
        llm,
        "原始提示，允许引用 FUND-GROWTH-01",
        role="bear",
    )

    assert len(llm.calls) == 1
    handoff_data, status = extract_yaml_mapping(response.content, "DECISION_HANDOFF")
    assert status == "valid"
    assert handoff_data["role"] == "bear"


def test_missing_decision_handoff_retries_only_bounded_structured_tail():
    llm = _FakeLLM(["空头正文" + "很长" * 10_000, _decision_tail()])
    original_prompt = "IC 决策包 FUND-GROWTH-01 与 NEWS-CAT-02" + "原始材料" * 10_000

    response = invoke_decision_response(llm, original_prompt, role="bear")

    assert len(llm.calls) == 2
    retry_messages = llm.calls[1]
    retry_chars = sum(len(str(message.content)) for message in retry_messages)
    assert retry_chars <= 14_000
    assert "FUND-GROWTH-01" in str(retry_messages)
    assert "NEWS-CAT-02" in str(retry_messages)
    assert "DECISION_HANDOFF:" in response.content
    assert "空头正文" not in response.content


def test_missing_decision_handoff_fails_closed_after_one_retry():
    llm = _FakeLLM(["第一轮没有交接", RuntimeError("provider timeout")])

    response = invoke_decision_response(llm, "原始提示", role="bull")

    handoff_data, status = extract_yaml_mapping(response.content, "DECISION_HANDOFF")
    assert len(llm.calls) == 2
    assert status == "valid"
    assert handoff_data["role"] == "bull"
    assert handoff_data["quality_status"] == "invalid"
    assert handoff_data["accepted_claim_ids"] == []
    assert "第一轮没有交接" not in response.content


def test_failed_decision_handoff_repair_does_not_infer_direction_from_mentions():
    llm = _FakeLLM([
        "空方判断：FUND-GROWTH-01 与 NEWS-CAT-02 支撑谨慎结论。",
        RuntimeError("provider timeout"),
    ])

    response = invoke_decision_response(
        llm,
        "允许引用 FUND-GROWTH-01、NEWS-CAT-02",
        role="bear",
    )

    handoff_data, status = extract_yaml_mapping(response.content, "DECISION_HANDOFF")
    assert status == "valid"
    assert handoff_data["quality_status"] == "invalid"
    assert handoff_data["accepted_claim_ids"] == []
    assert handoff_data["decision_judgments"] == []
    assert "空方判断" not in response.content
    assert response.content.count("DECISION_HANDOFF:") == 1


def test_explicitly_invalid_decision_handoff_is_repaired_before_use():
    invalid = """空头正文

```yaml
DECISION_HANDOFF:
  role: bear
  as_of: 2026-08-14
  accepted_claim_ids: []
  challenged_claims: []
  decision_judgments: []
  unresolved_dissent: [缺少证据]
  conditions_to_revisit: []
  quality_status: invalid
```"""
    llm = _FakeLLM([invalid, _decision_tail()])

    response = invoke_decision_response(
        llm,
        "允许引用 FUND-GROWTH-01",
        role="bear",
    )

    handoff_data, status = extract_yaml_mapping(response.content, "DECISION_HANDOFF")
    assert len(llm.calls) == 2
    assert status == "valid"
    assert handoff_data["quality_status"] == "complete"
    assert handoff_data["decision_judgments"]
    assert "空头正文" not in response.content
    assert response.content.count("DECISION_HANDOFF:") == 1


def test_bull_and_bear_nodes_use_validated_decision_responses():
    for relative in (
        "tradingagents/agents/researchers/bull_researcher.py",
        "tradingagents/agents/researchers/bear_researcher.py",
    ):
        source = (ROOT / relative).read_text(encoding="utf-8")
        assert "invoke_decision_response" in source, relative
        assert "response = llm.invoke(prompt)" not in source, relative


def test_partial_handoff_is_bounded_by_role_budget():
    report = _handoff().replace("短线结构偏强", "结构" * 5000)
    result = extract_handoff(report, "market")
    assert len(result["specialist_view"]["conclusion"]) < 6000
    assert result["quality"]["status"] == "partial"
    assert "超过角色字符预算" in result["quality"]["missing_fields"]


def test_invalid_handoff_does_not_fall_back_to_unvalidated_summary():
    report = """# report

```yaml
SUMMARY:
  research_rating: BUY
  target_price: 999
```
"""

    packed = pack_report_handoffs({"market": report}, budget_chars=2000)

    assert "quality" in packed
    assert "invalid" in packed
    assert "research_rating" not in packed
    assert "target_price" not in packed
    assert "SUMMARY 降级交接" not in packed


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
