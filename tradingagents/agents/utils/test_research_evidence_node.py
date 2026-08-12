from __future__ import annotations

import sys

from tradingagents.agents.utils.research_evidence_node import (
    compile_research_evidence,
    create_research_evidence_node,
    render_decision_attribution,
    render_ic_packet,
)


def _report(key: str, body: str) -> str:
    return f"# report\n\n```yaml\n{key}:\n{body}\n```\n"


def _complete_state() -> dict:
    return {
        "company_of_interest": "688114",
        "company_name": "华大智造",
        "trade_date": "2026-08-12",
        "market_report": _report(
            "SUMMARY",
            """  trend_weekly: 震荡
  trend_daily: 上行
  momentum: 强
  key_support: 82.5
  key_resistance: 91.2
  price_data_status: official_daily
  price_data_date: 2026-08-12
  price_data_source: tushare_daily
  capital_flow_regime: 强势
  data_implied_direction: 偏多
  data_implied_reasoning: 日线动量与资金共振
  confidence: 4""",
        ),
        "fundamentals_report": _report(
            "SUMMARY",
            """  pe_ttm: 42.1
  pe_zone: 合理
  pe_industry_median: 45.0
  growth_yoy_revenue: 31.2
  growth_yoy_profit: 47.8
  roe: 12.4
  earnings_quality: 高
  governance_score: 高
  red_flags: []
  rating: 正面
  data_implied_direction: 偏多
  data_implied_reasoning: 盈利增长快于收入""",
        ),
        "news_report": _report(
            "SUMMARY",
            """  net_sentiment: 正面
  key_events:
    - title: 新平台获批
      category: 公司
      event_date: 2026Q3
      source_date: 2026-08-11
      horizon: 中期(1-3月)
      priced_in_p: 40
      impact: +中
      credibility: 高
      thesis_relevance: 核心
    - title: 新平台获批
      category: 公司
      event_date: 2026Q3
      source_date: 2026-08-11
      horizon: 中期(1-3月)
      priced_in_p: 40
      impact: +中
      credibility: 高
      thesis_relevance: 核心
  research_consensus_rating: BUY
  research_consensus_target_price: 96.0
  data_implied_direction: 偏多
  data_implied_reasoning: 催化尚未充分定价""",
        ),
        "sentiment_report": _report(
            "SUMMARY",
            """  net_sentiment: 分歧
  bull_post_pct: 45
  bear_post_pct: 35
  neutral_post_pct: 20
  kol_consensus: 分歧
  sentiment_trend_7d: 12
  rating: HOLD
  data_implied_direction: 中性
  data_implied_reasoning: 多空观点分歧""",
        ),
        "quant_score": _report(
            "QUANT_SCORE",
            """  price_data_status: official_daily
  price_data_date: 2026-08-12
  composite: 72.5
  factor_scores:
    momentum: 78
    value: 48
    quality: 74
    growth: 81""",
        ),
        "sector_comparison": "- vs 沪深300（30d）：✓ 跑赢大盘（RS = +12.4%）",
        "stock_profile": (
            "SYS_SHORT_TERM_STRUCTURE: class=healthy_trend | ma10_slope_5d_pct=1.2 "
            "| price_vs_ma10_pct=2.1 | breakout_confirmed=false"
        ),
        "market_risk_snapshot": {
            "as_of_date": "2026-08-12",
            "risk_level": "中",
            "entry_gate": "CONDITIONAL",
            "position_cap_pct": 6,
            "data_status": "fresh",
        },
    }


def test_compiler_builds_stable_cards_and_deduplicates_events():
    ledger = compile_research_evidence(_complete_state())
    by_id = {card["claim_id"]: card for card in ledger["cards"]}

    assert by_id["MKT-TREND-01"]["decision_variable"] == "short_term_trend"
    assert by_id["MKT-TREND-01"]["direction"] == "bullish"
    assert by_id["MKT-STRUCT-01"]["decision_variable"] == "entry_timing"
    assert by_id["MKT-STRUCT-01"]["direction"] == "bullish"
    assert by_id["FUND-GROWTH-01"]["decision_variable"] == "earnings_outlook_12m"
    assert by_id["FUND-VAL-01"]["decision_variable"] == "target_price"
    assert by_id["NEWS-CAT-01"]["direction"] == "bullish"
    assert by_id["QUANT-COMP-01"]["direction"] == "bullish"
    assert by_id["SECTOR-RS-01"]["direction"] == "bullish"
    assert by_id["RISK-GATE-01"]["decision_variable"] == "position"
    assert [card["claim_id"] for card in ledger["cards"]].count("NEWS-CAT-01") == 1
    assert "NEWS-CAT-02" not in by_id


def test_unknown_news_date_and_t_minus_one_price_are_partial():
    state = _complete_state()
    state["market_report"] = state["market_report"].replace(
        "official_daily\n  price_data_date: 2026-08-12",
        "t_minus_1\n  price_data_date: 2026-08-11",
    )
    state["news_report"] = state["news_report"].replace(
        "source_date: 2026-08-11", "source_date: 未知"
    )

    ledger = compile_research_evidence(state)
    by_id = {card["claim_id"]: card for card in ledger["cards"]}

    assert by_id["MKT-TREND-01"]["quality_status"] == "partial"
    assert by_id["NEWS-CAT-01"]["quality_status"] == "partial"


def test_future_dated_evidence_is_invalid_to_prevent_lookahead():
    state = _complete_state()
    state["market_report"] = state["market_report"].replace(
        "price_data_date: 2026-08-12", "price_data_date: 2026-08-13"
    )
    state["news_report"] = state["news_report"].replace(
        "source_date: 2026-08-11", "source_date: 2026-08-13"
    )

    ledger = compile_research_evidence(state)
    by_id = {card["claim_id"]: card for card in ledger["cards"]}

    assert by_id["MKT-TREND-01"]["quality_status"] == "invalid"
    assert by_id["NEWS-CAT-01"]["quality_status"] == "invalid"


def test_missing_market_risk_still_emits_fail_closed_position_evidence():
    state = _complete_state()
    state["market_risk_snapshot"] = {}

    ledger = compile_research_evidence(state)
    by_id = {card["claim_id"]: card for card in ledger["cards"]}

    assert by_id["RISK-GATE-01"]["decision_variable"] == "position"
    assert by_id["RISK-GATE-01"]["quality_status"] == "partial"
    assert "WAIT" in by_id["RISK-GATE-01"]["claim"]
    assert ledger["coverage"]["risk"] == "partial"


def test_valuation_zone_maps_low_to_bullish_and_high_to_bearish():
    low = compile_research_evidence(_complete_state())
    low_card = next(card for card in low["cards"] if card["claim_id"] == "FUND-VAL-01")
    assert low_card["direction"] == "neutral"

    state = _complete_state()
    state["fundamentals_report"] = state["fundamentals_report"].replace(
        "pe_zone: 合理", "pe_zone: 低估"
    )
    cheap = compile_research_evidence(state)
    cheap_card = next(card for card in cheap["cards"] if card["claim_id"] == "FUND-VAL-01")
    assert cheap_card["direction"] == "bullish"

    state["fundamentals_report"] = state["fundamentals_report"].replace(
        "pe_zone: 低估", "pe_zone: 高估"
    )
    expensive = compile_research_evidence(state)
    expensive_card = next(card for card in expensive["cards"] if card["claim_id"] == "FUND-VAL-01")
    assert expensive_card["direction"] == "bearish"


def test_invalid_yaml_degrades_one_domain_without_aborting():
    state = _complete_state()
    state["sentiment_report"] = "```yaml\nSUMMARY:\n  broken: [\n```"

    ledger = compile_research_evidence(state)

    assert ledger["coverage"]["sentiment"] == "invalid"
    assert any("舆情" in warning for warning in ledger["warnings"])
    assert any(card["owner"] == "market" for card in ledger["cards"])


def test_type_invalid_summary_degrades_one_domain_without_aborting():
    state = _complete_state()
    state["news_report"] = _report("SUMMARY", "  key_events: 3")

    ledger = compile_research_evidence(state)

    assert ledger["coverage"]["news"] == "invalid"
    assert any("新闻摘要字段类型无效" in warning for warning in ledger["warnings"])
    assert any(card["owner"] == "market" for card in ledger["cards"])


def test_node_fails_open_when_compilation_raises(monkeypatch=None):
    import tradingagents.agents.utils.research_evidence_node as module

    original = module.compile_research_evidence
    module.compile_research_evidence = lambda _state: (_ for _ in ()).throw(RuntimeError("boom"))
    try:
        result = create_research_evidence_node()({"company_of_interest": "688114"})
    finally:
        module.compile_research_evidence = original

    assert result["research_evidence_ledger"]["cards"] == []
    assert "已降级为空账本" in result["ic_packet"]


def test_opposite_long_term_directions_create_conflict_without_resolving_it():
    state = _complete_state()
    state["news_report"] = state["news_report"].replace(
        "data_implied_direction: 偏多", "data_implied_direction: 偏空"
    )
    state["news_report"] = state["news_report"].replace(
        "impact: +中", "impact: -中"
    )

    ledger = compile_research_evidence(state)
    conflicts = [
        item for item in ledger["conflicts"]
        if item["decision_variable"] == "long_term_rating"
    ]

    assert conflicts
    assert set(conflicts[0]["directions"]) == {"bullish", "bearish"}
    assert "FUND-GROWTH-01" in conflicts[0]["evidence_ids"]
    assert "NEWS-CAT-01" in conflicts[0]["evidence_ids"]


def test_ic_packet_lists_quality_conflicts_and_evidence_index():
    state = _complete_state()
    state["news_report"] = state["news_report"].replace(
        "data_implied_direction: 偏多", "data_implied_direction: 偏空"
    ).replace("impact: +中", "impact: -中")
    ledger = compile_research_evidence(state)

    packet = render_ic_packet(
        ledger,
        ticker="688114",
        company_name="华大智造",
        trade_date="2026-08-12",
    )

    assert packet.startswith("# IC 决策包：688114 华大智造")
    assert "## 数据完整度" in packet
    assert "## 未来三日与入场" in packet
    assert "## 一年期盈利与估值" in packet
    assert "## 冲突清单" in packet
    assert "MKT-TREND-01" in packet
    assert "FUND-GROWTH-01" in packet
    assert "模型不得编造证据 ID" in packet


def _pm_summary(**overrides) -> str:
    values = {
        "pm_rating": "OVERWEIGHT",
        "pm_size_low_pct": "2",
        "pm_size_high_pct": "3",
        "pm_tp1": "96",
        "pm_tp2": "104",
        "pm_tp3": "112",
        "short_term_evidence_ids": "MKT-TREND-01|RISK-GATE-01",
        "long_term_evidence_ids": "FUND-GROWTH-01|NEWS-CAT-01",
        "position_evidence_ids": "RISK-GATE-01",
        "target_price_evidence_ids": "FUND-VAL-01|NEWS-TP-01",
    }
    values.update(overrides)
    body = "\n".join(f"  {key}: {value}" for key, value in values.items())
    return f"```yaml\nPM_SUMMARY:\n{body}\n```"


def _rm_summary() -> str:
    return """```yaml
RM_SUMMARY:
  target_price_low: 86
  target_price_mid: 101
  target_price_high: 118
```"""


def test_attribution_renderer_marks_valid_partial_missing_and_unauthorized_refs():
    ledger = compile_research_evidence(_complete_state())
    content = _pm_summary(
        short_term_evidence_ids="MKT-TREND-01|DOES-NOT-EXIST",
        position_evidence_ids="FUND-GROWTH-01",
        target_price_evidence_ids="null",
    )

    table = render_decision_attribution(
        content,
        {"effective_action": "等回踩"},
        ledger,
        rm_content=_rm_summary(),
    )

    assert "| 未来三日 | 等回踩 | 市场分析/风险 |" in table
    assert (
        "| 未来三日 | 等回踩 | 市场分析/风险 | "
        "MKT-STRUCT-01, RISK-GATE-01, MKT-TREND-01 |" in table
    )
    assert "部分：剔除不存在的证据 DOES-NOT-EXIST" in table
    assert "| 一年期评级 | OVERWEIGHT | 基本面/RM |" in table
    assert "部分：证据不完整" in table
    assert "| 新建仓位 | 2-3% | 风险/PM |" in table
    assert "部分：剔除权限不匹配的证据 FUND-GROWTH-01" in table
    assert "| 一年期目标价 | 86-118（中位 101） | 基本面/RM |" in table
    assert "缺失：PM 未完成证据归因" in table


def test_market_direction_uses_trend_and_momentum_instead_of_llm_wording_drift():
    state = _complete_state()
    state["market_report"] = state["market_report"].replace(
        "trend_daily: 上行\n  momentum: 强",
        "trend_daily: 震荡\n  momentum: 弱",
    )
    # Keep the analyst's prose-style direction bullish to reproduce the real drift.
    assert "data_implied_direction: 偏多" in state["market_report"]

    ledger = compile_research_evidence(state)
    trend = next(card for card in ledger["cards"] if card["claim_id"] == "MKT-TREND-01")

    assert trend["direction"] == "bearish"


def test_news_impact_direction_accepts_audited_suffixes():
    state = _complete_state()
    state["news_report"] = state["news_report"].replace("impact: +中", "impact: -中（已修复）")

    ledger = compile_research_evidence(state)
    news = next(card for card in ledger["cards"] if card["claim_id"] == "NEWS-CAT-01")

    assert news["direction"] == "bearish"


def test_unknown_news_event_date_and_future_quant_date_degrade_quality():
    state = _complete_state()
    state["news_report"] = state["news_report"].replace("event_date: 2026Q3", "event_date: 未知")
    state["quant_score"] = state["quant_score"].replace(
        "price_data_date: 2026-08-12", "price_data_date: 2026-08-13"
    )

    ledger = compile_research_evidence(state)
    by_id = {card["claim_id"]: card for card in ledger["cards"]}

    assert by_id["NEWS-CAT-01"]["quality_status"] == "partial"
    assert by_id["QUANT-COMP-01"]["quality_status"] == "invalid"


def test_attribution_rejects_invalid_and_wrong_variable_cards():
    state = _complete_state()
    state["market_report"] = state["market_report"].replace(
        "price_data_date: 2026-08-12", "price_data_date: 2026-08-13"
    )
    ledger = compile_research_evidence(state)
    content = _pm_summary(
        short_term_evidence_ids="MKT-TREND-01",
        target_price_evidence_ids="NEWS-CAT-01",
    )

    table = render_decision_attribution(
        content,
        {"effective_action": "等回踩"},
        ledger,
        rm_content=_rm_summary(),
    )

    assert "剔除无效证据 MKT-TREND-01" in table
    assert "剔除决策变量不匹配的证据 NEWS-CAT-01" in table


def test_attribution_renderer_accepts_valid_authorized_references():
    ledger = compile_research_evidence(_complete_state())
    table = render_decision_attribution(
        _pm_summary(short_term_evidence_ids="MKT-TREND-01|RISK-GATE-01"),
        {"effective_action": "小仓试探"},
        ledger,
        rm_content=_rm_summary(),
    )

    assert "MKT-STRUCT-01, RISK-GATE-01, MKT-TREND-01" in table
    assert "| 新建仓位 | 2-3% | 风险/PM | RISK-GATE-01 | 完整 |" in table
    assert "| 一年期评级 | OVERWEIGHT | 基本面/RM | FUND-GROWTH-01, NEWS-CAT-01 | 部分：证据不完整 |" in table


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
