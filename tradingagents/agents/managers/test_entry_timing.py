"""Deterministic entry-timing gate tests.

Run: .venv/bin/python tradingagents/agents/managers/test_entry_timing.py
"""

from __future__ import annotations

import sys
from pathlib import Path

from tradingagents.agents.managers.rm_tools import compute_entry_timing
from tradingagents.agents.managers.research_manager import (
    _derive_entry_timing_from_profile,
    _enforce_entry_timing_truth,
    _extract_rm_rating,
)
from tradingagents.agents.managers.portfolio_manager import (
    AIMessage as PortfolioAIMessage,
    _format_pm_decision,
)
from tradingagents.agents.utils.agent_utils import RISK_DEBATE_PHRASING_RULES


def _timing(structure_class="trend_pullback", market_mode="risk_on", **kwargs):
    return compute_entry_timing(
        structure_class=structure_class,
        market_mode=market_mode,
        recurring_loss=kwargs.get("recurring_loss", False),
        earnings_revision=kwargs.get("earnings_revision", "停修"),
        valuation_regime=kwargs.get("valuation_regime", "ride"),
        has_peak_signal=kwargs.get("has_peak_signal", False),
        retail_concentration_signal=kwargs.get("retail_concentration_signal", "中性"),
        rsi_percentile_1y=kwargs.get("rsi_percentile_1y", 60),
        capital_flow_regime=kwargs.get("capital_flow_regime", "中性"),
        main_force_streak_days=kwargs.get("main_force_streak_days", 0),
        long_term_rating=kwargs.get("long_term_rating"),
    )


def test_structure_to_base_action_mapping():
    expected = {
        "trend_pullback": "分批介入",
        "breakout_ready": "等放量突破",
        "healthy_trend": "等回踩",
        "exhaustion": "暂不介入",
        "broken": "退出观察",
        "neutral": "继续观察",
        "insufficient_data": "数据不足",
    }
    for structure_class, action in expected.items():
        result = _timing(structure_class)
        assert result["base_action"] == action, result
        assert result["effective_action"] == action, result


def test_conditional_only_downgrades_active_entry():
    assert _timing("trend_pullback", "conditional")["effective_action"] == "小仓试探"
    assert _timing("breakout_ready", "conditional")["effective_action"] == "等放量突破"
    assert _timing("healthy_trend", "conditional")["effective_action"] == "等回踩"


def test_risk_off_vetoes_positive_actions_but_preserves_broken():
    for structure_class in ("trend_pullback", "breakout_ready", "healthy_trend"):
        result = _timing(structure_class, "risk_off")
        assert result["effective_action"] == "暂不介入", result
        assert result["market_mode"] == "risk_off", result
    assert _timing("broken", "risk_off")["effective_action"] == "退出观察"


def test_unknown_market_mode_fails_closed():
    result = _timing("trend_pullback", "unexpected")
    assert result["market_mode"] == "risk_off", result
    assert result["effective_action"] == "暂不介入", result


def test_fundamental_vetoes_block_active_entry():
    cases = [
        {"recurring_loss": True},
        {"earnings_revision": "下修"},
        {"valuation_regime": "discipline"},
        {"has_peak_signal": True},
        {"retail_concentration_signal": "散户高接盘", "rsi_percentile_1y": 88},
        {"capital_flow_regime": "恶化"},
        {"main_force_streak_days": -3},
    ]
    for case in cases:
        result = _timing(**case)
        assert result["effective_action"] == "暂不介入", (case, result)
        assert result["vetoed"] is True, (case, result)
        assert result["reasons"], (case, result)


def test_outflow_is_not_veto_when_earnings_are_revised_up():
    result = _timing(capital_flow_regime="恶化", main_force_streak_days=-5,
                     earnings_revision="上修")
    assert result["effective_action"] == "分批介入", result
    assert result["vetoed"] is False, result


def test_unknown_structure_returns_data_insufficient():
    result = _timing("new_unrecognized_state")
    assert result["base_action"] == "数据不足", result
    assert result["effective_action"] == "数据不足", result


def test_long_term_rating_caps_positive_timing():
    hold = _timing(long_term_rating="HOLD")
    assert hold["effective_action"] == "等回踩", hold
    for rating in ("UNDERWEIGHT", "SELL"):
        result = _timing(long_term_rating=rating)
        assert result["effective_action"] == "暂不介入", result
        assert result["vetoed"] is True, result


def test_positive_ratings_allow_active_timing():
    for rating in ("BUY", "OVERWEIGHT"):
        result = _timing(long_term_rating=rating)
        assert result["effective_action"] == "分批介入", result


def test_profile_truth_lines_drive_entry_timing():
    profile = """
SYS_VALUATION_REGIME: ride
SYS_EARNINGS_REVISION: 上修（卖方盈利预期方向）
SYS_SHORT_TERM_STRUCTURE: class=trend_pullback | ma10_slope_5d_pct=2.35 | price_vs_ma10_pct=0.50 | volume_ratio_5d_20d=0.71 | breakout_confirmed=false
SYS_ENTRY_RECURRING_LOSS: false
SYS_ENTRY_HAS_PEAK_SIGNAL: false
SYS_ENTRY_RETAIL_CONCENTRATION: 中性
SYS_ENTRY_RSI_PERCENTILE_1Y: 60
SYS_ENTRY_CAPITAL_FLOW_REGIME: 强势
SYS_ENTRY_MAIN_FORCE_STREAK_DAYS: 2
"""
    result = _derive_entry_timing_from_profile(profile, "conditional")
    assert result["structure_class"] == "trend_pullback", result
    assert result["base_action"] == "分批介入", result
    assert result["effective_action"] == "小仓试探", result


def test_missing_profile_truth_fails_closed():
    result = _derive_entry_timing_from_profile("", "risk_on")
    assert result["structure_class"] == "unknown", result
    assert result["effective_action"] == "数据不足", result


def test_rm_rating_extraction_prefers_summary_field():
    report = "正文曾讨论 BUY\nRM_SUMMARY:\n  rm_rating: HOLD\n"
    assert _extract_rm_rating(report) == "HOLD"
    assert _extract_rm_rating("无摘要") is None


def test_output_truth_overrides_m3_summary_and_trade_ticket_drift():
    content = """| 结构时机 | 等回踩；结构=healthy_trend |
RM_SUMMARY:
  market_mode: risk_on
  short_term_structure: neutral
  entry_timing: 等回踩
"""
    timing = {
        "structure_class": "healthy_trend",
        "effective_action": "暂不介入",
    }
    fixed = _enforce_entry_timing_truth(content, timing)
    assert "short_term_structure: healthy_trend" in fixed
    assert "entry_timing: 暂不介入" in fixed
    assert "| 结构时机 | 暂不介入；结构=healthy_trend |" in fixed
    assert "entry_timing: 等回踩" not in fixed


def test_portfolio_manager_has_ai_message_available_for_output_enforcement():
    assert PortfolioAIMessage.__name__ == "AIMessage"


def test_pm_decision_starts_with_salient_action_and_removes_working_preamble():
    content = """我需要重新核算赔率并调用工具。
现在正式撰写决策报告。

## Trade Ticket 交易票

| 结构时机 | 暂不介入；结构=healthy_trend |

---
PM_SUMMARY:
  pm_rating: OVERWEIGHT
  pm_action_keyword: WAIT
  pm_size_low_pct: 2
  pm_size_high_pct: 3
  short_term_structure: healthy_trend
  entry_timing: 暂不介入
"""
    result = _format_pm_decision(
        content,
        {"structure_class": "healthy_trend", "effective_action": "暂不介入"},
    )
    assert result.startswith("# 短期操作结论：暂不介入\n")
    assert "**当前动作：WAIT｜新建仓位：0%｜长期评级：OVERWEIGHT**" in result
    assert "## Trade Ticket 交易票" in result
    assert "我需要重新核算" not in result
    assert result.rstrip().endswith("entry_timing: 暂不介入")


def test_pm_decision_without_trade_ticket_preserves_original_content():
    content = "PM_SUMMARY:\n  pm_rating: HOLD\n  entry_timing: 继续观察\n"
    result = _format_pm_decision(
        content,
        {"structure_class": "neutral", "effective_action": "继续观察"},
    )
    assert result.startswith("# 短期操作结论：继续观察\n")
    assert content.strip() in result


def test_exit_observation_removes_entry_instructions_and_zeros_new_position():
    content = """## Trade Ticket 交易票

| **Size** 仓位规模 | 新建仓 2-3% |

### 1.3 未来 3 个交易日趋势
入场条件：等回踩 98 元企稳，分批试探单笔 1%。

---
PM_SUMMARY:
  pm_rating: HOLD
  pm_action_keyword: WAIT
  pm_size_low_pct: 2
  pm_size_high_pct: 3
  pm_entry_low: 98
  pm_entry_high: 100
  entry_timing: 退出观察
"""
    result = _format_pm_decision(
        content,
        {"structure_class": "broken", "effective_action": "退出观察"},
    )
    assert "新建仓 0%" in result
    assert "pm_size_low_pct: 0" in result
    assert "pm_size_high_pct: 0" in result
    assert "pm_entry_low: null" in result
    assert "pm_entry_high: null" in result
    assert "分批试探" not in result
    assert "入场条件：等回踩" not in result
    assert "重新评估条件：结构修复并重新通过风险门控" in result


def test_stale_market_snapshot_forces_three_day_trend_to_data_insufficient():
    content = """## Trade Ticket 交易票
| 未来 3 个交易日趋势 | **下行**（置信度：低） |

### 1.3 未来 3 个交易日趋势
**下行**（置信度：低，数据陈旧）。

PM_SUMMARY:
  pm_rating: SELL
  pm_action_keyword: WAIT
  short_term_trend: 下跌
  short_term_confidence: 低
  entry_timing: 退出观察
"""
    result = _format_pm_decision(
        content,
        {"structure_class": "broken", "effective_action": "退出观察"},
        market_risk_snapshot={"data_status": "stale", "required_checkpoint": "14:30"},
    )
    assert "| 未来 3 个交易日趋势 | **数据不足**（盘中风险快照陈旧，需 14:30 检查点） |" in result
    assert "short_term_trend: 数据不足" in result
    assert "**下行**（置信度：低，数据陈旧）" not in result


def test_rm_and_pm_prompt_contracts_keep_rating_and_timing_separate():
    root = Path(__file__).resolve().parents[3]
    for relative in (
        "tradingagents/agents/managers/research_manager.py",
        "tradingagents/agents/managers/portfolio_manager.py",
    ):
        source = (root / relative).read_text(encoding="utf-8")
        assert "SYS_SHORT_TERM_STRUCTURE" in source, relative
        assert "entry_timing" in source, relative
        assert "短线结构不得改变长期评级" in source, relative


def test_shared_agent_rules_reject_uncalibrated_precision_and_unsourced_facts():
    assert "未经历史校准" in RISK_DEBATE_PHRASING_RULES
    assert "事实性描述必须能追溯" in RISK_DEBATE_PHRASING_RULES


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
