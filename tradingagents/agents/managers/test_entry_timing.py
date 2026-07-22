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
from tradingagents.harness.extractor import _find_yaml_block


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


def test_stale_snapshot_replaces_unnumbered_three_day_section_entirely():
    content = """## Trade Ticket 交易票
| 未来 3 个交易日趋势 | **震荡偏弱**（置信度：中） |

### 未来 3 个交易日趋势

**判定**：**震荡偏弱**（置信度：中）

领先风险信号：这里是一段基于陈旧快照的详细预测。

### 12 个月主题判断

兑现期。

```yaml
PM_SUMMARY:
  pm_rating: OVERWEIGHT
  pm_action_keyword: WAIT
  short_term_trend: 震荡
  entry_timing: 退出观察
```
"""
    result = _format_pm_decision(
        content,
        {"structure_class": "broken", "effective_action": "退出观察"},
        market_risk_snapshot={"data_status": "stale", "required_checkpoint": "14:30"},
    )
    assert "### 未来 3 个交易日趋势" in result
    assert "**数据不足**（盘中风险快照陈旧，需 14:30 检查点；当前仅执行 WAIT、0%）" in result
    assert "震荡偏弱" not in result
    assert "领先风险信号" not in result
    assert _find_yaml_block(result, "PM_SUMMARY")["short_term_trend"] == "数据不足"


def test_no_buy_recomputes_holder_levels_from_current_price():
    content = """## Trade Ticket 交易票

| Conviction 信心 | ⭐⭐⭐ Medium（anchor_sensitive=true） |
| 结构时机 | 退出观察；结构=broken |
| **Entry** 入场区间 | 待结构修复后再定 | 未来参考 |
| **1R** 风险单元 | **17.00 元** | 以参考入场 107 元为基准 |
| **TP1** 止盈 1 | **124.00 元** | 减仓 |
| **TP2** 止盈 2 | **141.00 元** | 减仓 |
| **TP3** 止盈 3 | **158.00 元** | 清仓 |
| **SL_soft** 软止损 | **96.80 元** | 减仓 |
| **SL_hard** 硬止损 | **90.00 元** | 清仓 |
| 当前赔率 | R = 4.50（基于参考入场 107 元） |
| 资金面快照 | capital_flow_score 76/100，regime=中性，winner_rate_pct 59.1% |
| 12 个月主题判断 | quant_anticrowding=85，quant value=5，lowvol=5 |

```yaml
PM_SUMMARY:
  current_price: 118.34
  pm_rating: OVERWEIGHT
  pm_conviction_stars: 3
  pm_action_keyword: WAIT
  pm_size_low_pct: 4
  pm_size_high_pct: 6
  pm_entry_low: 107
  pm_entry_high: 118
  pm_tp1: 124.00
  pm_tp2: 141.00
  pm_tp3: 158.00
  pm_sl_soft: 96.80
  pm_sl_hard: 90.00
  entry_timing: 退出观察
```
"""
    result = _format_pm_decision(
        content,
        {"structure_class": "broken", "effective_action": "退出观察"},
    )
    assert "| **Entry** 入场区间 | **—** | 当前不建仓；重新评估后重算入场区间 |" in result
    assert "| **1R** 风险单元 | **28.34 元** | 以当前价 118.34 元管理已有仓位 |" in result
    assert "| **TP1** 止盈 1 | **146.68 元**" in result
    assert "| **TP2** 止盈 2 | **175.02 元**" in result
    assert "| **TP3** 止盈 3 | **203.36 元**" in result
    assert "| **SL_soft** 软止损 | **101.34 元**" in result
    assert "| Conviction 信心 | ⭐⭐⭐ **中等**（长期判断与短期时机分开，当前不具备新建仓条件） |" in result
    assert "| 结构时机 | 退出观察（短线结构已破坏） |" in result
    assert "| 当前赔率 | 当前不建仓；赔率在重新评估入场条件后重算 |" in result
    assert "资金流评分 76/100，状态=中性，获利盘 59.1%" in result
    assert "反拥挤因子=85，量化价值因子=5，低波因子=5" in result
    assert "anchor_sensitive" not in result
    assert "quant_anticrowding" not in result
    assert "capital_flow_score" not in result
    assert "winner_rate_pct" not in result
    summary = _find_yaml_block(result, "PM_SUMMARY")
    assert summary["pm_tp1"] == 146.68
    assert summary["pm_tp2"] == 175.02
    assert summary["pm_tp3"] == 203.36
    assert summary["pm_sl_soft"] == 101.34


def test_no_buy_structure_explanation_matches_the_actual_timing_state():
    content = """## Trade Ticket 交易票

| Conviction 信心 | ⭐⭐⭐ Medium |
| 结构时机 | 等回踩 |

## 系统归档数据（供程序读取）

```yaml
PM_SUMMARY:
  pm_rating: HOLD
  pm_conviction_stars: 3
  pm_action_keyword: WAIT
  entry_timing: 等回踩
```
"""
    result = _format_pm_decision(
        content,
        {"structure_class": "trend_pullback", "effective_action": "等回踩"},
    )
    assert "| 结构时机 | 等回踩（趋势仍在，等待更好赔率） |" in result
    assert "短线结构已破坏" not in result
    assert "长期评级保持正面" not in result
    assert "系统归档数据" not in result


def test_pm_decision_hides_internal_audit_and_surfaces_position_specific_actions():
    content = """模型正在调用工具并检查不变量。

## Trade Ticket 交易票

| 字段 | 内容 |
|------|------|
| Rating 评级 | **HOLD** |
| 入场判断 | **DON'T BUY** |
| 结构时机 | 退出观察；结构=broken |

### 核心交易参数（Trade Parameters）

| 参数 | 数值 | 中文说明 |
|------|------|---------|
| **Size** 仓位规模 | 新建仓 2-3% |
| **Time Stop** 时间止损 | **6 月 / 12 月** | 6 月内未兑现减半，12 月内未兑现退出 |

### 各 Agent 核心结论一览

| Agent | 核心结论 |
|------|------|
| 股票画像识别官 | SYS_VALUATION_REGIME=ride |
| 研究主管 | 工具强制降档，effective_action=退出观察 |

## 一、投资决策与入场时机

### 1.2 入场时机：**DON'T BUY**

短期结构已破坏，effective_action="退出观察"逐字采用，禁止改写。

## 二、操作计划

### 2.1 操作动作表（按持仓场景）

**Scenario A：你当前空仓**（不建仓 / WAIT）

**Scenario B：你当前已持仓**（不加仓）

### 2.2 执行细节

风控缓释采用“突破确认+回调加仓”。

## 三、情景概率与赔率

### 3.1 四情景分布

| 情景 | 概率 |
|------|------|
| Base | 50% |

### 3.2 工具返回验证

工具返回 effective_size_low_pct=0.0，概率加总通过。

## 四、风险、触发与监控

### 4.1 关键监控指标

跌破软止损后减仓。

### 风控审查回应（与三位分析师交叉验证）

回应流动性分析师的内部过程。

### 不一致性最终自检（内部确认，不输出）

Action、Size 和不变量全部通过。

## 五、附录：自检与归档

### 5.1 历史教训应用自检

内部自检内容。

### 5.3 评级调整说明

PM 2A 与 PM 2B 通过，不变量 A 通过。

### 5.4 PM_SUMMARY YAML

```yaml
PM_SUMMARY:
  ticker: "603629"
  trade_date: "2026-07-22"
  current_price: 118.34
  pm_rating: HOLD
  pm_action_keyword: WAIT
  pm_size_low_pct: 2
  pm_size_high_pct: 3
  pm_tp1: 136.68
  pm_tp2: 155.02
  pm_tp3: 173.36
  pm_sl_soft: 107.34
  pm_sl_hard: 100.00
  short_term_structure: broken
  entry_timing: 退出观察
```
"""
    result = _format_pm_decision(
        content,
        {"structure_class": "broken", "effective_action": "退出观察"},
    )

    assert "| **空仓** | **不买，保持新建仓位 0%** |" in result
    assert "| **已持仓** | **不加仓**；反弹处理位 136.68 / 155.02 / 173.36 元；软止损 107.34 元，硬止损 100 元 |" in result
    assert "6 个月 / 12 个月" in result
    assert "工具返回验证" not in result
    assert "附录：自检与归档" not in result
    assert "SYS_VALUATION_REGIME" not in result
    assert "effective_action" not in result
    assert "PM 2A" not in result
    assert "不变量 A" not in result
    assert "回调加仓" not in result
    assert "风控审查回应" not in result
    assert "不一致性最终自检" not in result
    assert _find_yaml_block(result, "PM_SUMMARY")["current_price"] == 118.34


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


def test_pm_prompt_requests_only_user_facing_sections():
    source = Path(__file__).with_name("portfolio_manager.py").read_text(encoding="utf-8")
    assert "用户版报告不得输出以下内容" in source
    assert "只输出 Trade Ticket + 四个一级标题" in source
    assert "| **Time Stop** 时间止损 | 6 个月 / 12 个月 |" in source


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
