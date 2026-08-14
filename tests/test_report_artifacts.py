import tempfile
from pathlib import Path

from tradingagents.reporting import write_consolidated_reports


def test_user_report_is_decision_first_and_audit_report_keeps_working_material():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        result = write_consolidated_reports(
            root,
            ticker="603629",
            user_decision="# 短期操作结论：继续观察\n\n**空仓：不买**",
            audit_sections=["## 分析师团队报告\n\n第一步：分析数据", "## 风险辩论\n\n内部讨论"],
            generated_at="2026-08-13 16:00:00",
        )
        user_text = result.read_text(encoding="utf-8")
        audit_text = (root / "audit_report.md").read_text(encoding="utf-8")

    assert "短期操作结论" in user_text
    assert "第一步：分析数据" not in user_text
    assert "内部讨论" not in user_text
    assert "第一步：分析数据" in audit_text
    assert "内部讨论" in audit_text


def test_user_report_keeps_decision_card_but_moves_long_form_to_audit():
    decision = """# 短期操作结论：继续观察

> **当前动作：WAIT｜新建仓位：0%｜长期评级：OVERWEIGHT**

## 现在怎么做

**空仓：不买**

## Trade Ticket 交易票

### 核心交易参数（Trade Parameters）

| **Action** 操作 | **WAIT** |

## 一、投资决策与入场时机

这是模型生成的长篇论证，不应进入日常用户版。

## 二、操作计划

更多长篇操作细节。
"""
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        result = write_consolidated_reports(
            root,
            ticker="603629",
            user_decision=decision,
            audit_sections=[decision],
            generated_at="2026-08-13 16:00:00",
        )
        user_text = result.read_text(encoding="utf-8")
        audit_text = (root / "audit_report.md").read_text(encoding="utf-8")

    assert "短期操作结论" in user_text
    assert "核心交易参数" in user_text
    assert "## 一、投资决策与入场时机" not in user_text
    assert "长篇论证" not in user_text
    assert "## 一、投资决策与入场时机" in audit_text


def test_user_report_detects_common_long_form_heading_variants():
    decision = """# 短期操作结论：等待条件确认

## Trade Ticket 交易票

| **Action** 操作 | **WAIT** |

## 1. 投资决策与入场时机（未来三日）

这是只应进入审计报告的长篇论证。
"""
    with tempfile.TemporaryDirectory() as directory:
        result = write_consolidated_reports(
            Path(directory), ticker="603629", user_decision=decision,
            audit_sections=[decision], generated_at="2026-08-13 16:00:00",
        )
        user_text = result.read_text(encoding="utf-8")

    assert "Trade Ticket" in user_text
    assert "长篇论证" not in user_text


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"{len(tests)}/{len(tests)} passed")
