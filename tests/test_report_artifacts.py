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


def test_user_report_replaces_decision_tables_with_mobile_digest():
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
    assert "## 现在怎么做" in user_text
    assert "Trade Ticket" not in user_text
    assert not any(line.lstrip().startswith("|") for line in user_text.splitlines())
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

    assert "短期操作结论" in user_text
    assert "Trade Ticket" not in user_text
    assert "长篇论证" not in user_text


def test_mobile_report_distills_research_domains_without_wide_tables():
    decision = """# 短期操作结论：暂不介入

> **一年期研究评级：OVERWEIGHT｜当前动作：WAIT｜新建仓位：0%**
>
> **趋势判断：未来 3 日 下跌（置信度 中）｜未来 12 个月主题 兑现**
>
> **重新评估条件：资金转正、业绩兑现且价格结构企稳后重新评估**

## 现在怎么做

| 持仓状态 | 当前建议 |
|---|---|
| **空仓** | **不买，保持新建仓位 0%** |
| **已持仓** | **不加仓**；反弹处理位 430 / 463.48 元；硬止损 350 元 |

### 关键 Agent 贡献

- **fundamentals / FUND-GROWTH-01**：营收同比 105.76%，归母净利同比 76.8%（作用：强支撑）
- **news / NEWS-CAT-01**：H1 净利预增 77.6%-102.9%，事件日期 2026-07-20（作用：强支撑）

## Trade Ticket 交易票（决策卡）

### 顶部导航（At-a-glance）

| 字段 | 内容 |
|---|---|
| 入场判断 | WAIT（板块 RS 30d -20.0%，仅为导航摘要） |
| 未来 3 个交易日趋势 | 下行；置信度：中 |
| 12 个月主题判断 | 兑现（增长正在验证，拥挤度已部分释放） |

### 关键背景

| 字段 | 内容 |
|---|---|
| 目标价区间 | 463.48-616.85 元（中值 540.17，12 个月） |
| **资金面快照（主力 vs 散户）** | 主力近 5 日净流出 36.86 亿，股东户数环比 +11.48%，主力减仓、散户承接 |
| Core Thesis 核心逻辑 | 1. 盈利重新加速。2. PE(TTM) 40.63，盈利上修消化估值。3. 等事件落地再建仓 |
| Key Risks 核心风险 | 1. 估值无缓冲（PB 28.3，PE 40.6）。2. 波动率高。3. 主力连续流出 |

### 热门概念归属（这股属于哪个最热的板块/概念，占主营多少）

| 概念/板块 | 相关度 | 占主营营收% | 当前热度 |
|---|---|---|---|
| CPO/光模块 | 核心 | ~100% | 高（1.6T 放量催化密集） |

## 一、投资决策与入场时机

板块 RS 30d -18.6% 跑输沪深300，主题内排名第 2/4，个股相对行业 ETF +4.9%。
"""
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        result = write_consolidated_reports(
            root,
            ticker="300502",
            user_decision=decision,
            audit_sections=[decision],
            generated_at="2026-08-21 04:24:08",
        )
        user_text = result.read_text(encoding="utf-8")
        audit_text = (root / "audit_report.md").read_text(encoding="utf-8")

    assert not any(line.lstrip().startswith("|") for line in user_text.splitlines())
    assert "未来 3 日：下跌（置信度 中）" in user_text
    for heading in (
        "## 现在怎么做",
        "## 基本面",
        "## 消息面与催化",
        "## 赛道与前景",
        "## 近期轮动与资金",
        "## 估值、风险与重估条件",
    ):
        assert heading in user_text
    assert "营收同比 105.76%" in user_text
    assert "H1 净利预增 77.6%-102.9%" in user_text
    assert "CPO/光模块" in user_text
    assert "\n- 板块 RS 30d -18.6%" in user_text
    assert "主力近 5 日净流出 36.86 亿" in user_text
    assert "463.48-616.85 元" in user_text
    assert "FUND-VAL-01" not in user_text
    assert "\n- PE(TTM) 40.63，盈利上修消化估值。" in user_text
    assert "\n- 主要风险：估值无缓冲（PB 28.3，PE 40.6）。" in user_text
    assert "\n- 主要风险：波动率高。" in user_text
    assert "| 概念/板块 |" in audit_text


def test_mobile_report_classifies_core_thesis_when_contribution_ledger_is_absent():
    decision = """# 短期操作结论：等待条件确认

> **一年期研究评级：HOLD｜当前动作：WAIT｜新建仓位：0%**
>
> **趋势判断：未来 3 日 震荡（置信度 中）｜未来 12 个月主题 验证**

## Trade Ticket 交易票（决策卡）

### 关键背景

| 字段 | 内容 |
|---|---|
| Core Thesis 核心逻辑 | 1. 营收同比 45%，归母净利同比 38%，盈利继续增长。2. PE(TTM) 40.63，估值处于区间上沿。3. CPO 赛道需求仍强但竞争加剧 |
| 目标价区间 | 90-110 元 |

### 核心交易参数

| 参数 | 数值 | 中文说明 |
|---|---|---|
| Time Stop 时间止损 | 6 个月 / 12 个月 | 关注 2026-08-31 中报与后续政策落地 |
"""
    with tempfile.TemporaryDirectory() as directory:
        result = write_consolidated_reports(
            Path(directory),
            ticker="000001",
            user_decision=decision,
            audit_sections=[decision],
            generated_at="2026-08-21 08:30:00",
        )
        user_text = result.read_text(encoding="utf-8")

    fundamentals = user_text.split("## 基本面", 1)[1].split("## 消息面与催化", 1)[0]
    news = user_text.split("## 消息面与催化", 1)[1].split("## 赛道与前景", 1)[0]
    sector = user_text.split("## 赛道与前景", 1)[1].split("## 近期轮动与资金", 1)[0]
    valuation = user_text.split("## 估值、风险与重估条件", 1)[1]

    assert "营收同比 45%" in fundamentals
    assert "2026-08-31 中报" in news
    assert "CPO 赛道需求仍强" in sector
    assert "PE(TTM) 40.63" not in sector
    assert "营收同比 45%" not in sector
    assert "PE(TTM) 40.63" in valuation


def test_mobile_report_preserves_balanced_parentheses_in_table_only_rotation_fallback():
    decision = """# 短期操作结论：继续观察

## Trade Ticket 交易票（决策卡）

| 字段 | 内容 |
|---|---|
| 入场判断 | WAIT（板块 RS 30d -18.6%（vs 沪深300）） |
"""
    with tempfile.TemporaryDirectory() as directory:
        result = write_consolidated_reports(
            Path(directory),
            ticker="000001",
            user_decision=decision,
            audit_sections=[decision],
            generated_at="2026-08-21 08:30:00",
        )
        user_text = result.read_text(encoding="utf-8")

    assert "板块 RS 30d -18.6%（vs 沪深300）" in user_text


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"{len(tests)}/{len(tests)} passed")
