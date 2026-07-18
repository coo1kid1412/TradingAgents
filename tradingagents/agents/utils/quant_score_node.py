"""Quant Score Officer 节点（纯 Python，无 LLM）

在 Capital Flow Officer 完成后、Macro Context Officer 之前运行。
通过 7 类因子（动量/价值/质量/成长/低波/反拥挤/资金流）输出 0-100 量化综合分，
作为下游 RM 在 Step 6 评级映射时的"独立量化锚"。

设计原则：
- **无 LLM**：所有数值由 Python 确定性计算，复跑同股得到同分
- **数据自取**：直接调 route_to_vendor，不依赖 analyst 已生成的 markdown 报告
- **第 7 因子外部注入**：capital_flow_score 由 Capital Flow Officer 预计算（state.capital_flow_metrics）
- **缺失容错**：单个因子取不到数据时归零权重，剩余因子按比例补全
- **输出标准化**：markdown 报告 + YAML 摘要，供 RM/PM 引用
"""

from __future__ import annotations

import io
import logging
import re
from typing import Optional

import pandas as pd

from tradingagents.dataflows.factor_calc import (
    compute_price_factors,
    compute_quant_score,
    DEFAULT_FACTOR_WEIGHTS,
)
from tradingagents.dataflows.interface import route_to_vendor
from tradingagents.dataflows.akshare_vendor import get_industry_pe_table
from tradingagents.dataflows.intraday_quote import parse_price_metadata

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 数据获取
# ---------------------------------------------------------------------------
def _fetch_price_df(ticker: str, trade_date: str) -> Optional[pd.DataFrame]:
    """拉取近 365 天 OHLCV，返回 DataFrame。"""
    import datetime as _dt
    end = _dt.datetime.strptime(trade_date, "%Y-%m-%d")
    start = end - _dt.timedelta(days=400)  # 多拉 35 天作缓冲
    try:
        csv_str = route_to_vendor(
            "get_stock_data",
            ticker,
            start.strftime("%Y-%m-%d"),
            end.strftime("%Y-%m-%d"),
        )
    except Exception as e:
        logger.warning("quant_score 获取价格数据失败: %s", e)
        return None

    if not csv_str or "未找到" in csv_str[:200]:
        return None

    price_meta = parse_price_metadata(csv_str)

    # 跳过以 # 开头的 header 行
    lines = [ln for ln in csv_str.splitlines() if not ln.startswith("#") and ln.strip()]
    if not lines:
        return None
    try:
        df = pd.read_csv(io.StringIO("\n".join(lines)))
    except Exception as e:
        logger.warning("quant_score 解析价格 CSV 失败: %s", e)
        return None
    df.attrs["price_metadata"] = price_meta
    return df


def _fetch_fundamentals_raw(ticker: str, trade_date: str) -> str:
    """拉取原始 fundamentals 字符串（含估值指标段 + 财务分析指标表）。"""
    try:
        return route_to_vendor("get_fundamentals", ticker, trade_date)
    except Exception as e:
        logger.warning("quant_score 获取基本面数据失败: %s", e)
        return ""


# ---------------------------------------------------------------------------
# 解析 fundamentals 字符串
# ---------------------------------------------------------------------------
_PE_TTM_RE = re.compile(r"动态PE\(TTM\)\s*[:：]\s*([0-9.]+)\s*倍")
_PB_RE = re.compile(r"PB\s*[:：]\s*([0-9.]+)")
_INDUSTRY_RE = re.compile(r"所属行业\s*[:：]\s*(\S+)")


def _safe_float(s: str) -> Optional[float]:
    try:
        return float(s)
    except (ValueError, TypeError):
        return None


def _parse_table_row(table_str: str, label_keyword: str) -> Optional[float]:
    """从 Markdown 表格字符串中找含 label_keyword 的行，返回第一个数据列（最新报告期）。

    AKShare 的 extract_and_format 表格列从左到右依次是 最新 → 较旧。
    """
    for line in table_str.splitlines():
        if not line.startswith("|"):
            continue
        if label_keyword not in line:
            continue
        # 切分单元格：| label | v1 | v2 | v3 |
        cells = [c.strip() for c in line.split("|")]
        # 过滤前后空字符串
        cells = [c for c in cells if c]
        if len(cells) < 2:
            continue
        # 第一个 cell 是 label，后续是数据，跳过 "------" 分隔行
        if "---" in cells[0]:
            continue
        # 找第一个非空、非 N/A 的数值
        for v in cells[1:]:
            if v in ("", "N/A", "-", "—", "null", "None"):
                continue
            f = _safe_float(v)
            if f is not None:
                return f
        return None
    return None


def _parse_fundamentals(fund_str: str) -> dict:
    """从 fundamentals 原始字符串中抽取因子计算所需的数值。

    返回的 dict 字段都是 Optional[float]：缺失字段留 None。
    """
    out = {
        "pe_ttm": None,
        "pb": None,
        "roe_ttm_pct": None,
        "gross_margin_pct": None,
        "net_margin_pct": None,
        "revenue_yoy_pct": None,
        "net_profit_yoy_pct": None,
        "deducted_profit_yoy_pct": None,
        "recurring_loss": None,
        "industry": None,
    }
    if not fund_str:
        return out

    # 1. 估值指标段（动态PE TTM、PB）
    m = _PE_TTM_RE.search(fund_str)
    if m:
        out["pe_ttm"] = _safe_float(m.group(1))
    m = _PB_RE.search(fund_str)
    if m:
        out["pb"] = _safe_float(m.group(1))

    # 2. 所属行业（用于行业 PE 对照）
    m = _INDUSTRY_RE.search(fund_str)
    if m:
        out["industry"] = m.group(1).strip()

    # 3. 财务分析指标表（最近报告期）
    out["roe_ttm_pct"] = _parse_table_row(fund_str, "净资产收益率")
    out["gross_margin_pct"] = _parse_table_row(fund_str, "销售毛利率")
    out["net_margin_pct"] = _parse_table_row(fund_str, "销售净利率")
    out["revenue_yoy_pct"] = _parse_table_row(fund_str, "营业收入同比增长率")
    out["net_profit_yoy_pct"] = _parse_table_row(fund_str, "净利润同比增长率")
    quality = re.search(
        r"SYS_GROWTH_QUALITY.*?recurring_loss\s*=\s*(yes|no).*?扣非净利YoY年度\s*=\s*([+-]?[0-9.]+)%",
        fund_str, re.S,
    )
    if quality:
        out["recurring_loss"] = quality.group(1) == "yes"
        out["deducted_profit_yoy_pct"] = _safe_float(quality.group(2))

    return out


def _try_get_industry_pe(industry: Optional[str], trade_date: str) -> Optional[float]:
    """根据所属行业从巨潮 cninfo 表中匹配行业 PE 中位数。"""
    if not industry:
        return None
    try:
        table_str = get_industry_pe_table(trade_date)
    except Exception as e:
        logger.warning("quant_score 获取行业 PE 表失败: %s", e)
        return None
    if not table_str:
        return None

    # 巨潮表格按"证监会行业分类"分级，行业名可能是大类（如"计算机、通信和其他电子设备制造业"）
    # 而 fundamentals 的"所属行业"是东财细分（如"半导体"）。先做精确匹配，失败做包含匹配。
    candidates: list[float] = []
    for line in table_str.splitlines():
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.split("|") if c.strip()]
        if len(cells) < 2 or "---" in cells[0]:
            continue
        # 期望表格列：行业 / 静态市盈率-中位数 等
        row_industry = cells[0]
        # 包含匹配（一方包含另一方）
        if (
            industry == row_industry
            or industry in row_industry
            or row_industry in industry
        ):
            # 找第一个看起来是 PE 数值的单元格
            for v in cells[1:]:
                f = _safe_float(v)
                if f is not None and 0 < f < 1000:
                    candidates.append(f)
                    break
    if not candidates:
        return None
    # 多个匹配时取中位数
    candidates.sort()
    return candidates[len(candidates) // 2]


# ---------------------------------------------------------------------------
# 报告格式化
# ---------------------------------------------------------------------------
_INTERPRETATION_RANGE = [
    (30, "显著负面"),
    (50, "偏弱"),
    (65, "中性"),
    (80, "偏强"),
    (101, "显著正面"),
]


def _factor_label(name: str) -> str:
    return {
        "momentum": "动量 Momentum",
        "value": "价值 Value",
        "quality": "质量 Quality",
        "growth": "成长 Growth",
        "lowvol": "低波 LowVol",
        "anticrowding": "反拥挤 AntiCrowding",
        "capital_flow": "资金流 CapitalFlow",
    }.get(name, name)


def _fmt(v) -> str:
    if v is None:
        return "N/A"
    if isinstance(v, float):
        return f"{v:.2f}"
    return str(v)


def _format_report(
    ticker: str,
    company_name: str,
    trade_date: str,
    result,
    raw_inputs: dict,
    price_meta: Optional[dict] = None,
) -> str:
    composite = result.composite
    interp = result.interpretation

    lines: list[str] = []
    lines.append(f"# {ticker} {company_name} 量化因子打分报告")
    lines.append("")
    lines.append(f"**分析日期**：{trade_date}")
    if price_meta:
        lines.append(
            f"**价格数据状态**：{price_meta.get('status', 'unknown')} | "
            f"截止 {price_meta.get('date') or '未知'} | "
            f"来源 {price_meta.get('source') or 'unknown'}"
        )
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 一、综合分数")
    lines.append("")
    if composite is None:
        lines.append("**Composite Score**：N/A（数据严重不足，无法计算）")
    else:
        lines.append(f"**Composite Score**：**{composite:.1f} / 100** —— {interp}")
    lines.append("")
    lines.append("> 解读区间：0-30 显著负面 | 30-50 偏弱 | 50-65 中性 | 65-80 偏强 | 80-100 显著正面")
    lines.append("")

    # 因子总表
    lines.append("## 二、因子分项")
    lines.append("")
    lines.append("| 因子 | 权重 | 分数 (0-100) | 状态 |")
    lines.append("|------|------|-------------|------|")
    weights = result.weights_used
    for name in ["momentum", "value", "quality", "growth", "lowvol", "anticrowding", "capital_flow"]:
        score = result.factor_scores.get(name)
        w = weights.get(name, 0)
        status = "✅" if score is not None else "⚠ 数据缺失"
        score_str = f"{score:.1f}" if score is not None else "N/A"
        lines.append(f"| {_factor_label(name)} | {w*100:.0f}% | {score_str} | {status} |")
    lines.append("")
    lines.append(
        f"> 覆盖率：{len(result.coverage['available'])} / 7 因子有数据，"
        f"实际使用权重和 = {result.coverage['total_weight_used']:.2f}"
    )
    lines.append("")

    # 因子明细
    lines.append("## 三、因子明细（原始输入 + 子分）")
    lines.append("")
    for name in ["momentum", "value", "quality", "growth", "lowvol", "anticrowding", "capital_flow"]:
        bd = result.factor_breakdowns.get(name, {})
        if not bd:
            continue
        lines.append(f"### {_factor_label(name)}")
        lines.append("")
        lines.append("| 指标 | 数值 |")
        lines.append("|------|------|")
        for k, v in bd.items():
            lines.append(f"| {k} | {_fmt(v)} |")
        lines.append("")

    # 数据缺失字段（仅在有缺失时显示，避免与第三节重复）
    missing_fields = [k for k, v in raw_inputs.items() if v is None]
    next_section_idx = 4
    if missing_fields:
        lines.append("## 四、数据缺失字段（未参与因子打分）")
        lines.append("")
        lines.append(f"以下 {len(missing_fields)} 个字段取数失败：`{', '.join(missing_fields)}`")
        lines.append("")
        lines.append("（其余字段的具体数值见上方「三、因子明细」各因子表格）")
        lines.append("")
        next_section_idx = 5

    # 给 RM 的提示
    _section_label = {4: "四", 5: "五"}[next_section_idx]
    lines.append(f"## {_section_label}、给 RM / PM 的使用指引")
    lines.append("")
    if composite is None:
        lines.append("- 量化分缺失，RM 应在 Step 6 评级映射中**不引用**本节，按估值偏离度逻辑独立判断。")
    else:
        lines.append(f"- 量化综合分 **{composite:.1f}**（{interp}），作为 RM Step 6 评级映射的独立锚。")
        lines.append("- **背离检查**：若 RM 机械映射给出的评级方向（OVERWEIGHT / HOLD / UNDERWEIGHT）与量化分严重背离（差距≥35 分），需在 COT 中说明背离原因；若无法解释，应往保守方向修正一档。")
        lines.append("  - 量化 < 30 + RM 给 OVERWEIGHT → 必须降为 HOLD 并说明理由")
        lines.append("  - 量化 > 80 + RM 给 UNDERWEIGHT → 必须升为 HOLD 并说明理由")
        lines.append("- **置信度增强**：若 RM 评级与量化方向一致（同高或同低），Conviction 可加一档（中→中高）")
        lines.append("- **薄弱因子警示**：分数 < 30 的单一因子代表该维度有显著风险，PM 在 Trade Ticket 中应单独列入 Key Risks。")
    lines.append("")

    # YAML 摘要
    lines.append("---")
    lines.append("")
    lines.append("## YAML 摘要")
    lines.append("")
    lines.append("```yaml")
    lines.append("QUANT_SCORE:")
    price_meta = price_meta or {}
    for key in ("status", "date", "time", "source"):
        value = price_meta.get(key)
        rendered = f'"{value}"' if value is not None else "null"
        lines.append(f"  price_data_{key}: {rendered}")
    lines.append(f"  composite: {composite if composite is not None else 'null'}")
    lines.append(f'  interpretation: "{interp}"')
    lines.append("  factor_scores:")
    for name in ["momentum", "value", "quality", "growth", "lowvol", "anticrowding", "capital_flow"]:
        v = result.factor_scores.get(name)
        lines.append(f"    {name}: {v if v is not None else 'null'}")
    lines.append("  weights:")
    for name, w in weights.items():
        lines.append(f"    {name}: {w}")
    lines.append("  coverage:")
    lines.append(f"    available: {result.coverage['available']}")
    lines.append(f"    missing: {result.coverage['missing']}")
    lines.append("```")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 节点工厂
# ---------------------------------------------------------------------------
def create_quant_score_node():
    """工厂函数：返回 quant_score 节点。无需 LLM 参数（纯 Python 计算）。"""

    def quant_score_node(state) -> dict:
        ticker = state["company_of_interest"]
        company_name = state.get("company_name", "")
        trade_date = state.get("trade_date", "")

        # 从 Capital Flow Officer 读取第 7 因子分数（可能为 None / 空 dict）
        cf_metrics = state.get("capital_flow_metrics") or {}
        capital_flow_score_input = cf_metrics.get("capital_flow_score")  # 0-100 or None

        # 1. 拉取原始数据
        price_df = _fetch_price_df(ticker, trade_date)
        price_meta = (
            price_df.attrs.get("price_metadata", {}) if price_df is not None else {}
        )
        fund_str = _fetch_fundamentals_raw(ticker, trade_date)

        # 2. 计算价格因子（动量 / 低波 / 反拥挤）
        price_factors = compute_price_factors(price_df) if price_df is not None else {
            "r3m_pct": None, "r6m_pct": None, "r12m_pct": None,
            "r60d_pct": None, "realized_vol_annualized_pct": None,
            "turnover_ratio_30d_to_90d": None,
        }

        # 3. 解析基本面字段
        fund_inputs = _parse_fundamentals(fund_str)

        # 4. 行业 PE 中位数（可能为 None）
        industry_pe = _try_get_industry_pe(fund_inputs.get("industry"), trade_date)

        # 5. 调因子计算
        result = compute_quant_score(
            r3m_pct=price_factors["r3m_pct"],
            r6m_pct=price_factors["r6m_pct"],
            r12m_pct=price_factors["r12m_pct"],
            r60d_pct=price_factors["r60d_pct"],
            realized_vol_annualized_pct=price_factors["realized_vol_annualized_pct"],
            turnover_ratio_30d_to_90d=price_factors["turnover_ratio_30d_to_90d"],
            pe_ttm=fund_inputs["pe_ttm"],
            pb=fund_inputs["pb"],
            pe_industry_median=industry_pe,
            roe_ttm_pct=fund_inputs["roe_ttm_pct"],
            gross_margin_pct=fund_inputs["gross_margin_pct"],
            net_margin_pct=fund_inputs["net_margin_pct"],
            revenue_yoy_pct=fund_inputs["revenue_yoy_pct"],
            net_profit_yoy_pct=fund_inputs["net_profit_yoy_pct"],
            recurring_loss=fund_inputs["recurring_loss"],
            deducted_profit_yoy_pct=fund_inputs["deducted_profit_yoy_pct"],
            holder_num_qoq_pct=cf_metrics.get("holder_num_qoq_pct"),
            winner_rate_pct=cf_metrics.get("winner_rate_pct"),
            capital_flow_score_input=capital_flow_score_input,
        )

        # 6. 组装审计用原始输入表
        raw_inputs = {
            **price_factors,
            **{k: v for k, v in fund_inputs.items() if k != "industry"},
            "industry": fund_inputs.get("industry"),
            "pe_industry_median": industry_pe,
        }

        # 7. 格式化 markdown 报告
        report = _format_report(
            ticker, company_name, trade_date, result, raw_inputs,
            price_meta=price_meta,
        )

        return {"quant_score": report}

    return quant_score_node
