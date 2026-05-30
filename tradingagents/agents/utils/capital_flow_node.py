"""资金流 Officer 节点（Capital Flow Officer）

业内对标：买方私募的「筹码分析师」 + 卖方金工的「资金流策略组」
- 职责：把 DDX/DDY/DDZ-like、北向、龙虎榜、股东户数 揉成「资金面综合状态」
- 不调 LLM：所有判定都是 capital_flow_utils 的纯 Python 函数（确定性、可审计、可回放）

在 graph 中位置：
- 上游依赖：market_analyst（已知 trade_date / company_of_interest）
- 下游消费：market_analyst 第四节、quant_score_node 第 7 因子、build_report_context

输出 state 字段：
- capital_flow_yaml      : YAML 字符串（供 prompt 占位符注入与下游程序化解析）
- capital_flow_report    : markdown 报告（capital_flow.md 直接落盘）
- capital_flow_metrics   : dict，含全部全名字段（供 quant_score_node / RM 直接读取）
"""

from __future__ import annotations

import logging
from typing import Optional

from tradingagents.dataflows.capital_flow_utils import (
    FIELD_LABEL_ZH,
    assemble_capital_flow_metrics,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 字段格式化（_yi → "亿" / _pct → "%" / _ratio → "" / 其余原值）
# ---------------------------------------------------------------------------
def _format_field_value(field: str, value) -> str:
    if value is None:
        return "N/A"
    if field.endswith("_yi"):
        return f"{value:.2f} 亿"
    if field.endswith("_pct") or field.endswith("_pct_1y"):
        return f"{value:.2f}%"
    if field.endswith("_ratio"):
        return f"{value:.3f}"
    if field.endswith("_days") or field.endswith("_count") or field.endswith("_count_30d"):
        return str(int(value))
    if field == "capital_flow_score":
        return f"{value:.1f}"
    if field == "holder_num_latest":
        return f"{int(value):,} 户"
    return str(value)


# ---------------------------------------------------------------------------
# YAML 序列化（不依赖 PyYAML，避免引入新依赖）
# ---------------------------------------------------------------------------
def _to_yaml(metrics: dict) -> str:
    """把 metrics dict 序列化为 YAML 字符串（仅扁平 key-value，无嵌套 list）。"""
    lines = ["CAPITAL_FLOW:"]
    for field, value in metrics.items():
        if field == "capital_flow_score_breakdown":
            # breakdown 是嵌套 dict，单独处理
            lines.append(f"  {field}:")
            if isinstance(value, dict):
                for k, v in value.items():
                    if v is None:
                        lines.append(f"    {k}: null")
                    elif isinstance(v, str):
                        lines.append(f"    {k}: \"{v}\"")
                    else:
                        lines.append(f"    {k}: {v}")
            continue
        if field == "capital_flow_votes":
            lines.append(f"  {field}:")
            if isinstance(value, dict):
                for k, v in value.items():
                    lines.append(f"    {k}: \"{v}\"")
            continue
        if value is None:
            lines.append(f"  {field}: null")
        elif isinstance(value, str):
            lines.append(f"  {field}: \"{value}\"")
        elif isinstance(value, bool):
            lines.append(f"  {field}: {str(value).lower()}")
        else:
            lines.append(f"  {field}: {value}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Markdown 报告生成
# ---------------------------------------------------------------------------
_REPORT_FIELD_GROUPS: list[tuple[str, list[str]]] = [
    ("一、主力资金核心指标", [
        "main_force_net_inflow_5d_yi",
        "main_force_net_inflow_20d_yi",
        "ddx_like_5d_pct",
        "ddx_like_5d_pct_1y",
        "large_order_net_inflow_5d_yi",
        "ddz_like_20d_pct",
        "net_inflow_streak_days",
    ]),
    ("二、散户参与度", [
        "retail_buy_amount_rate_5d_pct",
        "retail_concentration_signal",
    ]),
    ("三、北向资金", [
        "northbound_5d_direction",
        "northbound_20d_direction",
        "northbound_data_status",
        "northbound_latest_date",
    ]),
    ("四、龙虎榜与股东户数", [
        "lhb_count_30d",
        "lhb_inst_net_buy_30d_yi",
        "lhb_inst_direction",
        "holder_num_latest",
        "holder_num_qoq_pct",
        "holder_num_4q_trend",
        "chip_concentration_signal",
    ]),
    ("五、综合判定", [
        "circulating_market_value_yi",
        "capital_flow_regime",
        "capital_flow_regime_reasoning",
        "capital_flow_score",
    ]),
]


def _build_markdown_report(
    symbol: str,
    company_name: str,
    trade_date: str,
    metrics: dict,
    data_source_breakdown: dict,
) -> str:
    """生成 capital_flow.md 报告。"""
    lines: list[str] = []
    lines.append(f"# 资金流综合分析 —— {company_name}（{symbol}）")
    lines.append("")
    lines.append(f"- 分析日期：{trade_date}")
    lines.append(
        "- 数据源：moneyflow="
        f"{data_source_breakdown.get('moneyflow', 'missing')}，"
        f"流通市值={data_source_breakdown.get('circ_mv', 'missing')}，"
        f"龙虎榜={data_source_breakdown.get('lhb', 'missing')}"
    )
    lines.append("")

    for section_title, fields in _REPORT_FIELD_GROUPS:
        lines.append(f"## {section_title}")
        lines.append("")
        lines.append("| 字段 | 中文标签 | 取值 |")
        lines.append("|---|---|---|")
        for f in fields:
            label = FIELD_LABEL_ZH.get(f, f)
            value = metrics.get(f)
            lines.append(f"| `{f}` | {label} | {_format_field_value(f, value)} |")
        lines.append("")

    # 综合解读
    regime = metrics.get("capital_flow_regime")
    score = metrics.get("capital_flow_score")
    reasoning = metrics.get("capital_flow_regime_reasoning", "")
    lines.append("## 一句话解读")
    lines.append("")
    if regime == "数据不足":
        lines.append(f"- 资金面综合状态：**{regime}**（≥3 个维度数据缺失，不计算 capital_flow_score）")
    else:
        score_str = f"{score:.1f}" if score is not None else "N/A"
        lines.append(f"- 资金面综合状态：**{regime}**，capital_flow_score = **{score_str}** / 100")
        lines.append(f"- 解释：{reasoning}")
    lines.append("")

    # 投票明细
    votes = metrics.get("capital_flow_votes", {})
    if votes:
        lines.append("## 五维投票明细")
        lines.append("")
        lines.append("| 维度 | 票 | 含义 |")
        lines.append("|---|---|---|")
        vote_label = {
            "streak": "连续天数（净流入/流出）",
            "ddx_pct_1y": "DDX-like 5 日 1 年分位",
            "northbound": "北向资金 5 日方向",
            "lhb": "龙虎榜 30 日上榜数",
            "retail_takeover": "散户接盘度",
        }
        for k, v in votes.items():
            lines.append(f"| {vote_label.get(k, k)} | `{v}` | "
                         f"{'+ 多头票' if v=='+' else '- 空头票' if v=='-' else '0 中性' if v=='0' else 'X 数据缺失'} |")
        lines.append("")

    # 评分明细
    breakdown = metrics.get("capital_flow_score_breakdown", {})
    if breakdown and "raw_score_before_regime_clamp" in breakdown:
        lines.append("## capital_flow_score 明细")
        lines.append("")
        lines.append("| 子项 | 取值 |")
        lines.append("|---|---|")
        for k, v in breakdown.items():
            lines.append(f"| `{k}` | {v} |")
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 节点工厂
# ---------------------------------------------------------------------------
def create_capital_flow_node():
    """创建资金流 Officer 节点（不依赖 LLM，纯 Python 计算）。"""

    def capital_flow_node(state) -> dict:
        symbol = state.get("company_of_interest", "")
        company_name = state.get("company_name", "") or symbol
        trade_date = state.get("trade_date", "")

        if not symbol or not trade_date:
            logger.warning("capital_flow_node: 缺少 symbol 或 trade_date，跳过")
            return {
                "capital_flow_yaml": "CAPITAL_FLOW:\n  capital_flow_regime: \"数据不足\"\n  capital_flow_score: null\n",
                "capital_flow_report": "# 资金流分析\n\n数据准备阶段缺少必要参数，跳过本节。",
                "capital_flow_metrics": {},
            }

        # 仅对 A 股启用（北向、moneyflow_dc 等接口都仅支持 A 股）
        from tradingagents.dataflows.ticker_utils import is_a_share
        if not is_a_share(symbol):
            logger.info("capital_flow_node: %s 非 A 股，跳过资金流分析", symbol)
            return {
                "capital_flow_yaml": "CAPITAL_FLOW:\n  capital_flow_regime: \"非A股不适用\"\n  capital_flow_score: null\n",
                "capital_flow_report": f"# 资金流综合分析 —— {company_name}（{symbol}）\n\n非 A 股标的，资金流（DDX/DDY/北向）框架不适用。",
                "capital_flow_metrics": {},
            }

        from tradingagents.dataflows.interface import route_to_vendor

        # 1. 资金流主数据（含 moneyflow_df / circulating_market_value_yi / lhb_count_30d）
        cap_data: Optional[dict] = None
        try:
            cap_data = route_to_vendor(
                "get_capital_flow", symbol, trade_date, lookback_days=120
            )
        except Exception as e:
            logger.warning("capital_flow_node: get_capital_flow 全部失败: %s", e)

        # 2. 股东户数（季报）
        holder_df = None
        try:
            holder_df = route_to_vendor("get_holder_number", symbol, lookback_quarters=8)
        except Exception as e:
            logger.info("capital_flow_node: get_holder_number 失败（不影响主流程）: %s", e)

        # 3. 北向持股
        north_df = None
        try:
            north_df = route_to_vendor(
                "get_north_hold", symbol, trade_date, lookback_days=30
            )
        except Exception as e:
            logger.info("capital_flow_node: get_north_hold 失败（数据停滞预期内）: %s", e)

        # 装配
        if cap_data is None:
            cap_data = {
                "moneyflow_df": None,
                "circulating_market_value_yi": None,
                "lhb_count_30d": None,
                "latest_trade_date": trade_date.replace("-", ""),
                "data_source_breakdown": {"moneyflow": "missing", "circ_mv": "missing", "lhb": "missing"},
            }

        metrics = assemble_capital_flow_metrics(
            moneyflow_df=cap_data.get("moneyflow_df"),
            north_df=north_df,
            holder_df=holder_df,
            circulating_market_value_yi=cap_data.get("circulating_market_value_yi"),
            lhb_count_30d=cap_data.get("lhb_count_30d"),
            lhb_inst_net_buy_30d_yi=cap_data.get("lhb_inst_net_buy_30d_yi"),
            latest_trade_date=cap_data.get("latest_trade_date"),
        )

        # 序列化
        # 注意：metrics 内的 capital_flow_votes 需保留为嵌套；YAML 已专门处理
        yaml_str = _to_yaml(metrics)
        md_report = _build_markdown_report(
            symbol=symbol,
            company_name=company_name,
            trade_date=trade_date,
            metrics=metrics,
            data_source_breakdown=cap_data.get("data_source_breakdown", {}),
        )

        return {
            "capital_flow_yaml": yaml_str,
            "capital_flow_report": md_report,
            "capital_flow_metrics": metrics,
        }

    return capital_flow_node
