"""Deterministic research evidence ledger and IC packet compiler.

The node translates existing analyst YAML summaries into auditable evidence
cards. It does not call an LLM, fetch data, or change investment decisions.
"""

from __future__ import annotations

import datetime as dt
import re
from collections.abc import Mapping
from typing import Any

import yaml


_DIRECTION = {
    "偏多": "bullish",
    "正面": "bullish",
    "BUY": "bullish",
    "上行": "bullish",
    "强": "bullish",
    "合理": "neutral",
    "中性": "neutral",
    "分歧": "neutral",
    "HOLD": "neutral",
    "震荡": "neutral",
    "偏空": "bearish",
    "负面": "bearish",
    "SELL": "bearish",
    "下行": "bearish",
    "弱": "bearish",
}

_IMPACT_DIRECTION = {
    "+大": "bullish",
    "+中": "bullish",
    "+小": "bullish",
    "0": "neutral",
    "-小": "bearish",
    "-中": "bearish",
    "-大": "bearish",
}

ATTRIBUTION_OWNER_POLICY = {
    "short_term": frozenset({"market", "news", "sentiment", "quant", "sector", "risk"}),
    "long_term": frozenset({"fundamentals", "news", "quant", "sector", "research_manager"}),
    "position": frozenset({"risk", "market", "portfolio_manager"}),
    "target_price": frozenset({"fundamentals", "news", "research_manager"}),
}

_CONFLICT_FAMILY = {
    "earnings_outlook_12m": "long_term_rating",
    "long_term_rating": "long_term_rating",
    "short_term_trend": "short_term_trend",
    "entry_confidence": "short_term_trend",
}


def _extract_yaml_mapping(report: str, key: str) -> tuple[dict[str, Any] | None, str]:
    """Return a top-level YAML mapping and parse status."""
    if not report or key not in report:
        return None, "missing"
    saw_candidate = False
    for block in reversed(re.findall(r"```yaml\s*\n(.*?)\n```", report, re.DOTALL)):
        if not re.search(rf"(?m)^\s*{re.escape(key)}\s*:", block):
            continue
        saw_candidate = True
        try:
            parsed = yaml.safe_load(block)
        except yaml.YAMLError:
            continue
        value = parsed.get(key) if isinstance(parsed, dict) else None
        if isinstance(value, dict):
            return value, "valid"
    return None, "invalid" if saw_candidate else "missing"


def _as_date(value: Any) -> dt.date | None:
    if value in (None, "", "未知", "null", "None"):
        return None
    try:
        return dt.date.fromisoformat(str(value)[:10])
    except ValueError:
        return None


def _direction(value: Any, fallback: Any = None) -> str:
    return _DIRECTION.get(str(value), _DIRECTION.get(str(fallback), "neutral"))


def _confidence(value: Any) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "medium"
    if numeric >= 4:
        return "high"
    if numeric <= 2:
        return "low"
    return "medium"


def _card(
    claim_id: str,
    owner: str,
    decision_variable: str,
    claim: str,
    direction: str,
    horizon: str,
    as_of: str | None,
    source: str,
    *,
    confidence: str = "medium",
    falsifier: str = "原始数据或前提发生反向变化",
    quality_status: str = "valid",
) -> dict[str, Any]:
    return {
        "claim_id": claim_id,
        "owner": owner,
        "decision_variable": decision_variable,
        "claim": claim,
        "direction": direction,
        "horizon": horizon,
        "confidence": confidence,
        "as_of": as_of,
        "source": source,
        "falsifier": falsifier,
        "quality_status": quality_status,
    }


def _market_cards(summary: Mapping[str, Any], trade_date: str) -> list[dict[str, Any]]:
    price_date = _as_date(summary.get("price_data_date"))
    trade_day = _as_date(trade_date)
    price_status = str(summary.get("price_data_status") or "unknown")
    quality = "valid"
    if price_status in {"unknown", "t_minus_1"} or price_date is None:
        quality = "partial"
    elif trade_day and price_date < trade_day:
        quality = "stale"
    direction = _direction(summary.get("data_implied_direction"), summary.get("trend_daily"))
    as_of = price_date.isoformat() if price_date else None
    cards = [
        _card(
            "MKT-TREND-01", "market", "short_term_trend",
            f"周线{summary.get('trend_weekly', '未知')}、日线{summary.get('trend_daily', '未知')}，"
            f"动量{summary.get('momentum', '未知')}",
            direction, "3d", as_of, "market_report.SUMMARY",
            confidence=_confidence(summary.get("confidence")),
            falsifier=f"跌破关键支撑 {summary.get('key_support', '未知')} 或价格数据过期",
            quality_status=quality,
        ),
        _card(
            "MKT-LEVEL-01", "market", "entry_timing",
            f"关键支撑 {summary.get('key_support', '未知')}，关键阻力 {summary.get('key_resistance', '未知')}",
            "neutral", "3d", as_of, "market_report.SUMMARY",
            falsifier="支撑或阻力被有效突破",
            quality_status=quality,
        ),
    ]
    flow = summary.get("capital_flow_regime")
    if flow not in (None, "数据不足", "非A股不适用", "不适用"):
        cards.append(_card(
            "MKT-FLOW-01", "market", "short_term_trend", f"资金面状态为{flow}",
            "bullish" if flow == "强势" else "bearish" if flow == "恶化" else "neutral",
            "3d", as_of, "market_report.SUMMARY", quality_status=quality,
        ))
    return cards


def _fundamental_cards(summary: Mapping[str, Any], trade_date: str) -> list[dict[str, Any]]:
    direction = _direction(summary.get("data_implied_direction"), summary.get("rating"))
    # Existing SUMMARY does not carry the financial statement period, so these
    # cards are intentionally partial until that upstream schema is extended.
    quality = "partial"
    return [
        _card(
            "FUND-GROWTH-01", "fundamentals", "earnings_outlook_12m",
            f"营收同比 {summary.get('growth_yoy_revenue', '未知')}%，归母净利同比 "
            f"{summary.get('growth_yoy_profit', '未知')}%",
            direction, "12m", trade_date, "fundamentals_report.SUMMARY",
            falsifier="后续财报显示收入或利润增速显著反转",
            quality_status=quality,
        ),
        _card(
            "FUND-VAL-01", "fundamentals", "target_price",
            f"PE(TTM)={summary.get('pe_ttm', '未知')}，估值区间={summary.get('pe_zone', '未知')}，"
            f"行业中位数={summary.get('pe_industry_median', '未知')}",
            _direction(summary.get("pe_zone")), "12m", trade_date,
            "fundamentals_report.SUMMARY",
            falsifier="盈利预测或可比估值中枢发生显著变化",
            quality_status=quality,
        ),
        _card(
            "FUND-QUALITY-01", "fundamentals", "long_term_rating",
            f"盈利质量={summary.get('earnings_quality', '未知')}，治理={summary.get('governance_score', '未知')}，"
            f"ROE={summary.get('roe', '未知')}%",
            direction, "12m", trade_date, "fundamentals_report.SUMMARY",
            falsifier="现金流、应收或治理出现新增红旗",
            quality_status=quality,
        ),
    ]


def _news_cards(summary: Mapping[str, Any], trade_date: str) -> list[dict[str, Any]]:
    cards: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for event in summary.get("key_events") or []:
        if not isinstance(event, dict):
            continue
        identity = (
            str(event.get("title") or ""),
            str(event.get("source_date") or ""),
            str(event.get("event_date") or ""),
        )
        if identity in seen:
            continue
        seen.add(identity)
        source_date = _as_date(event.get("source_date"))
        quality = "valid" if source_date else "partial"
        impact = str(event.get("impact") or "0")
        cards.append(_card(
            f"NEWS-CAT-{len(cards) + 1:02d}", "news", "long_term_rating",
            f"{event.get('title', '未命名事件')}；影响={impact}；已定价={event.get('priced_in_p', '未知')}%",
            _IMPACT_DIRECTION.get(impact, "neutral"), str(event.get("horizon") or "event"),
            source_date.isoformat() if source_date else None, "news_report.SUMMARY.key_events",
            confidence={"高": "high", "中": "medium", "低": "low"}.get(str(event.get("credibility")), "medium"),
            falsifier=f"事件未在 {event.get('event_date', '约定窗口')} 落地或出现反向公告",
            quality_status=quality,
        ))
    target = summary.get("research_consensus_target_price")
    if target not in (None, "null", "None", "不适用"):
        cards.append(_card(
            "NEWS-TP-01", "news", "target_price", f"卖方一致目标价 {target}",
            _direction(summary.get("research_consensus_rating")), "12m", trade_date,
            "news_report.SUMMARY", falsifier="一致预期目标价发生下修", quality_status="partial",
        ))
    return cards


def _sentiment_cards(summary: Mapping[str, Any], trade_date: str) -> list[dict[str, Any]]:
    return [_card(
        "SENT-CROWD-01", "sentiment", "entry_confidence",
        f"情绪={summary.get('net_sentiment', '未知')}，KOL={summary.get('kol_consensus', '未知')}，"
        f"7日趋势={summary.get('sentiment_trend_7d', '未知')}",
        _direction(summary.get("data_implied_direction"), summary.get("net_sentiment")),
        "3d", trade_date, "sentiment_report.SUMMARY",
        falsifier="样本覆盖或多空比例出现显著反转", quality_status="partial",
    )]


def _quant_cards(summary: Mapping[str, Any], trade_date: str) -> list[dict[str, Any]]:
    score = summary.get("composite")
    try:
        numeric = float(score)
    except (TypeError, ValueError):
        numeric = None
    direction = "bullish" if numeric is not None and numeric >= 65 else (
        "bearish" if numeric is not None and numeric < 35 else "neutral"
    )
    quality = "valid" if numeric is not None else "invalid"
    return [_card(
        "QUANT-COMP-01", "quant", "long_term_rating", f"量化综合分={score}",
        direction, "1-12m", str(summary.get("price_data_date") or trade_date),
        "quant_score.QUANT_SCORE", falsifier="综合分跨越 35/65 分界",
        quality_status=quality,
    )]


def _sector_cards(report: str, trade_date: str) -> list[dict[str, Any]]:
    match = re.search(r"vs\s+沪深300（30d）.*?RS\s*=\s*([+-]?[0-9.]+)%", report or "")
    if not match:
        return []
    rs_value = float(match.group(1))
    return [_card(
        "SECTOR-RS-01", "sector", "long_term_rating", f"30日相对沪深300强弱={rs_value:+.1f}%",
        "bullish" if rs_value > 0 else "bearish" if rs_value < 0 else "neutral",
        "1-3m", trade_date, "sector_comparison", falsifier="30日相对强弱方向反转",
    )]


def _risk_cards(snapshot: Mapping[str, Any], trade_date: str) -> list[dict[str, Any]]:
    if not snapshot:
        return []
    status = str(snapshot.get("data_status") or "missing")
    quality = "valid" if status == "fresh" else "stale" if status == "stale" else "partial"
    gate = str(snapshot.get("entry_gate") or "WAIT")
    return [_card(
        "RISK-GATE-01", "risk", "position",
        f"市场风险={snapshot.get('risk_level', '数据不足')}，入场门={gate}，"
        f"新仓上限={snapshot.get('position_cap_pct', 0)}%",
        "bullish" if gate == "OPEN" else "neutral" if gate == "CONDITIONAL" else "bearish",
        "intraday", str(snapshot.get("as_of_date") or trade_date), "market_risk_snapshot",
        falsifier="市场风险灯号或数据状态发生变化", quality_status=quality,
    )]


def _detect_conflicts(cards: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for card in cards:
        if card["quality_status"] == "invalid" or card["direction"] == "neutral":
            continue
        family = _CONFLICT_FAMILY.get(str(card["decision_variable"]))
        if family:
            grouped.setdefault(family, []).append(card)
    conflicts = []
    for variable, items in grouped.items():
        directions = {item["direction"] for item in items}
        if directions == {"bullish", "bearish"}:
            conflicts.append({
                "conflict_id": f"CONFLICT-{len(conflicts) + 1:02d}",
                "decision_variable": variable,
                "directions": sorted(directions),
                "evidence_ids": [item["claim_id"] for item in items],
            })
    return conflicts


def compile_research_evidence(state: Mapping[str, Any]) -> dict[str, Any]:
    """Compile state reports into an auditable, JSON-serializable ledger."""
    trade_date = str(state.get("trade_date") or "")
    cards: list[dict[str, Any]] = []
    warnings: list[str] = []
    coverage: dict[str, str] = {}
    specs = (
        ("market", "市场", state.get("market_report", ""), "SUMMARY", _market_cards),
        ("fundamentals", "基本面", state.get("fundamentals_report", ""), "SUMMARY", _fundamental_cards),
        ("news", "新闻", state.get("news_report", ""), "SUMMARY", _news_cards),
        ("sentiment", "舆情", state.get("sentiment_report", ""), "SUMMARY", _sentiment_cards),
        ("quant", "量化", state.get("quant_score", ""), "QUANT_SCORE", _quant_cards),
    )
    for domain, label, report, key, builder in specs:
        summary, status = _extract_yaml_mapping(str(report or ""), key)
        coverage[domain] = status
        if summary is None:
            warnings.append(f"{label}摘要{status}，该领域证据未进入归因")
            continue
        cards.extend(builder(summary, trade_date))

    sector_cards = _sector_cards(str(state.get("sector_comparison") or ""), trade_date)
    coverage["sector"] = "valid" if sector_cards else "missing"
    cards.extend(sector_cards)
    if not sector_cards:
        warnings.append("板块相对强弱缺失，该领域证据未进入归因")

    risk_cards = _risk_cards(state.get("market_risk_snapshot") or {}, trade_date)
    coverage["risk"] = risk_cards[0]["quality_status"] if risk_cards else "missing"
    cards.extend(risk_cards)
    if not risk_cards:
        warnings.append("市场风险快照缺失，仓位归因不完整")

    ids = [card["claim_id"] for card in cards]
    if len(ids) != len(set(ids)):
        raise ValueError("research evidence contains duplicate claim IDs")
    return {
        "schema_version": "research-evidence-v1",
        "cards": cards,
        "conflicts": _detect_conflicts(cards),
        "warnings": warnings,
        "coverage": coverage,
    }


def _quality_label(value: str) -> str:
    return {"valid": "完整", "partial": "部分", "stale": "过期", "invalid": "无效"}.get(value, "缺失")


def render_ic_packet(
    ledger: Mapping[str, Any], *, ticker: str, company_name: str, trade_date: str
) -> str:
    """Render the evidence ledger as a compact investment-committee packet."""
    cards = list(ledger.get("cards") or [])
    sections = [
        ("未来三日与入场", {"short_term_trend", "entry_timing", "entry_confidence", "position"}),
        ("一年期盈利与估值", {"earnings_outlook_12m", "long_term_rating", "target_price"}),
        ("量化、板块与风险交叉验证", {"position"}),
    ]
    lines = [
        f"# IC 决策包：{ticker} {company_name}", "", f"**决策日期**：{trade_date}", "",
        "> 本报告由代码从分析师 YAML 摘要编译，不新增 LLM 调用。模型不得编造证据 ID。", "",
        "## 数据完整度", "", "| 领域 | 状态 |", "|---|---|",
    ]
    for domain, status in (ledger.get("coverage") or {}).items():
        lines.append(f"| {domain} | {_quality_label(str(status))} |")
    warnings = list(ledger.get("warnings") or [])
    if warnings:
        lines.extend(["", *[f"- ⚠️ {warning}" for warning in warnings]])

    rendered_ids: set[str] = set()
    for title, variables in sections:
        lines.extend(["", f"## {title}", "", "| 证据 ID | 结论 | 方向 | 质量 |", "|---|---|---|---|"])
        selected = [card for card in cards if card.get("decision_variable") in variables]
        if title == "量化、板块与风险交叉验证":
            selected = [card for card in cards if card.get("owner") in {"quant", "sector", "risk"}]
        for card in selected:
            rendered_ids.add(str(card["claim_id"]))
            lines.append(
                f"| {card['claim_id']} | {card['claim']} | {card['direction']} | "
                f"{_quality_label(str(card['quality_status']))} |"
            )
        if not selected:
            lines.append("| - | 数据不足 | neutral | 缺失 |")

    lines.extend(["", "## 冲突清单", ""])
    conflicts = list(ledger.get("conflicts") or [])
    if conflicts:
        for conflict in conflicts:
            lines.append(
                f"- **{conflict['conflict_id']}**：{conflict['decision_variable']} 存在方向冲突，"
                f"证据 {', '.join(conflict['evidence_ids'])}；由 RM/PM 判断，不由代码裁决。"
            )
    else:
        lines.append("- 未检测到可用证据之间的直接方向冲突。")

    lines.extend(["", "## 全部证据 ID 索引", ""])
    for card in cards:
        if card["claim_id"] not in rendered_ids:
            lines.append(f"- `{card['claim_id']}`：{card['claim']}")
    lines.append("- 引用规则：只能使用本包出现的证据 ID；模型不得编造证据 ID。")
    return "\n".join(lines).rstrip() + "\n"


def create_research_evidence_node():
    """Return a LangGraph-compatible deterministic evidence compiler node."""

    def research_evidence_node(state: Mapping[str, Any]) -> dict[str, Any]:
        ledger = compile_research_evidence(state)
        packet = render_ic_packet(
            ledger,
            ticker=str(state.get("company_of_interest") or ""),
            company_name=str(state.get("company_name") or ""),
            trade_date=str(state.get("trade_date") or ""),
        )
        return {"research_evidence_ledger": ledger, "ic_packet": packet}

    return research_evidence_node
