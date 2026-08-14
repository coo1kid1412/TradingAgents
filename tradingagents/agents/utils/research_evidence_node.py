"""Deterministic research evidence ledger and IC packet compiler.

The node translates existing analyst YAML summaries into auditable evidence
cards. It does not call an LLM, fetch data, or change investment decisions.
"""

from __future__ import annotations

import datetime as dt
import re
from collections.abc import Mapping
from typing import Any

from tradingagents.agents.utils.report_calendar import (
    has_traceable_official_reference,
    normalize_reporting_event,
)
from tradingagents.agents.utils.yaml_summary import extract_yaml_mapping


_DIRECTION = {
    "偏多": "bullish",
    "正面": "bullish",
    "BUY": "bullish",
    "上行": "bullish",
    "强": "bullish",
    "合理": "neutral",
    "低估": "bullish",
    "高估": "bearish",
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

ATTRIBUTION_VARIABLE_POLICY = {
    "short_term": frozenset({"short_term_trend", "entry_timing", "entry_confidence", "position"}),
    "long_term": frozenset({"earnings_outlook_12m", "long_term_rating", "target_price"}),
    "position": frozenset({"position", "entry_timing"}),
    "target_price": frozenset({"target_price"}),
}

CONTRIBUTION_OWNER_POLICY = {
    "thesis": frozenset({"fundamentals", "news", "quant", "sector"}),
    "valuation": frozenset({"fundamentals", "news", "research_manager"}),
    "catalyst": frozenset({"news", "fundamentals", "sector", "sentiment"}),
    "durability": frozenset({"fundamentals", "news", "quant", "sector", "sentiment"}),
}

_CONTRIBUTION_FIELDS = {
    "thesis": ("pillar_thesis", "thesis_evidence_ids", "经营与盈利"),
    "valuation": ("pillar_valuation", "valuation_evidence_ids", "估值"),
    "catalyst": ("pillar_catalyst", "catalyst_evidence_ids", "催化"),
    "durability": ("pillar_durability", "durability_evidence_ids", "持续性"),
}

_PILLAR_EFFECTS = {
    "thesis": {
        "strong": "强支撑", "adequate": "支撑", "mixed": "中性",
        "weak": "评级上限", "invalid": "拒绝方向性评级",
    },
    "valuation": {
        "attractive": "支撑", "fair": "中性", "stretched": "评级上限",
        "invalid": "拒绝方向性评级",
    },
    "catalyst": {
        "strong": "强支撑", "visible": "支撑", "weak": "弱支撑",
        "absent": "中性", "adverse": "反对",
    },
    "durability": {
        "resilient": "强支撑", "acceptable": "支撑", "fragile": "评级上限",
        "broken": "反对",
    },
}

_CONFLICT_FAMILY = {
    "earnings_outlook_12m": "long_term_rating",
    "long_term_rating": "long_term_rating",
    "short_term_trend": "short_term_trend",
    "entry_confidence": "short_term_trend",
}


def _extract_yaml_mapping(report: str, key: str) -> tuple[dict[str, Any] | None, str]:
    """Return a top-level YAML mapping and parse status."""
    return extract_yaml_mapping(report, key)


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


def _impact_direction(value: Any) -> str:
    impact = str(value).strip()
    for prefix, direction in _IMPACT_DIRECTION.items():
        if impact.startswith(prefix):
            return direction
    return "neutral"


def _market_direction(summary: Mapping[str, Any]) -> str:
    """Derive direction from structured trend fields, not model prose."""
    score = 0
    for field in ("trend_weekly", "trend_daily", "momentum"):
        direction = _direction(summary.get(field))
        score += 1 if direction == "bullish" else -1 if direction == "bearish" else 0
    return "bullish" if score > 0 else "bearish" if score < 0 else "neutral"


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
    source_tier: str = "system",
    source_name: str | None = None,
    source_url: str | None = None,
    document_id: str | None = None,
    verification_status: str = "verified",
    decision_eligible: bool = True,
    date_basis: str | None = None,
    event_date: str | None = None,
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
        "source_tier": source_tier,
        "source_name": source_name or source,
        "source_url": source_url,
        "document_id": document_id,
        "verification_status": verification_status,
        "decision_eligible": decision_eligible,
        "date_basis": date_basis,
        "event_date": event_date,
    }


def _market_cards(summary: Mapping[str, Any], trade_date: str) -> list[dict[str, Any]]:
    price_date = _as_date(summary.get("price_data_date"))
    trade_day = _as_date(trade_date)
    price_status = str(summary.get("price_data_status") or "unknown")
    quality = "valid"
    if price_status in {"unknown", "t_minus_1"} or price_date is None:
        quality = "partial"
    elif trade_day and price_date > trade_day:
        quality = "invalid"
    elif trade_day and price_date < trade_day:
        quality = "stale"
    direction = _market_direction(summary)
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
    period = str(summary.get("financial_period") or "unknown").upper()
    match = re.fullmatch(r"(20\d{2})(Q1|H1|Q3|FY)", period)
    period_ends = {"Q1": "03-31", "H1": "06-30", "Q3": "09-30", "FY": "12-31"}
    period_as_of = f"{match.group(1)}-{period_ends[match.group(2)]}" if match else None
    quality = "valid" if period_as_of else "partial"
    flags = summary.get("data_quality_flags")
    if not isinstance(flags, list) or flags:
        quality = "partial"
    trade_day = _as_date(trade_date)
    period_date = _as_date(period_as_of)
    if period_date and trade_day and period_date > trade_day:
        quality = "invalid"
    period_note = f"；财务期间={period}" if period_as_of else "；财务期间=未知"
    return [
        _card(
            "FUND-GROWTH-01", "fundamentals", "earnings_outlook_12m",
            f"营收同比 {summary.get('growth_yoy_revenue', '未知')}%，归母净利同比 "
            f"{summary.get('growth_yoy_profit', '未知')}%{period_note}",
            direction, "12m", period_as_of, "fundamentals_report.SUMMARY",
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
            f"ROE={summary.get('roe', '未知')}%（{summary.get('roe_basis', 'unknown')}）{period_note}",
            direction, "12m", period_as_of, "fundamentals_report.SUMMARY",
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
        trade_day = _as_date(trade_date)
        normalized_event = normalize_reporting_event(
            str(event.get("title") or ""),
            str(event.get("event_date") or "未知"),
            str(event.get("source_date") or "未知"),
            event_date_basis=event.get("event_date_basis"),
            source_tier=event.get("source_tier"),
            verification_status=event.get("verification_status"),
            source_url=event.get("source_url"),
            document_id=event.get("document_id"),
        )
        event_date = normalized_event["event_date"]
        date_basis = normalized_event["date_basis"]
        source_tier = str(event.get("source_tier") or "unknown").strip().lower()
        verification_status = str(event.get("verification_status") or "unverified").strip().lower()
        decision_eligible = (
            has_traceable_official_reference(
                source_tier=source_tier,
                verification_status=verification_status,
                source_url=event.get("source_url"),
                document_id=event.get("document_id"),
            )
        ) or (
            source_tier in {"mainstream", "research"}
            and verification_status in {"verified", "corroborated"}
        )
        quality = "valid" if source_date and event_date not in (None, "", "未知", "null") else "partial"
        if not decision_eligible:
            quality = "partial"
        if source_date and trade_day and source_date > trade_day:
            quality = "invalid"
        impact = str(event.get("impact") or "0")
        date_labels = {
            "exact_reservation": f"交易所预约披露日={event_date}",
            "legal_deadline": f"法定最晚披露日={event_date}（非预约日）",
            "publication_date": f"公告发布日期={event_date}",
            "reported": f"事件日期={event_date}",
        }
        date_claim = date_labels.get(date_basis, f"日期={event_date}")
        cards.append(_card(
            f"NEWS-CAT-{len(cards) + 1:02d}", "news", "long_term_rating",
            f"{event.get('title', '未命名事件')}；{date_claim}；"
            f"影响={impact}；已定价={event.get('priced_in_p', '未知')}%",
            _impact_direction(impact), str(event.get("horizon") or "event"),
            source_date.isoformat() if source_date else None, "news_report.SUMMARY.key_events",
            confidence={"高": "high", "中": "medium", "低": "low"}.get(str(event.get("credibility")), "medium"),
            falsifier=(
                f"在{date_claim}前后核对正式披露或反向公告"
                if date_basis in {"legal_deadline", "exact_reservation"}
                else "后续出现反向公告或事件事实被更正"
            ),
            quality_status=quality,
            source_tier=source_tier,
            source_name=str(event.get("source_name") or "未知来源"),
            source_url=event.get("source_url"),
            document_id=event.get("document_id"),
            verification_status=verification_status,
            decision_eligible=decision_eligible,
            date_basis=date_basis,
            event_date=event_date,
        ))
    target = summary.get("research_consensus_target_price")
    if target not in (None, "null", "None", "不适用"):
        target_date = _as_date(summary.get("research_consensus_as_of"))
        trade_day = _as_date(trade_date)
        target_eligible = target_date is not None and not (
            trade_day is not None and target_date > trade_day
        )
        target_quality = "partial" if target_eligible else (
            "invalid" if target_date is not None else "partial"
        )
        cards.append(_card(
            "NEWS-TP-01", "news", "target_price", f"卖方一致目标价 {target}",
            _direction(summary.get("research_consensus_rating")), "12m",
            target_date.isoformat() if target_date else None,
            "news_report.SUMMARY", falsifier="一致预期目标价发生下修",
            quality_status=target_quality, source_tier="research",
            verification_status="corroborated" if target_eligible else "unverified",
            decision_eligible=target_eligible,
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
    price_date = _as_date(summary.get("price_data_date"))
    trade_day = _as_date(trade_date)
    price_status = str(summary.get("price_data_status") or "unknown")
    if numeric is None:
        quality = "invalid"
    elif trade_day and price_date and price_date > trade_day:
        quality = "invalid"
    elif price_status in {"unknown", "t_minus_1"} or price_date is None:
        quality = "partial"
    elif trade_day and price_date < trade_day:
        quality = "stale"
    else:
        quality = "valid"
    coverage = summary.get("coverage")
    missing_factors = coverage.get("missing") if isinstance(coverage, Mapping) else []
    if isinstance(missing_factors, str):
        missing_factors = [missing_factors]
    missing_factors = [str(item) for item in (missing_factors or []) if item not in (None, "")]
    if missing_factors and quality == "valid":
        quality = "partial"
    coverage_note = f"；缺失因子={','.join(missing_factors)}" if missing_factors else ""
    return [_card(
        "QUANT-COMP-01", "quant", "long_term_rating", f"量化综合分={score}{coverage_note}",
        direction, "1-12m", price_date.isoformat() if price_date else None,
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


def _stock_structure_cards(profile: str, trade_date: str) -> list[dict[str, Any]]:
    match = re.search(r"(?m)^SYS_SHORT_TERM_STRUCTURE:\s*class=([a-z_]+)", profile or "")
    if not match:
        return []
    structure = match.group(1)
    direction = {
        "trend_pullback": "bullish",
        "breakout_ready": "bullish",
        "healthy_trend": "bullish",
        "weak_rebound_in_downtrend": "bearish",
        "exhaustion": "bearish",
        "broken": "bearish",
        "neutral": "neutral",
        "insufficient_data": "neutral",
    }.get(structure, "neutral")
    quality = "valid" if structure in {
        "trend_pullback", "breakout_ready", "healthy_trend", "weak_rebound_in_downtrend",
        "exhaustion", "broken", "neutral",
    } else "partial"
    return [_card(
        "MKT-STRUCT-01", "market", "entry_timing", f"确定性短线结构={structure}",
        direction, "3d", trade_date, "stock_profile.SYS_SHORT_TERM_STRUCTURE",
        falsifier="短线结构分类在下一交易日发生变化", quality_status=quality,
    )]


def _risk_cards(snapshot: Mapping[str, Any], trade_date: str) -> list[dict[str, Any]]:
    if not snapshot:
        return [_card(
            "RISK-GATE-01", "risk", "position",
            "市场风险快照缺失，方向未知；执行数据门按 WAIT / 新仓上限 0% 处理",
            "neutral", "intraday", trade_date, "market_risk_snapshot",
            falsifier="取得有效市场风险快照且入场门不再为 WAIT",
            quality_status="invalid", verification_status="missing",
            decision_eligible=False,
        )]
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
        if summary is None:
            coverage[domain] = status
            warnings.append(f"{label}摘要{status}，该领域证据未进入归因")
            continue
        try:
            domain_cards = builder(summary, trade_date)
        except Exception as exc:  # Evidence attribution must never abort stock analysis.
            coverage[domain] = "invalid"
            warnings.append(f"{label}摘要字段类型无效，已跳过该领域证据：{type(exc).__name__}")
            continue
        if status == "recovered":
            for card in domain_cards:
                if card.get("quality_status") == "valid":
                    card["quality_status"] = "partial"
        cards.extend(domain_cards)
        qualities = {str(card["quality_status"]) for card in domain_cards}
        domain_status = next(
            (quality for quality in ("invalid", "stale", "partial") if quality in qualities),
            "valid" if domain_cards else "missing",
        )
        if status == "recovered":
            domain_status = "partial"
            warnings.append(f"{label}摘要经受限语法自动修复后解析，已标记为部分覆盖")
        coverage[domain] = domain_status

    structure_cards = _stock_structure_cards(str(state.get("stock_profile") or ""), trade_date)
    coverage["structure"] = structure_cards[0]["quality_status"] if structure_cards else "missing"
    cards.extend(structure_cards)
    if not structure_cards:
        warnings.append("确定性短线结构缺失，入场时机归因不完整")

    sector_cards = _sector_cards(str(state.get("sector_comparison") or ""), trade_date)
    coverage["sector"] = "valid" if sector_cards else "missing"
    cards.extend(sector_cards)
    if not sector_cards:
        warnings.append("板块相对强弱缺失，该领域证据未进入归因")

    risk_cards = _risk_cards(state.get("market_risk_snapshot") or {}, trade_date)
    coverage["risk"] = risk_cards[0]["quality_status"]
    cards.extend(risk_cards)
    if not state.get("market_risk_snapshot"):
        warnings.append("市场风险快照缺失，已按 WAIT / 新仓上限 0% 进行归因")

    ids = [card["claim_id"] for card in cards]
    if len(ids) != len(set(ids)):
        raise ValueError("research evidence contains duplicate claim IDs")
    if any(status in {"missing", "invalid"} for status in coverage.values()):
        analysis_status = "incomplete"
    elif any(status in {"partial", "stale", "recovered"} for status in coverage.values()):
        analysis_status = "partial"
    else:
        analysis_status = "complete"
    return {
        "schema_version": "research-evidence-v1",
        "analysis_status": analysis_status,
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


def _summary_value(content: str, key: str) -> str | None:
    matches = re.findall(rf"(?m)^\s*{re.escape(key)}:\s*([^#\r\n]+?)\s*$", content or "")
    if not matches:
        return None
    value = matches[-1].strip().strip('"\'')
    return None if value.lower() in {"null", "none", ""} else value


def _reference_ids(content: str, key: str) -> list[str]:
    raw = _summary_value(content, key)
    if not raw:
        return []
    return [item.strip() for item in re.split(r"[|,，]", raw) if item.strip()]


def compile_decision_contribution_ledger(
    rm_content: str,
    evidence_ledger: Mapping[str, Any],
) -> dict[str, Any]:
    """Compile RM pillar references into one auditable contribution per claim."""
    by_id = {
        str(card.get("claim_id")): card
        for card in evidence_ledger.get("cards") or []
        if card.get("claim_id")
    }
    seen: set[str] = set()
    items: list[dict[str, Any]] = []
    pillars: dict[str, dict[str, Any]] = {}
    warnings: list[str] = []

    for dimension, (state_field, evidence_field, label) in _CONTRIBUTION_FIELDS.items():
        state = (_summary_value(rm_content, state_field) or "数据不足").lower()
        effect = _PILLAR_EFFECTS.get(dimension, {}).get(state, "数据不足")
        references = _reference_ids(rm_content, evidence_field)
        accepted_ids: list[str] = []
        rejected_ids: list[str] = []

        for claim_id in references:
            if claim_id in seen:
                warnings.append(f"{claim_id} 已归入其他支柱，忽略重复引用")
                continue
            seen.add(claim_id)
            card = by_id.get(claim_id)
            rejection_reason = None
            if card is None:
                rejection_reason = "证据 ID 不存在"
            elif str(card.get("quality_status") or "invalid") == "invalid":
                rejection_reason = "证据质量无效"
            elif not bool(card.get("decision_eligible", True)):
                rejection_reason = "证据不具备硬决策资格"
            elif str(card.get("owner") or "") not in CONTRIBUTION_OWNER_POLICY[dimension]:
                rejection_reason = "角色无权支撑该研究支柱"

            if rejection_reason:
                rejected_ids.append(claim_id)
            else:
                accepted_ids.append(claim_id)
            quality = str(card.get("quality_status") or "invalid") if card else "missing"
            confidence = str(card.get("confidence") or "low") if card else "low"
            items.append({
                "claim_id": claim_id,
                "role": str(card.get("owner") or "unknown") if card else "unknown",
                "decision_dimension": dimension,
                "conclusion": str(card.get("claim") or "未找到证据") if card else "未找到证据",
                "accepted_effect": "rejected" if rejection_reason else effect,
                "direction": str(card.get("direction") or "neutral") if card else "neutral",
                "materiality": "high" if confidence == "high" else "medium" if confidence == "medium" else "low",
                "confidence": confidence,
                "claim_ids": [claim_id],
                "quality_status": quality,
                "rejection_reason": rejection_reason,
            })

        pillars[dimension] = {
            "label": label,
            "state": state,
            "accepted_effect": effect,
            "accepted_claim_ids": accepted_ids,
            "rejected_claim_ids": rejected_ids,
        }

    return {
        "schema_version": "decision-contribution-v1",
        "research_rating": _summary_value(rm_content, "research_rating")
        or _summary_value(rm_content, "rm_rating")
        or "数据不足",
        "pillars": pillars,
        "items": items,
        "warnings": warnings,
    }


def render_ic_contribution_summary(
    rm_content: str,
    contribution_ledger: Mapping[str, Any],
) -> str:
    """Render four pillars and selected/rejected analyst contributions."""
    rating = str(
        contribution_ledger.get("research_rating")
        or _summary_value(rm_content, "research_rating")
        or "数据不足"
    )
    lines = [
        "## 四支柱投委会结论", "",
        f"一年期研究评级：**{rating}**", "",
        "| 支柱 | 状态 | 对评级作用 | 已采纳证据 |",
        "|---|---|---|---|",
    ]
    for dimension in _CONTRIBUTION_FIELDS:
        pillar = (contribution_ledger.get("pillars") or {}).get(dimension, {})
        evidence = ", ".join(pillar.get("accepted_claim_ids") or []) or "-"
        lines.append(
            f"| {pillar.get('label', dimension)} | {pillar.get('state', '数据不足')} | "
            f"{pillar.get('accepted_effect', '数据不足')} | {evidence} |"
        )

    accepted = [
        item for item in contribution_ledger.get("items") or []
        if not item.get("rejection_reason")
    ]
    if accepted:
        lines.extend(["", "### 关键 Agent 贡献", ""])
        for item in accepted[:4]:
            lines.append(
                f"- **{item['role']} / {item['claim_id']}**：{item['conclusion']}"
                f"（作用：{item['accepted_effect']}）"
            )

    rejected = [
        item for item in contribution_ledger.get("items") or []
        if item.get("rejection_reason")
    ]
    if rejected:
        lines.extend(["", "### 未采纳的关键观点", ""])
        for item in rejected[:4]:
            lines.append(
                f"- `{item['claim_id']}`：{item['rejection_reason']}"
            )
    return "\n".join(lines)


def _display_number(value: str | None) -> str:
    if value is None:
        return "数据不足"
    try:
        number = float(value)
    except ValueError:
        return value
    return str(int(number)) if number.is_integer() else f"{number:.2f}".rstrip("0").rstrip(".")


def _display_position(content: str) -> str:
    low = _summary_value(content, "pm_size_low_pct")
    high = _summary_value(content, "pm_size_high_pct")
    if low is None and high is None:
        return "数据不足"
    low_text = _display_number(low or "0")
    high_text = _display_number(high or low or "0")
    return f"{low_text}%" if low_text == high_text else f"{low_text}-{high_text}%"


def _validate_references(
    ids: list[str], ledger: Mapping[str, Any], policy_key: str,
    *, expected_direction: str | None = None,
) -> tuple[str, str]:
    if not ids:
        return "-", "缺失：PM 未完成证据归因"
    by_id = {str(card.get("claim_id")): card for card in ledger.get("cards") or []}
    missing = [claim_id for claim_id in ids if claim_id not in by_id]
    allowed = ATTRIBUTION_OWNER_POLICY[policy_key]
    allowed_variables = ATTRIBUTION_VARIABLE_POLICY[policy_key]
    invalid = [
        claim_id for claim_id in ids
        if claim_id in by_id and str(by_id[claim_id].get("quality_status")) == "invalid"
    ]
    ineligible = [
        claim_id for claim_id in ids
        if claim_id in by_id
        and claim_id not in invalid
        and not bool(by_id[claim_id].get("decision_eligible", True))
    ]
    unauthorized = [
        claim_id for claim_id in ids
        if claim_id in by_id
        and claim_id not in invalid
        and claim_id not in ineligible
        and str(by_id[claim_id].get("owner")) not in allowed
    ]
    wrong_variable = [
        claim_id for claim_id in ids
        if claim_id in by_id
        and claim_id not in invalid
        and claim_id not in unauthorized
        and claim_id not in ineligible
        and str(by_id[claim_id].get("decision_variable")) not in allowed_variables
    ]
    accepted = [
        claim_id for claim_id in ids
        if claim_id in by_id
        and claim_id not in invalid
        and claim_id not in unauthorized
        and claim_id not in ineligible
        and claim_id not in wrong_variable
    ]
    issues = []
    if missing:
        issues.append(f"剔除不存在的证据 {', '.join(missing)}")
    if invalid:
        issues.append(f"剔除无效证据 {', '.join(invalid)}")
    if ineligible:
        issues.append(f"剔除不可用于硬决策的证据 {', '.join(ineligible)}")
    if unauthorized:
        issues.append(f"剔除权限不匹配的证据 {', '.join(unauthorized)}")
    if wrong_variable:
        issues.append(f"剔除决策变量不匹配的证据 {', '.join(wrong_variable)}")
    if not accepted:
        return "-", f"缺失：{'；'.join(issues)}"
    if expected_direction:
        trend_directions = {
            str(by_id[claim_id].get("direction") or "neutral")
            for claim_id in accepted
            if str(by_id[claim_id].get("decision_variable")) == "short_term_trend"
        }
        if expected_direction not in trend_directions:
            opposite = {"bullish": "bearish", "bearish": "bullish"}.get(expected_direction)
            issues.append(
                "证据方向与结论不一致"
                if opposite in trend_directions
                else "缺少同向趋势证据"
            )
        elif len(trend_directions) > 1:
            issues.append("趋势证据方向冲突")
    qualities = [str(by_id[claim_id].get("quality_status") or "invalid") for claim_id in accepted]
    if "stale" in qualities:
        issues.append("有效证据时点过期")
    elif "partial" in qualities:
        issues.append("证据不完整")
    return ", ".join(accepted), f"部分：{'；'.join(issues)}" if issues else "完整"


def render_decision_attribution(
    pm_content: str,
    timing: Mapping[str, Any],
    ledger: Mapping[str, Any],
    *,
    rm_content: str = "",
) -> str:
    """Render four PM decisions with validated evidence ownership and quality."""
    rating = _summary_value(pm_content, "pm_rating") or "数据不足"
    short_term = _summary_value(pm_content, "short_term_trend") or "数据不足"
    target_low = _display_number(_summary_value(rm_content, "target_price_low"))
    target_mid = _display_number(_summary_value(rm_content, "target_price_mid"))
    target_high = _display_number(_summary_value(rm_content, "target_price_high"))
    target_display = (
        f"{target_low}-{target_high}（中位 {target_mid}）"
        if "数据不足" not in {target_low, target_mid, target_high}
        else "数据不足"
    )
    rows = (
        (
            "未来三日", short_term, "市场分析/风险", "short_term_evidence_ids", "short_term",
        ),
        (
            "一年期评级", rating, "基本面/RM", "long_term_evidence_ids", "long_term",
        ),
        (
            "新建仓位", _display_position(pm_content), "风险/PM", "position_evidence_ids", "position",
        ),
        (
            "一年期目标价", target_display, "基本面/RM", "target_price_evidence_ids", "target_price",
        ),
    )
    lines = [
        "## 为什么这样决定", "",
        "| 决策项 | 最终结论 | 主责团队 | 关键证据 | 归因状态 |",
        "|---|---|---|---|---|",
    ]
    for label, conclusion, team, field, policy in rows:
        ids = _reference_ids(pm_content, field)
        deterministic_ids = {
            "short_term_evidence_ids": ("MKT-STRUCT-01", "RISK-GATE-01"),
            "position_evidence_ids": ("RISK-GATE-01",),
        }.get(field, ())
        available = {str(card.get("claim_id")) for card in ledger.get("cards") or []}
        ids = [claim_id for claim_id in deterministic_ids if claim_id in available] + ids
        ids = list(dict.fromkeys(ids))
        expected_direction = None
        if policy == "short_term":
            if "上涨" in conclusion:
                expected_direction = "bullish"
            elif "下跌" in conclusion:
                expected_direction = "bearish"
            elif "震荡" in conclusion:
                expected_direction = "neutral"
        evidence, status = _validate_references(
            ids, ledger, policy, expected_direction=expected_direction,
        )
        lines.append(f"| {label} | {conclusion} | {team} | {evidence} | {status} |")
    lines.extend([
        "",
        "> 归因状态由代码核验。它只说明证据是否可追踪、时效可用且角色有权支撑该决策，"
        "不自动改写评级或仓位。",
    ])
    return "\n".join(lines)


def create_research_evidence_node():
    """Return a LangGraph-compatible deterministic evidence compiler node."""

    def research_evidence_node(state: Mapping[str, Any]) -> dict[str, Any]:
        try:
            ledger = compile_research_evidence(state)
            packet = render_ic_packet(
                ledger,
                ticker=str(state.get("company_of_interest") or ""),
                company_name=str(state.get("company_name") or ""),
                trade_date=str(state.get("trade_date") or ""),
            )
        except Exception as exc:  # Never let reporting attribution abort analysis.
            warning = f"研究证据编译异常，已降级为空账本：{type(exc).__name__}"
            ledger = {
                "schema_version": "research-evidence-v1",
                "cards": [],
                "conflicts": [],
                "warnings": [warning],
                "coverage": {},
            }
            packet = f"# IC 决策包\n\n- ⚠️ {warning}\n"
        return {"research_evidence_ledger": ledger, "ic_packet": packet}

    return research_evidence_node
