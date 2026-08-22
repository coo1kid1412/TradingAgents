"""Shared user-facing and audit report writers."""

from __future__ import annotations

import datetime as dt
import re
from pathlib import Path
from typing import Iterable


_TABLE_SEPARATOR_RE = re.compile(r"^:?-{3,}:?$")
_MARKDOWN_HEADING_RE = re.compile(r"(?m)^#{1,6}\s+")
_CORE_KEYWORDS = {
    "news": ("公告", "预告", "中报", "季报", "年报", "股东会", "政策", "规则", "FCC", "事件", "落地", "验证节点"),
    "valuation": ("PE", "PB", "PEG", "估值", "目标价", "赔率", "溢价", "安全边际"),
    "fundamentals": ("营收", "净利", "利润", "毛利", "ROE", "ROIC", "现金流", "OCF", "EPS", "盈利", "订单", "产能"),
    "sector": ("赛道", "板块", "主题", "行业", "CPO", "光模块", "AI 算力", "竞争格局", "市占率"),
}


def _plain_text(value: str) -> str:
    value = re.sub(r"\*\*|__|`", "", value or "")
    value = value.replace("\\|", "|")
    return re.sub(r"\s+", " ", value).strip()


def _clip(value: str, limit: int = 220) -> str:
    value = _plain_text(value)
    if len(value) <= limit:
        return value
    return value[: limit - 1].rstrip("，,；;。 ") + "…"


def _table_cells(line: str) -> list[str]:
    """Split a Markdown row while preserving escaped pipes inside cells."""
    line = line.strip()
    if not (line.startswith("|") and line.endswith("|")):
        return []
    body = line[1:-1]
    cells = re.split(r"(?<!\\)\|", body)
    return [_plain_text(cell) for cell in cells]


def _table_rows(markdown: str) -> list[list[str]]:
    rows: list[list[str]] = []
    for line in (markdown or "").splitlines():
        cells = _table_cells(line)
        if not cells:
            continue
        if all(_TABLE_SEPARATOR_RE.fullmatch(cell.replace(" ", "")) for cell in cells):
            continue
        rows.append(cells)
    return rows


def _row_value(rows: list[list[str]], *labels: str) -> str | None:
    for row in rows:
        if len(row) < 2:
            continue
        key = row[0].lower()
        if any(label.lower() in key for label in labels):
            return "；".join(cell for cell in row[1:] if cell)
    return None


def _row_primary_value(rows: list[list[str]], *labels: str) -> str | None:
    for row in rows:
        if len(row) < 2:
            continue
        key = row[0].lower()
        if any(label.lower() in key for label in labels):
            return row[1]
    return None


def _summary_value(decision: str, key: str) -> str | None:
    match = re.search(
        rf"(?mi)^\s*{re.escape(key)}\s*:\s*(.*?)\s*$",
        decision or "",
    )
    if not match:
        return None
    value = match.group(1).strip().strip("'\"")
    return None if value.lower() in {"", "null", "none"} else value


def _section(markdown: str, heading: str) -> str:
    match = re.search(
        rf"(?mi)^#{{1,6}}\s+[^\n]*{re.escape(heading)}[^\n]*\n",
        markdown or "",
    )
    if not match:
        return ""
    remainder = markdown[match.end():]
    next_heading = _MARKDOWN_HEADING_RE.search(remainder)
    return remainder[: next_heading.start()] if next_heading else remainder


def _contribution_points(decision: str, role: str, limit: int = 2) -> list[str]:
    points: list[str] = []
    section = _section(decision, "关键 Agent 贡献")
    pattern = re.compile(
        rf"^\s*-\s+\*\*{re.escape(role)}\s*/[^*]+\*\*[：:]\s*(.+)$",
        re.IGNORECASE,
    )
    for line in section.splitlines():
        match = pattern.match(line)
        if not match:
            continue
        point = re.sub(r"[（(]作用[：:].*?[）)]\s*$", "", match.group(1)).strip()
        if point:
            points.append(_clip(point))
        if len(points) >= limit:
            break
    return points


def _numbered_points(value: str | None, limit: int = 3) -> list[str]:
    if not value:
        return []
    text = _plain_text(value)
    marker_re = re.compile(r"(?:^|(?<=[。；;\s]))\d+[.、](?!\d)\s*")
    markers = list(marker_re.finditer(text))
    points = []
    for index, marker in enumerate(markers):
        end = markers[index + 1].start() if index + 1 < len(markers) else len(text)
        point = text[marker.end():end].strip(" ；;。")
        if point:
            points.append(point + ("。" if text[end - 1:end] == "。" else ""))
    if not points:
        points = [text]
    return [_clip(point) for point in points[:limit] if point.strip()]


def _classify_core_points(value: str | None) -> dict[str, list[str]]:
    buckets = {name: [] for name in _CORE_KEYWORDS}
    for point in _numbered_points(value, limit=8):
        upper = point.upper()
        for name in ("news", "valuation", "fundamentals", "sector"):
            if any(keyword.upper() in upper for keyword in _CORE_KEYWORDS[name]):
                buckets[name].append(point)
                break
    return buckets


def _inline_value(markdown: str, label: str) -> str | None:
    match = re.search(
        rf"\*\*{re.escape(label)}[：:]\s*(.*?)\*\*",
        markdown or "",
    )
    return _plain_text(match.group(1)) if match else None


def _stock_title(decision: str, ticker: str) -> str:
    match = re.search(rf">\s*\*\*{re.escape(ticker)}\s+([^*|]+)\*\*", decision or "")
    company = _plain_text(match.group(1)) if match else ""
    return f"{ticker} {company}".strip()


def _position_size(decision: str) -> str:
    match = re.search(r"新建仓位[：:]\s*([^*｜|\n]+)", decision or "")
    if match:
        return _plain_text(match.group(1))
    low = _summary_value(decision, "pm_size_low_pct")
    high = _summary_value(decision, "pm_size_high_pct")
    if low is None and high is None:
        return "数据不足"
    if low == high or high is None:
        return f"{low}%"
    return f"{low or 0}%-{high}%"


def _short_term_outlook(decision: str, rows: list[list[str]]) -> tuple[str, str]:
    trend = _summary_value(decision, "short_term_trend")
    confidence = _summary_value(decision, "short_term_confidence")
    visible = re.search(
        r"未来\s*3\s*日\s*[：:]?\s*([^（｜|*\n]+?)\s*"
        r"[（(]\s*置信度\s*[：:]?\s*([^）)]+)[）)]",
        decision or "",
    )
    if visible:
        trend = trend or _plain_text(visible.group(1))
        confidence = confidence or _plain_text(visible.group(2))
    table_value = _row_primary_value(rows, "未来 3 个交易日趋势", "未来3个交易日趋势")
    if table_value:
        if not trend:
            trend = _plain_text(re.split(r"[；;（(]", table_value, maxsplit=1)[0])
        if not confidence:
            match = re.search(r"置信度\s*[：:]\s*([^；;，,）)]+)", table_value)
            if match:
                confidence = _plain_text(match.group(1))
    return trend or "数据不足", confidence or "数据不足"


def _concept_points(decision: str, limit: int = 2) -> list[str]:
    rows = _table_rows(_section(decision, "热门概念归属"))
    points: list[str] = []
    for row in rows:
        if len(row) < 2 or "概念/板块" in row[0]:
            continue
        details = "；".join(cell for cell in row[1:] if cell)
        points.append(_clip(f"{row[0]}：{details}"))
        if len(points) >= limit:
            break
    return points


def _rotation_point(decision: str) -> str | None:
    for line in (decision or "").splitlines():
        if line.lstrip().startswith("|"):
            continue
        match = re.search(r"板块\s*RS|主题内\s*\d+d\s*收益排名", line, re.IGNORECASE)
        if match:
            return _clip(line[match.start():])
    for row in _table_rows(decision):
        for cell in row[1:]:
            match = re.search(r"板块\s*RS|主题内\s*\d+d\s*收益排名", cell, re.IGNORECASE)
            if match:
                value = cell[match.start():].strip()
                prefix = cell[:match.start()]
                if prefix.count("（") > prefix.count("）") and value.endswith("）"):
                    value = value[:-1].rstrip()
                elif prefix.count("(") > prefix.count(")") and value.endswith(")"):
                    value = value[:-1].rstrip()
                return _clip(value)
    return None


def _render_points(points: list[str], fallback: str) -> str:
    useful = [point for point in points if point]
    if not useful:
        useful = [fallback]
    return "\n".join(f"- {point}" for point in useful)


def render_mobile_report(
    decision: str,
    *,
    ticker: str,
    generated_at: str,
) -> str:
    """Render the Feishu-facing report as a concise, table-free mobile digest."""
    decision = (decision or "").strip()
    rows = _table_rows(decision)

    timing_match = re.search(r"(?mi)^#\s*短期操作结论[：:]\s*(.+?)\s*$", decision)
    timing = _plain_text(timing_match.group(1)) if timing_match else "数据不足"
    rating = _summary_value(decision, "pm_rating")
    if not rating:
        rating_match = re.search(r"一年期研究评级[：:]\s*([^*｜|\n]+)", decision)
        rating = _plain_text(rating_match.group(1)) if rating_match else "数据不足"
    action = _summary_value(decision, "pm_action_keyword") or _row_value(rows, "Action 操作")
    action = _plain_text(action or "数据不足").split("；", 1)[0]
    short_trend, short_confidence = _short_term_outlook(decision, rows)
    theme_outlook = _summary_value(decision, "theme_outlook_12m")
    if not theme_outlook:
        theme_outlook = _row_value(rows, "12 个月主题判断") or "数据不足"

    empty_advice = _row_value(rows, "空仓") or _inline_value(decision, "空仓")
    holder_advice = _row_value(rows, "已持仓") or _inline_value(decision, "已持仓")
    if not empty_advice:
        empty_advice = "本次未形成可执行的新建仓方案，按当前动作等待复核。"
    if not holder_advice:
        holder_advice = "本次摘要未形成独立持仓方案，按原风控计划执行。"

    fundamentals = _contribution_points(decision, "fundamentals")
    news = _contribution_points(decision, "news")
    core_thesis = _row_value(rows, "Core Thesis", "核心逻辑")
    core_points = _classify_core_points(core_thesis)
    fundamentals = fundamentals or core_points["fundamentals"][:2]
    news = news or core_points["news"][:2]
    if not news:
        time_stop = _row_value(rows, "Time Stop 时间止损", "Time Stop")
        if time_stop:
            news = [_clip(time_stop)]
    concepts = _concept_points(decision)
    rotation = _rotation_point(decision)
    capital_flow = _row_value(rows, "资金面快照")
    target = _row_primary_value(rows, "目标价区间", "一年期目标价")
    risks = _numbered_points(_row_value(rows, "Key Risks", "核心风险"))
    trigger_match = re.search(r"重新评估条件[：:]\s*([^*\n]+)", decision)
    trigger = _plain_text(trigger_match.group(1)) if trigger_match else "数据不足"

    outlook_points = [f"12 个月主题判断：{_clip(theme_outlook)}"]
    outlook_points.extend(core_points["sector"][:2])
    rotation_points = []
    if rotation:
        rotation_points.append(rotation)
    if capital_flow:
        rotation_points.append(_clip(capital_flow))
    valuation_points = []
    if target:
        valuation_points.append(f"一年期目标价：{_clip(target)}")
    valuation_points.extend(core_points["valuation"][:2])
    valuation_points.extend(f"主要风险：{risk}" for risk in risks)
    valuation_points.append(f"重新评估：{_clip(trigger)}")

    return (
        f"# {_stock_title(decision, ticker)}｜短期操作结论：{timing}\n\n"
        f"更新：{generated_at}\n\n"
        f"> **当前动作：{action}｜新建仓位：{_position_size(decision)}**\n>\n"
        f"> **未来 3 日：{short_trend}（置信度 {short_confidence}）｜"
        f"一年期评级：{rating}**\n\n"
        f"## 现在怎么做\n\n"
        f"**空仓**\n\n- {_clip(empty_advice)}\n\n"
        f"**已持仓**\n\n- {_clip(holder_advice)}\n\n"
        f"## 基本面\n\n"
        f"{_render_points(fundamentals, '本次未提取到可核验的基本面摘要，详见本地审计报告。')}\n\n"
        f"## 消息面与催化\n\n"
        f"{_render_points(news, '本次未提取到可核验的消息面催化，详见本地审计报告。')}\n\n"
        f"## 赛道与前景\n\n"
        f"{_render_points(concepts + outlook_points, '本次未形成可核验的赛道与前景摘要。')}\n\n"
        f"## 近期轮动与资金\n\n"
        f"{_render_points(rotation_points, '本次未提取到可核验的轮动与资金摘要。')}\n\n"
        f"## 估值、风险与重估条件\n\n"
        f"{_render_points(valuation_points, '本次未形成可核验的估值与风险摘要。')}\n"
    )


def write_consolidated_reports(
    save_path: Path,
    *,
    ticker: str,
    user_decision: str,
    audit_sections: Iterable[str],
    generated_at: str | None = None,
) -> Path:
    save_path.mkdir(parents=True, exist_ok=True)
    generated_at = generated_at or dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    decision = user_decision or "# 短期操作结论：数据不足\n\n本次分析未形成可用的 PM 决策。"
    user_path = save_path / "complete_report.md"
    user_path.write_text(
        render_mobile_report(decision, ticker=ticker, generated_at=generated_at),
        encoding="utf-8",
    )

    audit_header = f"# 交易分析审计报告：{ticker}\n\n生成时间: {generated_at}\n\n"
    audit_body = "\n\n".join(section.strip() for section in audit_sections if section and section.strip())
    (save_path / "audit_report.md").write_text(audit_header + audit_body + "\n", encoding="utf-8")
    return user_path
