"""Deterministic, typed Markdown reports for market-warning decisions."""

from __future__ import annotations

import os
import re
import tempfile
from collections.abc import Mapping
from datetime import datetime
from math import isfinite
from numbers import Real
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from .domain import (
    DataStatus,
    DecisionSource,
    FinalWarningDecision,
    LLMContextAssessment,
    Market,
    MarketPhase,
    RiskLevel,
    RunnerResult,
)


_MARKET_NAMES = {Market.A_SHARE: "A股", Market.US: "美股"}
_MARKET_ZONES = {
    Market.A_SHARE: ZoneInfo("Asia/Shanghai"),
    Market.US: ZoneInfo("America/New_York"),
}
_LAMPS = {
    RiskLevel.GREEN: "绿灯 GREEN",
    RiskLevel.YELLOW: "黄灯 YELLOW",
    RiskLevel.ORANGE: "橙灯 ORANGE",
    RiskLevel.RED: "红灯 RED",
    RiskLevel.UNKNOWN: "未知 UNKNOWN",
}
_RULE_LAMPS = {
    RiskLevel.GREEN: "绿灯：环境稳定",
    RiskLevel.YELLOW: "黄灯：风险升温",
    RiskLevel.ORANGE: "橙灯：提前防守",
    RiskLevel.RED: "红灯：风险确认",
    RiskLevel.UNKNOWN: "未知：数据不足",
}
_IMMEDIATE_ACTIONS = {
    RiskLevel.GREEN: "可按既定计划参与，但仍须执行个股止损。",
    RiskLevel.YELLOW: "不追高，优先等待回踩确认。",
    RiskLevel.ORANGE: "暂停追涨，仅允许小仓位条件单，并复核高波动持仓。",
    RiskLevel.RED: "停止新增仓位，主动降低高波动暴露。",
    RiskLevel.UNKNOWN: "暂停新增仓位，先修复数据或模型，再判断风险。",
}
_PHASES = {
    MarketPhase.FIRST_SHOCK: "首次冲击（尚未进入明显续跌区间）",
    MarketPhase.CONTINUATION: "延续下跌（市场已处于回撤后的脆弱阶段）",
}
_THINK_RE = re.compile(r"<think>[\s\S]*?</think>", re.IGNORECASE)
_SENSITIVE_RE = re.compile(
    r"traceback|api\s*error|provider\s+error|authorization\s*:|bearer\s+\S+|"
    r"api[_ -]?key\s*[:=]|minimax_api_key|(?:access_)?token\s*[:=]|"
    r"password\s*[:=]|private[_ -]?key|\bsecret\b|\bsk-[A-Za-z0-9_-]+|"
    r"system\s+(?:prompt|message)|internal\s+prompt|return\s+json\s+only|"
    r"strict_json_output|output_schema|提示词|密钥|令牌|\{\s*[\"']",
    re.IGNORECASE,
)


def _require_complete(result: RunnerResult) -> FinalWarningDecision:
    if not isinstance(result, RunnerResult) or result.decision is None:
        raise ValueError("result must contain a final decision")
    return result.decision


def _safe_text(value: Any, fallback: str = "[内容已脱敏]") -> str:
    if not isinstance(value, str):
        return fallback
    cleaned = _THINK_RE.sub("", value).strip()
    if not cleaned or _SENSITIVE_RE.search(cleaned) or "<think>" in cleaned.lower():
        return fallback
    return cleaned.replace("\r", " ").replace("\n", " ")[:500]


def _safe_code(value: Any) -> str:
    cleaned = _safe_text(value)
    if cleaned == "[内容已脱敏]" or not re.fullmatch(r"[A-Za-z0-9._:/+@=-]{1,200}", cleaned):
        return "[内容已脱敏]"
    return cleaned


def _percentage(value: Any) -> str:
    if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(float(value)):
        return "不可用"
    return f"{float(value) * 100:.2f}%"


def _cap(value: float) -> str:
    return f"{value:.0f}%" if float(value).is_integer() else f"{value:.1f}%"


def _first_block(decision: FinalWarningDecision) -> str:
    return "\n".join(
        (
            f"> **【{_LAMPS[decision.final_level]}】立即操作：{_IMMEDIATE_ACTIONS[decision.final_level]}**",
            f"> 入场门：`{decision.entry_gate}` | 新增仓位上限：`{_cap(decision.new_position_cap_pct)}` | "
            f"持仓动作：`{decision.holding_action}`",
        )
    )


def _latest_data_time(result: RunnerResult) -> datetime:
    snapshot = result.feature_snapshot
    if snapshot is None or not snapshot.source_times:
        return result.as_of_time
    return max(snapshot.source_times.values())


def _rule_action_block(result: RunnerResult) -> str:
    decision = _require_complete(result)
    assessment = result.rule_assessment
    if assessment is None:
        raise ValueError("rule decision requires rule assessment")
    local_data_time = _latest_data_time(result).astimezone(_MARKET_ZONES[result.market])
    return "\n".join(
        (
            f"## 【{_RULE_LAMPS[decision.final_level]}】",
            f"- **立即操作：{_IMMEDIATE_ACTIONS[decision.final_level]}**",
            f"- **入场门：`{decision.entry_gate}`**",
            f"- **新增仓位上限：`{_cap(decision.new_position_cap_pct)}`**",
            f"- **持仓动作：`{decision.holding_action}`**",
            f"- 运行模式：`{_MARKET_NAMES[result.market]}规则生产版`",
            f"- 数据截至：`{local_data_time.isoformat(timespec='minutes')}`",
            f"- 可靠度：`{_safe_code(assessment.reliability_grade)}`",
            f"- 规则分数：`{assessment.risk_score:.1f}/10`（规则分数不是概率）",
        )
    )


def _observed_value(value: Any) -> str:
    if isinstance(value, bool):
        return "是" if value else "否"
    if isinstance(value, Real) and not isinstance(value, bool) and isfinite(float(value)):
        return f"{float(value):.4f}"
    return _safe_text(value, "不可用")


def _rule_trigger_section(result: RunnerResult, *, limit: int | None = None) -> str:
    assessment = result.rule_assessment
    if assessment is None:
        raise ValueError("rule decision requires rule assessment")
    triggered = assessment.triggered_rules if limit is None else assessment.triggered_rules[:limit]
    if not triggered:
        return "## 触发证据\n本次没有规则越过阈值。"
    rows = [
        f"{index}. 规则 `{_safe_code(rule.rule_id)}`：观测值 `{_observed_value(rule.observed_value)}`；"
        f"阈值：{_safe_text(rule.threshold_description)}；分值 `+{rule.severity_points}`"
        for index, rule in enumerate(triggered, start=1)
    ]
    return "## 触发证据\n" + "\n".join(rows)


def _rule_layer_section(result: RunnerResult) -> str:
    assessment = result.rule_assessment
    if assessment is None:
        raise ValueError("rule decision requires rule assessment")
    points: dict[str, int] = {}
    for rule in assessment.triggered_rules:
        points[rule.layer.value] = points.get(rule.layer.value, 0) + rule.severity_points
    rendered = "；".join(f"`{layer}` {score}分" for layer, score in points.items())
    missing = "、".join(_safe_text(item) for item in assessment.missing_optional_groups)
    lines = [
        "## 风险结构",
        f"- 市场阶段：`{assessment.market_phase.value}`",
        f"- 分层得分：{rendered or '无触发得分'}",
        f"- 规则版本：`{_safe_code(assessment.engine_version)}`；清单：`{assessment.manifest_sha256[:12]}`",
    ]
    if missing:
        lines.append(f"- 缺失的可选数据组：{missing}")
    return "\n".join(lines)


def _rule_shadow_section(result: RunnerResult) -> str | None:
    quant = result.shadow_quant_assessment
    if quant is None or quant.reliability_grade == "UNAVAILABLE":
        return None
    return "\n".join(
        (
            "## 影子模型观察",
            "该模型仅用于对照，不参与本次灯号和操作决策。",
            f"- 1日估计：{_percentage(quant.crash_1d_probability)}",
            f"- 3日估计：{_percentage(quant.crash_3d_probability)}",
            f"- 可靠度：`{_safe_code(quant.reliability_grade)}`",
        )
    )


def _rule_context_section(context: LLMContextAssessment) -> str:
    if context.reasoning_status != "validated":
        return "## M3 情景解释\nM3 本次不可用；规则灯号和操作约束保持不变。"
    causal = " -> ".join(_safe_text(item) for item in context.causal_chain)
    conflicting = "；".join(
        f"`{_safe_code(item)}`" for item in context.conflicting_evidence_ids
    ) or "无"
    overlooked = "；".join(_safe_text(item) for item in context.overlooked_risks) or "无"
    warning = (
        "\n- 契约提示：M3 返回的决策字段已忽略。"
        if context.error_class == "decision_override_ignored"
        else ""
    )
    return "\n".join(
        (
            "## M3 情景解释",
            "M3 仅解释触发背景，不改变规则灯号和操作约束。",
            f"- 场景：{_safe_text(context.market_scenario)}",
            f"- 因果链：{causal or '[内容已脱敏]'}",
            f"- 反向证据ID：{conflicting}",
            f"- 遗漏风险：{overlooked}{warning}",
        )
    )


def _previous_session_section(result: RunnerResult) -> str:
    summary = result.previous_session_summary
    if summary is None:
        return "## 上一交易日盘中轨迹\n暂无可用的上一交易日盘中规则记录。"
    changes = " -> ".join(_safe_code(item) for item in summary.state_changes) or "无跨级变化"
    return "\n".join(
        (
            "## 上一交易日盘中轨迹",
            f"- 交易日：`{summary.trade_date.isoformat()}`",
            f"- 最高灯号：`{summary.highest_level.value}`",
            f"- 状态变化：{changes}",
        )
    )


def _render_rule_premarket(
    result: RunnerResult, previous: FinalWarningDecision | None
) -> str:
    decision = _require_complete(result)
    market_name = _MARKET_NAMES[result.market]
    sections = [
        f"# {market_name}大盘骤跌预警",
        _rule_action_block(result),
        _previous_section(decision, previous),
        _previous_session_section(result),
        _rule_layer_section(result),
        _rule_trigger_section(result),
    ]
    shadow = _rule_shadow_section(result)
    if shadow is not None:
        sections.append(shadow)
    if result.context_assessment is not None:
        sections.append(_rule_context_section(result.context_assessment))
    return "\n\n".join(sections) + "\n"


def _render_rule_upgrade(
    result: RunnerResult, previous: FinalWarningDecision | None
) -> str:
    decision = _require_complete(result)
    local_time = result.as_of_time.astimezone(_MARKET_ZONES[result.market])
    action_lines = _rule_action_block(result).splitlines()[1:]
    sections = (
        f"# 【{_RULE_LAMPS[decision.final_level]}】",
        f"{_MARKET_NAMES[result.market]}盘中预警 | `{local_time.isoformat(timespec='minutes')}`",
        "\n".join(action_lines),
        _previous_section(decision, previous).replace("## 相比上一份", "## 状态变化", 1),
        _rule_trigger_section(result, limit=3),
    )
    return "\n\n".join(sections) + "\n"


def _probability_section(result: RunnerResult) -> str:
    quant = result.quant_assessment
    if quant is None or quant.reliability_grade == "UNAVAILABLE":
        rows = "| 1日 | 不可用 | 不可用 |\n| 3日 | 不可用 | 不可用 |"
    else:
        rows = (
            f"| 1日 | {_percentage(quant.crash_1d_probability)} | {_percentage(quant.base_rate_1d)} |\n"
            f"| 3日 | {_percentage(quant.crash_3d_probability)} | {_percentage(quant.base_rate_3d)} |"
        )
    return (
        "## 概率判断\n"
        "以下是校准模型的概率估计，不代表确定会发生。\n\n"
        "| 观察窗口 | 骤跌概率 | 历史基准率 |\n"
        "|---|---:|---:|\n"
        f"{rows}"
    )


def _phase_section(result: RunnerResult) -> str:
    quant = result.quant_assessment
    phase = _PHASES.get(quant.market_phase) if quant is not None else None
    if quant is None or phase is None:
        rendered = "当前阶段不可可靠判定。"
    else:
        rendered = f"`{quant.market_phase.value}`：{phase}"
    return f"## 市场阶段\n{rendered}"


def _previous_section(
    decision: FinalWarningDecision, previous: FinalWarningDecision | None
) -> str:
    if previous is None:
        change = "无可比的上一份有效决策。"
    elif previous.final_level == decision.final_level:
        change = f"维持 {decision.final_level.value}，状态为 `{decision.state_transition}`。"
    else:
        change = (
            f"{previous.final_level.value} -> {decision.final_level.value}，"
            f"状态为 `{decision.state_transition}`。"
        )
    return f"## 相比上一份\n{change}"


def _contributor_section(result: RunnerResult) -> str:
    quant = result.quant_assessment
    rows = [] if quant is None else list(quant.top_contributors[:3])
    if not rows:
        return "## 主要驱动\n暂无可审计的模型驱动项。"
    rendered = []
    for index, item in enumerate(rows, start=1):
        if isinstance(item, Mapping):
            feature = _safe_text(str(item.get("feature", "未命名特征")), "未命名特征")
            contribution = item.get("contribution")
            numeric = (
                f"{float(contribution):+.3f}"
                if isinstance(contribution, Real)
                and not isinstance(contribution, bool)
                and isfinite(float(contribution))
                else "不可用"
            )
            rendered.append(f"{index}. `{feature}`：模型贡献 {numeric}")
        else:
            rendered.append(f"{index}. 未命名驱动项")
    return "## 主要驱动\n" + "\n".join(rendered)


def _context_section(context: LLMContextAssessment | None) -> str:
    if context is None:
        return "## M3 情景校验\n本时点未调用 M3；量化与代码规则独立有效。"
    if context.reasoning_status != "validated":
        return "## M3 情景校验\nM3 本次不可用；未改变代码基线。"
    causal = " -> ".join(_safe_text(item) for item in context.causal_chain)
    conflicting_ids = "；".join(
        f"`{_safe_code(item)}`" for item in context.conflicting_evidence_ids
    ) or "无"
    overlooked = "；".join(_safe_text(item) for item in context.overlooked_risks) or "无"
    return "\n".join(
        (
            "## M3 情景校验",
            f"- 场景：{_safe_text(context.market_scenario)}",
            f"- 因果链：{causal or '[内容已脱敏]'}",
            f"- 反向证据ID：{conflicting_ids}",
            f"- 遗漏风险：{overlooked}",
            f"- 建议：{context.recommended_risk_level.value}，置信度 {_percentage(context.confidence)}；"
            f"{_safe_text(context.action_reason)}",
        )
    )


def _metadata_section(result: RunnerResult) -> str:
    snapshot = result.feature_snapshot
    quant = result.quant_assessment
    status = snapshot.data_quality.value if snapshot is not None else "insufficient"
    reliability = _safe_code(snapshot.reliability_grade) if snapshot is not None else "UNAVAILABLE"
    feature_version = _safe_code(snapshot.feature_version) if snapshot is not None else "unavailable"
    model_version = _safe_code(quant.model_version) if quant is not None else "unavailable"
    calibration = _safe_code(quant.calibration_version) if quant is not None else "unavailable"
    source_times = []
    if snapshot is not None:
        source_times = [
            f"  - `{_safe_code(str(source))}`：`{timestamp.isoformat()}`"
            for source, timestamp in sorted(snapshot.source_times.items())
        ]
    source_block = "\n- 源数据时间：\n" + "\n".join(source_times) if source_times else "\n- 源数据时间：不可用"
    shadow = (
        "\n- 运行限制：影子运行，仅供观察，不联动个股生产硬门控。"
        if status == DataStatus.SHADOW.value
        else ""
    )
    unknown = (
        "\n- 解释：数据不足不等于低风险；UNKNOWN 也不等于 RED。"
        if result.decision and result.decision.final_level == RiskLevel.UNKNOWN
        else ""
    )
    return (
        "## 数据与模型\n"
        f"- 数据状态：`{status}`；可靠度：`{reliability}`\n"
        f"- 特征版本：`{feature_version}`\n"
        f"- 模型版本：`{model_version}`；校准版本：`{calibration}`"
        f"{source_block}"
        f"{shadow}{unknown}"
    )


def _trigger_section(result: RunnerResult) -> str:
    snapshot = result.feature_snapshot
    transition_fields = (
        "pressure_transition_signal",
        "volatility_acceleration_transition",
        "abnormal_range_weak_close_transition",
        "breadth_deterioration_transition",
        "credit_volatility_transition",
        "equity_dispersion_transition",
    )
    rows = []
    if snapshot is not None:
        rows.extend(
            f"- 代码信号：`{name}`"
            for name in transition_fields
            if snapshot.features.get(name) is True
        )
    quant = result.quant_assessment
    if quant is not None:
        for item in quant.top_contributors[:3]:
            if isinstance(item, Mapping):
                feature = _safe_code(str(item.get("feature", "unavailable")))
                contribution = item.get("contribution")
                numeric = (
                    f"{float(contribution):+.3f}"
                    if isinstance(contribution, Real)
                    and not isinstance(contribution, bool)
                    and isfinite(float(contribution))
                    else "不可用"
                )
                rows.append(f"- 模型驱动：`{feature}`（贡献 {numeric}）")
    context = result.context_assessment
    if context is not None and context.reasoning_status == "validated":
        rows.append(
            "- M3 支持证据："
            + "；".join(f"`{_safe_code(item)}`" for item in context.supporting_evidence_ids)
        )
    return "## 触发证据\n" + ("\n".join(rows) if rows else "未形成可展示的具体触发证据。")


def render_premarket_report(
    result: RunnerResult, previous: FinalWarningDecision | None
) -> str:
    """Render the fixed reading order used for every premarket report."""

    decision = _require_complete(result)
    if decision.decision_source == DecisionSource.RULE_V1:
        return _render_rule_premarket(result, previous)
    market_name = _MARKET_NAMES[result.market]
    local_time = result.as_of_time.astimezone(_MARKET_ZONES[result.market])
    sections = (
        _first_block(decision),
        f"# {market_name}大盘骤跌概率预警\n\n评估时间：`{local_time.isoformat(timespec='minutes')}`",
        _probability_section(result),
        _phase_section(result),
        _previous_section(decision, previous),
        _contributor_section(result),
        _context_section(result.context_assessment),
        _metadata_section(result),
    )
    return "\n\n".join(sections) + "\n"


def render_upgrade_report(
    result: RunnerResult, previous: FinalWarningDecision | None
) -> str:
    """Render an intraday alert with the same first-screen action contract."""

    decision = _require_complete(result)
    if decision.decision_source == DecisionSource.RULE_V1:
        return _render_rule_upgrade(result, previous)
    local_time = result.as_of_time.astimezone(_MARKET_ZONES[result.market])
    mode = "盘中升级" if decision.push_required else "盘中评估"
    sections = (
        _first_block(decision),
        f"# {_MARKET_NAMES[result.market]}大盘骤跌概率预警（{mode}）\n\n"
        f"评估时间：`{local_time.isoformat(timespec='minutes')}`",
        _previous_section(decision, previous).replace("## 相比上一份", "## 变化", 1),
        _trigger_section(result),
    )
    return "\n\n".join(sections) + "\n"


def report_path(result: RunnerResult, root: Path | str) -> Path:
    local_time = result.as_of_time.astimezone(_MARKET_ZONES[result.market])
    slot = re.sub(r"[^a-z0-9_-]+", "-", result.session_slot.strip().lower()).strip("-")
    if not slot:
        slot = "evaluation"
    return (
        Path(root)
        / result.market.value
        / local_time.date().isoformat()
        / f"{local_time:%H%M}-{slot}.md"
    )


def write_report(
    result: RunnerResult,
    previous: FinalWarningDecision | None,
    root: Path | str = Path("reports/market_warning"),
) -> Path:
    """Atomically write a report and return its deterministic path."""

    target = report_path(result, root)
    target.parent.mkdir(parents=True, exist_ok=True)
    content = (
        render_premarket_report(result, previous)
        if "premarket" in result.session_slot.lower()
        else render_upgrade_report(result, previous)
    )
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return target
