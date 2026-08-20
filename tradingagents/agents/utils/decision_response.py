"""Guarantee a bounded, machine-readable tail for research decision roles."""

from __future__ import annotations

import re
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage

from tradingagents.agents.utils.handoff import extract_final_report_artifact
from tradingagents.agents.utils.yaml_summary import extract_yaml_mapping


_REQUIRED_FIELDS = {
    "role",
    "as_of",
    "accepted_claim_ids",
    "challenged_claims",
    "decision_judgments",
    "unresolved_dissent",
    "conditions_to_revisit",
    "quality_status",
}
_LIST_FIELDS = {
    "accepted_claim_ids",
    "challenged_claims",
    "decision_judgments",
    "unresolved_dissent",
    "conditions_to_revisit",
}
_EVIDENCE_ID = re.compile(
    r"(?<![A-Z0-9_])[A-Z][A-Z0-9_]{1,24}-[A-Z0-9][A-Z0-9_-]{0,48}(?![A-Z0-9_-])"
)


def _referenced_claim_ids(handoff: dict[str, Any]) -> set[str]:
    result = {str(item) for item in handoff.get("accepted_claim_ids", [])}
    for challenge in handoff.get("challenged_claims", []):
        if isinstance(challenge, dict) and challenge.get("claim_id"):
            result.add(str(challenge["claim_id"]))
    for judgment in handoff.get("decision_judgments", []):
        if isinstance(judgment, dict):
            result.update(str(item) for item in judgment.get("claim_ids", []))
    return result


def _has_complete_handoff(content: str, role: str, allowed_ids: set[str]) -> bool:
    handoff, status = extract_yaml_mapping(content, "DECISION_HANDOFF")
    if handoff is None or status not in {"valid", "recovered"}:
        return False
    if not _REQUIRED_FIELDS.issubset(handoff) or str(handoff.get("role")) != role:
        return False
    if any(not isinstance(handoff.get(field), list) for field in _LIST_FIELDS):
        return False
    if str(handoff.get("quality_status")) not in {"complete", "partial"}:
        return False
    if not handoff.get("decision_judgments"):
        return False
    referenced = _referenced_claim_ids(handoff)
    return not referenced or referenced.issubset(allowed_ids)


def _fallback_handoff(role: str) -> str:
    return f"""```yaml
DECISION_HANDOFF:
  role: {role}
  as_of: unknown
  accepted_claim_ids: []
  challenged_claims: []
  decision_judgments: []
  unresolved_dissent:
    - 模型未生成可解析交接单；该角色意见不得单独主导决策
  conditions_to_revisit:
    - 补齐结构化交接单后重新评估
  quality_status: invalid
```"""


def invoke_decision_response(llm, prompt: str, *, role: str) -> AIMessage:
    """Invoke once, repair only a bounded decision handoff, then fail closed."""
    allowed_ids = set(_EVIDENCE_ID.findall(prompt or ""))
    first = llm.invoke(prompt)
    first_content = extract_final_report_artifact(str(getattr(first, "content", "") or ""))
    if _has_complete_handoff(first_content, role, allowed_ids):
        return AIMessage(content=first_content)

    evidence_allowlist = ", ".join(sorted(allowed_ids)) or "（无可引用正式证据 ID）"
    bounded_first = first_content[-10_000:]
    retry_prompt = f"""上一轮输出缺少可解析的 DECISION_HANDOFF。不要重复正文，只输出一个完整 YAML 代码块。
role 必须为 {role}；必须包含契约要求的全部字段。不得新增事实，只能引用以下正式证据 ID：
{evidence_allowlist}

上一轮可见结论（限长，仅供补交接单）：
{bounded_first}
"""
    try:
        second = llm.invoke([HumanMessage(content=retry_prompt)])
        second_content = extract_final_report_artifact(
            str(getattr(second, "content", "") or "")
        )
    except Exception:
        second_content = ""
    if not _has_complete_handoff(second_content, role, allowed_ids):
        return AIMessage(content=_fallback_handoff(role))
    return AIMessage(content=second_content.strip())
