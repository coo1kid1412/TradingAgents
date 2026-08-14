"""Guarantee a bounded, machine-readable tail for each risk response."""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage

from tradingagents.agents.utils.risk_consensus import _last_view


_REQUIRED_FIELDS = {
    "role", "severity", "cap_pct", "cap_basis", "evidence_ids", "data_supported",
}


def _has_complete_view(content: str, role: str) -> bool:
    view = _last_view(content)
    return bool(
        view
        and _REQUIRED_FIELDS.issubset(view)
        and str(view.get("role")) == role
        and isinstance(view.get("evidence_ids"), list)
        and isinstance(view.get("data_supported"), bool)
    )


def _fallback_view(role: str) -> str:
    return f"""```yaml
RISK_VIEW:
  role: {role}
  severity: unknown
  cap_pct: null
  cap_basis: null
  evidence_ids: []
  data_supported: false
```"""


def _drop_unclosed_yaml_tail(content: str) -> str:
    """Keep prose, but never concatenate a retry inside an unclosed fence."""
    marker = "```yaml"
    start = content.lower().rfind(marker)
    if start < 0:
        return content.strip()
    tail = content[start + len(marker):]
    if "```" not in tail:
        return content[:start].rstrip()
    return content.strip()


def invoke_risk_response(llm, prompt: str, *, role: str) -> AIMessage:
    """Invoke once, retry only the structured tail, then fail closed."""
    first = llm.invoke(prompt)
    first_content = str(getattr(first, "content", "") or "")
    if _has_complete_view(first_content, role):
        return AIMessage(content=first_content)

    retry_prompt = (
        "上一条回答被截断或缺少完整结构化结尾。不要重复正文，只输出一个 YAML 代码块；"
        f"顶层键必须为 RISK_VIEW，role 必须为 {role}，并完整包含 severity、cap_pct、"
        "cap_basis、evidence_ids、data_supported。无法追溯时必须 data_supported=false、"
        "cap_pct=null、evidence_ids=[]。"
    )
    try:
        second = llm.invoke([
            HumanMessage(content=prompt),
            AIMessage(content=first_content),
            HumanMessage(content=retry_prompt),
        ])
        second_content = str(getattr(second, "content", "") or "")
    except Exception:
        second_content = ""
    tail = second_content if _has_complete_view(second_content, role) else _fallback_view(role)
    safe_first = _drop_unclosed_yaml_tail(first_content)
    combined = "\n\n".join(part for part in (safe_first, tail.strip()) if part)
    return AIMessage(content=combined)
