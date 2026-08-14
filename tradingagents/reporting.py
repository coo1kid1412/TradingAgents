"""Shared user-facing and audit report writers."""

from __future__ import annotations

import datetime as dt
import re
from pathlib import Path
from typing import Iterable


def _user_decision_excerpt(decision: str) -> str:
    """Keep the scan-friendly decision card; move long-form PM prose to audit."""
    decision = (decision or "").strip()
    long_form = re.search(
        r"(?mi)^##\s*(?:一|1)\s*[、.．:：)]?\s*投资决策与入场时机(?:[（(][^\n]*[）)])?\s*$",
        decision,
    )
    if long_form:
        return decision[:long_form.start()].rstrip()
    return decision


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
    header = f"# 交易分析报告：{ticker}\n\n生成时间: {generated_at}\n\n"
    decision = _user_decision_excerpt(user_decision)
    decision = decision or "# 决策未生成\n\n本次分析未形成可用的 PM 决策。"
    user_path = save_path / "complete_report.md"
    user_path.write_text(header + decision + "\n", encoding="utf-8")

    audit_header = f"# 交易分析审计报告：{ticker}\n\n生成时间: {generated_at}\n\n"
    audit_body = "\n\n".join(section.strip() for section in audit_sections if section and section.strip())
    (save_path / "audit_report.md").write_text(audit_header + audit_body + "\n", encoding="utf-8")
    return user_path
