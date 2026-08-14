"""Small, bounded recovery helpers for analyst YAML contracts."""

from __future__ import annotations

import re
from typing import Any

import yaml


_PARTIALLY_QUOTED_LIST_ITEM = re.compile(
    r'(?m)^(?P<prefix>\s*-\s*)"(?P<quoted>[^"\n]*)"(?P<tail>[^#\n]+?)(?P<space>\s*)$'
)
_RESERVED_SCALAR_PREFIX = re.compile(
    r"(?m)^(?P<prefix>\s*[A-Za-z_][A-Za-z0-9_]*:\s*)(?P<value>[>|][^\s\n].*?)\s*$"
)


def _repair_partially_quoted_list_items(block: str) -> str:
    """Repair `- "quoted"trailing text` without broad YAML rewriting."""

    def replace(match: re.Match[str]) -> str:
        value = f"{match.group('quoted')}{match.group('tail')}".strip()
        escaped = value.replace("\\", "\\\\").replace('"', '\\"')
        return f'{match.group("prefix")}"{escaped}"{match.group("space")}'

    return _PARTIALLY_QUOTED_LIST_ITEM.sub(replace, block)


def _repair_reserved_scalar_prefixes(block: str) -> str:
    """Quote values such as `>4%`; bare `>` and `|` remain YAML block markers."""

    def replace(match: re.Match[str]) -> str:
        value = match.group("value").replace("\\", "\\\\").replace('"', '\\"')
        return f'{match.group("prefix")}"{value}"'

    return _RESERVED_SCALAR_PREFIX.sub(replace, block)


def extract_yaml_mapping(report: str, key: str) -> tuple[dict[str, Any] | None, str]:
    """Extract a fenced top-level mapping, with one bounded syntax repair."""
    if not report or key not in report:
        return None, "missing"
    saw_candidate = False
    blocks = re.findall(r"```yaml\s*\n(.*?)\n```", report, re.DOTALL | re.IGNORECASE)
    for block in reversed(blocks):
        if not re.search(rf"(?m)^\s*{re.escape(key)}\s*:", block):
            continue
        saw_candidate = True
        repaired = _repair_reserved_scalar_prefixes(
            _repair_partially_quoted_list_items(block)
        )
        candidates = (block, repaired)
        for candidate in dict.fromkeys(candidates):
            try:
                parsed = yaml.safe_load(candidate)
            except yaml.YAMLError:
                continue
            value = parsed.get(key) if isinstance(parsed, dict) else None
            if isinstance(value, dict):
                return value, "valid" if candidate == block else "recovered"
    return None, "invalid" if saw_candidate else "missing"
