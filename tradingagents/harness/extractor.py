"""从报告目录提取 RM_SUMMARY + PM_SUMMARY YAML 块。

YAML 块位于 manager.md / decision.md 末尾，由 RM/PM prompt 强制输出。
本模块只做"找到 YAML 块 + 用 PyYAML 解析"，不做任何 regex 推断。
"""

from __future__ import annotations

import datetime as _dt
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)


@dataclass
class ExtractResult:
    """提取结果，包含 RM_SUMMARY / PM_SUMMARY 字典 + 解析告警。"""
    rm_summary: dict | None = None
    pm_summary: dict | None = None
    rm_parsed: bool = False
    pm_parsed: bool = False
    warnings: list[str] = field(default_factory=list)


# 报告目录名格式：<ticker>_<name>_<YYYYMMDD>_<HHMMSS>
_DIR_NAME_RE = re.compile(r"^(\w+)_(.+?)_(\d{8})_(\d{6})$")


def parse_report_dir_name(dir_name: str) -> dict | None:
    """从目录名解析 ticker / company_name / trade_date / timestamp。

    返回 None 表示目录名不符合规范。
    """
    m = _DIR_NAME_RE.match(dir_name)
    if not m:
        return None
    ticker, name, date_str, time_str = m.groups()
    try:
        dt = _dt.datetime.strptime(f"{date_str}{time_str}", "%Y%m%d%H%M%S")
    except ValueError:
        return None
    return {
        "ticker": ticker,
        "company_name": name,
        "trade_date": dt.date().isoformat(),
        "report_timestamp": dt.isoformat(timespec="seconds"),
    }


def classify_window(report_timestamp: str) -> str:
    """根据报告时间戳分类为 4 档时间窗。

    pre_market:    < 09:30
    morning:       09:30-11:30
    afternoon:     11:30-15:00
    post_market:   >= 15:00
    """
    dt = _dt.datetime.fromisoformat(report_timestamp)
    t = dt.time()
    if t < _dt.time(9, 30):
        return "pre_market"
    if t < _dt.time(11, 30):
        return "morning"
    if t < _dt.time(15, 0):
        return "afternoon"
    return "post_market"


def _find_yaml_block(text: str, key: str) -> dict | None:
    """从 markdown 文本里找到 ```yaml ... ``` 代码块，且块内顶层 key 匹配。

    Returns:
        解析后的 dict（key 之下的内容），或 None 表示未找到/解析失败。
    """
    # 找所有 ```yaml ... ``` 块，从后往前遍历（YAML 一般在文末）
    blocks = re.findall(r"```yaml\s*\n(.*?)\n```", text, flags=re.DOTALL)
    for block in reversed(blocks):
        if key not in block:
            continue
        lines = block.splitlines()
        while lines and lines[0].strip() == "---":
            lines.pop(0)
        while lines and lines[-1].strip() == "---":
            lines.pop()
        block = "\n".join(lines)
        try:
            parsed = yaml.safe_load(block)
        except yaml.YAMLError as e:
            logger.warning("YAML 解析失败: %s", e)
            continue
        if isinstance(parsed, dict) and key in parsed:
            return parsed[key]
    return None


def extract_from_report(report_dir: Path) -> ExtractResult:
    """从报告目录提取 RM/PM YAML 摘要。

    manager.md 找 RM_SUMMARY；decision.md 找 PM_SUMMARY。
    """
    result = ExtractResult()
    manager_path = report_dir / "2_research" / "manager.md"
    decision_path = report_dir / "5_portfolio" / "decision.md"

    if manager_path.exists():
        text = manager_path.read_text(encoding="utf-8", errors="replace")
        rm = _find_yaml_block(text, "RM_SUMMARY")
        if rm is not None:
            result.rm_summary = rm
            result.rm_parsed = True
        else:
            result.warnings.append("manager.md 中未找到 RM_SUMMARY YAML 块")
    else:
        result.warnings.append(f"manager.md 不存在: {manager_path}")

    if decision_path.exists():
        text = decision_path.read_text(encoding="utf-8", errors="replace")
        pm = _find_yaml_block(text, "PM_SUMMARY")
        if pm is not None:
            result.pm_summary = pm
            result.pm_parsed = True
        else:
            result.warnings.append("decision.md 中未找到 PM_SUMMARY YAML 块")
    else:
        result.warnings.append(f"decision.md 不存在: {decision_path}")

    return result
