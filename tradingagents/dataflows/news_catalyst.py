"""新闻催化剂确定性聚合：把新闻分析师 SUMMARY 块的 key_events 聚合成净催化信号。

背景：新闻分析师已产出结构化 SUMMARY（key_events 带 impact/credibility/horizon/
priced_in_p），但下游 RM 只当散文读、自己判该不该信——同股不同跑松紧不一。
这里把"判单个事件"(LLM，结构化)和"聚合成方向"(Python，确定性)分开：LLM 判事件、
Python 定方向、下游确定性消费，RM 不再二次解读。
"""

from __future__ import annotations

import re
from typing import Optional

import yaml

# impact 措辞 → 数值（+大..-大）
_IMPACT_MAP = {
    "+大": 2.0, "+中": 1.0, "+小": 0.5, "0": 0.0,
    "-小": -0.5, "-中": -1.0, "-大": -2.0,
}
_CRED_MAP = {"高": 1.0, "中": 0.7, "低": 0.4}
# 近端催化对评级窗（~12 月，但短期择时敏感）权重更高
_HORIZON_MAP = {"短期": 1.0, "中期": 0.7, "长期": 0.4}


def _find_summary_yaml(news_report: str) -> Optional[dict]:
    """从新闻报告里抽 ```yaml ... ``` 的 SUMMARY 块并解析。"""
    if not news_report:
        return None
    for block in re.findall(r"```yaml\s*\n(.*?)\n```", news_report, flags=re.DOTALL):
        if "SUMMARY" not in block:
            continue
        try:
            parsed = yaml.safe_load(block)
        except yaml.YAMLError:
            continue
        if isinstance(parsed, dict) and isinstance(parsed.get("SUMMARY"), dict):
            return parsed["SUMMARY"]
    return None


def aggregate_news_catalyst(news_report: str) -> Optional[dict]:
    """把 SUMMARY.key_events 聚合成净催化信号。

    单事件分 = impact × 可信度 × (1 − priced_in%) × 时间窗权重。
    已定价的事件不再驱动（priced_in 高→权重低）；可信度低、远端的事件权重低。

    Returns: {net, direction(+1/0/-1), strength(high/medium/low), score(-30..30),
              n_events, nearest} 或 None（无 SUMMARY/无事件）。
    """
    summary = _find_summary_yaml(news_report)
    if not summary:
        return None
    events = summary.get("key_events") or []
    if not isinstance(events, list) or not events:
        return None

    net = 0.0
    counted = 0
    for e in events:
        if not isinstance(e, dict):
            continue
        impact = _IMPACT_MAP.get(str(e.get("impact", "0")).strip())
        if impact is None:
            continue
        cred = _CRED_MAP.get(str(e.get("credibility", "中")).strip(), 0.7)
        priced = e.get("priced_in_p")
        try:
            priced_frac = float(priced) / 100.0 if priced not in (None, "null", "") else 0.5
        except (TypeError, ValueError):
            priced_frac = 0.5
        priced_frac = min(max(priced_frac, 0.0), 1.0)
        hz_key = next((k for k in _HORIZON_MAP if k in str(e.get("horizon", ""))), None)
        hz = _HORIZON_MAP.get(hz_key, 0.7)
        net += impact * cred * (1.0 - priced_frac) * hz
        counted += 1

    if counted == 0:
        return None

    net = round(net, 2)
    direction = 1 if net > 0.5 else (-1 if net < -0.5 else 0)
    strength = "high" if abs(net) >= 2.0 else ("medium" if abs(net) >= 1.0 else "low")
    score = int(max(-30, min(30, round(net * 15))))   # 喂催化腿，量级同 sell_side(±30)
    # 最近端事件标题（供 PM Time Stop）
    nearest = None
    for e in events:
        if isinstance(e, dict) and "短期" in str(e.get("horizon", "")):
            nearest = str(e.get("title", ""))[:30]
            break
    return {"net": net, "direction": direction, "strength": strength,
            "score": score, "n_events": counted, "nearest": nearest}


_SYS_CATALYST_RE = re.compile(
    r"SYS_CATALYST:\s*direction=(?P<dir>[-\d]+)\s*\|\s*strength=(?P<str>\w+)"
    r"\s*\|\s*score=(?P<score>[-\d]+)"
)


def parse_sys_catalyst(text: str) -> Optional[dict]:
    """从注入文本解析 SYS_CATALYST 回来（下游确定性消费用）。"""
    if not text:
        return None
    m = _SYS_CATALYST_RE.search(text)
    if not m:
        return None
    try:
        return {"direction": int(m.group("dir")), "strength": m.group("str"),
                "score": int(m.group("score"))}
    except ValueError:
        return None
