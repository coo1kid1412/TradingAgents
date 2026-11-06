"""Validate bounded technical judgments without repairing them with another LLM."""

import json
import math
import re


class TechnicalReportError(RuntimeError):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


JUDGMENT_CHOICES = {
    "trend_daily": ("上行", "下行", "震荡"),
    "trend_weekly": ("上行", "下行", "震荡"),
    "momentum": ("强", "中", "弱"),
    "next_3_sessions_bias": ("偏多", "偏空", "中性"),
}
TEXT_LIMITS = {
    "rationale": 120, "key_risk": 80,
    "entry_condition": 80, "invalidation_condition": 80,
}
LEVEL_KEYS = {
    "sma-50", "sma-200", "ema-10", "weekly-ma20", "weekly-ma52",
    "prior-20d-low", "prior-20d-high",
}
JUDGMENT_FIELDS = set(JUDGMENT_CHOICES) | set(TEXT_LIMITS) | {
    "confidence", "evidence_keys", "support_key", "resistance_key",
}


def _fail(code="incomplete_artifact"):
    raise TechnicalReportError(code, "技术面判断不满足精简 JSON 契约，停止分析")


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            _fail()
        result[key] = value
    return result


def parse_technical_judgment(content, facts, current_price):
    """Accept only complete judgments grounded in the supplied canonical facts."""
    if not isinstance(content, str) or len(content) > 6000:
        _fail()
    try:
        value = json.loads(content, object_pairs_hook=_unique_object,
                           parse_constant=lambda _value: _fail())
    except (ValueError, TypeError, RecursionError):
        _fail()
    if not isinstance(value, dict) or set(value) != JUDGMENT_FIELDS:
        _fail()
    for key, choices in JUDGMENT_CHOICES.items():
        if value[key] not in choices:
            _fail("missing_judgment")
    if type(value["confidence"]) is not int or not 1 <= value["confidence"] <= 5:
        _fail()
    for key, limit in TEXT_LIMITS.items():
        text = value[key]
        if not isinstance(text, str) or not text.strip() or len(text) > limit or re.search(
            r"[\d\n\r<>`\[\]*_#!\\]|https?://|www\.|reasoning_content|chain[_ -]?of[_ -]?thought|HANDOFF|SUMMARY",
            text, re.I,
        ):
            _fail()
    known = {fact["local_fact_key"]: fact for fact in facts}
    keys = value["evidence_keys"]
    if (not isinstance(keys, list) or not 1 <= len(keys) <= 5
            or not all(isinstance(key, str) and key in known for key in keys)
            or len(set(keys)) != len(keys)):
        _fail("unknown_fact_reference")
    for name in ("support_key", "resistance_key"):
        key = value[name]
        if key is None:
            continue
        if not isinstance(key, str) or key not in LEVEL_KEYS or key not in known:
            _fail("unknown_fact_reference")
        level = known[key]["value"]
        if type(level) not in (int, float) or not math.isfinite(level):
            _fail("unknown_fact_reference")
        if (name == "support_key" and level > current_price
                or name == "resistance_key" and level < current_price):
            _fail("conflicting_price")
    return value
