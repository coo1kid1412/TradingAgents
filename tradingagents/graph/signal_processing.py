# TradingAgents/graph/signal_processing.py

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

# ── 7 级评级体系 ────────────────────────────────────────────────
# 从强烈买入到强烈卖出，共 7 档，覆盖 A 股研报常见评级
_RATING_TIER = [
    "STRONG_BUY",
    "BUY",
    "OVERWEIGHT",
    "HOLD",
    "UNDERWEIGHT",
    "SELL",
    "STRONG_SELL",
]

VALID_RATINGS = set(_RATING_TIER)

# 中文 → 英文映射（覆盖 A 股/港股/美股常见表述）
_CN_TO_EN = {
    "强烈买入": "STRONG_BUY",
    "强力买入": "STRONG_BUY",
    "买入": "BUY",
    "增持": "OVERWEIGHT",
    "跑赢大盘": "OVERWEIGHT",
    "持有": "HOLD",
    "中性": "HOLD",
    "观望": "HOLD",
    "减持": "UNDERWEIGHT",
    "跑输大盘": "UNDERWEIGHT",
    "卖出": "SELL",
    "强烈卖出": "STRONG_SELL",
    "强力卖出": "STRONG_SELL",
}

# 英文 → 中文标签（用于双语输出）
_EN_TO_CN = {v: k for k, v in _CN_TO_EN.items()}
# 补充多个中文 key 映射到同一英文时，取最常用的那个
_EN_TO_CN.update({
    "STRONG_BUY": "强烈买入",
    "BUY": "买入",
    "OVERWEIGHT": "增持",
    "HOLD": "持有",
    "UNDERWEIGHT": "减持",
    "SELL": "卖出",
    "STRONG_SELL": "强烈卖出",
})

_DEFAULT_RATING = "HOLD"

# ── 标点清理正则 ────────────────────────────────────────────────
_PUNCT_RE = re.compile(r"[*#.。，、！!？?():：\s]+")


def _bilingual(rating: str) -> str:
    """Return bilingual format: '中文 ENGLISH'."""
    cn = _EN_TO_CN.get(rating, rating)
    return f"{cn} {rating}"


class SignalProcessor:
    """Processes trading signals to extract actionable decisions."""

    def __init__(self, quick_thinking_llm: Any):
        """Initialize with an LLM for processing."""
        self.quick_thinking_llm = quick_thinking_llm

    def process_signal(self, full_signal: str) -> str:
        """
        Process a full trading signal to extract the core decision.

        Args:
            full_signal: Complete trading signal text

        Returns:
            Bilingual rating string, e.g. "买入 BUY" or "强烈卖出 STRONG_SELL".
            Defaults to "持有 HOLD" if no valid rating can be extracted.
        """
        messages = [
            (
                "system",
                "You are an efficient assistant that extracts the trading decision from analyst reports. "
                "The report may be written in Chinese or English. Extract the rating regardless of the report language. "
                "Extract the rating as exactly one of: STRONG_BUY, BUY, OVERWEIGHT, HOLD, UNDERWEIGHT, SELL, STRONG_SELL. "
                "Output only the single rating word, nothing else.",
            ),
            ("human", full_signal),
        ]

        raw = self.quick_thinking_llm.invoke(messages).content.strip()
        rating = self._extract_rating(raw)

        if rating not in VALID_RATINGS:
            logger.warning(
                "无法从 LLM 输出中提取有效评级: raw=%r, extracted=%r, 回退为 %s",
                raw[:100], rating, _DEFAULT_RATING,
            )
            rating = _DEFAULT_RATING

        return _bilingual(rating)

    @staticmethod
    def _extract_rating(raw: str) -> str:
        """Extract a rating word from raw LLM output.

        Tries in order:
        1. Direct English rating match (after stripping punctuation)
        2. Chinese-to-English mapping (handles LLM outputting Chinese)
        3. Falls back to raw text upper-cased (caller validates)
        """
        # 1. English match
        for word in raw.split():
            cleaned = _PUNCT_RE.sub("", word).upper()
            if cleaned in VALID_RATINGS:
                return cleaned

        # 2. Chinese match — scan for longest Chinese key first
        cleaned_raw = _PUNCT_RE.sub("", raw)
        for cn_term in sorted(_CN_TO_EN, key=len, reverse=True):
            if cn_term in cleaned_raw:
                return _CN_TO_EN[cn_term]

        # 3. No match
        return raw.upper().strip()
