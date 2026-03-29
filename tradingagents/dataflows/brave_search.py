"""Brave Search API integration for real-time news retrieval."""

import os
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional

import requests

logger = logging.getLogger(__name__)

_BRAVE_API_URL = "https://api.search.brave.com/res/v1/web/search"

# Domains to exclude (encyclopedias)
_EXCLUDED_DOMAINS = {
    "baike.baidu.com",
    "zh.wikipedia.org",
    "en.wikipedia.org",
    "zh.m.wikipedia.org",
    "wikipedia.org",
    "wikidata.org",
}


def _get_api_key() -> Optional[str]:
    """Read BRAVE_SEARCH_API_KEY from environment."""
    return os.environ.get("BRAVE_SEARCH_API_KEY")


def _is_excluded(url: str) -> bool:
    """Check if a URL belongs to an excluded domain."""
    url_lower = url.lower()
    for domain in _EXCLUDED_DOMAINS:
        if domain in url_lower:
            return True
    return False


def _is_within_days(date_str: Optional[str], days: int = 7) -> bool:
    """Check if a date string is within the last N days. If unparseable, keep it."""
    if not date_str:
        return True  # No date info → keep
    cutoff = datetime.now() - timedelta(days=days)
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d", "%Y-%m-%dT%H:%M:%S.%f"):
        try:
            dt = datetime.strptime(date_str[:len(fmt) + 3], fmt)
            return dt >= cutoff
        except (ValueError, IndexError):
            continue
    return True  # Unparseable → keep


def search_news(query: str, count: int = 20, freshness: str = "pw") -> str:
    """
    Search for recent news using Brave Search API.

    Args:
        query: Search query string (e.g. "贵州茅台 600519 新闻")
        count: Number of results to request (will be filtered down to top 10)
        freshness: Brave freshness filter; 'pd'=past day, 'pw'=past week, 'pm'=past month

    Returns:
        Formatted string of top 10 news results (after filtering), or error message.
    """
    api_key = _get_api_key()
    if not api_key:
        return "Brave Search API key 未配置，请在 .env 中设置 BRAVE_SEARCH_API_KEY。"

    headers = {
        "Accept": "application/json",
        "Accept-Encoding": "gzip",
        "X-Subscription-Token": api_key,
    }
    params = {
        "q": query,
        "count": count,
        "freshness": freshness,
        "text_decorations": "false",
        "search_lang": "zh-hans",
    }

    try:
        resp = requests.get(_BRAVE_API_URL, headers=headers, params=params, timeout=15)
        resp.raise_for_status()
    except requests.RequestException as e:
        logger.warning("Brave Search request failed: %s", e)
        return f"Brave Search 请求失败: {e}"

    data = resp.json()
    web_results: List[Dict] = data.get("web", {}).get("results", [])

    if not web_results:
        return f"Brave Search 未找到与 '{query}' 相关的新闻结果。"

    # Filter: exclude encyclopedias + enforce 7-day freshness
    filtered: List[Dict] = []
    for item in web_results:
        url = item.get("url", "")
        if _is_excluded(url):
            continue
        # Brave returns 'page_age' or 'age' for freshness; also check description dates
        if not _is_within_days(item.get("page_age") or item.get("age")):
            continue
        filtered.append(item)

    if not filtered:
        return f"Brave Search 返回了结果，但经过过滤（排除百科、仅保留7天内）后无有效新闻。"

    # Take top 10
    top_results = filtered[:10]

    # Format output
    lines = [f"=== Brave Search 实时新闻 (query: {query}) ===\n"]
    for i, item in enumerate(top_results, 1):
        title = item.get("title", "无标题")
        url = item.get("url", "")
        description = item.get("description", "无摘要")
        age = item.get("page_age") or item.get("age") or "未知"
        source = item.get("meta_url", {}).get("hostname", "") if isinstance(item.get("meta_url"), dict) else ""

        lines.append(f"[{i}] {title}")
        if source:
            lines.append(f"    来源: {source}")
        lines.append(f"    时间: {age}")
        lines.append(f"    链接: {url}")
        lines.append(f"    摘要: {description}")
        lines.append("")

    lines.append(f"共 {len(top_results)} 条新闻（已过滤百科类结果，仅保留7天内内容）")
    return "\n".join(lines)
