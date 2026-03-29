"""Xueqiu (雪球) social media sentiment data fetching.

Searches Xueqiu for stock-related posts, comments and sub-replies.
Uses HTTP API first; falls back to Playwright headless browser if blocked.
"""

import html
import os
import re
import time as _time
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import requests

from .config import get_config


# ---------------------------------------------------------------------------
# Internal exception for API blocking detection
# ---------------------------------------------------------------------------
class _XueqiuAPIBlocked(Exception):
    """Raised when Xueqiu API returns 403, login redirect, or error code."""


# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------
_XUEQIU_BASE = "https://xueqiu.com"
_SEARCH_API = f"{_XUEQIU_BASE}/query/v1/search/status.json"
_COMMENTS_API = f"{_XUEQIU_BASE}/statuses/comments.json"

_MAX_POSTS = 8
_MAX_COMMENTS = 20
_POST_CONTENT_LIMIT = 650
_COMMENT_CONTENT_LIMIT = 150
_PLAYWRIGHT_TIMEOUT_MS = 90_000
_MAX_LOOKBACK_DAYS = 7


def _get_token() -> str:
    """Retrieve Xueqiu token from env or config."""
    token = os.environ.get("XUEQIU_TOKEN", "").strip()
    if not token:
        token = get_config().get("xueqiu_token", "").strip()
    return token


def _get_headers(token: str) -> Dict[str, str]:
    return {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/131.0.0.0 Safari/537.36"
        ),
        "Cookie": f"xq_a_token={token}",
        "Referer": _XUEQIU_BASE,
        "Accept": "application/json, text/plain, */*",
    }


def _strip_html(text: str) -> str:
    """Remove HTML tags and unescape entities."""
    if not text:
        return ""
    text = re.sub(r"<[^>]+>", "", text)
    text = html.unescape(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "..."


def _ts_to_datetime(ts) -> Optional[datetime]:
    """Convert Xueqiu Unix-millisecond timestamp to datetime."""
    if not ts:
        return None
    try:
        return datetime.fromtimestamp(int(ts) / 1000)
    except (ValueError, TypeError, OSError):
        return None


def _format_time(dt: Optional[datetime]) -> str:
    if not dt:
        return "未知时间"
    return dt.strftime("%Y-%m-%d %H:%M")


def _parse_xueqiu_date_str(date_str: str) -> Optional[datetime]:
    """Parse Xueqiu's display date strings like '03-24 11:24', '昨天 11:24', '2026-03-24'."""
    if not date_str:
        return None
    date_str = date_str.strip()
    # Remove source text like '· 来自iPad'
    date_str = re.sub(r"·.*$", "", date_str).strip()

    now = datetime.now()
    try:
        if "今天" in date_str:
            time_part = re.search(r"(\d{1,2}:\d{2})", date_str)
            if time_part:
                t = datetime.strptime(time_part.group(1), "%H:%M")
                return now.replace(hour=t.hour, minute=t.minute, second=0, microsecond=0)
        elif "昨天" in date_str:
            time_part = re.search(r"(\d{1,2}:\d{2})", date_str)
            yesterday = now - timedelta(days=1)
            if time_part:
                t = datetime.strptime(time_part.group(1), "%H:%M")
                return yesterday.replace(
                    hour=t.hour, minute=t.minute, second=0, microsecond=0
                )
            return yesterday.replace(hour=0, minute=0, second=0, microsecond=0)
        elif re.match(r"\d{4}-\d{2}-\d{2}", date_str):
            if len(date_str) >= 16:
                return datetime.strptime(date_str[:16].strip(), "%Y-%m-%d %H:%M")
            return datetime.strptime(date_str[:10], "%Y-%m-%d")
        elif re.match(r"\d{2}-\d{2}", date_str):
            time_part = re.search(r"(\d{2}-\d{2})\s+(\d{1,2}:\d{2})", date_str)
            if time_part:
                return datetime.strptime(
                    f"{now.year}-{time_part.group(1)} {time_part.group(2)}",
                    "%Y-%m-%d %H:%M",
                )
            return datetime.strptime(f"{now.year}-{date_str[:5]}", "%Y-%m-%d")
    except (ValueError, AttributeError):
        pass
    return None


# ---------------------------------------------------------------------------
# HTTP API layer
# ---------------------------------------------------------------------------
def _try_http_search(query: str, token: str) -> List[dict]:
    """Search Xueqiu via HTTP API. Returns list of parsed post dicts.

    Raises _XueqiuAPIBlocked if the API is inaccessible.
    """
    session = requests.Session()
    session.headers.update(_get_headers(token))

    # Xueqiu requires a pre-visit to set cookies
    try:
        session.get(_XUEQIU_BASE, timeout=10)
    except requests.RequestException:
        pass  # best-effort cookie init

    resp = session.get(
        _SEARCH_API,
        params={"q": query, "sort": "time", "count": 10, "comment": 0},
        timeout=15,
    )

    if resp.status_code in (403, 401, 429):
        raise _XueqiuAPIBlocked(f"HTTP {resp.status_code}")

    # Detect WAF / HTML response (Aliyun WAF returns JS challenge page)
    content_type = resp.headers.get("Content-Type", "")
    if "text/html" in content_type or resp.text.lstrip().startswith("<"):
        raise _XueqiuAPIBlocked("WAF blocked: received HTML instead of JSON")

    # Check for login redirect
    if "login" in resp.url and resp.url != _SEARCH_API:
        raise _XueqiuAPIBlocked("Redirected to login page")

    data = resp.json()

    # Xueqiu error response format
    if isinstance(data, dict) and data.get("error_code"):
        raise _XueqiuAPIBlocked(
            f"API error: {data.get('error_code')} - {data.get('error_description', '')}"
        )

    # Extract post list
    statuses = []
    raw_list = data.get("list") or data.get("statuses") or []
    for item in raw_list:
        post = item if not isinstance(item, dict) or "id" in item else item
        statuses.append({
            "id": post.get("id"),
            "user_id": post.get("user_id") or (post.get("user", {}) or {}).get("id"),
            "screen_name": (post.get("user", {}) or {}).get("screen_name", "匿名用户"),
            "title": _strip_html(post.get("title", "")),
            "text": _strip_html(post.get("text", "") or post.get("description", "")),
            "created_at": _ts_to_datetime(post.get("created_at")),
            "reply_count": post.get("reply_count", 0),
            "like_count": post.get("like_count", 0),
            "target": f"{_XUEQIU_BASE}/{post.get('user_id', '')}/{post.get('id', '')}",
        })

    return statuses


def _try_http_comments(status_id, token: str) -> List[dict]:
    """Fetch comments for a specific post via HTTP API.

    Returns list of comment dicts. Raises _XueqiuAPIBlocked if blocked.
    """
    session = requests.Session()
    session.headers.update(_get_headers(token))

    try:
        session.get(_XUEQIU_BASE, timeout=10)
    except requests.RequestException:
        pass

    resp = session.get(
        _COMMENTS_API,
        params={"id": status_id, "count": _MAX_COMMENTS, "page": 1},
        timeout=15,
    )

    if resp.status_code in (403, 401):
        raise _XueqiuAPIBlocked(f"Comments HTTP {resp.status_code}")

    data = resp.json()

    if isinstance(data, dict) and data.get("error_code"):
        raise _XueqiuAPIBlocked(
            f"Comments API error: {data.get('error_code')}"
        )

    comments = []
    raw_comments = data.get("comments") or []
    for c in raw_comments:
        comment = {
            "author": (c.get("user", {}) or {}).get("screen_name", "匿名"),
            "text": _strip_html(c.get("text", "")),
            "created_at": _ts_to_datetime(c.get("created_at")),
            "like_count": c.get("like_count", 0),
            "replies": [],
        }
        # Sub-replies
        for r in (c.get("reply_list") or c.get("sub_comments") or []):
            comment["replies"].append({
                "author": (r.get("user", {}) or {}).get("screen_name", "匿名"),
                "text": _strip_html(r.get("text", "")),
                "created_at": _ts_to_datetime(r.get("created_at")),
            })
        comments.append(comment)

    return comments


# ---------------------------------------------------------------------------
# Playwright browser fallback
# ---------------------------------------------------------------------------
def _try_playwright(query: str, token: str) -> List[dict]:
    """Search Xueqiu via headless browser. Returns list of post dicts with comments.

    Strategy: visit homepage first (to pass WAF), then use the search input,
    click the 讨论 tab, extract posts, then navigate to each post for comments.
    """
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        raise RuntimeError(
            "Playwright 未安装，请运行: pip install playwright && playwright install chromium"
        )

    posts: List[dict] = []

    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=True,
            args=["--disable-blink-features=AutomationControlled"],
        )
        try:
            context = browser.new_context(
                viewport={"width": 1280, "height": 800},
                user_agent=(
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/131.0.0.0 Safari/537.36"
                ),
            )
            context.add_cookies([{
                "name": "xq_a_token",
                "value": token,
                "domain": ".xueqiu.com",
                "path": "/",
            }])

            page = context.new_page()
            page.add_init_script(
                "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"
            )
            page.set_default_timeout(_PLAYWRIGHT_TIMEOUT_MS)

            # --- Step 1: Visit homepage to establish session / pass WAF ---
            page.goto(_XUEQIU_BASE + "/", wait_until="networkidle")
            _time.sleep(2)

            # --- Step 2: Use the search input to type query and submit ---
            search_input = page.query_selector('input[name="q"]')
            if not search_input:
                return []
            search_input.click()
            search_input.fill(query)
            _time.sleep(0.5)
            search_input.press("Enter")
            _time.sleep(4)

            # --- Step 3: Click the 讨论 tab ---
            discussion_tab = page.query_selector('a[href="#/timeline"]')
            if discussion_tab:
                discussion_tab.click()
                _time.sleep(4)

            # --- Step 4: Extract posts from timeline ---
            post_elements = page.query_selector_all("article.timeline__item")

            for el in post_elements[:_MAX_POSTS]:
                try:
                    # Author
                    author_el = el.query_selector(".user-name")
                    screen_name = author_el.inner_text().strip() if author_el else "未知"

                    # Date and post URL
                    date_el = el.query_selector("a.date-and-source")
                    date_str = date_el.inner_text().strip() if date_el else ""
                    href = date_el.get_attribute("href") if date_el else ""
                    post_url = (_XUEQIU_BASE + href) if href and not href.startswith("http") else href

                    # Content preview
                    content_el = el.query_selector(".content--description")
                    content = _strip_html(content_el.inner_text() or "") if content_el else ""

                    # Engagement (likes, comments, shares)
                    stats = el.query_selector_all(".timeline__item__control span")
                    like_count = 0
                    reply_count = 0
                    if len(stats) >= 2:
                        try:
                            reply_count = int(stats[0].inner_text().strip() or 0)
                        except (ValueError, TypeError):
                            pass
                        try:
                            like_count = int(stats[-1].inner_text().strip() or 0)
                        except (ValueError, TypeError):
                            pass

                    # Parse post ID from URL
                    post_id = post_url.rstrip("/").split("/")[-1] if post_url else ""

                    # Parse date string to datetime
                    created_at = _parse_xueqiu_date_str(date_str)

                    posts.append({
                        "id": post_id,
                        "user_id": "",
                        "screen_name": screen_name,
                        "title": "",
                        "text": content,
                        "created_at": created_at,
                        "reply_count": reply_count,
                        "like_count": like_count,
                        "target": post_url,
                        "comments": [],
                    })
                except Exception:
                    continue

            # --- Step 5: For each post, navigate and extract full content + comments ---
            for post in posts[:_MAX_POSTS]:
                if not post.get("target"):
                    continue
                try:
                    page.goto(post["target"], wait_until="domcontentloaded")
                    _time.sleep(3)

                    # Full post content
                    detail_el = page.query_selector(".article__bd__detail")
                    if detail_el:
                        post["text"] = _strip_html(detail_el.inner_text() or "")

                    # Try to expand comments
                    for _ in range(2):
                        try:
                            expand_btn = page.query_selector(
                                "a:has-text('查看更多评论'), "
                                "a:has-text('展开更多'), "
                                "[class*='load-more']"
                            )
                            if expand_btn and expand_btn.is_visible():
                                expand_btn.click()
                                _time.sleep(1.5)
                            else:
                                break
                        except Exception:
                            break

                    # Extract comments
                    comment_items = page.query_selector_all(".comment__item")
                    for ci in comment_items[:_MAX_COMMENTS]:
                        try:
                            c_author_el = ci.query_selector(
                                ".comment__item__main__hd a, .user-name"
                            )
                            c_author = c_author_el.inner_text().strip() if c_author_el else "匿名"

                            c_content_el = ci.query_selector(
                                ".comment__item__main .content, "
                                ".comment__item__main p"
                            )
                            c_text = _strip_html(
                                c_content_el.inner_text() or ""
                            ) if c_content_el else ""

                            post["comments"].append({
                                "author": c_author,
                                "text": c_text,
                                "created_at": None,
                                "like_count": 0,
                                "replies": [],
                            })
                        except Exception:
                            continue
                except Exception:
                    continue

        finally:
            browser.close()

    return posts


# ---------------------------------------------------------------------------
# Date filtering
# ---------------------------------------------------------------------------
def _filter_by_date(
    posts: List[dict], start_date: str, end_date: str
) -> List[dict]:
    """Filter posts by date range. Posts without timestamps are kept."""
    try:
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=1)
    except ValueError:
        return posts  # if dates are unparseable, skip filtering

    filtered = []
    for post in posts:
        dt = post.get("created_at")
        if dt is None:
            filtered.append(post)  # keep posts without timestamps
        elif start_dt <= dt < end_dt:
            filtered.append(post)
    return filtered


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------
def _format_results(
    query: str,
    posts: List[dict],
    start_date: str,
    end_date: str,
) -> str:
    """Format posts + comments into a markdown string for LLM consumption."""
    if not posts:
        return f"未在雪球找到与 '{query}' 相关的帖子（{start_date} 至 {end_date}）"

    lines = [
        f"# 雪球社媒舆情: \"{query}\"",
        f"## 搜索时间范围: {start_date} 至 {end_date}",
        f"## 共找到 {len(posts)} 条相关帖子",
        "",
    ]

    for i, post in enumerate(posts, 1):
        title = post.get("title") or "(无标题)"
        author = post.get("screen_name", "未知")
        time_str = _format_time(post.get("created_at"))
        content = _truncate(post.get("text", ""), _POST_CONTENT_LIMIT)
        likes = post.get("like_count", 0)
        replies = post.get("reply_count", 0)
        url = post.get("target", "")

        lines.append("---")
        lines.append(f"### 帖子 {i}: {title}")
        lines.append(f"- **作者**: {author} | **发布时间**: {time_str}")
        if likes or replies:
            lines.append(f"- **点赞**: {likes} | **评论数**: {replies}")
        if url:
            lines.append(f"- **链接**: {url}")
        lines.append(f"- **内容**: {content}")
        lines.append("")

        # Comments (from HTTP API path or Playwright path)
        comments = post.get("comments", [])
        if comments:
            lines.append(f"#### 评论 (共 {len(comments)} 条):")
            for j, c in enumerate(comments[:10], 1):
                c_author = c.get("author", "匿名")
                c_text = _truncate(c.get("text", ""), _COMMENT_CONTENT_LIMIT)
                c_time = _format_time(c.get("created_at"))
                c_likes = c.get("like_count", 0)
                like_str = f" [{c_likes}赞]" if c_likes else ""
                lines.append(f"{j}. **{c_author}** ({c_time}){like_str}: {c_text}")

                # Sub-replies
                for r in c.get("replies", [])[:3]:
                    r_author = r.get("author", "匿名")
                    r_text = _truncate(r.get("text", ""), _COMMENT_CONTENT_LIMIT)
                    lines.append(f"   - 回复: **{r_author}**: {r_text}")

            lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def fetch_xueqiu_posts(query: str, start_date: str, end_date: str) -> str:
    """Search Xueqiu for stock-related posts and comments.

    Args:
        query: Search term (stock code, company name, or slang).
        start_date: Start date in yyyy-mm-dd format (will be capped to T-7).
        end_date: End date in yyyy-mm-dd format (T).

    Returns:
        Formatted markdown string with posts and comments.
        Never raises — returns a descriptive error string on failure.
    """
    # 强制限定日期范围为近 _MAX_LOOKBACK_DAYS 天，不依赖 LLM 传入的 start_date
    try:
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    except ValueError:
        end_dt = datetime.now()
        end_date = end_dt.strftime("%Y-%m-%d")
    enforced_start_dt = end_dt - timedelta(days=_MAX_LOOKBACK_DAYS)
    start_date = enforced_start_dt.strftime("%Y-%m-%d")

    token = _get_token()
    if not token:
        return "雪球 Token 未配置，请在 .env 中设置 XUEQIU_TOKEN"

    posts: List[dict] = []
    used_browser = False

    # --- Phase 1: Try HTTP API ---
    try:
        posts = _try_http_search(query, token)
    except _XueqiuAPIBlocked:
        posts = []  # will fall through to Playwright
    except Exception:
        posts = []

    # Fetch comments for API-returned posts
    if posts:
        for post in posts[:_MAX_POSTS]:
            if post.get("id"):
                try:
                    post["comments"] = _try_http_comments(post["id"], token)
                except (_XueqiuAPIBlocked, Exception):
                    post["comments"] = []

    # --- Phase 2: Playwright fallback if API returned nothing ---
    if not posts:
        try:
            posts = _try_playwright(query, token)
            used_browser = True
        except RuntimeError as e:
            # Playwright not installed
            return str(e)
        except Exception as e:
            return f"雪球数据获取失败 (API 和浏览器均失败): {str(e)}"

    # --- Phase 3: Filter by date and format ---
    if not used_browser:
        posts = _filter_by_date(posts, start_date, end_date)

    return _format_results(query, posts[:_MAX_POSTS], start_date, end_date)
