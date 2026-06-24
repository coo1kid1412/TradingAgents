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


def _parse_iso_date(s) -> Optional["datetime.date"]:
    """宽松解析 YYYY-MM-DD / YYYY/MM/DD / YYYY.MM.DD / YYYYMMDD → date；失败/季度/未知 → None。"""
    import datetime
    if s in (None, "", "未知", "null"):
        return None
    text = str(s).strip()
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y.%m.%d", "%Y%m%d"):
        try:
            return datetime.datetime.strptime(text[:10] if "-" in text or "/" in text or "." in text
                                              else text[:8], fmt).date()
        except ValueError:
            continue
    return None


def _recency_weight(source_date, current_date) -> float:
    """新闻见报日期 → 新鲜度权重（对标投研：信息越陈旧，边际驱动力越弱）。

    ≤7 日=1.0 / ≤21 日=0.75 / ≤45 日=0.5 / >45 日=0.35。
    缺见报日期或当前日期无法解析 → 1.0（向后兼容，不惩罚缺失，只在有日期时生效）。
    与 priced_in 是不同的轴：priced_in 问"市场吸收没"，recency 问"信息新不新"。
    """
    sd = _parse_iso_date(source_date)
    cd = _parse_iso_date(current_date)
    if sd is None or cd is None:
        return 1.0
    age = (cd - sd).days
    if age <= 7:
        return 1.0
    if age <= 21:
        return 0.75
    if age <= 45:
        return 0.5
    return 0.35


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


def aggregate_news_catalyst(news_report: str, current_date: Optional[str] = None) -> Optional[dict]:
    """把 SUMMARY.key_events 聚合成净催化信号。

    单事件分 = impact × 可信度 × (1 − priced_in%) × 时间窗权重 × 新鲜度权重。
    已定价的事件不再驱动（priced_in 高→权重低）；可信度低、远端的事件权重低；
    见报日期(source_date)越陈旧→新鲜度权重越低（缺日期或缺 current_date 则不衰减）。

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
        rec = _recency_weight(e.get("source_date"), current_date)
        net += impact * cred * (1.0 - priced_frac) * hz * rec
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


# 卖方盈利预期上修/下修关键词（成长股 ride 真判据——对标投研：revision 方向决定该骑还是该收，
# 而非 TTM 后视镜增速）。从新闻 SUMMARY 的累积模式 + 机构类事件里抽，做粗代理(零成本，无 report_rc)。
_REVISION_UP_KW = (
    "评级上调", "上调评级", "上调目标价", "目标价上调", "上修", "盈利预测上调",
    "调高盈利", "上调盈利", "一致预期改善", "预期改善", "上调至", "增持评级",
)
_REVISION_DOWN_KW = (
    "评级下调", "下调评级", "下调目标价", "目标价下调", "下修", "盈利预测下调",
    "调低盈利", "下调盈利", "一致预期恶化", "预期恶化", "下调至", "减持评级",
)


def compute_earnings_revision(news_report: str) -> Optional[dict]:
    """从新闻 SUMMARY 抽确定性"盈利预期修正方向"（上修/停修/下修）——喂 regime earnings 腿。

    背景：earnings 腿原只看 TTM 后视镜增速，主升浪里龙头单季高基数回落被判 decelerating→-1，
    但卖方此时常在**上修前瞻预期**（真正的 ride 判据）。这条把"上修方向"确定性抽出来，让前瞻
    修正能中和后视镜减速（见 compute_valuation_regime 3c）。report_rc 没权限前的零成本粗代理。

    取数优先级：① cumulative_patterns（agent 已蒸馏的"多次评级上调/下调"≥2家口径，可信度高）；
    ② 机构类 key_events 标题。两者命中关键词计票，净方向定上修/停修/下修。

    Returns: {direction: 上修/停修/下修, score: up-down, up, down, evidence:[...]}
             或 None（无 SUMMARY 块）。
    """
    summary = _find_summary_yaml(news_report)
    if not summary:
        return None
    blobs: list[str] = []
    patterns = summary.get("cumulative_patterns")
    if isinstance(patterns, list):
        blobs += [str(p) for p in patterns if p]
    for e in (summary.get("key_events") or []):
        if isinstance(e, dict) and "机构" in str(e.get("category", "")):
            blobs.append(str(e.get("title", "")))
    up = down = 0
    evidence: list[str] = []
    for blob in blobs:
        if any(k in blob for k in _REVISION_UP_KW):
            up += 1
            evidence.append(blob[:30])
        if any(k in blob for k in _REVISION_DOWN_KW):
            down += 1
            evidence.append(blob[:30])
    if up == 0 and down == 0:
        return {"direction": "停修", "score": 0, "up": 0, "down": 0, "evidence": []}
    direction = "上修" if up > down else ("下修" if down > up else "停修")
    return {"direction": direction, "score": up - down, "up": up, "down": down,
            "evidence": evidence[:3]}


_SYS_REVISION_RE = re.compile(r"SYS_EARNINGS_REVISION:\s*(?P<dir>上修|停修|下修)")


def parse_sys_earnings_revision(text: str) -> Optional[str]:
    """从注入文本解析 SYS_EARNINGS_REVISION 方向回来（下游确定性消费用）。"""
    if not text:
        return None
    m = _SYS_REVISION_RE.search(text)
    return m.group("dir") if m else None


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


_IMPACT_SIGN = {"+大": "+", "+中": "+", "+小": "+", "0": "·",
                "-小": "-", "-中": "-", "-大": "-"}


def aggregate_catalyst_calendar(news_report: str, max_items: int = 6) -> Optional[list[dict]]:
    """从 SUMMARY.key_events 抽确定性催化日历——对标投研"催化剂日历驱动仓位时机/止损"。

    只收**有日期、且 thesis 相关度≥相关、impact≠0** 的事件，按日期排序（未知日期排末）。
    每条：{date, title, direction(+/-/·), impact, thesis_relevance, priced_in_p}。
    供 PM 时间止损/监控段直读，不再让 LLM 凭空写"下一验证点"。
    """
    summary = _find_summary_yaml(news_report)
    if not summary:
        return None
    events = summary.get("key_events") or []
    if not isinstance(events, list):
        return None

    cal = []
    for e in events:
        if not isinstance(e, dict):
            continue
        impact = str(e.get("impact", "0")).strip()
        if impact in ("0", "", "null", None):
            continue
        relevance = str(e.get("thesis_relevance", "")).strip()
        if relevance not in ("核心", "相关"):
            continue   # 只要 thesis 相关的（边缘/无标的不进监控）
        date = str(e.get("event_date", "未知")).strip() or "未知"
        cal.append({
            "date": date,
            "title": str(e.get("title", ""))[:30],
            "direction": _IMPACT_SIGN.get(impact, "·"),
            "impact": impact,
            "thesis_relevance": relevance,
            "priced_in_p": e.get("priced_in_p"),
        })

    if not cal:
        return None
    # 有日期的优先且按日期升序；未知日期排末（保持原序）
    dated = [c for c in cal if c["date"] != "未知"]
    undated = [c for c in cal if c["date"] == "未知"]
    dated.sort(key=lambda c: c["date"])
    return (dated + undated)[:max_items]


# 舆情措辞 → 方向符号（社媒用 偏多/偏空/分歧；新闻用 正面/负面/中性）
_SENTIMENT_SIGN = {
    "偏多": 1, "正面": 1, "看多": 1,
    "偏空": -1, "负面": -1, "看空": -1,
    "分歧": 0, "中性": 0, "无明显": 0,
}
# sentiment_trend_7d（-100~+100）转向多少算"明显切换"——对标投研把 7 日舆情动能
# 超过约 1/3 量程视为风向已变（leading，先于价格/基本面确认）
_NARRATIVE_TREND_TH = 30.0


def _sentiment_sign(value) -> Optional[int]:
    if value is None:
        return None
    s = str(value).strip()
    for k, v in _SENTIMENT_SIGN.items():
        if k in s:
            return v
    return None


def compute_narrative_shift(sentiment_report: str, news_report: str) -> Optional[dict]:
    """叙事切换早期预警——舆情水位 vs 7 日动能/新闻论调的背离，先于价格预警。

    对标投研"风向先于价格"：人群水位还偏多、但 7 日动能已转负，或新闻论调先于社媒
    转空（聪明钱/卖方叙事先变），是顶部叙事见顶回落的领先信号；反之是底部筑底回升。
    这不是评级信号（不参与确定性评级链），是给 PM 监控段的**早期预警观察项**。

    输入：社媒 SUMMARY(net_sentiment 偏多/偏空/分歧 + sentiment_trend_7d -100~100)、
          新闻 SUMMARY(net_sentiment 正面/负面/中性)。
    Returns: {status(见顶回落预警/筑底回升预警/无明显切换), trend_7d, social_sign,
              news_sign, note} 或 None（两份都缺 SUMMARY）。
    """
    s_sum = _find_summary_yaml(sentiment_report) or {}
    n_sum = _find_summary_yaml(news_report) or {}
    if not s_sum and not n_sum:
        return None

    social_sign = _sentiment_sign(s_sum.get("net_sentiment"))
    news_sign = _sentiment_sign(n_sum.get("net_sentiment"))
    try:
        trend = float(s_sum.get("sentiment_trend_7d")) if s_sum.get("sentiment_trend_7d") not in (None, "null", "") else None
    except (TypeError, ValueError):
        trend = None

    if social_sign is None and news_sign is None and trend is None:
        return None

    reasons = []
    bearish = bearish_news = bullish = bullish_news = False
    # 见顶回落：水位未转空(≥0) 但 7 日动能明显转负
    if social_sign is not None and social_sign >= 0 and trend is not None and trend <= -_NARRATIVE_TREND_TH:
        bearish = True
        reasons.append(f"舆情水位仍{'偏多' if social_sign>0 else '中性'}但 7 日动能 {trend:.0f}≤-{_NARRATIVE_TREND_TH:.0f}（动能先转负）")
    # 新闻论调先于社媒转空（聪明钱/卖方叙事先变）
    if news_sign is not None and news_sign < 0 and social_sign is not None and social_sign > 0:
        bearish_news = True
        reasons.append("新闻论调已转空、社媒仍偏多（叙事先于人群转向）")
    # 筑底回升：水位未转多(≤0) 但 7 日动能明显转正
    if social_sign is not None and social_sign <= 0 and trend is not None and trend >= _NARRATIVE_TREND_TH:
        bullish = True
        reasons.append(f"舆情水位仍{'偏空' if social_sign<0 else '中性'}但 7 日动能 {trend:.0f}≥+{_NARRATIVE_TREND_TH:.0f}（动能先转正）")
    if news_sign is not None and news_sign > 0 and social_sign is not None and social_sign < 0:
        bullish_news = True
        reasons.append("新闻论调已转多、社媒仍偏空（叙事先于人群转向）")

    if bearish or bearish_news:
        status = "见顶回落预警"
    elif bullish or bullish_news:
        status = "筑底回升预警"
    else:
        status = "无明显切换"

    return {
        "status": status,
        "trend_7d": trend,
        "social_sign": social_sign,
        "news_sign": news_sign,
        "note": "；".join(reasons) if reasons else "舆情水位与动能、新闻论调方向一致，无背离",
    }
