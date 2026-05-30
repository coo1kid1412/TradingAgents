"""兄弟股可比 PE 锚：舆情/新闻共现挖掘 + tushare PE 快照缓存 + 清洗中位数。

为 stock_profile 的"目标 PE 上限"提供高质量同业锚（替代去 LLM 报告里抠数的脆弱做法）。

设计约束（2026-05-30 实测确认）：
- tushare `daily_basic` 限流：1次/分钟、**5次/天**，且是 EOD 收盘数据。
  → PE 快照必须"每天 1 次全市场 + 落盘缓存 + 跨分析共用 + 跨天复用"，
    严禁每 peer / 每次分析单独拉（瞬间爆 5/天）。
- `stock_basic`（名字/行业）变动极少 → 缓存 7 天。

兄弟股选取规则（用户定，宁缺毋滥）：
- 共现频次 ≥ 3 次（在 news + sentiment 文本里）
- 行业交叉校验：peer 与主股同/相关行业（干掉跨行业明星股 + 常用词误匹配）
- 剔除券商/银行/保险/财经平台（研报来源噪音）

PE 中位数清洗规则（用户定）：
- 剔除亏损（PE ≤ 0）、剔除 PE > 250（近谷底微利的失真高 PE）
- 至少 **1** 家有效 peer（宁缺毋滥的严筛在"选兄弟"阶段已做：频次≥3+行业校验）；
  **单标的(n=1)标记 confidence="low"**，下游 Conviction 减一档（无第二家纠偏）；
  ≥2 家则 confidence="normal"。0 家返回 None 走兜底。
"""

from __future__ import annotations

import json
import logging
import os
import statistics
import time
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 缓存目录
# ---------------------------------------------------------------------------
_CACHE_DIR = os.environ.get(
    "PEER_COMPS_CACHE_DIR",
    os.path.join(os.path.dirname(__file__), "..", "..", "harness_data", "peer_comps_cache"),
)
_STOCK_BASIC_TTL_SEC = 7 * 24 * 3600  # 名字/行业表缓存 7 天

# 兄弟股静态表（人工核验的种子；运行时新股查不到则现场挖掘）
_BROTHER_MAP_PATH = os.path.join(os.path.dirname(__file__), "brother_peer_map.json")

# 清洗阈值
_PE_LOW = 0.0       # PE ≤ 0 剔除（亏损）
_PE_HIGH_CAP = 250.0  # PE > 250 剔除（近谷底失真）
_MIN_VALID_PEERS = 1  # 至少 1 家有效（单标的→低置信，下游 Conviction 减一档）
_MIN_COMENTION_FREQ = 3  # 共现频次门槛

# 券商/财经平台黑名单（研报来源噪音，名字不含"证券"的那部分）
_FINANCIAL_MEDIA_BLOCK = {
    "东方财富", "同花顺", "大智慧", "指南针", "财富趋势", "顶点软件", "恒生电子",
    "太平洋", "中国银河", "中信建投", "中金公司", "锦龙股份", "哈投股份",
}


def _ensure_cache_dir() -> None:
    os.makedirs(_CACHE_DIR, exist_ok=True)


def is_financial_media(name: str) -> bool:
    """券商/银行/保险/信托/财经平台 → True（研报来源噪音，不作可比同业）。"""
    return (
        "证券" in name or "银行" in name or "保险" in name or name.endswith("信托")
        or name in _FINANCIAL_MEDIA_BLOCK
    )


# ---------------------------------------------------------------------------
# 行业归类（用于交叉校验）
# ---------------------------------------------------------------------------
# 电子科技核心簇：AI/算力链内部互为可比合理（半导体/光通信/连接元件/PCB 等）
_CORE_KEYWORDS = (
    "半导体", "集成电路", "元件", "元器件", "光学", "光电", "光通信", "通信设备",
    "电子", "印制电路", "PCB", "消费电子", "连接器", "电路", "芯片", "光器件",
)


def broad_sector(industry: Optional[str]) -> str:
    """把 tushare/东财 的行业字符串归到一个粗分类，用于 is_related 判断。"""
    s = industry or ""
    for kw in _CORE_KEYWORDS:
        if kw in s:
            return "电子科技"
    for kw, tag in (
        ("证券", "金融"), ("银行", "金融"), ("保险", "金融"), ("多元金融", "金融"),
        ("电网", "电力设备"), ("电力", "电力设备"), ("电气", "电力设备"),
        ("电池", "锂电"), ("锂", "锂电"), ("有色", "有色"), ("稀土", "有色"),
        ("化工", "化工"), ("白酒", "白酒"), ("软件", "软件"), ("汽车", "汽车"),
    ):
        if kw in s:
            return tag
    return s or "未知"


def is_related_industry(industry_a: Optional[str], industry_b: Optional[str]) -> bool:
    """两个行业是否同/相关（同粗分类即可；电子科技核心簇内部都算相关）。"""
    a, b = broad_sector(industry_a), broad_sector(industry_b)
    if a == "未知" or b == "未知":
        return False
    return a == b


# ---------------------------------------------------------------------------
# tushare 快照缓存：stock_basic（名字/行业） & daily_basic（PE）
# ---------------------------------------------------------------------------
def _get_pro():
    """复用 tushare_vendor 的客户端（含限流封装）。"""
    from tradingagents.dataflows.tushare_vendor import _get_tushare_api
    return _get_tushare_api()


def _safe(func, *args, api_name: str = "", **kwargs):
    from tradingagents.dataflows.tushare_vendor import _safe_call
    return _safe_call(func, *args, api_name=api_name, **kwargs)


def get_stock_basic() -> dict[str, dict]:
    """code6 → {name, industry}。缓存 7 天（变动极少）。失败时返回已有缓存或空。"""
    _ensure_cache_dir()
    path = os.path.join(_CACHE_DIR, "stock_basic.json")
    if os.path.exists(path) and (time.time() - os.path.getmtime(path)) < _STOCK_BASIC_TTL_SEC:
        try:
            return json.load(open(path, encoding="utf-8"))
        except Exception:
            pass
    try:
        pro = _get_pro()
        df = _safe(pro.stock_basic, exchange="", list_status="L",
                   fields="ts_code,symbol,name,industry", api_name="stock_basic")
        d = {r["symbol"]: {"name": r["name"], "industry": (r["industry"] or "未知")}
             for _, r in df.iterrows()}
        json.dump(d, open(path, "w", encoding="utf-8"), ensure_ascii=False)
        logger.info("peer_comps: stock_basic 刷新成功 %d 只", len(d))
        return d
    except Exception as e:
        logger.warning("peer_comps: stock_basic 拉取失败(%s)，用旧缓存", str(e)[:80])
        if os.path.exists(path):
            try:
                return json.load(open(path, encoding="utf-8"))
            except Exception:
                pass
        return {}


def get_pe_snapshot(trade_date: str) -> dict[str, Optional[float]]:
    """code6 → pe_ttm。每个交易日 1 次全市场 daily_basic + 落盘。

    限流安全：当天快照已缓存则直接读；未缓存才拉 1 次；拉取失败（5次/天 用尽等）
    则回退到"最近一个已缓存快照"（comp 锚是慢变量，用最近收盘完全够）。
    trade_date: "YYYY-MM-DD" 或 "YYYYMMDD"。
    """
    _ensure_cache_dir()
    ymd = trade_date.replace("-", "")
    # 周末用纯日期计算回退到上周五（零接口调用，避免 daily_basic 在非交易日返回空）；
    # 节假日仍由下方"最近缓存"兜底。
    try:
        from datetime import datetime, timedelta
        d = datetime.strptime(ymd, "%Y%m%d")
        if d.weekday() >= 5:  # 5=周六 6=周日
            d = d - timedelta(days=d.weekday() - 4)
            ymd = d.strftime("%Y%m%d")
    except ValueError:
        pass
    path = os.path.join(_CACHE_DIR, f"pe_{ymd}.json")
    if os.path.exists(path):
        try:
            return json.load(open(path, encoding="utf-8"))
        except Exception:
            pass
    try:
        pro = _get_pro()
        df = _safe(pro.daily_basic, trade_date=ymd, fields="ts_code,pe_ttm",
                   api_name="daily_basic")
        snap = {r["ts_code"][:6]: (float(r["pe_ttm"]) if r["pe_ttm"] == r["pe_ttm"] else None)
                for _, r in df.iterrows()}
        if snap:
            json.dump(snap, open(path, "w", encoding="utf-8"), ensure_ascii=False)
            logger.info("peer_comps: PE 快照 %s 拉取成功 %d 只", ymd, len(snap))
            return snap
    except Exception as e:
        logger.warning("peer_comps: daily_basic %s 失败(%s)，回退最近缓存", ymd, str(e)[:80])
    # 回退：最近一个已缓存快照
    snaps = sorted(f for f in os.listdir(_CACHE_DIR) if f.startswith("pe_") and f.endswith(".json"))
    if snaps:
        try:
            recent = json.load(open(os.path.join(_CACHE_DIR, snaps[-1]), encoding="utf-8"))
            logger.info("peer_comps: 用最近缓存快照 %s（%d 只）", snaps[-1], len(recent))
            return recent
        except Exception:
            pass
    return {}


# ---------------------------------------------------------------------------
# 共现挖掘：从 news + sentiment 文本里挖兄弟股
# ---------------------------------------------------------------------------
def mine_brother_peers(
    text: str,
    self_code: str,
    self_name: str = "",
    min_freq: int = _MIN_COMENTION_FREQ,
    topk: int = 5,
) -> list[tuple[str, str, int]]:
    """从文本挖共现兄弟股。返回 [(code6, name, freq), ...]（已过滤金融/媒体 + 行业校验）。

    规则：频次 ≥ min_freq、行业与主股相关、非金融/媒体、非自身。
    """
    if not text:
        return []
    basic = get_stock_basic()
    if not basic:
        return []
    self_ind = basic.get(self_code, {}).get("industry")
    # name → code（长度≥3 避免 2 字误匹配；排除金融/媒体）
    name2code = {
        info["name"]: code
        for code, info in basic.items()
        if len(info["name"]) >= 3 and "ST" not in info["name"]
        and not is_financial_media(info["name"])
    }
    cand: list[tuple[str, str, int]] = []
    for name, code in name2code.items():
        if code == self_code or name == self_name:
            continue
        c = text.count(name)
        if c < min_freq:
            continue
        # 行业交叉校验
        if self_ind and not is_related_industry(self_ind, basic.get(code, {}).get("industry")):
            continue
        cand.append((code, name, c))
    cand.sort(key=lambda x: -x[2])
    return cand[:topk]


# ---------------------------------------------------------------------------
# 兄弟股 PE 中位数
# ---------------------------------------------------------------------------
def peer_pe_median(
    peer_codes: list[str],
    trade_date: str,
    min_valid: int = _MIN_VALID_PEERS,
) -> Optional[dict]:
    """兄弟股 PE 中位数（清洗：剔亏损/剔PE>250，需 ≥min_valid 家）。

    返回 None 表示有效 peer 不足 → 上层走兜底。
    返回 {median, used:[(code,pe)], dropped:[(code,pe,reason)]}。
    """
    if not peer_codes:
        return None
    snap = get_pe_snapshot(trade_date)
    if not snap:
        return None
    used, dropped = [], []
    for code in peer_codes:
        pe = snap.get(code)
        if pe is None:
            dropped.append((code, None, "无PE/缺失"))
        elif pe <= _PE_LOW:
            dropped.append((code, pe, "亏损PE≤0"))
        elif pe > _PE_HIGH_CAP:
            dropped.append((code, pe, f"PE>{_PE_HIGH_CAP:.0f}失真"))
        else:
            used.append((code, pe))
    if len(used) < min_valid:
        return None
    n_valid = len(used)
    return {
        "median": statistics.median(pe for _, pe in used),
        "used": used,
        "dropped": dropped,
        "n_valid": n_valid,
        # 单标的 = 低置信（无第二家纠偏），下游 Conviction 应减一档
        "confidence": "low" if n_valid == 1 else "normal",
    }


# ---------------------------------------------------------------------------
# 静态兄弟股表（人工核验种子）
# ---------------------------------------------------------------------------
def load_brother_map() -> dict[str, list[str]]:
    """code6 → [peer_code6, ...]（人工核验的静态种子表）。"""
    if os.path.exists(_BROTHER_MAP_PATH):
        try:
            return json.load(open(_BROTHER_MAP_PATH, encoding="utf-8"))
        except Exception:
            pass
    return {}


def get_brother_pe_median(
    target_code: str,
    trade_date: str,
    comention_text: str = "",
    target_name: str = "",
) -> Optional[dict]:
    """主入口：取目标股的兄弟股 PE 中位数。

    兄弟股来源：静态种子表优先；查不到则从 comention_text（news+sentiment）现场挖掘。
    返回 None → 上层走兜底（宽口径行业 / all-null）。
    """
    brothers = load_brother_map().get(target_code)
    if not brothers:
        mined = mine_brother_peers(comention_text, target_code, target_name)
        brothers = [c for c, _, _ in mined]
    if not brothers:
        return None
    res = peer_pe_median(brothers, trade_date)
    if res:
        res["peer_codes"] = brothers
    return res
