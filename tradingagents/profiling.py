"""轻量级运行时性能分析器。

在 analyze_single_stock 结束时打印两张汇总表：
- LLM 调用按 agent 分组：调用次数、总耗时、平均耗时、占比
- 数据源调用按 (method, vendor) 分组：成功/失败次数、总耗时、平均

每支股票分析开始时调用 reset() 清空 collector；
跨进程（multiprocessing.Process）时各子进程独立计数。
"""
import threading
import time
from collections import defaultdict

_lock = threading.Lock()
_llm_records: list = []      # (agent_name, model, elapsed_sec)
_vendor_records: list = []   # (method, vendor, elapsed_sec, ok)
_run_start_ts: float = 0.0


# --- Agent 识别：基于 prompt 头部关键词 ---
# 顺序敏感：先匹配更具体的（避免"分析师"误伤"基本面分析师"）
_AGENT_PATTERNS = [
    ("RM",            ["投资研究总监"]),
    ("PM",            ["投资组合经理"]),
    ("Trader",        ["交易执行专家"]),  # DEPRECATED 但保留识别
    ("Bull",          ["You are a Bull Analyst", "Bull Analyst making the case for"]),
    ("Bear",          ["You are a Bear Analyst", "Bear Analyst making the case against"]),
    ("Liquidity",     ["流动性风险与执行分析师"]),
    ("Event",         ["事件风险与时机分析师"]),
    ("Tail",          ["尾部风险与压力测试分析师"]),
    ("Market",        ["专业技术分析师"]),
    ("Social",        ["专业舆情分析师"]),
    ("News",          ["专业的新闻分析师"]),
    ("Fundamentals",  ["专业的基本面分析师"]),
]


def detect_agent(prompt_text: str) -> str:
    """根据 prompt 头部关键词识别是哪个 agent 调用的 LLM。"""
    head = prompt_text[:2000] if prompt_text else ""
    for name, patterns in _AGENT_PATTERNS:
        if any(p in head for p in patterns):
            return name
    return "Unknown"


def record_llm(prompt_text: str, model: str, elapsed: float) -> None:
    """记录一次 LLM 调用耗时。"""
    agent = detect_agent(prompt_text)
    with _lock:
        _llm_records.append((agent, model, elapsed))


def record_vendor(method: str, vendor: str, elapsed: float, ok: bool) -> None:
    """记录一次 vendor 调用耗时。"""
    with _lock:
        _vendor_records.append((method, vendor, elapsed, ok))


def reset() -> None:
    """开始新一支股票分析前清空。"""
    global _run_start_ts
    with _lock:
        _llm_records.clear()
        _vendor_records.clear()
        _run_start_ts = time.time()


def print_summary(label: str = "") -> None:
    """打印两张汇总表（LLM by agent + Vendor by method,vendor）。"""
    import sys

    wall = time.time() - _run_start_ts if _run_start_ts else 0.0
    out = sys.stderr

    def w(s=""):
        out.write(s + "\n")
        out.flush()

    w()
    w("=" * 72)
    w(f"  性能分析摘要{(' [' + label + ']') if label else ''}")
    w("=" * 72)
    w(f"  Wall clock: {wall:.1f} 秒 ({wall/60:.1f} 分钟)")

    # --- LLM 汇总（按 agent 分组）---
    with _lock:
        llm_snap = list(_llm_records)
        vendor_snap = list(_vendor_records)

    if llm_snap:
        groups = defaultdict(list)
        for agent, _model, elapsed in llm_snap:
            groups[agent].append(elapsed)

        total_llm = sum(e for _, _, e in llm_snap)

        w()
        w("--- LLM 调用（按 agent 分组）---")
        w(f"{'Agent':<14} {'次数':>5} {'总耗时(s)':>10} {'平均(s)':>9} {'占总耗时':>9}")
        w("-" * 60)
        for agent in sorted(groups, key=lambda a: -sum(groups[a])):
            calls = groups[agent]
            total = sum(calls)
            avg = total / len(calls) if calls else 0
            pct = (total / wall * 100) if wall else 0
            w(f"{agent:<14} {len(calls):>5} {total:>10.1f} {avg:>9.1f} {pct:>8.1f}%")
        w("-" * 60)
        w(f"{'合计':<14} {len(llm_snap):>5} {total_llm:>10.1f} "
          f"{total_llm/len(llm_snap):>9.1f} {total_llm/wall*100 if wall else 0:>8.1f}%")

    # --- Vendor 汇总（按 method, vendor 分组）---
    if vendor_snap:
        groups = defaultdict(lambda: {"ok": 0, "fail": 0, "elapsed": 0.0})
        for method, vendor, elapsed, ok in vendor_snap:
            key = (method, vendor)
            if ok:
                groups[key]["ok"] += 1
            else:
                groups[key]["fail"] += 1
            groups[key]["elapsed"] += elapsed

        total_vendor = sum(g["elapsed"] for g in groups.values())

        w()
        w("--- 数据源调用（按 method × vendor 分组）---")
        w(f"{'Method':<22} {'Vendor':<10} {'OK':>4} {'失败':>4} "
          f"{'总耗时(s)':>10} {'占总耗时':>9}")
        w("-" * 72)
        for (method, vendor), g in sorted(groups.items(), key=lambda kv: -kv[1]["elapsed"]):
            pct = (g["elapsed"] / wall * 100) if wall else 0
            w(f"{method:<22} {vendor:<10} {g['ok']:>4} {g['fail']:>4} "
              f"{g['elapsed']:>10.1f} {pct:>8.1f}%")
        w("-" * 72)
        w(f"{'合计':<33} {'':>4} {'':>4} {total_vendor:>10.1f} "
          f"{total_vendor/wall*100 if wall else 0:>8.1f}%")

    # --- 未计入耗时（系统/编排/网络等）---
    if wall:
        accounted = sum(e for _, _, e in llm_snap) + sum(g["elapsed"] for g in
                       defaultdict(lambda: {"elapsed": 0.0}, {(m,v): {"elapsed": el}
                       for m, v, el, _ in vendor_snap}).values())
        # 简化版未计入：wall - LLM 总耗时 - vendor 总耗时（注意 LLM/vendor 可能并发，是粗估）
        total_llm = sum(e for _, _, e in llm_snap)
        total_vendor = sum(e for _, _, e, _ in vendor_snap)
        other = max(0.0, wall - total_llm - total_vendor)
        w()
        w(f"未计入耗时（编排/嵌入/IO/网络其他）: {other:.1f} 秒（粗估，未考虑并发）")
    w("=" * 72)
    w()
