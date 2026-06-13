"""板块对照官（Sector Comparison Officer）——纯 Python 节点，无 LLM。

输入：股票代码 + stock_profile（含 theme_name）
处理：拉本股 + 大盘三档 + 行业 ETF + 主题 ETF + 主题代表股 的近 60 日 OHLCV，
     算 RS（Relative Strength）= 本股 N 日涨幅 − 基准 N 日涨幅
输出：state["sector_comparison"]（markdown 报告）

注入下游（RM/PM）：让 LLM 在评级决策时多一个"板块相对强弱"维度。

设计原则：
- 纯 Python，无 LLM 主观空间
- 用 harness.price_cache 缓存 OHLCV，避免重复拉
- 行业 ETF / 主题 ETF / 主题代表股 全部按规则自动匹配
"""

from __future__ import annotations

import datetime as _dt
import io
import logging
import re
from pathlib import Path
from typing import Optional

import pandas as pd

from tradingagents.dataflows.theme_etf_map import (
    resolve_theme,
    resolve_industry_etf,
    resolve_market_etf_by_ticker,
    DEFAULT_FALLBACK_ETF,
)

logger = logging.getLogger(__name__)

# RS 计算时段（交易日偏移）
_RS_HORIZONS = [("5d", 5), ("30d", 30), ("60d", 60)]


# ---------------------------------------------------------------------------
# Theme parsing
# ---------------------------------------------------------------------------
def _parse_theme_from_profile(stock_profile_md: str) -> str | None:
    """从 stock_profile.md 的 YAML 摘要里抠出 theme_name 字段。"""
    if not stock_profile_md:
        return None
    m = re.search(r"theme_name\s*:\s*([^\n#]+)", stock_profile_md)
    if not m:
        return None
    raw = m.group(1).strip().strip('"').strip("'")
    if raw.lower() in ("none", "null", "—", "-") or raw in ("不在主题", "无"):
        return None
    return raw


def _parse_industry_from_profile(stock_profile_md: str) -> str | None:
    """从 stock_profile.md 的 YAML 摘要里抠出 industry 字段。"""
    if not stock_profile_md:
        return None
    m = re.search(r"\bindustry\s*:\s*([^\n#]+)", stock_profile_md)
    if not m:
        return None
    raw = m.group(1).strip().strip('"').strip("'")
    if raw.lower() in ("none", "null", "—", "-") or raw == "无":
        return None
    return raw


# ---------------------------------------------------------------------------
# Price fetching（用 harness.price_cache 增量缓存）
# ---------------------------------------------------------------------------
def _fetch_ohlcv(ticker: str, base_date: _dt.date) -> pd.DataFrame | None:
    """拉本股或基准 OHLCV（覆盖最长 RS 窗口 60 交易日）。优先用 harness.price_cache 命中缓存。

    窗口须覆盖 max(_RS_HORIZONS)=60 交易日 → 需 >61 根 K 线。原取 80 日历日 ≈56 交易日，
    60d 收益恒返 None（所有票 60d 全 N/A 的根）。120 日历日 ≈84 交易日，含春节长假缺口仍稳过 61 根。
    """
    try:
        from tradingagents.harness import price_cache as _pcache
        start = base_date - _dt.timedelta(days=120)
        end = base_date
        df = _pcache.fetch_with_cache(ticker, start, end)
        return df
    except Exception as e:
        logger.warning("拉 %s 价格失败: %s", ticker, e)
        return None


# ---------------------------------------------------------------------------
# RS computation
# ---------------------------------------------------------------------------
def _compute_return_pct(df: pd.DataFrame, n_days: int) -> float | None:
    """计算近 n_days 个交易日的累计收益率（%）。

    df 按 Date 升序，取最后一行作为当前价，倒推 n_days 行作为基准。
    """
    if df is None or "Close" not in df.columns:
        return None
    closes = pd.to_numeric(df["Close"], errors="coerce").dropna().reset_index(drop=True)
    if len(closes) <= n_days:
        return None
    current = float(closes.iloc[-1])
    past = float(closes.iloc[-n_days - 1])
    if past <= 0:
        return None
    return round((current / past - 1) * 100, 2)


def _format_pct(v: float | None) -> str:
    if v is None:
        return "N/A"
    return f"{v:+.1f}%"


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------
def _format_report(
    ticker: str,
    company_name: str,
    trade_date: str,
    theme_match: dict,
    industry_name: str | None,
    industry_etf: str | None,
    returns_table: dict[str, dict[str, float | None]],
    benchmark_labels: dict[str, str],
    fallback_log: list[str],
) -> str:
    """拼装 markdown 报告。

    returns_table: {ticker: {"5d": float, "30d": float, "60d": float}}
    benchmark_labels: {ticker: 显示名}
    """
    lines: list[str] = []
    lines.append(f"# 板块对照报告 - {ticker} {company_name}")
    lines.append("")
    lines.append(f"**分析日期**：{trade_date}")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 一、对照集匹配路径（fallback 层级）")
    lines.append("")
    for entry in fallback_log:
        lines.append(f"- {entry}")
    lines.append("")
    matched = theme_match.get("matched_theme")
    if matched:
        peers = theme_match.get("peers", []) or []
        etf = theme_match.get("etf")
        lines.append(f"**最终对照集**：主题=**{matched}** / 主题 ETF={etf or '无'} / 主题代表股 {len(peers)} 只 / industry={industry_name or '无'} → {industry_etf or '无'}")
    else:
        lines.append(f"**最终对照集**：industry={industry_name or '无'} → {industry_etf or '无'}（主题降级）")
    lines.append("")

    # 拼涨幅对照表
    lines.append("## 二、近期涨跌幅对照（本股 vs 所有基准）")
    lines.append("")

    # 列顺序：本股 + 大盘三档 + 行业 ETF + 主题 ETF + 主题代表股
    cols = list(returns_table.keys())
    header = "| 时段 | " + " | ".join(benchmark_labels.get(t, t) for t in cols) + " |"
    sep = "|------|" + "|".join(["---"] * len(cols)) + "|"
    lines.append(header)
    lines.append(sep)
    for h, _n in _RS_HORIZONS:
        cells = [_format_pct(returns_table[c].get(h)) for c in cols]
        lines.append(f"| {h:5s} | " + " | ".join(cells) + " |")
    lines.append("")

    # 拼 RS 表（RS = 本股 - 基准；只对非本股列）
    if len(cols) > 1:
        lines.append("## 三、相对强弱 RS = 本股收益 − 基准收益")
        lines.append("")
        rs_cols = [c for c in cols if c != ticker]
        rs_header = "| 时段 | " + " | ".join(f"vs {benchmark_labels.get(t, t)}" for t in rs_cols) + " |"
        rs_sep = "|------|" + "|".join(["---"] * len(rs_cols)) + "|"
        lines.append(rs_header)
        lines.append(rs_sep)
        for h, _n in _RS_HORIZONS:
            row = []
            for c in rs_cols:
                self_v = returns_table[ticker].get(h)
                bench_v = returns_table[c].get(h)
                if self_v is None or bench_v is None:
                    row.append("N/A")
                else:
                    row.append(f"{self_v - bench_v:+.1f}%")
            lines.append(f"| {h:5s} | " + " | ".join(row) + " |")
        lines.append("")

    # 自动判定
    lines.append("## 四、判定")
    lines.append("")
    self_ret_30 = returns_table[ticker].get("30d")
    if self_ret_30 is not None:
        # vs 沪深300
        hs300_ret = returns_table.get("510300", {}).get("30d")
        if hs300_ret is not None:
            rs_vs_hs300 = self_ret_30 - hs300_ret
            verdict = "✓ 跑赢大盘" if rs_vs_hs300 > 0 else "✗ 跑输大盘"
            lines.append(f"- vs 沪深300（30d）：{verdict}（RS = {rs_vs_hs300:+.1f}%）")
        # vs 主题 ETF（如有）
        theme_etf = theme_match.get("etf")
        if theme_etf and theme_etf in returns_table:
            theme_ret = returns_table[theme_etf].get("30d")
            if theme_ret is not None:
                rs_vs_theme = self_ret_30 - theme_ret
                verdict = "✓ 跑赢主题板块" if rs_vs_theme > 0 else "✗ 跑输主题板块"
                lines.append(f"- vs 主题 ETF（30d）：{verdict}（RS = {rs_vs_theme:+.1f}%）")

        # 主题内排名（跟 peers 比 30d 收益）
        peers = theme_match.get("peers", []) or []
        # 防止本股自己出现在 peers 里被重复计入
        peers = [p for p in peers if p != ticker]
        peer_returns = [
            (p, returns_table.get(p, {}).get("30d"))
            for p in peers if returns_table.get(p, {}).get("30d") is not None
        ]
        if peer_returns:
            all_returns = [(ticker, self_ret_30)] + peer_returns
            sorted_returns = sorted(all_returns, key=lambda x: x[1], reverse=True)
            rank = next(i for i, (t, _) in enumerate(sorted_returns) if t == ticker) + 1
            lines.append(
                f"- 主题内 30d 收益排名：**第 {rank} / {len(all_returns)}**（"
                + " > ".join(f"{t}({v:+.1f}%)" for t, v in sorted_returns)
                + "）"
            )

    lines.append("")
    lines.append("> 用法：RM 在 Step 6 评级 COT 时引用 RS 判定；PM 在 Trade Ticket 入场判断时引用主题板块强弱。")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 主节点
# ---------------------------------------------------------------------------
def create_sector_comparison_node():
    """工厂函数：返回 sector_comparison 节点。纯 Python，无需 LLM 参数。"""

    def sector_comparison_node(state) -> dict:
        ticker = state["company_of_interest"]
        company_name = state.get("company_name", "")
        trade_date = state.get("trade_date", "")
        stock_profile = state.get("stock_profile", "")

        # 1. 解析主题 + 行业
        theme_name = _parse_theme_from_profile(stock_profile)
        industry_name = _parse_industry_from_profile(stock_profile)
        theme_match = resolve_theme(theme_name)

        # 2. 按 fallback 层级决定对照集（精 → 粗）
        all_tickers: list[str] = [ticker]
        labels: dict[str, str] = {ticker: ticker}
        fallback_log: list[str] = []

        # 层级 1：主题命中（最精细）
        theme_etf = theme_match.get("etf")
        theme_peers = theme_match.get("peers", []) or []
        # 防止本股出现在 peers 里被重复计入
        theme_peers = [p for p in theme_peers if p != ticker]
        matched_theme = theme_match.get("matched_theme")
        if matched_theme:
            fallback_log.append(f"层级1 主题命中: {matched_theme}")
            if theme_etf and theme_etf not in all_tickers:
                all_tickers.append(theme_etf)
                labels[theme_etf] = f"{theme_etf}主题ETF"
            for peer in theme_peers:
                if peer not in all_tickers:
                    all_tickers.append(peer)
                    labels[peer] = peer
        else:
            fallback_log.append(f"层级1 主题未命中（theme_name={theme_name or 'None'}），降级到行业匹配")

        # 层级 2：行业 ETF（当 1 没命中或主题没专门 ETF 时启用）
        # 即使主题命中但有时主题 ETF 不存在（如 PCB），仍可加行业 ETF 做补充
        industry_etf = resolve_industry_etf(industry_name)
        if industry_etf and industry_etf not in all_tickers:
            all_tickers.append(industry_etf)
            labels[industry_etf] = f"{industry_etf}行业ETF"
            fallback_log.append(f"层级2 行业命中: {industry_name} → {industry_etf}")
        elif not matched_theme:
            fallback_log.append(f"层级2 行业未命中（industry={industry_name or 'None'}），降级到市场指数")

        # 层级 3：本股所在市场指数（按 ticker 段判断）
        market = resolve_market_etf_by_ticker(ticker)
        if market is not None:
            mkt_code, mkt_label = market
            if mkt_code not in all_tickers:
                all_tickers.append(mkt_code)
                labels[mkt_code] = mkt_label
            fallback_log.append(f"层级3 市场指数: {mkt_label}（{mkt_code}）")
        else:
            fallback_log.append("层级3 市场指数: 无（北交所等其他）")

        # 层级 4：兜底加沪深300（如果上面没把它包括进来）
        fb_code, fb_label = DEFAULT_FALLBACK_ETF
        if fb_code not in all_tickers:
            all_tickers.append(fb_code)
            labels[fb_code] = fb_label
            fallback_log.append(f"层级4 大盘兜底: {fb_label}（{fb_code}）")

        # 3. 拉所有 ticker 的 OHLCV
        try:
            base_date = _dt.date.fromisoformat(trade_date)
        except (ValueError, TypeError):
            base_date = _dt.date.today()

        returns_table: dict[str, dict[str, float | None]] = {}
        for t in all_tickers:
            df = _fetch_ohlcv(t, base_date)
            returns_table[t] = {h: _compute_return_pct(df, n) for h, n in _RS_HORIZONS}

        # 4. 拼装 markdown
        report = _format_report(
            ticker, company_name, trade_date,
            theme_match, industry_name, industry_etf,
            returns_table, labels, fallback_log,
        )

        return {"sector_comparison": report}

    return sector_comparison_node
