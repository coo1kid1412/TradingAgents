"""股票画像程序化判定（pure deterministic Python，无 LLM）

为 stock_profile_node 提供"已确定字段"，杜绝 LLM 自由裁定导致的同股不同跑次抖动。

确定性字段（Python 直接判，LLM 不允许覆盖）：
- market_cap_tier: 总市值阈值
- liquidity: 日均成交额阈值
- style: 基于 market_cap + momentum + vol 联合判定
- peak_signal: RSI 极端 + 乖离率 + 量价背离 联合检测
- report_weights_base: style 查表

仍由 LLM 负责的字段：
- industry / industry framework card 识别（需要语义匹配）
- theme_name（需要 news 上下文）
- event_windows（需要 news/fundamentals 上下文）
- 文字说明 + 行业风险叙述

特别约定：
- "ETF/LOF" 直接由 ticker 模式识别（5 位数字开头 51/15/56/58）
- "cyclical" 由 LLM 在行业卡基础上加注，Python 默认不输出
"""

from __future__ import annotations

from typing import Optional


# ---------------------------------------------------------------------------
# 市值层级
# ---------------------------------------------------------------------------
def compute_market_cap_tier(market_cap_yi: Optional[float]) -> Optional[str]:
    """根据总市值（亿元）判定市值层级。"""
    if market_cap_yi is None or market_cap_yi <= 0:
        return None
    if market_cap_yi >= 1000:
        return "large_cap"
    if market_cap_yi >= 200:
        return "mid_cap"
    if market_cap_yi >= 30:
        return "small_cap"
    return "micro_cap"


def market_cap_tier_label(tier: Optional[str]) -> str:
    return {
        "large_cap": "大盘（>1000 亿）",
        "mid_cap": "中盘（200-1000 亿）",
        "small_cap": "小盘（30-200 亿）",
        "micro_cap": "微盘（<30 亿）",
    }.get(tier or "", "未知")


# ---------------------------------------------------------------------------
# 流动性档
# ---------------------------------------------------------------------------
def compute_liquidity_tier(avg_daily_turnover_yi: Optional[float]) -> Optional[str]:
    """根据近 60 日日均成交额（亿元）判定流动性档。"""
    if avg_daily_turnover_yi is None or avg_daily_turnover_yi <= 0:
        return None
    if avg_daily_turnover_yi >= 5:
        return "deep"
    if avg_daily_turnover_yi >= 0.5:
        return "medium"
    return "shallow"


def liquidity_tier_label(tier: Optional[str]) -> str:
    return {
        "deep": "深（日均 > 5 亿）",
        "medium": "中（日均 5000 万-5 亿）",
        "shallow": "浅（日均 < 5000 万）",
    }.get(tier or "", "未知")


# ---------------------------------------------------------------------------
# 风格 style
# ---------------------------------------------------------------------------
def derive_style(
    market_cap_tier: Optional[str],
    momentum_score: Optional[float],
    lowvol_score: Optional[float],
    liquidity_tier: Optional[str],
    is_etf: bool = False,
) -> Optional[str]:
    """基于市值 + 量化动量分 + 量化低波分（反向）+ 流动性 联合判定 style。

    cyclical 不在此处自动识别——需要 LLM 根据 industry 字段叠加判断（如银行/钢铁/煤炭等）。
    Python 默认输出基础 style，LLM 仅当 industry 命中周期清单时可改写为 cyclical。
    """
    if is_etf:
        return "etf"
    if liquidity_tier == "shallow":
        return "illiquid"
    if market_cap_tier is None or momentum_score is None or lowvol_score is None:
        return None

    # lowvol_score 反向：分高 = 波动低；分低 = 波动高
    is_high_vol = lowvol_score < 40
    is_very_high_vol = lowvol_score < 25
    is_high_momentum = momentum_score >= 65
    is_very_high_momentum = momentum_score >= 80

    # 题材炒作只锁定"小盘 / 微盘 + 极端动量 + 极端波动"
    # 大盘 / 中盘即使动量极强也归 high_beta_growth（如澜起 / 寒武纪都是实业 growth）
    if market_cap_tier in ("micro_cap", "small_cap") and is_very_high_momentum and is_very_high_vol:
        return "theme_speculation"

    # 中盘 + 小盘 + 中等以上动量 / 波动 → high_beta_growth
    if market_cap_tier in ("micro_cap", "small_cap"):
        return "high_beta_growth"  # 小盘默认偏成长

    # mid_cap / large_cap
    if is_high_momentum or is_high_vol:
        return "high_beta_growth"
    return "blue_chip"


# ---------------------------------------------------------------------------
# Peak 信号检测（强制 conservative）
# ---------------------------------------------------------------------------
def detect_peak_signals(
    rsi_value: Optional[float],
    rsi_percentile_1y: Optional[float],
    deviation_pct: Optional[float],
    has_vol_divergence: Optional[bool] = None,
) -> dict:
    """检测当前是否处于技术面"顶部期"（peak）。

    任一强信号触发 → force_peak = True，下游 stock_profile 必须把 theme_stage 设为 peak
    （即使 LLM 觉得是 acceleration）。

    Args:
        rsi_value: 当前 RSI 值（0-100）
        rsi_percentile_1y: 近 1 年 RSI 分位（0-100）
        deviation_pct: 当前价相对 20MA 的乖离率（百分比）
        has_vol_divergence: 是否存在量价背离（价格新高但量能萎缩）

    Returns:
        dict: {
            'force_peak': bool,
            'signals_triggered': list[str],
            'rsi_trigger': bool,
            'deviation_trigger': bool,
            'vol_divergence_trigger': bool,
        }
    """
    triggered: list[str] = []
    rsi_trigger = False
    deviation_trigger = False
    vol_div_trigger = False

    # RSI 极端：当前 RSI > 80 或 1Y 分位 ≥ 85
    if rsi_value is not None and rsi_value >= 80:
        rsi_trigger = True
        triggered.append(f"RSI={rsi_value:.1f} ≥ 80（极度超买）")
    elif rsi_percentile_1y is not None and rsi_percentile_1y >= 85:
        rsi_trigger = True
        triggered.append(f"RSI 1Y 分位={rsi_percentile_1y:.0f}% ≥ 85（历史超买极端）")

    # 乖离率极端：|乖离率| ≥ 30%（高弹性成长股）或 ≥ 20%（蓝筹）
    if deviation_pct is not None and abs(deviation_pct) >= 30:
        deviation_trigger = True
        triggered.append(f"乖离率={deviation_pct:+.1f}% ≥ ±30%（价格远超均线）")

    # 量价背离
    if has_vol_divergence:
        vol_div_trigger = True
        triggered.append("量价背离（价格新高但量能萎缩）")

    # 强制 peak 条件：至少 2 个信号触发，或单 RSI 极度超买
    n_signals = sum([rsi_trigger, deviation_trigger, vol_div_trigger])
    force_peak = n_signals >= 2 or (rsi_value is not None and rsi_value >= 88)

    return {
        "force_peak": force_peak,
        "signals_triggered": triggered,
        "rsi_trigger": rsi_trigger,
        "deviation_trigger": deviation_trigger,
        "vol_divergence_trigger": vol_div_trigger,
        "n_signals": n_signals,
    }


# ---------------------------------------------------------------------------
# REPORT_WEIGHTS 基础值（按 style 查表）
# ---------------------------------------------------------------------------
STYLE_DEFAULT_WEIGHTS: dict[str, dict[str, int]] = {
    "blue_chip":         {"fundamentals": 45, "market": 25, "news": 20, "sentiment": 10},
    "high_beta_growth":  {"fundamentals": 35, "market": 30, "news": 20, "sentiment": 15},
    "theme_speculation": {"fundamentals": 15, "market": 30, "news": 25, "sentiment": 30},
    "cyclical":          {"fundamentals": 30, "market": 25, "news": 30, "sentiment": 15},
    "illiquid":          {"fundamentals": 25, "market": 35, "news": 15, "sentiment": 25},
    "etf":               {"fundamentals": 15, "market": 45, "news": 30, "sentiment": 10},
}


def get_default_weights(style: Optional[str]) -> Optional[dict[str, int]]:
    """根据 style 返回 4 份报告的基础权重（整数，加总 = 100）。"""
    if style is None:
        return None
    return STYLE_DEFAULT_WEIGHTS.get(style, None)


# ---------------------------------------------------------------------------
# 价格 / 估值数据提取（从 OHLCV / fundamentals 原始数据）
# ---------------------------------------------------------------------------
def compute_price_signals(price_df) -> dict:
    """从 OHLCV DataFrame 提取技术面信号：RSI / 乖离率 / 量价背离 / 60 日日均成交额。

    依赖 stockstats（已是项目依赖）来算 RSI。
    """
    import pandas as pd

    result = {
        "rsi_value": None,
        "rsi_percentile_1y": None,
        "deviation_pct": None,
        "has_vol_divergence": None,
        "avg_daily_turnover_yi": None,
    }
    if price_df is None or len(price_df) == 0 or "Close" not in price_df.columns:
        return result

    closes = pd.to_numeric(price_df["Close"], errors="coerce").dropna().reset_index(drop=True)
    if len(closes) < 30:
        return result

    # 20-MA 乖离率
    if len(closes) >= 20:
        ma20 = closes.tail(20).mean()
        if ma20 > 0:
            result["deviation_pct"] = float((closes.iloc[-1] / ma20 - 1) * 100)

    # RSI（用 stockstats 算）
    try:
        from stockstats import StockDataFrame
        df_for_rsi = price_df.copy()
        # stockstats 需要小写列名
        df_for_rsi.columns = [c.lower() for c in df_for_rsi.columns]
        if "close" in df_for_rsi.columns:
            sdf = StockDataFrame.retype(df_for_rsi)
            rsi_series = sdf["rsi_14"]
            if rsi_series is not None and len(rsi_series.dropna()) > 0:
                rsi_clean = rsi_series.dropna()
                result["rsi_value"] = float(rsi_clean.iloc[-1])
                if len(rsi_clean) >= 200:
                    # 1Y 分位 = 当前 RSI 在过去 1 年所有 RSI 值中的百分比排名
                    sorted_rsi = sorted(rsi_clean.tail(252).tolist())
                    pos = sum(1 for v in sorted_rsi if v <= result["rsi_value"])
                    result["rsi_percentile_1y"] = float(pos / len(sorted_rsi) * 100)
    except Exception:
        pass  # 算不出来不影响主流程

    # 量价背离：近 10 日价格创新高 + 同期量能 30 日均量 < 60 日均量
    if "Volume" in price_df.columns and len(price_df) >= 60:
        volumes = pd.to_numeric(price_df["Volume"], errors="coerce").dropna()
        if len(volumes) >= 60:
            recent_high = closes.tail(10).max()
            prior_high = closes.iloc[-30:-10].max() if len(closes) >= 30 else None
            v_recent = volumes.tail(10).mean()
            v_prior = volumes.iloc[-30:-10].mean() if len(volumes) >= 30 else None
            if prior_high is not None and v_prior is not None and v_prior > 0:
                # 价格新高 + 量能萎缩
                price_new_high = recent_high > prior_high
                vol_shrink = (v_recent / v_prior) < 0.85
                result["has_vol_divergence"] = bool(price_new_high and vol_shrink)

    # 60 日日均成交额（亿元）
    if "Volume" in price_df.columns and len(price_df) >= 60:
        # 成交额 = 收盘价 × 成交量；A 股 Volume 单位通常是"股"
        # 转换为亿元：(price × volume) / 1e8
        recent_close = closes.tail(60)
        recent_vol = pd.to_numeric(price_df["Volume"].tail(60), errors="coerce").reset_index(drop=True)
        if len(recent_close) == len(recent_vol) and len(recent_close) > 0:
            turnover = (recent_close.reset_index(drop=True) * recent_vol).dropna()
            if len(turnover) > 0:
                result["avg_daily_turnover_yi"] = float(turnover.mean() / 1e8)

    return result


def parse_market_cap_from_fundamentals(fund_str: str) -> Optional[float]:
    """从原始 fundamentals 字符串中提取总市值（亿元）。

    AKShare get_fundamentals 输出的"公司基本信息"段含 "总市值: XX" 行。
    可能的单位：元 / 万元 / 亿元 —— 用启发式判定。
    """
    import re

    if not fund_str:
        return None

    # 模式 1: "总市值(亿元): 3020.18"（_append_valuation_section 系统计算段格式）
    m = re.search(r"总市值[(（]亿元[)）]\s*[:：]\s*([0-9.]+)", fund_str)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass

    # 模式 2: "总市值: 3030.93亿"（公司基本信息段 EM 源格式）
    m = re.search(r"总市值\s*[:：]\s*([0-9.]+)\s*亿", fund_str)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass

    # 模式 3: "总市值: 30,309,300,000"（纯数字，可能是元 / 万元）
    m = re.search(r"总市值\s*[:：]\s*([0-9.,]+)\b", fund_str)
    if m:
        try:
            v = float(m.group(1).replace(",", ""))
            # 启发式：数值 > 1e10 视为元，转亿元
            if v > 1e10:
                return v / 1e8
            if v > 1e6:
                return v / 1e4  # 万元 → 亿元
            return v  # 已是亿元
        except ValueError:
            pass

    return None


def is_etf_ticker(ticker: str) -> bool:
    """根据 A 股 ETF/LOF 代码模式判断：5 位数字开头 51/15/56/58/16/50。"""
    if not ticker or len(ticker) < 6:
        return False
    head = ticker[:2]
    return head in {"51", "15", "56", "58", "16", "50"}
