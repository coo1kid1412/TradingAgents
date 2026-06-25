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

import re
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


# ---------------------------------------------------------------------------
# Layer 1: 硬规则 —— 亏损股屏蔽 PE / 行业铁律
# ---------------------------------------------------------------------------
def parse_eps_ttm(fund_str: str) -> Optional[float]:
    """从 fundamentals 报告抽 EPS_TTM（每股收益 TTM）。

    匹配优先级（从专属到通用）：
    1. "EPS(TTM): X" / "扣非 EPS(TTM): X"（_append_valuation_section 系统计算段）
    2. "每股收益(TTM): X" / "归母 EPS: X"
    3. "基本每股收益: X"（季报口径，作为兜底）
    """
    import re

    if not fund_str:
        return None

    patterns = [
        r"EPS\s*[(（]\s*TTM\s*[)）]\s*[:：]\s*(-?[0-9.]+)",
        r"每股收益\s*[(（]\s*TTM\s*[)）]\s*[:：]\s*(-?[0-9.]+)",
        r"扣非\s*EPS\s*[(（]\s*TTM\s*[)）]\s*[:：]\s*(-?[0-9.]+)",
        r"归母\s*EPS\s*[:：]\s*(-?[0-9.]+)",
        r"基本每股收益\s*[(（]?元[)）]?\s*[:：]\s*(-?[0-9.]+)",
    ]
    for pat in patterns:
        m = re.search(pat, fund_str)
        if m:
            try:
                v = float(m.group(1))
                # 合理性检查：A 股个股 EPS 一般在 -10 ~ +30 元区间
                if -50 < v < 100:
                    return v
            except ValueError:
                continue
    return None


def is_loss_making(eps_ttm: Optional[float]) -> bool:
    """是否亏损股（EPS_TTM ≤ 0）。EPS 缺失时返回 False（保守）。"""
    return eps_ttm is not None and eps_ttm <= 0


# 行业 → 强制估值范式（仅 PE 失真行业，其他行业 LLM 自由发挥）
# ============================================================================
# 周期股识别与正常化估值（对标投研：周期股 TTM EPS 不能代表正常盈利——
# 林奇铁律：周期股顶部 PE 最低（该卖）、谷底 PE 最高（该买），TTM 口径整个反着）
# ============================================================================

# 传统强周期：tushare stock_basic.industry 关键词匹配（行业粒度够细，安全）
_CYCLICAL_STRONG_INDUSTRY_KW = (
    "钢", "煤", "焦炭", "有色", "铝", "铜", "锌", "铅", "镍", "锂矿",
    "小金属", "稀有金属", "稀土", "黄金",   # 锗/铟/钨/钼/稀土/黄金等资源型，盈利随商品价周期波动
    "化工原料", "化纤", "石油开采", "炼油", "航运", "船舶",
    "水泥", "玻璃", "造纸", "养殖", "畜禽", "猪",
)
# 科技周期：tushare industry 太粗（"半导体"既有周期的存储也有 secular 的澜起/海光），
# 用公司名单确定性判（维护成本低、零误伤）。LLM 仍可对漏网者加注 cyclical（只能加注不能摘帽）。
_CYCLICAL_STRONG_NAMES = (
    "京东方", "TCL科技", "彩虹股份", "深天马",                 # 面板
    "兆易创新", "佰维存储", "江波龙", "德明利", "普冉股份",      # 存储
    "中远海控", "招商轮船",                                   # 海运（industry 关键词外的补充）
)
_CYCLICAL_SEMI_NAMES = (
    "长电科技", "通富微电", "华天科技", "晶方科技",              # 封测（有成长β但跟半导体周期）
    "中芯国际", "华虹",                                       # 代工
    "三安光电",                                               # LED
)


def detect_cyclical(industry: Optional[str], company_name: Optional[str]) -> Optional[str]:
    """周期股确定性识别 → "strong" / "semi" / None。

    Python 优先判（防 LLM 跑间漂移）；LLM 只能对漏网者加注 cyclical，不能摘帽。
    """
    name = (company_name or "").strip()
    if name:
        for kw in _CYCLICAL_STRONG_NAMES:
            if kw in name:
                return "strong"
        for kw in _CYCLICAL_SEMI_NAMES:
            if kw in name:
                return "semi"
    ind = (industry or "").strip()
    if ind:
        for kw in _CYCLICAL_STRONG_INDUSTRY_KW:
            if kw in ind:
                return "strong"
    return None


# ============================================================================
# 范式成长（secular hardtech）识别 —— AI/算力硬科技结构性成长
# ----------------------------------------------------------------------------
# 对标投研：secular re-rating 期间倍数可多年不回归（NVDA/中际旭创式），不能用历史 PE band
# 均值回归估。识别后让 regime 在"真加速期"走 ride-by-default（见 compute_valuation_regime
# 范式反转），但 peak/派发/破位任一出现即回纪律（不骑顶）。
# tushare industry 太粗（分不出 CPO/光刻胶/PCB），主用龙头名单（同周期股，零误伤优先）+
# 少量干净行业关键词。维护：新龙头进名单。AI应用(软件)/固态电池暂不纳入。
# ============================================================================
_PARADIGM_INDUSTRY_KW = ("半导体",)   # tushare industry 里能干净映射的只有半导体；其余靠名单
_PARADIGM_NAMES = (
    # CPO / 光模块 / 光器件
    "中际旭创", "新易盛", "天孚通信", "光迅科技", "华工科技", "剑桥科技", "太辰光", "仕佳光子",
    # 算力 / AI 芯片 / GPU
    "海光信息", "寒武纪", "景嘉微", "龙芯中科", "芯原股份", "瑞芯微", "澜起科技",
    # 光刻胶 / 半导体材料
    "晶瑞电材", "南大光电", "彤程新材", "华懋科技", "雅克科技", "鼎龙股份", "安集科技",
    # PCB（AI 服务器 / 算力）
    "沪电股份", "生益科技", "深南电路", "胜宏科技", "兴森科技", "广合科技",
    # MLCC / 被动元件
    "三环集团", "风华高科", "洁美科技",
    # 算力租赁 / IDC
    "润泽科技", "光环新网", "数据港", "云赛智联", "奥飞数据", "科华数据",
    # 机器人（盈利兑现度较弱；regime 的盈利动能门控会自然过滤未兑现的）
    "汇川技术", "绿的谐波", "埃斯顿", "拓斯达", "鸣志电器", "雷赛智能", "三花智控",
)


def detect_paradigm_growth(industry: Optional[str], company_name: Optional[str]) -> Optional[str]:
    """范式成长确定性识别 → "paradigm" / None。

    ⛔ 周期优先：已是周期股（detect_cyclical 命中，如存储/面板）→ 让位周期轨（返回 None）。
    存储等结构性周期股的成长 β 已由周期轨的滑动权重（谷底偏成长前瞻）表达，不让它双轨打架。
    Python 优先判（防 LLM 漂移）；LLM 只能对漏网者加注，不能摘帽（同周期股）。
    """
    if detect_cyclical(industry, company_name) is not None:
        return None
    name = (company_name or "").strip()
    if name:
        for kw in _PARADIGM_NAMES:
            if kw in name:
                return "paradigm"
    ind = (industry or "").strip()
    if ind:
        for kw in _PARADIGM_INDUSTRY_KW:
            if kw in ind:
                return "paradigm"
    return None


_SYS_PARADIGM_RE = re.compile(r"【SYS_PARADIGM[^】]*】\s*class=(?P<cls>paradigm)")


def parse_sys_paradigm(text: str) -> bool:
    """从 fundamentals 报告解析 SYS_PARADIGM 机读行（Python 转录保证在场）。命中→True。"""
    if not text:
        return False
    return bool(_SYS_PARADIGM_RE.search(text))


_SYS_MAIN_BUSINESS_RE = re.compile(r"【SYS_MAIN_BUSINESS[^】]*】\s*(?P<seg>.+?)（按")


def parse_sys_main_business(text: str) -> Optional[str]:
    """从 fundamentals 原始数据解析 SYS_MAIN_BUSINESS 的产品营收占比段。

    画像识别官据此把它确定性转录到画像末尾，PM 直读画像、不经基本面分析师改写——
    防"分析师转写丢真值"（实测虽多数转录无误，但应去掉这道 LLM 转手）。
    Returns: "芯片量产 47% / 芯片设计 28% / ..." 或 None。
    """
    if not text:
        return None
    m = _SYS_MAIN_BUSINESS_RE.search(text)
    if not m:
        return None
    seg = m.group("seg").strip()
    return seg or None


_AI_MAIN_UPTREND_KEYWORDS = (
    "AI服务器", "AI芯片", "AI算力", "算力", "CPO", "光模块", "光器件", "光通信", "800G", "1.6T", "数据中心",
    "服务器", "交换机", "PCB", "高速互联", "液冷", "英伟达", "NVIDIA",
    "GPU", "云厂商", "半导体", "芯片",
)


def compute_ai_main_uptrend_signal(
    *,
    company_name: Optional[str] = None,
    industry: Optional[str] = None,
    main_business: Optional[str] = None,
    is_paradigm: bool = False,
    net_profit_growth: Optional[float] = None,
    revenue_growth: Optional[float] = None,
    earnings_revision: Optional[str] = None,
    has_hard_order_evidence: bool = False,
    momentum_score: Optional[float] = None,
    theme_stage_inferred: Optional[str] = None,
    sector_rs_30d: Optional[float] = None,
    valuation_regime: Optional[str] = None,
    recurring_loss: Optional[bool] = None,
    has_peak_signal: bool = False,
    retail_concentration_signal: Optional[str] = None,
    rsi_percentile_1y: Optional[float] = None,
    winner_rate_pct: Optional[float] = None,
    capital_flow_regime: Optional[str] = None,
    main_force_streak_days: Optional[int] = None,
) -> dict:
    """识别 AI 算力链主升兑现票（纯确定性信号，不读外部状态）。

    该信号只表达"是否有资格在市场 risk_on 时获得克制升档"，不是买入建议。
    排除条件优先级最高：纪律 regime、扣非亏损、价格 blowoff、下修/资金恶化等都
    会关闭信号，避免把纯叙事或会亏钱的票误抬。
    """
    text = " ".join([company_name or "", industry or "", main_business or ""])
    upper_text = text.upper()
    ai_chain = bool(is_paradigm) or any(kw.upper() in upper_text for kw in _AI_MAIN_UPTREND_KEYWORDS)

    reasons: list[str] = []
    blockers: list[str] = []

    if ai_chain:
        reasons.append("AI算力链/硬科技赛道命中")
    else:
        blockers.append("未命中AI算力链硬科技赛道")

    if net_profit_growth is not None and net_profit_growth >= 0.40:
        reasons.append(f"净利润兑现增长{net_profit_growth * 100:.0f}%")
    if revenue_growth is not None and revenue_growth >= 0.30:
        reasons.append(f"营收兑现增长{revenue_growth * 100:.0f}%")
    if earnings_revision == "上修":
        reasons.append("卖方盈利预期上修")
    if has_hard_order_evidence:
        reasons.append("订单/核心客户/产能放量硬证据")

    has_confirmed_delivery = any(
        marker in reason for reason in reasons
        for marker in ("净利润兑现", "营收兑现", "上修")
    )
    has_delivery = has_confirmed_delivery or has_hard_order_evidence

    trend_reasons: list[str] = []
    if momentum_score is not None and momentum_score >= 65:
        trend_reasons.append(f"momentum={momentum_score:.0f}>=65")
    if theme_stage_inferred == "acceleration":
        trend_reasons.append("theme_stage=acceleration")
    if sector_rs_30d is not None and sector_rs_30d > 5:
        trend_reasons.append(f"板块RS 30d={sector_rs_30d:+.1f}%")
    reasons.extend(trend_reasons)

    reg = (valuation_regime or "").strip().lower()
    if reg == "discipline":
        blockers.append("valuation_regime=discipline")
    if recurring_loss is True:
        blockers.append("扣非/主业亏损")
    if has_peak_signal:
        blockers.append("peak信号触发")
    price_extreme = (
        (rsi_percentile_1y is not None and rsi_percentile_1y >= 85)
        or (winner_rate_pct is not None and winner_rate_pct >= _BLOWOFF_WINNER_EUPHORIA_PCT)
    )
    if retail_concentration_signal == "散户高接盘" and price_extreme:
        blockers.append("散户高接盘+价格极端 blowoff")
    if earnings_revision == "下修":
        blockers.append("卖方盈利预期下修")
    strong_outflow = (
        capital_flow_regime == "恶化"
        or (main_force_streak_days is not None and main_force_streak_days <= -3)
    )
    if strong_outflow and earnings_revision != "上修":
        blockers.append("资金面持续恶化且无卖方上修")

    if not has_delivery:
        blockers.append("缺少业绩/订单/上修兑现证据")
    if not trend_reasons:
        blockers.append("缺少趋势确认")

    enabled = ai_chain and has_delivery and bool(trend_reasons) and not blockers
    if not enabled:
        return {"enabled": False, "class": "none", "reasons": reasons, "blockers": blockers}

    signal_class = "confirmed" if has_confirmed_delivery else "early"
    return {"enabled": True, "class": signal_class, "reasons": reasons, "blockers": []}


# 强周期股目标价的「正常化 vs 成长前瞻」滑动权重，按周期位置确定（防 RM 自选权重致摆动）。
# 结构性上行周期（存储/面板）既有周期风险又有 AI 结构性需求——位置决定该信哪边更多：
#   顶部 → 偏正常化（谨慎，但承认结构性成长，不一杆打到纯正常化）；
#   谷底 → 偏成长（周期底盈利差是常态，看前瞻修复）；中段 → 各半。权重锁死，RM 不得自选。
# ⚠️ TODO(harness 超参优化)：这组档位是人工先验，待 T+30 回测按"周期位置×后续收益"
#   网格搜索校准（尤其顶部 normalized 权重——0.5 偏成长 vs 0.7 偏周期纪律，对存储类
#   结构性周期股的评级方向影响最大）。当前顶部取 0.5（承认 AI 结构需求，顶部谨慎更多
#   交给 regime 派发信号表达，而非估值锚单独打到极低）。
_CYCLICAL_TARGET_WEIGHTS = {
    "top":    (0.5, 0.5),   # (正常化, 成长前瞻)；顶部谨慎但不极端，结构性成长留一半权重
    "mid":    (0.5, 0.5),
    "trough": (0.3, 0.7),
}


def cyclical_target_weights(position: Optional[str]) -> tuple[float, float]:
    """周期位置 → (正常化腿权重, 成长前瞻腿权重)。未知/数据不足 → 偏谨慎 (0.6, 0.4)。"""
    return _CYCLICAL_TARGET_WEIGHTS.get((position or "").strip().lower(), (0.6, 0.4))


_SYS_CYCLICAL_RE = re.compile(
    r"【SYS_CYCLICAL[^】]*】\s*class=(?P<cls>strong|semi)"
    r"(?:\s*\|\s*position=(?P<pos>top|mid|trough|数据不足))?"
    r"(?:\s*\|\s*roe_pct_rank=(?P<rank>[\d.]+))?"
    r"(?:\s*\|\s*roe_10y_median=(?P<roe_med>[-\d.]+)%)?"
    r"(?:\s*\|\s*roe_latest=(?P<roe_now>[-\d.]+)%)?"
    r"(?:\s*\|\s*normalized_eps=(?P<neps>[-\d.]+))?"
    r"(?:\s*\|\s*pe_on_normalized=(?P<npe>[-\d.]+|N/A))?"
)


def parse_sys_cyclical(fund_str: str) -> Optional[dict]:
    """从 fundamentals 报告解析 SYS_CYCLICAL 机读行（Python 转录保证在场）。

    Returns: {class, position, roe_pct_rank, roe_10y_median, roe_latest,
              normalized_eps, pe_on_normalized} 或 None（非周期股/行缺失）。
    pe_on_normalized = 当前价÷正常化EPS（"贵不贵"读数，不是目标倍数）。
    """
    if not fund_str:
        return None
    m = _SYS_CYCLICAL_RE.search(fund_str)
    if not m:
        return None

    def _f(key):
        v = m.group(key)
        if v in (None, "N/A"):
            return None
        try:
            return float(v)
        except ValueError:
            return None

    return {
        "class": m.group("cls"),
        "position": m.group("pos"),
        "roe_pct_rank": _f("rank"),
        "roe_10y_median": _f("roe_med"),
        "roe_latest": _f("roe_now"),
        "normalized_eps": _f("neps"),
        "pe_on_normalized": _f("npe"),
    }


INDUSTRY_FORCED_VALUATION: dict[str, str] = {
    "银行": "pb",
    "保险": "pb",
    "REIT": "ddm",
    "公用事业": "ddm",
    "电力": "ddm",  # 公用事业子集
    "水务": "ddm",
}


def detect_forced_valuation_method(
    industry: Optional[str],
    eps_ttm: Optional[float],
) -> dict:
    """检测是否触发"强制估值范式"硬规则。

    优先级：
    1. 亏损股（EPS_TTM ≤ 0）→ 强制 target_pe_range = null，primary_method 必须是 pb/ps/ev_ebitda
    2. PE 失真行业（银行/保险/REIT/公用事业）→ 强制 primary_method = pb / ddm
    3. 其他 → 不强制，LLM 自由发挥

    Returns:
        dict: {
            'force_valuation': bool,
            'forced_primary_method': Optional[str],   # 强制选这个 primary_method
            'forbid_pe': bool,                         # 是否禁止使用 PE 估值
            'reason': str,
        }
    """
    # 优先级 1: 亏损股
    if is_loss_making(eps_ttm):
        return {
            "force_valuation": True,
            "forced_primary_method": "pb",  # 默认推荐 pb，LLM 可选 ps/ev_ebitda
            "allowed_primary_methods": ["pb", "ps", "ev_ebitda"],
            "forbid_pe": True,
            "reason": f"EPS_TTM = {eps_ttm:.4f} ≤ 0（亏损股 PE 公式失效）",
        }
    # 优先级 2: 行业铁律
    if industry:
        for key, primary in INDUSTRY_FORCED_VALUATION.items():
            if key in industry:
                return {
                    "force_valuation": True,
                    "forced_primary_method": primary,
                    "allowed_primary_methods": [primary],
                    "forbid_pe": (primary != "pe_eps"),
                    "reason": f"行业 '{industry}' 命中 {key} → 必须用 {primary.upper()} 估值（PE 失真行业）",
                }
    return {
        "force_valuation": False,
        "forced_primary_method": None,
        "allowed_primary_methods": None,
        "forbid_pe": False,
        "reason": "",
    }


def parse_growth_quality(fund_str: str) -> dict:
    """从 SYS_GROWTH_QUALITY 行抽扣非口径成长质量（前瞻路由/盈利腿的质量闸）。

    优先读 tushare 确定性行 `【SYS_GROWTH_QUALITY...】 ... recurring_loss=yes/no | 扣非净利YoY年度=X%`；
    缺失时退回散文兜底（"扣非净利润亏损" / "扣非...为负" → recurring_loss）。

    Returns:
        dict: {"recurring_loss": Optional[bool], "deducted_yoy": Optional[float]}
    """
    import re
    out = {"recurring_loss": None, "deducted_yoy": None}
    if not fund_str:
        return out
    m = re.search(r"SYS_GROWTH_QUALITY.*?recurring_loss\s*=\s*(yes|no)", fund_str, re.S)
    if m:
        out["recurring_loss"] = (m.group(1) == "yes")
        m2 = re.search(r"SYS_GROWTH_QUALITY.*?扣非净利YoY年度\s*=\s*(-?[0-9.]+)%", fund_str, re.S)
        if m2:
            try:
                out["deducted_yoy"] = float(m2.group(1)) / 100.0
            except ValueError:
                pass
        return out
    # 散文兜底：扣非亏损/为负
    if re.search(r"扣非[^。\n]{0,12}(亏损|为负|-?\d[\d.,]*\s*亿?元?\s*亏)", fund_str):
        out["recurring_loss"] = True
    return out


def recommend_growth_primary_method(
    style: Optional[str],
    net_profit_growth: Optional[float],
    forced_valuation: dict,
    valuation_regime: Optional[str] = None,
    recurring_loss: Optional[bool] = None,
    deducted_yoy: Optional[float] = None,
) -> dict:
    """成长股目标价口径路由：high_beta_growth 应以**前瞻 PEG** 为主方法，而非 TTM PE×EPS。

    动机（实测两样本对标）：12 个月目标价对成长股本该前瞻（卖方做法 = 次年 EPS × 目标倍数）。
    若主方法用 PE×EPS(TTM)（占 50% 权重），会把快速成长龙头的目标价系统性压低 15-25%
    → 偏离度虚高、base-case 隐含负收益过大（中际旭创 300308 primary 误设 PE×EPS TTM →
    目标 917 vs 卖方中位 ~1050，base -21%；而天孚 300394 正确用 PEG 前瞻）。

    **口径安全**：本函数只改"主方法选谁"（→ 改各腿权重），**不**改任何腿的 EPS 口径——
    PEG 腿仍用前瞻 EPS、PE×EPS/同业腿仍用 TTM EPS，不违反"反双重计入"铁律。
    漂移护栏：PEG 腿倍数仍受同业锚 + PEG 有界溢价封顶（见 compute_peer_anchored_pe_cap）。

    路由规则：
    - forced_valuation 生效（亏损/银行/REIT 等）→ 不介入，沿用强制方法。
    - high_beta_growth + 有正增速（≥15%）+ 非 discipline → 推荐 primary=peg（前瞻主导），
      secondary=[pe_eps(TTM 作下限参考), 同业可比/卖方目标价]。
    - 其余 style → 不强制（返回 recommend=None，沿用行业卡/现状）。

    Returns:
        dict: {"recommend": Optional[str], "weight_hint": str, "reason": str}
    """
    if forced_valuation.get("force_valuation"):
        return {"recommend": None, "weight_hint": "",
                "reason": "forced_valuation 生效，前瞻路由不介入"}

    # 成长质量闸（对标投研：不拿基数效应/非经常性增速 PEG 一个主业亏损的公司，防淳中式价值陷阱）
    # ① 扣非亏损 → 主业实亏，归母高增速多为非经常性 → 不走前瞻 PEG
    if recurring_loss is True:
        return {"recommend": None, "weight_hint": "",
                "reason": "成长质量闸：扣非净利亏损（主业实亏）→ 归母高增速多属非经常性/基数效应，不走前瞻 PEG，回退保守口径"}
    # ② 头条高增速但扣非增速很弱（基数效应）→ 用扣非口径判断，不被归母假高增速误导
    if (deducted_yoy is not None and net_profit_growth is not None
            and net_profit_growth >= 0.50 and deducted_yoy < 0.15):
        return {"recommend": None, "weight_hint": "",
                "reason": (f"成长质量闸：归母增速 {net_profit_growth*100:.0f}% 但扣非增速仅 "
                           f"{deducted_yoy*100:.0f}%（基数效应/非经常性主导）→ 不走前瞻 PEG")}

    g = net_profit_growth
    is_growth_style = style in ("high_beta_growth",)
    has_growth = g is not None and g >= 0.15
    not_discipline = (valuation_regime or "") != "discipline"

    if is_growth_style and has_growth and not_discipline:
        return {
            "recommend": "peg",
            "weight_hint": "PEG(前瞻) 50% / 卖方目标价或同业可比 30% / PE×EPS(TTM) ≤20%（仅作下限参考）",
            "reason": (
                f"high_beta_growth + 归母净利增速 {g*100:.0f}% → 12 个月目标按前瞻 EPS×PEG 主导"
                f"（对标卖方做法）；TTM PE×EPS 腿降权当下限参考，避免系统性压低成长龙头目标价"
            ),
        }
    # 高成长但减速/无增速数据 → 不强制前瞻，留给现状（防 PEG 在负增速下失真）
    why = "非 high_beta_growth" if not is_growth_style else (
        "增速数据缺失或<15%" if not has_growth else "regime=discipline（基本面恶化，不前瞻主导）")
    return {"recommend": None, "weight_hint": "", "reason": f"前瞻路由不介入：{why}"}


# ---------------------------------------------------------------------------
# Layer 2: 数据参照 —— 三源 PE / 默认 premium / theme_inferred / leadership_bonus
# ---------------------------------------------------------------------------
def parse_sell_side_pe_consensus(news_str: str) -> Optional[tuple[float, float]]:
    """从 news 报告抽卖方一致预期 PE 区间。

    匹配（按优先级）：
    1. "卖方一致预期 PE 60-85 倍" / "目标 PE 60-85 倍"
    2. "卖方目标价隐含 PE 60-85x"
    3. "卖方平均 PE 75 倍"（单值 → ± 15% 区间）

    Returns (low, high) or None.
    """
    import re

    if not news_str:
        return None

    # 模式 1: 区间
    range_patterns = [
        r"卖方[^0-9\n]{1,20}PE[^0-9\n]{1,8}([0-9.]+)\s*[-~到至]\s*([0-9.]+)\s*[倍xX]?",
        r"卖方[^0-9\n]{1,20}目标价[^0-9\n]{1,15}隐含\s*PE[^0-9\n]{0,8}([0-9.]+)\s*[-~到至]\s*([0-9.]+)",
        r"(?:目标|合理|forward)\s*PE[^0-9\n]{0,8}([0-9.]+)\s*[-~到至]\s*([0-9.]+)\s*[倍xX]",
        r"一致[预期]{0,2}\s*PE[^0-9\n]{0,8}([0-9.]+)\s*[-~到至]\s*([0-9.]+)",
    ]
    for pat in range_patterns:
        m = re.search(pat, news_str)
        if m:
            try:
                low, high = float(m.group(1)), float(m.group(2))
                if 0 < low <= high <= 1000:
                    return (low, high)
            except ValueError:
                continue

    # 模式 2: 单值 → 扩为 ± 15% 区间
    single_patterns = [
        r"卖方[^0-9\n]{1,15}平均\s*PE\s*[:：]?\s*([0-9.]+)",
        r"卖方[^0-9\n]{1,15}中位\s*PE\s*[:：]?\s*([0-9.]+)",
    ]
    for pat in single_patterns:
        m = re.search(pat, news_str)
        if m:
            try:
                v = float(m.group(1))
                if 0 < v <= 500:
                    return (v * 0.85, v * 1.15)
            except ValueError:
                continue

    return None


def compute_self_pe_p80(price_df, eps_ttm: Optional[float]) -> Optional[float]:
    """计算自身近 1 年 PE 的 80% 分位（用于"超共识溢价"对照）。

    简化：用过去 252 个交易日的收盘价 / 当前 EPS_TTM。
    严格做法应该用滚动 TTM EPS，但 fundamentals 历史 EPS 难获取，简化版可接受。
    亏损股或 EPS 缺失时返回 None。
    """
    import pandas as pd

    if price_df is None or len(price_df) == 0 or "Close" not in price_df.columns:
        return None
    if eps_ttm is None or eps_ttm <= 0:
        return None

    closes = pd.to_numeric(price_df["Close"], errors="coerce").dropna()
    if len(closes) < 60:
        return None

    recent = closes.tail(252)
    pe_series = recent / eps_ttm
    return float(pe_series.quantile(0.80))


def parse_peer_pe_median(news_str: str, fund_str: str) -> Optional[float]:
    """从 news/fundamentals 报告抽同业/行业 PE 中位数。

    优先级：
    1. "巨潮行业 PE 中位数 88.6"（fundamentals 行业 PE 表）
    2. "同业 PE 中位数 75 倍"
    3. "可比公司 PE 平均 80 倍"
    """
    import re

    sources = (news_str or "") + "\n" + (fund_str or "")
    if not sources.strip():
        return None

    patterns = [
        r"巨潮[^0-9\n]{0,30}PE[^0-9\n]{0,15}中位[数值]?\s*[:：=]?\s*([0-9.]+)",
        r"行业\s*PE\s*中位[数值]?\s*[:：=]?\s*([0-9.]+)",
        r"同业[^0-9\n]{0,15}PE\s*中位[数值]?\s*[:：=]?\s*([0-9.]+)",
        r"可比[^0-9\n]{0,15}PE\s*(?:平均|中位)\s*[:：=]?\s*([0-9.]+)",
        r"行业\s*平均\s*PE\s*[:：=]?\s*([0-9.]+)",
    ]
    for pat in patterns:
        m = re.search(pat, sources)
        if m:
            try:
                v = float(m.group(1))
                if 0 < v <= 500:
                    return v
            except ValueError:
                continue
    return None


def detect_leadership_bonus(fund_str: str, news_str: str) -> tuple[int, str]:
    """检测是否享受"龙头/稀缺标的"额外溢价（用于 premium 计算）。

    判定规则（按优先级取最高）：
    - 全球唯三/唯一/国内唯一 → +30%
    - 行业市占率 > 30% → +30%
    - 卖方一致评级 BUY 数量 ≥ 10 家 → +20%
    - 否则 0

    Returns (bonus_pct, reason)
    """
    import re

    sources = (fund_str or "") + "\n" + (news_str or "")
    if not sources.strip():
        return 0, ""

    # 全球唯几 / 国内唯一
    m = re.search(r"全球[^\n]{0,8}(唯一|唯[二三四]|仅[一二三四])\s*(?:供应商|厂商|企业)?", sources)
    if m:
        return 30, f"全球稀缺标的：{m.group(0)[:30]}"
    m = re.search(r"国内\s*唯一\s*(?:供应商|厂商|企业|龙头)?", sources)
    if m:
        return 30, "国内唯一标的"

    # 市占率
    m = re.search(r"(?:全球|国内|世界)?\s*市[占场][率分]?\s*[:：=]?\s*([0-9.]+)\s*%", sources)
    if m:
        try:
            v = float(m.group(1))
            if v >= 30:
                return 30, f"市占率 {v:.1f}% ≥ 30%（行业龙头）"
        except ValueError:
            pass

    # 卖方 BUY 数量（多模式）
    patterns_buy = [
        r"BUY\s*[评级]{0,2}\s*[:：]?\s*([0-9]+)\s*家",
        r"([0-9]+)\s*家\s*(?:券商|机构|卖方)\s*(?:给予|评级)\s*BUY",
        r"([0-9]+)\s*份\s*研报\s*(?:全部|均|一致)?\s*(?:BUY|买入|增持)",
    ]
    for pat in patterns_buy:
        m = re.search(pat, sources)
        if m:
            try:
                n = int(m.group(1))
                if n >= 10:
                    return 20, f"卖方一致评级 BUY {n} 家（共识强龙头）"
            except ValueError:
                continue

    return 0, ""


# Theme stage 默认 premium 模板（百分点）
PREMIUM_DEFAULT_TABLE: dict[str, int] = {
    "initiation": 30,
    "acceleration": 50,
    "peak": 20,
    "fading": -20,
    "none": 0,
}


def compute_default_premium(
    theme_stage_inferred: str,
    leadership_bonus_pct: int = 0,
    macro_adjustment_pct: int = 0,
) -> tuple[int, str]:
    """计算默认 premium_tolerance_pct（按 theme_stage 查表 + 龙头溢价 + 宏观修正）。

    Returns (premium_pct, formula_explanation)
    """
    # initiation_or_acceleration / none_or_acceleration 这些"二选一"标签取保守值
    stage_for_lookup = theme_stage_inferred
    if stage_for_lookup == "initiation_or_acceleration":
        stage_for_lookup = "initiation"  # 二选一取保守端
    elif stage_for_lookup == "none_or_acceleration":
        stage_for_lookup = "none"

    base = PREMIUM_DEFAULT_TABLE.get(stage_for_lookup, 0)
    total = base + leadership_bonus_pct + macro_adjustment_pct
    formula = (
        f"{base}（{stage_for_lookup}）"
        f" + {leadership_bonus_pct}（龙头）"
        f" + {macro_adjustment_pct}（宏观）"
        f" = {total}"
    )
    return total, formula


_REGIME_PREMIUM_FACTOR = {"ride": 1.0, "neutral": 0.5, "discipline": 0.0}


def gate_premium_by_regime(
    premium_pct: Optional[int], valuation_regime: Optional[str],
) -> tuple[Optional[int], str]:
    """按 regime 闸门"主题溢价容忍度"：ride 满 / neutral 半 / discipline 零。

    依据（对标投研，非结果倒推）：主题溢价 = 对"今天的高估值会被加速的盈利长进去"的容忍。
    这个容忍该由**基本面动能**(regime)来挣，而不是由"在不在热门赛道"来发——给一只基本面
    恶化的票发主题容忍，正是"买热门赛道里已掉头的票"那类错误的根源。
    - ride（基本面强，能长进去）        → 容忍全给（×1.0）
    - neutral（多空混杂）              → 容忍减半（×0.5）
    - discipline（恶化，长不进去，纯贵）→ 容忍归零（×0.0）

    只压"正溢价"(放宽容忍)；负溢价(fading / 宏观收紧)原样保留——收紧不放松，避免反向松绑。
    regime 未知时不介入（向后兼容）。

    Returns: (gated_premium_pct, 说明)
    """
    if premium_pct is None or valuation_regime not in _REGIME_PREMIUM_FACTOR:
        return premium_pct, "regime 未知或溢价缺失 → 不闸"
    factor = _REGIME_PREMIUM_FACTOR[valuation_regime]
    pos, neg = max(premium_pct, 0), min(premium_pct, 0)
    gated = int(round(pos * factor)) + neg
    return gated, (f"regime={valuation_regime} → 正溢价容忍 ×{factor}："
                   f"max({premium_pct},0)×{factor}{f'+{neg}' if neg else ''} = {gated}")


def infer_theme_stage_from_data(
    momentum_score: Optional[float],
    sector_rs_30d: Optional[float],
    rsi_percentile_1y: Optional[float],
    has_peak_signal: bool,
) -> tuple[str, str]:
    """基于量化 + 板块数据推断 theme_stage（不强制 LLM，仅作"参照值"喂给 LLM）。

    判定优先级：
    1. peak_signal 触发 → peak（已是硬规则）
    2. momentum ≥ 70 + sector_rs > 0 + RSI 分位 < 70 → acceleration（趋势确认）
    3. sector_rs < -15 + momentum < 40 → fading（趋势退潮）
    4. sector_rs > 5 → initiation_or_acceleration（早期识别，LLM 二选一）
    5. 其他 → none_or_acceleration（LLM 二选一）

    Returns (inferred_stage, reason_text)
    """
    if has_peak_signal:
        return "peak", "Peak 信号已强制触发"

    has_strong_momentum = momentum_score is not None and momentum_score >= 70
    has_positive_sector = sector_rs_30d is not None and sector_rs_30d > 0
    has_extreme_rsi = rsi_percentile_1y is not None and rsi_percentile_1y >= 70
    has_weak_sector = sector_rs_30d is not None and sector_rs_30d < -15
    has_weak_momentum = momentum_score is not None and momentum_score < 40

    if has_strong_momentum and has_positive_sector and not has_extreme_rsi:
        return "acceleration", (
            f"momentum={momentum_score:.1f} ≥ 70 + sector_rs_30d={sector_rs_30d:+.1f}% > 0"
            f" + RSI 分位未极端 → 趋势确认"
        )

    if has_weak_sector and has_weak_momentum:
        return "fading", (
            f"sector_rs_30d={sector_rs_30d:+.1f}% < -15"
            f" + momentum={momentum_score:.1f} < 40 → 趋势退潮"
        )

    if sector_rs_30d is not None and sector_rs_30d > 5:
        return "initiation_or_acceleration", (
            f"sector_rs_30d={sector_rs_30d:+.1f}% > 5 → 早期主题候选，LLM 在 initiation/acceleration 二选一"
        )

    return "none_or_acceleration", "无明确趋势信号 → LLM 在 none/acceleration 二选一"


def parse_pe_ttm_from_fundamentals(fund_str: str) -> Optional[float]:
    """从 fundamentals 报告抽 PE(TTM) 数值。

    作为"全部 Layer 2 三源都缺失"时的最后兜底锚：
    fallback_peer_pe_median = PE(TTM) × 0.7（向卖方一致 PE 方向收敛）

    匹配优先级：
    1. "PE(TTM) | XX 倍 | 系统计算值"（fundamentals 表格格式）
    2. "PE(TTM) X.XX" / "当前 PE(TTM) X.XX 倍"
    3. "动态 PE / 静态 PE" 不在此函数处理（口径不同）
    """
    import re

    if not fund_str:
        return None

    patterns = [
        r"PE\s*[(（]\s*TTM\s*[)）]\s*\|\s*\*{0,2}\s*([0-9.]+)\s*倍",
        r"PE\s*[(（]\s*TTM\s*[)）]\s*[:：]?\s*([0-9.]+)\s*倍",
        r"当前\s*PE\s*[(（]\s*TTM\s*[)）]\s*[:：]?\s*([0-9.]+)",
    ]
    for pat in patterns:
        m = re.search(pat, fund_str)
        if m:
            try:
                v = float(m.group(1))
                if 0 < v <= 5000:  # 合理 PE 范围
                    return v
            except ValueError:
                continue
    return None


def parse_net_profit_growth(fund_str: str, strict: bool = False) -> Optional[float]:
    """从 fundamentals 报告抽"归母净利润增速（年度）"，返回小数（如 +51.20% → 0.512）。

    strict=True：只认确定性 SYS_GROWTH_YOY（tushare），抽不到直接返回 None，**不回退散文**。
    用于 valuation_regime 的 earnings 腿——散文增速跑跑之间会漂，会让 regime 在 discipline/neutral
    间乱翻（澜起式 SELL↔HOLD 摆动根源）。earnings 腿宁可"无信号取 0"也不靠散文猜。

    用途（不做循环 forward_pe = PE_TTM/(1+g) 公式——那是恒等式无观点）：
    1. PEG 校验：合理 PE ≈ 增速值（PEG=1），用来给同业锚一个有界溢价上限
    2. 提示下游 RM：目标 PE 应配"前瞻 EPS = EPS_TTM×(1+g)"，而非 TTM EPS

    口径优先级（年度优先于单季，避开 Q1 淡季噪音）：
    1. 归母口径（最可靠，不会误抓扣非/营收）——"润"字可选、"同比"可选、增速/增长率/增长
       覆盖：归母净利润增速 / 归母净利润增长率 / 归母净利同比增速 / 归母净利增速
    2. "净利润增速(年度) | +XX%"（无归母前缀的年度表格）
    3. 兜底："净利润同比 +XX%"（排除扣非）

    标签每跑必漂（增速/增长率、归母前缀、润字有无、同比插入），故用弹性正则。
    """
    import re

    if not fund_str:
        return None

    # 最高优先：确定性 SYS_GROWTH_YOY（tushare fina_indicator，固定格式，不受 LLM 散文漂移影响）
    m_sys = re.search(r"SYS_GROWTH_YOY[^\n]*?归母净利YoY[^\n]*?年度=([+-]?[0-9.]+)%", fund_str)
    if not m_sys:  # 年度=NA 时退用单季
        m_sys = re.search(r"SYS_GROWTH_YOY[^\n]*?归母净利YoY\s*单季=([+-]?[0-9.]+)%", fund_str)
    if m_sys:
        try:
            v = float(m_sys.group(1))
            if -90.0 <= v <= 500.0:
                return v / 100.0
        except ValueError:
            pass

    if strict:
        return None  # 严格模式：SYS 抽不到就认输，不回退散文（防 earnings 腿漂移）

    # 归母口径（首选）：归母 + 净利(润可选) + (同比可选) + 增速/增长率/增长；表格或紧邻散文
    gm_label = r"归母净利(?:润)?(?:同比)?\s*(?:增速|增长率|增长)"
    gm_patterns = [
        gm_label + r"\s*\|\s*\*{0,2}\s*([+-]?[0-9.]+)\s*%",        # 表格行
        gm_label + r"[^\n%]{0,6}?\*{0,2}\+?([+-]?[0-9.]+)\s*%",    # 散文/冒号紧邻
    ]
    # 年度表格（无归母前缀）
    annual_patterns = [
        r"净利(?:润)?\s*(?:增速|增长率)\s*[(（]\s*年度\s*[)）]\s*\|\s*\*{0,2}\s*([+-]?[0-9.]+)\s*%",
    ]
    # 兜底：净利润同比（排除"扣非净利"——负向断言前一字不是"非"）
    fallback_patterns = [
        r"(?<!非)净利润\s*(?:同比)?\s*(?:增速|增长率|增长)\s*[:：|]?\s*\*{0,2}\+?([+-]?[0-9.]+)\s*%",
    ]

    def _first_valid(patterns):
        for pat in patterns:
            for m in re.finditer(pat, fund_str):
                try:
                    pct = float(m.group(1))
                except ValueError:
                    continue
                # 合理性：年度净利增速 -90% ~ +500%（剔除明显误抓）
                if -90.0 <= pct <= 500.0:
                    return pct / 100.0
        return None

    for group in (gm_patterns, annual_patterns, fallback_patterns):
        val = _first_valid(group)
        if val is not None:
            return val
    return None


def parse_sys_net_growth_components(fund_str: str) -> dict:
    """从确定性 SYS_GROWTH_YOY 抽归母净利的「单季 / 年度」增速（小数）。抽不到为 None。

    用于 PEG 确定性增速 + 低基数护栏（单季尖峰 vs 年度基线）。格式：
    `归母净利YoY 单季=X% 年度=Y%`（X/Y 可为 NA）。
    """
    import re
    res = {"annual": None, "quarter": None}
    if not fund_str:
        return res
    m = re.search(
        r"SYS_GROWTH_YOY[^\n]*?归母净利YoY\s*单季=([+-]?[0-9.]+|NA)%\s*年度=([+-]?[0-9.]+|NA)%",
        fund_str)
    if m:
        for key, raw in (("quarter", m.group(1)), ("annual", m.group(2))):
            if raw != "NA":
                try:
                    res[key] = float(raw) / 100.0
                except ValueError:
                    pass
    return res


_PEG_GROWTH_CAP = 0.60        # 可持续性封顶：>60% 前瞻增速极少长期持续
_PEG_TRUST_BAND = 0.40        # 全采信区间：≤40% 增速视为可持续（约等于优质成长行业长期复合上限）
_PEG_HALFLIFE = 0.5           # 超出可持续区间的部分打五折（高增的均值回归）


def _peg_forward_growth(annual_growth: float) -> float:
    """trailing 年度增速 → forward 前瞻代理（分段衰减）。

    旧公式 min(g, 60%)×0.5 一刀切半衰，前瞻增速被压到最高 30%：
    - 50% 增速的票（天孚）前瞻只剩 25%，隐含 PE = PEG区间×25 ≈ 25-37 倍，
      公式目标价系统性偏低 → LLM 曾因"觉得太低"手算造反（82↔410 摆动的诱因）；
    - 100%+ 增速的一线光模块龙头（中际旭创/新易盛式）被压得更狠。
    对标卖方：12 个月目标价用 NTM 一致预期 EPS；我们无一致预期数据源，
    用分段衰减做代理——可持续区间全采信、超出部分均值回归打折、超高增封顶：
      g ≤ 40%          → 前瞻 = g（可持续高增全采信）
      40% < g          → 前瞻 = 40% + (g − 40%) × 0.5（超出部分五折）
      封顶 60%          →（g ≥ 80% 时触顶；超高增几乎必回落）
    """
    if annual_growth <= _PEG_TRUST_BAND:
        return annual_growth
    return min(_PEG_TRUST_BAND + (annual_growth - _PEG_TRUST_BAND) * _PEG_HALFLIFE,
               _PEG_GROWTH_CAP)


def compute_deterministic_peg_inputs(
    eps_ttm: Optional[float],
    annual_net_growth: Optional[float],
    q_net_growth: Optional[float] = None,
) -> Optional[dict]:
    """确定性 PEG 输入：钉死「前瞻增速 + 前瞻 EPS + 低基数置信度」，杜绝 LLM 现场选增速/EPS
    致 PEG 目标价跑跑之间摆动（协创式 320↔180 → OW↔UW 翻转的根）。

    口径（Python 确定性、不让 LLM 自选）：
    - **低基数护栏**：用「年度」归母增速做基，**单季尖峰（如 +343%）不进 PEG**。
    - 前瞻增速 = 分段衰减（见 _peg_forward_growth：≤40% 全采信 / 超出部分五折 / 封顶 60%）。
    - 前瞻 EPS = EPS_TTM × (1 + 前瞻增速)。
    - **置信度**：单季 >> 年度（>2× 且 >100%，低基数尖峰，前瞻高度不确定）→ "low"
      → 下游 RM 降 Conviction / 偏离度近阈值时偏 HOLD（数据本就说不清，不下强方向单）。

    缺确定性年度增速 / EPS_TTM / 年度增速≤0（PEG 不适用衰退）→ 返回 None（RM 走原 LLM 路径，至少不更差）。
    """
    if eps_ttm is None or annual_net_growth is None or annual_net_growth <= 0:
        return None
    fwd_growth = _peg_forward_growth(annual_net_growth)
    fwd_growth_pct = round(fwd_growth * 100)
    forward_eps = round(eps_ttm * (1 + fwd_growth), 2)
    low_base = (q_net_growth is not None and annual_net_growth > 0
                and q_net_growth > max(annual_net_growth * 2.0, 1.0))
    return {
        "peg_growth_pct": fwd_growth_pct,
        "forward_eps": forward_eps,
        "confidence": "low" if low_base else "normal",
        "annual_growth_pct": round(annual_net_growth * 100),
        "quarter_growth_pct": round(q_net_growth * 100) if q_net_growth is not None else None,
        "capped": fwd_growth < annual_net_growth,   # 是否被分段衰减打了折
        "low_base_spike": low_base,
    }


# 确定性 PEG 倍数带（low, high），治 RM 自拍 PEG 倍数致目标价摆动（澜起 274↔189 根）。
# 基准 PEG=1.0（Lynch 合理估值）；regime 决定能给多高成长溢价（主题溢价已包含在 regime 闸门）。
# ⚠️ TODO(harness 超参)：这组带是人工先验，待回测按"PEG×后续收益"校准（尤其 ride 上沿 1.5）。
_PEG_BAND_BY_REGIME = {
    "ride":       (1.0, 1.5),   # 强基本面+主题：可给成长溢价
    "neutral":    (0.9, 1.2),   # 中性：围绕合理估值
    "discipline": (0.8, 1.0),   # 弱基本面（减速/派发/流出）：折价，不追
}

# 范式成长 ride 档（Phase 2 ②）：AI/算力硬科技 secular 龙头在确认 ride 时，比通用 ride
# (1.0-1.5) 抬一档——对标卖方对 AI 龙头爬坡期常给 PEG 1.5-2.5，此处取**保守**上沿避免追顶。
# 闸门严（见 compute_peg_band：is_paradigm_ride 须 paradigm+ride+earnings腿==1，且 confidence
# 非 low 才享高沿）。⚠️ TODO(harness 超参)：上沿 1.8 待 T+30 回测校准，偏多则下调。
_PARADIGM_RIDE_PEG_BAND = (1.2, 1.8)


def compute_peg_band(valuation_regime: Optional[str],
                     peg_confidence: Optional[str] = "",
                     is_paradigm_ride: bool = False) -> tuple[float, float]:
    """regime → (PEG 下限, PEG 上限)。低置信前瞻时上沿压回（不为不确定的 EPS 付溢价）。

    is_paradigm_ride=True（范式股 + 确认 ride + earnings腿=+1，由调用方判定）→ 用范式 ride 档
    (1.2-1.8) 表达 secular 多年跑道；但**低置信前瞻(低基数尖峰)不享高沿**，退回通用 ride 再压。
    """
    conf_low = (peg_confidence or "").strip().lower() == "low"
    if is_paradigm_ride and not conf_low:
        return _PARADIGM_RIDE_PEG_BAND   # 范式 ride 档；多年 secular 跑道只由 PEG 这一通道表达
    low, high = _PEG_BAND_BY_REGIME.get((valuation_regime or "").strip().lower(), (0.9, 1.2))
    if conf_low:
        high = min(high, 1.1)
    return (low, high)


def compute_peg_leg_target(
    forward_eps: Optional[float],
    growth_pct: Optional[float],
    peg_low: float,
    peg_high: float,
) -> Optional[dict]:
    """确定性 PEG 腿目标价（= compute_peg_target_price 同公式），Python 算死供 RM Step4 直读。

    根治成长股 PEG 腿被 LLM 现场乱填参数致目标价摆动：天孚同股同输入(前瞻EPS 3.8/增速45/
    PEG 0.9-1.2)三跑 PEG 腿 194↔269↔342-456 乱跳，把 SELL 抬成 HOLD——根因是 RM 调
    compute_peg_target_price 时无视 SYS_PEG_BAND/SYS_PEG_GROWTH_PCT，自塞高一倍的增速/PEG。
    钉死后 RM 直读 SYS_PEG_TARGET_PRICE，不再有塞错参数的入口。

    目标价 = forward_eps × (PEG × 增速%)，隐含 PE = PEG × 增速%（同 rm_tools.compute_peg_target_price）。

    Returns: {low, mid, high, implied_pe_range} 或 None（缺前瞻 EPS/增速）。
    """
    if (forward_eps is None or growth_pct is None
            or forward_eps <= 0 or growth_pct <= 0):
        return None
    low = round(forward_eps * peg_low * growth_pct, 2)
    high = round(forward_eps * peg_high * growth_pct, 2)
    mid = round((low + high) / 2, 2)
    return {
        "low": low, "mid": mid, "high": high,
        "implied_pe_range": [round(peg_low * growth_pct, 1), round(peg_high * growth_pct, 1)],
    }


# 强周期股正常化腿的 mid-cycle PE 带（对标投研：周期股峰值盈利不可线性外推，给跨周期合理倍数）。
# 存储/面板类 10-15x 是行业惯用 mid-cycle 区间。⚠️ TODO(harness 超参)：待回测校准。
_CYCLICAL_NORMALIZE_PE_BAND = (10.0, 15.0)
# 两腿离散度 ≥ 此倍数 = 双峰（周期崩 vs 结构成长分歧大）→ 综合目标低置信（中间数不可信）
_CYCLICAL_DISPERSION_BIMODAL = 2.5


def compute_cyclical_scenario_target(
    normalized_eps: Optional[float],
    forward_eps: Optional[float],
    forward_growth_pct: Optional[float],
    position: Optional[str],
    peg_low: float,
    peg_high: float,
    normalize_pe_band: tuple = _CYCLICAL_NORMALIZE_PE_BAND,
) -> Optional[dict]:
    """强周期股双轨情景目标价（确定性，替代 RM 现场硬平均两条腿致摆动）。

    两条腿是**互斥的未来**，不是同一估值的不同输入：
      - 周期均值回归(bear)：normalized_eps × mid-cycle PE —— 峰值盈利不可持续、回归正常化；
      - 结构成长(bull)：forward_eps × (PEG × 前瞻增速) —— AI 结构需求支撑前瞻溢价。
    按周期位置滑动权重做概率加权出 base；两腿离散 ≥2.5x 时标低置信（双峰，中间数谁都不信）。
    Python 算死、RM 直读 SYS_CYCLICAL_TARGET 不再现场算 → 根治残留摆动（兆易 SELL↔UW）。
    对标投研：估值分歧 5 倍时不做算术平均，而是情景概率加权 + 显式标注双峰不确定性。

    Returns: {bear_low/high, bull_low/high, base_low/high, weights, dispersion, confidence,
              normalize_pe_band} 或 None（缺正常化/前瞻 EPS）。
    """
    if (normalized_eps is None or forward_eps is None or forward_growth_pct is None
            or normalized_eps <= 0 or forward_eps <= 0 or forward_growth_pct <= 0):
        return None
    pe_lo, pe_hi = normalize_pe_band
    bear_low = round(normalized_eps * pe_lo, 2)
    bear_high = round(normalized_eps * pe_hi, 2)
    # 成长腿：target_PE = PEG × 增速%；price = forward_eps × target_PE（同 compute_peg_target_price）
    bull_low = round(forward_eps * peg_low * forward_growth_pct, 2)
    bull_high = round(forward_eps * peg_high * forward_growth_pct, 2)
    w_norm, w_growth = cyclical_target_weights(position)
    base_low = round(w_norm * bear_low + w_growth * bull_low, 2)
    base_high = round(w_norm * bear_high + w_growth * bull_high, 2)
    bear_mid = (bear_low + bear_high) / 2
    bull_mid = (bull_low + bull_high) / 2
    dispersion = round(bull_mid / bear_mid, 2) if bear_mid > 0 else None
    confidence = ("low" if dispersion is not None and dispersion >= _CYCLICAL_DISPERSION_BIMODAL
                  else "normal")
    return {
        "bear_low": bear_low, "bear_high": bear_high,     # 周期均值回归视角（正常化）
        "bull_low": bull_low, "bull_high": bull_high,     # 结构成长视角（前瞻 PEG）
        "base_low": base_low, "base_high": base_high,     # 概率加权（位置滑动权重）
        "weights": {"normalize": w_norm, "growth": w_growth},
        "dispersion": dispersion, "confidence": confidence,
        "normalize_pe_band": [pe_lo, pe_hi],
    }


def compute_peer_anchored_pe_cap(
    peer_pe_median: Optional[float],
    pe_ttm: Optional[float],
    net_profit_growth: Optional[float],
    leadership_bonus_pct: int = 0,
    target_peg: float = 1.0,
) -> Optional[dict]:
    """计算 target_pe_high 的硬天花板（两腿取友好者 + 绝对上限 PE_TTM）。

    投研团队做法：目标 PE 锚同业，成长更快者按 PEG 给溢价；但绝不超过当前 PE_TTM
    （不能用现价的贵倍数来证明目标价——那是无观点的循环）。

    两腿取 max（让超高增长股不被同业 comps 死压）：
    - 腿 A（comps）：peer_median × (1+有界溢价)，溢价按超额增速给、封顶 +40%
    - 腿 B（PEG 正当化）：min(增速,100%) × target_peg —— 增速 100% 的票可正当化到 PE≈100
      （增速封顶 100% 防一次性暴增正当化离谱 PE；target_peg=1.0 即 PEG=1 公允不偏贵）

    绝对上限：两腿结果都 ≤ PE_TTM（防漂移核心护栏；PEG 给空间但不许超现价倍数）。

    返回 None 表示 peer 锚不可用（交由调用方走全-null 兜底）。
    返回 dict: {cap, premium_pct_used, anchor_used, formula}
    """
    if not peer_pe_median or peer_pe_median <= 0:
        return None

    # --- 腿 A：同业 comps 锚 + 有界溢价（增速越高溢价越大，封顶 +40%）---
    premium_pct = 0
    if net_profit_growth is not None and net_profit_growth > 0:
        # 经验映射：增速每超同业基准（25% 作为行业一般成长基准）10pp，给 +10% 溢价
        excess = (net_profit_growth - 0.25) * 100.0  # 单位 pp
        if excess > 0:
            premium_pct = min(30, int(excess / 10.0) * 10)
    if leadership_bonus_pct > 0:
        premium_pct = min(40, premium_pct + 10)
    comps_anchor = peer_pe_median * (1 + premium_pct / 100.0)

    # --- 腿 B：PEG 正当化 PE（高增长股的该有溢价，不被 comps 死压）---
    peg_anchor = None
    if net_profit_growth is not None and net_profit_growth > 0:
        g_pct = min(net_profit_growth * 100.0, 100.0)  # 增速封顶 100%
        peg_anchor = g_pct * target_peg

    # 取两腿更友好者
    if peg_anchor is not None and peg_anchor > comps_anchor:
        cap = peg_anchor
        anchor_used = "peg"
        formula = (
            f"PEG 正当化 = min(增速,100%)×PEG{target_peg:g} = {peg_anchor:.1f}"
            f"（> comps 锚 {comps_anchor:.1f}）"
        )
    else:
        cap = comps_anchor
        anchor_used = "comps"
        formula = f"peer_median {peer_pe_median:.1f}×(1+{premium_pct}%) = {comps_anchor:.1f}"

    # --- 绝对上限：≤ PE_TTM（堵死"目标=现价贵倍数"漂移）---
    if pe_ttm is not None and pe_ttm > 0 and cap > pe_ttm:
        cap = pe_ttm
        anchor_used += "+pe_ttm_capped"
        formula += f"；绝对上限 ≤ PE_TTM {pe_ttm:.1f}"

    return {
        "cap": cap,
        "premium_pct_used": premium_pct,
        "anchor_used": anchor_used,
        "formula": formula,
    }


def parse_growth_deceleration(fund_str: str, strict: bool = False) -> Optional[str]:
    """从 fundamentals 抽营收增速方向（最近季 vs 年度）→ 减速/加速/平稳。

    格式：`| 营收同比增速 | +19.51% | +49.94% | ... |`（列序：最近季 / 年度 / 上年度）。
    澜起：Q1 19.5% << 年度 49.9% → 减速（盈利动能转弱，regime earnings 腿据此投负）。

    strict=True：只认确定性 SYS_GROWTH_YOY（路径0），抽不到直接返回 None，**不回退散文**
    （路径1/2 的散文抽取跑跑之间会漂，是 earnings 腿乱翻的根源）。

    Returns: "decelerating" / "accelerating" / "stable" / None（抽不到）
    """
    import re
    if not fund_str:
        return None

    # 路径0（最高优先）：确定性 SYS_GROWTH_YOY（tushare fina_indicator）
    msys = re.search(
        r"SYS_GROWTH_YOY[^\n]*?营收YoY\s*单季=([+-]?[0-9.]+)%\s*年度=([+-]?[0-9.]+)%", fund_str)
    if msys:
        try:
            q, annual = float(msys.group(1)), float(msys.group(2))
            if annual > 0:
                if q < annual * 0.6:
                    return "decelerating"
                if q >= annual * 0.95:
                    return "accelerating"
                return "stable"
        except ValueError:
            pass
    # SYS 行只有单季（年度=NA）→ 用单季绝对水平
    msq = re.search(r"SYS_GROWTH_YOY[^\n]*?营收YoY\s*单季=([+-]?[0-9.]+)%\s*年度=NA", fund_str)
    if msq:
        try:
            q = float(msq.group(1))
            return "decelerating" if q < 15.0 else ("accelerating" if q >= 45.0 else "stable")
        except ValueError:
            pass

    if strict:
        return None  # 严格模式：SYS 抽不到就认输，不回退散文（防 earnings 腿漂移）

    # 路径1：营收同比增速 两列（最近季 | 年度），比较得方向
    m = re.search(
        r"营(?:业收入|收)同比\s*(?:增速|增长率)?\s*\*{0,2}\s*\|\s*"
        r"\*{0,2}([+-]?[0-9.]+)%\*{0,2}\s*\|\s*\*{0,2}([+-]?[0-9.]+)%",
        fund_str,
    )
    if m:
        try:
            q, annual = float(m.group(1)), float(m.group(2))
            if annual > 0:
                if q < annual * 0.6:
                    return "decelerating"
                if q >= annual * 0.95:
                    return "accelerating"
                return "stable"
        except ValueError:
            pass

    # 路径2：只有最近季（单季）增速——格式如 "营收同比增速(Q1单季) | 4.58%" 或散文 "Q1营收仅同比+4.58%"
    # 没有年度基线时，用绝对水平判：单季营收增速 <15% = 弱/减速（高 PE 成长股的红旗）；≥45% = 强
    # 用 [^0-9%\n] 排除数字，避免贪婪回溯误抓单个数字（如把 +58% 抓成 8）
    mq = re.search(
        r"(?:Q[1-4]|单季|最近季)[^0-9%\n]{0,8}营(?:业收入|收)[^0-9%\n]{0,8}同比[^0-9%\n]{0,4}([+-]?[0-9.]+)%",
        fund_str,
    )
    if not mq:
        mq = re.search(
            r"营(?:业收入|收)同比增速\s*[(（]\s*Q[1-4]\s*单季\s*[)）]\s*\|\s*\*{0,2}([+-]?[0-9.]+)%",
            fund_str,
        )
    if mq:
        try:
            q1 = float(mq.group(1))
            if q1 < 15.0:
                return "decelerating"   # 单季营收个位数/十几→弱（澜起 4.58% 即此）
            if q1 >= 45.0:
                return "accelerating"
            return "stable"
        except ValueError:
            pass
    return None


# 减持/机构减仓正向证据词（排除否定语境）
_DISTRIBUTION_PATTERNS = (
    r"询价转让[^。\n]{0,20}折价",
    r"折价[^。\n]{0,8}转让",
    r"[0-9]+\s*余?家机构[^。\n]{0,10}(?:减持|减仓)",
    r"机构[^。\n]{0,6}(?:大幅|集中)?(?:减持|减仓)",
    r"套现[^。\n]{0,6}[0-9.]+\s*亿",
    r"大股东[^。\n]{0,8}减持",
    r"股东户数[^。\n]{0,10}(?:增加|上升|持续增)",
)
_DISTRIBUTION_NEGATION = (r"未(?:发现|出现)[^。\n]{0,20}减持", r"无[^。\n]{0,10}减持")

# 减持/派发证据的"陈旧"阈值：附近日期距分析日 >此天数 → 视为旧事，不投 distribution 腿。
# 对标投研：内部人减持信号约 1 季度内有意义，半年前在 1/3 价位的减持对当前判断无增量。
# 治范式龙头被陈旧减持敲出 ride（天孚 06-25：2026-01 高管@198-219 / 2025-03 大股东@98 误投）。
_DISTRIBUTION_STALE_DAYS = 120
# 匹配 YYYY-MM-DD / YYYY/MM/DD / YYYY.MM.DD / YYYY年MM月[DD日]（日可缺，缺则按当月1日）
_EVENT_DATE_RE = re.compile(r"(20\d{2})\s*[-/.年]\s*(\d{1,2})(?:\s*[-/.月]\s*(\d{1,2}))?")


def _nearest_event_date(text: str, pos: int, radius: int = 45):
    """匹配位置 pos 附近窗口内、距 pos 最近的日期 → datetime.date 或 None。"""
    import datetime
    lo, hi = max(0, pos - radius), min(len(text), pos + radius)
    window = text[lo:hi]
    best, best_dist = None, 1 << 30
    for m in _EVENT_DATE_RE.finditer(window):
        try:
            y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3) or 1)
            dt = datetime.date(y, mo, d)
        except (ValueError, TypeError):
            continue
        center = lo + (m.start() + m.end()) // 2
        dist = abs(center - pos)
        if dist < best_dist:
            best, best_dist = dt, dist
    return best


def parse_distribution_signals(news_str: str, fund_str: str = "", sentiment_str: str = "",
                               current_date: Optional[str] = None) -> dict:
    """检测"大股东/机构派发"硬证据（减持/折价询价转让/机构减仓/筹码分散）。

    这是澜起式派发股最硬的看空信号，但散落在 news/fundamentals/sentiment，
    之前没喂进 regime。返回 {detected, reasons, stale_skipped}（用于 regime 的 distribution 腿投负）。

    recency 门（current_date 给定时生效）：每条减持/派发证据附近若有日期且 >120 天，视为陈旧、
    不计入——治软口径把一年前低位减持当当前派发、把范式龙头敲出 ride。无日期的当期结构信号
    （如"股东户数增加"）照常保留。缺 current_date 则不过滤（向后兼容）。
    """
    import datetime
    text = "\n".join([news_str or "", fund_str or "", sentiment_str or ""])
    if not text.strip():
        return {"detected": False, "reasons": [], "stale_skipped": 0}
    cd = None
    if current_date:
        try:
            cd = datetime.date.fromisoformat(str(current_date)[:10])
        except ValueError:
            cd = None
    reasons, stale_skipped = [], 0
    for pat in _DISTRIBUTION_PATTERNS:
        for m in re.finditer(pat, text):
            seg = m.group(0)
            # 否定语境跳过（如"未发现高管密集减持"）
            ctx = text[max(0, m.start() - 8):m.start()]
            if any(re.search(neg, ctx + seg) for neg in _DISTRIBUTION_NEGATION):
                continue
            # recency 门：附近日期 >120 天 → 陈旧，跳过（无日期则保留，当期结构信号不丢）
            if cd is not None:
                ev = _nearest_event_date(text, m.start())
                if ev is not None and (cd - ev).days > _DISTRIBUTION_STALE_DAYS:
                    stale_skipped += 1
                    continue
            reasons.append(seg.strip())
    # 去重
    reasons = list(dict.fromkeys(reasons))[:5]
    return {"detected": len(reasons) > 0, "reasons": reasons, "stale_skipped": stale_skipped}


# blowoff 护栏价格极端门：获利盘≥此值=狂热接盘(顶部温床)，与 RSI 1年分位≥85 并列作"价格极端"。
# 同 capital_flow_utils._WINNER_RATE_EUPHORIA_PCT(派发腿2)；范式股 blowoff 须"派发+价格极端"共振。
_BLOWOFF_WINNER_EUPHORIA_PCT = 85.0


def compute_valuation_regime(
    *,
    momentum_score: Optional[float] = None,
    rsi_percentile_1y: Optional[float] = None,
    has_peak_signal: bool = False,
    capital_flow_regime: Optional[str] = None,
    main_force_streak_days: Optional[int] = None,
    lhb_inst_direction: Optional[int] = None,
    net_profit_growth: Optional[float] = None,
    growth_direction: Optional[str] = None,
    retail_concentration_signal: Optional[str] = None,
    theme_stage_inferred: Optional[str] = None,
    quant_anticrowding: Optional[float] = None,
    distribution_detected: bool = False,
    capital_flow_score: Optional[float] = None,
    recurring_loss: Optional[bool] = None,
    cyclical_class: Optional[str] = None,
    roe_pct_rank_10y: Optional[float] = None,
    is_paradigm: bool = False,
    earnings_revision: Optional[str] = None,
    winner_rate_pct: Optional[float] = None,
) -> dict:
    """客观估值 regime（骑趋势 / 中性 / 收纪律）——六路分析师信号合成，纯 Python 确定性。

    原则：每条腿**客观反映该信号本意**（对标投研），不带"该升该降"先验。
    决定"贵要不要紧 / 骑还是收"的不是估值本身，而是 盈利动能 + 技术趋势 + 资金面 +
    舆情拥挤 + 主题阶段 + 派发证据 的合成。

    六路，每路投 +1(ride)/0/-1(discipline)：
      1 技术/动量：强趋势且未极端超买 → +1；破位/弱 或 极端超买顶 → -1
      2 资金面  ：用连续 capital_flow_score(≥60→+1/≤40→-1) + 强势/机构净买/主力连续净流入(+1)、
                  恶化/机构净卖/主力连续净流出(-1)；流出优先
      3 盈利动能：高增速(≥40%)且不减速 → +1（含高位稳定增长，投研认可可持续高增）；
                  减速 或 停滞(<10%) → -1
      4 舆情拥挤：不拥挤(anticrowding≥60) → +1；拥挤+散户高接盘 → -1
      5 主题阶段：acceleration → +1；peak/fading 或 peak信号 → -1
      6 派发证据：大股东/机构减持等 → -1；**但当下资金面强流入时视为已被吸收/陈旧，不投**
                  （净当前持仓口径：smart money 在吸筹时，旧减持非红旗）

    合成：净分 ≥ +2 → ride；≤ -2 → discipline；其余 neutral（对称，无方向先验）。
    peak 信号强制触发时，ride 降级为 neutral（不骑进顶部）。

    Returns: {valuation_regime, score, legs:{...}, reasoning}
    """
    legs: dict[str, int] = {}

    # 1 技术/动量
    if momentum_score is not None:
        overbought = rsi_percentile_1y is not None and rsi_percentile_1y >= 85
        if momentum_score >= 65 and not overbought and not has_peak_signal:
            legs["tech"] = 1
        elif momentum_score <= 35 or has_peak_signal:
            legs["tech"] = -1
        else:
            legs["tech"] = 0

    # 2 资金面：用连续 score（信息量大于"强势/恶化"标签）+ 方向/streak/机构 共同判
    strong_inflow = (
        capital_flow_regime == "强势" or lhb_inst_direction == 1
        or (main_force_streak_days is not None and main_force_streak_days >= 3)
        or (capital_flow_score is not None and capital_flow_score >= 60)
    )
    strong_outflow = (
        capital_flow_regime == "恶化" or lhb_inst_direction == -1
        or (main_force_streak_days is not None and main_force_streak_days <= -3)
        or (capital_flow_score is not None and capital_flow_score <= 40)
    )
    cap_vote = 0
    if strong_inflow:
        cap_vote = 1
    if strong_outflow:
        cap_vote = -1  # 流出优先
    if (capital_flow_regime is not None or lhb_inst_direction is not None
            or main_force_streak_days is not None or capital_flow_score is not None):
        legs["capital"] = cap_vote

    # 3 盈利动能（方向优先；高增速且不减速 → +1，含高位稳定增长）
    # 成长质量闸：扣非亏损（主业实亏）时，归母高增速多属非经常性/基数效应 → 不投 +1（防淳中式假高增）
    if recurring_loss is True:
        legs["earnings"] = -1   # 主业亏损直接判盈利动能为负
    elif net_profit_growth is not None or growth_direction is not None:
        if growth_direction == "decelerating" or (
                net_profit_growth is not None and net_profit_growth < 0.10):
            legs["earnings"] = -1
        elif growth_direction == "accelerating" or (
                net_profit_growth is not None and net_profit_growth >= 0.40
                and growth_direction != "decelerating"):
            legs["earnings"] = 1   # 加速 OR 高增速(≥40%)且未减速（高位稳定也算偏多）
        else:
            legs["earnings"] = 0

    # 3b 强周期股的盈利动能语义反转（林奇铁律，只对 strong；半周期保留成长β语义）：
    #   - 周期顶部（ROE 10年分位 ≥0.8）：高增速是"周期顶部现象"而非动能证据，
    #     +1 压到 0——否则系统在最该下车的位置判 ride、把 SELL 托底成 HOLD
    #   - 周期谷底（≤0.2）：负增长/低增速是周期常态而非基本面恶化，
    #     -1 抬到 0——否则在最该布局的位置判 discipline、封死 BUY
    cyc_note = ""
    if cyclical_class == "strong" and roe_pct_rank_10y is not None and "earnings" in legs:
        if roe_pct_rank_10y >= 0.8 and legs["earnings"] > 0:
            legs["earnings"] = 0
            cyc_note = f"；强周期顶部(ROE分位{roe_pct_rank_10y:.2f})高增速不算动能，earnings腿压0"
        elif roe_pct_rank_10y <= 0.2 and legs["earnings"] < 0:
            legs["earnings"] = 0
            cyc_note = f"；强周期谷底(ROE分位{roe_pct_rank_10y:.2f})盈利差是周期常态，earnings腿抬0"

    # 3c 前瞻盈利上修中和后视镜减速（对标投研：revision 方向才是"骑还是收"的真判据）——
    #   主升浪里龙头单季高基数回落被 SYS_GROWTH 判 decelerating(-1)，但卖方此时在上修前瞻预期，
    #   把 -1 中和到 0（前瞻方向优先于后视镜）；下修则把高增速 +1 削到 0（预期恶化预警）。
    #   ⚠️ 范围克制：只动 earnings 腿，不碰 blowoff 护栏(硬派发仍生效)、不碰 PEG 带——避免把
    #   澜起式顶部(卖方维持但目标价低于现价、非上修)重新放松。源自新闻粗代理(report_rc 没权限前)。
    rev_note = ""
    if earnings_revision and "earnings" in legs:
        if earnings_revision == "上修" and legs["earnings"] == -1:
            legs["earnings"] = 0
            rev_note = "；卖方上修前瞻→earnings后视镜减速-1中和到0(revision方向优先)"
        elif earnings_revision == "下修" and legs["earnings"] == 1:
            legs["earnings"] = 0
            rev_note = "；卖方下修前瞻→earnings高增速+1削到0(预期恶化预警)"

    # 4 舆情拥挤
    crowd_vote = 0
    if retail_concentration_signal == "散户高接盘" or (
        quant_anticrowding is not None and quant_anticrowding <= 30):
        crowd_vote = -1
    elif quant_anticrowding is not None and quant_anticrowding >= 60:
        crowd_vote = 1
    if retail_concentration_signal is not None or quant_anticrowding is not None:
        legs["crowding"] = crowd_vote

    # 5 主题阶段
    if theme_stage_inferred is not None or has_peak_signal:
        ts = theme_stage_inferred or ""
        if has_peak_signal or "peak" in ts or "fading" in ts:
            legs["theme"] = -1
        elif ts == "acceleration":  # 仅精确确认的加速；模糊二选一不算
            legs["theme"] = 1
        else:
            legs["theme"] = 0

    # 6 派发证据（减持/机构减仓等）→ -1；但当下资金面强流入时视为已被吸收/陈旧，不投
    #   (净当前持仓口径：天孚主力净流入33亿时，舆情里的旧减仓不是红旗)
    if distribution_detected and not strong_inflow:
        legs["distribution"] = -1

    # 范式成长反转（镜像周期反转 3b）：确认范式股(secular hardtech) + 真加速(earnings=+1) +
    # 无 blowoff 证据 → ride 门槛 +2 降到 +1，且"反拥挤致的 crowding -1"在主升浪属常态(非顶部
    # 证据)抬 0。对标投研：secular re-rating 期间，拥挤/高位是加速特征不是该收的理由。
    # 护栏(防骑顶)只认**硬证据**：peak信号 / 破位(tech=-1) / 散户高接盘(硬派发合成确认)。
    # ⚠️ 不用 distribution_detected(软)做护栏——它是 parse_distribution_signals 读新闻散文的
    # 旧减持口径，18倍股必有陈旧减持新闻(中际旭创实测：5个月前大股东减持 0.5% 触发软派发，
    # 但硬数据股东户数 -15.78%=吸筹、大宗无折价=无派发，软信号误杀 ride)。硬派发由 peak/破位/
    # 散户高接盘三路把关，陈旧减持新闻不该否决范式骑乘(内部人减持硬口径见待办 A)。
    paradigm_note = ""
    ride_threshold = 2
    if is_paradigm and legs.get("earnings") == 1:
        # blowoff 护栏（Option A）：peak信号 / 趋势破位 是**价格行为**硬证据，单独成立即否决 ride；
        # 但"散户高接盘"是**筹码派发(流向)**信号、不等于价格 blowoff——必须叠加**价格极端**
        # (RSI 1年分位≥85 或 获利盘≥85% 狂热) 才算真抛物线顶。否则把"已回调/趋势健康但散户参与
        # 升高"的龙头误判见顶(天孚实测：6/12 已跌32%、获利盘74%、RSI中段，无价格 blowoff，却被
        # 筹码派发信号否决 ride)。派发的看空已由 capital/crowding 两腿承担，不让同一信号第三次否决。
        price_extreme = (
            (rsi_percentile_1y is not None and rsi_percentile_1y >= 85)
            or (winner_rate_pct is not None and winner_rate_pct >= _BLOWOFF_WINNER_EUPHORIA_PCT)
        )
        blowoff = (has_peak_signal
                   or legs.get("tech") == -1
                   or (retail_concentration_signal == "散户高接盘" and price_extreme))
        if not blowoff:
            ride_threshold = 1
            if legs.get("crowding") == -1:   # 此时非散户高接盘(否则 blowoff)，纯反拥挤分 → 主升浪常态抬0
                legs["crowding"] = 0
                paradigm_note = "；范式成长加速期：拥挤属主升浪常态(crowding抬0)+ride门槛降至+1"
            else:
                paradigm_note = "；范式成长加速期：ride门槛降至+1（无blowoff证据）"
        else:
            paradigm_note = "；范式股但 blowoff硬证据(peak/破位/散户高接盘+价格极端)→反转失效，回纪律"

    score = sum(legs.values())
    # 有效路 < 3 → 数据不足，给 neutral（不轻易骑/收）
    if len(legs) < 3:
        regime = "neutral"
    elif score >= ride_threshold:   # 对称阈值；范式加速期门槛降至 +1（见上）
        regime = "ride"
    elif score <= -2:
        regime = "discipline"
    else:
        regime = "neutral"

    # peak 信号不允许 ride
    if has_peak_signal and regime == "ride":
        regime = "neutral"

    reasoning = (f"六路净分={score}（{legs}）→ {regime}"
                 + ("；peak信号压制不骑" if has_peak_signal else "")
                 + cyc_note + rev_note + paradigm_note)
    return {"valuation_regime": regime, "score": score, "legs": legs, "reasoning": reasoning}


def parse_sector_rs_30d(sector_str: str) -> Optional[float]:
    """从 sector_comparison 报告抽本股 vs 主题 ETF (或行业 ETF) 的 30d RS。"""
    import re

    if not sector_str:
        return None

    # 表格匹配：| 30d | ... |
    m = re.search(r"\|\s*30d\s*\|\s*([+-]?[0-9.]+)\s*%", sector_str)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass

    # 文字匹配
    patterns = [
        r"30d\s*RS\s*[:：=]\s*([+-]?[0-9.]+)\s*%",
        r"vs[^0-9\n]{1,30}(?:主题|行业)\s*ETF[^0-9\n]{0,15}([+-]?[0-9.]+)\s*%",
        r"30d[^0-9\n]{1,15}([+-]?[0-9.]+)\s*%\s*(?:跑赢|跑输)?",
    ]
    for pat in patterns:
        m = re.search(pat, sector_str)
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                continue
    return None
