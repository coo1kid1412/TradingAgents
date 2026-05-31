"""资金流（DDE/DDX/DDY/DDZ-like）计算工具 —— 纯 Python，不调 LLM。

设计原则
- 所有计算确定性，相同输入必得相同输出（可复现、可审计、无幻觉）
- 字段全名规范：禁止简写。`main_force_net_inflow_5d_yi`，不是 `main_5d`
- 单位后缀必显式：`_pct` / `_yi`（亿元）/ `_days` / `_count` / `_ratio`
- 缺失值统一返回 None（不返回空字符串、不返回 0、不返回 -999）

下游消费方：tradingagents/agents/utils/capital_flow_node.py
"""
from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 单位常量
# ---------------------------------------------------------------------------
WAN_TO_YI = 1e-4  # 万元 → 亿元
YI_TO_WAN = 1e4   # 亿元 → 万元


# ---------------------------------------------------------------------------
# 投票/打分阈值（提议默认，可用真实数据校准 —— TODO: calibrate）
# ---------------------------------------------------------------------------
# 散户高承接：小单+中单买入成交占比 ≥ 此值（A 股该占比常 50-70%，>65 偏高）
_RETAIL_HIGH_RATE_PCT = 65.0     # 毛买占比口径：散户买盘占比 ≥ 65% = 高承接
_RETAIL_NET_HIGH_PCT = 8.0       # 净流入占比口径：散户(中+小单)净流入占比 ≥ +8% = 高承接
# 主力派发：连续净流出 ≥ 3 日（收紧自旧逻辑的 streak<0，消除 streak=-1 虚假票）
_STREAK_DISTRIBUTION = -3
# 龙虎榜机构净买方向阈值：30 日机构净买额绝对值 ≥ 此值（亿元）才算明确方向，否则 0（去零附近噪音）
_LHB_INST_NET_THRESHOLD_YI = 0.05


# ---------------------------------------------------------------------------
# 工具函数：归一化（线性区间映射到 0-100）
# ---------------------------------------------------------------------------
def _clip(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _linear_map(v: float, src_lo: float, src_hi: float, dst_lo: float = 0.0, dst_hi: float = 100.0) -> float:
    """线性映射：v 从 [src_lo, src_hi] 映射到 [dst_lo, dst_hi]，超出范围 clip。"""
    if src_hi == src_lo:
        return (dst_lo + dst_hi) / 2
    v = _clip(v, src_lo, src_hi)
    return dst_lo + (v - src_lo) / (src_hi - src_lo) * (dst_hi - dst_lo)


# ---------------------------------------------------------------------------
# 1. DDE-like 指标派生（基于 tushare moneyflow_dc / akshare 个股资金流）
# ---------------------------------------------------------------------------
def compute_dde_like_metrics(
    moneyflow_df: Optional[pd.DataFrame],
    circulating_market_value_yi: Optional[float],
) -> dict:
    """从主力/超大单/大单/中单/小单原始净额派生 DDX/DDY/DDZ-like 指标。

    Args:
        moneyflow_df: 资金流 DataFrame，按日期升序，必含列：
            - trade_date (str YYYYMMDD)
            - main_force_net_amount_yi      主力净流入（亿元）
            - extra_large_net_amount_yi     超大单净流入（亿元）
            - large_net_amount_yi           大单净流入（亿元）
            - medium_buy_amount_rate_pct    中单买入成交占比（%，可选）
            - small_buy_amount_rate_pct     小单买入成交占比（%，可选）
            - daily_amount_yi               当日成交额（亿元，可选；缺失时 ddz 用 None）
        circulating_market_value_yi: 流通市值（亿元）。无值时 ddx_like 系列返回 None。

    Returns:
        dict（全部字段全名命名）：
        - main_force_net_inflow_5d_yi      : 5 日主力累计净流入（亿元）
        - main_force_net_inflow_20d_yi     : 20 日主力累计净流入
        - ddx_like_5d_pct                  : 5 日超大单净额 / 流通市值 × 100（%）
        - ddx_like_5d_pct_1y               : ddx_like_5d_pct 的 1 年百分位（0-100）
        - large_order_net_inflow_5d_yi     : 5 日大单累计净额（亿元，展示用，不进打分/投票）
                                             （旧称 ddy_like；实为大单净额，非经典 DDY「涨跌动因/主动性」）
        - ddz_like_20d_pct                 : 20 日主力净额 / 20 日成交额 × 100（%，20日主力强度比，
                                             非经典 DDZ「主力库存」累计量）
        - net_inflow_streak_days           : 连续净流入(+) / 净流出(-) 天数（截断 ±10）

    注：散户接盘度不在此函数算——旧的 retail_takeover_ratio 用"零和近似"恒等 1.0（零信息），
    已废弃。散户信号改由 compute_retail_amount_rate 的真实小单+中单买占比驱动（见 assemble）。
    """
    out: dict = {
        "main_force_net_inflow_5d_yi": None,
        "main_force_net_inflow_20d_yi": None,
        "ddx_like_5d_pct": None,
        "ddx_like_5d_pct_1y": None,
        "large_order_net_inflow_5d_yi": None,
        "ddz_like_20d_pct": None,
        "net_inflow_streak_days": None,
    }

    if moneyflow_df is None or len(moneyflow_df) == 0:
        return out

    df = moneyflow_df.copy().reset_index(drop=True)
    n = len(df)

    # 主力累计净流入
    if "main_force_net_amount_yi" in df.columns:
        if n >= 1:
            out["main_force_net_inflow_5d_yi"] = round(
                float(df["main_force_net_amount_yi"].tail(5).sum()), 2
            )
        if n >= 20:
            out["main_force_net_inflow_20d_yi"] = round(
                float(df["main_force_net_amount_yi"].tail(20).sum()), 2
            )

    # DDX-like 5 日：超大单净额 / 流通市值
    if (
        "extra_large_net_amount_yi" in df.columns
        and circulating_market_value_yi is not None
        and circulating_market_value_yi > 0
        and n >= 1
    ):
        xl_5d = float(df["extra_large_net_amount_yi"].tail(5).sum())
        out["ddx_like_5d_pct"] = round(xl_5d / circulating_market_value_yi * 100, 4)

    # DDX-like 1 年百分位（要求至少 60 个交易日）
    if (
        "extra_large_net_amount_yi" in df.columns
        and circulating_market_value_yi is not None
        and circulating_market_value_yi > 0
        and n >= 60
    ):
        # 滚动 5 日累计 / mv
        rolling_5d_pct = (
            df["extra_large_net_amount_yi"].rolling(window=5).sum() / circulating_market_value_yi * 100
        ).dropna()
        if len(rolling_5d_pct) >= 60 and out["ddx_like_5d_pct"] is not None:
            current = out["ddx_like_5d_pct"]
            # 取最近 252 个交易日（不足则全部）
            history = rolling_5d_pct.tail(252)
            below = (history <= current).sum()
            out["ddx_like_5d_pct_1y"] = round(float(below) / len(history) * 100, 1)

    # 大单 5 日累计净额（旧称 DDY-like；实为大单净额，非经典 DDY「涨跌动因/主动性」，故诚实命名）
    if "large_net_amount_yi" in df.columns and n >= 1:
        out["large_order_net_inflow_5d_yi"] = round(float(df["large_net_amount_yi"].tail(5).sum()), 2)

    # DDZ-like 20 日：20 日主力净额 / 20 日成交额
    if (
        "main_force_net_amount_yi" in df.columns
        and "daily_amount_yi" in df.columns
        and n >= 20
    ):
        net_20 = float(df["main_force_net_amount_yi"].tail(20).sum())
        amt_20 = float(df["daily_amount_yi"].tail(20).sum())
        if amt_20 > 0:
            out["ddz_like_20d_pct"] = round(net_20 / amt_20 * 100, 4)

    # 连续净流入/流出天数（基于主力净额符号）
    if "main_force_net_amount_yi" in df.columns and n >= 1:
        out["net_inflow_streak_days"] = _compute_streak_days(df["main_force_net_amount_yi"])

    return out


def compute_retail_concentration_signal(
    retail_buy_amount_rate_5d_pct: Optional[float],
    net_inflow_streak_days: Optional[int],
    retail_net_inflow_rate_5d_pct: Optional[float] = None,
) -> Optional[str]:
    """散户接盘信号（替代恒等 1.0 的旧 retail_takeover_ratio）。

    真接盘 = 主力持续派发（streak ≤ -3）+ 散户大举承接。两种数据口径分别判：
    - 毛买占比口径（tushare）：散户买占比 ≥ 65%；
    - 净流入占比口径（akshare）：散户(中+小单)净流入占比 ≥ +8%（散户在净买入接盘）。
    与"主力流出 + 散户也跑（踩踏，净流入为负）"区分开。

    Returns: "散户高接盘" / "中性" / None（数据缺失）
    """
    if net_inflow_streak_days is None:
        return None
    distributing = net_inflow_streak_days <= _STREAK_DISTRIBUTION
    # 毛买占比口径优先
    if retail_buy_amount_rate_5d_pct is not None:
        if distributing and retail_buy_amount_rate_5d_pct >= _RETAIL_HIGH_RATE_PCT:
            return "散户高接盘"
        return "中性"
    # 净流入占比口径（akshare）
    if retail_net_inflow_rate_5d_pct is not None:
        if distributing and retail_net_inflow_rate_5d_pct >= _RETAIL_NET_HIGH_PCT:
            return "散户高接盘"
        return "中性"
    return None


def _compute_streak_days(series: pd.Series, max_streak: int = 10) -> int:
    """连续净流入(+) / 净流出(-) 天数，截断至 ±max_streak。

    从最后一天往前数：若最后一天 > 0，往前数连续 > 0 的天数（正数）
    若最后一天 < 0，往前数连续 < 0 的天数（负数）；若最后一天 = 0，返回 0
    """
    if len(series) == 0:
        return 0
    s = series.astype(float).reset_index(drop=True)
    last = s.iloc[-1]
    if last > 0:
        sign = 1
    elif last < 0:
        sign = -1
    else:
        return 0

    count = 0
    for i in range(len(s) - 1, -1, -1):
        v = s.iloc[i]
        if (sign > 0 and v > 0) or (sign < 0 and v < 0):
            count += 1
        else:
            break
    return int(sign * min(count, max_streak))


# ---------------------------------------------------------------------------
# 2. 北向资金指标派生
# ---------------------------------------------------------------------------
def compute_northbound_metrics(
    north_df: Optional[pd.DataFrame],
    data_freshness_days: int = 7,
    latest_trade_date: Optional[str] = None,
) -> dict:
    """从北向持股序列派生 5 日 / 20 日方向。

    Args:
        north_df: DataFrame，按日期升序，含列：
            - trade_date (YYYYMMDD)
            - hold_share_count            北向持股数量（股）
            - hold_market_value_yi        北向持股市值（亿元，可选）
        data_freshness_days: 若最新数据距 latest_trade_date 超过该阈值，则视为「数据停滞」
        latest_trade_date: 当前交易日 YYYYMMDD（用于判断数据新鲜度）

    Returns:
        dict（全部字段全名）：
        - northbound_5d_direction    : -1 / 0 / +1 / None（数据缺失或停滞返回 None）
        - northbound_20d_direction   : -1 / 0 / +1 / None
        - northbound_data_status     : "fresh" / "stale" / "missing"
        - northbound_latest_date     : 数据最新日期 YYYYMMDD
    """
    out: dict = {
        "northbound_5d_direction": None,
        "northbound_20d_direction": None,
        "northbound_data_status": "missing",
        "northbound_latest_date": None,
    }

    if north_df is None or len(north_df) == 0 or "hold_share_count" not in north_df.columns:
        return out

    df = north_df.copy().sort_values("trade_date").reset_index(drop=True)
    out["northbound_latest_date"] = str(df.iloc[-1]["trade_date"])

    # 数据新鲜度判定
    if latest_trade_date and out["northbound_latest_date"]:
        try:
            from datetime import datetime
            latest = datetime.strptime(latest_trade_date.replace("-", ""), "%Y%m%d")
            data_latest = datetime.strptime(out["northbound_latest_date"], "%Y%m%d")
            gap = (latest - data_latest).days
            out["northbound_data_status"] = "fresh" if gap <= data_freshness_days else "stale"
        except (ValueError, TypeError):
            out["northbound_data_status"] = "fresh"  # 无法解析时默认 fresh
    else:
        out["northbound_data_status"] = "fresh"

    # 数据停滞时不计算方向（防止用过期数据投票）
    if out["northbound_data_status"] == "stale":
        return out

    # 5 日方向：(最新持股 - 5 日前持股) 符号
    if len(df) >= 6:
        delta_5d = float(df.iloc[-1]["hold_share_count"]) - float(df.iloc[-6]["hold_share_count"])
        # 阈值：变化超过当前持股 0.5% 才算明确方向
        base = float(df.iloc[-6]["hold_share_count"])
        threshold = abs(base) * 0.005 if base != 0 else 1
        if delta_5d > threshold:
            out["northbound_5d_direction"] = 1
        elif delta_5d < -threshold:
            out["northbound_5d_direction"] = -1
        else:
            out["northbound_5d_direction"] = 0

    # 20 日方向
    if len(df) >= 21:
        delta_20d = float(df.iloc[-1]["hold_share_count"]) - float(df.iloc[-21]["hold_share_count"])
        base = float(df.iloc[-21]["hold_share_count"])
        threshold = abs(base) * 0.01 if base != 0 else 1
        if delta_20d > threshold:
            out["northbound_20d_direction"] = 1
        elif delta_20d < -threshold:
            out["northbound_20d_direction"] = -1
        else:
            out["northbound_20d_direction"] = 0

    return out


# ---------------------------------------------------------------------------
# 3. 散户户数（季报，stk_holdernumber）
# ---------------------------------------------------------------------------
def compute_holder_number_metrics(holder_df: Optional[pd.DataFrame]) -> dict:
    """从季报股东户数派生筹码集中度信号。

    Args:
        holder_df: DataFrame，按报告期升序，含列：
            - end_date (YYYYMMDD 报告期)
            - holder_num （股东户数，单位 户）

    Returns:
        dict（全部字段全名）：
        - holder_num_latest                 : 最新季报户数
        - holder_num_qoq_pct                : 环比上一季度变化（%）
        - holder_num_4q_trend               : "持续下降" / "持续上升" / "震荡" / None
        - chip_concentration_signal         : "筹码集中" / "筹码分散" / "中性"
        - holder_num_latest_report_date     : 最新报告期 YYYYMMDD
    """
    out: dict = {
        "holder_num_latest": None,
        "holder_num_qoq_pct": None,
        "holder_num_4q_trend": None,
        "chip_concentration_signal": None,
        "holder_num_latest_report_date": None,
    }

    if holder_df is None or len(holder_df) == 0 or "holder_num" not in holder_df.columns:
        return out

    df = holder_df.dropna(subset=["holder_num"]).sort_values("end_date").reset_index(drop=True)
    if len(df) == 0:
        return out

    latest = int(df.iloc[-1]["holder_num"])
    out["holder_num_latest"] = latest
    out["holder_num_latest_report_date"] = str(df.iloc[-1]["end_date"])

    # 环比
    if len(df) >= 2:
        prev = float(df.iloc[-2]["holder_num"])
        if prev > 0:
            out["holder_num_qoq_pct"] = round((latest - prev) / prev * 100, 2)

    # 4 季度趋势
    if len(df) >= 4:
        last_4 = df["holder_num"].tail(4).astype(float).reset_index(drop=True)
        diffs = last_4.diff().dropna()
        if all(d < 0 for d in diffs):
            out["holder_num_4q_trend"] = "持续下降"
        elif all(d > 0 for d in diffs):
            out["holder_num_4q_trend"] = "持续上升"
        else:
            out["holder_num_4q_trend"] = "震荡"

    # 筹码集中度信号（环比 < -3% → 集中；> +3% → 分散；其余中性）
    if out["holder_num_qoq_pct"] is not None:
        if out["holder_num_qoq_pct"] < -3:
            out["chip_concentration_signal"] = "筹码集中"
        elif out["holder_num_qoq_pct"] > 3:
            out["chip_concentration_signal"] = "筹码分散"
        else:
            out["chip_concentration_signal"] = "中性"

    return out


# ---------------------------------------------------------------------------
# 4. 龙虎榜计数（30 天）
# ---------------------------------------------------------------------------
def compute_lhb_metrics(
    lhb_count_30d: Optional[int],
    lhb_inst_net_buy_30d_yi: Optional[float] = None,
) -> dict:
    """龙虎榜：上榜次数（关注度）+ 机构席位净买方向（真方向信号）。

    上榜次数本身是"异动/关注度"，不代表方向（机构出货/游资派发同样上榜）。
    方向由机构专用席位 30 日净买额定：净买→+1 / 净卖→-1 / 持平→0 / 缺失→None。

    Args:
        lhb_count_30d: 30 天上榜次数；缺失返回 None（仅作关注度展示）
        lhb_inst_net_buy_30d_yi: 30 天龙虎榜机构席位净买额（亿元）；缺失 None

    Returns:
        dict:
        - lhb_count_30d            : int 或 None（关注度，不投方向）
        - lhb_inst_net_buy_30d_yi  : float 或 None
        - lhb_inst_direction       : -1 / 0 / +1 / None（投票依据）
    """
    out: dict = {
        "lhb_count_30d": int(lhb_count_30d) if lhb_count_30d is not None else None,
        "lhb_inst_net_buy_30d_yi": (
            round(float(lhb_inst_net_buy_30d_yi), 4) if lhb_inst_net_buy_30d_yi is not None else None
        ),
        "lhb_inst_direction": None,
    }
    if lhb_inst_net_buy_30d_yi is not None:
        if lhb_inst_net_buy_30d_yi > _LHB_INST_NET_THRESHOLD_YI:
            out["lhb_inst_direction"] = 1
        elif lhb_inst_net_buy_30d_yi < -_LHB_INST_NET_THRESHOLD_YI:
            out["lhb_inst_direction"] = -1
        else:
            out["lhb_inst_direction"] = 0
    return out


# ---------------------------------------------------------------------------
# 5. 散户成交占比（日度，moneyflow_dc 内置 rate 字段）
# ---------------------------------------------------------------------------
def compute_retail_amount_rate(moneyflow_df: Optional[pd.DataFrame]) -> dict:
    """从 moneyflow rate 字段计算散户（中单+小单）5 日占比均值——**区分毛买占比与净流入占比**。

    两种数据源口径不同，严禁混用：
    - 毛买盘占比（*_buy_amount_rate_pct，如 tushare buy_sm/md_amount_rate）：A 股常 50-70%；
    - 净流入占比（*_net_inflow_rate_pct，如 akshare「中/小单净流入-净占比」）：通常 ±个位数。

    Args:
        moneyflow_df: 含 *_buy_amount_rate_pct（毛）或 *_net_inflow_rate_pct（净）字段

    Returns:
        dict:
        - retail_buy_amount_rate_5d_pct  : 5 日散户毛买盘占比均值（%，仅毛口径源有）
        - retail_net_inflow_rate_5d_pct  : 5 日散户净流入占比均值（%，仅净口径源有，可负）
    """
    out = {"retail_buy_amount_rate_5d_pct": None, "retail_net_inflow_rate_5d_pct": None}
    if moneyflow_df is None or len(moneyflow_df) == 0:
        return out
    df = moneyflow_df.copy().reset_index(drop=True)

    def _sum5(col_md, col_sm):
        parts = []
        if col_md in df.columns:
            parts.append(df[col_md].astype(float))
        if col_sm in df.columns:
            parts.append(df[col_sm].astype(float))
        if not parts:
            return None
        return round(float(sum(parts).tail(5).mean()), 2)

    out["retail_buy_amount_rate_5d_pct"] = _sum5(
        "medium_buy_amount_rate_pct", "small_buy_amount_rate_pct")      # 毛口径
    out["retail_net_inflow_rate_5d_pct"] = _sum5(
        "medium_net_inflow_rate_pct", "small_net_inflow_rate_pct")      # 净口径
    return out


# ---------------------------------------------------------------------------
# 6. Regime 五维投票（强势 / 分化 / 恶化 / 中性 / 数据不足）
# ---------------------------------------------------------------------------
def compute_capital_flow_regime(metrics: dict) -> dict:
    """五维投票判定资金面综合 regime。

    五个独立维度（每维投 + / - / 0 / X）：
    1. streak（连续天数）：≥+3 投 +；≤-3 投 -；≤-5 单维度即可触发"恶化独立票"
    2. ddx_pct_1y（DDX 1 年分位）：≥80 投 +；≤20 投 -
    3. northbound_5d_direction：+1 投 +；-1 投 -；0 投 0；data_status=stale/missing 投 X
    4. lhb_inst_direction（龙虎榜机构席位净买方向）：+1 投 +；-1 投 -；0 投 0；缺失投 X
       （上榜次数 lhb_count_30d 仅作关注度展示，不投方向）
    5. retail（小单+中单买占比）：≥65% 且 streak≤-3 → 投 -（散户高接盘）；其他 0

    Regime 判定：
    - 数据不足：valid 维度数（非 X）< 3
    - 恶化     ：streak ≤ -5  OR  ≥3 维度投 -
    - 强势     ：≥3 维度投 + AND 0 维度投 -
    - 分化     ：同时存在 + 票 和 - 票
    - 中性     ：其他

    Returns:
        dict:
        - capital_flow_regime           : "强势" / "分化" / "恶化" / "中性" / "数据不足"
        - capital_flow_regime_reasoning : 一句话解释（Python 模板生成）
        - capital_flow_votes            : dict 含五维投票结果
        - capital_flow_valid_dimensions : int，有效维度数
    """
    votes = {
        "streak": "X",
        "ddx_pct_1y": "X",
        "northbound": "X",
        "lhb": "X",
        "retail_takeover": "X",
    }

    # 1. streak
    streak = metrics.get("net_inflow_streak_days")
    streak_extreme_negative = False
    if streak is not None:
        if streak <= -5:
            votes["streak"] = "-"
            streak_extreme_negative = True
        elif streak <= -3:
            votes["streak"] = "-"
        elif streak >= 3:
            votes["streak"] = "+"
        else:
            votes["streak"] = "0"

    # 2. ddx_pct_1y
    ddx_pct = metrics.get("ddx_like_5d_pct_1y")
    if ddx_pct is not None:
        if ddx_pct >= 80:
            votes["ddx_pct_1y"] = "+"
        elif ddx_pct <= 20:
            votes["ddx_pct_1y"] = "-"
        else:
            votes["ddx_pct_1y"] = "0"

    # 3. northbound（数据停滞时投 X 不投 0）
    nb_status = metrics.get("northbound_data_status", "missing")
    nb_dir = metrics.get("northbound_5d_direction")
    if nb_status == "fresh" and nb_dir is not None:
        if nb_dir > 0:
            votes["northbound"] = "+"
        elif nb_dir < 0:
            votes["northbound"] = "-"
        else:
            votes["northbound"] = "0"

    # 4. lhb（机构席位方向：净买→+ / 净卖→- / 持平→0 / 缺失→X）
    #    不再用"上榜次数≥2→多头"——上榜是异动/关注度，不是方向
    lhb_dir = metrics.get("lhb_inst_direction")
    if lhb_dir is not None:
        votes["lhb"] = "+" if lhb_dir > 0 else ("-" if lhb_dir < 0 else "0")

    # 5. retail（散户高接盘：主力派发 streak≤-3 + 散户买占比≥65% → 投 -）
    #    用真实小单+中单买占比，不再用恒等 1.0 的旧 retail_takeover_ratio；
    #    收紧到 streak≤-3（不是 <0），消除"主力流出一天"的虚假票
    retail_rate = metrics.get("retail_buy_amount_rate_5d_pct")
    if retail_rate is not None and streak is not None:
        if retail_rate >= _RETAIL_HIGH_RATE_PCT and streak <= _STREAK_DISTRIBUTION:
            votes["retail_takeover"] = "-"
        else:
            votes["retail_takeover"] = "0"

    # 统计
    valid_count = sum(1 for v in votes.values() if v != "X")
    plus_count = sum(1 for v in votes.values() if v == "+")
    minus_count = sum(1 for v in votes.values() if v == "-")

    # Regime 判定
    if valid_count < 3:
        regime = "数据不足"
    elif streak_extreme_negative or minus_count >= 3:
        regime = "恶化"
    elif plus_count >= 3 and minus_count == 0:
        regime = "强势"
    elif plus_count > 0 and minus_count > 0:
        regime = "分化"
    else:
        regime = "中性"

    # 一句话解释
    reasoning = _build_regime_reasoning(regime, votes, metrics)

    return {
        "capital_flow_regime": regime,
        "capital_flow_regime_reasoning": reasoning,
        "capital_flow_votes": votes,
        "capital_flow_valid_dimensions": valid_count,
    }


def _build_regime_reasoning(regime: str, votes: dict, metrics: dict) -> str:
    """根据 regime 和投票结果生成 Python 模板化解释（不调 LLM）。"""
    streak = metrics.get("net_inflow_streak_days")
    ddx_pct = metrics.get("ddx_like_5d_pct_1y")
    nb_status = metrics.get("northbound_data_status", "missing")

    parts = []
    if streak is not None:
        if streak <= -5:
            parts.append(f"主力连续净流出 {abs(streak)} 日（极端恶化）")
        elif streak <= -3:
            parts.append(f"主力连续净流出 {abs(streak)} 日")
        elif streak >= 3:
            parts.append(f"主力连续净流入 {streak} 日")

    if ddx_pct is not None:
        if ddx_pct >= 80:
            parts.append(f"DDX 1 年分位 {ddx_pct:.0f}（高位）")
        elif ddx_pct <= 20:
            parts.append(f"DDX 1 年分位 {ddx_pct:.0f}（低位）")

    if nb_status != "fresh":
        parts.append(f"北向数据 {nb_status}")

    body = "；".join(parts) if parts else "各维度均无显著信号"
    return f"{regime}：{body}。"


# ---------------------------------------------------------------------------
# 7. capital_flow_score 第 7 因子打分（0-100，进 quant_score 加权）
# ---------------------------------------------------------------------------
def compute_capital_flow_score(metrics: dict, regime: str) -> tuple[Optional[float], dict]:
    """capital_flow_score 公式（第 7 因子）：

    cf_score (0-100) = 加权（对标投研资金面权重排序：机构出处 > DDE 推断）
        0.25 × ddx_like_5d_pct_1y                        （DDE 推断，降权）
      + 0.20 × normalize(ddz_like_20d_pct, [-3, +3], [0, 100])  （DDE 推断）
      + 0.15 × clip_normalize(net_inflow_streak_days, [-10, +10], [0, 100])（DDE 推断）
      + 0.20 × dir_to_score(northbound_5d_direction)    （真机构北向；停滞则按比例缩权剔除）
      + 0.15 × dir_to_score(lhb_inst_direction)         （真机构龙虎榜席位）
      + 0.05 × inv_normalize(retail_buy_amount_rate_5d_pct, [50, 75] → [100, 0])（辅助）

    DDE（按单笔大小分主力/散户）是散户级启发式、受算法拆单削弱，故降权；机构出处信号提权。
    缺失子项时按比例缩放剩余权重（total_w 归一），北向停滞时其 0.20 自动剔除、不留空洞。

    Regime 硬约束（业内"双轨制"）：
    - regime == "恶化" → cf_score 强制 ≤ 40
    - regime == "强势" → cf_score 强制 ≥ 60
    - regime == "数据不足" → 返回 None（不打分）

    Returns:
        (score, breakdown_dict) — score ∈ [0, 100] 或 None
    """
    if regime == "数据不足":
        return None, {"reason": "数据不足，cf_score 不计算"}

    ddx_pct = metrics.get("ddx_like_5d_pct_1y")
    ddz_20d = metrics.get("ddz_like_20d_pct")
    streak = metrics.get("net_inflow_streak_days")
    retail_rate = metrics.get("retail_buy_amount_rate_5d_pct")
    nb_dir = metrics.get("northbound_5d_direction")
    nb_status = metrics.get("northbound_data_status", "missing")
    lhb_inst_dir = metrics.get("lhb_inst_direction")

    parts: list[tuple[float, float]] = []  # (sub_score, weight)
    breakdown: dict = {}

    # 方向(-1/0/+1) → 0-100 子分：-1→0, 0→50, +1→100
    def _dir_to_score(d: int) -> float:
        return 50.0 + 50.0 * float(d)

    # 子分 1: ddx_like_5d_pct_1y（已是 0-100，权重 0.25；DDE 推断降权）
    if ddx_pct is not None:
        parts.append((float(ddx_pct), 0.25))
        breakdown["ddx_like_5d_pct_1y"] = round(ddx_pct, 1)

    # 子分 2: ddz_like_20d_pct → 线性映射 [-3, +3] → [0, 100]，权重 0.20
    if ddz_20d is not None:
        sub = _linear_map(float(ddz_20d), -3.0, 3.0)
        parts.append((sub, 0.20))
        breakdown["ddz_like_20d_pct"] = round(ddz_20d, 2)
        breakdown["ddz_sub_score"] = round(sub, 1)

    # 子分 5: 北向 5 日方向（真机构，权重 0.20）；停滞/缺失时跳过 → total_w 自动缩权
    if nb_dir is not None and nb_status == "fresh":
        sub = _dir_to_score(nb_dir)
        parts.append((sub, 0.20))
        breakdown["northbound_5d_direction"] = int(nb_dir)
        breakdown["northbound_sub_score"] = round(sub, 1)

    # 子分 6: 龙虎榜机构席位方向（真机构，权重 0.15）；缺失时跳过
    if lhb_inst_dir is not None:
        sub = _dir_to_score(lhb_inst_dir)
        parts.append((sub, 0.15))
        breakdown["lhb_inst_direction"] = int(lhb_inst_dir)
        breakdown["lhb_inst_sub_score"] = round(sub, 1)

    # 子分 3: net_inflow_streak_days → 截断 [-10,+10] 后线性映射，权重 0.15
    if streak is not None:
        sub = _linear_map(float(streak), -10.0, 10.0)
        parts.append((sub, 0.15))
        breakdown["net_inflow_streak_days"] = int(streak)
        breakdown["streak_sub_score"] = round(sub, 1)

    # 子分 4: 散户承接（散户接盘越重=派发期越偏空→打分越低），权重 0.05。两种口径分别归一化：
    #   毛买占比[50,75]→[100,0]；净流入占比[-10,+10]→[100,0]（净流入越高=散户接盘越重→越空）
    retail_net = metrics.get("retail_net_inflow_rate_5d_pct")
    if retail_rate is not None:
        sub = _linear_map(float(retail_rate), 50.0, 75.0, dst_lo=100.0, dst_hi=0.0)
        parts.append((sub, 0.05))
        breakdown["retail_buy_amount_rate_5d_pct"] = round(retail_rate, 2)
        breakdown["retail_sub_score"] = round(sub, 1)
    elif retail_net is not None:
        sub = _linear_map(float(retail_net), -10.0, 10.0, dst_lo=100.0, dst_hi=0.0)
        parts.append((sub, 0.05))
        breakdown["retail_net_inflow_rate_5d_pct"] = round(retail_net, 2)
        breakdown["retail_sub_score"] = round(sub, 1)

    if not parts:
        return None, {"reason": "所有子项缺失"}

    total_w = sum(w for _, w in parts)
    raw_score = sum(s * w for s, w in parts) / total_w
    breakdown["raw_score_before_regime_clamp"] = round(raw_score, 1)
    breakdown["effective_weight_sum"] = round(total_w, 3)

    # Regime 硬约束
    if regime == "恶化":
        final_score = min(raw_score, 40.0)
        breakdown["regime_clamp"] = "恶化 → ≤40"
    elif regime == "强势":
        final_score = max(raw_score, 60.0)
        breakdown["regime_clamp"] = "强势 → ≥60"
    else:
        final_score = raw_score
        breakdown["regime_clamp"] = "无"

    final_score = _clip(final_score, 0.0, 100.0)
    return round(final_score, 1), breakdown


# ---------------------------------------------------------------------------
# 8. 组装最终 metrics dict（capital_flow_node 调用）
# ---------------------------------------------------------------------------
def assemble_capital_flow_metrics(
    moneyflow_df: Optional[pd.DataFrame] = None,
    north_df: Optional[pd.DataFrame] = None,
    holder_df: Optional[pd.DataFrame] = None,
    circulating_market_value_yi: Optional[float] = None,
    lhb_count_30d: Optional[int] = None,
    lhb_inst_net_buy_30d_yi: Optional[float] = None,
    latest_trade_date: Optional[str] = None,
) -> dict:
    """一次性组装所有资金流字段（含 regime 与 cf_score）。

    Returns:
        dict（全部字段全名）：
        - 见 compute_dde_like_metrics / compute_northbound_metrics /
              compute_holder_number_metrics / compute_lhb_metrics /
              compute_retail_amount_rate / compute_capital_flow_regime
        - capital_flow_score              : 0-100 或 None
        - capital_flow_score_breakdown    : dict 子分明细
        - circulating_market_value_yi     : 流通市值（亿元）
    """
    metrics: dict = {"circulating_market_value_yi": circulating_market_value_yi}

    metrics.update(compute_dde_like_metrics(moneyflow_df, circulating_market_value_yi))
    metrics.update(compute_northbound_metrics(north_df, latest_trade_date=latest_trade_date))
    metrics.update(compute_holder_number_metrics(holder_df))
    metrics.update(compute_lhb_metrics(lhb_count_30d, lhb_inst_net_buy_30d_yi))
    metrics.update(compute_retail_amount_rate(moneyflow_df))

    # 散户接盘信号（散户承接 + 主力派发，替代恒等 1.0 的旧 ratio）——毛买占比/净流入占比两口径
    metrics["retail_concentration_signal"] = compute_retail_concentration_signal(
        metrics.get("retail_buy_amount_rate_5d_pct"),
        metrics.get("net_inflow_streak_days"),
        retail_net_inflow_rate_5d_pct=metrics.get("retail_net_inflow_rate_5d_pct"),
    )

    regime_info = compute_capital_flow_regime(metrics)
    metrics.update(regime_info)

    score, breakdown = compute_capital_flow_score(metrics, metrics["capital_flow_regime"])
    metrics["capital_flow_score"] = score
    metrics["capital_flow_score_breakdown"] = breakdown

    return metrics


# ---------------------------------------------------------------------------
# 9. 中文标签映射（仅供 prompt 与 markdown 报告使用）
# ---------------------------------------------------------------------------
FIELD_LABEL_ZH: dict[str, str] = {
    "main_force_net_inflow_5d_yi":      "5日主力净流入(亿)",
    "main_force_net_inflow_20d_yi":     "20日主力净流入(亿)",
    "ddx_like_5d_pct":                  "DDX-like 5日(%)",
    "ddx_like_5d_pct_1y":               "DDX 1年分位(0-100)",
    "large_order_net_inflow_5d_yi":     "大单净流入5日(亿)",
    "ddz_like_20d_pct":                 "20日主力强度比(%)",
    "net_inflow_streak_days":           "连续净流入/流出天数",
    "retail_buy_amount_rate_5d_pct":    "散户毛买盘占比5日均值(%)",
    "retail_net_inflow_rate_5d_pct":    "散户净流入占比5日均值(%,可负)",
    "retail_concentration_signal":      "散户接盘信号",
    "northbound_5d_direction":          "北向资金5日方向",
    "northbound_20d_direction":         "北向资金20日方向",
    "northbound_data_status":           "北向数据状态",
    "northbound_latest_date":           "北向最新数据日期",
    "lhb_count_30d":                    "龙虎榜30日上榜次数(关注度)",
    "lhb_inst_net_buy_30d_yi":          "龙虎榜机构30日净买(亿)",
    "lhb_inst_direction":               "龙虎榜机构净买方向",
    "holder_num_latest":                "最新股东户数",
    "holder_num_qoq_pct":               "户数环比变化(%)",
    "holder_num_4q_trend":              "户数4季度趋势",
    "chip_concentration_signal":        "筹码集中度信号",
    "circulating_market_value_yi":      "流通市值(亿)",
    "capital_flow_regime":              "资金面综合状态",
    "capital_flow_regime_reasoning":    "资金面状态解释",
    "capital_flow_score":               "资金面综合打分(0-100)",
}
