"""股票画像识别节点（Stock Profile Node）

在 4 个 analyst + Quant Score + Macro Context 完成后、Consensus Officer 之前运行。

杠杆 2 改造（2026-05）：所有"客观可程序化"字段由 Python 确定性判定，LLM 只负责
"语义判断"字段（行业卡识别、theme_name、event_windows、文字说明）。

程序化字段（Python 直接判，LLM 不允许覆盖）：
- market_cap_tier / liquidity / style / instrument_type
- peak_signal（强制把 LLM 的 theme_stage 上锁为 peak）
- REPORT_WEIGHTS 基础值（按 style 查表）

LLM 仍负责：
- industry / 行业框架卡识别
- theme_name + theme_stage（非 peak 时由 LLM 判断）
- event_windows + 权重事件调整
- 文字说明
"""

import logging
import re

from tradingagents.agents.utils.agent_utils import build_instrument_context
from tradingagents.dataflows.factor_calc import compute_price_factors
from tradingagents.dataflows.interface import route_to_vendor
from tradingagents.dataflows.ticker_utils import is_a_share
from tradingagents.dataflows.profile_calc import (
    compute_market_cap_tier,
    compute_liquidity_tier,
    compute_price_signals,
    derive_style,
    detect_peak_signals,
    get_default_weights,
    is_etf_ticker,
    liquidity_tier_label,
    market_cap_tier_label,
    parse_market_cap_from_fundamentals,
    # Layer 1: 硬规则
    parse_eps_ttm,
    detect_forced_valuation_method,
    # Layer 2: 数据参照
    parse_sell_side_pe_consensus,
    compute_self_pe_p80,
    parse_peer_pe_median,
    detect_leadership_bonus,
    compute_default_premium,
    infer_theme_stage_from_data,
    parse_sector_rs_30d,
    parse_pe_ttm_from_fundamentals,
    parse_net_profit_growth,
    compute_peer_anchored_pe_cap,
    compute_valuation_regime,
    parse_growth_deceleration,
    parse_distribution_signals,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 解析 quant_score 报告里的 YAML，提取 factor_scores
# ---------------------------------------------------------------------------
def _parse_quant_score_yaml(quant_md: str) -> dict:
    """从 quant_score 报告末尾的 YAML 摘要中提取 momentum / lowvol / composite 等关键值。

    YAML 块在两个 ``` 之间。容错：解析失败返回 None 字段。
    """
    out = {
        "composite": None,
        "momentum": None,
        "lowvol": None,
        "value": None,
        "quality": None,
        "growth": None,
        "anticrowding": None,
    }
    if not quant_md:
        return out

    in_yaml = False
    in_factor_scores = False
    for line in quant_md.splitlines():
        stripped = line.strip()
        if stripped == "```yaml":
            in_yaml = True
            continue
        if in_yaml and stripped.startswith("```"):
            break
        if not in_yaml:
            continue

        m = re.match(r"^\s*composite:\s*(.+)$", line)
        if m:
            v = m.group(1).strip()
            if v not in ("null", "None", ""):
                try:
                    out["composite"] = float(v)
                except ValueError:
                    pass

        if stripped == "factor_scores:":
            in_factor_scores = True
            continue
        # 离开 factor_scores 段
        if in_factor_scores and not line.startswith("    "):
            in_factor_scores = False
        if in_factor_scores:
            m = re.match(
                r"^\s+(momentum|lowvol|value|quality|growth|anticrowding):\s*(.+)$",
                line,
            )
            if m:
                k, v = m.group(1), m.group(2).strip()
                if v not in ("null", "None", ""):
                    try:
                        out[k] = float(v)
                    except ValueError:
                        pass
    return out


# ---------------------------------------------------------------------------
# 数据拉取（与 quant_score_node 重复，可接受的 V1 冗余）
# ---------------------------------------------------------------------------
def _fetch_price_df_for_profile(ticker: str, trade_date: str):
    import datetime as _dt
    import io
    import pandas as pd

    end = _dt.datetime.strptime(trade_date, "%Y-%m-%d")
    start = end - _dt.timedelta(days=400)
    try:
        csv_str = route_to_vendor(
            "get_stock_data",
            ticker,
            start.strftime("%Y-%m-%d"),
            end.strftime("%Y-%m-%d"),
        )
    except Exception as e:
        logger.warning("stock_profile 获取价格数据失败: %s", e)
        return None
    if not csv_str or "未找到" in csv_str[:200]:
        return None
    lines = [ln for ln in csv_str.splitlines() if not ln.startswith("#") and ln.strip()]
    if not lines:
        return None
    try:
        return pd.read_csv(io.StringIO("\n".join(lines)))
    except Exception as e:
        logger.warning("stock_profile 解析价格 CSV 失败: %s", e)
        return None


def _fetch_fundamentals_raw(ticker: str, trade_date: str) -> str:
    try:
        return route_to_vendor("get_fundamentals", ticker, trade_date)
    except Exception as e:
        logger.warning("stock_profile 获取基本面数据失败: %s", e)
        return ""


def _parse_capital_flow_signals(cf_yaml: str) -> dict:
    """从 capital_flow_yaml 抽 valuation_regime 需要的资金面信号（容错，缺失返回 None）。"""
    out = {"regime": None, "streak": None, "lhb_inst_dir": None, "retail_signal": None}
    if not cf_yaml:
        return out

    def _grab(key):
        # 抓 key 后到行尾（. 不跨行），去引号/空白；null/空 → None
        m = re.search(rf'{key}:\s*(.+)', cf_yaml)
        if not m:
            return None
        v = m.group(1).strip().strip('"').strip()
        return None if v in ("null", "") else v

    def _grab_int(key):
        v = _grab(key)
        if v is None:
            return None
        try:
            return int(v)
        except ValueError:
            return None

    out["regime"] = _grab("capital_flow_regime")  # 注：capital_flow_regime_reasoning 的"_"≠":"，不会误匹配
    out["streak"] = _grab_int("net_inflow_streak_days")
    out["lhb_inst_dir"] = _grab_int("lhb_inst_direction")
    out["retail_signal"] = _grab("retail_concentration_signal")
    return out


def _enforce_target_pe_cap(content: str, cap: float) -> str:
    """出口处硬截断 LLM 输出的 target_pe_range 高位到 cap（防 LLM 无视 prompt 软约束）。

    背景：cap 原本只在 prompt 里"软约束"，LLM 有时无视（如澜起 172516 给 [89.6,116.5]
    却无视 PE_TTM×0.6=72.4 上限 → 目标价漂到 294 → 误评 OVERWEIGHT）。此处把它变成
    程序化硬天花板：只改 YAML 的 target_pe_range 数值字段（下游 RM 据此提取），不动散文。

    仅处理 [low, high] 数值形式；[null, null]（亏损/无锚）不匹配、跳过。
    """
    pat = re.compile(r"(target_pe_range:\s*\[\s*)([0-9.]+)\s*,\s*([0-9.]+)(\s*\])")

    def _repl(m: "re.Match") -> str:
        low, high = float(m.group(2)), float(m.group(3))
        if high <= cap + 1e-6:
            return m.group(0)  # 未超上限，不动
        new_high = round(cap, 1)
        new_low = round(min(low, new_high), 1)
        if new_low >= new_high:  # 防退化成单点，保留一个窄带
            new_low = round(new_high * 0.9, 1)
        logger.warning(
            "stock_profile target_pe 出口硬截断: [%.1f, %.1f] → [%.1f, %.1f] (cap=%.1f)",
            low, high, new_low, new_high, cap,
        )
        return f"{m.group(1)}{new_low}, {new_high}{m.group(4)}  # ⚠️出口硬截断 high≤{cap:.1f}"

    return pat.sub(_repl, content)


# ---------------------------------------------------------------------------
# 主节点
# ---------------------------------------------------------------------------
def create_stock_profile_node(llm):
    def stock_profile_node(state) -> dict:
        ticker = state["company_of_interest"]
        company_name = state.get("company_name", "")
        trade_date = state.get("trade_date", "")
        instrument_context = build_instrument_context(ticker, company_name)

        market_report = state.get("market_report", "")
        sentiment_report = state.get("sentiment_report", "")
        news_report = state.get("news_report", "")
        fundamentals_report = state.get("fundamentals_report", "")
        macro_context = state.get("macro_context", "")
        quant_score_md = state.get("quant_score", "")
        sector_comparison_md = state.get("sector_comparison", "")
        capital_flow_yaml = state.get("capital_flow_yaml", "")

        # === 程序化判定开始 ===
        is_etf = is_etf_ticker(ticker)

        # 1. 拉数据
        price_df = _fetch_price_df_for_profile(ticker, trade_date)
        fund_raw = _fetch_fundamentals_raw(ticker, trade_date)

        # 2. 解析 fundamentals 中的总市值
        market_cap_yi = parse_market_cap_from_fundamentals(fund_raw)

        # 3. 计算价格信号（RSI / 乖离率 / 量价背离 / 日均成交额）
        price_signals = compute_price_signals(price_df)

        # 4. 量化分数（来自 quant_score state）
        quant_yaml = _parse_quant_score_yaml(quant_score_md)
        momentum_score = quant_yaml.get("momentum")
        lowvol_score = quant_yaml.get("lowvol")

        # 5. 程序化判定
        market_cap_tier = compute_market_cap_tier(market_cap_yi)
        liquidity_tier = compute_liquidity_tier(price_signals.get("avg_daily_turnover_yi"))
        style = derive_style(
            market_cap_tier=market_cap_tier,
            momentum_score=momentum_score,
            lowvol_score=lowvol_score,
            liquidity_tier=liquidity_tier,
            is_etf=is_etf,
        )
        peak_check = detect_peak_signals(
            rsi_value=price_signals.get("rsi_value"),
            rsi_percentile_1y=price_signals.get("rsi_percentile_1y"),
            deviation_pct=price_signals.get("deviation_pct"),
            has_vol_divergence=price_signals.get("has_vol_divergence"),
        )
        base_weights = get_default_weights(style)

        # === Layer 1: 硬规则（亏损股 + 行业铁律）===
        # 注：行业铁律的 industry 字段在 LLM 输出后才有，本节点入口只能用 EPS 判定
        # 行业铁律由 prompt 强约束（已存在的行业框架卡 + 下面 LLM 看到 forced_valuation 后自检）
        eps_ttm_val = parse_eps_ttm(fund_raw + "\n" + fundamentals_report)
        # industry 由 LLM 输出，此处只用 EPS 触发 Layer 1，industry 字段传 None
        forced_valuation = detect_forced_valuation_method(industry=None, eps_ttm=eps_ttm_val)

        # === Layer 2: 数据参照（三源 PE / 龙头溢价 / theme_inferred / default premium）===
        sell_side_pe_range = parse_sell_side_pe_consensus(news_report)
        self_pe_p80 = compute_self_pe_p80(price_df, eps_ttm_val)
        peer_pe_median = parse_peer_pe_median(news_report, fundamentals_report)
        peer_pe_source = "report_scrape" if peer_pe_median is not None else None

        # ---- 兄弟股可比 PE（优先源，质量高于报告抠数）：news+sentiment 共现挖掘 + 行业校验 + ≥1家中位 ----
        # 仅 A 股（PE 快照/兄弟表都是 A 股口径）；取数失败或 0 家有效时保持原 peer_pe_median 不变。
        brother_pe = None
        brother_single_comp = False  # 单标的(n=1)低置信 → 下游 Conviction 减一档
        if is_a_share(ticker):
            code_m = re.search(r"\d{6}", ticker)
            if code_m:
                try:
                    from tradingagents.dataflows.peer_comps import get_brother_pe_median
                    comention_text = (news_report or "") + "\n" + (sentiment_report or "")
                    brother_pe = get_brother_pe_median(
                        code_m.group(0), trade_date, comention_text, company_name,
                    )
                except Exception as e:
                    logger.warning("兄弟股可比 PE 取数失败: %s", str(e)[:120])
            if brother_pe:
                peer_pe_median = brother_pe["median"]
                peer_pe_source = "brother_comps"
                brother_single_comp = (brother_pe.get("confidence") == "low")
                logger.info(
                    "兄弟股可比 PE 命中: median=%.1f, n_valid=%s, conf=%s, used=%s",
                    brother_pe["median"], brother_pe.get("n_valid"),
                    brother_pe.get("confidence"), brother_pe.get("used"),
                )

        # ---- Layer 2 兜底：三源全 null 时用 PE_TTM × 0.7 做最后锚（机构 PM "无锚定时保守" 原则）----
        # 这里的 0.7 = "向卖方一致 PE 方向收敛"——A 股 PE 通常比卖方目标 PE 高 30-50%，
        # 取 0.7 倍作为兜底锚等于强制 stock_profile 不能给"PE_TTM 之上"的乐观估值
        # PE_TTM 与净利增速：所有分支都先抽（PE_TTM 用作硬天花板的绝对上限；增速用作 PEG 有界溢价 + 前瞻 EPS 提示）
        pe_ttm_actual = parse_pe_ttm_from_fundamentals(fundamentals_report + "\n" + fund_raw)
        net_profit_growth = parse_net_profit_growth(fundamentals_report + "\n" + fund_raw)

        layer2_all_null = (sell_side_pe_range is None and self_pe_p80 is None and peer_pe_median is None)
        pe_ttm_fallback = None
        if layer2_all_null:
            pe_ttm_fallback = pe_ttm_actual
            if pe_ttm_fallback is not None:
                # 用 PE_TTM × 0.7 灌到 peer_pe_median 位置（标识为 fallback）
                peer_pe_median = pe_ttm_fallback * 0.7
        leadership_bonus_pct, leadership_reason = detect_leadership_bonus(
            fundamentals_report, news_report,
        )
        # 同业锚硬天花板（peer 有值且非全-null 兜底时生效；防 target_pe_high 漂到当前 PE）
        # ⚠️ 仅限 A 股：peer_pe_median 抽自巨潮 cninfo 行业表（A 股专有），美股/港股的"行业 PE 中位数"
        #    若被填成巨潮 A 股口径会错锚——非 A 股不套此 cap，交回 LLM 用本地同业/卖方（亏损股已由 Layer 1 转 PB/PS）
        is_a_share_stock = is_a_share(ticker)
        peer_pe_cap = None
        if is_a_share_stock and not layer2_all_null and peer_pe_median:
            peer_pe_cap = compute_peer_anchored_pe_cap(
                peer_pe_median=peer_pe_median,
                pe_ttm=pe_ttm_actual,
                net_profit_growth=net_profit_growth,
                leadership_bonus_pct=leadership_bonus_pct,
            )
        sector_rs_30d = parse_sector_rs_30d(sector_comparison_md)
        theme_inferred, theme_reason = infer_theme_stage_from_data(
            momentum_score=momentum_score,
            sector_rs_30d=sector_rs_30d,
            rsi_percentile_1y=price_signals.get("rsi_percentile_1y"),
            has_peak_signal=peak_check["force_peak"],
        )
        # 宏观修正（macro_context 由 LLM 已经生成；此处简化用 0，LLM 在 YAML 输出时自填）
        default_premium_pct, default_premium_formula = compute_default_premium(
            theme_stage_inferred=theme_inferred,
            leadership_bonus_pct=leadership_bonus_pct,
            macro_adjustment_pct=0,
        )

        # 客观估值 regime（六路合成：技术/资金/盈利/拥挤/主题/派发 → ride/neutral/discipline）
        # 决定估值姿态（cap 松紧）的是这六路分析师，不是估值锚本身
        cf_sig = _parse_capital_flow_signals(capital_flow_yaml)
        growth_dir = parse_growth_deceleration(fundamentals_report + "\n" + fund_raw)
        dist_sig = parse_distribution_signals(news_report, fundamentals_report, sentiment_report)
        regime_info = compute_valuation_regime(
            momentum_score=momentum_score,
            rsi_percentile_1y=price_signals.get("rsi_percentile_1y"),
            has_peak_signal=peak_check["force_peak"],
            capital_flow_regime=cf_sig["regime"],
            main_force_streak_days=cf_sig["streak"],
            lhb_inst_direction=cf_sig["lhb_inst_dir"],
            net_profit_growth=net_profit_growth,
            growth_direction=growth_dir,
            retail_concentration_signal=cf_sig["retail_signal"],
            theme_stage_inferred=theme_inferred,
            quant_anticrowding=quant_yaml.get("anticrowding"),
            distribution_detected=dist_sig["detected"],
        )
        valuation_regime = regime_info["valuation_regime"]
        if dist_sig["detected"]:
            logger.info("派发证据: %s", dist_sig["reasons"][:3])
        logger.info("valuation_regime: %s | %s", valuation_regime, regime_info["reasoning"])

        # === 程序化判定结束 ===

        # 拼装"已确定字段"展示段，喂给 LLM 当 ground truth
        prog_lines = ["## 已确定字段（Python 程序化判定，禁止修改）", ""]
        prog_lines.append("| 字段 | 取值 | 程序化判定依据 |")
        prog_lines.append("|------|------|--------------|")
        prog_lines.append(
            f"| market_cap_tier | **{market_cap_tier or 'N/A'}**（{market_cap_tier_label(market_cap_tier)}） | "
            f"总市值 ≈ {market_cap_yi:.0f} 亿（来源：fundamentals 公司信息）|"
            if market_cap_yi else
            f"| market_cap_tier | **{market_cap_tier or 'N/A'}** | 总市值数据缺失 |"
        )
        prog_lines.append(
            f"| liquidity | **{liquidity_tier or 'N/A'}**（{liquidity_tier_label(liquidity_tier)}） | "
            f"60 日日均成交额 ≈ {price_signals['avg_daily_turnover_yi']:.2f} 亿（OHLCV 直接算）|"
            if price_signals.get("avg_daily_turnover_yi") is not None else
            f"| liquidity | **{liquidity_tier or 'N/A'}** | 价格数据缺失 |"
        )
        prog_lines.append(
            f"| style | **{style or 'N/A'}** | "
            f"market_cap_tier={market_cap_tier} / quant.momentum={momentum_score} / quant.lowvol={lowvol_score} / "
            f"is_etf={is_etf} 联合判定 |"
        )
        prog_lines.append(
            f"| instrument_type | **{'etf' if is_etf else 'a_share_stock'}** | 代码模式识别 |"
        )
        prog_lines.append(
            f"| valuation_regime | **{valuation_regime}**（ride骑趋势/neutral中性/discipline收纪律）| "
            f"六路合成(技术/资金/盈利/拥挤/主题/派发)：{regime_info['reasoning']}。"
            f"→ ride 放松估值 cap、骑趋势；discipline 收紧 cap、均值回归；下游 RM/PM 据此定姿态 |"
        )

        # peak 检测
        prog_lines.append("")
        if peak_check["force_peak"]:
            sig_str = "; ".join(peak_check["signals_triggered"])
            prog_lines.append(f"**⛔ Peak 信号强制触发**：{sig_str}")
            prog_lines.append("→ theme_stage **必须** = peak（即使新闻面看起来仍在 acceleration 也不允许）")
        else:
            prog_lines.append(f"Peak 信号未触发（n_signals={peak_check['n_signals']}），theme_stage 可由你结合 news 判断。")

        # 基础权重
        prog_lines.append("")
        if base_weights:
            prog_lines.append(f"**REPORT_WEIGHTS 基础值（style={style} 查表）**：")
            prog_lines.append(
                f"- fundamentals: {base_weights['fundamentals']}% / market: {base_weights['market']}% "
                f"/ news: {base_weights['news']}% / sentiment: {base_weights['sentiment']}%"
            )
            prog_lines.append(
                "→ 这是基础权重。你只能在此基础上做**事件驱动微调**（如财报前 ±5-10%），不能整体重写。"
            )
        else:
            prog_lines.append("REPORT_WEIGHTS 基础值缺失（style 无法判定），你按通用 30/25/25/20 默认处理。")

        # 调试用：价格信号
        prog_lines.append("")
        prog_lines.append("**底层量化信号（供你参考）**：")
        prog_lines.append(
            f"- RSI={price_signals.get('rsi_value')} (1Y 分位={price_signals.get('rsi_percentile_1y')}) "
            f"/ 20MA 乖离率={price_signals.get('deviation_pct')}% "
            f"/ 量价背离={price_signals.get('has_vol_divergence')}"
        )
        prog_lines.append(
            f"- 量化复合分={quant_yaml.get('composite')} / momentum={momentum_score} "
            f"/ lowvol={lowvol_score} / value={quant_yaml.get('value')} "
            f"/ quality={quant_yaml.get('quality')} / growth={quant_yaml.get('growth')}"
        )

        # === Layer 1: 估值范式硬规则 ===
        prog_lines.append("")
        prog_lines.append("## 估值范式硬规则（Layer 1，必须遵守）")
        if forced_valuation["force_valuation"]:
            prog_lines.append(f"**⛔ 强制约束**：{forced_valuation['reason']}")
            if forced_valuation["forbid_pe"]:
                prog_lines.append(
                    f"→ `target_pe_range` **必须**填 `[null, null]`，**禁止**强行套 PE 公式"
                )
            allowed = forced_valuation.get("allowed_primary_methods")
            if allowed:
                prog_lines.append(
                    f"→ `primary_method` **必须**从 `{allowed}` 中选（推荐 `{forced_valuation['forced_primary_method']}`）"
                )
        else:
            eps_str = f"{eps_ttm_val:.2f}" if eps_ttm_val is not None else "未抽到"
            prog_lines.append(f"EPS_TTM = {eps_str}，盈利股，PE 估值可用。如行业属银行/保险/REIT/公用事业，仍需走 PB/DDM。")

        # === Layer 2: 数据参照（喂给 LLM 当估值锚定参照，不强制）===
        prog_lines.append("")
        prog_lines.append("## 估值锚定三源参照（Layer 2，LLM 主选但要透明）")
        prog_lines.append("")
        prog_lines.append("**三源 PE 参照值**（用于 `target_pe_range` 决策）：")
        if sell_side_pe_range:
            prog_lines.append(f"- 卖方一致预期 PE: [{sell_side_pe_range[0]:.1f}, {sell_side_pe_range[1]:.1f}]（来自 news 报告）")
        else:
            prog_lines.append("- 卖方一致预期 PE: 未抽到（news 报告无明确卖方一致 PE 区间）")
        if self_pe_p80:
            prog_lines.append(f"- 自身历史 1Y PE 80% 分位: {self_pe_p80:.1f}")
        else:
            prog_lines.append("- 自身历史 1Y PE 80% 分位: 不适用（亏损或数据不足）")
        if peer_pe_median:
            if pe_ttm_fallback is not None:
                prog_lines.append(
                    f"- 同业/行业 PE 中位数: {peer_pe_median:.1f}（**⚠️ Layer 2 三源全 null，用 PE_TTM×0.7 = "
                    f"{pe_ttm_fallback:.1f}×0.7 兜底**）"
                )
            else:
                src_label = {
                    "brother_comps": "兄弟股可比中位（共现挖掘+行业校验，tushare 实测 PE）",
                    "report_scrape": "来自 news/fundamentals",
                }.get(peer_pe_source, "来自 news/fundamentals")
                prog_lines.append(f"- 同业/行业 PE 中位数: {peer_pe_median:.1f}（{src_label}）")
                if brother_single_comp:
                    prog_lines.append(
                        "  - ⚠️ **兄弟股可比仅 1 家（单标的低置信）**：无第二家纠偏，估值锚可靠性打折。"
                        "→ 下游 RM/PM **Conviction 必须减一档**（见 TRANSPARENCY.peer_anchor_single_comp）。"
                    )
        else:
            prog_lines.append("- 同业/行业 PE 中位数: 未抽到（且 PE_TTM 也抽不到，无任何兜底锚）")

        # ---- Layer 2 三源全 null 时的硬约束（防 213401 类 "宽锚伪 HOLD" 漂移）----
        if layer2_all_null:
            prog_lines.append("")
            prog_lines.append("⛔ **Layer 2 数据缺失硬约束（防漂移）**：")
            prog_lines.append("Layer 2 三源（卖方一致 / 自身历史 P80 / 同业 PE）全部为 null。")
            if peer_pe_median:
                pe_cap = pe_ttm_fallback * 0.6
                prog_lines.append(
                    f"- 已用 PE_TTM = {pe_ttm_fallback:.1f} 做最后兜底锚（peer_pe_median 字段填 {peer_pe_median:.1f}）"
                )
                prog_lines.append(
                    f"- **target_pe_high 严禁超过 PE_TTM × 0.6 = {pe_cap:.1f}**"
                    f"（机构 PM `无外部锚定时保守`原则——禁止 LLM 在数据缺失时自由发挥给宽 PE）"
                )
            else:
                prog_lines.append("- PE_TTM 也抽不到，无任何外部锚")
                prog_lines.append(
                    "- **target_pe_range 必须填 [null, null]**，primary_method 必须从 [pb, ps, ev_ebitda] 选"
                )
            prog_lines.append(
                "- 这是为了避免 `数据没抓到时 LLM 自由发挥给乐观 target_pe → 评级被宽锚漂移` 的 bug"
            )

        # ---- Layer 2 有同业锚时的硬天花板（防 target_pe_high 漂到当前 PE，按投研团队做法）----
        elif peer_pe_cap:
            prog_lines.append("")
            prog_lines.append("⛔ **target_pe_high 硬天花板（同业锚 + PEG 有界溢价，防漂移）**：")
            prog_lines.append(
                f"- **target_pe_high 严禁超过 {peer_pe_cap['cap']:.1f}**"
                f"（= {peer_pe_cap['formula']}）"
            )
            growth_str = f"{net_profit_growth*100:.0f}%" if net_profit_growth is not None else "未抽到"
            prog_lines.append(
                f"- 同业锚 = {peer_pe_median:.1f}（**TTM 口径**，来自巨潮行业中位）；归母净利增速(年度) = {growth_str}；"
                f"PEG 逻辑：增速越快给越高溢价，已按 PEG 证据封顶（≤+40%）"
            )
            prog_lines.append(
                "- ⚠️ **口径一致铁律**：target_pe 与同业锚同为 **TTM 口径**，目标价 ≈ `target_pe × EPS_TTM`。"
                "**严禁把 TTM 口径 PE 乘以前瞻 EPS（EPS_TTM×(1+增速)）**——那样会双重计入成长、把目标价虚高近 50%。"
            )
            prog_lines.append(
                "- 理由：成熟投研团队相对估值要求『PE 口径与 EPS 口径一致』；"
                "禁止用『现价的贵倍数』反推目标价（绝对上限 ≤ PE_TTM）"
            )

        # ---- EPS 口径锁定（所有分支通用；修"TTM 倍数 × 前瞻 EPS"双重计入，澜起派发期被错抬成 OW 的根因）----
        if not (forced_valuation["force_valuation"] and forced_valuation["forbid_pe"]):
            eps_ttm_disp = f"{eps_ttm_val:.2f}" if eps_ttm_val is not None else "未抽到"
            prog_lines.append("")
            prog_lines.append("⛔ **EPS 口径锁定（防 TTM 倍数 × 前瞻 EPS 双重计入）**：")
            prog_lines.append(f"- **EPS_TTM = {eps_ttm_disp} 元**（系统给值；下游 RM 直接用，禁止重算/换口径）")
            if eps_ttm_val is not None and net_profit_growth is not None:
                fwd_eps = eps_ttm_val * (1 + net_profit_growth)
                prog_lines.append(
                    f"- 前瞻 EPS 参考 ≈ {fwd_eps:.2f} 元（= EPS_TTM×(1+增速{net_profit_growth*100:.0f}%)，**仅 PEG 腿用**）"
                )
            prog_lines.append(
                "- **铁律**：`target_pe_range` / 同业中位 / `PE_TTM×0.x` 全是 **TTM 倍数** → 下游 RM 的 "
                "`PE×EPS` 腿 和 `同业可比` 腿 **必须乘 EPS_TTM**；**只有 PEG 腿配前瞻 EPS**。"
            )
            prog_lines.append(
                "- ❌ 严禁：同业中位(TTM) × 前瞻/2026E EPS（双重计入成长 → 目标价虚高 ~50% → 高估值股被错抬成强买）。"
            )

        prog_lines.append("")
        prog_lines.append("**主题阶段量化推断（参照值，LLM 可不采纳但要解释）**：")
        prog_lines.append(f"- `theme_stage_inferred` = **{theme_inferred}**")
        prog_lines.append(f"- 推断依据: {theme_reason}")

        prog_lines.append("")
        prog_lines.append("**默认 premium_tolerance_pct（按 theme_stage 查表 + 龙头溢价 + 宏观）**：")
        prog_lines.append(f"- `default_premium_pct` = **{default_premium_pct}**")
        prog_lines.append(f"- 计算公式: {default_premium_formula}")
        if leadership_bonus_pct > 0:
            prog_lines.append(f"- 龙头溢价识别: +{leadership_bonus_pct}% （{leadership_reason}）")
        if sector_rs_30d is not None:
            prog_lines.append(f"- 板块 RS 30d: {sector_rs_30d:+.1f}%（来自 sector_comparison）")

        # === Layer 3: 透明化标注要求 ===
        prog_lines.append("")
        prog_lines.append("## 透明化标注（Layer 3，YAML 末尾 TRANSPARENCY 段必填）")
        prog_lines.append("你的 target_pe_range 和 premium_tolerance_pct 可在 Layer 2 参照值基础上自由调，**但必须在 YAML 末尾透明标注偏离程度**——下游 RM/PM 会按你的偏离幅度调 Conviction。")

        programmatic_block = "\n".join(prog_lines)

        # 给 LLM YAML 模板填空用的中间变量（避免 f-string 模板里复杂取值）
        sell_side_low_str = f"{sell_side_pe_range[0]:.1f}" if sell_side_pe_range else "null"
        sell_side_high_str = f"{sell_side_pe_range[1]:.1f}" if sell_side_pe_range else "null"
        self_p80_str = f"{self_pe_p80:.1f}" if self_pe_p80 else "null"
        peer_median_str = f"{peer_pe_median:.1f}" if peer_pe_median else "null"
        peer_anchor_source_str = peer_pe_source or "none"
        peer_single_comp_str = "true" if brother_single_comp else "false"

        prompt = f"""【语言要求】你必须使用中文撰写以下分析。股票代码和技术指标名称可保留英文。

你是投研团队的**股票画像识别官（Stock Profile Officer）**。

{instrument_context}
交易日期：{trade_date}

---

{programmatic_block}

---

## 你的任务（在已确定字段基础上补全）

1. **直接采用**上方"已确定字段"中的值（market_cap_tier / liquidity / style / instrument_type / 基础权重）
2. **行业识别 + 行业框架卡**：你负责判断 industry 并匹配下方行业卡（必须）
3. **theme_stage**：若上方 Peak 信号已强制触发，必须用 peak；否则你根据 news/sentiment 判定（initiation / acceleration / peak / fading / none）
4. **theme_name + premium**：根据当前主流主题清单判断 + 叠加宏观修正
5. **EVENT_WINDOWS**：识别近 30 天事件，给出权重微调（不超过 ±10%）
6. **最终 REPORT_WEIGHTS**：基础权重 + 事件调整，加总仍 = 100%
7. **DECISION_STYLE + VALUATION_METHOD**：按 style 选择
8. **文字说明**：补充每个字段的判断理由

---

## 输出结构

### 一、股票画像（直接 echo 已确定字段，加文字补充）

| 维度 | 取值 | 判断依据 |
|------|------|---------|
| **市值层级** | （直接采用已确定字段）| 总市值数据来源 |
| **行业** | （你判断）| 引用 fundamentals 行业归类 |
| **股性风格** | （直接采用已确定字段 style）| Python 程序化判定 |
| **流动性档** | （直接采用已确定字段）| 60 日日均成交额数据 |
| **品种类型** | （直接采用已确定字段 instrument_type）| 代码识别 |

### 二、4 份报告推荐权重（基础值 + 事件调整）

| 报告 | 基础权重 | 事件调整 | 最终权重 |
|------|---------|---------|---------|
| 📊 Fundamentals | （已确定字段）| __ | __ |
| 📈 Market | （已确定字段）| __ | __ |
| 📰 News | （已确定字段）| __ | __ |
| 💬 Sentiment | （已确定字段）| __ | __ |
| **合计** | 100% | — | **100%** |

事件调整必须可追溯到具体事件；单项调整不超过 ±10%；总和仍 = 100%。

### 三、决策风格

按 style 选 1 种（参考下表）：

| style | 推荐 DECISION_STYLE |
|-------|-------------------|
| blue_chip | value_anchor |
| high_beta_growth | catalyst_driven |
| theme_speculation | momentum |
| cyclical | event_driven 或 value_anchor |
| illiquid | catalyst_driven（长 Time Stop）|
| etf | momentum（看技术面）|

输出：`推荐风格：[XXX]。理由：__`

### 四、时间窗口事件（近 30 天）

| 事件 | 日期 | 类型 | 对权重的临时影响 |
|------|------|------|---------------|
| 例：Q2 财报披露 | 2026-08-15 | 财报 | 基本面权重 +5% |

无则填"无（保持基础权重）"。

### 五、行业框架卡（必填，按 fundamentals.industry 匹配）

| 行业 | 估值方法（主→辅）| 关键驱动 | 景气信号 | 典型风险 |
|------|----------------|---------|---------|---------|
| **半导体设计**（含 IC 设计/存储/接口）| PEG / PE×EPS / 历史分位 | 技术节点 / 客户认证 / 周期位置 | HBM 价格、DDR5 渗透率、北美 capex | 出口管制 / 客户集中 / 周期下行 |
| **半导体设备**（光刻/刻蚀/封测设备）| PE×EPS / PB / 同业可比 | 资本开支周期 / 国产化率 | 晶圆厂资本开支、国产替代节奏 | 周期回落 / 技术追赶失败 |
| **CPO/光通信**（光模块/光器件）| PEG / 同业可比 / 卖方目标价 | AI 算力订单 / 800G/1.6T 时间表 | 北美 hyperscaler 订单、客户结构 | 单一大客户依赖 / 技术迭代风险 |
| **新能源车**（整车/三电）| PE×EPS / PEG / 同业可比 | 销量 / 渗透率 / 单车价值量 | 月度销量、单价、毛利率 | 价格战 / 补贴退坡 / 库存 |
| **消费白马**（食品饮料/必选）| PE / DCF / EV/EBITDA | 渠道 / 品牌 / 客单价 | 同店销售、经销商库存 | 库存堆积 / 大宗成本 / 渠道变化 |
| **互联网平台** | PE / PS / DCF | 用户活跃 / 货币化率 | MAU/DAU、ARPU、留存 | 监管 / 流量见顶 / 竞争加剧 |
| **生物医药**（创新药）| DCF（管线 NPV）/ EV/Sales / 同业可比 | 临床进度 / FDA/NMPA 批准 / 销售放量 | 临床数据公告、商业化进度 | 临床失败 / 集采 / 专利悬崖 |
| **CRO/CDMO** | PE / PEG / DCF | 订单 / 产能利用率 | 新签订单、产能 | 海外订单转移 / 价格战 |
| **银行** | PB / DDM（股息折现）/ 历史分位 | 利差 / 资产质量 / ROE | NIM / 不良率 / 拨备覆盖 / 信贷增速 | 利率倒挂 / 房地产敞口 / 资产质量恶化 |
| **券商** | PB / PE / 同业可比 | 市场成交量 / 投行业务 / 自营 | 日均成交、IPO/再融资节奏 | 市场低迷 / 监管收紧 |
| **房地产** | PB / NAV（资产重估）| 销售 / 土地储备 / 杠杆 | 销售面积、回款、融资成本 | 销售失速 / 债务违约 / 政策 |
| **公用事业**（电力/水务）| DCF / PB / 股息率 | 价格调整 / 容量增长 / 现金流 | 上网电价、煤价、装机容量 | 政策风险 / 燃料成本 |
| **算力租赁/IDC** | PE×EPS / EV/EBITDA / 同业可比 | 上架率 / 单机柜价格 / 电力成本 | 上架率、新签客户、电力成本 | 上架率不及预期 / 电力价格上行 |
| **AI 应用**（SaaS / 工具）| PS / 用户增长率 / DCF | ARR / 用户增长 / NRR | MAU、ARR 增速、NRR | 流量见顶 / 商业化不及预期 |

匹配规则：根据 fundamentals/news 中的 industry 字段选最接近的卡；冷门行业标"无标准行业卡"。

### 六、估值方法推荐（按 style）

| style | primary_method | secondary_methods |
|-------|--------------|-------------------|
| blue_chip | dcf / pe_eps（看公司类型）| 历史分位、同业可比 |
| high_beta_growth | peg | pe_eps、同业可比 |
| theme_speculation | 历史分位 | 市值天花板、卖方目标价 |
| cyclical | pb | 周期顶/底 PE、同业可比 |
| illiquid | pb × 0.8（流动性折价）| pe_eps、历史分位 |
| etf | nav / 跟踪指数估值 | 折溢价率 |

**目标 PE/PB 区间**（如适用）必须有依据（行业平均/历史分位/同业），不能凭印象。

### 七、主题热度识别

#### 主题清单（参考）

AI 算力 / CPO 光通信 / 算力租赁 / 算电 / 可控核聚变 / 量子计算 / 低空经济 / 智能驾驶 / 人形机器人 / 国产替代

#### 主题阶段判定（4 选 1）

| 阶段 | 触发信号 | 容忍系数 |
|------|---------|---------|
| initiation 启动期 | 第一次集中报道 + 少数龙头启动 + 卖方开始覆盖 | +30% |
| acceleration 加速期 | 主题已持续 3-6 月 + 多股共振 + 主流财经媒体反复报道 | **+50%** |
| peak 顶部期 | 持续 6+ 月 + 调整出现 + 部分龙头 RSI 极端 | +20% |
| fading 退潮期 | 主题热度消退 + 利空增多 + 板块整体下跌 | **-20%** |
| none 不在主题 | 标的不属于任何当前活跃主题 | 0% |

⚠️ **Peak 信号已强制触发时（见已确定字段顶部），theme_stage 必须 = peak，不允许给 acceleration**。

#### 宏观修正（叠加在 premium_tolerance_pct 之上）

最终 = 基础（按 theme_stage）+ macro_context.premium_adjustment_pct

如：AI 算力 acceleration 基础 +50% + macro 紧缩 -20% = **最终 +30%**

---

## 输入资料

**【最重要】宏观上下文**：

{macro_context if macro_context else "（宏观上下文缺失，按中性环境处理）"}

[置信度:高] Company fundamentals report:
{fundamentals_report}

[置信度:中高] Market research report:
{market_report}

[置信度:中] Latest world affairs news:
{news_report}

[置信度:中低] Social media sentiment report:
{sentiment_report}

---

**最终输出要求**：
- 用中文撰写，结构严格按上述七部分
- 末尾输出 YAML 摘要（已确定字段必须按 Python 程序化结果填写，事件调整后的最终 REPORT_WEIGHTS 填到下方）：

```yaml
PROFILE:
  market_cap_tier: {market_cap_tier or 'null'}        # 已确定，直接复制
  industry: <你判断的行业>
  style: {style or 'null'}                            # 已确定，直接复制
  liquidity: {liquidity_tier or 'null'}               # 已确定，直接复制
  instrument_type: {'etf' if is_etf else 'a_share_stock'}  # 已确定，直接复制

REPORT_WEIGHTS:    # 基础权重 + 事件调整后的最终值
  fundamentals: __
  market: __
  news: __
  sentiment: __

DECISION_STYLE: value_anchor / catalyst_driven / momentum / event_driven

VALUATION_METHOD:
  primary_method: dcf / pe_eps / peg / pb_bps / ev_ebitda / historical_quantile / nav
  secondary_methods:
    - <方法 1>
    - <方法 2>
  target_pe_range: [__, __]
  target_pb_range: [__, __]
  data_completeness: L0 / L1 / L2 / L3
  rationale: <1-2 句话>

EVENT_WINDOWS:
  - event: <事件描述>
    date: YYYY-MM-DD
    impact: <对权重的临时调整>

THEMATIC_PREMIUM:
  is_active_theme: yes / no
  theme_name: <如 AI算力 / CPO / 不在主题 等>
  theme_stage: {'peak（已被 Peak 信号强制锁定）' if peak_check['force_peak'] else 'initiation / acceleration / peak / fading / none'}
  premium_tolerance_pct: <整数>
  rationale: <1-2 句话，引用具体信号>

# Layer 3 透明化标注（强制必填，用于下游 RM/PM Conviction 校准）
TRANSPARENCY:
  # —— 估值锚定偏离度（target_pe_range 高位 vs 三源参照）——
  target_pe_high_vs_sell_side_pct: <整数百分比，如 +35 表示 LLM 选的 target_pe_high 比卖方一致 PE 高 35%；卖方未抽到时填 null>
  target_pe_high_vs_self_p80_pct: <整数百分比；自身历史 P80 未抽到时填 null>
  target_pe_high_vs_peer_median_pct: <整数百分比；同业 PE 未抽到时填 null>
  # —— 主题阶段是否与量化推断一致 ——
  theme_stage_inferred_by_data: {theme_inferred}
  theme_stage_llm_chosen: <你最终选的 theme_stage>
  theme_divergence_reason: <若 inferred 与 chosen 不同，必填理由；一致填 "consistent">
  # —— premium 偏离默认模板 ——
  premium_default_template: {default_premium_pct}
  premium_llm_chosen: <你最终选的 premium_tolerance_pct>
  premium_divergence_reason: <若偏离 default ±15 以上，必填理由；否则填 "within_default_range">
  # —— 参照值原始数据（Python 已算好，直接抄）——
  sell_side_pe_low_ref: {sell_side_low_str}
  sell_side_pe_high_ref: {sell_side_high_str}
  self_pe_p80_ref: {self_p80_str}
  peer_pe_median_ref: {peer_median_str}
  peer_anchor_source: {peer_anchor_source_str}        # brother_comps / report_scrape / none（Python 预填，直接抄）
  peer_anchor_single_comp: {peer_single_comp_str}     # true=兄弟股仅1家(低置信)→下游 Conviction 减一档（Python 预填，直接抄）
  leadership_bonus_pct: {leadership_bonus_pct}
  sector_rs_30d_pct: {sector_rs_30d if sector_rs_30d is not None else 'null'}
```

**TRANSPARENCY 字段填写规则**：
1. 偏离百分比按 LLM 自选值 vs 参照值算：`(llm_chosen - reference) / reference × 100`
2. 参照值缺失（null）时对应偏离字段填 null，**不要编造**
3. `theme_divergence_reason` 和 `premium_divergence_reason` 是给下游 RM/PM 看的——超共识时必须有产业证据支撑（如"刚出 H100 量产订单 / Q1 业绩超预期 60% / 主题情绪急速升温"），否则下游会自动降 Conviction

⛔ **TRANSPARENCY schema 封闭约束**：
- TRANSPARENCY 段**字段集严格封闭**——只能输出上述模板列出的字段名
- **禁止**自创字段（如 `_v2` / `_corrected` / `_adjusted` 后缀，或任何其他自定义字段名）
- 偏离值如有不同算法解释，统一在对应 `_reason` 字段的文字里说明，不要新增字段
- YAML 字段顺序按模板出现顺序，不要重排
- 这是下游 Python 程序化解析依赖的固定 schema，schema 漂移会导致归档失败
"""

        response = llm.invoke(prompt)
        content = response.content

        # 出口硬截断 target_pe_high —— regime 条件化（修"主升浪被低 cap 压死"）
        # discipline/neutral：用纪律 cap（同业锚 / PE_TTM×0.6）；
        # ride（主升浪）：解除纪律 cap，仅用 PE_TTM 兜底（不许超当前贵倍数、但不向下压，允许倍数不收缩）
        if is_a_share_stock:
            discipline_cap = None
            if peer_pe_cap:
                discipline_cap = peer_pe_cap["cap"]
            elif layer2_all_null and pe_ttm_fallback:
                discipline_cap = pe_ttm_fallback * 0.6

            eff_cap = pe_ttm_actual if valuation_regime == "ride" else discipline_cap
            if eff_cap is not None:
                content = _enforce_target_pe_cap(content, eff_cap)
                logger.info(
                    "target_pe 出口 cap: regime=%s → eff_cap=%.1f", valuation_regime, eff_cap,
                )

        return {"stock_profile": content}

    return stock_profile_node
