import logging
import json
import re

from langchain_core.messages import AIMessage, HumanMessage

from tradingagents.agents.utils.agent_utils import build_instrument_context, get_language_instruction, RISK_DEBATE_PHRASING_RULES
from tradingagents.agents.managers.pm_tools import (
    PM_TOOLS,
    PM_TOOLS_BY_NAME,
    compute_r_multiple_levels,
)
from tradingagents.agents.managers.research_manager import (
    _derive_entry_timing_from_profile,
    _enforce_entry_timing_truth,
    _extract_rm_rating,
    _run_tool_calling_loop,
)
from tradingagents.agents.managers.rm_tools import derive_market_mode

logger = logging.getLogger(__name__)

_MAX_TOOL_ITERATIONS = 6

_ENTRY_PRESENTATION = {
    "分批介入": (
        "结构与风险门控允许分批进入，避免一次性追价",
        "按计划分批执行，并持续检查资金与市场环境",
    ),
    "小仓试探": (
        "条件尚未完全确认，只允许小仓验证",
        "结构确认且风险条件改善后再评估扩大仓位",
    ),
    "等回踩": (
        "趋势仍在，但当前位置不具备理想赔率",
        "回踩企稳且资金未恶化后重新评估",
    ),
    "等放量突破": (
        "突破条件尚未得到量价确认",
        "放量突破关键位置并站稳后重新评估",
    ),
    "暂不介入": (
        "当前风险条件未解除，不追高、不新开仓",
        "资金转正、业绩兑现且价格结构企稳后重新评估",
    ),
    "退出观察": (
        "短期结构已经破坏，当前不具备参与条件",
        "结构修复并重新通过风险门控后再观察",
    ),
    "继续观察": (
        "当前结构缺少明确方向，暂不执行交易",
        "趋势或催化形成明确信号后重新评估",
    ),
    "数据不足": (
        "关键数据不足，无法形成可靠的短期操作判断",
        "补齐价格、资金和市场环境数据后重新评估",
    ),
}


def _extract_pm_summary_value(content: str, key: str) -> str | None:
    matches = re.findall(rf"(?m)^\s*{re.escape(key)}:\s*([^#\r\n]+?)\s*$", content or "")
    return matches[-1].strip().strip('"\'') if matches else None


def _format_position_size(low: str | None, high: str | None) -> str:
    def clean(value: str | None) -> str | None:
        if value is None or value.lower() == "null":
            return None
        try:
            number = float(value)
        except ValueError:
            return value
        return str(int(number)) if number.is_integer() else str(number)

    low_value = clean(low)
    high_value = clean(high)
    if low_value is None and high_value is None:
        return "数据不足"
    if low_value == high_value or high_value is None:
        return f"{low_value}%"
    if low_value is None:
        return f"0-{high_value}%"
    return f"{low_value}-{high_value}%"


def _format_price(value: str | None) -> str | None:
    if value is None or value.lower() == "null":
        return None
    try:
        number = float(value)
    except ValueError:
        return value
    return str(int(number)) if number.is_integer() else f"{number:.2f}".rstrip("0").rstrip(".")


def _summary_float(content: str, key: str) -> float | None:
    value = _extract_pm_summary_value(content, key)
    if value is None or value.lower() == "null":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _enforce_holder_r_levels(
    report_body: str,
    summary_block: str | None,
    source: str,
) -> tuple[str, str | None]:
    """Use current price as the holder risk basis while new entry is blocked."""
    summary_source = summary_block or source
    current_price = _summary_float(summary_source, "current_price")
    sl_hard = _summary_float(summary_source, "pm_sl_hard")
    if current_price is None or sl_hard is None or sl_hard >= current_price:
        return report_body, summary_block

    levels = compute_r_multiple_levels.invoke({
        "entry_price": current_price,
        "sl_hard_price": sl_hard,
    })
    if "error" in levels:
        return report_body, summary_block

    current = _format_price(str(levels["entry_price"]))
    replacements = {
        "Entry": "| **Entry** 入场区间 | **—** | 当前不建仓；重新评估后重算入场区间 |",
        "1R": f"| **1R** 风险单元 | **{_format_price(str(levels['one_r']))} 元** | 以当前价 {current} 元管理已有仓位 |",
        "TP1": f"| **TP1** 止盈 1 | **{_format_price(str(levels['tp1']))} 元** (+1R) | 持仓者逢强减仓 1/3 |",
        "TP2": f"| **TP2** 止盈 2 | **{_format_price(str(levels['tp2']))} 元** (+2R) | 持仓者再减仓 1/3 |",
        "TP3": f"| **TP3** 止盈 3 | **{_format_price(str(levels['tp3']))} 元** (+3R) | 持仓者清仓 |",
        "SL_soft": f"| **SL_soft** 软止损 | **{_format_price(str(levels['sl_soft']))} 元** (−0.6R) | 减仓 50% 并复核风险 |",
        "SL_hard": f"| **SL_hard** 硬止损 | **{_format_price(str(levels['sl_hard']))} 元** (−1R) | 全部退出 |",
    }
    for label, replacement in replacements.items():
        report_body = re.sub(
            rf"(?m)^\|\s*\*\*{re.escape(label)}\*\*[^|]*\|.*$",
            replacement,
            report_body,
        )

    if summary_block:
        yaml_values = {
            "pm_tp1": levels["tp1"],
            "pm_tp2": levels["tp2"],
            "pm_tp3": levels["tp3"],
            "pm_sl_soft": levels["sl_soft"],
            "pm_sl_hard": levels["sl_hard"],
        }
        for key, value in yaml_values.items():
            summary_block = re.sub(
                rf"(?m)^(\s*{re.escape(key)}:\s*).*$",
                rf"\g<1>{value:.2f}",
                summary_block,
            )
    return report_body, summary_block


def _normalize_no_new_position_rows(
    report_body: str,
    summary_source: str,
    entry_timing: str,
) -> str:
    timing_context = {
        "等回踩": "趋势仍在，等待更好赔率",
        "等放量突破": "结构待确认，等待量价确认",
        "暂不介入": "风险条件尚未解除",
        "退出观察": "短线结构已破坏",
        "继续观察": "短线方向尚不明确",
        "数据不足": "关键数据不足",
    }.get(entry_timing, "当前不具备新建仓条件")
    stars_value = _summary_float(summary_source, "pm_conviction_stars")
    stars = max(1, min(5, int(stars_value))) if stars_value is not None else 0
    if stars:
        labels = {1: "很低", 2: "较低", 3: "中等", 4: "较高", 5: "很高"}
        report_body = re.sub(
            r"(?m)^\|\s*Conviction\s+信心\s*\|.*$",
            f"| Conviction 信心 | {'⭐' * stars} **{labels[stars]}**"
            "（长期判断与短期时机分开，当前不具备新建仓条件） |",
            report_body,
        )
    report_body = re.sub(
        r"(?m)^\|\s*结构时机\s*\|.*$",
        f"| 结构时机 | {entry_timing}（{timing_context}） |",
        report_body,
    )
    report_body = re.sub(
        r"(?m)^\|\s*当前赔率\s*\|.*$",
        "| 当前赔率 | 当前不建仓；赔率在重新评估入场条件后重算 |",
        report_body,
    )
    return report_body


def _extract_pm_summary_block(content: str) -> tuple[str | None, int, bool]:
    """Return the final PM_SUMMARY YAML, its rendered start, and fence state."""
    matches = list(re.finditer(r"(?m)^PM_SUMMARY:\s*$", content or ""))
    if not matches:
        return None, len(content or ""), False

    key_start = matches[-1].start()
    prefix = content[:key_start]
    fences = list(re.finditer(r"(?m)^```yaml\s*$", prefix))
    fenced = bool(fences and prefix.rfind("```") == fences[-1].start())
    rendered_start = fences[-1].start() if fenced else key_start
    tail = content[key_start:]
    if fenced:
        closing = re.search(r"(?m)^```\s*$", tail)
        block = tail[:closing.start()].rstrip() if closing else tail.rstrip()
    else:
        block = tail.rstrip()
    return block, rendered_start, fenced


_INTERNAL_PM_LINE = re.compile(
    r"SYS_[A-Z0-9_]+|effective_(?:action|size|entry)|工具返回|工具强制|"
    r"不变量\s*[A-Z]|PM\s*2[AB]|逐字采用|禁止改写|本 section|mark 闸门|"
    r"系统归档数据",
    re.I,
)


def _strip_internal_pm_content(content: str) -> str:
    """Remove model/audit material that is not useful in a user decision."""
    cleaned = content
    cleaned = cleaned.replace("工具强制降档", "综合风险约束使评级下调")
    cleaned = cleaned.replace("系统强制改写为", "因此当前结论为")
    cleaned = cleaned.replace("market_mode=risk_off", "市场环境偏防守")
    cleaned = cleaned.replace("data_status=stale", "风险快照陈旧")
    cleaned = cleaned.replace("as_of_date", "快照时间")
    cleaned = cleaned.replace("anchor_sensitive=true", "估值锚敏感")
    cleaned = cleaned.replace("quant_anticrowding", "反拥挤因子")
    cleaned = cleaned.replace("capital_flow_score", "资金流评分")
    cleaned = cleaned.replace("winner_rate_pct", "获利盘")
    cleaned = cleaned.replace("regime=", "状态=")
    cleaned = cleaned.replace("quant value", "量化价值因子")
    cleaned = cleaned.replace("lowvol", "低波因子")
    cleaned = cleaned.replace("ma10_slope_5d", "10 日均线 5 日斜率")
    cleaned = cleaned.replace("PEG 未 attempt 定价", "PEG 尚未充分计入")
    cleaned = cleaned.replace("短线结构 broken", "短线结构已破坏")
    cleaned = cleaned.replace("结构 broken", "结构已破坏")
    cleaned = cleaned.replace("thesis", "核心逻辑")
    cleaned = re.sub(
        r'effective_action=["\']?退出观察["\']?逐字采用[^。\n]*',
        "当前短线结构已破坏，当前不新建仓",
        cleaned,
    )
    cleaned = re.sub(
        r"(?ms)^###\s+各 Agent 核心结论一览.*?(?=^---\s*$|^##\s|\Z)",
        "",
        cleaned,
    )
    cleaned = re.sub(
        r"(?ms)^###\s+(?:\d+\.\d+\s+)?工具返回验证.*?(?=^###\s|^##\s|^---\s*$|\Z)",
        "",
        cleaned,
    )
    cleaned = re.sub(
        r"(?ms)^###\s+风控审查回应.*?(?=^###\s|^##\s|^---\s*$|\Z)",
        "",
        cleaned,
    )
    cleaned = re.sub(
        r"(?ms)^###\s+不一致性最终自检.*?(?=^###\s|^##\s|^---\s*$|\Z)",
        "",
        cleaned,
    )
    cleaned = re.sub(
        r"(?ms)^##\s+五、附录：自检与归档.*\Z",
        "",
        cleaned,
    )

    cleaned = cleaned.replace("**6 月 / 12 月**", "**6 个月 / 12 个月**")
    cleaned = cleaned.replace("6 月 / 12 月", "6 个月 / 12 个月")
    cleaned = cleaned.replace("6 月内 thesis", "6 个月内 thesis")
    cleaned = cleaned.replace("12 月内 thesis", "12 个月内 thesis")
    cleaned = cleaned.replace("6 月内", "6 个月内")
    cleaned = cleaned.replace("12 月内", "12 个月内")
    cleaned = cleaned.replace("12 月主题判断", "12 个月主题判断")
    cleaned = cleaned.replace("**6 月检查点**", "**第一检查点**")
    cleaned = cleaned.replace("**9 月检查点**", "**第二检查点**")
    cleaned = cleaned.replace("**12 月强制退出**", "**最终检查点**")
    cleaned = cleaned.replace("突破确认+回调加仓", "分批执行并预留滑点")

    lines = [line for line in cleaned.splitlines() if not _INTERNAL_PM_LINE.search(line)]
    cleaned = "\n".join(lines)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def _position_action_rows(content: str, entry_timing: str, size: str, rating: str) -> str:
    no_new_position_actions = {
        "等回踩", "等放量突破", "暂不介入", "退出观察", "继续观察", "数据不足",
    }
    if entry_timing not in no_new_position_actions:
        empty_action = f"按计划分批执行，目标新建仓位 {size}"
        holding_action = "按总仓位上限管理，不追价"
    else:
        empty_action = "不买，保持新建仓位 0%"
        levels = [
            _format_price(_extract_pm_summary_value(content, key))
            for key in ("pm_tp1", "pm_tp2", "pm_tp3")
        ]
        soft = _format_price(_extract_pm_summary_value(content, "pm_sl_soft"))
        hard = _format_price(_extract_pm_summary_value(content, "pm_sl_hard"))
        if rating in {"UNDERWEIGHT", "SELL"}:
            prefix = "优先降低风险仓位，不加仓"
        else:
            prefix = "不加仓"
        details = []
        if all(levels):
            details.append(f"反弹处理位 {' / '.join(levels)} 元")
        if soft and hard:
            details.append(f"软止损 {soft} 元，硬止损 {hard} 元")
        holding_action = prefix + ("；" + "；".join(details) if details else "")

    return (
        "## 现在怎么做\n\n"
        "| 持仓状态 | 当前建议 |\n"
        "|------|------|\n"
        f"| **空仓** | **{empty_action}** |\n"
        f"| **已持仓** | **{holding_action.split('；', 1)[0]}**"
        f"{('；' + holding_action.split('；', 1)[1]) if '；' in holding_action else ''} |"
    )


def _format_pm_decision(
    content: str,
    timing: dict,
    market_risk_snapshot: dict | None = None,
) -> str:
    """Remove model working text and prepend a deterministic action summary."""
    original = (content or "").strip()
    summary_block, summary_start, summary_was_fenced = _extract_pm_summary_block(original)
    user_source = original[:summary_start].rstrip() if summary_block else original
    trade_ticket = re.search(r"(?m)^#{1,2}\s+Trade Ticket\b.*$", user_source)
    if trade_ticket:
        report_body = user_source[trade_ticket.start():].strip()
    else:
        report_body = user_source
        logger.warning("PM 输出未找到 Trade Ticket 标题，保留原文并仅添加操作摘要")

    entry_timing = timing.get("effective_action") or "数据不足"
    if entry_timing not in _ENTRY_PRESENTATION:
        entry_timing = "数据不足"
    reason, trigger = _ENTRY_PRESENTATION[entry_timing]
    rating = _extract_pm_summary_value(original, "pm_rating") or "数据不足"
    action = _extract_pm_summary_value(original, "pm_action_keyword") or "数据不足"
    no_new_position_actions = {
        "等回踩", "等放量突破", "暂不介入", "退出观察", "继续观察", "数据不足",
    }
    if entry_timing in no_new_position_actions:
        report_body = re.sub(
            r"(?m)^(\|\s*\*\*Size\*\*\s*仓位规模\s*\|).*$",
            r"\1 新建仓 0% |",
            report_body,
        )
        report_body = re.sub(r"(?m)^(\s*pm_size_(?:low|high)_pct:\s*).*$", r"\g<1>0", report_body)
        report_body = re.sub(r"(?m)^(\s*pm_entry_(?:low|high):\s*).*$", r"\g<1>null", report_body)
        report_body = re.sub(
            r"(?m)^入场条件：.*$",
            "重新评估条件：结构修复并重新通过风险门控后再评估；当前不新开仓。",
            report_body,
        )
        if summary_block:
            summary_block = re.sub(
                r"(?m)^(\s*pm_size_(?:low|high)_pct:\s*).*$", r"\g<1>0", summary_block,
            )
            summary_block = re.sub(
                r"(?m)^(\s*pm_entry_(?:low|high):\s*).*$", r"\g<1>null", summary_block,
            )
    if (market_risk_snapshot or {}).get("data_status") == "stale":
        checkpoint = (market_risk_snapshot or {}).get("required_checkpoint") or "最新盘中"
        report_body = re.sub(
            r"(?m)^\|\s*未来 3 个交易日趋势\s*\|.*$",
            f"| 未来 3 个交易日趋势 | **数据不足**（盘中风险快照陈旧，需 {checkpoint} 检查点） |",
            report_body,
        )
        report_body = re.sub(
            r"(?ms)^###\s+(?:1\.3\s+)?未来\s*3\s*个交易日趋势[^\n]*\n.*?(?=^###\s|^##\s|\Z)",
            f"### 未来 3 个交易日趋势\n\n"
            f"**数据不足**（盘中风险快照陈旧，需 {checkpoint} 检查点；当前仅执行 WAIT、0%）。\n\n",
            report_body,
            count=1,
        )
        report_body = re.sub(
            r"(?m)^(\s*short_term_trend:\s*).*$", r"\g<1>数据不足", report_body,
        )
        if summary_block:
            summary_block = re.sub(
                r"(?m)^(\s*short_term_trend:\s*).*$", r"\g<1>数据不足", summary_block,
            )
    size = "0%" if entry_timing in no_new_position_actions else _format_position_size(
        _extract_pm_summary_value(original, "pm_size_low_pct"),
        _extract_pm_summary_value(original, "pm_size_high_pct"),
    )

    if entry_timing in no_new_position_actions:
        report_body, summary_block = _enforce_holder_r_levels(
            report_body, summary_block, original,
        )
        report_body = _normalize_no_new_position_rows(
            report_body, summary_block or original, entry_timing,
        )

    report_body = _strip_internal_pm_content(report_body)
    position_actions = _position_action_rows(
        summary_block or original, entry_timing, size, rating,
    )

    summary = (
        f"# 短期操作结论：{entry_timing}\n\n"
        f"> **当前动作：{action}｜新建仓位：{size}｜长期评级：{rating}**\n>\n"
        f"> **核心原因：{reason}**\n>\n"
        f"> **重新评估条件：{trigger}**\n\n"
        f"{position_actions}\n\n---"
    )
    rendered = f"{summary}\n\n{report_body}".rstrip()
    if summary_block:
        archive = f"```yaml\n{summary_block}\n```" if summary_was_fenced else summary_block
        rendered += f"\n\n---\n\n{archive}"
    return rendered.rstrip() + "\n"


def _pm_tool_loop(llm_with_tools, initial_messages):
    """PM 工具调用循环——复用 RM 的共享循环（含空输出/截断续写兜底）。

    完成标记 PM_SUMMARY：prompt 强制的收尾 YAML，缺了说明正文被 think-only
    剥空或中途截断（603629 事故：decision.md 缺失，5_portfolio 整段跳过）。
    """
    return _run_tool_calling_loop(
        llm_with_tools, initial_messages,
        tools_by_name=PM_TOOLS_BY_NAME, role="PM",
        completion_token="PM_SUMMARY", max_iterations=_MAX_TOOL_ITERATIONS,
    )


def create_portfolio_manager(llm, memory):
    def portfolio_manager_node(state) -> dict:

        instrument_context = build_instrument_context(state["company_of_interest"], state.get("company_name", ""))

        history = state["risk_debate_state"]["history"]
        risk_debate_state = state["risk_debate_state"]
        research_plan = state["investment_plan"]

        # PM 直接读 4 个 analyst 原始报告 + consensus + RM thesis + quant_score
        market_report = state.get("market_report", "")
        sentiment_report = state.get("sentiment_report", "")
        news_report = state.get("news_report", "")
        fundamentals_report = state.get("fundamentals_report", "")
        consensus_snapshot = state.get("consensus_snapshot", "")
        stock_profile = state.get("stock_profile", "")
        quant_score = state.get("quant_score", "")
        sector_comparison = state.get("sector_comparison", "")
        market_risk_snapshot = state.get("market_risk_snapshot") or {}
        market_mode = derive_market_mode(market_risk_snapshot)
        rm_rating = _extract_rm_rating(research_plan)
        entry_timing = _derive_entry_timing_from_profile(
            stock_profile, market_mode, long_term_rating=rm_rating,
        )
        market_risk_block = (
            json.dumps(market_risk_snapshot, ensure_ascii=False, indent=2)
            if market_risk_snapshot
            else "未找到当日市场风险快照：不得假设低风险；短期动作只能为 WAIT，仓位为 0%。"
        )

        # 决策卡头部信息
        pm_ticker = state["company_of_interest"]
        pm_company_name = state.get("company_name", "")
        pm_trade_date = state.get("trade_date", "")

        # 基于 RM 方案 + analyst 综合做 memory 检索
        curr_situation = f"{research_plan}\n\n{fundamentals_report}\n\n{market_report}"
        past_memories = memory.get_memories(curr_situation, n_matches=3)

        past_memory_str = ""
        for i, rec in enumerate(past_memories, 1):
            past_memory_str += f"【适用场景】{rec['matched_situation']}\n【经验教训】{rec['recommendation']}\n\n"

        prompt = fr"""【语言要求】你必须使用中文撰写以下所有分析内容和回复。评级关键词（Buy/Overweight/Hold/Underweight/Sell）、股票代码、专业交易术语（Action/Size/R/TP/SL/Time Stop）可保留英文，但需带中文注释。

你是**投资组合经理（Portfolio Manager）**，对标头部对冲基金 PM 角色，输出**专业交易票（Trade Ticket）**风格的决策。

## ⚠️ 数值计算必须调用工具（强制约束）

你已绑定 4 个计算工具。**以下数值必须通过工具调用完成，禁止心算**：

| 计算场景 | 必须调用的工具 |
|---------|-------------|
| 1R / TP1-3 / SL_soft / SL_hard 完整 R-multiple 价位体系 | `compute_r_multiple_levels`（输入 Entry + SL_hard）|
| Conviction 五星 + 仓位上限（基于 RM Conviction + 赔率 R，\|d\| 仅微调）| `compute_conviction_position_map` |
| 4 情景概率加权 E（含黑天鹅档）| `compute_pm_scenario_e` |
| 市场风险闸门后的实际动作与仓位 | `apply_market_risk_gate` |

**规则**：
- 决定 Entry 和 SL_hard 后**必须**调 `compute_r_multiple_levels` 算 TP1/TP2/TP3/SL_soft，**禁止心算"+1R/+2R/+3R"**
- 决定 Conviction 时**必须**调 `compute_conviction_position_map`，仓位严格采用工具返回的区间
- 输出 4 情景表后**必须**调 `compute_pm_scenario_e` 算加权 E，**禁止自己加权**
- 决定原始 Action/Size 后**必须**调 `apply_market_risk_gate`；Trade Ticket 与 PM_SUMMARY 只能使用工具返回的 effective 值

{instrument_context}

## 当日 Market Risk Officer 快照（硬约束）

```json
{market_risk_block}
```

该快照不改变 RM 的长期评级，但严格约束短期动作：`WAIT` 时禁止 BUY_NOW/追高；
`CONDITIONAL` 时 BUY_NOW 必须改为条件入场；建议仓位不得超过 `position_cap_pct`。
快照缺失等同于 WAIT + 0% 仓位。必须在决策卡与入场时机中引用 as_of_date/as_of_time。

## 确定性短线结构与入场时机（权威约束）

```json
{json.dumps(entry_timing, ensure_ascii=False, indent=2)}
```

该结果由画像 `SYS_SHORT_TERM_STRUCTURE`、RM 长期评级和 `market_risk_daily` 共同确定。
**短线结构不得改变长期评级**，只决定当前时机。Trade Ticket 的“结构时机”必须逐字采用
`effective_action`；原有 `apply_market_risk_gate` 继续决定最终 Action/Size，若两者不一致，采用更保守者。
禁止把“等回踩/等放量突破/暂不介入”改写成 BUY NOW。

---

## 你的角色边界（必读）

你是**最终决策官**，把 Research Manager 的 thesis 转化为**机构级 trade ticket**。

**核心产出**（按以下顺序）：
1. **Trade Ticket 决策卡**（交易票风格，机构标准格式）
2. 该不该投资？YES / NO / CONDITIONAL（投资判断段）
3. 现阶段该不该买？BUY NOW / WAIT / DON'T BUY（入场时机段）
4. 完整操作动作表（含 R-multiple、TP1/TP2/TP3、SL、Time Stop）
5. 情景概率分布表（含黑天鹅尾部）
6. 反向证伪触发器（What would change my mind）
7. 核心风险、触发条件 + 必要时的减仓资金去向（机会成本）

**你比 RM 多的信息**：4 个 analyst 原始报告 + consensus + risk 三方辩论。
**你不该做的**：重新做方向判断（评级以 RM 为准，仅 ±1 档微调）、重新算 PE/EPS。

## 用户版报告可读性（最高优先级）

用户版报告不得输出以下内容：工具调用过程、工具返回字段、prompt 合规说明、
`SYS_*`/`effective_*` 内部变量、PM 2A/2B、不变量检查、历史教训自检表、
价位一致性自检表、评级链路自检、归档附录。上述检查只在内部完成，结论融入
核心理由、风险和操作动作。不得使用“逐字采用”“禁止改写”“工具强制降档”
等面向开发者的措辞。

只输出 Trade Ticket + 四个一级标题：投资决策与入场时机、操作计划、情景概率与赔率、
风险触发与监控。正文控制在可扫描的机构交易票长度，避免重复同一数据和 thesis。
`PM_SUMMARY` 作为最后一个 YAML 代码块保留，之前不增加归档章节。

---

## 决策流程（仅内部思考使用，**禁止把流程章节写入最终报告**）

⚠️ **关键格式约束**：以下"第一步～第八步"是你**内部思考流程**，不是报告章节。**最终 decision.md 必须直接以 `## Trade Ticket 决策卡` 开头**，禁止出现"第一步：吸收股票画像"、"第二步：评级微调" 等流程性标题。所有思考结果直接体现在"完整报告结构"列出的十六个正式章节里。

### 第一步：吸收股票画像 + 上下文（内部思考，不输出章节）

从画像识别官的输出（在"输入资料"区"股票画像"段）提取并显式列出：
- **决策风格**（value_anchor / catalyst_driven / momentum / event_driven）
- **4 份报告最终权重**（用于校验 RM 评分是否合理）
- **关键时间窗口事件**

**决策风格→操作动作的映射规则**（必须严格遵守）：

| 决策风格 | Time Stop | Entry 节奏 | TP/SL 节奏 | 监控指标侧重 |
|---------|-----------|----------|----------|------------|
| **value_anchor 价值锚定** | 12-18 月 | 等 PE 跌至历史中位数附近建仓 | 宽 TP（≥3R），宽 SL（≥1R），不追求精确 | 季度财报、ROE、毛利率趋势 |
| **catalyst_driven 催化驱动** | 6-9 月 | 等关键催化前 2 周建仓 | TP 1R/2R 阶梯（催化兑现阶段性减仓） | 催化进度、行业事件、机构持仓 |
| **momentum 动量** | 1-3 月 | 突破/回踩均线建仓 | 紧 TP（1R 立刻减半），紧 SL（0.7R） | RSI、MACD、成交量、舆情拥挤度 |
| **event_driven 事件驱动** | 至事件结束（1-2 月）| 事件前 1 周内 | 事件后立即清仓（无视价位） | 事件日历、政策细则、公告 |

**从 RM thesis 中提取**（RM 8 步 COT 综合判断产出）：

- **最终评级 R + Conviction**（RM 一、评级与置信度）→ 默认采纳，仅 ±1 档微调；Conviction 映射五星 + 仓位
- **综合目标价区间 + Bull/Base/Bear 目标价 + 概率**（RM 一 / Step 5）→ 直接复用为情景分布三档，1R 基于综合区间
- **业绩拐点 + 下一检验点**（RM Step 3）→ Time Stop 触发条件
- **行业框架 + 决策风格**（RM Step 1 + stock_profile）→ 操作节奏（紧/松 TP/SL）
- **风险清单**（RM 六）→ 映射到风控辩论缓释表
- **多空辩论 Bull/Bear/d**（RM 辅助分析）→ 仅作 Conviction 参考，不影响方向

**从 quant_score 提取**（Python 确定性输出，独立第二眼）：

| 字段 | quant_score 输出位置 | 你的用途 |
|------|--------------------|---------|
| **QUANT_SCORE.composite**（0-100） | YAML 摘要 | 评级一致性交叉校验：若 RM 评级方向与 quant 严重背离（如 RM=OVERWEIGHT 但 composite<30），需在 2B 评级微调中说明 |
| **factor_scores 中 <30 分的因子** | YAML 摘要 / 因子分项表 | **必须列入 Trade Ticket 的 Key Risks 段**（如 lowvol=5 → "极端高波动"；value=18 → "估值显著偏贵"）|
| **Conviction 强化**（评级与 quant 方向一致时）| —— | RM=OVERWEIGHT 且 composite≥70 → Conviction 可在 RM 给的基础上 +1 档 |

⚠️ **强制约束**：本节列出的薄弱因子（<30）**必须**出现在 Trade Ticket 的"Key Risks"段，禁止以"已在 RM 风险清单覆盖"为由跳过——这是 Python 量化锚，是独立信号来源。

**从 sector_comparison 提取**（板块对照官的 Python 确定性输出）：

| 字段 | 你的用途 |
|------|---------|
| **fallback 匹配路径**（层级 1→2→3→4）| 判断对照集可靠度。命中"层级 1 主题"最强；降到"层级 4 大盘兜底"则只能粗略对比 |
| **本股 vs 主题 ETF 的 30d RS** | Trade Ticket "投资判断" / "入场判断" **必须引用一句** |
| **主题内 30d 收益排名** | Trade Ticket 决策时引用——若排名靠后则信号弱化（同主题更好选择） |
| **本股 vs 大盘指数 30d RS** | 用于"宏观背景下本股是否抗跌" 判断 |

⚠️ **强制约束**：Trade Ticket 的 "投资判断" 或 "入场判断" 字段**必须含一句板块 RS 引用**。
例：
- "板块 RS 30d +12% 跑赢大盘 + 主题内排名第 2/5，板块β 仍正向，CONDITIONAL（等回调）"
- "板块 RS 30d -8% 跑输 + 主题内倒数 → 板块走弱强化卖出信号，DON'T BUY"

**从 CAPITAL_FLOW YAML 提取（资金流官的 Python 确定性输出，market_report 内）——填"资金面快照"行**：

| 字段 | 用途 |
|------|------|
| `主力净流入(5日)` / `主力净流入(20日)` | 主力近期方向与力度（亿元） |
| `capital_flow_score`（0-100）/ `capital_flow_regime` / `ddx_like_5d_pct_1y` | 综合资金面强弱、定性、主力强度 1 年分位 |
| `net_inflow_streak_days` | 主力连续净流入(+)/净流出(−)天数 |
| **`holder_num_qoq_pct`（股东户数环比%）** + **`holder_num_latest_report_date`（截止日）** | **筹码迁徙——散户 vs 机构最有体感的信号**：环比**↑**=户数增加=机构把筹码派给散户=**派发/顶部信号**；环比**↓**=户数减少=筹码向机构集中=**吸筹**。⚠️ **必须带上截止日**（季度数据有滞后，区别于主力资金的日频）：写成"股东户数环比 +X%（截至 YYYY-MM-DD 季报）"。缺失则写"股东户数数据缺失"，**不要拿别的指标硬凑** |
| **`winner_rate_pct`（获利盘%，cyq 日频可靠）** + `winner_rate_chg_5d` | **散户套牢度——首选**：≤50%=多数套牢（高位接盘被埋/抛压小）；≥85%=普遍获利（止盈派发压力/顶部）。价涨但获利盘5日暴跌=新买盘高位被套 |
| `retail_concentration_signal` | 散户接盘信号：`散户高接盘`=主力派发 **且** 获利盘低位/散户高位承接（看空增强）；`中性`=否（已升级筹码口径优先） |

⚠️ **散户口径排雷（强制）**：`retail_buy_amount_rate_5d_pct`（毛买盘占比）**当前数据口径不可靠**（实测稳定 6-10%，远低于 A 股应有的 50-70%）、`retail_net_inflow_rate_5d_pct`（净流入占比）**几乎恒为 null**——**这两个一律不要写进快照、更禁止据其编"散户参与度极低/异常"之类的解读**（是坏数据不是真信号）。散户侧用 **获利盘比例(winner_rate)** + **股东户数环比** + **散户接盘信号** 表达。

⚠️ **强制约束**：Trade Ticket"关键背景"的 **资金面快照（主力 vs 散户）行必填**，必须给出 ① **主力**近 5/20 日净流入 + DDX 分位；② **筹码迁徙**（股东户数环比，缺失则注明）；③ **散户接盘信号**；并点明**当前主导方**（如"主力连续净流出 7 日、股东户数环比 +15% 筹码向散户分散 → 主力派发中"）。某项数据缺失写"缺失/不可用"，不得留空、不得用坏口径补位。

**⚠️ stock_profile TRANSPARENCY 段（Layer 3 标注）必读**：

stock_profile 末尾的 `TRANSPARENCY:` 段标注了"超共识程度"，按以下规则用于 **Conviction 五星调档**：

| 触发条件 | Conviction 调档 |
|---------|----------------|
| `target_pe_high_vs_sell_side_pct` > +50 且 `premium_divergence_reason` 无 ≥2 条产业证据 | -1 档 |
| `target_pe_high_vs_sell_side_pct` > +100 | 强制 ≤ 3★ Medium |
| `theme_stage_llm_chosen` ≠ `theme_stage_inferred_by_data` 且 `theme_divergence_reason` 不充分 | -1 档 |
| `premium_llm_chosen` > `premium_default_template` + 30 | -1 档 |
| `peer_anchor_single_comp` = true | -1 档（兄弟股可比仅 1 家，单标的低置信）|
| 三源 PE 全部 null | 强制 ≤ 2★ Low |

**Trade Ticket Key Risks 段**：若 TRANSPARENCY 任一字段触发降档，必须在 Key Risks 写入"超共识溢价风险（vs 卖方/同业偏离 N%）"作为独立一条。

**核心理念（机构对照）**：跟 IC 复议要求"超共识 target 必须 defend 产业证据"完全一致。这里不强制改评级方向，只通过 Conviction 调档间接压仓位——机构 PM 内部 risk dashboard 标准做法。

⛔ **内部应用要求**：按上述阈值完成 Conviction 调档，但用户版报告只写成
“估值锚偏离导致置信度下降”及对应结果，不输出 `TRANSPARENCY.*` 字段名、阈值检查过程或开发者留痕。

### 第二步：评级微调（含对 RM thesis 的反向质疑）

#### 2A. 对 RM thesis 的反向质疑（内部完成，不输出独立表）

模拟真实投研团队中 PM 对 RM 的双向沟通：PM 是 thesis 的"第一个怀疑者"，要找出 RM 论证里**最薄弱的 1-2 个假设**，并显式回答"如果这些假设不成立，会怎样"。

**内部检查格式**（只用于推理，结论融入 Core Thesis / Key Risks）：

| # | RM thesis 中的薄弱假设 | 假设来源 | 假设不成立的概率 | 假设不成立后的影响 |
|---|---------------------|---------|---------------|------------------|
| 1 | <一句话描述被质疑的假设> | RM 第 X 步 / Bull/Bear 论据第 Y 条 | 低 / 中 / 高 | 评级会变成 __ / 目标价会降到 __ |
| 2 | <同上> | <同上> | <同上> | <同上> |

**质疑要求**：
- 必须聚焦"假设"层面，不是数据错误（数据错误应在 RM 评分时已被 Hard Data 校验剔除）
- 必须可证伪——质疑的假设应该有明确的"何时验证、用什么数据验证"
- 至少 1 条质疑必须针对 anchor 论据（RM 评分表中得分×权重最高的多空各 1 条）
- 如果你认为 RM 的所有关键假设都很扎实，写一句"无显著质疑：RM 的 anchor 论据 X 和 Y 都有 hard data 支撑，假设链条无明显漏洞"

**质疑示例**：

> | 1 | RM 假设 Q2 营收同比 >25% 是 Base case | 第 5 步 Bull case 3 | **中** | 若 Q2 仅 +15%，Base case 概率从 55% 降到 30%，加权 E 从 +12% 转为 -3%，评级实际应降至 HOLD |
> | 2 | RM 用三星 CXL 量产 Q3 落地作为 anchor 1 | Bull 论据 1 | **中** | 若三星推迟到 Q4，anchor 失效，d' 跨档至 HOLD，目标价应砍 20% 至 220 元 |

#### 2B. 评级微调（仅限执行层因素）

1. 默认采纳 RM 评级 R2
2. 仅允许 ±1 档微调，禁止跨方向翻转
3. 调整必须留痕（写明触发理由）

**质疑（2A）→ Conviction 修正**：
- 若 2A 中有 ≥1 条"假设不成立概率 = 高"的质疑 → Conviction 下调一档（仓位对应下移）
- 若 2A 中有 ≥2 条"假设不成立概率 = 中"的质疑 → Conviction 下调一档
- 质疑**不能**修改评级方向（仍按 2B 规则采纳 RM ± 1 档）
- 质疑结论写入最终决策卡的"评级调整说明"段留痕

### 第三步：Conviction Score 五星制 + 仓位映射

**主输入 = RM Conviction（高/中/低）+ 赔率 R**；\|d\|（辩论比分差）只作减分微调。
为什么：仓位是执行端最重要的参数，RM Conviction 由数据完整度/估值收敛度/拐点确认度等硬条件校准，是证据质量的代理；辩论比分是 LLM 打分的软信号，且本系统规定"辩论不影响方向只影响置信"——让它主导仓位自相矛盾（对标真实 PM：仓位 = 信念强度 × 赔率 × 风险预算）。

**强制**：调 `compute_conviction_position_map`，输入 `rm_conviction`（照抄 RM thesis 的 高/中/低）+ `odds_r`（赔率 R）+ `abs_d`（RM 辅助分析的 \|d\|，如实填）+ `anchor_sensitive`，星级与仓位区间**严格采用工具返回值**。

工具内部规则（你不算，只需理解）：
- 基础星：RM 高 → 4★ / 中 → 3★ / 低 → 2★
- 赔率：R ≥ 2.0 → +1★；R < 1.0 → −1★
- \|d\| < 0.5（多空胶着）→ −1★（辩论仅减分，不加分）
- 5★ 门槛：RM=高 且 R≥2.0 且 anchor 不敏感，缺一封 4★
- 仓位上限：5★ 15-20% / 4★ 8-12% / 3★ 4-6% / 2★ 2-3% / 1★ ≤1%（试探仓或观望）

**风控修正**：若风控辩论任一维度"高风险"未缓释，Conviction 可下调一档，仓位对应下移。

### 第四步：R-multiple 设计（核心：风险单元化）

R-multiple 是头部 PM 报告的核心工具，让止盈止损天然对称、自动校验赔率。

**定义**：
- **1R = 建仓价 − 硬止损价**（每股承担的最大风险）
- 例：建仓 240 元、硬止损 215 元 → 1R = 25 元
- TP1 = 建仓价 + 1R（赚 1R 减 1/3 仓位）
- TP2 = 建仓价 + 2R（赚 2R 再减 1/3）
- TP3 = 建仓价 + 3R（赚 3R 清仓）

**自校验**：如果 RM 的上行目标 P_up < TP1，说明赔率假设不成立，必须重新审视。

**评级↔R 用法映射**：
- BUY/OVERWEIGHT：用 R-multiple 表达建仓/止盈/止损
- HOLD：建仓段空着，止盈/止损用 R-multiple 表达（针对已持有者）
- UNDERWEIGHT/SELL：用 R 反向表达——R 用作减仓节奏（每跌 1R 减 X%）

### 第五步：Time Stop（时间止损）

价位止损解决"如果错了"，时间止损解决"如果僵尸"。

**强制设置**：
- **6 个月检查点**：若 thesis 核心进展无任何兑现（具体里程碑见 RM 证伪触发器反向），持仓减半
- **12 个月强制退出**：thesis 全无进展则清仓（除非有新证据延长 thesis 有效期）

PM 必须明确"thesis 兑现"的具体里程碑（如"Q2 营收增速 >25%"、"CXL 量产订单 >X 亿"），可观测可证伪。

⛔ **里程碑必须直读催化日历，不许凭空编日期**：新闻报告末尾若有 `SYS_CATALYST_CALENDAR:` 段（Python 从新闻 thesis 相关事件抽取，带真实日期+方向），**Time Stop 检查点和监控里程碑必须以它为准**——把日历里 thesis 相关度=核心 的事件（如"2026-07-15 中报""2026Q4 DDR5量产"）作为验证节点，方向(+/-)对应 thesis 兑现/受损。日历里的事件是真实新闻提取的，比你自己想的"下一验证点"可靠。无该段时才退回自行判断。

⚠️ **叙事切换早期预警（领先观察项）**：新闻报告末尾若有 `SYS_NARRATIVE:` 段（Python 比对舆情水位 vs 7 日动能 / 新闻论调的背离），**必须在监控段单列一行早期预警**——`见顶回落预警`=人群还偏多但动能/新闻论调已先转空（顶部领先信号，收紧止盈、降新进仓位节奏）；`筑底回升预警`=人群还偏空但动能/论调已先转多（左侧建仓观察）。⛔这是**领先观察项不是评级信号**：不据此改 RM 评级，只作为监控触发器——预警方向与持仓方向相反时，把它列为最优先复核项。无该段=无背离，不提。

### 第六步：情景概率分布表（含尾部）

**核心规则（条件性继承）**：

| 你在第二步 2B 的决策 | 第六步该怎么做 |
|------|---------|
| **采纳 RM 评级（不微调）**| Bull/Base/Bear 三档**完全沿用 RM Step 5**（目标价 + 概率 + 核心假设原文照抄），你只新增 **黑天鹅 Tail** 一档 |
| **微调 RM 评级（±1 档，如 HOLD → OVERWEIGHT）**| 允许调整 Bull/Base/Bear 概率（**单档调整不超过 ±15pp**），但**目标价仍沿用 RM**；必须显式输出"RM 原始 vs PM 调整"对照表，并归因到第二步反向质疑的具体条目 |

**为什么目标价必须沿用 RM**：目标价是估值模型（PE×EPS / PEG / 同业可比）的输出。如果你认为目标价错，应该回到第二步质疑 RM Step 4 的估值方法，**而不是在这里悄悄改数字**。概率是对未来路径的主观判断，PM 在这上面有合理裁量空间。

#### 情形 A：评级未微调（直接沿用 RM）

| 情景 | 概率 | 12 月目标价 | 收益 | 触发条件 |
|------|------|------------|------|---------|
| 乐观 Bullish | **（沿用 RM）**| **（沿用 RM）**| __% | （沿用 RM）|
| 基础 Base | **（沿用 RM）**| **（沿用 RM）**| __% | （沿用 RM）|
| 悲观 Bearish | **（沿用 RM）**| **（沿用 RM）**| __% | （沿用 RM）|
| 黑天鹅 Tail | __%（一般 5-15%）| __ 元 | __% | __（必须来自尾部风险分析师辩论的极端情形）|
| **概率加权 E** | 100% | __ 元 | __% | — |

加入 Tail 后 Bull/Base/Bear 原始概率之和需从 100% 等比例收缩到 (100% − tail%)。例：RM 给 25/50/25，加 10% Tail，三档变 22.5/45/22.5。

#### 情形 B：评级微调（必须输出对照表）

| 情景 | RM 原始概率 | PM 调整后概率 | 12 月目标价（沿用 RM）| 调整归因 |
|------|------------|--------------|---------------------|---------|
| 乐观 Bullish | __% | __% | （沿用 RM）| 引用第二步反向质疑第 N 条：__ |
| 基础 Base | __% | __% | （沿用 RM）| __ |
| 悲观 Bearish | __% | __% | （沿用 RM）| __ |
| 黑天鹅 Tail | — | __% | __ 元 | （PM 新增）|
| **概率加权 E** | 100% | 100% | __ 元 | — |

**约束（情形 B 专属）**：
- 单档概率调整**不能超过 ±15pp**（如 Bear 从 25% 降到 10% 不允许，最低只能到 10%）
- 调整方向必须与微调方向一致：升档（HOLD→OVERWEIGHT）只能 Bull 加 / Bear 减；降档反之
- 每条调整必须**显式引用**第二步反向质疑的对应条目编号（如"第二步质疑 #2 指出 Q2 营收兑现概率被 RM 低估 → Bull 概率从 25% 升至 35%"）
- 若你在第二步没有给出对应质疑就修改概率 → 视为静默改写，禁止

#### 共同约束（情形 A / B 都适用）

- 黑天鹅档**必须**显式引用尾部风险分析师的论据
- 概率加权 E **必须**通过工具 `compute_pm_scenario_e` 计算（输入 4 档目标价 + 概率），禁止心算
- 若工具计算结果与 RM 的 3 档 E 偏差超 8pct，需要说明黑天鹅档的贡献
- 所有概率必须标注为“主观情景权重，未经历史校准”；不得引用无来源历史胜率、复合概率或精确事件概率增强权威感

### 第七步：Sell Trigger 对称详细（BUY/OVERWEIGHT 也必须有）

不只是降档触发器，还要四维度退出信号：

| 维度 | 触发条件 | 退出动作 |
|------|---------|---------|
| **基本面** | 例：毛利率连续 2 季 <60% / 扣非增速 <15% / 大客户流失公告 | 减仓 50% |
| **估值** | 例：动态 PE > 历史 95 分位 | 减仓 30% |
| **技术** | 例：跌破 50 日均线 + 成交放量 >均量 2 倍 | 减仓 30% |
| **情绪** | 例：共识从偏多翻为偏空 + 舆情多头占比 <30% | 减仓 20% |

### 第八步：反向证伪（What would change my mind）

防止 anchoring bias。如果你当前是 UNDERWEIGHT，**什么条件能让你翻为 BUY/OVERWEIGHT**？

⚠️ **核心约束**：**反向证伪只描述触发条件，禁止给具体价位区间**——否则会与减仓表的清仓位冲突（同一价位既要清仓又要建仓）。评级翻转后报告会重新生成，新 Entry/TP/SL 届时重算。

#### 输出格式（强制：只写触发条件 + 时间窗口 + 翻转后的评级方向）

| # | 反向证伪触发条件（可观测，**禁止给具体价位**）| 时间窗口 | 翻转后评级 |
|---|---------------------------------------------|---------|----------|
| 1 | 例：Q2 营收同比 >35% + 净利率 >50% + 公告 CXL 量产订单 | 2026 年 Q2 财报 | BUY |
| 2 | 例：技术面经过 ≥5 个交易日企稳 + 日线连续 2 日收阳 + 成交量缩至 20 日均量 60% 以下 + RSI 跌至 30 分位以下 | 任意时点（技术反弹）| OVERWEIGHT |

**末尾必须加一行**：
> ⚠️ 评级翻转后报告将重新生成，综合估值区间会重新计算，**届时给出新的 Entry/TP/SL 价位，不在本报告里预定**。

⛔ 触发条件只写信号特征（如 "RSI<30 + 日线企稳 + 量缩"）和业绩门槛（如 "Q2 营收 >X%"），不写"回调至 220-230 建仓"这种具体价位。

---

## Trade Ticket 决策卡格式（严格按以下输出）

> **格式规则（强制）**：报告中任何出现在 markdown 表格单元格内的绝对值竖线（如 \|d\|、\|R\|）一律转义为 `\|`，**禁止裸 `|`**（会被当成列分隔符破坏表格）和 **`&#124;`**（部分渲染器显示为字面字符）。

```markdown
## Trade Ticket 交易票

> **{pm_ticker} {pm_company_name}** | 决策日期 {pm_trade_date}

### 顶部导航（At-a-glance）

| 字段 | 内容 |
|------|------|
| Rating 评级 | <BUY / OVERWEIGHT / HOLD / UNDERWEIGHT / SELL> |
| Conviction 信心 | <⭐⭐⭐ Medium>（RM=高/中/低，R=X.XX，\|d\|=X.XX） |
| 投资判断 | <YES / NO / CONDITIONAL>（含等待条件） |
| 入场判断 | <BUY NOW / WAIT / DON'T BUY>（含等待条件） |
| 结构时机 | <逐字填写上方确定性 entry_timing.effective_action>；结构=<structure_class> |
| Market Risk 市场风险 | <风险等级> / <T+1 偏向>，快照生效 <as_of_date as_of_time> |
| 未来 3 个交易日趋势 | <上行 / 横盘修复 / 下行>（置信度：高/中/低；领先风险信号：__；已触发信号：__；失效条件：__） |
| 12 个月主题判断 | <扩张 / 兑现 / 降速 / 破裂>（增长兑现/估值溢价/拥挤度：__） |

### 核心交易参数（Trade Parameters）

| 参数 | 数值 | 中文说明 |
|------|------|---------|
| **Action** 操作 | BUY NOW / WAIT @<价位> / REDUCE / EXIT | 当前应执行的具体动作 |
| **Size** 仓位规模 | X-Y% | 仓位区间（来自 Conviction 表）；**SELL/UNDERWEIGHT 或入场=DON'T BUY 时，新建仓 Size 必须填 0%**，Conviction 区间挪到括注"（若已持仓，减仓上限参考 X-Y%）" |
| **Entry** 入场区间 | A-B 元 | 建仓价位区间（HOLD/SELL 填"—"） |
| **1R** 风险单元 | X.XX 元 | 1R = Entry − SL_hard |
| **TP1** 止盈 1 | <价位> 元 (+1R) | 减仓 1/3（**看空票改标"持仓者逢强减仓位 1"**）|
| **TP2** 止盈 2 | <价位> 元 (+2R) | 再减 1/3（同上）|
| **TP3** 止盈 3 | <价位> 元 (+3R) | 清仓（同上）|
| **SL_soft** 软止损 | <价位> 元 (−0.6R) | 减仓 50%，预警 |
| **SL_hard** 硬止损 | <价位> 元 (−1R) | 全部清仓 |
| **Time Stop** 时间止损 | 6 个月 / 12 个月 | 6 个月内 thesis 无进展减半，12 个月内全无进展清仓；若有真实催化日期则直接写日期 |
| **Horizon** 时间窗 | 短期(1-3 月) / 中期(3-12 月) / 长期(>12 月) | 持有目标周期 |

### 关键背景

| 字段 | 内容 |
|------|------|
| 目标价区间 | P_dn <价位> ↔ P_up <价位> |
| 当前赔率 | R = U/D = X.XX |
| 概率加权期望收益 | E = X.XX% |
| **资金面快照（主力 vs 散户）** | **主力**：近5日净流入 __亿/近20日 __亿（capital_flow_score __/100，regime __，DDX 1年分位 __）；**筹码迁徙**：股东户数环比 __%（**截至 __ 季报**，填 `holder_num_latest_report_date`；↑=机构派发给散户/顶部，↓=筹码集中/吸筹；缺失则写"股东户数数据缺失"）；**散户接盘**：__（散户高接盘/中性）；近 __ 日主力连续净流入/流出 → 主导方为 **主力/散户** |
| Core Thesis 核心逻辑 | 1. __ 2. __ 3. __（每条 ≤30 字，单行编号）|
| Key Risks 核心风险 | 1. __ 2. __ 3. __（每条 ≤30 字，单行编号）|
| 市场闸门 | <OPEN / CONDITIONAL / WAIT>；新增仓位上限 <X>%；必须为 `apply_market_risk_gate` 的工具返回值 |

### 热门概念归属（这股属于哪个最热的板块/概念，占主营多少）

| 概念/板块 | 相关度 | 占主营营收% | 当前热度 |
|------|------|------|------|
| <最热的概念，如 CPO/磷化铟/PCB/MLCC/氮化硅> | 核心/相关/边缘 | ~X%（对应产品线） | 🔥高/中/低 |
| <次相关概念> | … | ~X% | … |

---
```

**"热门概念归属"填写约束**：
- **占主营营收% 必须直读 `SYS_MAIN_BUSINESS:` 行**（Python 从 fina_mainbz 确定性算的产品营收占比，**画像识别官已确定性转录此行，优先读画像里的**；基本面报告里的同名内容次之），**禁止自己推断占比**。把相关产品线归并到概念时，% 为对应产品线占比之和，并在括号注明是哪些产品线（如"磷化铟/化合物半导体 ~13%（化合物半导体材料）"）。
- **概念归属**：把产品线映射到市场在炒的热门概念（CPO/PCB/MLCC/磷化铟/氮化硅/HBM/算力租赁/光刻胶 等）；一个产品线可对应一个概念，相关产品线可合并。
- **当前热度**：由新闻/舆情/板块报告判（news 的 SYS_CATALYST、sentiment 的 net_sentiment、sector 的主题 RS），最热的概念排第一行；🔥高=近期密集催化/板块大涨，中=有关注，低=冷。
- **相关度**：核心（占主营≥30% 或公司被市场当该概念龙头）/ 相关（10-30%）/ 边缘（<10% 或仅蹭概念）。
- ⛔ 无 `SYS_MAIN_BUSINESS:` 行（fina_mainbz 无数据）时：占比列填"未披露"，概念归属仍按主营业务描述+新闻判，但**不得编造百分比**。

各 Agent 的分歧只提炼进 Core Thesis 和 Key Risks，不在用户版报告重复输出逐 Agent 对照表。

**字段填写约束**：
- 全部字段必填，**禁止** "TBD"/"待评估"/"灵活调整"
- 评级为 HOLD/UNDERWEIGHT/SELL 时：
  - Entry 填 "—"（不建仓）
  - Action 填 "WAIT @<回调位>" 或 "REDUCE -X%" 或 "EXIT @<价位>"
  - ⛔ **TP1/TP2/TP3/SL_soft/SL_hard 仍必须按 R-multiple 计算具体价位（针对已持有者）**，**禁止填 "—"**
  - 1R 取当前价 P_0 作为 Entry 基准（而非空仓者建仓价）：`1R = P_0 − SL_hard`
  - 例：HOLD 评级当前价 271.83 元，SL_hard 选 215 元，则 1R=56.83，TP1=328.66，TP2=385.49，TP3=442.32，SL_soft=237.74
  - 同步在 PM_SUMMARY YAML 中 pm_tp1/pm_tp2/pm_tp3/pm_sl_soft/pm_sl_hard **必须填具体数字，禁止填 null**
- ⛔ **方向铁律（对标真实研报，禁止"评级不看多却处处暗示做多"）——触发条件：评级=UNDERWEIGHT/SELL，或综合目标价中位在现价下方**：
  - **Size（新建仓）必须 = 0%**——不看多的票操作建议是"不建仓/减仓"，Conviction 区间只作"若已持仓的减仓上限"括注，不放头部 Size
  - 现价上方的 TP1/TP2/TP3，**必须改标为"持仓者逢强减仓位（trim into strength），非看多止盈"**——12 个月综合目标价在现价下方，上方价位只是给已套牢/持仓者一个反弹离场参考，不是"目标上看"
  - **关键背景必须有一行点明方向，措辞按评级自适应**（别把 HOLD 叫"看空"）：
    * UNDERWEIGHT/SELL → "本票评级看空：综合目标价中位 <P_mid> 在现价下方（隐含 <−X%>）；上方 TP 仅供已持仓者反弹减仓，不构成看多目标"
    * HOLD（但目标价在现价下方）→ "本票评级中性偏谨慎：综合目标价中位 <P_mid> 低于现价（隐含 <−X%>），不追多；上方 TP 仅供已持仓者反弹减仓，不构成看多目标"
  - **禁止**出现"止盈""目标上看""上行空间"等暗示做多的措辞与"不看多/目标在下方"并存（自相矛盾，下游审计会标记）
- BUY/OVERWEIGHT 时 Entry 必须给具体区间

---

## 第九步：历史教训与一致性检查（仅内部执行，不输出）

- 将匹配的历史教训转化成一条具体动作或风险约束；不适用时忽略，不输出自检表。
- 检查空仓场景没有减仓/清仓指令，持仓场景没有正向建仓/加仓指令。
- 检查 Entry、TP、SL 的角色不冲突；反向证伪只写信号和业绩门槛。
- 检查评级、当前动作、新建仓位和市场闸门一致，修正后只输出最终结论。

## 最终报告结构

只输出以下内容，每项只出现一次：

1. **Trade Ticket 交易票**：顶部导航、核心交易参数、关键背景、热门概念归属。
2. **## 一、投资决策与入场时机**：投资判断、当前入场判断、未来 3 日趋势、12 个月主题判断。
3. **## 二、操作计划**：明确分开空仓者与持仓者；执行细节；仅在需要减仓时说明资金去向。
4. **## 三、情景概率与赔率**：四情景表和概率加权结果；不输出工具验证过程。
5. **## 四、风险、触发与监控**：反向证伪、使用真实日期或“第 N 检查点”的 Time Stop、关键监控和风控措施。
6. 报告最后直接输出 `PM_SUMMARY` YAML 代码块，不增加附录、自检、归档或评级链路说明。

输出完成后立即结束，不追加总结、致谢、prompt 合规说明或重复章节。

---

## 减仓资金去向（机会成本）规则

UNDERWEIGHT/SELL 评级时**必须输出**："减下来的资金该去哪？"

| 选项 | 适用场景 | 预期年化收益 |
|------|---------|-------------|
| 现金（货币基金/T+0 理财）| 短期观望，等待该标的回调入场 | ~1.5% |
| 国债 ETF | 中期避险 | ~2.5% |
| 同行业更优标的 | 若存在比该标的更佳的标的 | 需另估 |
| 行业 ETF | 保留行业 beta，降低个股 alpha 风险 | 行业平均 |

PM 必须明确推荐其中之一，并解释理由。

---

## 输入资料

### Research Manager 的 thesis（核心输入）
{research_plan}

### 股票画像（决定决策风格 + 报告使用权重 + Time Stop / Entry 节奏）
{stock_profile if stock_profile else "（未提供）"}

### 量化打分官（独立第二眼，Python 确定性输出 0-100 综合分 + 6 因子分项）
{quant_score if quant_score else "（量化锚未生成，PM 跳过量化交叉校验）"}

### 板块对照（Python 确定性输出，本股 vs 主题/行业/市场 ETF + 主题代表股的 RS）
{sector_comparison if sector_comparison else "（板块对照未生成，PM 跳过相对强弱判断）"}

### 共识快照（用于 entry timing 判断）
{consensus_snapshot if consensus_snapshot else "（未提供）"}

### 4 个 analyst 原始报告（PM 独享，用于校验 RM thesis + 操作细节）

[置信度:高] Company fundamentals report:
{fundamentals_report}

[置信度:中高] Market research report:
{market_report}

[置信度:中] Latest world affairs news:
{news_report}

[置信度:中低] Social media sentiment report:
{sentiment_report}

### 风险团队辩论记录
{history}

### 历史教训（BM25 检索出的最相关 3 条，仅供内部调整动作，不输出自检表）
{past_memory_str if past_memory_str else "（本次未检索到相关教训）"}

---

{RISK_DEBATE_PHRASING_RULES}

**重要**：请用中文撰写。评级关键词、股票代码、交易术语（Action/Size/R/TP/SL/Time Stop）保留英文但带中文注释。

---

## ⚠️ 报告末尾强制输出 PM_SUMMARY YAML（用于 harness 自动归档）

报告**完成后**，必须在最末尾输出一段 YAML 摘要，**字段名严格按以下格式**，否则归档失败。
所有数值直接采用 Trade Ticket 中已经定下的值，不要再调整。

```yaml
PM_SUMMARY:
  ticker: "{pm_ticker}"
  trade_date: "{pm_trade_date}"
  current_price: <float>                 # 当前价 P_0
  pm_rating: BUY / OVERWEIGHT / HOLD / UNDERWEIGHT / SELL
  pm_conviction_stars: <int 1-5>
  pm_invest_judgment: YES / NO / CONDITIONAL
  pm_entry_judgment: BUY_NOW / WAIT / DONT_BUY
  pm_action_keyword: BUY_NOW / WAIT / REDUCE / EXIT  # Trade Ticket Action 字段的关键词部分
  pm_size_low_pct: <float>               # 仓位区间下沿百分比（如 2.0 表示 2%）
  pm_size_high_pct: <float>              # 仓位区间上沿百分比
  pm_entry_low: <float or null>          # BUY/OVERWEIGHT 时必填；HOLD/UNDERWEIGHT/SELL 填 null
  pm_entry_high: <float or null>
  pm_tp1: <float>                        # ⛔ 持仓者止盈位 1（所有评级都必填具体数字，禁止 null）
  pm_tp2: <float>                        # ⛔ 持仓者止盈位 2（所有评级都必填具体数字，禁止 null）
  pm_tp3: <float>                        # ⛔ 持仓者止盈位 3（所有评级都必填具体数字，禁止 null）
  pm_sl_soft: <float>                    # ⛔ 软止损（所有评级都必填具体数字，禁止 null）
  pm_sl_hard: <float>                    # ⛔ 硬止损（所有评级都必填具体数字，禁止 null）
  pm_horizon_months_low: <int>           # Time Stop 时间窗口下沿（月）
  pm_horizon_months_high: <int>          # Time Stop 时间窗口上沿
  pm_rating_adjusted_from_rm: <bool>     # PM 是否相对 RM 评级做了 ±1 档微调
  market_risk_level: <低 / 中 / 高 / 极高 / 数据不足>
  market_entry_gate: <OPEN / CONDITIONAL / WAIT>
  market_position_cap_pct: <float>
  short_term_structure: trend_pullback / breakout_ready / healthy_trend / exhaustion / broken / neutral / insufficient_data
  entry_timing: 分批介入 / 小仓试探 / 等回踩 / 等放量突破 / 暂不介入 / 退出观察 / 继续观察 / 数据不足
  short_term_trend: <上涨 / 震荡 / 下跌>
  short_term_confidence: <高 / 中 / 低>
  theme_outlook_12m: <扩张 / 兑现 / 降速 / 破裂>
```

**约束**：
- 缺数据填 `null`，禁止编造
- 不要嵌套、不要加注释行；本节是供 Python 解析的固定格式
- 该 YAML 必须是报告最后一段，前后用 `---` 分隔，方便提取器定位

Be decisive and ground every conclusion in specific evidence from the analysts.{get_language_instruction()}"""

        # 绑定 PM 计算工具，让 LLM 调工具算 R-multiple / Conviction / 4 情景 E
        llm_with_tools = llm.bind_tools(PM_TOOLS)
        response = _pm_tool_loop(llm_with_tools, [HumanMessage(content=prompt)])
        pm_rating_match = re.findall(
            r"(?m)^\s*pm_rating:\s*(BUY|OVERWEIGHT|HOLD|UNDERWEIGHT|SELL)\s*$",
            response.content or "",
        )
        final_rating = pm_rating_match[-1] if pm_rating_match else rm_rating
        final_entry_timing = _derive_entry_timing_from_profile(
            stock_profile, market_mode, long_term_rating=final_rating,
        )
        enforced_content = _enforce_entry_timing_truth(response.content, final_entry_timing)
        response = AIMessage(content=_format_pm_decision(
            enforced_content, final_entry_timing, market_risk_snapshot=market_risk_snapshot,
        ))
        logger.info("PM entry_timing 出口真值: %s", final_entry_timing)

        new_risk_debate_state = {
            "judge_decision": response.content,
            "history": risk_debate_state["history"],
            "aggressive_history": risk_debate_state["aggressive_history"],
            "conservative_history": risk_debate_state["conservative_history"],
            "neutral_history": risk_debate_state["neutral_history"],
            "latest_speaker": "Judge",
            "current_aggressive_response": risk_debate_state["current_aggressive_response"],
            "current_conservative_response": risk_debate_state["current_conservative_response"],
            "current_neutral_response": risk_debate_state["current_neutral_response"],
            "count": risk_debate_state["count"],
        }

        return {
            "risk_debate_state": new_risk_debate_state,
            "final_trade_decision": response.content,
        }

    return portfolio_manager_node
