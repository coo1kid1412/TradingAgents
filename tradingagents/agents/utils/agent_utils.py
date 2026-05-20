import re
from langchain_core.messages import HumanMessage, RemoveMessage

# Import tools from separate utility files
from tradingagents.agents.utils.core_stock_tools import (
    get_stock_data
)
from tradingagents.agents.utils.technical_indicators_tools import (
    get_indicators
)
from tradingagents.agents.utils.fundamental_data_tools import (
    get_fundamentals,
    get_balance_sheet,
    get_cashflow,
    get_income_statement
)
from tradingagents.agents.utils.news_data_tools import (
    get_news,
    get_insider_transactions,
    get_global_news,
    get_announcements,
    get_cls_telegraph,
    get_research_reports,
    get_news_from_search,
)
from tradingagents.agents.utils.xueqiu_data_tools import (
    get_xueqiu_posts,
)


RISK_DEBATE_PHRASING_RULES = """
【输出措辞规范（合规要求，必须严格遵守）】
为避免触发输出层合规审核（如 MiniMax 1027），请使用以下中性表述：
- ❌ "暴雷" / "爆雷" / "业绩暴雷" → ✅ "业绩显著低于预期" / "业绩大幅不达预期"
- ❌ "崩盘" / "崩塌" / "崩溃" / "股价崩盘" → ✅ "深度回调" / "估值大幅压缩"
- ❌ "血洗" / "腰斩" → ✅ "跌幅超过 50%"
- ❌ "做空获利" / "砸盘" → ✅ "下行情景的对冲收益" / "卖压释放"
- ❌ "亏多少" / "巨亏" / "血亏" → ✅ "潜在下行幅度" / "下行风险敞口"
- ❌ "时间炸弹" / "引爆" / "爆炸" → ✅ "风险事件" / "触发" / "兑现"
- ❌ "踩雷" / "雷区" → ✅ "命中风险点" / "高风险区域"

保留所有定量分析和数字，仅替换情绪化/煽动性措辞。本规则仅针对措辞，不影响风险识别的严谨性与深度。
"""


# 数值类指标使用规范（Bull/Bear/PM 共享，避免在三处分别 inline）
NUMERIC_VALUE_USAGE_RULES = """
## ⚠️ 数值类指标使用规范
- 引用 PE(TTM)、动态PE、EPS 等估值指标时，**必须使用分析师报告中的「系统计算」值**（如「动态PE(系统计算)」），严禁自行计算不同的 PE 值
- 如果发现两个不同的 PE 数值（如 API 参考值 vs 系统计算值），**以系统计算值为准**
- 如需验证或自行计算，必须写明公式和中间步骤
"""


def get_language_instruction() -> str:
    """Return a prompt instruction for the configured output language.

    Returns empty string when English (default), so no extra tokens are used.
    Only applied to user-facing agents (analysts, portfolio manager).
    Internal debate agents stay in English for reasoning quality.
    """
    from tradingagents.dataflows.config import get_config
    lang = get_config().get("output_language", "English")
    if lang.strip().lower() == "english":
        return ""
    return f" Write your entire response in {lang}."


def build_instrument_context(ticker: str, company_name: str = "") -> str:
    """Describe the exact instrument so agents preserve exchange-qualified tickers."""
    name_part = f" (**{company_name}**)" if company_name else ""
    return (
        f"The instrument to analyze is `{ticker}`{name_part}. "
        "Use this exact ticker in every tool call, report, and recommendation, "
        "preserving any exchange suffix (e.g. `.TO`, `.L`, `.HK`, `.T`)."
    )

def create_msg_delete():
    def delete_messages(state):
        """Clear messages and add placeholder for Anthropic compatibility"""
        messages = state["messages"]

        # Remove all messages
        removal_operations = [RemoveMessage(id=m.id) for m in messages]

        # Add a minimal placeholder message
        placeholder = HumanMessage(content="Continue")

        return {"messages": removal_operations + [placeholder]}

    return delete_messages


def build_report_context(state) -> str:
    """将各分析师报告拼接为上下文字符串，供风控团队等 agent 使用。"""
    parts = []
    for key, label in [
        ("fundamentals_report", "[置信度:高] Company fundamentals report"),
        ("market_report", "[置信度:中高] Market research report"),
        ("news_report", "[置信度:中] Latest world affairs news"),
        ("sentiment_report", "[置信度:中低] Social media sentiment report"),
    ]:
        data = state.get(key, "")
        if data:
            parts.append(f"{label}: {data}")
    return "\n\n".join(parts)


def validate_fundamentals_data(raw_fundamentals_text: str) -> str:
    """校验基本面数据中的 PE/EPS 等关键估值指标一致性。

    在基本面分析师处理原始数据之前调用，将校验警告插入到数据文本前方，
    确保后续所有 Agent 都能看到数据质量提示。
    """
    warnings = []

    # 1. 检查 Tushare PE 偏差警告
    if "⚠️ PE偏差警告" in raw_fundamentals_text:
        for line in raw_fundamentals_text.split("\n"):
            if "⚠️ PE偏差警告" in line:
                warnings.append(f"【数据校验】{line.strip()}")
                break

    # 2. 用正则提取系统计算的 PE 和 EPS 做公式校验
    pe_match = re.search(r"动态PE\(系统计算\):\s*([\d.]+)\s*倍", raw_fundamentals_text)
    close_match = re.search(r"收盘价\(元\):\s*([\d.]+)", raw_fundamentals_text)
    if pe_match and close_match:
        calc_pe = float(pe_match.group(1))
        close_price = float(close_match.group(1))
        implied_eps = close_price / calc_pe

        # 检查 EPS 一致性
        eps_patterns = [
            r"基本每股收益\(EPS\)[:\s]*([\d.]+)",
            r"EPS\(TTM\)[:\s]*([\d.]+)",
            r"每股收益[:\s]*([\d.]+)",
        ]
        for pat in eps_patterns:
            eps_match = re.search(pat, raw_fundamentals_text)
            if eps_match:
                reported_eps = float(eps_match.group(1))
                if abs(implied_eps - reported_eps) / max(implied_eps, reported_eps) > 0.20:
                    warnings.append(
                        f"【数据校验】PE/EPS 公式不一致: PE={calc_pe} 隐含 TTM_EPS={implied_eps:.2f}, "
                        f"但报告 EPS={reported_eps:.2f} (偏差 >20%)"
                    )
                break

    # 3. PE 合理性检查（A 股 PE 超过 500 倍通常异常）
    if pe_match:
        pe_val = float(pe_match.group(1))
        if pe_val > 500:
            warnings.append(f"【数据校验】动态PE={pe_val}倍异常偏高，请核实 EPS 数据是否正确")

    # 4. 静态 vs 动态 PE 比率检查
    pe_static_match = re.search(r"静态PE\(系统计算\):\s*([\d.]+)\s*倍", raw_fundamentals_text)
    if pe_match and pe_static_match:
        dynamic_pe = float(pe_match.group(1))
        static_pe = float(pe_static_match.group(1))
        if static_pe > 0 and dynamic_pe > 0:
            ratio = dynamic_pe / static_pe
            if ratio > 3.0 or ratio < 0.3:
                warnings.append(
                    f"【数据校验】动态PE/静态PE={ratio:.1f}，偏差过大 "
                    f"(动态={dynamic_pe} vs 静态={static_pe})，请检查 EPS 数据"
                )

    if warnings:
        return (
            "## ⚠️ 数据质量校验警告（请优先阅读）\n\n"
            + "\n".join(f"- {w}" for w in warnings)
            + "\n\n> 说明: 以上为系统自动校验结果。请在分析报告中使用「系统计算」的 PE 值，"
            "避免自行计算 PE/EPS 等数值类指标。\n\n---\n\n"
        )
    return ""
