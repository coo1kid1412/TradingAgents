import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from langchain_core.messages import AIMessage

from tradingagents.agents.utils.agent_utils import (
    build_instrument_context,
    get_language_instruction,
)
from tradingagents.dataflows.interface import route_to_vendor

logger = logging.getLogger(__name__)


def _safe_fetch(method: str, *args, **kwargs) -> str:
    """Call route_to_vendor with graceful error handling."""
    try:
        return route_to_vendor(method, *args, **kwargs)
    except Exception as e:
        logger.warning("基本面数据获取失败 (%s): %s", method, e)
        return ""


def _fetch_all_fundamentals(ticker: str, current_date: str) -> dict:
    """Fetch all fundamental data in parallel via ThreadPoolExecutor."""
    methods = {
        "fundamentals": ("get_fundamentals", (ticker, current_date)),
        "balance_sheet": ("get_balance_sheet", (ticker, "quarterly", current_date)),
        "cashflow": ("get_cashflow", (ticker, "quarterly", current_date)),
        "income_statement": ("get_income_statement", (ticker, "quarterly", current_date)),
    }

    results = {}
    with ThreadPoolExecutor(max_workers=4) as executor:
        future_to_key = {
            executor.submit(_safe_fetch, method, *args): key
            for key, (method, args) in methods.items()
        }
        for future in as_completed(future_to_key):
            key = future_to_key[future]
            try:
                results[key] = future.result()
            except Exception as e:
                logger.warning("基本面数据线程异常 (%s): %s", key, e)
                results[key] = ""

    return results


def _format_structured_data(raw_data: dict, ticker: str, current_date: str) -> str:
    """Format raw fundamental data into structured text for LLM analysis."""
    sections = []

    section_map = {
        "fundamentals": ("公司基本面概览", "综合基本面数据"),
        "balance_sheet": ("资产负债表", "最近季度资产负债表数据"),
        "cashflow": ("现金流量表", "最近季度现金流量表数据"),
        "income_statement": ("利润表", "最近季度利润表数据"),
    }

    for key, (title, desc) in section_map.items():
        data = raw_data.get(key, "")
        if data:
            sections.append(f"## {title}\n{desc}：\n\n{data}")
        else:
            sections.append(f"## {title}\n（数据获取失败，请基于其他可用数据进行分析）")

    header = f"# {ticker} 基本面数据（截至 {current_date}）\n\n"
    return header + "\n\n---\n\n".join(sections)


def create_fundamentals_analyst(llm):
    def fundamentals_analyst_node(state):
        ticker = state["company_of_interest"]
        current_date = state["trade_date"]
        instrument_context = build_instrument_context(
            ticker, state.get("company_name", "")
        )

        # 1. Programmatically fetch all fundamental data in parallel
        raw_data = _fetch_all_fundamentals(ticker, current_date)

        # 2. Format into structured text
        structured_data = _format_structured_data(raw_data, ticker, current_date)

        # 3. Single LLM call for analysis
        lang_instruction = get_language_instruction()

        system_message = f"""【语言要求】你必须使用中文撰写所有分析报告和回复内容。股票代码、财务指标英文缩写（如 EPS、P/E、ROE 等）以及评级关键词（BUY/SELL/HOLD）请保留英文原文。

你是一名专业的基本面分析师，负责分析公司的基本面信息并撰写全面的研究报告。

## 报告结构（必须按此顺序）

### 一、公司概况
公司名称、所属行业、上市日期、市值规模等基本信息。

### 二、估值分析
分析 PE(TTM)、PB、PS 等估值指标的水平及历史分位，判断当前估值处于高估/合理/低估区间，说明理由。

### 三、盈利能力
分析 ROE、ROA、净利润率、毛利率等盈利指标的近几期变化趋势，判断盈利能力是否在改善或恶化，指出关键驱动因素。

### 四、偿债能力与财务风险
分析资产负债率、流动比率、速动比率、利息保障倍数等指标，评估公司的债务压力和短期偿债风险。

### 五、成长性分析
分析营收增长率、净利润增长率等成长指标的近几期变化，判断成长性趋势及可持续性。

### 六、现金流分析
分析经营性现金流、投资性现金流、筹资性现金流的规模和趋势，重点关注经营性现金流与净利润的匹配度（盈利质量）。

### 七、投资建议与风险提示
基于以上分析，给出明确的基本面判断（正面/中性/负面），列出 2-3 个核心支撑论据和 1-2 个主要风险点。

### 八、关键指标汇总表（必须包含）
在报告末尾附加一个 Markdown 表格，**必须**包含以下指标（数据可用时）：

| 指标 | 数值 | 变动趋势 | 说明 |
|------|------|---------|------|
| PE(TTM) | | | |
| PB | | | |
| PS(TTM) | | | |
| 总市值 | | | |
| ROE | | | |
| ROA | | | |
| 净利润率 | | | |
| 资产负债率 | | | |
| 流动比率 | | | |
| 营收增长率 | | | |
| 净利润增长率 | | | |
| 经营性现金流 | | | |
| 自由现金流 | | | |
| EPS(TTM) | | | |

注意：如果某项指标在提供的数据中不可用（如 ETF/基金无财务报表），在表格中标注「不适用」并在正文中说明原因，不要编造数据。

当前日期：{current_date}。{instrument_context}{lang_instruction}"""

        messages = [
            {"role": "system", "content": system_message},
            {
                "role": "user",
                "content": f"请基于以下基本面数据撰写详细的分析报告：\n\n{structured_data}",
            },
        ]

        result = llm.invoke(messages)

        report = result.content if hasattr(result, "content") else str(result)

        return {
            "messages": [
                AIMessage(content=report, name="Fundamentals Analyst")
            ],
            "fundamentals_report": report,
        }

    return fundamentals_analyst_node
