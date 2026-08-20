import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

import yaml

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from tradingagents.agents.utils.agent_utils import (
    build_instrument_context,
    get_language_instruction,
    validate_fundamentals_data,
)
from tradingagents.agents.analysts.fundamentals_tools import (
    FUNDAMENTALS_TOOLS, FUNDAMENTALS_TOOLS_BY_NAME,
)
from tradingagents.dataflows.interface import route_to_vendor
from tradingagents.dataflows.akshare_vendor import get_industry_pe_table
from tradingagents.agents.utils.yaml_summary import extract_yaml_mapping
from tradingagents.agents.utils.handoff import (
    analyst_handoff_contract,
    extract_final_report_artifact,
)

# 工具调用循环上限（防止 LLM 反复调同一工具）
_MAX_TOOL_ITERATIONS = 8

_FUNDAMENTALS_SUMMARY_FIELDS = {
    "pe_ttm", "pe_zone", "pe_industry_median", "pe_industry_median_source",
    "pe_vs_industry", "growth_yoy_revenue", "growth_yoy_profit",
    "growth_yoy_profit_recurring", "roe", "roe_basis", "financial_period",
    "cashflow_period_basis", "inventory_comparison_basis", "data_quality_flags",
    "debt_ratio", "fcf_quality", "business_model_type",
    "customer_concentration_top5_pct", "governance_score", "governance_red_flags",
    "ocf_to_net_profit_ratio", "receivable_vs_revenue_growth_gap",
    "recurring_profit_ratio", "earnings_quality", "red_flags", "rating",
    "data_implied_direction", "data_implied_reasoning",
}

logger = logging.getLogger(__name__)


def _fundamentals_summary_is_complete(report: str) -> bool:
    summary, status = extract_yaml_mapping(report or "", "SUMMARY")
    if not (
        status in {"valid", "recovered"}
        and isinstance(summary, dict)
        and _FUNDAMENTALS_SUMMARY_FIELDS.issubset(summary)
    ):
        return False
    if not all(isinstance(summary.get(key), list) for key in (
        "data_quality_flags", "governance_red_flags", "red_flags",
    )):
        return False
    enum_fields = {
        "pe_zone": {"高估", "合理", "低估"},
        "pe_industry_median_source": {"cninfo", "news_research", "unavailable"},
        "pe_vs_industry": {"高于", "接近", "低于", "不可比"},
        "roe_basis": {"ttm", "annual", "quarterly_unannualized"},
        "cashflow_period_basis": {"cumulative", "single_quarter", "annual", "unknown"},
        "inventory_comparison_basis": {"same_period", "year_end", "incomparable", "unknown"},
        "fcf_quality": {"高", "中", "低"},
        "governance_score": {"高", "中", "低"},
        "earnings_quality": {"高", "中", "低"},
        "rating": {"正面", "中性", "负面"},
        "data_implied_direction": {"偏多", "偏空", "中性"},
    }
    for key, allowed in enum_fields.items():
        if summary.get(key) is not None and summary.get(key) not in allowed:
            return False
    period = summary.get("financial_period")
    if period not in {None, "unknown"} and not re.fullmatch(r"20\d{2}(?:Q1|H1|Q3|FY)", str(period)):
        return False
    numeric_fields = {
        "pe_ttm", "pe_industry_median", "growth_yoy_revenue", "growth_yoy_profit",
        "growth_yoy_profit_recurring", "roe", "debt_ratio",
        "customer_concentration_top5_pct", "ocf_to_net_profit_ratio",
        "receivable_vs_revenue_growth_gap", "recurring_profit_ratio",
    }
    return all(
        summary.get(key) is None
        or isinstance(summary.get(key), (int, float)) and not isinstance(summary.get(key), bool)
        for key in numeric_fields
    )


def _fundamentals_data_digest(structured_data: str, max_chars: int = 16000) -> str:
    """Keep high-signal source rows for an artifact-only retry."""
    keywords = (
        "SYS_", "PE", "PB", "PS", "EPS", "ROE", "ROA", "报告期", "营收",
        "净利润", "扣非", "资产负债率", "流动比率", "现金流", "应收", "存货",
        "总市值", "客户", "行业", "同比", "毛利率", "净利率",
    )
    selected = [
        line.strip() for line in (structured_data or "").splitlines()
        if line.strip() and any(keyword in line for keyword in keywords)
    ]
    digest = "\n".join(selected)
    return digest[:max_chars] if digest else (structured_data or "")[:max_chars]


def _compact_fundamentals_prompt(structured_data: str, tool_results: list[str]) -> str:
    return f"""基本面分析已经完成，但上一轮没有留下可见正文。

只输出可交付内容，禁止输出 <think>、思维过程、工具调用或过程说明，也不要再调用工具。
先用不超过 700 个中文字写出：估值、增长、盈利质量、治理/负债风险和基本面方向；
随后输出完整可解析的 `SUMMARY` YAML，字段必须与原任务模板完全一致，缺失值填 null。
不得编造客户、合同、财务期间或行业数据。

已执行的确定性计算工具结果：
{chr(10).join(tool_results)[-12000:] if tool_results else "无；相关派生字段必须填 null"}

高信号原始数据：
{_fundamentals_data_digest(structured_data)}
"""


def _mark_fundamentals_retry_partial(report: str) -> str:
    """Audit compact-retry output as partial even when its YAML is parseable."""
    matches = list(re.finditer(r"```yaml\s*\n(.*?)\n```", report or "", re.S | re.I))
    for match in reversed(matches):
        try:
            parsed = yaml.safe_load(match.group(1))
        except yaml.YAMLError:
            continue
        summary = parsed.get("SUMMARY") if isinstance(parsed, dict) else None
        if not isinstance(summary, dict):
            continue
        flags = summary.get("data_quality_flags")
        flags = list(flags) if isinstance(flags, list) else []
        marker = "模型压缩重试输出，工具口径已保留但需复核"
        if marker not in flags:
            flags.append(marker)
        summary["data_quality_flags"] = flags
        replacement = "```yaml\n" + yaml.safe_dump(
            {"SUMMARY": summary}, allow_unicode=True, sort_keys=False,
        ).rstrip() + "\n```"
        return report[:match.start()] + replacement + report[match.end():]
    return report


def _run_fundamentals_tool_loop(
    llm_with_tools,
    initial_messages: list,
    *,
    structured_data: str,
    max_iterations: int = _MAX_TOOL_ITERATIONS,
    max_finalization_attempts: int = 2,
):
    """Run calculation tools, then require a visible, parseable analyst artifact."""
    messages = list(initial_messages)
    visible_segments: list[str] = []
    tool_result_summaries: list[str] = []

    for iteration in range(max_iterations):
        try:
            result = llm_with_tools.invoke(messages)
        except Exception as exc:
            logger.warning("Fundamentals 主流程 LLM 调用失败，转入压缩收敛：%s", exc)
            break
        messages.append(result)
        content = str(getattr(result, "content", "") or "").strip()
        if content:
            visible_segments.append(content)

        tool_calls = getattr(result, "tool_calls", None) or []
        if not tool_calls:
            joined = "\n\n".join(visible_segments)
            if _fundamentals_summary_is_complete(joined):
                return AIMessage(content=extract_final_report_artifact(joined))
            logger.warning(
                "Fundamentals 可见正文缺失或 SUMMARY 不完整，第 %d 轮转入压缩收敛",
                iteration + 1,
            )
            break

        logger.info("Fundamentals 第 %d 轮工具调用：%d 个", iteration + 1, len(tool_calls))
        for tc in tool_calls:
            tool_name = tc.get("name")
            tool_args = tc.get("args", {})
            tool_id = tc.get("id", "")
            tool = FUNDAMENTALS_TOOLS_BY_NAME.get(tool_name)
            if tool is None:
                messages.append(ToolMessage(content=f"未知工具：{tool_name}", tool_call_id=tool_id))
                continue
            try:
                tool_result = tool.invoke(tool_args)
                payload = json.dumps(tool_result, ensure_ascii=False) if isinstance(tool_result, dict) else str(tool_result)
                tool_result_summaries.append(f"{tool_name}: {payload[:3000]}")
                messages.append(ToolMessage(content=payload, tool_call_id=tool_id))
            except Exception as exc:
                error = f"工具 {tool_name} 失败: {exc}"
                tool_result_summaries.append(f"{tool_name}: {error}")
                messages.append(ToolMessage(content=error, tool_call_id=tool_id))
    else:
        logger.warning("Fundamentals 达到工具调用上限 %d 轮，转入压缩收敛", max_iterations)

    final_messages = [
        SystemMessage(content="你是基本面分析师。内部分析已完成，现在只负责输出最终可见报告。"),
        HumanMessage(content=_compact_fundamentals_prompt(structured_data, tool_result_summaries)),
    ]
    for attempt in range(max_finalization_attempts):
        try:
            result = llm_with_tools.invoke(final_messages)
        except Exception as exc:
            logger.warning("Fundamentals 压缩收敛第 %d 次调用失败：%s", attempt + 1, exc)
            continue
        content = str(getattr(result, "content", "") or "").strip()
        if content and _fundamentals_summary_is_complete(content):
            return AIMessage(content=extract_final_report_artifact(
                _mark_fundamentals_retry_partial(content)
            ))
        logger.warning("Fundamentals 压缩收敛第 %d 次仍无完整 SUMMARY", attempt + 1)
        final_messages.append(HumanMessage(
            content="上一轮仍未生成可见且完整的 SUMMARY。直接输出短报告和完整 YAML，禁止思考过程。"
        ))
    return AIMessage(content=extract_final_report_artifact(
        "\n\n".join(visible_segments)
    ))


def _find_float(text: str, pattern: str):
    match = re.search(pattern, text or "", re.I)
    if not match or match.group(1).upper() in {"NA", "N/A"}:
        return None
    try:
        return round(float(match.group(1)), 2)
    except (TypeError, ValueError):
        return None


def _yaml_number(value) -> str:
    return "null" if value is None else str(value)


def _build_deterministic_fundamentals_fallback(vendor_text: str, current_date: str) -> str:
    """Emit a conservative partial SUMMARY when visible output stays unavailable."""
    pe_ttm = _find_float(vendor_text, r"PE\s*\(TTM\)\s*[:：]\s*([-+]?\d+(?:\.\d+)?)")
    revenue_yoy = _find_float(vendor_text, r"营收YoY[^\n]*?年度=([-+]?\d+(?:\.\d+)?|NA)%")
    profit_yoy = _find_float(vendor_text, r"归母净利YoY[^\n]*?年度=([-+]?\d+(?:\.\d+)?|NA)%")
    recurring_yoy = _find_float(vendor_text, r"扣非净利YoY[^\n]*?年度=([-+]?\d+(?:\.\d+)?|NA)%")
    debt_ratio = _find_float(vendor_text, r"资产负债率=([-+]?\d+(?:\.\d+)?)%")
    return f"""## 基本面摘要（确定性兜底）

LLM可见正文缺失，以下仅保留可从数据源直接解析的指标；其余字段按缺失处理，
下游必须将该领域视为部分覆盖，不得据此单独提高评级或仓位。

```yaml
SUMMARY:
  pe_ttm: {_yaml_number(pe_ttm)}
  pe_zone: null
  pe_industry_median: null
  pe_industry_median_source: unavailable
  pe_vs_industry: 不可比
  growth_yoy_revenue: {_yaml_number(revenue_yoy)}
  growth_yoy_profit: {_yaml_number(profit_yoy)}
  growth_yoy_profit_recurring: {_yaml_number(recurring_yoy)}
  roe: null
  roe_basis: null
  financial_period: unknown
  cashflow_period_basis: unknown
  inventory_comparison_basis: unknown
  data_quality_flags:
    - LLM可见正文缺失，使用确定性兜底
  debt_ratio: {_yaml_number(debt_ratio)}
  fcf_quality: null
  business_model_type: null
  customer_concentration_top5_pct: null
  governance_score: null
  governance_red_flags: []
  ocf_to_net_profit_ratio: null
  receivable_vs_revenue_growth_gap: null
  recurring_profit_ratio: null
  earnings_quality: null
  red_flags:
    - 基本面需在下次分析重试
  rating: null
  data_implied_direction: null
  data_implied_reasoning: 仅确定性指标可用，完整分析缺失
```
"""


def _safe_fetch(method: str, *args, **kwargs) -> str:
    """Call route_to_vendor with graceful error handling."""
    try:
        return route_to_vendor(method, *args, **kwargs)
    except Exception as e:
        logger.warning("基本面数据获取失败 (%s): %s", method, e)
        return ""


def _safe_fetch_industry_pe(target_date: str) -> str:
    """Wrap industry PE table fetch with graceful error handling."""
    try:
        return get_industry_pe_table(target_date)
    except Exception as e:
        logger.warning("行业 PE 表获取失败: %s", e)
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
    with ThreadPoolExecutor(max_workers=5) as executor:
        future_to_key = {
            executor.submit(_safe_fetch, method, *args): key
            for key, (method, args) in methods.items()
        }
        # 并行多拉一个行业 PE 表（来源 akshare 巨潮，独立于 route_to_vendor）
        future_to_key[executor.submit(_safe_fetch_industry_pe, current_date)] = "industry_pe_table"

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

    # 行业 PE 中位数表（巨潮 cninfo，独立于其他财务数据）
    industry_pe = raw_data.get("industry_pe_table", "")
    if industry_pe:
        sections.append(industry_pe)
    else:
        sections.append("# 行业 PE 中位数表\n（数据不可用，pe_industry_median 字段请尝试从 news 报告中提取卖方研报的同业可比 PE）")

    header = f"# {ticker} 基本面数据（截至 {current_date}）\n\n"
    return header + "\n\n---\n\n".join(sections)


# 必须确定性传递的机读行前缀：源数据(tushare_vendor)里有，但 LLM 写报告时经常丢——
# 300308_20260612 实跑：SYS_GROWTH_YOY 被丢 → 画像 peg_det=None → 无 SYS_PEG 注入
# → RM 退回自估 PEG 还伪造了"SYS_PEG_CONFIDENCE=low"的来源标注。
# 修法与 stock_profile_node 同款：Python 原样转录追加在报告末尾，不靠 LLM 自觉。
_SYS_LINE_PREFIXES = ("【SYS_GROWTH_YOY", "【SYS_LANDMINE", "【SYS_CYCLICAL", "EPS(TTM):")

_PRICE_STATUS_LABELS = {
    "intraday_provisional": "盘中临时价",
    "official_daily": "正式收盘价",
    "t_minus_1": "最近正式收盘价（T-1）",
}


def _extract_price_field(vendor_text: str, label: str):
    match = re.search(rf"^{re.escape(label)}:\s*(.+?)\s*$", vendor_text or "", re.M)
    if not match:
        return None
    value = match.group(1).strip()
    return None if value in {"", "N/A", "None"} else value


def _enforce_price_context(report: str, vendor_text: str) -> str:
    """Prepend the vendor's deterministic valuation-price context to the report."""
    status = _extract_price_field(vendor_text, "价格数据状态")
    price = _extract_price_field(vendor_text, "估值基准价(元)")
    if not status or not price:
        return report

    label = _PRICE_STATUS_LABELS.get(status, status)
    date = _extract_price_field(vendor_text, "估值基准日期")
    quote_time = _extract_price_field(vendor_text, "估值基准时间")
    source = _extract_price_field(vendor_text, "估值基准来源")
    official_close = _extract_price_field(vendor_text, "前收参考价(元,非当前价)")
    official_date = _extract_price_field(vendor_text, "前收日期")

    parts = [f"> **估值价格口径：{label} {price} 元**"]
    if date:
        parts.append(f"日期：{date}")
    if quote_time:
        parts.append(f"报价时间：{quote_time}")
    if source:
        parts.append(f"来源：{source}")
    if official_close and status != "official_daily" and official_date != date:
        parts.append(f"前收参考：{official_close} 元（非当前价）")
        if official_date:
            parts.append(f"前收日期：{official_date}")
    return "｜".join(parts) + "\n\n" + report.lstrip()


def _extract_sys_lines(vendor_text: str) -> list[str]:
    """从源数据文本里抽出机读行（防御式：空文本返回空列表）。"""
    return [ln.strip() for ln in (vendor_text or "").splitlines()
            if ln.strip().startswith(_SYS_LINE_PREFIXES)]


def _append_sys_lines(report: str, vendor_text: str) -> str:
    """把源数据机读行追加到 LLM 报告末尾（已被 LLM 原样转抄的不重复追加）。"""
    missing = [ln for ln in _extract_sys_lines(vendor_text) if ln not in report]
    if not missing:
        return report
    return (report
            + "\n\n<!-- ⚠️SYS 机读行｜Python 从源数据原样转录(LLM 报告丢行不再断确定性链)，下游解析直读 -->\n"
            + "\n".join(missing) + "\n")


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

        # 2.5. Run data cross-validation on PE/EPS consistency
        validation_warning = validate_fundamentals_data(raw_data.get("fundamentals", ""))
        if validation_warning:
            structured_data = validation_warning + structured_data

        # 3. Single LLM call for analysis
        lang_instruction = get_language_instruction()

        system_message = f"""【语言要求】你必须使用中文撰写所有分析报告和回复内容。股票代码、财务指标英文缩写（如 EPS、P/E、ROE 等）以及评级关键词（BUY/SELL/HOLD）请保留英文原文。

你是一名专业的基本面分析师，负责分析公司的基本面信息并撰写全面的研究报告。

## ⚠️ 价格口径（强制约束）

- 「估值基准价(元)」是本报告唯一允许使用的当前价格；PE、PB、PS、总市值必须采用标有「按估值基准价调整」的数值
- `价格数据状态=intraday_provisional` 时，必须写作「盘中价/盘中临时价」，严禁称为「收盘价」
- `价格数据状态=official_daily` 时，同日正式日线是唯一当前价；严禁被收盘后实时回报价降级成「盘中临时价」
- 「前收参考价(元,非当前价)」仅用于比较涨跌，严禁写成当前价、现价或估值计算基准
- 报告标题、公司概况、估值分析、关键指标表中的价格和市值口径必须一致；不得混用盘中价与前收价

## ⚠️ 数值计算必须调用工具（强制约束）

你已绑定 7 个计算工具。**所有以下数值必须通过工具调用完成，禁止心算或偷懒填 null**：

| 计算场景 | 必须调用的工具 |
|---------|-------------|
| 自由现金流 FCF = OCF − CapEx | `compute_fcf` |
| 营收/净利润同比增速 | `compute_yoy_growth` |
| 扣非净利润同比（扣非 EPS × 总股本）| `compute_recurring_profit_yoy` |
| 应收增速 vs 营收增速差 | `compute_receivable_revenue_gap` |
| 经营现金流/净利润比（盈利质量）| `compute_ocf_to_profit_ratio` |
| 应付/应收比（议价能力）| `compute_payable_receivable_ratio` |
| TTM 年化 ROE（滚动 4 季）| `compute_ttm_roe` |

**规则**：
- 涉及上述任何数值时，**必须**先调用对应工具，再把工具结果写入报告
- raw_data 里有原始数据但**没有现成的派生指标**（如 FCF / 同比增速 / TTM ROE）时，调工具自算
- 工具返回的数值直接采用，不要"调整"或"修正"
- ⛔ **禁止偷懒**：发现 raw_data 里没有"净利润同比"字段就填 null —— 这是过往 bug 来源。raw_data 里几乎一定有本期和去年同期的绝对值，必须调工具自算

## ⚠️ 财务期间与单位一致性（强制约束）

- 任何同比、环比、现金流/利润比只允许同口径期间相除：累计对累计、单季对单季、年度对年度；禁止把累计现金流称为“单季现金流”
- 存货、应收等资产负债表项目只能与同类报告期或明确年末基准比较；没有同口径基期时写“不可比”，不得判断改善/恶化
- 现金流/净利润比绝对值异常（如 >3 或 <-1）时，先标记“期间或单位待核”，在核验前不得据此把盈利质量评为高
- 单季 CapEx 不能单独证明扩产周期启动或结束；生命周期判断至少需要连续两个可比期或官方产能公告
- 每个核心财务结论必须注明 `financial_period` 与口径；发现混期/单位冲突写入 `data_quality_flags`

## 报告结构（必须按此顺序）

### 一、公司概况
公司名称、所属行业、上市日期、市值规模等基本信息。

### 二、商业模式分析（新增，专业 RM 重点）
不只看财务数字，必须分析公司**怎么赚钱**：

| 维度 | 输出 | 数据缺失时 |
|------|------|------------|
| **收入构成（按产品/业务线）** | Top 3 产品/业务占比，如"DDR5 内存接口 65% / CXL 15% / 其他 20%" | 标注"未披露，仅依据年报概要分析" |
| **收入构成（按客户）** | 客户集中度 Top5 占比，是否过度依赖单一大客户 | 同上 |
| **收入构成（按地区）** | 国内 vs 海外占比，是否存在地缘集中风险 | 同上 |
| **上游议价能力** | 关键原材料/供应商集中度，是否受供应链单点制约 | 同上 |
| **下游议价能力** | 客户应收账款周转、价格调整频次 | 同上 |
| **商业模式核心特征** | 一句话总结：典型成长股 / 周期股 / 平台型 / 重资产制造 / 渠道驱动 / 技术壁垒型 | — |

⚠️ **若 raw_data 中无具体收入构成数据，明确标注"数据不可用"，禁止凭印象编造比例**。

**客户集中度数据提取要求**（必读）：
- akshare/tushare 的财报数据**不直接提供**客户集中度，但年报附注/卖方研报通常有
- **优先从 news 报告中查找**"前 5 大客户占营收 X%" / "客户集中度 X%" / "前 5 名客户合计销售 X 亿元"等表述
- 若 news 报告里有提到，**必须提取数值**填入 SUMMARY 的 `customer_concentration_top5_pct` 字段
- 例如：news 报告中"前五大客户营收占比 47.92%" → `customer_concentration_top5_pct: 47.92`
- 仅当 news 报告也确实没有客户集中度信息时，才填 null
- ⛔ **禁止偷懒填 null**：必须先在 news 报告里搜索过

### 三、管理层与公司治理评估（新增）

| 维度 | 输出 |
|------|------|
| **股东结构** | 大股东背景（民企/国资/外资）、前 3 大股东持股比例、是否一致行动人 |
| **管理层背景** | 核心高管（CEO/CFO/CTO）行业经验年数、过往业绩、是否核心团队稳定 |
| **激励是否对齐股东** | 是否有股权激励计划、激励解锁条件是否合理（如对应业绩目标）|
| **过往承诺兑现率** | 历史业绩指引兑现情况、上一次定增/再融资资金使用是否符合预期 |
| **治理红旗** | 大股东高比例质押 / 关联交易激增 / 高管减持密集 / 审计师变更 等任一触发即列出 |

⚠️ 数据缺失时明确标注，不编造。

### 四、估值分析
分析 PE(TTM)、PB、PS 等估值指标的水平及历史分位，判断当前估值处于高估/合理/低估区间，说明理由。

**行业 PE 对照（数据源三级 fallback，必须执行）**：

下方"输入资料"区可能包含一段"# 行业 PE 中位数表（来源：巨潮 cninfo）"——这是从巨潮接口拉来的全行业 PE 数据，含 19 个一级行业 + 91 个二级行业的静态市盈率中位数。

**Step 1**: 检查是否拼进来了"行业 PE 中位数表"
- ✅ 有 → 走 Step 2
- ❌ 无 → 跳到 Step 3（news fallback）

**Step 2**: 从巨潮行业表中**自行选最匹配的行业**
- 优先选**二级行业**（精度更高，如 C39 计算机、通信和其他电子设备制造业；C27 医药制造业）
- 若二级行业不直接命中，fallback 到对应**一级行业**（如 C 制造业 / I 信息传输 / J 金融业）
- 取该行业的"静态市盈率-中位数"作为 pe_industry_median
- 显式说明匹配过程："本标的属于'XXX'，匹配巨潮行业表中的'C39 计算机...'（二级），PE 中位数 = 88.6"
- SUMMARY 的 `pe_industry_median_source` 填 `cninfo`

**Step 3**（fallback）: 巨潮表不可用时，从 news 报告中查找
- 搜索 news 报告中"行业 PE" / "同业可比 PE" / "可比公司平均 PE" / "卖方对标 X / Y / Z 公司 PE 平均 N 倍"等表述
- 找到则提取，SUMMARY 的 `pe_industry_median_source` 填 `news_research`
- 注明数据局限性（卖方研报"可比公司"口径，不是行业全样本中位数）

**Step 4**（最后兜底）: 上述两源都不可用
- pe_industry_median 填 null
- pe_industry_median_source 填 `unavailable`
- pe_vs_industry 填 "不可比"

**输出格式**（必须执行）：
- 在估值分析正文中显式说明数据来源 + 匹配过程
- 例：'当前 PE 80.73 vs 巨潮行业表 C39（计算机/通信/电子设备制造业）中位数 88.6 倍 → 略低于行业'

⛔ **禁止**：
- 凭印象给出"行业平均约 25 倍"（若无明确数据来源）
- 拼进来了巨潮行业表却不去用、直接填 null
- 把可比公司"算术平均"当成"行业中位数"——这是两个口径，必须区分

### 五、盈利能力
分析 ROE、ROA、净利润率、毛利率等盈利指标的近几期变化趋势，判断盈利能力是否在改善或恶化，指出关键驱动因素。

⚠️ **ROE 口径要求（必读，避免误导下游）**：
- **优先采用 TTM 年化 ROE**：滚动 4 个季度净利润 ÷ 当期净资产
- **不能直接用单季 ROE 代替年度 ROE**：例如 Q1 单季 ROE = 0.11% 不等于年化 0.44%（季节性高的公司不能线性外推）
- 报告中**每次引用 ROE 必须显式标注口径**：'ROE(TTM)' / 'ROE(年度)' / 'ROE(Q1 单季未年化)'
- 若公司季节性极强（如 Q1 占全年利润比例 <20%），仅引用 Q1 单季 ROE 会严重低估盈利能力，必须说明这一点
- 若 raw_data 中既有季度也有年度 ROE，**SUMMARY 优先填年度 ROE**，季度数据放在正文里讨论

### 六、盈利质量深度分析（新增，避免"利润虚胖"陷阱）

盈利质量比绝对利润数字更重要——同样 +30% 净利润，如果是经营驱动 vs 一次性收益，含金量天差地别。

| 维度 | 公式/标准 | 当前数值 | 评分 |
|------|----------|---------|------|
| **经营现金流/净利润比** | 经营现金流 ÷ 净利润 | __ | >0.8 健康 / 0.5-0.8 警惕 / <0.5 红旗 |
| **应收账款增速 vs 营收增速** | 应收增速 − 营收增速 | __ pp | < 0 健康 / 0-5pp 正常 / >5pp 警惕（应收增长过快可能虚增收入）|
| **扣非净利润 vs 归母净利润** | 扣非占归母比例 | __% | >85% 健康 / 60-85% 中等 / <60% 红旗（一次性损益占比过高）|
| **存货周转率变化** | 同比变化方向 | __ | 改善/平稳/恶化 |
| **应付/应收比** | 应付账款 ÷ 应收账款 | __ | >1 议价能力强 / <1 现金流压力 |

**盈利质量综合评级**（必输出）：高 / 中 / 低 + 一句话理由。

### 七、偿债能力与财务风险
分析资产负债率、流动比率、速动比率、利息保障倍数等指标，评估公司的债务压力和短期偿债风险。

### 八、成长性分析
分析营收增长率、净利润增长率等成长指标的近几期变化，判断成长性趋势及可持续性。

⚠️ **同比增速计算（强制）**：
- 优先采用 raw_data 现成的"营收/净利润同比增长率"字段；缺失时**必须**调 `compute_yoy_growth`（公式 `(本期 − 去年同期)/|去年同期|×100%`，基期 Q1 对 Q1、年度对年度），并显式展开 'X 亿元 vs Y 亿元，同比 +Z%'
- 区分 `growth_yoy_profit_reported`（归母）vs `growth_yoy_profit_recurring`（扣非，调 `compute_recurring_profit_yoy`：扣非 EPS × 总股本 → 同比）
- ⛔ raw_data 里几乎一定有多期数据和扣非 EPS，**禁止偷懒填 null**——只有真没有去年同期数据时才允许 null

### 九、现金流分析
分析经营性现金流、投资性现金流、筹资性现金流的规模和趋势，重点关注经营性现金流与净利润的匹配度（盈利质量）。

**FCF 计算（强制）**：必须调 `compute_fcf`（公式 `FCF = OCF − CapEx`，分别取 cashflow 里"经营活动现金流量净额"和"购建固定资产/无形资产/其他长期资产支付的现金"），显式展开 'FCF Q = OCF X − CapEx Y = Z 亿'，多季度时给 TTM 累加和环比。⛔ raw_data 几乎一定有 OCF/CapEx 两字段，**禁止填"数据不可用"**——只有 cashflow 整段为空（如 ETF）才允许"不适用"。

### 十、投资建议与风险提示
基于以上分析，给出明确的基本面判断（正面/中性/负面），列出 2-3 个核心支撑论据和 1-2 个主要风险点。

### 十一、关键指标汇总表（必须包含）
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

## 强制输出：SUMMARY 块（位于报告末尾）
在报告所有正文章节和汇总表格之后，**必须**附加一个 YAML 代码块，
格式严格如下（字段名、单位、取值集合不可变）：

```yaml
SUMMARY:
  pe_ttm: <数值>
  pe_zone: 高估 / 合理 / 低估
  pe_industry_median: <数值或 null>
  pe_industry_median_source: cninfo / news_research / unavailable  # 数据来源标注
  pe_vs_industry: 高于 / 接近 / 低于 / 不可比
  growth_yoy_revenue: <百分比>                          # 优先 TTM/年度同比，缺失时按计算要求自算
  growth_yoy_profit: <百分比>                           # 归母净利润同比（优先 TTM/年度）
  growth_yoy_profit_recurring: <百分比 或 null>          # 扣非净利润同比（如可计算）
  roe: <百分比>                                          # 优先填 TTM 年化 ROE，否则填年度 ROE
  roe_basis: ttm / annual / quarterly_unannualized      # ROE 取值口径（强制标注，不允许省略）
  financial_period: <YYYYQ1 / YYYYH1 / YYYYQ3 / YYYYFY / unknown>
  cashflow_period_basis: cumulative / single_quarter / annual / unknown
  inventory_comparison_basis: same_period / year_end / incomparable / unknown
  data_quality_flags:
    - <期间/单位异常，≤30字；无则空列表>
  debt_ratio: <百分比>
  fcf_quality: 高 / 中 / 低
  # 商业模式（新增）
  business_model_type: 成长股 / 周期股 / 平台型 / 重资产制造 / 渠道驱动 / 技术壁垒型 / 其他
  customer_concentration_top5_pct: <0-100 或 null>      # 客户集中度（Top5 占比）
  # 管理层与治理（新增）
  governance_score: 高 / 中 / 低                         # 综合治理评分
  governance_red_flags:
    - <治理红旗，≤20 字>
  # 盈利质量深化（新增）
  ocf_to_net_profit_ratio: <数值或 null>                # 经营现金流/净利润
  receivable_vs_revenue_growth_gap: <百分点或 null>     # 应收增速 − 营收增速
  recurring_profit_ratio: <0-100 或 null>               # 扣非净利占归母比例
  earnings_quality: 高 / 中 / 低                         # 盈利质量综合
  red_flags:
    - <财务红旗描述，≤20 字>
  rating: 正面 / 中性 / 负面                             # 措辞评级（保守表达）
  data_implied_direction: 偏多 / 偏空 / 中性             # 数据真实隐含方向（穿透措辞）
  data_implied_reasoning: <≤30 字说明>
```

## SUMMARY 规则
- 字段缺失时填 null 或 "不适用"，不允许省略字段名
- 取值必须落在 schema 允许的集合内
- 数值字段保留 2 位小数；百分比字段直接填数字（不带 % 符号）
- 该 SUMMARY 块是下游 RM / 风控团队的核心信息源，宁缺勿错

## ⚠️ 数值类指标使用规范（PE/EPS/PB 等）
- **PE(TTM)、动态PE、静态PE、EPS 等有标准计算公式的指标**：必须优先使用数据中「## PE估值（系统计算）」段落的「系统计算」值
- **仅在「系统计算」值不可用时**，才可使用「API参考」值或自行计算
- **如需自行计算**：必须写明公式并验证结果（示例: PE = 估值基准价/年度EPS = 210.27/1.97 = 106.7），不允许凭记忆估算
- **数据校验警告**：如果数据前方出现「⚠️ 数据质量校验警告」段落，请优先处理其建议

当前日期：{current_date}。{instrument_context}{lang_instruction}"""

        system_message += analyst_handoff_contract(
            "fundamentals",
            "判断未来一年盈利质量、竞争力和估值锚是否支持持有",
            "交接财务期间与口径、增长和现金流变化桥、商业壁垒与治理、12个月盈利驱动、估值方法与区间、bear/base/bull 假设范围及 thesis breaker；不得判断三日买点或绕过市场风险门。",
        )

        messages = [
            SystemMessage(content=system_message),
            HumanMessage(content=f"请基于以下基本面数据撰写详细的分析报告：\n\n{structured_data}"),
        ]

        # 绑定 Fundamentals 计算工具，让 LLM 调工具算 FCF / 同比 / TTM ROE 等
        llm_with_tools = llm.bind_tools(FUNDAMENTALS_TOOLS)

        result = _run_fundamentals_tool_loop(
            llm_with_tools,
            messages,
            structured_data=structured_data,
        )
        report = result.content if hasattr(result, "content") else str(result)
        if not _fundamentals_summary_is_complete(report):
            fallback = _build_deterministic_fundamentals_fallback(
                raw_data.get("fundamentals", ""), current_date,
            )
            report = f"{report.rstrip()}\n\n{fallback}" if report.strip() else fallback
        report = _enforce_price_context(report, raw_data.get("fundamentals", ""))
        # 机读行确定性传递（Python 兜底，不靠 LLM 转抄）
        report = _append_sys_lines(report, raw_data.get("fundamentals", ""))

        return {
            "messages": [
                AIMessage(content=report, name="Fundamentals Analyst")
            ],
            "fundamentals_report": report,
        }

    return fundamentals_analyst_node
