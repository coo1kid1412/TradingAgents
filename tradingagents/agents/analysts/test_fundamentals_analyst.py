from langchain_core.messages import AIMessage, HumanMessage

from tradingagents.agents.analysts.fundamentals_analyst import (
    _build_deterministic_fundamentals_fallback,
    _fundamentals_summary_is_complete,
    _run_fundamentals_tool_loop,
)


_COMPLETE_SUMMARY = """基本面可见摘要。

```yaml
SUMMARY:
  pe_ttm: 61.0
  pe_zone: 高估
  pe_industry_median: null
  pe_industry_median_source: unavailable
  pe_vs_industry: 不可比
  growth_yoy_revenue: 47.16
  growth_yoy_profit: 1088.59
  growth_yoy_profit_recurring: 3906.41
  roe: null
  roe_basis: annual
  financial_period: 2025FY
  cashflow_period_basis: annual
  inventory_comparison_basis: incomparable
  data_quality_flags: []
  debt_ratio: 75.1
  fcf_quality: 中
  business_model_type: 其他
  customer_concentration_top5_pct: null
  governance_score: 中
  governance_red_flags: []
  ocf_to_net_profit_ratio: null
  receivable_vs_revenue_growth_gap: null
  recurring_profit_ratio: null
  earnings_quality: 中
  red_flags: []
  rating: 中性
  data_implied_direction: 中性
  data_implied_reasoning: 数据完整度有限
```
"""


class _SequenceLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.messages = []

    def invoke(self, messages):
        self.messages.append(messages)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def test_fundamentals_tool_loop_retries_when_think_body_is_empty():
    llm = _SequenceLLM([AIMessage(content=""), AIMessage(content=_COMPLETE_SUMMARY)])

    result = _run_fundamentals_tool_loop(
        llm,
        [HumanMessage(content="原始基本面输入")],
        structured_data="PE(TTM): 61\n资产负债率: 75.1%",
    )

    assert _fundamentals_summary_is_complete(result.content)
    assert "M3压缩重试输出" in result.content
    assert len(llm.messages) == 2
    assert "只输出可交付" in llm.messages[-1][-1].content


def test_deterministic_fallback_is_parseable_and_marks_partial_quality():
    report = _build_deterministic_fundamentals_fallback(
        """PE(TTM): 61.25
【SYS_GROWTH_YOY｜tushare】 营收YoY 单季=41.61% 年度=47.16% | 归母净利YoY 单季=NA% 年度=1088.59% | 扣非净利YoY 年度=3906.41%
【SYS_LANDMINE｜tushare】 ST=否 | 连续两年亏损=否 | 资产负债率=75.1% | 流动比率=1.06
""",
        current_date="2026-08-14",
    )

    assert _fundamentals_summary_is_complete(report)
    assert "financial_period: unknown" in report
    assert "pe_zone: null" in report
    assert "fcf_quality: null" in report
    assert "LLM可见正文缺失" in report


def test_fundamentals_summary_rejects_wrong_schema_types():
    malformed = _COMPLETE_SUMMARY.replace("data_quality_flags: []", "data_quality_flags: not-a-list")
    assert not _fundamentals_summary_is_complete(malformed)


def test_compact_retry_carries_forward_tool_results():
    first = AIMessage(
        content="",
        tool_calls=[{
            "name": "compute_fcf",
            "args": {"operating_cash_flow": 10, "capex": 4},
            "id": "fcf-1",
            "type": "tool_call",
        }],
    )
    llm = _SequenceLLM([first, AIMessage(content=""), AIMessage(content=_COMPLETE_SUMMARY)])

    result = _run_fundamentals_tool_loop(
        llm,
        [HumanMessage(content="原始基本面输入")],
        structured_data="经营现金流 10，资本开支 4",
    )

    assert _fundamentals_summary_is_complete(result.content)
    assert "compute_fcf" in llm.messages[-1][-1].content


def test_llm_errors_degrade_to_caller_fallback_instead_of_escaping():
    llm = _SequenceLLM([TimeoutError("primary timeout"), TimeoutError("retry timeout")])

    result = _run_fundamentals_tool_loop(
        llm,
        [HumanMessage(content="原始基本面输入")],
        structured_data="PE(TTM): 30",
        max_finalization_attempts=1,
    )

    assert result.content == ""


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"  PASS {test.__name__}")
    print(f"\n{len(tests)}/{len(tests)} passed")
