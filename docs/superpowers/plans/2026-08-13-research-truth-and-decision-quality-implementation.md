# Research Truth and Decision Quality Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the 603629 review findings into deterministic, reusable quality gates across data, evidence, decision, and report output.

**Architecture:** Extend existing tools and pure-Python nodes rather than replacing agents. Official and normalized facts enter a claim-level evidence ledger; downstream LLM reasoning is constrained by decision eligibility, while output guards reconcile execution and presentation.

**Tech Stack:** Python, LangGraph, LangChain tools, Tushare Pro, AKShare, YAML, pytest-style project scripts.

## Global Constraints

- Preserve existing public report paths used by the harness.
- Do not hard-code 603629-specific values.
- Missing data must fail closed for execution without being labeled bearish.
- All production changes require a failing regression test first.

---

### Task 1: Official disclosure and evidence provenance

**Files:**
- Modify: `tradingagents/agents/utils/news_data_tools.py`
- Modify: `tradingagents/dataflows/tushare_vendor.py`
- Modify: `tradingagents/dataflows/interface.py`
- Modify: `tradingagents/agents/analysts/news_analyst.py`
- Modify: `tradingagents/agents/utils/research_evidence_node.py`
- Modify: `tradingagents/dataflows/news_catalyst.py`
- Test: `tradingagents/agents/utils/test_news_data_tools.py`
- Test: `tradingagents/agents/utils/test_research_evidence_node.py`
- Test: `tradingagents/dataflows/test_news_catalyst.py`

- [x] Write tests for 120-day disclosure lookup, structured forecasts, and source-tier eligibility.
- [x] Run tests and verify the current implementation fails for the intended reason.
- [x] Implement the tool and evidence contract.
- [x] Run focused tests to green.

### Task 2: Financial and technical normalization

**Files:**
- Modify: `tradingagents/agents/analysts/fundamentals_analyst.py`
- Modify: `tradingagents/agents/utils/quant_score_node.py`
- Modify: `tradingagents/dataflows/factor_calc.py`
- Modify: `tradingagents/dataflows/profile_calc.py`
- Modify: `tradingagents/agents/managers/rm_tools.py`
- Test: `tradingagents/dataflows/test_factor_quality_gates.py`
- Test: `tradingagents/dataflows/test_profile_calc.py`
- Test: `tradingagents/agents/managers/test_entry_timing.py`

- [x] Write tests for normalized ROE, partial factor coverage, and weak rebound classification.
- [x] Verify red failures.
- [x] Implement minimal normalization and classification logic.
- [x] Run focused tests to green.

### Task 3: Market-risk semantics, risk consensus, and executable levels

**Files:**
- Modify: `tradingagents/harness/market_risk.py`
- Modify: `tradingagents/graph/propagation.py`
- Modify: `tradingagents/agents/managers/research_manager.py`
- Modify: `tradingagents/agents/managers/portfolio_manager.py`
- Create: `tradingagents/agents/utils/risk_consensus.py`
- Create: `tradingagents/agents/utils/report_calendar.py`
- Test: `tradingagents/harness/test_market_risk.py`
- Test: `tradingagents/agents/managers/test_entry_timing.py`
- Test: `tradingagents/agents/utils/test_risk_consensus.py`

- [x] Write tests for unknown market state, on-demand preflight, reporting windows, and one effective risk cap.
- [x] Verify red failures.
- [x] Implement deterministic preflight and reconciliation.
- [x] Run focused tests to green.

### Task 4: User and audit report separation

**Files:**
- Modify: `main.py`
- Modify: `cli/main.py`
- Create: `tradingagents/reporting.py`
- Test: `tests/test_report_artifacts.py`

- [x] Write a test that requires a concise user artifact and a full audit artifact.
- [x] Verify the current writers fail.
- [x] Implement shared report rendering and wire both entry points.
- [x] Run focused tests to green.

### Task 5: Regression and smoke verification

- [x] Run all changed-module tests under `/Users/lailixiang/WorkSpace/QoderWorkspace/TradingAgents/.venv`.
- [x] Run the broader project test suite.
- [x] Run one full stock analysis under `.venv`.
- [x] Inspect generated reports for official price, official forecast, source eligibility, reporting windows, one risk cap, action clarity, and artifact separation.
