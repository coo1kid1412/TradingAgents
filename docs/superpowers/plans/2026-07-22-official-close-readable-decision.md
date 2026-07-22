# Official Close And Readable Decision Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make same-day official closes authoritative across agents and turn `decision.md` into a concise, action-first user report without breaking harness extraction.

**Architecture:** Fix price selection at the fundamentals vendor boundary, then enforce the precedence in downstream prompts. Extend the existing deterministic PM formatter to build a user-facing action summary and remove model-generated audit/process sections while preserving `PM_SUMMARY`.

**Tech Stack:** Python 3.13, pandas, regex-based Markdown normalization, existing LangChain nodes and harness YAML extractor.

## Global Constraints

- Preserve existing `PM_SUMMARY` keys and harness compatibility.
- Preserve unrelated local modifications.
- Use `.venv/bin/python` for all tests and smoke analysis.
- Do not add dependencies.

---

### Task 1: Official-close price precedence

**Files:**
- Modify: `tradingagents/dataflows/test_intraday_quote.py`
- Modify: `tradingagents/dataflows/tushare_vendor.py`
- Modify: `tradingagents/agents/analysts/fundamentals_analyst.py`
- Modify: `tradingagents/agents/utils/macro_context_node.py`

**Interfaces:**
- Consumes: `_resolve_fundamentals_price_context(ticker, curr_date, pro, daily_basic)`
- Produces: a context whose `status`, `price`, `date`, `time`, and `source` describe one authoritative price.

- [ ] Add a failing test where `daily_basic.trade_date == curr_date` and a same-day realtime quote is also available; assert `official_daily`, `tushare_daily_basic`, no quote timestamp, and no previous-close banner.
- [ ] Run `.venv/bin/python tradingagents/dataflows/test_intraday_quote.py` and confirm the new assertion fails.
- [ ] Return immediately from realtime overlay logic when a valid same-day official row exists, and suppress previous-close display for `official_daily`.
- [ ] Add explicit downstream prompt language for `official_daily > intraday_provisional > t_minus_1`.
- [ ] Re-run the intraday quote and Tushare valuation tests and confirm they pass.

### Task 2: Action-first PM report formatter

**Files:**
- Modify: `tradingagents/agents/managers/test_entry_timing.py`
- Modify: `tradingagents/agents/managers/portfolio_manager.py`

**Interfaces:**
- Consumes: model Markdown, deterministic entry timing, optional market-risk snapshot.
- Produces: concise user Markdown with a stable action header and intact terminal `PM_SUMMARY`.

- [ ] Add a failing representative test containing tool verification, self-check/archive sections, internal `SYS_*` text, ambiguous time-stop labels, and a no-buy/add-position contradiction.
- [ ] Run `.venv/bin/python tradingagents/agents/managers/test_entry_timing.py` and confirm the new test fails.
- [ ] Add helpers that split the terminal `PM_SUMMARY`, remove internal sections, normalize time-stop wording, and build explicit empty-position/existing-position actions from summary fields.
- [ ] Simplify the PM output contract so model output no longer requests a public audit appendix or tool-verification prose.
- [ ] Re-run entry-timing and harness extractor tests and confirm `PM_SUMMARY` remains parseable.

### Task 3: Regression and report smoke test

**Files:**
- Verify: all modified files
- Generate: one report under `reports/` (not committed)

**Interfaces:**
- Consumes: project `.env`, `.venv`, and current analysis entry point.
- Produces: a real report whose price status and decision readability satisfy the design.

- [ ] Run focused price, PM, market-risk, and harness tests.
- [ ] Run the repository's relevant broader test commands.
- [ ] Generate a real A-share report in `.venv` and wait for normal completion unless logs show a clear abnormal state.
- [ ] Verify the report's price against Tushare official daily data, check that no same-day official price is labelled intraday, and inspect the final `decision.md` action summary and internal-text filters.
- [ ] Commit only task-related files, push the feature branch, merge through the authorized repository workflow, and verify remote `main` contains the commit.
