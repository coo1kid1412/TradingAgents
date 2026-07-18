# Intraday Price Overlay Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a validated realtime A-share quote overlay so same-day analyses use a provisional current OHLCV bar instead of silently stopping at T-1.

**Architecture:** Keep Tushare completed daily bars as the historical base, normalize realtime quotes behind one focused service, and append a non-persistent current bar only after date/session/freshness validation. Feed the merged series into stock data, indicators, profile, and Quant, with explicit metadata for downstream agents.

**Tech Stack:** Python 3.13, pandas, requests, Tushare Pro, stockstats, native script-style regression tests.

## Global Constraints

- A realtime failure must not abort the stock analysis.
- Never merge a quote from a different date or a stale continuous-session timestamp.
- Never persist a provisional intraday bar as completed historical data.
- Capital-flow freshness is outside this feature's scope.
- Preserve all unrelated user worktree changes.

---

### Task 1: Normalized realtime quote service

**Files:**
- Create: `tradingagents/dataflows/intraday_quote.py`
- Create: `tradingagents/dataflows/test_intraday_quote.py`

**Interfaces:**
- Produces: `IntradayQuote`, `fetch_intraday_quote(symbol, analysis_date, tushare_api=None, now=None)`, `merge_intraday_quote(daily_df, quote)`.

- [x] Write parser, validation, session-freshness, provider fallback, and merge tests.
- [x] Run the test script and verify failures are caused by the missing module/API.
- [x] Implement the minimal normalized service with Tushare `rt_k` then Sina fallback.
- [x] Run the test script and verify all cases pass.

### Task 2: Tushare stock and indicator integration

**Files:**
- Modify: `tradingagents/dataflows/tushare_vendor.py`
- Modify: `tradingagents/dataflows/test_intraday_quote.py`

**Interfaces:**
- Consumes: `fetch_intraday_quote` and `merge_intraday_quote` from Task 1.
- Produces: stock CSV freshness headers and indicator values calculated from the merged current bar.

- [x] Add failing integration tests for current-bar append, official-row precedence, and T-1 metadata.
- [x] Run the focused test script and confirm the new tests fail.
- [x] Integrate the overlay into `get_stock` and `get_indicator` without changing completed historical queries.
- [x] Run the focused tests and existing dataflow regressions.

### Task 3: Quant and Market freshness propagation

**Files:**
- Modify: `tradingagents/agents/utils/quant_score_node.py`
- Modify: `tradingagents/agents/analysts/market_analyst.py`
- Modify: `tradingagents/dataflows/test_intraday_quote.py`

**Interfaces:**
- Consumes: stock CSV metadata headers.
- Produces: deterministic `QUANT_SCORE.price_data_*` fields and Market `SUMMARY.price_data_*` contract.

- [x] Add failing tests for metadata parsing and required Market summary fields.
- [x] Run focused tests and confirm expected failures.
- [x] Implement metadata parsing, Quant report propagation, and Market prompt requirements.
- [x] Run focused tests and manager/dataflow regressions.

### Task 4: Verification and smoke

**Files:**
- Modify only if verification reveals a defect.

- [x] Run `compileall` and `git diff --check`.
- [x] Run all native regression scripts affected by stock data, Quant, entry timing, and market risk.
- [x] Run a 300308 quote/data smoke in `.venv`; if the market is closed, verify provider behavior and fixture replay without claiming a live current bar.
- [x] Review the diff for accidental inclusion of `main.py`, `.agents/`, local docs, or the harness database.
