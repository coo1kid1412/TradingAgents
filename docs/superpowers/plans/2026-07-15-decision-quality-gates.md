# Decision Quality Gates Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make three-day trade guidance timely and internally consistent while preventing low-quality earnings from inflating one-year valuation.

**Architecture:** Keep financial rules deterministic in the existing Python calculation layer. Let LLM agents explain validated outputs, but enforce report invariants at the PM output boundary and fail closed on stale market-risk snapshots.

**Tech Stack:** Python, LangGraph state, SQLite market-risk snapshots, pytest-compatible script tests.

## Global Constraints

- Preserve existing report schemas unless adding backward-compatible fields.
- Do not modify the user's local `main.py` ticker change.
- Use `.venv/bin/python` for verification and a real-stock smoke run.

---

### Task 1: Earnings-quality valuation gate

**Files:** `tradingagents/dataflows/profile_calc.py`, `tradingagents/agents/utils/stock_profile_node.py`, `tradingagents/dataflows/test_valuation_regime.py`

- [ ] Add a failing test proving recurring losses invalidate deterministic PEG inputs.
- [ ] Pass growth-quality evidence into the PEG input calculation.
- [ ] Emit an explicit machine-readable PEG invalid reason and no PEG target price.
- [ ] Run the valuation tests.

### Task 2: Quality-adjusted quant factors

**Files:** `tradingagents/dataflows/factor_calc.py`, `tradingagents/agents/utils/quant_score_node.py`, relevant factor tests

- [ ] Add failing tests for recurring-loss growth caps and trapped/distributed crowding penalties.
- [ ] Prefer deducted-profit growth and cap headline growth when recurring profit is negative.
- [ ] Apply holder-distribution and trapped-position evidence to anti-crowding.
- [ ] Apply recency contradiction penalties to capital flow.
- [ ] Run factor and capital-flow tests.

### Task 3: Three-day decision and output invariants

**Files:** `tradingagents/agents/managers/portfolio_manager.py`, `tradingagents/agents/managers/research_manager.py`, `tradingagents/agents/managers/test_entry_timing.py`

- [ ] Add failing tests that an exit-observation report contains no new-entry instruction or nonzero new-position size.
- [ ] Rename the short horizon from five days to three trading days.
- [ ] Sanitize contradictory entry instructions at the deterministic PM output boundary.
- [ ] Replace cost-basis-driven holder guidance with thesis and risk-budget guidance in the prompt contract.
- [ ] Run manager tests.

### Task 4: Market-risk freshness

**Files:** `tradingagents/harness/market_risk.py`, `tradingagents/graph/propagation.py`, market-risk tests

- [ ] Add failing tests for analysis-time-aware snapshot freshness.
- [ ] Mark same-day premarket-only snapshots stale during market hours when an intraday checkpoint should exist.
- [ ] Fail the short-term gate closed when the latest required checkpoint is missing.
- [ ] Run harness tests.

### Task 5: Verification

- [ ] Run all directly affected test modules under `.venv`.
- [ ] Run a real-stock analysis smoke test.
- [ ] Inspect the generated `decision.md` for three-day wording, stale-data disclosure, PEG gating, zero-new-position consistency, and absence of contradictory buy instructions.
