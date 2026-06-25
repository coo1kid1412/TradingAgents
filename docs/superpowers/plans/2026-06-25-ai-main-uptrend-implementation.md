# AI Main Uptrend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a deterministic AI main-uptrend signal that can modestly relax RM ratings only when broad market risk permits and individual stock quality is confirmed.

**Architecture:** `profile_calc.py` computes a pure-Python `SYS_AI_MAIN_UPTREND` signal. `stock_profile_node.py` appends the signal to stock profiles. `research_manager.py` passes the signal and market mode into `compute_step6_final_rating`, which applies a bounded adjustment before the existing trend overlay and invariants.

**Tech Stack:** Python, LangChain tool wrappers, pytest-style module tests, existing `.venv` project environment.

## Global Constraints

- Do not derive rules from desired historical outcomes.
- Market environment is controlled by `market_risk_daily` snapshots; missing snapshots are `risk_off`.
- AI main-uptrend can never bypass `discipline`, crowded-long BUY ceiling, or target-price sign invariant.
- Pure AI narrative without earnings/order/cash-flow confirmation must not be upgraded.
- Run a real single-stock regression from `.venv` after unit tests.

---

### Task 1: Add AI Main-Uptrend Pure Function

**Files:**
- Modify: `tradingagents/dataflows/profile_calc.py`
- Test: `tradingagents/dataflows/test_valuation_regime.py`

**Interfaces:**
- Produces: `compute_ai_main_uptrend_signal(...) -> dict`
- Return keys: `enabled: bool`, `class: str`, `reasons: list[str]`, `blockers: list[str]`

- [ ] **Step 1: Write failing tests**

Add tests that cover confirmed, early, non-AI, discipline-blocked, loss-blocked, and blowoff-blocked cases in `tradingagents/dataflows/test_valuation_regime.py`.

- [ ] **Step 2: Run tests to verify failure**

Run: `.venv/bin/python -m pytest tradingagents/dataflows/test_valuation_regime.py -q`
Expected: FAIL because `compute_ai_main_uptrend_signal` is missing.

- [ ] **Step 3: Implement minimal pure function**

Add `compute_ai_main_uptrend_signal` to `profile_calc.py`, using only passed-in deterministic values.

- [ ] **Step 4: Run tests to verify pass**

Run: `.venv/bin/python -m pytest tradingagents/dataflows/test_valuation_regime.py -q`
Expected: PASS.

### Task 2: Emit SYS_AI_MAIN_UPTREND In Stock Profile

**Files:**
- Modify: `tradingagents/agents/utils/stock_profile_node.py`

**Interfaces:**
- Consumes: `compute_ai_main_uptrend_signal(...) -> dict`
- Produces profile footer lines:
  - `SYS_AI_MAIN_UPTREND_ENABLED: true/false`
  - `SYS_AI_MAIN_UPTREND_CLASS: confirmed/early/none`
  - `SYS_AI_MAIN_UPTREND_REASONS: ...`
  - `SYS_AI_MAIN_UPTREND_BLOCKERS: ...`

- [ ] **Step 1: Add import and call**

Import and call `compute_ai_main_uptrend_signal` after `regime_info` and `rev_info` are available.

- [ ] **Step 2: Append deterministic footer**

Append SYS lines near the other deterministic SYS blocks.

- [ ] **Step 3: Run profile-related tests**

Run: `.venv/bin/python -m pytest tradingagents/dataflows/test_valuation_regime.py -q`
Expected: PASS.

### Task 3: Add Market Mode And Step 6 Adjustment

**Files:**
- Modify: `tradingagents/agents/managers/rm_tools.py`
- Test: `tradingagents/agents/managers/test_step6_final_rating.py`

**Interfaces:**
- Produces: `derive_market_mode(market_risk_snapshot: dict | None) -> str`
- Extends `compute_step6_final_rating` args:
  - `ai_main_uptrend: Optional[bool] = False`
  - `ai_main_uptrend_class: Optional[str] = ""`
  - `market_mode: Optional[str] = ""`

- [ ] **Step 1: Write failing Step 6 tests**

Add tests for risk_on confirmed upgrade, risk_on confirmed strong-confirm BUY upgrade, conditional cap, risk_off no-op, discipline ceiling, and crowded-long BUY ceiling.

- [ ] **Step 2: Run tests to verify failure**

Run: `.venv/bin/python -m pytest tradingagents/agents/managers/test_step6_final_rating.py -q`
Expected: FAIL because new args/behavior are absent.

- [ ] **Step 3: Implement market mode and AI adjustment**

Apply AI adjustment after symmetric adjustment and before trend overlay. Clamp through existing bounds.

- [ ] **Step 4: Run tests to verify pass**

Run: `.venv/bin/python -m pytest tradingagents/agents/managers/test_step6_final_rating.py -q`
Expected: PASS.

### Task 4: Wire RM Prompt And Tool Call

**Files:**
- Modify: `tradingagents/agents/managers/research_manager.py`

**Interfaces:**
- Consumes profile footer lines and `market_risk_snapshot` from state.
- Passes `ai_main_uptrend`, `ai_main_uptrend_class`, and `market_mode` to `compute_step6_final_rating`.

- [ ] **Step 1: Add prompt instructions**

Update Step 6 input table and tool return text to mention AI main-uptrend and market mode.

- [ ] **Step 2: Add deterministic parsing helpers if needed**

Keep parsing simple and local; absent fields must default to disabled/risk_off.

- [ ] **Step 3: Run manager tests**

Run: `.venv/bin/python -m pytest tradingagents/agents/managers/test_step6_final_rating.py -q`
Expected: PASS.

### Task 5: Verify And Single-Stock Regression

**Files:**
- No planned source changes unless tests expose a bug.

**Interfaces:**
- Uses `.venv/bin/python`.

- [ ] **Step 1: Run targeted unit tests**

Run:
`.venv/bin/python -m pytest tradingagents/dataflows/test_valuation_regime.py tradingagents/agents/managers/test_step6_final_rating.py tradingagents/harness/test_market_risk.py -q`

- [ ] **Step 2: Run one real-stock regression**

Run one AI main-uptrend candidate through the project entrypoint using `.venv`. Prefer `300308` unless runtime/data limits make another A/B candidate more practical.

- [ ] **Step 3: Inspect generated report**

Check the latest report for:
`SYS_AI_MAIN_UPTREND`, `market_mode`, Step 6 AI main-uptrend adjustment, final RM rating, and PM market-risk gate behavior.

- [ ] **Step 4: Report outcome**

Report test commands, real-stock command, generated report path, and whether the behavior matches the spec.
