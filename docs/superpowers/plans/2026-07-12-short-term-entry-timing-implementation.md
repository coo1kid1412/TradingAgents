# Short-Term Entry Timing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a deterministic short-term structure classifier and market/fundamental-gated entry-timing output without changing long-term rating arithmetic.

**Architecture:** Compute structure once from OHLCV in `profile_calc.py`, transcribe it into the stock profile as a machine-readable truth line, and map it to entry timing through a deterministic manager helper. Existing Market Analyst, RM, and PM prompts explain and display the result but cannot override it.

**Tech Stack:** Python 3, pandas, NumPy, LangChain tools, existing TradingAgents state and `.venv` test runner.

## Global Constraints

- Long-term BUY/OVERWEIGHT/HOLD/UNDERWEIGHT/SELL arithmetic must remain unchanged.
- `market_risk_daily` remains the market gate; missing snapshots are `risk_off`.
- `recurring_loss`, `earnings_revision=下修`, and `valuation_regime=discipline` veto positive entry actions.
- No new market-data provider or machine-learning dependency.
- Classification and gating are deterministic Python; M3 may explain but may not reclassify.
- Preserve all unrelated dirty-worktree changes.

---

### Task 1: Deterministic OHLCV Structure Classifier

**Files:**
- Create: `tradingagents/dataflows/test_short_term_structure.py`
- Modify: `tradingagents/dataflows/profile_calc.py`

**Interfaces:**
- Consumes: pandas DataFrame with case-insensitive OHLCV columns and optional `rsi_percentile_1y: float | None`, `has_vol_divergence: bool`.
- Produces: `compute_short_term_structure(price_df, *, rsi_percentile_1y=None, has_vol_divergence=False) -> dict` with the exact schema from the approved design.

- [ ] **Step 1: Write failing classifier tests**

Create deterministic synthetic frames for `trend_pullback`, `breakout_ready`, `healthy_trend`, `exhaustion`, `broken`, and `insufficient_data`. Assertions must check both `structure_class` and the numeric evidence that caused it; a conflict test must prove `broken` wins over positive classes.

- [ ] **Step 2: Verify RED**

Run: `.venv/bin/python -m pytest tradingagents/dataflows/test_short_term_structure.py -q`

Expected: collection fails because `compute_short_term_structure` does not exist.

- [ ] **Step 3: Implement the minimal classifier**

Add small internal helpers for case-insensitive columns, finite numeric cleaning, `clamp`, ATR14, and reason construction. Use the approved precedence:

```python
broken > exhaustion > trend_pullback > breakout_ready > healthy_trend > neutral
```

Use prior 20-day highs that exclude the current bar. Return `insufficient_data` when fewer than 20 valid rows remain, and suppress positive volume classifications when recent volume is zero.

- [ ] **Step 4: Verify GREEN**

Run: `.venv/bin/python -m pytest tradingagents/dataflows/test_short_term_structure.py -q`

Expected: all classifier tests pass.

- [ ] **Step 5: Commit classifier**

```bash
git add tradingagents/dataflows/profile_calc.py tradingagents/dataflows/test_short_term_structure.py
git commit -m "feat: add deterministic short-term structure classifier"
```

### Task 2: Deterministic Entry-Timing Gate

**Files:**
- Create: `tradingagents/agents/managers/test_entry_timing.py`
- Modify: `tradingagents/agents/managers/rm_tools.py`

**Interfaces:**
- Consumes: `structure_class`, `market_mode`, `recurring_loss`, `earnings_revision`, `valuation_regime`, `has_peak_signal`, `retail_concentration_signal`, `rsi_percentile_1y`, `capital_flow_regime`, and `main_force_streak_days`.
- Produces: `compute_entry_timing(...) -> dict` containing `base_action`, `effective_action`, `market_mode`, `vetoed`, and `reasons`.

- [ ] **Step 1: Write failing mapping tests**

Cover all seven structure classes. Add matrix assertions proving `risk_on` preserves actions, `conditional` changes only `分批介入` to `小仓试探`, and `risk_off` changes every positive action to `暂不介入`. Add separate tests for recurring loss, downward revision, discipline, peak, retail-plus-RSI-extreme, and sustained outflow vetoes.

- [ ] **Step 2: Verify RED**

Run: `.venv/bin/python -m pytest tradingagents/agents/managers/test_entry_timing.py -q`

Expected: collection fails because `compute_entry_timing` does not exist.

- [ ] **Step 3: Implement the minimal mapping helper**

Keep the mapping in one constant and apply individual-stock vetoes before the market-mode downgrade. Unknown classes must map to `数据不足`, and unknown market modes must fail closed as `risk_off`.

- [ ] **Step 4: Verify GREEN and existing gates**

Run: `.venv/bin/python -m pytest tradingagents/agents/managers/test_entry_timing.py tradingagents/agents/managers/test_market_risk_gate.py tradingagents/agents/managers/test_step6_final_rating.py -q`

Expected: all tests pass and existing rating behavior is unchanged.

- [ ] **Step 5: Commit timing gate**

```bash
git add tradingagents/agents/managers/rm_tools.py tradingagents/agents/managers/test_entry_timing.py
git commit -m "feat: add deterministic entry timing gate"
```

### Task 3: Profile Truth-Line Integration

**Files:**
- Modify: `tradingagents/agents/utils/stock_profile_node.py`
- Modify: `tradingagents/dataflows/test_short_term_structure.py`

**Interfaces:**
- Consumes: Task 1 `compute_short_term_structure(...) -> dict` plus existing `price_signals`.
- Produces: profile line `SYS_SHORT_TERM_STRUCTURE: class=... | ma10_slope_5d_pct=... | price_vs_ma10_pct=... | volume_ratio_5d_20d=... | breakout_confirmed=...`.

- [ ] **Step 1: Write a failing serialization test**

Extract a pure formatter `_format_short_term_structure_line(signal: dict) -> str` and test exact field names, stable ordering, lowercase booleans, and `N/A` handling.

- [ ] **Step 2: Verify RED**

Run: `.venv/bin/python -m pytest tradingagents/dataflows/test_short_term_structure.py -q`

Expected: fails because the formatter is missing.

- [ ] **Step 3: Implement and wire the truth line**

Call the classifier immediately after existing `compute_price_signals(price_df)`, pass the existing RSI percentile and volume-divergence signal, log the classification, append the machine line next to `SYS_AI_MAIN_UPTREND`, and include a concise human-readable structure row in the profile.

- [ ] **Step 4: Verify GREEN and profile regressions**

Run: `.venv/bin/python -m pytest tradingagents/dataflows/test_short_term_structure.py tradingagents/dataflows/test_valuation_regime.py -q`

Expected: all tests pass.

- [ ] **Step 5: Commit profile integration**

```bash
git add tradingagents/agents/utils/stock_profile_node.py tradingagents/dataflows/test_short_term_structure.py
git commit -m "feat: expose short-term structure in stock profile"
```

### Task 4: Agent Consumption and M3 Output Contract

**Files:**
- Modify: `tradingagents/agents/analysts/market_analyst.py`
- Modify: `tradingagents/agents/managers/research_manager.py`
- Modify: `tradingagents/agents/managers/portfolio_manager.py`
- Modify: `tradingagents/agents/managers/test_entry_timing.py`

**Interfaces:**
- Consumes: `SYS_SHORT_TERM_STRUCTURE`, `market_mode`, and the deterministic `compute_entry_timing` helper.
- Produces: consistent `entry_timing` in RM output and PM Trade Ticket, separate from long-term rating.

- [ ] **Step 1: Write failing prompt-contract tests**

Add source-level contract assertions that all three agents mention `SYS_SHORT_TERM_STRUCTURE`, that RM and PM require `entry_timing`, and that prompts explicitly forbid changing long-term rating from the short-term signal.

- [ ] **Step 2: Verify RED**

Run: `.venv/bin/python -m pytest tradingagents/agents/managers/test_entry_timing.py -q`

Expected: prompt-contract assertions fail.

- [ ] **Step 3: Update agent contracts**

Market Analyst must cite the deterministic class and numeric evidence. RM must calculate long-term rating first and expose a separate entry-timing field. PM must show long-term rating, entry timing, trigger, and invalidation in the Trade Ticket. Keep existing M3 neutral-language safeguards and preserve current uncommitted changes in `market_analyst.py`.

- [ ] **Step 4: Add deterministic output fallback**

Where the RM/PM prompt currently requires entry timing, make the deterministic action authoritative in the prompt context so missing or malformed M3 prose cannot promote an action. Do not add a new LLM retry path; existing output-sanitization and compliance retries remain unchanged.

- [ ] **Step 5: Verify GREEN**

Run: `.venv/bin/python -m pytest tradingagents/agents/managers/test_entry_timing.py tradingagents/agents/managers/test_step6_final_rating.py tradingagents/agents/managers/test_market_risk_gate.py -q`

Expected: all tests pass.

- [ ] **Step 6: Commit agent integration**

```bash
git add tradingagents/agents/analysts/market_analyst.py tradingagents/agents/managers/research_manager.py tradingagents/agents/managers/portfolio_manager.py tradingagents/agents/managers/test_entry_timing.py
git commit -m "feat: surface deterministic entry timing in agent reports"
```

### Task 5: Full Regression and One-Stock Closure

**Files:**
- Modify only files required by failures directly caused by Tasks 1-4.

**Interfaces:**
- Consumes: completed classifier, gate, profile line, and agent contracts.
- Produces: regression evidence from `.venv` and one complete M3 report.

- [ ] **Step 1: Run focused regression suite**

Run:

```bash
.venv/bin/python -m pytest \
  tradingagents/dataflows/test_short_term_structure.py \
  tradingagents/dataflows/test_valuation_regime.py \
  tradingagents/agents/managers/test_entry_timing.py \
  tradingagents/agents/managers/test_step6_final_rating.py \
  tradingagents/agents/managers/test_market_risk_gate.py -q
```

Expected: all tests pass with no warnings introduced by this feature.

- [ ] **Step 2: Run one complete M3 stock analysis**

Use the existing `.venv` analysis entry point and current default MiniMax-M3 configuration on one A-share stock. Wait through normal long-running periods; inspect logs before terminating, and terminate only after evidence of a genuine abnormal hang.

- [ ] **Step 3: Inspect the generated report**

Confirm the profile contains `SYS_SHORT_TERM_STRUCTURE`, RM and PM agree on the structure-derived action, Trade Ticket separates long-term rating from entry timing, market/fundamental vetoes are respected, and no MiniMax moderation or malformed-output failure prevented report completion.

- [ ] **Step 4: Final verification commit if needed**

Only if regression fixes were necessary, stage the feature-owned files after reviewing `git diff`:

```bash
git add tradingagents/dataflows/profile_calc.py tradingagents/dataflows/test_short_term_structure.py tradingagents/agents/utils/stock_profile_node.py tradingagents/agents/managers/rm_tools.py tradingagents/agents/managers/test_entry_timing.py tradingagents/agents/analysts/market_analyst.py tradingagents/agents/managers/research_manager.py tradingagents/agents/managers/portfolio_manager.py
git commit -m "fix: close entry timing regression gaps"
```
