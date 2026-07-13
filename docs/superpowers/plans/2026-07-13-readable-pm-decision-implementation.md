# PM Decision Readability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make every new `decision.md` start with an attention-grabbing deterministic short-term action summary and remove model working text before the Trade Ticket.

**Architecture:** Add pure formatting helpers in `portfolio_manager.py` and invoke them once at the PM output boundary after deterministic timing enforcement. Keep persistence, Feishu delivery, and harness extraction unchanged because they consume the same finalized string.

**Tech Stack:** Python 3.13, Markdown, regex, existing executable test scripts.

## Global Constraints

- Do not change rating, timing, sizing, or market-risk calculations.
- Preserve `PM_SUMMARY` at the report end with unchanged fields.
- Use standard Markdown only.
- Historical reports are not rewritten.

---

### Task 1: Deterministic Decision Presentation

**Files:**
- Modify: `tradingagents/agents/managers/portfolio_manager.py`
- Test: `tradingagents/agents/managers/test_entry_timing.py`

**Interfaces:**
- Produces: `_format_pm_decision(content: str, timing: dict) -> str`
- Consumes: finalized PM content after `_enforce_entry_timing_truth` and the deterministic timing dictionary.

- [ ] **Step 1: Write failing tests**

Add tests proving that process text before `## Trade Ticket` is removed, the first line is `# 短期操作结论：暂不介入`, rating/action/size are extracted from `PM_SUMMARY`, and reports without a Trade Ticket are retained.

- [ ] **Step 2: Run tests and verify failure**

Run: `.venv/bin/python tradingagents/agents/managers/test_entry_timing.py`

Expected: import or assertion failure because `_format_pm_decision` does not exist.

- [ ] **Step 3: Implement the pure formatter**

Implement helpers that locate the first Trade Ticket heading, extract flat YAML-like summary fields without introducing a YAML dependency, render fixed reason/trigger templates from `entry_timing`, and prepend the Markdown summary.

- [ ] **Step 4: Apply formatter at the PM output boundary**

After `_enforce_entry_timing_truth`, call `_format_pm_decision` before assigning `judge_decision` and `final_trade_decision`.

- [ ] **Step 5: Run focused tests**

Run: `.venv/bin/python tradingagents/agents/managers/test_entry_timing.py`

Expected: all tests pass.

### Task 2: Archive Compatibility and Real Report Replay

**Files:**
- Test: `tradingagents/harness/test_archive_market_risk.py`
- Read: `reports/300308_中际旭创_20260713_092309/5_portfolio/decision.md`

**Interfaces:**
- Consumes: `_format_pm_decision(content, timing)` from Task 1.
- Produces: evidence that formatted reports remain archive-compatible.

- [ ] **Step 1: Add archive compatibility coverage**

Format a report containing `PM_SUMMARY`, parse it with `_find_yaml_block`, and assert rating and entry timing remain available.

- [ ] **Step 2: Run focused harness tests**

Run: `.venv/bin/python tradingagents/harness/test_archive_market_risk.py`

Expected: all tests pass.

- [ ] **Step 3: Replay the latest real report offline**

Run the formatter against the latest `300308` decision and verify the first line, absence of process preamble, preserved Trade Ticket, and preserved `PM_SUMMARY`.

- [ ] **Step 4: Run the complete relevant regression set**

Run entry timing, final rating, market risk, M3 compliance, and archive tests already used by this feature.

- [ ] **Step 5: Commit, push, open PR, merge, and sync `main`**

Commit only the spec, plan, formatter, and tests. Preserve unrelated local files.
