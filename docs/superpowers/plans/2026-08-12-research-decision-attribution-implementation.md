# Research Decision Attribution Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a deterministic evidence ledger and IC packet so RM/PM decisions cite valid analyst evidence and `decision.md` visibly attributes short-term, long-term, position, and target conclusions.

**Architecture:** A pure-Python Research Evidence Officer parses existing YAML summaries into stable evidence cards, validates freshness and role permissions, identifies conflicts, and renders a compact IC packet. The LangGraph state carries the ledger and packet into researchers, RM, and PM; PM evidence references are validated and rendered into the final report without changing any investment calculation.

**Tech Stack:** Python 3.11+, LangGraph state nodes, PyYAML, pytest-style function tests executed with the project `.venv`.

## Global Constraints

- Do not add a normal-path LLM stage, network call, or external dependency. Existing bounded RM/PM continuation may retry only when output is empty, truncated, or unparsable.
- Do not change rating, valuation, market-risk, position-sizing, or entry-timing calculations.
- Invalid, stale, missing, or unauthorized evidence must be visible and must never be silently replaced.
- `PM_SUMMARY` remains the final YAML block and existing harness fields retain their semantics.
- Missing attribution degrades reporting only; it must not abort stock analysis or mutate the decision.
- Preserve existing checkpoints by defaulting new state fields to empty values.

---

### Task 1: Evidence Ledger and IC Packet

**Files:**
- Create: `tradingagents/agents/utils/research_evidence_node.py`
- Create: `tradingagents/agents/utils/test_research_evidence_node.py`

**Interfaces:**
- Consumes: `compile_research_evidence(state: Mapping[str, Any]) -> dict[str, Any]`
- Produces: `render_ic_packet(ledger: Mapping[str, Any], *, ticker: str, company_name: str, trade_date: str) -> str` and `create_research_evidence_node() -> Callable[[Mapping[str, Any]], dict[str, Any]]`
- Ledger schema: `{"cards": list[dict], "conflicts": list[dict], "warnings": list[str], "coverage": dict[str, str]}`

- [ ] **Step 1: Write failing tests for stable cards and quality states**

```python
def test_compiler_builds_stable_market_and_fundamental_cards():
    ledger = compile_research_evidence(_complete_state())
    by_id = {card["claim_id"]: card for card in ledger["cards"]}
    assert by_id["MKT-TREND-01"]["decision_variable"] == "short_term_trend"
    assert by_id["FUND-GROWTH-01"]["decision_variable"] == "earnings_outlook_12m"

def test_unknown_news_date_and_t_minus_one_price_are_partial():
    ledger = compile_research_evidence(_state_with_partial_dates())
    by_id = {card["claim_id"]: card for card in ledger["cards"]}
    assert by_id["NEWS-CAT-01"]["quality_status"] == "partial"
    assert by_id["MKT-TREND-01"]["quality_status"] == "partial"
```

- [ ] **Step 2: Run the focused test and verify RED**

Run: `../../.venv/bin/python -m pytest tradingagents/agents/utils/test_research_evidence_node.py -q`

Expected: FAIL because `research_evidence_node` does not exist.

- [ ] **Step 3: Implement YAML parsing and evidence-card compilation**

Implement `_extract_yaml_mapping`, direction mapping, stable card builders for market/fundamentals/news/sentiment/quant/sector/risk, event de-duplication, and explicit `valid/partial/stale/invalid` quality states. All returned objects must be JSON serializable.

- [ ] **Step 4: Add failing tests for conflict detection and IC rendering**

```python
def test_opposite_long_term_directions_create_a_conflict_without_resolving_it():
    ledger = compile_research_evidence(_conflicting_state())
    assert ledger["conflicts"][0]["decision_variable"] == "long_term_rating"
    assert set(ledger["conflicts"][0]["directions"]) == {"bullish", "bearish"}

def test_ic_packet_lists_warnings_conflicts_and_evidence_ids():
    packet = render_ic_packet(ledger, ticker="688114", company_name="华大智造", trade_date="2026-08-12")
    assert "## 数据完整度" in packet
    assert "## 冲突清单" in packet
    assert "MKT-TREND-01" in packet
```

- [ ] **Step 5: Run the focused test and verify RED**

Run: `../../.venv/bin/python -m pytest tradingagents/agents/utils/test_research_evidence_node.py -q`

Expected: FAIL on conflict or IC rendering behavior.

- [ ] **Step 6: Implement conflict detection, permission policy, and IC rendering**

Define `ATTRIBUTION_OWNER_POLICY` for `short_term`, `long_term`, `position`, and `target_price`. Conflict detection compares usable bullish/bearish cards for the same decision variable but never chooses a winner.

- [ ] **Step 7: Run focused tests and commit**

Run: `../../.venv/bin/python -m pytest tradingagents/agents/utils/test_research_evidence_node.py -q`

Expected: PASS.

Commit: `feat: compile analyst evidence into ic packet`

---

### Task 2: Graph, State, Logging, and Report Persistence

**Files:**
- Modify: `tradingagents/agents/utils/agent_states.py`
- Modify: `tradingagents/graph/propagation.py`
- Modify: `tradingagents/graph/setup.py`
- Modify: `tradingagents/graph/trading_graph.py`
- Modify: `main.py`
- Create: `tradingagents/graph/test_research_evidence_integration.py`

**Interfaces:**
- Consumes: `create_research_evidence_node()` from Task 1.
- Produces state fields `research_evidence_ledger: dict` and `ic_packet: str`.

- [ ] **Step 1: Write failing state and graph-order tests**

```python
def test_initial_state_contains_empty_research_evidence_fields(monkeypatch):
    state = Propagator().create_initial_state("688114", "2026-08-12")
    assert state["research_evidence_ledger"] == {}
    assert state["ic_packet"] == ""

def test_graph_routes_consensus_through_research_evidence_before_bull():
    source = Path("tradingagents/graph/setup.py").read_text()
    assert 'workflow.add_edge("Consensus Officer", "Research Evidence Officer")' in source
    assert 'workflow.add_edge("Research Evidence Officer", "Bull Researcher")' in source
```

- [ ] **Step 2: Run integration test and verify RED**

Run: `../../.venv/bin/python -m pytest tradingagents/graph/test_research_evidence_integration.py -q`

Expected: FAIL because state fields and graph node are absent.

- [ ] **Step 3: Add the deterministic node to state and graph**

Add annotated state fields, initialize them for fresh and checkpoint-compatible runs, instantiate/add the node, and replace `Consensus -> Bull` with `Consensus -> Research Evidence -> Bull`.

- [ ] **Step 4: Persist the new artifacts**

Add `ic_packet` to progress labels/order, state logging, returned final-state views, and `1_analysts/ic_packet.md`. The raw ledger remains in the JSON state log, not as a separate user report.

- [ ] **Step 5: Run tests and commit**

Run: `../../.venv/bin/python -m pytest tradingagents/graph/test_research_evidence_integration.py tradingagents/agents/utils/test_research_evidence_node.py -q`

Expected: PASS.

Commit: `feat: wire research evidence officer into graph`

---

### Task 3: RM and PM Decision Rights

**Files:**
- Modify: `tradingagents/agents/researchers/bull_researcher.py`
- Modify: `tradingagents/agents/researchers/bear_researcher.py`
- Modify: `tradingagents/agents/managers/research_manager.py`
- Modify: `tradingagents/agents/managers/portfolio_manager.py`
- Create: `tradingagents/agents/managers/test_decision_attribution_prompts.py`

**Interfaces:**
- Consumes: `state["ic_packet"]` and stable evidence IDs.
- Produces RM fields `rating_evidence_ids`, `target_price_evidence_ids`, `earnings_evidence_ids`, `key_conflict_ids`; PM fields `short_term_evidence_ids`, `long_term_evidence_ids`, `position_evidence_ids`, `target_price_evidence_ids`.

- [ ] **Step 1: Write failing prompt-contract tests**

```python
def test_rm_and_researchers_receive_ic_packet_and_evidence_rules():
    for path in RM_AND_RESEARCHER_PATHS:
        source = path.read_text(encoding="utf-8")
        assert 'state.get("ic_packet", "")' in source
        assert "证据 ID" in source

def test_pm_uses_ic_packet_instead_of_four_full_analyst_reports():
    source = PM_PATH.read_text(encoding="utf-8")
    assert 'state.get("ic_packet", "")' in source
    assert "### 4 个 analyst 原始报告" not in source
    assert "short_term_evidence_ids" in source
```

- [ ] **Step 2: Run prompt tests and verify RED**

Run: `../../.venv/bin/python -m pytest tradingagents/agents/managers/test_decision_attribution_prompts.py -q`

Expected: FAIL because IC packet and ID fields are absent.

- [ ] **Step 3: Inject IC packet into Bull/Bear and RM**

Make the packet the primary evidence index. Keep the four raw reports in Bull/Bear and RM as conflict drill-down material for P0. Require every material argument and RM summary decision to cite existing IDs.

- [ ] **Step 4: Replace PM raw-report injection with IC packet**

Use `research_plan + ic_packet` for memory retrieval and PM context. Retain stock profile, quant, sector, consensus, risk debate, and prior lessons. Add the four attribution fields to `PM_SUMMARY` without changing existing fields.

- [ ] **Step 5: Run prompt and existing manager tests, then commit**

Run: `../../.venv/bin/python -m pytest tradingagents/agents/managers/test_decision_attribution_prompts.py tradingagents/agents/managers/test_entry_timing.py tradingagents/agents/managers/test_market_risk_gate.py -q`

Expected: PASS.

Commit: `feat: enforce evidence references in rm and pm`

---

### Task 4: Deterministic Decision Attribution Rendering

**Files:**
- Modify: `tradingagents/agents/utils/research_evidence_node.py`
- Modify: `tradingagents/agents/managers/portfolio_manager.py`
- Modify: `tradingagents/agents/managers/test_entry_timing.py`
- Modify: `tradingagents/harness/test_archive_market_risk.py`

**Interfaces:**
- Consumes: `render_decision_attribution(pm_content: str, timing: Mapping[str, Any], ledger: Mapping[str, Any]) -> str`.
- Produces: Markdown `## 为什么这样决定` table with final conclusion, accountable team, evidence IDs, and validation state.

- [ ] **Step 1: Write failing renderer tests**

```python
def test_attribution_renderer_marks_valid_partial_missing_and_unauthorized_refs():
    table = render_decision_attribution(_pm_summary(), _timing(), _ledger())
    assert "| 未来三日 | 等回踩 |" in table
    assert "完整" in table
    assert "部分：证据不完整" in table
    assert "权限不匹配" in table
    assert "缺失：PM 未完成证据归因" in table

def test_pm_formatter_inserts_attribution_before_trade_ticket_and_preserves_yaml():
    result = _format_pm_decision(content, timing, research_evidence_ledger=ledger)
    assert result.index("## 为什么这样决定") < result.index("## Trade Ticket")
    assert _find_yaml_block(result, "PM_SUMMARY")["pm_rating"] == "OVERWEIGHT"
```

- [ ] **Step 2: Run renderer tests and verify RED**

Run: `../../.venv/bin/python -m pytest tradingagents/agents/utils/test_research_evidence_node.py tradingagents/agents/managers/test_entry_timing.py -q`

Expected: FAIL because rendering and formatter integration are absent.

- [ ] **Step 3: Implement evidence-reference validation and table rendering**

Parse pipe/comma-separated IDs from the four PM fields. For each row, verify ID existence, quality, and owner permission. Render the final short-term action, long-term rating, new-position size, and TP1-TP3 range from deterministic PM values.

- [ ] **Step 4: Integrate into PM output without changing fallback behavior**

Pass `state["research_evidence_ledger"]` into `_format_pm_decision`. Insert the table only when a ledger object is provided; legacy calls without a ledger keep existing formatting. Preserve the final YAML fence and archive extraction.

- [ ] **Step 5: Run focused regression and commit**

Run: `../../.venv/bin/python -m pytest tradingagents/agents/utils/test_research_evidence_node.py tradingagents/agents/managers/test_entry_timing.py tradingagents/harness/test_archive_market_risk.py -q`

Expected: PASS.

Commit: `feat: render auditable decision attribution`

---

### Task 5: Full Verification and Stock Smoke Test

**Files:**
- Modify only if verification reveals a task-related defect.

**Interfaces:**
- Consumes all deliverables from Tasks 1-4.
- Produces a verified report and GitHub integration.

- [ ] **Step 1: Run the attribution and manager suites**

Run: `../../.venv/bin/python -m pytest tradingagents/agents/utils/test_research_evidence_node.py tradingagents/graph/test_research_evidence_integration.py tradingagents/agents/managers/test_decision_attribution_prompts.py tradingagents/agents/managers/test_entry_timing.py tradingagents/agents/managers/test_market_risk_gate.py tradingagents/harness/test_archive_market_risk.py -q`

Expected: PASS with zero failures.

- [ ] **Step 2: Run the broader existing regression suites**

Run: `../../.venv/bin/python -m pytest tradingagents/agents/managers tradingagents/dataflows tradingagents/harness -q`

Expected: PASS; unrelated external-data tests may be reported separately only if they demonstrably predate this branch.

- [ ] **Step 3: Run one A-share smoke analysis in `.venv`**

Run the project entry point for a recently used A-share ticker. Do not kill the process merely for running longer than five minutes; inspect logs and wait unless behavior is clearly abnormal.

Expected: a new report directory with `1_analysts/ic_packet.md` and `5_portfolio/decision.md`.

- [ ] **Step 4: Inspect the generated report**

Verify price timestamp, four attribution rows, evidence validity, consistency with Trade Ticket/`PM_SUMMARY`, final YAML extraction, and absence of working COT before the decision card.

- [ ] **Step 5: Review diff, commit residual fixes, push, open PR, and merge**

Run: `git diff --check`, `git status --short`, and inspect the final diff. Push `codex/research-decision-attribution`, create a ready PR, merge after checks pass, and update local `main` without touching the user's dirty main worktree.

Expected: remote `main` contains the feature and the original main worktree's dirty files remain unchanged.
