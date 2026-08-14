# DeepSeek V4 And Four-Pillar IC Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Migrate every active LLM role to explicit DeepSeek V4 Think routing and replace valuation-first stock ratings with an auditable four-pillar investment-committee recommendation that is separate from the current trade action.

**Architecture:** Specialist agents continue to create auditable reports, while the evidence officer validates claims and compiles a bounded IC packet. The RM classifies four research pillars and calls a deterministic recommendation tool; the PM inherits that research rating and applies timing, market-risk, and position constraints. DeepSeek model selection is resolved by role through one policy module, and raw reasoning never crosses an agent boundary.

**Tech Stack:** Python 3.11+, LangChain/LangGraph, `langchain-openai`, pytest-style executable tests, YAML summaries, existing TradingAgents graph and `.venv`.

## Global Constraints

- Use `https://api.deepseek.com` with `deepseek-v4-pro` and `deepseek-v4-flash`.
- Explicitly enable Think for every DeepSeek call; use `high` normally and `max` for RM, PM, and the market-warning slow path.
- Never persist or pass raw `reasoning_content` across agents, reports, databases, or `llm_calls`.
- Do not automatically fall back from DeepSeek to MiniMax.
- LLMs classify evidence and explain causal judgments; Python validates evidence, calculates numbers, maps ratings, and enforces risk gates.
- Preserve the existing deterministic market-warning level, probability, push, and position rules.
- Keep unrelated dirty files in the main workspace untouched; implement only in the isolated `codex/deepseek-v4-routing` worktree.

---

### Task 1: Deterministic Four-Pillar Recommendation Engine

**Files:**
- Create: `tradingagents/agents/managers/ic_recommendation.py`
- Create: `tradingagents/agents/managers/test_ic_recommendation.py`
- Modify: `tradingagents/agents/managers/rm_tools.py`

**Interfaces:**
- Produces: `compute_ic_recommendation(...) -> dict` as a LangChain tool.
- Returns: `research_rating`, `rating_reason_codes`, `scenario_expected_return_pct`, `thresholds`, `pillar_effects`, `bounds`, and `explanation`.
- Reuses: existing style/theme dynamic thresholds, evidence-quality semantics, and five-level rating order.

- [ ] **Step 1: Write failing tests for positive, neutral, negative, and override paths**

```python
def test_buy_requires_return_thesis_catalyst_and_no_veto():
    result = invoke_ic(expected_return=45, thesis="strong", catalyst="visible")
    assert result["research_rating"] == "BUY"

def test_cheap_stock_with_weak_thesis_cannot_be_positive():
    result = invoke_ic(expected_return=30, thesis="weak", catalyst="visible")
    assert result["research_rating"] == "HOLD"

def test_long_term_positive_rating_is_independent_of_short_term_timing():
    result = invoke_ic(expected_return=22, thesis="adequate", catalyst="visible")
    assert result["research_rating"] == "OVERWEIGHT"

def test_verified_thesis_breaker_can_override_stale_positive_return():
    result = invoke_ic(expected_return=5, thesis="weak", durability="broken", hard_veto=True)
    assert result["research_rating"] == "SELL"
    assert "THESIS_BREAK_OVERRIDE" in result["rating_reason_codes"]
```

- [ ] **Step 2: Run the new test file and verify it fails because the module does not exist**

Run: `.venv/bin/python tradingagents/agents/managers/test_ic_recommendation.py`

Expected: import failure for `ic_recommendation`.

- [ ] **Step 3: Implement strict enums, input validation, dynamic thresholds, and the rating matrix**

```python
@tool
def compute_ic_recommendation(
    scenario_expected_return_pct: float,
    downside_pct: float,
    payoff_ratio: float,
    thesis_state: str,
    thesis_direction: str,
    valuation_state: str,
    catalyst_state: str,
    priced_in: str,
    durability_state: str,
    hard_veto: bool,
    evidence_quality: str,
    style: str = "",
    theme_premium_pct: float = 0.0,
    theme_stage: str = "",
    crowded_long: bool = False,
) -> dict:
    """Return one deterministic 12-month IC recommendation."""
```

The implementation must reject invalid enum values, calculate style-aware positive and negative thresholds, apply pillar gates, apply evidence/crowding caps, and emit stable reason codes. It must not consume entry timing, market-risk action, or PM position fields.

- [ ] **Step 4: Register the new tool and run old plus new rating tests**

Run: `.venv/bin/python tradingagents/agents/managers/test_ic_recommendation.py`

Run: `.venv/bin/python tradingagents/agents/managers/test_step6_final_rating.py`

Expected: both pass; the old tool remains available for compatibility but is no longer selected by the RM prompt in Task 2.

- [ ] **Step 5: Commit the rating engine**

```bash
git add tradingagents/agents/managers/ic_recommendation.py tradingagents/agents/managers/test_ic_recommendation.py tradingagents/agents/managers/rm_tools.py
git commit -m "feat: add four-pillar IC recommendation engine"
```

---

### Task 2: Research Manager Four-Pillar Contract

**Files:**
- Modify: `tradingagents/agents/managers/research_manager.py`
- Modify: `tradingagents/agents/managers/test_decision_attribution_prompts.py`
- Modify: `tradingagents/agents/managers/test_entry_timing.py`

**Interfaces:**
- Consumes: `compute_ic_recommendation` and the existing `ic_packet`.
- Produces: `RM_SUMMARY` fields `research_rating`, `pillar_thesis`, `pillar_valuation`, `pillar_catalyst`, `pillar_durability`, `scenario_expected_return_pct`, `rating_reason_codes`, and per-pillar evidence IDs.
- Keeps compatibility: `rm_rating` mirrors `research_rating` during the transition.

- [ ] **Step 1: Add failing source-contract and summary-normalization tests**

```python
def test_rm_uses_four_pillar_tool_as_only_rating_authority():
    source = Path("tradingagents/agents/managers/research_manager.py").read_text()
    assert "compute_ic_recommendation" in source
    assert "最终权威评级不得调用 compute_step6_final_rating" in source

def test_rm_summary_preserves_four_pillar_fields_after_retry_merge():
    result = normalize_rm_summary(partial_then_complete_messages)
    summary = _find_yaml_block(result, "RM_SUMMARY")
    assert summary["research_rating"] == summary["rm_rating"]
    assert summary["pillar_thesis"] == "strong"
```

- [ ] **Step 2: Run focused tests and confirm the new assertions fail**

Run: `.venv/bin/python tradingagents/agents/managers/test_decision_attribution_prompts.py`

Run: `.venv/bin/python tradingagents/agents/managers/test_entry_timing.py`

- [ ] **Step 3: Replace the RM rating stage and compact the prompt**

Require the RM to classify each pillar with accepted evidence IDs, call `compute_ic_recommendation` exactly once, copy the returned rating and reason codes, and keep target prices as scenario inputs rather than the initial rating. Remove narrative that says market/news direction votes form the long-term rating.

- [ ] **Step 4: Extend `RM_SUMMARY` without breaking archive readers**

```yaml
research_rating: OVERWEIGHT
rm_rating: OVERWEIGHT
pillar_thesis: adequate
pillar_valuation: attractive
pillar_catalyst: visible
pillar_durability: acceptable
scenario_expected_return_pct: 18.4
rating_reason_codes: THESIS_OK|EXPECTED_RETURN_POSITIVE|CATALYST_VISIBLE
thesis_evidence_ids: FUND-GROWTH-01
valuation_evidence_ids: FUND-VAL-01
catalyst_evidence_ids: NEWS-CAT-01
durability_evidence_ids: RISK-GATE-01
```

- [ ] **Step 5: Run RM, entry-timing, and extractor tests**

Run: `.venv/bin/python tradingagents/agents/managers/test_decision_attribution_prompts.py`

Run: `.venv/bin/python tradingagents/agents/managers/test_entry_timing.py`

Run: `.venv/bin/python tradingagents/harness/test_archive_market_risk.py`

- [ ] **Step 6: Commit RM integration**

```bash
git add tradingagents/agents/managers/research_manager.py tradingagents/agents/managers/test_decision_attribution_prompts.py tradingagents/agents/managers/test_entry_timing.py
git commit -m "feat: make RM use four-pillar recommendation"
```

---

### Task 3: Decision Contribution Ledger And PM Read-Only Rating

**Files:**
- Modify: `tradingagents/agents/utils/research_evidence_node.py`
- Modify: `tradingagents/agents/utils/test_research_evidence_node.py`
- Modify: `tradingagents/agents/managers/portfolio_manager.py`
- Modify: `tradingagents/agents/managers/test_decision_attribution_prompts.py`
- Modify: `tradingagents/agents/managers/test_entry_timing.py`

**Interfaces:**
- Produces: `compile_decision_contribution_ledger(rm_content, evidence_ledger) -> dict`.
- Produces: `render_ic_contribution_summary(rm_content, contribution_ledger) -> str`.
- PM consumes `research_rating` as read-only and may only produce `trade_action`, size, entry, exits, and monitoring.

- [ ] **Step 1: Write failing tests for accepted, rejected, duplicate, and unauthorized contributions**

```python
def test_contribution_ledger_maps_each_claim_once_to_one_pillar():
    ledger = compile_decision_contribution_ledger(_rm_four_pillars(), evidence)
    assert [row["claim_id"] for row in ledger["items"]].count("NEWS-CAT-01") == 1

def test_pm_cannot_change_rm_research_rating():
    result = _format_pm_decision(_pm_summary(pm_rating="SELL"), research_plan=_rm_rating("OVERWEIGHT"))
    assert _find_yaml_block(result, "PM_SUMMARY")["pm_rating"] == "OVERWEIGHT"
```

- [ ] **Step 2: Run evidence and PM tests and verify the assertions fail**

Run: `.venv/bin/python tradingagents/agents/utils/test_research_evidence_node.py`

Run: `.venv/bin/python tradingagents/agents/managers/test_entry_timing.py`

- [ ] **Step 3: Compile the ledger deterministically from RM fields and eligible cards**

Reject unknown IDs, invalid cards, wrong-owner/wrong-dimension references, and duplicate claim reuse. Record `accepted_effect`, `quality_status`, and a concrete `rejection_reason`; do not infer missing evidence as neutral.

- [ ] **Step 4: Make PM rating read-only and update the prompt contract**

Remove the `±1` rating adjustment authority. PM must use the RM research rating, while the existing market-risk gate and entry-timing truth remain authoritative for execution.

- [ ] **Step 5: Render the user-facing decision header and contribution summary**

```markdown
# 短期操作结论：等待条件确认

> **一年期研究评级：OVERWEIGHT｜当前动作：WAIT｜新建仓位：0%**
>
> **四支柱：经营与盈利 adequate｜估值 attractive｜催化 visible｜持续性 acceptable**
>
> **暂不买入原因：市场风险门限制新增仓位**
```

- [ ] **Step 6: Run focused PM/evidence/report tests**

Run: `.venv/bin/python tradingagents/agents/utils/test_research_evidence_node.py`

Run: `.venv/bin/python tradingagents/agents/managers/test_entry_timing.py`

Run: `.venv/bin/python tests/test_report_artifacts.py`

- [ ] **Step 7: Commit contribution and PM changes**

```bash
git add tradingagents/agents/utils/research_evidence_node.py tradingagents/agents/utils/test_research_evidence_node.py tradingagents/agents/managers/portfolio_manager.py tradingagents/agents/managers/test_decision_attribution_prompts.py tradingagents/agents/managers/test_entry_timing.py
git commit -m "feat: expose agent contributions in PM decisions"
```

---

### Task 4: Role-Specific Handoff Contracts And Context Bounds

**Files:**
- Modify: `tradingagents/agents/analysts/market_analyst.py`
- Modify: `tradingagents/agents/analysts/fundamentals_analyst.py`
- Modify: `tradingagents/agents/analysts/news_analyst.py`
- Modify: `tradingagents/agents/analysts/social_media_analyst.py`
- Modify: `tradingagents/agents/utils/macro_context_node.py`
- Modify: `tradingagents/agents/utils/stock_profile_node.py`
- Modify: `tradingagents/agents/utils/consensus_node.py`
- Modify: `tradingagents/agents/researchers/bull_researcher.py`
- Modify: `tradingagents/agents/researchers/bear_researcher.py`
- Modify: `tradingagents/agents/risk_mgmt/aggressive_debator.py`
- Modify: `tradingagents/agents/risk_mgmt/neutral_debator.py`
- Modify: `tradingagents/agents/risk_mgmt/conservative_debator.py`
- Create: `tradingagents/agents/utils/handoff.py`
- Create: `tradingagents/agents/utils/test_handoff.py`

**Interfaces:**
- Produces: `extract_handoff(report, role) -> dict` and `pack_agent_context(...) -> str`.
- Enforces: role schema, role character budget, evidence/date priority, and no `reasoning_content`/`<think>` leakage.

- [ ] **Step 1: Write failing schema, permission, and budget tests**

```python
def test_market_handoff_rejects_long_term_target_price():
    result = extract_handoff(market_report_with_target_price, "market")
    assert result["quality"]["status"] == "invalid"

def test_context_packer_keeps_hard_constraints_before_narrative():
    packed = pack_agent_context(items, budget_chars=1200)
    assert "RISK-GATE-01" in packed
    assert "reasoning_content" not in packed
```

- [ ] **Step 2: Run the new tests and verify missing-module failures**

Run: `.venv/bin/python tradingagents/agents/utils/test_handoff.py`

- [ ] **Step 3: Implement the common envelope plus role validators**

Use the exact `HANDOFF` and `DECISION_HANDOFF` fields from the approved spec. Enforce role permissions in Python; malformed handoffs become `partial` or `invalid` and never abort unrelated evidence domains.

- [ ] **Step 4: Add concise handoff blocks to every active role prompt**

Keep the existing full Markdown report for artifacts, but instruct downstream prompts to consume the bounded handoff/IC packet. Bull/Bear and risk roles may cite formal evidence IDs but may not introduce new facts.

- [ ] **Step 5: Run handoff, evidence, and graph integration tests**

Run: `.venv/bin/python tradingagents/agents/utils/test_handoff.py`

Run: `.venv/bin/python tradingagents/agents/utils/test_research_evidence_node.py`

Run: `.venv/bin/python tradingagents/graph/test_research_evidence_integration.py`

- [ ] **Step 6: Commit handoff contracts**

```bash
git add tradingagents/agents
git commit -m "feat: add role-specific research handoffs"
```

---

### Task 5: DeepSeek Client, Defaults, And Role Routing

**Files:**
- Create: `tradingagents/llm_clients/deepseek_client.py`
- Create: `tradingagents/llm_clients/role_policy.py`
- Create: `tradingagents/llm_clients/test_deepseek_client.py`
- Modify: `tradingagents/llm_clients/factory.py`
- Modify: `tradingagents/llm_clients/model_catalog.py`
- Modify: `tradingagents/default_config.py`
- Modify: `tradingagents/graph/trading_graph.py`
- Modify: `tradingagents/graph/setup.py`
- Modify: `main.py`
- Modify: `cli/utils.py`
- Modify: `cli/main.py`
- Modify: `.env.example`
- Modify: `tests/test_model_validation.py`

**Interfaces:**
- Produces: `DeepSeekClient` using `DEEPSEEK_API_KEY` and `https://api.deepseek.com`.
- Produces: `resolve_role_policy(config, role) -> RoleLLMPolicy(model, reasoning_effort, max_tokens)`.
- TradingGraph creates role LLMs through `_create_role_llm(role, temperature)`.

- [ ] **Step 1: Write failing provider and routing tests**

```python
def test_deepseek_client_enables_think_explicitly(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test-deepseek")
    llm = DeepSeekClient("deepseek-v4-pro").get_llm()
    assert llm.extra_body["thinking"] == {"type": "enabled"}

def test_role_policy_routes_rm_to_pro_max_and_news_to_flash_high():
    assert resolve_role_policy(config, "research_manager").model == "deepseek-v4-pro"
    assert resolve_role_policy(config, "research_manager").reasoning_effort == "max"
    assert resolve_role_policy(config, "news").model == "deepseek-v4-flash"
```

- [ ] **Step 2: Run provider tests and verify they fail on missing classes/provider**

Run: `.venv/bin/python tradingagents/llm_clients/test_deepseek_client.py`

Run: `.venv/bin/python tests/test_model_validation.py`

- [ ] **Step 3: Implement the client and explicit Think payload**

Use Chat Completions through `NormalizedChatOpenAI`, pass `thinking={"type":"enabled"}` and role effort through the provider request body, forward timeouts/retries/max tokens, and raise immediately when the key is missing. Log only model, duration, token usage, and final-content length for DeepSeek calls.

- [ ] **Step 4: Implement role policy and graph wiring**

Route market/fundamentals/macro/profile to Pro high; RM/PM to Pro max; news/sentiment/consensus/Bull/Bear/risk/signal/reflection to Flash high. Add explicit macro/profile/consensus LLM arguments to `GraphSetup` so they no longer silently use the global quick instance.

- [ ] **Step 5: Make DeepSeek the default in batch and CLI configuration**

Set `llm_provider=deepseek`, `deep_think_llm=deepseek-v4-pro`, `quick_think_llm=deepseek-v4-flash`, and the official base URL. Preserve MiniMax as an explicitly selected provider only.

- [ ] **Step 6: Add reasoning-content boundary tests**

Verify tool-call subturn messages keep provider `reasoning_content`, while `normalize_content`, handoff packing, reports, and persisted state contain no raw reasoning or `<think>` blocks.

- [ ] **Step 7: Run provider, graph, configuration, and leakage tests**

Run: `.venv/bin/python tradingagents/llm_clients/test_deepseek_client.py`

Run: `.venv/bin/python tests/test_model_validation.py`

Run: `.venv/bin/python -m compileall tradingagents cli main.py`

- [ ] **Step 8: Commit the DeepSeek migration**

```bash
git add tradingagents/llm_clients tradingagents/default_config.py tradingagents/graph main.py cli .env.example tests/test_model_validation.py
git commit -m "feat: route active agents through DeepSeek V4"
```

---

### Task 6: Market-Warning Slow Path Migration

**Files:**
- Create: `tradingagents/harness/market_warning/adapters/deepseek_reasoning.py`
- Modify: `tradingagents/harness/market_warning/runner.py`
- Modify: `tradingagents/harness/market_warning/reasoning.py`
- Modify: `tradingagents/harness/market_warning/service.py`
- Modify: `tradingagents/harness/market_warning/rule_service.py`
- Modify: `tradingagents/harness/market_warning/reporting.py`
- Modify: `tradingagents/harness/market_warning/test_reasoning.py`
- Modify: `tradingagents/harness/market_warning/test_reporting.py`

**Interfaces:**
- Produces: `DeepSeekReasoningAdapter.from_environment(...)` with Pro/max Think.
- Preserves: explanation-only contract; no level, probability, push, action, or position override.

- [ ] **Step 1: Change tests to require DeepSeek Pro while preserving the rule contract**

```python
def test_warning_adapter_uses_deepseek_pro_max_think():
    adapter = DeepSeekReasoningAdapter.from_environment()
    assert adapter.model_name == "deepseek-v4-pro"

def test_rule_alert_explanation_cannot_override_red_light():
    result = adapter.assess_rule_alert(red_result, previous)
    assert result.recommended_risk_level == RiskLevel.RED
```

- [ ] **Step 2: Run warning reasoning/report tests and verify the migration assertions fail**

Run: `.venv/bin/python tradingagents/harness/market_warning/test_reasoning.py`

Run: `.venv/bin/python tradingagents/harness/market_warning/test_reporting.py`

- [ ] **Step 3: Implement the DeepSeek adapter by reusing validation and circuit-breaker contracts**

Rename user-visible `M3` labels to `LLM` or `DeepSeek` without changing deterministic decisions. Keep bounded retries, JSON repair, raw-I/O logging suppression, and fail-closed persistence behavior.

- [ ] **Step 4: Run the full market-warning suite**

Run: `.venv/bin/python -m pytest tradingagents/harness/market_warning -q`

- [ ] **Step 5: Commit warning migration**

```bash
git add tradingagents/harness/market_warning
git commit -m "feat: migrate warning explanations to DeepSeek V4"
```

---

### Task 7: Full Verification And One-Stock Regression

**Files:**
- Modify only files required by failures attributable to Tasks 1-6.
- Verify generated artifacts under `reports/`; do not commit report output or secrets.

**Interfaces:**
- Consumes: the complete DeepSeek/four-pillar pipeline.
- Produces: one finished A-share report with current price/date, four-pillar rating, separate three-day action, contribution summary, and no reasoning leakage.

- [ ] **Step 1: Run all focused executable test files**

Run: `.venv/bin/python -m pytest tradingagents tests -q`

Expected: all discovered tests pass.

- [ ] **Step 2: Run compilation and security scans**

Run: `.venv/bin/python -m compileall tradingagents cli main.py`

Run: `git grep -nE 'sk-[A-Za-z0-9_-]{16,}|reasoning_content|<think>' -- ':!docs/superpowers/specs/*' ':!docs/superpowers/plans/*'`

Expected: no API key; `reasoning_content` references are restricted to provider boundary code/tests and never report/persistence code.

- [ ] **Step 3: Verify Pro, Flash, and Think with minimal API calls**

Run a minimal non-tool call for each model and one Pro tool-call loop. Confirm model IDs, non-empty final content, and tool completion without persisting raw reasoning.

- [ ] **Step 4: Run one A-share analysis in `.venv`**

Use a liquid A-share selected at execution time and the current trade date. Wait for normal completion; only terminate the process after logs and elapsed time show a clear abnormal stall.

- [ ] **Step 5: Review every generated agent report and final `decision.md`**

Check the following observable outcomes:

```text
price/date = current official close or valid intraday snapshot
research_rating = RM four-pillar tool result
trade_action = PM/risk/timing result
four pillars = visible near the top
agent contributions = accepted/rejected with evidence IDs
raw think/reasoning_content = absent
MiniMax runtime calls = absent
```

- [ ] **Step 6: Run the market-warning dry run without notification**

Confirm the deterministic result persists even if the LLM explanation is unavailable, and no unrequested Feishu message is sent.

- [ ] **Step 7: Commit regression fixes and final verification notes**

```bash
git add tradingagents tests cli main.py .env.example
git commit -m "test: verify DeepSeek four-pillar workflow"
```
