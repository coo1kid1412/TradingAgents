# Dual-Market Crash Warning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a point-in-time, calibrated A-share and US broad-market crash warning system that estimates 1-day and 3-day crash probabilities, explains risk with MiniMax M3 Think, pushes only actionable transitions, and applies conservative short-term gates to individual-stock analysis.

**Architecture:** Add an isolated `tradingagents.harness.market_warning` package following ports-and-adapters boundaries. Deterministic code owns data quality, point-in-time features, model inference, policy, state transitions, persistence, scheduling, and hard gates; MiniMax M3 Think only returns a validated structured context assessment and may raise risk by one level. Keep `market_risk.py` as a compatibility and continuation-risk input, and compose the strictest effective gate at its existing read boundary.

**Tech Stack:** Python 3.13, pandas, NumPy, scikit-learn pipelines, joblib, exchange-calendars, Tushare, yfinance, existing MiniMax/OpenAI-compatible client, SQLite, existing Feishu sender, standard-library `unittest`.

## Global Constraints

- Preserve unrelated local changes in `main.py`, `.agents/`, `docs/capital_flow_refactor_review_guide.md`, and `tradingagents/harness/tradingagents.db`.
- Use `.venv/bin/python` for every test, CLI, training, and smoke command.
- Do not use random train/test splits, SMOTE, future revisions, or data whose `available_at` is later than the prediction time.
- Freeze labels at A-share `1d <= -4%`, A-share `3d worst <= -6%`, US `1d <= -3%`, and US `3d worst <= -5%`.
- Treat missing, stale, conflicted, and insufficient data conservatively; they must never become normal GREEN.
- Do not persist MiniMax Think text or pass it to downstream agents. Persist only validated structured fields.
- MiniMax failure must never prevent quant prediction persistence or report generation.
- Keep US intraday in `shadow` mode while Yahoo is the only source; shadow decisions cannot hard-gate individual US stocks.
- Keep individual-stock long-term ratings and one-year targets independent from this short-term market warning.
- Add abstractions only at external boundaries or where the A-share/US strategies genuinely differ.
- Use deterministic report rendering. The LLM must not write the final report Markdown.
- Run focused tests after every task and the complete market-warning suite after every integration task.

---

### Task 1: Establish the domain model and ports

**Files:**
- Create: `tradingagents/harness/market_warning/__init__.py`
- Create: `tradingagents/harness/market_warning/domain.py`
- Create: `tradingagents/harness/market_warning/ports.py`
- Create: `tradingagents/harness/market_warning/test_domain.py`

**Interfaces:**
- `Market(str, Enum)`: `A_SHARE`, `US`.
- `DataStatus(str, Enum)`: `FRESH`, `PARTIAL`, `CONFLICTED`, `STALE`, `INSUFFICIENT`, `SHADOW`.
- `RiskLevel(str, Enum)`: `GREEN`, `YELLOW`, `ORANGE`, `RED`, `UNKNOWN`.
- `MarketPhase(str, Enum)`: `FIRST_SHOCK`, `CONTINUATION`.
- Frozen dataclasses: `MarketDataPoint`, `RawMarketSnapshot`, `Evidence`, `FeatureSnapshot`, `QuantRiskAssessment`, `LLMContextAssessment`, `FinalWarningDecision`, `RunnerResult`.
- Protocols: `MarketDataPort`, `MarketContextPort`, `ProbabilityModelPort`, `ReasoningPort`, `WarningRepository`, `WarningNotifier`, `ClockPort`.

- [ ] Write failing tests proving dataclasses reject invalid probability, confidence, unsupported risk level, and timezone-naive `as_of_time`.
- [ ] Add a test proving `FeatureSnapshot.evidence_ids` is stable and duplicate evidence IDs are rejected.
- [ ] Run `.venv/bin/python tradingagents/harness/market_warning/test_domain.py` and confirm the tests fail because the package does not exist.
- [ ] Implement string enums and frozen dataclasses. Validate numeric ranges and timezone awareness in `__post_init__` without importing vendor, database, or LLM modules.
- [ ] Define the ports as `typing.Protocol`; keep each method narrow:

```python
class MarketDataPort(Protocol):
    def load_snapshot(
        self, market: Market, as_of_time: datetime, session_slot: str
    ) -> RawMarketSnapshot: ...

class ProbabilityModelPort(Protocol):
    def predict(self, snapshot: FeatureSnapshot) -> QuantRiskAssessment: ...

class ReasoningPort(Protocol):
    def assess(
        self,
        snapshot: FeatureSnapshot,
        quant: QuantRiskAssessment,
        previous: FinalWarningDecision | None,
    ) -> LLMContextAssessment: ...
```

- [ ] Export only stable domain types from `market_warning/__init__.py`.
- [ ] Re-run the domain test and confirm it passes.
- [ ] Run `rg -n 'tushare|yfinance|sqlite|langchain|minimax' tradingagents/harness/market_warning/domain.py tradingagents/harness/market_warning/ports.py` and confirm no infrastructure dependency leaked into the domain layer.

### Task 2: Add the warning schema and SQLite repository

**Files:**
- Modify: `tradingagents/harness/schema.sql`
- Create: `tradingagents/harness/market_warning/adapters/__init__.py`
- Create: `tradingagents/harness/market_warning/adapters/sqlite_repository.py`
- Create: `tradingagents/harness/market_warning/test_repository.py`

**Interfaces:**
- `SQLiteWarningRepository.save_feature_snapshot(snapshot) -> int`
- `save_prediction(feature_snapshot_id, assessment) -> int`
- `save_reasoning(feature_snapshot_id, assessment, model_name) -> int`
- `save_decision(feature_snapshot_id, prediction_ids, reasoning_id, decision) -> int`
- `load_latest_decision(market, as_of_time=None) -> FinalWarningDecision | None`
- `load_previous_decision(market, before_time) -> FinalWarningDecision | None`
- `claim_alert(idempotency_key, decision_id, payload_hash) -> bool`
- `finish_alert(idempotency_key, status, error_summary=None) -> None`
- `register_model(record)`, `load_active_model(market, horizon)`.

- [ ] Add a failing temporary-database test that calls `_db.connect()` and asserts all six tables from the design exist.
- [ ] Add failing round-trip tests for one feature snapshot, two horizon predictions, one reasoning result, and one final decision.
- [ ] Add a failing idempotency test: the first `claim_alert` returns true and the same key returns false, including from a second repository instance.
- [ ] Run `.venv/bin/python tradingagents/harness/market_warning/test_repository.py` and confirm failure.
- [ ] Add these tables and indexes to `schema.sql` using `IF NOT EXISTS`:
  - `market_warning_feature_snapshots`: integer id, market, as_of_time, session_slot, feature_version, data_status, reliability_grade, features/evidence/source-times JSON, created_at; unique `(market, as_of_time, feature_version)`.
  - `market_warning_predictions`: integer id, feature_snapshot_id FK, horizon, probability, base_rate, market_phase, reliability_grade, model_version, calibration_version, top_contributors JSON; unique `(feature_snapshot_id, model_version, horizon)`.
  - `market_warning_reasoning`: integer id, feature_snapshot_id FK, model_name, reasoning_status, structured JSON, error_class, created_at; never store raw response or Think content.
  - `market_warning_decisions`: integer id, feature_snapshot_id FK, baseline_level, final_level, transition, entry_gate, new_position_cap_pct, holding_action, push_required, data_status, reasons JSON, model_version, created_at; unique `feature_snapshot_id`.
  - `market_warning_alerts`: integer id, idempotency_key unique, decision_id FK, payload_hash, push_status, sent_at, error_summary, created_at.
  - `market_warning_model_registry`: model_version, market, horizon, feature_version, calibration_version, training_cutoff, artifact_path, artifact_sha256, metrics JSON, base_rate, active flag, created_at; primary key `(model_version, market, horizon)` and a partial lookup index for active rows.
- [ ] Implement JSON serialization at the repository boundary, explicit column lists, transactions, and immutable domain reconstruction. Do not use `pickle` or SQLite blobs.
- [ ] Make alert claiming atomic with a unique insert and catch only `sqlite3.IntegrityError` for duplicate claims.
- [ ] Re-run repository tests and existing `.venv/bin/python tradingagents/harness/test_market_risk.py`.

### Task 3: Implement point-in-time data quality rules

**Files:**
- Create: `tradingagents/harness/market_warning/quality.py`
- Create: `tradingagents/harness/market_warning/test_quality.py`

**Interfaces:**
- `QUALITY_POLICY_V1`: intraday core quote age at most 10 minutes, cross-source price deviation at most 0.5%, timestamp skew at most 120 seconds, 100% core-field coverage for `FRESH`, and at least 70% optional-field coverage for reliability A.
- `evaluate_data_quality(snapshot, policy, now) -> DataQualityAssessment`.
- `select_point_in_time(points, as_of_time) -> tuple[MarketDataPoint, ...]`.
- `combine_source_quotes(primary, secondary, tolerance) -> MarketDataPoint` or a conflicted result.

- [ ] Write failing tests that exclude a record whose market date is correct but whose `available_at` is after `as_of_time`.
- [ ] Add failing tests for fresh, partial, stale, conflicted, insufficient, and US single-source shadow cases.
- [ ] Add a regression test for the existing stale-cache failure: a cached US close with `data_time` before the required session cutoff cannot be marked fresh merely because it was fetched today.
- [ ] Add a test proving three index direction votes cannot populate `breadth_up_pct` or `breadth_above_ma20_pct`.
- [ ] Run the quality test and confirm failure.
- [ ] Implement quality evaluation from actual `data_time`, `fetched_at`, required core fields, optional fields, market, and session slot.
- [ ] Use explicit tolerances from a versioned `QUALITY_POLICY_V1`: index quote age, cross-source price deviation, cross-source timestamp skew, daily disclosure cutoff, and minimum field coverage.
- [ ] Map reliability deterministically: `A` for fresh dual-source/core-complete data with at least 70% optional coverage; `B` for fresh core-complete data with lower optional coverage; `C` for usable partial/shadow data; `UNAVAILABLE` for conflicted/stale/insufficient data.
- [ ] Return `SHADOW` only for otherwise usable US intraday data with one independent source; return `UNKNOWN`-forcing statuses for stale/conflicted/insufficient.
- [ ] Re-run the quality test and confirm every status and timestamp assertion passes.

### Task 4: Build deterministic A-share and US feature strategies

**Files:**
- Create: `tradingagents/harness/market_warning/features.py`
- Create: `tradingagents/harness/market_warning/test_features.py`

**Interfaces:**
- `FEATURE_VERSION = "market-warning-v1"`.
- `FEATURE_METADATA`: source, availability rule, missing strategy, direction, unit, and version for every feature.
- `AShareFeatureStrategy.build(raw, prior_history) -> FeatureSnapshot`.
- `USFeatureStrategy.build(raw, prior_history) -> FeatureSnapshot`.
- `derive_market_phase(drawdown_20d) -> MarketPhase`.

- [ ] Add failing tests for returns, rolling drawdowns, MA distances/slopes, realized-volatility ratios, range, close location, and volume z-score using hand-calculated fixtures.
- [ ] Add failing tests for `FIRST_SHOCK` when 20-day drawdown is greater than -5% and `CONTINUATION` at or below -5%.
- [ ] Add A-share tests for true stock breadth, margin growth/contraction, valuation/turnover percentile, limit-down diffusion, and Shibor changes.
- [ ] Add US tests for HYG/LQD relative strength, VIX/VIX3M term structure, Russell/Nasdaq/SOXX relative weakness, and credit-plus-volatility transition.
- [ ] Add missing-data tests proving unavailable breadth remains `None` and emits an evidence record instead of becoming zero or neutral.
- [ ] Run `.venv/bin/python tradingagents/harness/market_warning/test_features.py` and confirm failure.
- [ ] Implement shared pure helpers and two concrete strategies. Use pandas rolling windows with `min_periods` equal to the declared horizon; never backfill from future rows.
- [ ] Emit evidence IDs in the form `<market>:<feature_version>:<feature_name>:<as_of_time>` and include value, direction, source time, and a concise deterministic explanation.
- [ ] Sort contributor inputs and evidence deterministically so identical inputs produce byte-stable JSON.
- [ ] Re-run feature and quality tests.

### Task 5: Add historical and live market data adapters

**Files:**
- Create: `tradingagents/harness/market_warning/adapters/tushare_data.py`
- Create: `tradingagents/harness/market_warning/adapters/realtime_quote.py`
- Create: `tradingagents/harness/market_warning/adapters/us_market_data.py`
- Create: `tradingagents/harness/market_warning/adapters/data_cache.py`
- Create: `tradingagents/harness/market_warning/test_data_adapters.py`

**Interfaces:**
- `TushareAShareDataAdapter.load_snapshot(...) -> RawMarketSnapshot`.
- `TushareAShareDataAdapter.backfill(start_date, end_date) -> iterator[RawMarketSnapshot]`.
- `RealtimeAShareDataAdapter` reuses `fetch_intraday_quote()` and accepts an injectable cross-sectional loader.
- `YahooUSDataAdapter.load_snapshot(...) -> RawMarketSnapshot` and `.backfill(...)`.
- Raw normalized cache under `harness_data/market_warning/raw/`, partitioned by market/dataset/year and accompanied by `available_at` metadata.

- [ ] Write mocked tests for Tushare index, daily, daily-basic, margin, Shibor, limit-list, and money-flow responses; assert symbol, field, source, data time, and conservative disclosure availability.
- [ ] Write a regression test where Tushare `rt_k` is forbidden and Sina succeeds; assert source lineage and the actual quote timestamp are preserved.
- [ ] Add a test where two A-share quote sources disagree beyond tolerance and produce `CONFLICTED`.
- [ ] Write mocked Yahoo tests that normalize S&P 500, Nasdaq, Russell 2000, SOXX, VIX, VIX3M, HYG, LQD, Treasury, and dollar proxies without treating download time as market data time.
- [ ] Add a test proving US Yahoo-only intraday snapshots are `SHADOW` and cannot claim production reliability A.
- [ ] Run the adapter test and confirm failure.
- [ ] Implement Tushare calls behind an injectable `pro` object and reuse `tradingagents.dataflows.intraday_quote.fetch_intraday_quote` rather than adding another quote parser.
- [ ] Compute A-share cross-sectional breadth from point-in-time stock observations. If the stock universe cannot be loaded, leave breadth fields missing; do not use the existing three-index approximation.
- [ ] Implement Yahoo batch downloads with explicit ticker maps and actual last-row timestamps. Keep source count in the normalized snapshot.
- [ ] Write cache files atomically via a temporary file plus rename; include a small JSON manifest containing query, source, fetched time, min/max data time, row count, and schema version.
- [ ] Re-run adapter, intraday quote, quality, and feature tests.

### Task 6: Build leakage-safe datasets and four calibrated logistic pipelines

**Files:**
- Modify: `pyproject.toml`
- Create: `tradingagents/harness/market_warning/probability.py`
- Create: `tradingagents/harness/market_warning/training.py`
- Create: `tradingagents/harness/market_warning/test_probability.py`
- Create: `tradingagents/harness/market_warning/test_training.py`

**Interfaces:**
- Add `scikit-learn>=1.7,<2` and `exchange-calendars>=4.11,<5` dependencies.
- `build_labels(index_frame, market) -> DataFrame`.
- `time_partitions(frame) -> dev, validation, test` with a three-trading-day embargo.
- `fit_model(train, calibration, market, horizon) -> ModelBundle`.
- `evaluate_model(bundle, test) -> EvaluationReport`.
- `SklearnProbabilityModel.predict(snapshot) -> QuantRiskAssessment`.
- CLI subcommands: `backfill`, `train`, `evaluate`, `promote`.

- [ ] Add failing label tests around exact threshold boundaries and prove each label uses only future returns while each feature row uses only information at or before the prediction timestamp.
- [ ] Add failing partition tests for 2000-2012 development, 2013-2019 validation, 2020-2026-07-31 frozen test, and a three-trading-day embargo at both boundaries.
- [ ] Add a test that fails if `train_test_split`, `shuffle=True`, or SMOTE appears in the training module.
- [ ] Add synthetic imbalanced-data tests that fit a `Pipeline(SimpleImputer(add_indicator=True), StandardScaler(), LogisticRegression())`, calibrate with Platt scaling on the later calibration window, and return probabilities in `[0, 1]`.
- [ ] Add tests for missing model, checksum mismatch, feature-version mismatch, and stale registry entry; each must return unavailable rather than train on demand.
- [ ] Add evaluation tests for Brier Score, AUPRC, calibration bins, phase breakdown, crisis-period contribution, monthly alert budget, and comparison to constant-base-rate and old-market-risk baselines.
- [ ] Run probability/training tests and confirm failure.
- [ ] Add dependencies to `pyproject.toml`, install with `.venv/bin/python -m pip install -e .`, and record the resolved versions in the implementation notes.
- [ ] Implement labels and chronological partitions. Drop rows whose 3-day label window crosses a partition boundary.
- [ ] Fit four bundles (`a_share/1d`, `a_share/3d`, `us/1d`, `us/3d`) without synthetic oversampling. If class weights are used, calibrate the resulting score on the later chronological window.
- [ ] Persist one joblib bundle per market/horizon under `harness_data/models/market_warning/<model_version>/`; compute SHA-256 and register metadata in SQLite. Do not commit model binaries.
- [ ] Compute top contributors from standardized logistic coefficients times transformed feature values. Map transformed missing-indicator columns back to stable evidence names.
- [ ] Re-run probability/training tests and full market-warning test discovery.

### Task 7: Implement the deterministic warning policy and state machine

**Files:**
- Create: `tradingagents/harness/market_warning/policy.py`
- Create: `tradingagents/harness/market_warning/test_policy.py`

**Interfaces:**
- `POLICY_VERSION = "market-warning-policy-v1"`.
- `baseline_level(quant, snapshot) -> RiskLevel`.
- `apply_llm_adjustment(baseline, context, snapshot) -> RiskLevel`.
- `transition(previous, candidate, valid_snapshot_count) -> StateTransitionResult`.
- `build_final_decision(...) -> FinalWarningDecision`.

- [ ] Write table-driven failing tests for GREEN below 2x base rate, YELLOW at 2x, ORANGE at 4x plus a transition signal, RED at 8x or a hard trigger, and UNKNOWN for conflicted/stale/insufficient.
- [ ] Prove the stricter of 1-day and 3-day mappings wins.
- [ ] Define versioned hard triggers and test each independently:
  - market-level abnormal range z-score at least 3.0 plus close location at or below 0.15 and daily return at or below -2.0% for A-share or -1.5% for US;
  - A-share declining-stock share at least 85% or limit-down share at least 2%, confirmed by a negative broad-index return;
  - US HYG/LQD five-day relative return at or below -1.5%, confirmed by either a five-day VIX increase of at least 20% or `VIX/VIX3M >= 1.0`;
  - data-source failure is never a hard market-risk trigger.
- [ ] Add LLM adjustment tests: confidence below 0.70, fewer than two valid evidence IDs, invented evidence IDs, lowering ORANGE/RED, and jumping more than one level are all rejected.
- [ ] Add state-machine tests: upgrades are immediate; equal levels do not push; ORANGE/RED need two consecutive valid recovery snapshots; intraday recovery is persisted but not pushed; ORANGE-to-RED pushes.
- [ ] Add action mapping tests:
  - GREEN: `OPEN`, no warning cap;
  - YELLOW: `OPEN`, reminder only;
  - ORANGE: `CONDITIONAL`, new single-stock cap 3%, holding `HOLD_OR_REDUCE`;
  - RED: `WAIT`, cap 0%, holding `REDUCE`;
  - UNKNOWN: `WAIT`, cap 0%, holding `HOLD`, explicitly not labelled RED.
- [ ] Run the policy test and confirm failure.
- [ ] Implement pure policy functions with no repository, notifier, or model imports.
- [ ] Re-run policy and domain tests.

### Task 8: Add MiniMax M3 Think structured reasoning with circuit breaking

**Files:**
- Create: `tradingagents/harness/market_warning/reasoning.py`
- Create: `tradingagents/harness/market_warning/adapters/minimax_reasoning.py`
- Create: `tradingagents/harness/market_warning/test_reasoning.py`

**Interfaces:**
- `validate_context_assessment(payload, valid_evidence_ids) -> LLMContextAssessment`.
- `MiniMaxReasoningAdapter(llm, model_name="MiniMax-M3", timeout=90, breaker: CircuitBreaker | None = None)`.
- `CircuitBreaker(failure_threshold=3, cooldown=timedelta(minutes=30))`.
- `build_reasoning_prompt(...)` emits compact structured input and a strict JSON schema.

- [ ] Add failing tests for valid JSON, fenced JSON, invalid level, out-of-range confidence, missing causal chain, missing conflicting evidence, invented IDs, and empty output after Think stripping.
- [ ] Add a fake-LLM test for keyword/content filtering: first response is blocked/empty, one repair request is made, second failure returns `reasoning_status="fallback"` without raising.
- [ ] Add timeout and invalid-JSON tests proving exactly one repair attempt and no raw error/response persistence.
- [ ] Add circuit-breaker tests for three consecutive failures, no LLM invocation during cooldown, and reset after a successful call.
- [ ] Add call-policy tests: premarket always calls; ordinary GREEN/YELLOW intraday polling does not; candidate ORANGE/RED and ORANGE-to-RED call.
- [ ] Run the reasoning test and confirm failure.
- [ ] Implement strict JSON extraction and domain validation. The repair prompt may include the validation error class and required schema, but not the original Think text.
- [ ] Create the LLM via the existing `create_llm_client("minimax", model, ...)` path and `get_llm_wrapped()`. Read model/base URL/API key through existing configuration/environment conventions; do not modify user-owned `main.py`.
- [ ] Set market-warning-specific timeout and token budget through `MARKET_WARNING_LLM_TIMEOUT` and `MARKET_WARNING_LLM_MAX_TOKENS`, defaulting to 90 seconds and 4096 tokens.
- [ ] Ensure only `LLMContextAssessment` and a coarse `error_class` leave the adapter.
- [ ] Re-run reasoning plus existing MiniMax compliance/normalization tests.

### Task 9: Orchestrate evaluation, persistence, and deterministic reports

**Files:**
- Create: `tradingagents/harness/market_warning/service.py`
- Create: `tradingagents/harness/market_warning/reporting.py`
- Create: `tradingagents/harness/market_warning/test_service.py`
- Create: `tradingagents/harness/market_warning/test_reporting.py`

**Interfaces:**
- `MarketWarningService.evaluate(market, as_of_time, session_slot) -> RunnerResult`.
- `render_premarket_report(result, previous) -> str`.
- `render_upgrade_report(result, previous) -> str`.
- Reports saved under `reports/market_warning/<market>/<YYYY-MM-DD>/<HHMM>-<slot>.md`.

- [ ] Write an orchestration test using fake ports that asserts the exact order: load data, assess quality, build features, persist snapshot, predict, optionally reason, transition, persist decision, decide whether to notify.
- [ ] Add failure tests for adapter exception, missing model, LLM failure, repository write failure, and notifier failure. Data/model failures must produce UNKNOWN; LLM/notifier failures must preserve the quant decision and saved report.
- [ ] Add deterministic report golden tests for GREEN premarket, FIRST_SHOCK ORANGE, CONTINUATION RED, UNKNOWN stale data, US shadow, and intraday ORANGE-to-RED.
- [ ] Assert the first visible block always contains the lamp, immediate action, gate, and cap; report language must distinguish probability from certainty.
- [ ] Assert the premarket order is: action, probabilities/base rates, phase, previous change, top three contributors, M3 context and counter-evidence, data/model metadata.
- [ ] Assert reports never contain `<think>`, internal prompt text, raw JSON, stack traces, API errors, or secrets.
- [ ] Run service/reporting tests and confirm failure.
- [ ] Implement one transaction boundary per persistence stage so quant artifacts survive LLM/notifier failure.
- [ ] Render Markdown from typed objects only. Bold the immediate action and use a compact table for 1-day/3-day probabilities and baselines.
- [ ] Write report files atomically and include the report path in `RunnerResult`.
- [ ] Re-run service, reporting, repository, reasoning, and policy tests.

### Task 10: Add exchange-aware runner, idempotent Feishu delivery, and operations CLI

**Files:**
- Create: `tradingagents/harness/market_warning/runner.py`
- Create: `tradingagents/harness/market_warning/adapters/feishu_notifier.py`
- Create: `tradingagents/harness/market_warning/test_runner.py`
- Create: `scripts/install_market_warning_cron.sh`
- Create: `docs/market_warning.md`

**Interfaces:**
- CLI: `.venv/bin/python -m tradingagents.harness.market_warning.runner [--market ...] [--at ISO] [--dry-run] [--force]`.
- `due_evaluations(now) -> tuple[EvaluationSlot, ...]` backed by `exchange_calendars` and `zoneinfo`.
- Idempotency key: market, local trade date, session slot or 5-minute bucket, final level, transition, model version.
- One cron wake-up every five minutes; the runner decides whether a market is due.

- [ ] Add failing calendar tests for A-share lunch break, Chinese holiday, US holiday, US summer time, US winter time, early close, 08:30 local premarket, and open-plus-five-minutes intraday start.
- [ ] Add runner tests for A-share windows `09:35-11:25` and `13:05-14:55`, US open+5 through before close, and no weekend evaluation.
- [ ] Add duplicate-run tests proving repeated processes cannot send the same alert twice.
- [ ] Add notification tests: premarket always sends; intraday GREEN/YELLOW is silent; first ORANGE/RED and ORANGE-to-RED send; recovery persists but is silent.
- [ ] Add Feishu failure tests proving the alert row becomes `failed` while the decision and report remain saved and a later explicit retry can reuse the same alert record safely.
- [ ] Run runner tests and confirm failure.
- [ ] Implement calendar scheduling with `exchange_calendars`; use `Asia/Shanghai` and `America/New_York` via `zoneinfo`, never fixed UTC offsets.
- [ ] Reuse the existing Feishu application/webhook transport from `market_risk_daily` behind `WarningNotifier`; do not duplicate credentials or log them.
- [ ] Make `scripts/install_market_warning_cron.sh` print and install only this idempotent line after confirmation in the execution session:

```cron
*/5 * * * * cd /Users/lailixiang/WorkSpace/QoderWorkspace/TradingAgents && .venv/bin/python -m tradingagents.harness.market_warning.runner >> harness_data/logs/market_warning.log 2>&1
```

- [ ] Document report reading order, lamp/action meanings, UNKNOWN versus RED, US shadow limitations, manual dry run, model training/promotion, logs, and rollback.
- [ ] Re-run runner tests and dry-run the CLI at fixed A-share and US DST timestamps.

### Task 11: Compose the strictest warning gate into individual-stock analysis

**Files:**
- Modify: `tradingagents/harness/market_risk.py`
- Modify: `tradingagents/harness/test_market_risk.py`
- Modify: `tradingagents/agents/managers/test_market_risk_gate.py`
- Verify: `tradingagents/graph/propagation.py`
- Verify: `tradingagents/agents/managers/pm_tools.py`

**Interfaces:**
- `load_market_warning_for_ticker(ticker, trade_date, db_path, analysis_time) -> dict | None`.
- `compose_effective_market_gate(legacy_snapshot, warning_decision, market) -> dict`.
- Keep `load_market_risk_for_ticker(...)` as the graph-facing compatibility entry point.

- [ ] Add a failing test where legacy risk is OPEN and warning ORANGE; assert effective `entry_gate="CONDITIONAL"`, cap 3%, and long-term rating fields are absent/unchanged.
- [ ] Add a failing test where legacy risk is OPEN and warning RED/UNKNOWN; assert WAIT and cap 0%.
- [ ] Add a failing test where legacy WAIT is stricter than warning GREEN; assert the old WAIT remains effective.
- [ ] Add a failing US shadow test proving the warning is visible for context but does not override the production gate.
- [ ] Add freshness tests proving an outdated warning decision fails closed only when it is the configured production source, and that the reason names the stale snapshot rather than inventing market stress.
- [ ] Add PM tool tests proving ORANGE maps to the already-supported `CONDITIONAL` semantics and RED/UNKNOWN to `WAIT`; no new PM action vocabulary is required.
- [ ] Run market-risk tests and confirm the new assertions fail.
- [ ] Load the latest warning decision at the existing `load_market_risk_for_ticker` boundary and compose by gate severity `OPEN < CONDITIONAL < WAIT`, then by minimum position cap.
- [ ] Preserve legacy keys and add namespaced context keys: `legacy_market_risk`, `market_warning`, `effective_gate_source`, `warning_level`, `warning_phase`, `warning_probabilities`.
- [ ] Do not change `Propagator`, PM report schemas, long-term rating, or one-year target-price logic unless a focused failing test proves compatibility requires it.
- [ ] Re-run market-risk, PM gate, propagation/state, report formatter, and harness extractor tests.

### Task 12: Backfill point-in-time history, train, evaluate, and promote V1 models

**Files:**
- Generate: `harness_data/market_warning/raw/` (not committed)
- Generate: `harness_data/market_warning/features/` (not committed)
- Generate: `harness_data/models/market_warning/` (not committed)
- Generate: `reports/market_warning/model-evaluation/market-warning-v1.md` (commit only the compact evaluation report; do not commit raw data or model binaries)
- Modify: `.gitignore` if generated directories are not already ignored.

**Interfaces:**
- Four registered model artifacts with frozen features/labels and SHA-256 checksums.
- One evaluation report containing dev/validation/test dates, embargo proof, Brier, AUPRC, calibration bins, phase recall, alert budget, crisis concentration, and old-system comparison.

- [ ] Run a one-month A-share and US backfill first; inspect actual source times, disclosure lags, missingness, and duplicate keys before requesting the full range.
- [ ] Run full historical backfill from 2000-01-01 through 2026-07-31. Respect vendor rate limits and resume from cache after interruption.
- [ ] Build feature datasets and run a leakage audit that checks every input `available_at <= as_of_time` and every 3-day label window stays within its partition.
- [ ] Train and evaluate all four frozen pipelines with:

```bash
.venv/bin/python -m tradingagents.harness.market_warning.training train \
  --start 2000-01-01 --test-end 2026-07-31 --version market-warning-v1
```

- [ ] Fail promotion if any model has Brier no better than constant base rate, AUPRC no better than prevalence, expected calibration error above 0.05, one named crisis contributing more than 50% of all test-period true positives, or more than six distinct ORANGE/RED alert entries per calendar month on average.
- [ ] Compare the warning lead time and recall against legacy `market_risk` separately for FIRST_SHOCK and CONTINUATION. Do not tune thresholds on the frozen test interval.
- [ ] Inspect results by 2008, 2015, 2020, 2022, and non-crisis years so one crisis cannot dominate the claimed performance.
- [ ] If all gates pass, run `promote --version market-warning-v1`; otherwise leave models inactive and keep runtime UNKNOWN while reporting the failed gate plainly.
- [ ] Commit only code, tests, `.gitignore`, and the compact evaluation report.

### Task 13: End-to-end production smoke, scheduler activation, and delivery

**Files:**
- Verify: all task-related files
- Generate: A-share and US reports under `reports/market_warning/` (runtime artifacts, not committed unless a sanitized example is intentionally added)
- Verify: `harness_data/logs/market_warning.log`

**Interfaces:**
- A-share premarket report and live/close smoke based on actual data timestamps.
- US premarket report and Yahoo-only intraday shadow report.
- Existing individual-stock analysis receives the strictest effective gate.

- [ ] Run the complete focused suite:

```bash
.venv/bin/python -m unittest discover \
  -s tradingagents/harness/market_warning -p 'test_*.py'
.venv/bin/python tradingagents/harness/test_market_risk.py
.venv/bin/python tradingagents/agents/managers/test_market_risk_gate.py
.venv/bin/python tradingagents/dataflows/test_intraday_quote.py
.venv/bin/python tradingagents/agents/managers/test_entry_timing.py
```

- [ ] Run any broader repository test command discovered during implementation and record unrelated pre-existing failures separately.
- [ ] Run one fixed-time replay for known calm, first-shock, and continuation dates in both markets; verify no data timestamp or disclosure crosses the replay time.
- [ ] Run current A-share premarket/live evaluation and inspect report action prominence, actual source data times, probabilities, base rates, phase, reliability, M3 structured assessment, and effective gate.
- [ ] Run current US premarket and intraday evaluation; verify single-source intraday is visibly `SHADOW` and does not hard-gate US stocks.
- [ ] Simulate MiniMax timeout, blocked keyword, empty response, and malformed JSON; verify report generation and persistence remain 100% and no Think text appears.
- [ ] Simulate duplicate runner invocations and Feishu failure/retry; verify one logical alert, no duplicate push, and preserved decisions.
- [ ] Install the five-minute cron only after all model, smoke, and idempotency checks pass. Verify `crontab -l` contains one market-warning runner and no deleted legacy 08:30/20:30 market-risk tasks.
- [ ] Commit task-related changes on `codex/dual-market-crash-warning`, push the branch, create a PR, inspect CI/review, merge after tests pass under the user's standing authorization, and verify remote `main` contains the merge.
- [ ] Provide the user a concise Chinese handoff: current lamp/action, where to read 1-day/3-day probabilities, FIRST_SHOCK/CONTINUATION meaning, reliability/SHADOW/UNKNOWN meaning, and exactly which alerts will generate Feishu messages.

---

## Spec Coverage Review

- [ ] Goals 1-8 are covered by Tasks 3-13.
- [ ] Clean Architecture boundaries and named patterns are covered by Tasks 1-2, 7-10.
- [ ] A-share and US data inputs, point-in-time availability, and all six runtime statuses are covered by Tasks 3-5.
- [ ] Deterministic common/A-share/US feature groups are covered by Task 4.
- [ ] Frozen labels, four models, chronological partitions, embargo, Platt calibration, no SMOTE, metrics, drift metadata, and model registry are covered by Tasks 6 and 12.
- [ ] M3 invocation rules, strict contract, one repair retry, circuit breaker, and Think suppression are covered by Task 8.
- [ ] Probability multiples, hard triggers, LLM one-level ceiling, hysteresis, action caps, and UNKNOWN semantics are covered by Task 7.
- [ ] Premarket/intraday schedules, DST, idempotency, report order, and Feishu rules are covered by Tasks 9-10.
- [ ] Six storage tables and artifact checksums are covered by Tasks 2 and 6.
- [ ] Individual-stock compatibility and long-term-rating isolation are covered by Task 11.
- [ ] Unit, integration, model, and end-to-end acceptance criteria are covered by Tasks 1-13.
- [ ] A-share production smoke and US single-source shadow rollout are covered by Tasks 12-13.

## Completeness And Consistency Review

- [ ] Run `rg -n 'TODO|TBD|以后再|待定|某个' docs/superpowers/plans/2026-08-01-dual-market-crash-warning-implementation.md` and remove unresolved implementation gaps.
- [ ] Verify all referenced paths are either existing files or explicitly marked Create/Generate.
- [ ] Verify enum values, table names, status names, gate names, model versions, label thresholds, dates, and schedule windows match the approved spec.
- [ ] Verify all new tests have exact `.venv` commands and every external dependency has an adapter boundary and a failure-path test.
