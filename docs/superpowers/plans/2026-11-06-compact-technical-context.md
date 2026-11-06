# Compact Technical Context Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans to implement task-by-task; tests precede production edits.

**Goal:** Stop replaying daily indicator tables to the market LLM while retaining auditable full history and existing downstream contracts.

**Architecture:** Fetch the existing vendor OHLCV once, calculate indicators using the installed stockstats engine, persist full precision locally, and compile a bounded packet. The compact market path makes one artifact-producing LLM invocation without data tools; invalid/empty reports fail closed. The existing tool-loop path remains available by configuration for rollback.

**Tech Stack:** Existing pandas, stockstats, LangChain, stdlib JSON/hashlib/pathlib; no new dependencies.

**Spec:** User approved compact summaries plus recent windows, single-stage limited validation before any full analysis. Design choices below are engineering defaults, not additional user commitments.

## Constraints And Acceptance

- Keep MiniMax M3 and Think unchanged. Do not change ratings, risk gates, or underlying full-history calculations to save quota.
- Preserve raw vendor response and full derived series in a content-addressed local artifact; include its reference and checksum in the report.
- Only observations on/before analysis_date enter calculations; preserve current-price/status/date/source/time, including T-1 and provisional bars.
- Default recent windows: 20 trading days and 12 weeks. Historical indicators/percentiles use the full available data, not the display window. Insufficient history must be explicit; no NaN/Infinity JSON values.
- Default total compact prompt ceiling: 24000 characters; fail rather than truncate essential facts. No data-tool calls are bound on the compact path.
- Rollout remains opt-in (`market_context_mode=compact`) until a bounded live report passes; the production default stays legacy.
- Require parseable SUMMARY and valid market HANDOFF, retaining existing downstream keys and authoritative price enforcement. Empty output must not silently advance to another analyst.
- Limited live smoke: 301171, 2026-11-05; one provider request, output ceiling 8192, SDK/wall-clock/empty-output retries disabled; input character guard applies before any paid call. No full analysis or Feishu success delivery from this smoke.

## Tasks

- [x] Add failing packet tests: long/short histories, stockstats equivalence, no lookahead, bad CSV, T-1/intraday metadata, strict JSON, archival integrity and bounded size.
- [x] Implement `tradingagents/dataflows/technical_context.py`: CSV parsing, deterministic statistics, weekly aggregation and audit artifact.
- [x] Add failing market-node tests: one invocation/no bound tools, no raw-history leakage, old-summary compatibility, invalid report stops, input-size guard and legacy rollback.
- [x] Implement compact market node in a focused module; wire `create_market_analyst` and `market_context_mode` without changing the rest of the graph.
- [x] Add reproducible single-stage smoke command recording usage/request sizes and artifact paths, with a hard request budget; test offline before using the real model.
- [x] Run relevant regression checks, independent review, and one authorized limited smoke. Report actual success/failure and usage, not a full-analysis success claim.
- [ ] Commit, PR and merge according to standing authorization after tests pass.

## Review Focus

- A short display window must not change moving-average warmup or year-percentile samples.
- Future/provisional bars and stale metadata must not be mislabeled as official current prices.
- Bad/missing source data must not turn into a fabricated neutral/zero indicator.
- Legacy graph transitions and downstream SUMMARY consumers must remain compatible.
- A smoke command must not inherit hidden SDK/provider retries or start the remaining analysis graph.

## Preflight Evidence

- Real 301171 data-only preparation returned 483 bars, official daily close 34.47 dated 2026-11-05. Raw input was archived locally; no model call occurred.
- Initial compact prompt measured 8907 characters; subsequent grounding keys add a small bounded map. Old peak request was 152318 message characters.
- Current values for all 12 indicators were compared against the previous vendor warmup window: maximum absolute difference 4.23e-7 (display rounding); both SMA values were identical.
- Independent review findings addressed: HANDOFF calculation facts are generated from the packet, null core judgments fail closed, completed historical weeks are not excluded merely because their last bar was Thursday.

## Single-Request Live Result

- Source version 4777001; directory `logs/technical_smoke/301171_20261105_20261106_v1`.
- Exactly one MiniMax-M3 request, all retries disabled. Input: 9276 prompt characters / 4064 provider-reported input tokens (128 cached).
- Output: 8192 tokens, including 8190 reasoning tokens; finish_reason=length, visible_chars=0. Total reported tokens: 12256; elapsed 72.478 seconds.
- The compact packet and local audit passed, but no valid report/SUMMARY/HANDOFF was produced. This is a failed bounded report smoke, not successful end-to-end analysis or Feishu delivery.
- Input size dropped materially versus the prior peak of 80579 input tokens. Remaining failure is reasoning exhausting the shared output cap even with compact input; do not attribute it solely to long context or claim the account quota problem is solved.
- No automatic second request or full graph run occurred. Requested explicit approval for one 16K-output comparison; until approved/passing, compact remains opt-in and the existing production default is preserved.
- Final offline verification: 91 tests-directory unittest checks, 33 warning-reasoning unittest checks, 128 related standalone checks (252 total). Independent review approved the opt-in merge, not a live acceptance claim.
