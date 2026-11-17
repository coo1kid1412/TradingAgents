# Small-Cap Top10 Implementation Plan

**Goal:** Deliver a deterministic, PIT-safe A-share daily candidate report to the configured Feishu recipient at 19:30 Asia/Shanghai on trading days.

**Spec:** `docs/superpowers/specs/2026-11-16-small-cap-daily-top10-design.md`. User has now requested implementation and daily delivery; this supersedes the draft's no-scheduling-yet gate. No broker/LLM calls, no position recommendations, no V2 trading simulation.

**Architecture:** Independent package `tradingagents/strategies/small_cap`. Provider collects raw Tushare records with actual observation timestamps; selector is a pure function over a sealed JSON snapshot. Runner persists immutable inputs/results plus notification state, then deterministic reporting sends a mobile-friendly Feishu card. Codex automation invokes only the bounded CLI; its model must not re-analyze stocks or change code.

## Contracts

- Snapshot dict: `schema_version`, `mode=forward`, `data_date` (ISO date), `as_of_at` (aware ISO instant), `target_trade_date`, `calendar` (raw trading calendar rows), `records` (normalized per-security records), `coverage` (global data completeness), `raw` (sanitized endpoint responses).
- Record: `ts_code,name,exchange,market,curr_type,list_date,delist_date,industry,industry_basis,security_status` (`NORMAL/ST/DELISTING/UNKNOWN`), `status_effective_date`, `observed_at`, `trade_date,close,total_market_cap_yuan,volume,suspended`, `financials`.
- Annual financial row: `end_date,ann_date,f_ann_date,report_type,n_income_attr_p,revenue,observed_at,revision_id`. Dates may be YYYYMMDD and are normalized by selector. Only consolidated annual versions (`1/4/5`) are candidates; select newest available annual period before newest version; never use missing/newer rows as a reason to cherry-pick an older profitable year.
- `select_top10(snapshot, config=Config()) -> dict`: result fields from spec, plus `snapshot_id`, `selection_hash`, `top10`, `exclusions`, `coverage`, `warnings`. UNKNOWN rows that could sort ahead of the boundary block publication. No live access in selector.
- `TushareSource.collect(data_date: str | None = None) -> dict`: real current Shanghai-day forward snapshot only; closed days return `{'status':'CLOSED','data_date':...}`. No historical value masquerading; exact original snapshot replay is separate.
- `render_markdown(result) -> str`; `send_feishu(result, markdown, uuid) -> dict` with safe `status/message_id/recipient_hash`. New environment variable prefix `FEISHU_SMALL_CAP_*` falls back to the existing explicitly configured project recipient. Never print secrets/raw credential errors.
- CLI: `python -m tradingagents.strategies.small_cap run [--data-date DATE] [--no-notify] [--refresh] [--root PATH]`; `replay --snapshot PATH [--as-of AWARE_ISO] [--output-dir PATH]` is offline and cannot notify. Preflight is read-only and cannot notify. Default run loads repo `.env`, timezone fixed Shanghai regardless of host.

## Tasks

- [x] Selector: RED/GREEN coverage for threshold boundaries, dates/visibility, annual versions, future contamination, historic statuses, missing ranking boundary, ties and seasonal mode; pure validated selection and audit output implemented.
- [x] Provider: bounded HTTPS requests, current-day calendar/universe/status, exact-day quotes, annual versions and disjoint announcement partitions implemented; official SZSE identity-only verification added with its own offline tests. No stale fallback/vendor valuation prose.
- [x] Reporting: independent worker implemented mobile Markdown/card and business-code-checked Feishu delivery; 22 offline tests. No sends outside explicit `--notify` runner invocation.
- [x] Runner: RED/GREEN for immutable archive/replay, idempotent daily runs, delivery failure/ambiguity, concurrent lock, no-notify, cutoff and closed-day behavior; SQLite delivery state, atomic report bundles and CLI implemented. New raw-data directory ignored in Git.
- [x] Verification: root suite 263 passed; independent final proof review found no remaining P1/P2. Actual current-day SUCCESS Top10 inspected, replay hash matched, all ten prices/annual financials checked against raw facts, all ten fit in a 7,996-byte Feishu card, confirmed private Feishu send succeeded.
- [ ] After real-path checks, configure exactly one daily Codex automation with explicit Asia/Shanghai timezone; skip closed days in code, never rerun full TA/LLM. Verify saved scheduling and report receipt; use standing branch/PR/merge authorization, preserve dirty main.py.

## Review Focus / Acceptance

- Current financial captures prove only forward knowledge; old decisions consume frozen versions, not re-fetched current rows. Date-only announcements are visible D+1 Shanghai midnight.
- Current-day `bak_basic`/`stock_st` and stock identities distinguish real exclusion from missing quotes; all possibly ranking-changing unknowns block.
- Ranking is only market cap then ticker; no extra momentum/profit scoring, no backfit thresholds, no invented strong/weak state.
- Notifications check Feishu business response and never treat a timeout as confirmed delivery. Reports and delivery state are distinct; a resend does not reselect.
- Transport retries are finite, errors redacted, no tokens in snapshots; scheduling cannot drift with US daylight savings. Scheduler requires the local host/app to remain available; disclose that limitation.

## Execution Ledger

Working from `codex/small-cap-top10-spec`; existing `main.py`, `.agents/`, unrelated docs and harness database are user-owned and excluded from commits. The separate Top10 chat is idle; implementation is in this task to honor the latest direct request, without dispatching duplicate user-owned work.

Independent review identified stale zero-volume exclusions, uncertain newer annual fallback, refresh delivery identity, replay hash validation, send-time cutoff, partial report bundles and failed-cache recovery. These now have RED/GREEN regression tests. A later fetch of a retained pre-adjustment report cannot supersede a latest consolidated report merely because the HTTP request happened later.

Live verification: baseline suite 106 passed; final integrated suite 263 passed. Tushare permissions were verified without model calls. A transient ST transport failure recovered on a direct probe. `income_vip` offset pages returned 5,000 + 2,240 records with three overlaps, so finance collection now uses explicit report types and disjoint announcement-date partitions rather than accepting unstable page boundaries.

The first full capture correctly blocked a finance-only legacy identifier. A current official SZSE A-share XLSX census was then added solely for identity verification, covering all 2,905 active primary SZSE CNY identities. Negative membership requires both complete frozen censuses, consistent current provenance, contiguous unfiltered primary pages with terminal evidence, and no positive market/identity-feed contradiction. Unknown identities are not hardcoded away.

Accepted live run: data 2026-11-17, target 2026-11-18, sealed 18:03:33.053616+08:00, 10 candidates, snapshot `9ea644bf42eabf3461eb170534d8c36e60733f0207e75ba8517137b7aab49c7f`, selection hash `1e9d5d0442655ada94d954ccc6d47fe0f0cfcb26e581dee0f7dfb72b28a95382`. Capture took about 4m20s including transient retries. Replay under the final stricter selector produced the same hash. Raw prices, cap units, annual facts and cutoff were independently checked for all ten. One out-of-scope BSE suspension warning remains disclosed; it does not affect this mainboard ranking.

The accepted report was sent once to the configured private Feishu recipient; API-confirmed message ID is retained only in the local delivery ledger. No LLM or broker calls. Source snapshots are about 52 MiB/run before compression; V1 retains audit inputs and does not delete historical data automatically.

Scheduling tool rejected an immediate timezone-anchored RFC create and explicitly required a timezone-aware suggested create instead; do not silently schedule 19:30 in the host's America/Chicago zone or add unmanaged cron as a workaround. Verify the final saved schedule/activation state before claiming daily delivery is enabled.
