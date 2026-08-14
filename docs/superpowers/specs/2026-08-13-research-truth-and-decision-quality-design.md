# Research Truth and Decision Quality Design

## Goal

Prevent unverified claims, incorrect reporting windows, inconsistent financial
bases, and missing operational data from silently becoming investment facts.
Keep the user report concise while preserving a full audit artifact.

## Architecture

1. The data layer owns official disclosures, reporting calendars, price facts,
   financial bases, and market-risk availability.
2. The evidence layer assigns source tier, verification status, decision
   eligibility, freshness, and claim-level provenance before research debate.
3. LLM agents may reason over eligible evidence, but may not promote social or
   unverified claims into valuation, position, or event triggers.
4. Deterministic exit guards reconcile risk caps, market state, technical
   structure, and report timing before writing the final decision.
5. `complete_report.md` is the concise user artifact; `audit_report.md` stores
   analyst, debate, risk, and PM detail.

## Data Contracts

- A-share announcement discovery looks back 120 calendar days.
- Tushare `forecast` supplies structured official performance-forecast facts.
- News events carry `source_tier`, `source_name`, `source_url`, `document_id`,
  and `verification_status`.
- Social and unverified claims are leads only and are not decision eligible.
- Reporting periods use deterministic legal disclosure windows when no exact
  exchange reservation date is available.
- Missing market-risk data means `unknown` plus an execution block, not a
  bearish market opinion.
- Quant coverage distinguishes complete and partial factors.
- A short rebound below medium-term trend is not classified as healthy trend.

## Verification

- Unit tests reproduce every 603629 failure mode before implementation.
- Existing attribution, timing, market-risk, factor, and report tests remain
  green.
- A full stock smoke test runs under the project `.venv` and the generated
  reports are inspected for price, provenance, timing, action, and readability.
