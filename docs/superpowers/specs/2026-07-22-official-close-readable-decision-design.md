# Official Close And Readable Decision Design

## Problem

The 2026-07-22 Liquidity Electronics report used the correct official close of
118.34, but fundamentals and downstream agents labelled the same value as an
intraday provisional quote. The final decision also exposed tool/audit language,
repeated the same thesis, and buried the distinct actions for empty and existing
positions in a 344-line report.

## Price Semantics

Price source precedence is deterministic:

1. A same-day `daily_basic` row is `official_daily` and remains authoritative.
2. A fresh intraday quote is used only when the same-day official row is absent.
3. The latest earlier official row is `t_minus_1` when no fresh quote is available.

An `official_daily` price must not also be rendered as a previous-close reference.
Downstream prompts must state that official daily data outranks a closing-auction
or post-close realtime quote for the same date.

## User Decision Output

`decision.md` remains the harness-compatible report and keeps `PM_SUMMARY` at the
end. Its user-facing body is normalized after model generation:

- The first viewport shows separate empty-position and existing-position actions.
- Internal workflow, tool verification, self-check, and archive sections are
  removed from the user body.
- The scenario table and actionable risk/monitoring sections remain.
- Internal tokens such as `SYS_*`, tool return field names, and prompt compliance
  instructions are removed or replaced with user language.
- Time stops use elapsed periods or actual dates, never ambiguous labels such as
  `6 月` when the milestone is in August.
- A no-buy decision may not contain an instruction to add a position.

## Compatibility

- `PM_SUMMARY` field names and values remain parseable by the existing harness.
- The formatter fails open: if expected headings are absent, it preserves content
  rather than returning an empty report.
- No new dependencies are introduced.

## Verification

- Unit tests reproduce the same-day official-close override bug.
- Formatter tests use a representative verbose PM report and assert that the
  action summary remains while audit/process text is absent.
- Existing entry-timing, intraday quote, harness extraction, and price tests pass.
- A `.venv` report smoke test confirms official-close labelling and readable final
  output on a real A-share analysis.
