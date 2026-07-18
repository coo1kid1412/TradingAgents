# Intraday Price Overlay Design

## Problem

The A-share analysis path routes price data through Tushare `pro.daily`. During
the trading session this endpoint ends at T-1, but the resulting CSV is accepted
as a successful response. Market, Quant, profile, and technical indicators can
therefore analyze yesterday's close while same-day news describes today's move.

## Goal

For an analysis whose end date is today, append a validated same-day provisional
OHLCV bar when a fresh quote is available. Preserve the official historical
daily series as the base, disclose the quote source and timestamp, and fail back
to T-1 without fabricating a current bar when realtime data is unavailable.

## Architecture

### Realtime quote service

Create `tradingagents/dataflows/intraday_quote.py` with a normalized
`IntradayQuote` value object and provider orchestration:

1. Try Tushare `rt_k` when the current account has permission.
2. Fall back to the Sina single-symbol quote endpoint.
3. Cache successful quotes for 20 seconds and remember Tushare permission denial
   for the current process to avoid repeated rejected calls.

The service only returns a quote when all of these checks pass:

- instrument is an A-share;
- requested analysis date equals the current Asia/Shanghai date;
- quote trade date equals the requested analysis date;
- OHLC and last price are positive and internally coherent;
- quote timestamp is no more than five minutes old during continuous trading;
- the 11:30 quote remains valid during the lunch break;
- same-day final quotes remain valid after 15:00;
- no provisional bar is emitted before 09:25.

### Historical-plus-intraday merge

Tushare remains the authoritative source for completed daily bars. The Tushare
stock and indicator paths call the quote service after loading `pro.daily`:

- if the official series already contains the requested date, retain it;
- otherwise append one synthetic row using realtime open/high/low/last and
  cumulative volume;
- mark that row as provisional in response metadata;
- calculate technical indicators from the merged OHLCV series.

The merge is deterministic and side-effect free. It does not rewrite caches or
persist an incomplete daily bar as historical truth.

### Freshness propagation

Stock-data headers expose:

- `Price data status`: `intraday_provisional`, `official_daily`, or `t_minus_1`;
- `Latest bar source`;
- `Latest quote time` when applicable.

The Market Analyst must state this status near the beginning of its report and
include `price_data_date`, `price_data_time`, `price_data_status`, and
`price_data_source` in `SUMMARY`. Quant adds deterministic price-date/status
fields to `QUANT_SCORE`, so downstream RM/PM can distinguish provisional and
T-1 evidence.

## Failure Behavior

Realtime provider failures never abort a full stock analysis. They produce a
warning, keep the T-1 historical series, and label it `t_minus_1`. A stale,
wrong-date, malformed, zero-price, or pre-open quote is rejected. The service
must never silently describe T-1 data as realtime.

## Scope

This change covers A-share price OHLCV, technical indicators, and price-derived
Quant factors. Existing capital-flow APIs remain daily/T-1 and keep their own
freshness semantics. No realtime fund-flow approximation is introduced.

## Verification

- parser tests for valid and malformed Tushare/Sina payloads;
- freshness tests for continuous trading, lunch break, after close, pre-open,
  wrong date, and stale quotes;
- merge tests for append, official-row precedence, and no-quote fallback;
- Tushare stock/indicator integration tests with mocked providers;
- Quant metadata and Market prompt contract tests;
- live smoke call on 300308 when the market is open; outside market hours, use
  a provider connectivity smoke plus deterministic fixture replay.
