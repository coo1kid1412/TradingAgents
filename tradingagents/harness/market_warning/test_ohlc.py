"""Source-coherent OHLC selection and validation regressions."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys
from unittest import TestCase, main

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from tradingagents.harness.market_warning.domain import DataStatus, Market, MarketDataPoint
from tradingagents.harness.market_warning.ohlc import assess_ohlc_invariants


AS_OF = datetime(2026, 8, 31, 8, 0, tzinfo=timezone.utc)


def point(
    field: str,
    value: float,
    *,
    source: str,
    seconds_ago: int = 0,
    quality_status: DataStatus = DataStatus.FRESH,
) -> MarketDataPoint:
    at = AS_OF - timedelta(seconds=seconds_ago)
    return MarketDataPoint(
        market=Market.A_SHARE,
        symbol="INDEX",
        field=field,
        value=value,
        data_time=at,
        fetched_at=at,
        source=source,
        quality_status=quality_status,
    )


def candle(
    source: str,
    *,
    open_value: float = 100.0,
    high: float = 110.0,
    low: float = 90.0,
    close: float = 100.0,
    seconds_ago: int = 0,
    leg_statuses: dict[str, DataStatus] | None = None,
) -> tuple[MarketDataPoint, ...]:
    statuses = leg_statuses or {}
    return tuple(
        point(
            field,
            value,
            source=source,
            seconds_ago=seconds_ago,
            quality_status=statuses.get(field, DataStatus.FRESH),
        )
        for field, value in (
            ("open", open_value),
            ("high", high),
            ("low", low),
            ("index_price", close),
        )
    )


def assess(*points: MarketDataPoint):
    return assess_ohlc_invariants(
        points,
        Market.A_SHARE,
        "intraday",
        timestamp_skew=timedelta(seconds=120),
    )


class SourceCoherentOHLCTests(TestCase):
    def test_mixed_incomplete_sources_cannot_synthesize_complete_candle(self):
        result = assess(
            point("index_price", 100.0, source="close_source"),
            point("open", 100.0, source="close_source"),
            point("high", 110.0, source="range_source"),
            point("low", 90.0, source="range_source"),
        )

        self.assertFalse(result.complete)
        self.assertFalse(result.valid)
        self.assertLessEqual(len({item.source for item in result.points}), 1)

    def test_two_valid_sources_within_tolerance_select_one_complete_group(self):
        points = (
            point("open", 100.0, source="alpha", seconds_ago=20),
            point("high", 110.0, source="alpha"),
            point("low", 90.0, source="alpha", seconds_ago=20),
            point("index_price", 100.0, source="alpha"),
            point("open", 100.4, source="beta"),
            point("high", 110.4, source="beta", seconds_ago=20),
            point("low", 90.4, source="beta"),
            point("index_price", 100.4, source="beta", seconds_ago=20),
        )

        result = assess(*points)

        self.assertTrue(result.valid)
        self.assertFalse(result.conflicted)
        self.assertEqual(len({item.source for item in result.points}), 1)

    def test_structurally_invalid_complete_peer_fails_closed(self):
        result = assess(
            *candle("valid", open_value=100.0, high=100.2, low=99.8, close=100.0),
            *candle("invalid", open_value=100.1, high=99.9, low=100.0, close=100.05),
        )

        self.assertTrue(result.conflicted)
        self.assertFalse(result.valid)

    def test_stale_or_conflicted_selected_leg_is_never_usable(self):
        for status in (DataStatus.STALE, DataStatus.CONFLICTED, DataStatus.INSUFFICIENT):
            with self.subTest(status=status):
                result = assess(
                    *candle(
                        "single",
                        leg_statuses={"open": status},
                    )
                )

                self.assertFalse(result.valid)
                if status == DataStatus.STALE:
                    self.assertTrue(result.stale)
                elif status == DataStatus.INSUFFICIENT:
                    self.assertTrue(result.insufficient)
                else:
                    self.assertTrue(result.conflicted)

    def test_partial_and_shadow_legs_remain_eligible(self):
        for status in (DataStatus.PARTIAL, DataStatus.SHADOW):
            with self.subTest(status=status):
                result = assess(
                    *candle(
                        "single",
                        leg_statuses={"open": status},
                    )
                )

                self.assertTrue(result.valid)


if __name__ == "__main__":
    main()
