"""Point-in-time data quality contract tests."""

from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys
from unittest import TestCase, main

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from tradingagents.harness.market_warning.domain import DataStatus, Market, MarketDataPoint, RawMarketSnapshot
from tradingagents.harness.market_warning.quality import (
    QUALITY_POLICY_V1,
    combine_source_quotes,
    evaluate_data_quality,
    select_point_in_time,
)


UTC = timezone.utc
NOW = datetime(2026, 8, 3, 16, 0, tzinfo=UTC)


def point(
    field: str,
    value,
    *,
    market: Market = Market.A_SHARE,
    symbol: str = "000300",
    source: str = "primary",
    quality_status: DataStatus = DataStatus.FRESH,
    data_time: datetime = NOW - timedelta(minutes=5),
    fetched_at: datetime = NOW,
    available_at: datetime | None = None,
) -> MarketDataPoint:
    return MarketDataPoint(
        market=market,
        symbol=symbol,
        field=field,
        value=value,
        data_time=data_time,
        fetched_at=fetched_at,
        source=source,
        quality_status=quality_status,
        available_at=available_at,
    )


def snapshot(*points: MarketDataPoint, market: Market = Market.A_SHARE, session_slot: str = "intraday"):
    return RawMarketSnapshot(
        market=market,
        as_of_time=NOW,
        session_slot=session_slot,
        points=points,
    )


class PointInTimeSelectionTests(TestCase):
    def test_excludes_record_available_after_as_of_even_when_market_date_is_correct(self):
        late = point("index_price", 100, available_at=NOW + timedelta(seconds=1))
        visible = point("index_price", 99, source="secondary", available_at=NOW - timedelta(seconds=1))

        selected = select_point_in_time((late, visible), NOW)

        self.assertEqual(selected, (visible,))


class SourceCombinationTests(TestCase):
    def test_combines_quotes_within_tolerance(self):
        primary = point("index_price", 100, source="primary")
        secondary = point("index_price", 100.4, source="secondary")

        combined = combine_source_quotes(primary, secondary, QUALITY_POLICY_V1.cross_source_price_tolerance)

        self.assertEqual(combined.quality_status, DataStatus.FRESH)
        self.assertEqual(combined.value, 100)
        self.assertEqual(combined.source, "primary+secondary")

    def test_marks_quotes_over_tolerance_as_conflicted(self):
        primary = point("index_price", 100, source="primary")
        secondary = point("index_price", 101, source="secondary")

        combined = combine_source_quotes(primary, secondary, QUALITY_POLICY_V1.cross_source_price_tolerance)

        self.assertEqual(combined.quality_status, DataStatus.CONFLICTED)

    def test_timestamp_skew_at_boundary_is_allowed(self):
        primary = point("index_price", 100, source="primary", data_time=NOW - timedelta(seconds=300))
        secondary = point("index_price", 100, source="secondary", data_time=NOW - timedelta(seconds=180))

        combined = combine_source_quotes(primary, secondary, QUALITY_POLICY_V1.cross_source_price_tolerance)

        self.assertEqual(combined.quality_status, DataStatus.FRESH)
        self.assertEqual(combined.source, "primary+secondary")

    def test_timestamp_skew_just_over_boundary_is_conflicted(self):
        primary = point("index_price", 100, source="primary", data_time=NOW - timedelta(seconds=300))
        secondary = point("index_price", 100, source="secondary", data_time=NOW - timedelta(seconds=179))

        combined = combine_source_quotes(primary, secondary, QUALITY_POLICY_V1.cross_source_price_tolerance)

        self.assertEqual(combined.quality_status, DataStatus.CONFLICTED)

    def test_price_deviation_at_boundary_is_allowed(self):
        primary = point("index_price", 100, source="primary")
        secondary = point("index_price", 100.5, source="secondary")

        combined = combine_source_quotes(primary, secondary, QUALITY_POLICY_V1.cross_source_price_tolerance)

        self.assertEqual(combined.quality_status, DataStatus.FRESH)

    def test_price_deviation_just_over_boundary_is_conflicted(self):
        primary = point("index_price", 100, source="primary")
        secondary = point("index_price", 100.501, source="secondary")

        combined = combine_source_quotes(primary, secondary, QUALITY_POLICY_V1.cross_source_price_tolerance)

        self.assertEqual(combined.quality_status, DataStatus.CONFLICTED)

    def test_combined_lineage_is_still_dual_source_to_quality_layer(self):
        combined_price = combine_source_quotes(
            point("index_price", 100, market=Market.US, symbol="SPX", source="primary"),
            point("index_price", 100, market=Market.US, symbol="SPX", source="secondary"),
            QUALITY_POLICY_V1.cross_source_price_tolerance,
        )
        combined_change = combine_source_quotes(
            point("index_change_pct", -0.4, market=Market.US, symbol="SPX", source="primary"),
            point("index_change_pct", -0.4, market=Market.US, symbol="SPX", source="secondary"),
            QUALITY_POLICY_V1.cross_source_price_tolerance,
        )

        assessment = evaluate_data_quality(
            snapshot(combined_price, combined_change, market=Market.US, session_slot="intraday"),
            QUALITY_POLICY_V1,
            NOW,
        )

        self.assertEqual(assessment.status, DataStatus.FRESH)
        self.assertEqual(assessment.core_source_count, 2)


class QualityEvaluationTests(TestCase):
    def test_fresh_dual_source_core_complete_with_optional_coverage_is_grade_a(self):
        points = (
            point("index_price", 100, source="primary"),
            point("index_change_pct", -0.4, source="primary"),
            point("index_price", 100, source="secondary"),
            point("index_change_pct", -0.4, source="secondary"),
            point("breadth_up_pct", 30, source="primary"),
            point("breadth_above_ma20_pct", 25, source="primary"),
            point("volatility", 22, source="primary"),
            point("volume", 1_000_000, source="primary"),
        )

        assessment = evaluate_data_quality(snapshot(*points), QUALITY_POLICY_V1, NOW)

        self.assertEqual(assessment.status, DataStatus.FRESH)
        self.assertEqual(assessment.reliability_grade, "A")
        self.assertEqual(assessment.core_coverage, 1.0)
        self.assertGreaterEqual(assessment.optional_coverage, 0.70)

    def test_optional_second_source_cannot_upgrade_core_data_to_grade_a(self):
        points = (
            point("index_price", 100, source="primary"),
            point("index_change_pct", -0.4, source="primary"),
            point("index_price", 100, source="secondary"),
            point("breadth_up_pct", 30, source="secondary"),
            point("breadth_above_ma20_pct", 25, source="secondary"),
            point("volatility", 22, source="secondary"),
            point("volume", 1_000_000, source="secondary"),
        )

        assessment = evaluate_data_quality(snapshot(*points), QUALITY_POLICY_V1, NOW)

        self.assertEqual(assessment.status, DataStatus.FRESH)
        self.assertEqual(assessment.reliability_grade, "B")

    def test_partial_core_data_is_usable_but_grade_c(self):
        assessment = evaluate_data_quality(
            snapshot(point("index_price", 100)),
            QUALITY_POLICY_V1,
            NOW,
        )

        self.assertEqual(assessment.status, DataStatus.PARTIAL)
        self.assertEqual(assessment.reliability_grade, "C")

    def test_explicit_stale_core_point_forces_stale_and_unavailable(self):
        assessment = evaluate_data_quality(
            snapshot(
                point("index_price", 100, quality_status=DataStatus.STALE),
                point("index_change_pct", -0.4),
            ),
            QUALITY_POLICY_V1,
            NOW,
        )

        self.assertEqual(assessment.status, DataStatus.STALE)
        self.assertEqual(assessment.reliability_grade, "UNAVAILABLE")

    def test_core_age_at_exact_boundary_is_fresh_but_just_over_is_stale(self):
        exact = evaluate_data_quality(
            snapshot(
                point("index_price", 100, data_time=NOW - timedelta(seconds=600)),
                point("index_change_pct", -0.4, data_time=NOW - timedelta(seconds=600)),
            ),
            QUALITY_POLICY_V1,
            NOW,
        )
        over = evaluate_data_quality(
            snapshot(
                point("index_price", 100, data_time=NOW - timedelta(seconds=600, microseconds=1)),
                point("index_change_pct", -0.4, data_time=NOW - timedelta(seconds=600, microseconds=1)),
            ),
            QUALITY_POLICY_V1,
            NOW,
        )

        self.assertEqual(exact.status, DataStatus.FRESH)
        self.assertEqual(over.status, DataStatus.STALE)

    def test_optional_coverage_at_seventy_percent_is_a_but_just_below_is_b(self):
        optional_fields = (
            "optional_1",
            "optional_2",
            "optional_3",
            "optional_4",
            "optional_5",
            "optional_6",
            "optional_7",
            "optional_8",
            "optional_9",
            "optional_10",
        )
        policy = replace(QUALITY_POLICY_V1, optional_fields=optional_fields)
        core = (
            point("index_price", 100, source="primary"),
            point("index_change_pct", -0.4, source="primary"),
            point("index_price", 100, source="secondary"),
            point("index_change_pct", -0.4, source="secondary"),
        )
        at_boundary = evaluate_data_quality(
            snapshot(*core, *(point(field, 1) for field in optional_fields[:7])),
            policy,
            NOW,
        )
        below_boundary = evaluate_data_quality(
            snapshot(*core, *(point(field, 1) for field in optional_fields[:6])),
            policy,
            NOW,
        )

        self.assertEqual(at_boundary.optional_coverage, 0.70)
        self.assertEqual(at_boundary.reliability_grade, "A")
        self.assertLess(below_boundary.optional_coverage, 0.70)
        self.assertEqual(below_boundary.reliability_grade, "B")

    def test_stale_data_time_is_unavailable_even_when_fetched_today(self):
        cached_close = (
            point(
                "index_price",
                100,
                data_time=NOW - timedelta(days=1),
                fetched_at=NOW - timedelta(minutes=1),
            ),
            point(
                "index_change_pct",
                -0.4,
                data_time=NOW - timedelta(days=1),
                fetched_at=NOW - timedelta(minutes=1),
            ),
        )

        assessment = evaluate_data_quality(
            snapshot(*cached_close, market=Market.US, session_slot="close"),
            QUALITY_POLICY_V1,
            NOW,
        )

        self.assertEqual(assessment.status, DataStatus.STALE)
        self.assertEqual(assessment.reliability_grade, "UNAVAILABLE")

    def test_conflicting_sources_force_unavailable(self):
        assessment = evaluate_data_quality(
            snapshot(
                point("index_price", 100, source="primary"),
                point("index_price", 101, source="secondary"),
                point("index_change_pct", -0.4, source="primary"),
                point("index_change_pct", -0.4, source="secondary"),
            ),
            QUALITY_POLICY_V1,
            NOW,
        )

        self.assertEqual(assessment.status, DataStatus.CONFLICTED)
        self.assertEqual(assessment.reliability_grade, "UNAVAILABLE")

    def test_optional_volume_difference_does_not_conflict_the_snapshot(self):
        assessment = evaluate_data_quality(
            snapshot(
                point("index_price", 100, source="primary"),
                point("index_change_pct", -0.4, source="primary"),
                point("index_price", 100, source="secondary"),
                point("index_change_pct", -0.4, source="secondary"),
                point("volume", 100, source="primary"),
                point("volume", 110, source="secondary"),
            ),
            QUALITY_POLICY_V1,
            NOW,
        )

        self.assertNotEqual(assessment.status, DataStatus.CONFLICTED)

    def test_no_core_fields_is_insufficient(self):
        assessment = evaluate_data_quality(
            snapshot(point("volatility", 22)),
            QUALITY_POLICY_V1,
            NOW,
        )

        self.assertEqual(assessment.status, DataStatus.INSUFFICIENT)
        self.assertEqual(assessment.reliability_grade, "UNAVAILABLE")

    def test_us_intraday_single_source_is_shadow(self):
        assessment = evaluate_data_quality(
            snapshot(
                point("index_price", 100, market=Market.US, symbol="SPX"),
                point("index_change_pct", -0.4, market=Market.US, symbol="SPX"),
                market=Market.US,
                session_slot="intraday",
            ),
            QUALITY_POLICY_V1,
            NOW,
        )

        self.assertEqual(assessment.status, DataStatus.SHADOW)
        self.assertEqual(assessment.reliability_grade, "C")

    def test_three_index_votes_do_not_fill_breadth_fields(self):
        points = tuple(
            point("index_change_pct", -0.4, symbol=symbol)
            for symbol in ("000300", "000905", "000852")
        )

        assessment = evaluate_data_quality(snapshot(*points), QUALITY_POLICY_V1, NOW)

        self.assertEqual(assessment.optional_coverage, 0.0)
        self.assertNotIn("breadth_up_pct", assessment.covered_optional_fields)
        self.assertNotIn("breadth_above_ma20_pct", assessment.covered_optional_fields)

    def test_excludes_data_time_after_as_of_time(self):
        future = point("index_price", 101, data_time=NOW + timedelta(microseconds=1))
        visible = point("index_price", 100, source="secondary")

        selected = select_point_in_time((future, visible), NOW)

        self.assertEqual(selected, (visible,))


if __name__ == "__main__":
    main()
