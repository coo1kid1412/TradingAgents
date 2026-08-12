"""Point-in-time data quality contract tests."""

from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys
from unittest import TestCase, main
from zoneinfo import ZoneInfo

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
SHANGHAI = ZoneInfo("Asia/Shanghai")
NEW_YORK = ZoneInfo("America/New_York")


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
        secondary = point(
            "index_price",
            100 / (1 - QUALITY_POLICY_V1.cross_source_price_tolerance),
            source="secondary",
        )

        combined = combine_source_quotes(primary, secondary, QUALITY_POLICY_V1.cross_source_price_tolerance)

        self.assertEqual(combined.quality_status, DataStatus.FRESH)

    def test_price_deviation_just_over_boundary_is_conflicted(self):
        primary = point("index_price", 100, source="primary")
        secondary = point("index_price", 100 / (1 - QUALITY_POLICY_V1.cross_source_price_tolerance) + 0.001, source="secondary")

        combined = combine_source_quotes(primary, secondary, QUALITY_POLICY_V1.cross_source_price_tolerance)

        self.assertEqual(combined.quality_status, DataStatus.CONFLICTED)

    def test_price_conflict_result_is_independent_of_source_argument_order(self):
        over_boundary = 100 / (1 - QUALITY_POLICY_V1.cross_source_price_tolerance) + 0.001
        forward = combine_source_quotes(
            point("index_price", 100, source="primary"),
            point("index_price", over_boundary, source="secondary"),
            QUALITY_POLICY_V1.cross_source_price_tolerance,
        )
        reverse = combine_source_quotes(
            point("index_price", over_boundary, source="secondary"),
            point("index_price", 100, source="primary"),
            QUALITY_POLICY_V1.cross_source_price_tolerance,
        )

        self.assertEqual(forward.quality_status, DataStatus.CONFLICTED)
        self.assertEqual(reverse.quality_status, DataStatus.CONFLICTED)

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
    def test_complete_current_stale_ohlc_group_is_stale_and_unavailable(self):
        assessment = evaluate_data_quality(
            snapshot(
                point("index_price", 100.0, source="candle"),
                point("index_change_pct", 0.0, source="candle"),
                point("open", 100.0, source="candle", quality_status=DataStatus.STALE),
                point("high", 101.0, source="candle", quality_status=DataStatus.STALE),
                point("low", 99.0, source="candle", quality_status=DataStatus.STALE),
            ),
            QUALITY_POLICY_V1,
            NOW,
        )

        self.assertEqual(assessment.status, DataStatus.STALE)
        self.assertEqual(assessment.reliability_grade, "UNAVAILABLE")
        self.assertTrue(any("OHLC" in reason for reason in assessment.reasons))

    def test_complete_current_insufficient_ohlc_group_is_unavailable(self):
        assessment = evaluate_data_quality(
            snapshot(
                point("index_price", 100.0, source="candle"),
                point("index_change_pct", 0.0, source="candle"),
                point("open", 100.0, source="candle", quality_status=DataStatus.INSUFFICIENT),
                point("high", 101.0, source="candle"),
                point("low", 99.0, source="candle"),
            ),
            QUALITY_POLICY_V1,
            NOW,
        )

        self.assertEqual(assessment.status, DataStatus.INSUFFICIENT)
        self.assertEqual(assessment.reliability_grade, "UNAVAILABLE")

    def test_material_source_divergence_precedes_stale_ohlc(self):
        assessment = evaluate_data_quality(
            snapshot(
                *(
                    point(field, value, source="stale", quality_status=DataStatus.STALE)
                    for field, value in (
                        ("index_price", 100.0),
                        ("open", 100.0),
                        ("high", 101.0),
                        ("low", 99.0),
                    )
                ),
                *(
                    point(field, value, source="fresh")
                    for field, value in (
                        ("index_price", 110.0),
                        ("open", 110.0),
                        ("high", 111.0),
                        ("low", 109.0),
                    )
                ),
                point("index_change_pct", 0.0, source="fresh"),
            ),
            QUALITY_POLICY_V1,
            NOW,
        )

        self.assertEqual(assessment.status, DataStatus.CONFLICTED)
        self.assertEqual(assessment.reliability_grade, "UNAVAILABLE")

    def test_complete_aligned_invalid_ohlc_is_conflicted_and_unavailable(self):
        cases = (
            ("close_below_low", 100.0, 110.0, 90.0, 80.0),
            ("open_above_high", 111.0, 110.0, 90.0, 100.0),
            ("high_below_low", 100.0, 89.0, 90.0, 90.0),
            ("zero_open", 0.0, 110.0, 90.0, 100.0),
            ("negative_open", -1.0, 110.0, 90.0, 100.0),
            ("zero_high", 1.0, 0.0, 0.0, 0.0),
            ("negative_low", 100.0, 110.0, -1.0, 100.0),
            ("zero_close", 100.0, 110.0, 90.0, 0.0),
            ("negative_close", 100.0, 110.0, 90.0, -1.0),
            ("nan_open", float("nan"), 110.0, 90.0, 100.0),
            ("infinite_high", 100.0, float("inf"), 90.0, 100.0),
            ("boolean_low", 100.0, 110.0, False, 100.0),
            ("string_close", 100.0, 110.0, 90.0, "100"),
        )
        for name, open_value, high, low, close in cases:
            with self.subTest(name=name):
                assessment = evaluate_data_quality(
                    snapshot(
                        point("index_price", close),
                        point("index_change_pct", -1.0),
                        point("open", open_value),
                        point("high", high),
                        point("low", low),
                    ),
                    QUALITY_POLICY_V1,
                    NOW,
                )
                self.assertEqual(assessment.status, DataStatus.CONFLICTED)
                self.assertEqual(assessment.reliability_grade, "UNAVAILABLE")
                self.assertTrue(any("OHLC" in reason for reason in assessment.reasons))

    def test_valid_boundary_close_equal_low_is_not_conflicted(self):
        assessment = evaluate_data_quality(
            snapshot(
                point("index_price", 90.0),
                point("index_change_pct", -10.0),
                point("open", 100.0),
                point("high", 110.0),
                point("low", 90.0),
            ),
            QUALITY_POLICY_V1,
            NOW,
        )

        self.assertNotEqual(assessment.status, DataStatus.CONFLICTED)

    def test_incomplete_ohlc_keeps_missing_semantics_without_false_conflict(self):
        assessment = evaluate_data_quality(
            snapshot(
                point("index_price", 100.0),
                point("index_change_pct", 0.0),
                point("open", 100.0),
                point("high", 101.0),
            ),
            QUALITY_POLICY_V1,
            NOW,
        )

        self.assertNotEqual(assessment.status, DataStatus.CONFLICTED)

    def test_invalid_ohlc_on_non_benchmark_symbol_is_not_compared_to_benchmark(self):
        assessment = evaluate_data_quality(
            snapshot(
                point("index_price", 100.0, market=Market.US, symbol="SPX"),
                point("index_change_pct", 0.0, market=Market.US, symbol="SPX"),
                point("index_price", 80.0, market=Market.US, symbol="HYG"),
                point("open", 100.0, market=Market.US, symbol="HYG"),
                point("high", 110.0, market=Market.US, symbol="HYG"),
                point("low", 90.0, market=Market.US, symbol="HYG"),
                market=Market.US,
            ),
            QUALITY_POLICY_V1,
            NOW,
        )

        self.assertNotEqual(assessment.status, DataStatus.CONFLICTED)

    def test_invalid_primary_benchmark_is_not_hidden_by_valid_fallback_benchmark(self):
        assessment = evaluate_data_quality(
            snapshot(
                point("index_price", float("nan"), symbol="INDEX"),
                point("index_change_pct", 0.0, symbol="INDEX"),
                point("index_price", 100.0, symbol="000300.SH"),
                point("open", 100.0, symbol="000300.SH"),
                point("high", 101.0, symbol="000300.SH"),
                point("low", 99.0, symbol="000300.SH"),
            ),
            QUALITY_POLICY_V1,
            NOW,
        )

        self.assertEqual(assessment.status, DataStatus.CONFLICTED)
        self.assertEqual(assessment.reliability_grade, "UNAVAILABLE")

    def test_misaligned_ohlc_is_missing_not_cross_observation_conflict(self):
        assessment = evaluate_data_quality(
            snapshot(
                point("index_price", 80.0),
                point("index_change_pct", -20.0),
                point("open", 100.0, data_time=NOW - timedelta(days=1)),
                point("high", 110.0),
                point("low", 90.0),
            ),
            QUALITY_POLICY_V1,
            NOW,
        )

        self.assertNotEqual(assessment.status, DataStatus.CONFLICTED)

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
                point("index_price", 100, data_time=NOW - timedelta(seconds=300)),
                point("index_change_pct", -0.4, data_time=NOW - timedelta(seconds=300)),
            ),
            QUALITY_POLICY_V1,
            NOW,
        )
        over = evaluate_data_quality(
            snapshot(
                point("index_price", 100, data_time=NOW - timedelta(seconds=300, microseconds=1)),
                point("index_change_pct", -0.4, data_time=NOW - timedelta(seconds=300, microseconds=1)),
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

    def test_a_share_friday_close_is_fresh_on_monday_premarket_with_session_resolver(self):
        as_of = datetime(2026, 8, 3, 8, 30, tzinfo=SHANGHAI)
        data_time = datetime(2026, 7, 31, 15, 0, tzinfo=SHANGHAI)
        available_at = datetime(2026, 7, 31, 18, 0, tzinfo=SHANGHAI)
        raw = RawMarketSnapshot(
            market=Market.A_SHARE,
            as_of_time=as_of,
            session_slot="premarket",
            points=(
                point("index_price", 100, data_time=data_time, available_at=available_at),
                point("index_change_pct", -0.4, data_time=data_time, available_at=available_at),
            ),
        )

        policy = replace(QUALITY_POLICY_V1, previous_session=lambda market, current: current - timedelta(days=3))
        assessment = evaluate_data_quality(raw, policy, as_of)

        self.assertEqual(assessment.status, DataStatus.FRESH)

    def test_us_friday_close_is_fresh_on_monday_premarket_with_session_resolver(self):
        as_of = datetime(2026, 8, 3, 8, 30, tzinfo=NEW_YORK)
        data_time = datetime(2026, 7, 31, 16, 0, tzinfo=NEW_YORK)
        raw = RawMarketSnapshot(
            market=Market.US,
            as_of_time=as_of,
            session_slot="premarket",
            points=(
                point("index_price", 100, market=Market.US, symbol="SPX", data_time=data_time, available_at=data_time),
                point("index_change_pct", -0.4, market=Market.US, symbol="SPX", data_time=data_time, available_at=data_time),
            ),
        )

        policy = replace(QUALITY_POLICY_V1, previous_session=lambda market, current: current - timedelta(days=3))
        assessment = evaluate_data_quality(raw, policy, as_of)

        self.assertEqual(assessment.status, DataStatus.FRESH)

    def test_monday_close_and_postmarket_require_monday_observation(self):
        as_of = datetime(2026, 8, 3, 16, 30, tzinfo=NEW_YORK)
        data_time = datetime(2026, 7, 31, 16, 0, tzinfo=NEW_YORK)
        for session_slot in ("close", "post_market"):
            with self.subTest(session_slot=session_slot):
                raw = RawMarketSnapshot(
                    market=Market.US,
                    as_of_time=as_of,
                    session_slot=session_slot,
                    points=(
                        point("index_price", 100, market=Market.US, symbol="SPX", data_time=data_time, available_at=data_time),
                        point("index_change_pct", -0.4, market=Market.US, symbol="SPX", data_time=data_time, available_at=data_time),
                    ),
                )

                self.assertEqual(evaluate_data_quality(raw, QUALITY_POLICY_V1, as_of).status, DataStatus.STALE)

    def test_postmarket_accepts_same_session_daily_observation(self):
        as_of = datetime(2026, 8, 3, 17, 0, tzinfo=NEW_YORK)
        data_time = datetime(2026, 8, 3, 16, 0, tzinfo=NEW_YORK)
        raw = RawMarketSnapshot(
            market=Market.US,
            as_of_time=as_of,
            session_slot="post_market",
            points=(
                point("index_price", 100, market=Market.US, symbol="SPX", data_time=data_time, available_at=data_time),
                point("index_change_pct", -0.4, market=Market.US, symbol="SPX", data_time=data_time, available_at=data_time),
            ),
        )

        self.assertEqual(evaluate_data_quality(raw, QUALITY_POLICY_V1, as_of).status, DataStatus.FRESH)

    def test_premarket_resolver_rejects_observation_from_wrong_previous_session(self):
        as_of = datetime(2026, 8, 3, 8, 30, tzinfo=NEW_YORK)
        friday = datetime(2026, 7, 31, 16, 0, tzinfo=NEW_YORK)
        raw = RawMarketSnapshot(
            market=Market.US,
            as_of_time=as_of,
            session_slot="premarket",
            points=(
                point("index_price", 100, market=Market.US, symbol="SPX", data_time=friday, available_at=friday),
                point("index_change_pct", -0.4, market=Market.US, symbol="SPX", data_time=friday, available_at=friday),
            ),
        )
        policy = replace(QUALITY_POLICY_V1, previous_session=lambda market, current: current - timedelta(days=4))

        self.assertEqual(evaluate_data_quality(raw, policy, as_of).status, DataStatus.STALE)

    def test_same_t_minus_one_daily_close_is_stale_during_intraday_session(self):
        cases = (
            (Market.A_SHARE, SHANGHAI, 15, "000001.SH"),
            (Market.US, NEW_YORK, 16, "SPX"),
        )
        for market, zone, close_hour, symbol in cases:
            with self.subTest(market=market):
                as_of = datetime(2026, 8, 3, 10, 0, tzinfo=zone)
                data_time = datetime(2026, 7, 31, close_hour, 0, tzinfo=zone)
                raw = RawMarketSnapshot(
                    market=market,
                    as_of_time=as_of,
                    session_slot="intraday",
                    points=(
                        point("index_price", 100, market=market, symbol=symbol, data_time=data_time, available_at=data_time),
                        point("index_change_pct", -0.4, market=market, symbol=symbol, data_time=data_time, available_at=data_time),
                    ),
                )

                assessment = evaluate_data_quality(raw, QUALITY_POLICY_V1, as_of)

                self.assertEqual(assessment.status, DataStatus.STALE)

    def test_premarket_fallback_allows_fourteen_days_but_not_fifteen(self):
        as_of = datetime(2026, 8, 17, 8, 30, tzinfo=NEW_YORK)
        for age_days, expected in ((14, DataStatus.FRESH), (15, DataStatus.STALE)):
            with self.subTest(age_days=age_days):
                data_time = datetime(2026, 8, 17 - age_days, 16, 0, tzinfo=NEW_YORK)
                raw = RawMarketSnapshot(
                    market=Market.US,
                    as_of_time=as_of,
                    session_slot="premarket",
                    points=(
                        point("index_price", 100, market=Market.US, symbol="SPX", data_time=data_time, available_at=data_time),
                        point("index_change_pct", -0.4, market=Market.US, symbol="SPX", data_time=data_time, available_at=data_time),
                    ),
                )

                self.assertEqual(evaluate_data_quality(raw, QUALITY_POLICY_V1, as_of).status, expected)

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

    def test_timestamp_skew_applies_to_same_named_non_price_core_field(self):
        at_boundary = evaluate_data_quality(
            snapshot(
                point("index_price", 100, source="primary"),
                point("index_price", 100, source="secondary"),
                point("index_change_pct", -0.4, source="primary", data_time=NOW - timedelta(seconds=300)),
                point("index_change_pct", -0.4, source="secondary", data_time=NOW - timedelta(seconds=180)),
            ),
            QUALITY_POLICY_V1,
            NOW,
        )
        over_boundary = evaluate_data_quality(
            snapshot(
                point("index_price", 100, source="primary"),
                point("index_price", 100, source="secondary"),
                point("index_change_pct", -0.4, source="primary", data_time=NOW - timedelta(seconds=300)),
                point("index_change_pct", -0.4, source="secondary", data_time=NOW - timedelta(seconds=179)),
            ),
            QUALITY_POLICY_V1,
            NOW,
        )

        self.assertNotEqual(at_boundary.status, DataStatus.CONFLICTED)
        self.assertEqual(over_boundary.status, DataStatus.CONFLICTED)

    def test_large_breadth_and_volatility_differences_with_same_time_do_not_conflict(self):
        assessment = evaluate_data_quality(
            snapshot(
                point("index_price", 100, source="primary"),
                point("index_change_pct", -0.4, source="primary"),
                point("index_price", 100, source="secondary"),
                point("index_change_pct", -0.4, source="secondary"),
                point("breadth_up_pct", 1, source="primary"),
                point("breadth_up_pct", 99, source="secondary"),
                point("volatility", 1, source="primary"),
                point("volatility", 1000, source="secondary"),
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
