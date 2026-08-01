"""Hand-calculated contract tests for deterministic market-warning features."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import pstdev
import sys
from unittest import TestCase, main

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from tradingagents.harness.market_warning.domain import DataStatus, Market, MarketDataPoint, MarketPhase, RawMarketSnapshot
from tradingagents.harness.market_warning.features import (
    AShareFeatureStrategy,
    FEATURE_METADATA,
    FEATURE_VERSION,
    USFeatureStrategy,
    derive_market_phase,
)


UTC = timezone.utc
AS_OF = datetime(2026, 8, 31, 16, 0, tzinfo=UTC)


def point(
    field: str,
    value: float,
    at: datetime,
    *,
    market: Market = Market.A_SHARE,
    symbol: str = "INDEX",
    source: str = "fixture",
) -> MarketDataPoint:
    return MarketDataPoint(
        market=market,
        symbol=symbol,
        field=field,
        value=value,
        data_time=at,
        fetched_at=at,
        source=source,
    )


def snapshot(
    *points: MarketDataPoint,
    market: Market = Market.A_SHARE,
    at: datetime = AS_OF,
    session_slot: str = "close",
) -> RawMarketSnapshot:
    return RawMarketSnapshot(market=market, as_of_time=at, session_slot=session_slot, points=points)


def history_from_series(
    field: str,
    values: list[float],
    *,
    market: Market = Market.A_SHARE,
    symbol: str = "INDEX",
    source: str = "fixture",
) -> tuple[RawMarketSnapshot, ...]:
    return tuple(
        snapshot(
            point(field, value, AS_OF - timedelta(days=len(values) - index), market=market, symbol=symbol, source=source),
            market=market,
            at=AS_OF - timedelta(days=len(values) - index),
        )
        for index, value in enumerate(values)
    )


class CommonFeatureTests(TestCase):
    def test_common_price_features_use_declared_rolling_horizons(self):
        closes = [100.0] * 19 + [110.0, 100.0]
        history = history_from_series("index_price", closes[:-1])
        raw = snapshot(
            point("index_price", closes[-1], AS_OF),
            point("index_change_pct", -9.0909090909, AS_OF),
            point("open", 105.0, AS_OF),
            point("high", 110.0, AS_OF),
            point("low", 90.0, AS_OF),
            point("volume", 50.0, AS_OF),
        )

        result = AShareFeatureStrategy().build(raw, history)

        self.assertAlmostEqual(result.features["return_1d"], -0.0909090909)
        self.assertAlmostEqual(result.features["drawdown_20d"], 100 / 110 - 1)
        self.assertAlmostEqual(result.features["ma20_distance"], 100 / 100.5 - 1)
        self.assertAlmostEqual(result.features["ma20_slope"], 0.0)
        self.assertAlmostEqual(result.features["range_pct"], 20 / 105)
        self.assertAlmostEqual(result.features["close_location"], 0.5)
        self.assertAlmostEqual(result.features["return_5d"], 0.0)
        self.assertIsNone(result.features["ma50_distance"])
        self.assertIsNotNone(result.features["realized_volatility_5d"])
        self.assertIsNone(result.features["volume_zscore_20d"])

    def test_realized_volatility_ratio_and_volume_zscore_are_hand_calculated(self):
        closes = [100.0, 110.0, 99.0, 108.9, 98.01, 107.811]
        volumes = [10.0] * 19 + [20.0, 30.0]
        history = history_from_series("index_price", closes[:-1]) + history_from_series("volume", volumes[:-1])
        raw = snapshot(
            point("index_price", closes[-1], AS_OF),
            point("index_change_pct", 10.0, AS_OF),
            point("volume", volumes[-1], AS_OF),
        )

        result = AShareFeatureStrategy().build(raw, history)

        self.assertAlmostEqual(result.features["realized_volatility_5d"], 0.0979795897)
        self.assertIsNone(result.features["volatility_ratio_5d_20d"])
        self.assertAlmostEqual(result.features["volume_zscore_20d"], (30 - 11.5) / (22.75 ** 0.5))

    def test_phase_boundary_and_metadata_contract(self):
        self.assertEqual(derive_market_phase(-0.049999), MarketPhase.FIRST_SHOCK)
        self.assertEqual(derive_market_phase(-0.05), MarketPhase.CONTINUATION)
        self.assertIsNone(derive_market_phase(None))
        self.assertEqual(FEATURE_VERSION, "market-warning-v1")
        for name, metadata in FEATURE_METADATA.items():
            with self.subTest(name=name):
                self.assertEqual(set(metadata), {"source", "availability", "missing", "direction", "unit", "version"})
                self.assertEqual(metadata["version"], FEATURE_VERSION)

    def test_missing_drawdown_does_not_invent_market_phase(self):
        raw = snapshot(point("index_price", 100.0, AS_OF), point("index_change_pct", 0.0, AS_OF))

        result = AShareFeatureStrategy().build(raw, ())

        self.assertIsNone(result.features["drawdown_20d"])
        self.assertIsNone(result.features["market_phase"])

    def test_common_features_require_an_explicit_market_benchmark(self):
        market = Market.US
        history = history_from_series("index_price", [100.0, 101.0], market=market, symbol="HYG")
        raw = snapshot(
            point("index_price", 98.0, AS_OF, market=market, symbol="HYG"),
            point("index_price", 100.0, AS_OF, market=market, symbol="LQD"),
            point("index_price", 96.0, AS_OF, market=market, symbol="SOXX"),
            point("open", 97.0, AS_OF, market=market, symbol="HYG"),
            point("high", 99.0, AS_OF, market=market, symbol="HYG"),
            point("low", 95.0, AS_OF, market=market, symbol="HYG"),
            market=market,
            session_slot="intraday",
        )

        result = USFeatureStrategy().build(raw, history)

        for name in ("return_1d", "drawdown_20d", "ma20_distance", "realized_volatility_5d", "range_pct"):
            self.assertIsNone(result.features[name], name)

    def test_intraday_and_close_snapshots_on_one_market_day_count_once(self):
        market = Market.A_SHARE
        close_time = datetime(2026, 8, 31, 7, 0, tzinfo=UTC)
        history = tuple(
            snapshot(
                point("index_price", 100.0, close_time - timedelta(days=20 - index), market=market),
                market=market,
                at=close_time - timedelta(days=20 - index),
            )
            for index in range(19)
        ) + (
            snapshot(
                point("index_price", 90.0, close_time - timedelta(hours=3), market=market),
                market=market,
                at=close_time - timedelta(hours=3),
                session_slot="intraday",
            ),
        )
        raw = snapshot(
            point("index_price", 100.0, close_time, market=market),
            point("index_change_pct", 0.0, close_time, market=market),
            market=market,
            at=close_time,
        )

        result = AShareFeatureStrategy().build(raw, history)

        self.assertIsNone(result.features["return_20d"])
        self.assertEqual(result.features["drawdown_20d"], 0.0)

    def test_same_market_day_alignment_is_required_for_ohlc_and_multi_leg_features(self):
        market = Market.US
        history = tuple(
            snapshot(
                point("index_price", 100.0, AS_OF - timedelta(days=days), market=market, symbol=symbol),
                market=market,
                at=AS_OF - timedelta(days=days),
            )
            for days in range(1, 6)
            for symbol in ("SPX", "HYG", "LQD", "RUT")
        )
        raw = snapshot(
            point("index_price", 100.0, AS_OF, market=market, symbol="SPX"),
            point("index_change_pct", 0.0, AS_OF, market=market, symbol="SPX"),
            point("open", 98.0, AS_OF - timedelta(days=1), market=market, symbol="SPX"),
            point("high", 101.0, AS_OF, market=market, symbol="SPX"),
            point("low", 97.0, AS_OF, market=market, symbol="SPX"),
            point("index_price", 99.0, AS_OF, market=market, symbol="HYG"),
            point("index_price", 100.0, AS_OF - timedelta(days=1), market=market, symbol="LQD"),
            point("vix", 24.0, AS_OF, market=market, symbol="VIX"),
            point("vix3m", 23.0, AS_OF - timedelta(days=1), market=market, symbol="VIX3M"),
            point("index_price", 95.0, AS_OF - timedelta(days=1), market=market, symbol="RUT"),
            market=market,
            session_slot="close",
        )

        result = USFeatureStrategy().build(raw, history)

        for name in ("range_pct", "close_location", "hyg_lqd_relative_return_5d", "vix_vix3m_ratio", "russell_spx_relative_return_5d"):
            self.assertIsNone(result.features[name], name)

    def test_metadata_declares_actual_endpoint_requirements(self):
        self.assertEqual(FEATURE_METADATA["return_1d"]["availability"], "2 observations visible by as_of")
        self.assertEqual(FEATURE_METADATA["return_20d"]["availability"], "21 observations visible by as_of")
        self.assertEqual(FEATURE_METADATA["margin_balance_growth_20d"]["availability"], "21 disclosed observations visible by as_of")

    def test_nonzero_ma_slope_and_nonempty_volatility_ratio_use_rolling_inputs(self):
        closes = [100.0, 110.0, 99.0, 108.9, 98.01, 107.811] + [100.0 + index for index in range(15)]
        history = history_from_series("index_price", closes[:-1])
        raw = snapshot(
            point("index_price", closes[-1], AS_OF),
            point("index_change_pct", closes[-1] / closes[-2] - 1, AS_OF),
        )

        result = AShareFeatureStrategy().build(raw, history)

        returns = [closes[index] / closes[index - 1] - 1 for index in range(1, len(closes))]
        expected_ratio = pstdev(returns[-5:]) / pstdev(returns[-20:])
        expected_slope = (sum(closes[-20:]) / 20) / (sum(closes[-21:-1]) / 20) - 1
        self.assertAlmostEqual(result.features["volatility_ratio_5d_20d"], expected_ratio)
        self.assertAlmostEqual(result.features["ma20_slope"], expected_slope)

    def test_prior_close_is_valid_only_for_premarket_not_intraday(self):
        market = Market.US
        as_of = datetime(2026, 9, 1, 14, 0, tzinfo=UTC)
        prior_close = as_of - timedelta(days=1)
        points = (
            point("index_price", 100.0, prior_close, market=market, symbol="SPX"),
            point("index_change_pct", 0.0, prior_close, market=market, symbol="SPX"),
            point("open", 99.0, prior_close, market=market, symbol="SPX"),
            point("high", 101.0, prior_close, market=market, symbol="SPX"),
            point("low", 98.0, prior_close, market=market, symbol="SPX"),
        )
        premarket = snapshot(*points, market=market, at=as_of, session_slot="premarket")
        intraday = snapshot(*points, market=market, at=as_of, session_slot="intraday")

        self.assertAlmostEqual(USFeatureStrategy().build(premarket, ()).features["range_pct"], 3 / 99)
        self.assertIsNone(USFeatureStrategy().build(intraday, ()).features["range_pct"])

    def test_window_outside_gaps_do_not_invalidate_current_five_or_twenty_day_returns(self):
        market = Market.US
        history = tuple(
            snapshot(
                point("index_price", 100.0, AS_OF - timedelta(days=days), market=market, symbol="SPX"),
                market=market,
                at=AS_OF - timedelta(days=days),
            )
            for days in range(1, 23)
        )
        old_gap = snapshot(market=market, at=AS_OF - timedelta(days=22))
        raw = snapshot(
            point("index_price", 110.0, AS_OF, market=market, symbol="SPX"),
            point("index_change_pct", 10.0, AS_OF, market=market, symbol="SPX"),
            market=market,
            session_slot="intraday",
        )

        result = USFeatureStrategy().build(raw, (old_gap, *history))

        self.assertAlmostEqual(result.features["return_5d"], 0.1)
        self.assertAlmostEqual(result.features["return_20d"], 0.1)


class AShareFeatureTests(TestCase):
    def test_a_share_features_preserve_true_breadth_and_measure_transition(self):
        history = (
            *history_from_series("margin_balance", [100.0] * 19 + [120.0]),
            *history_from_series("valuation", list(range(1, 21))),
            *history_from_series("turnover_pct", list(range(1, 21))),
            *history_from_series("shibor_3m", [2.0] * 19 + [2.1]),
        )
        raw = snapshot(
            point("index_price", 100.0, AS_OF),
            point("index_change_pct", -1.0, AS_OF),
            point("breadth_up_pct", 25.0, AS_OF),
            point("breadth_above_ma20_pct", 30.0, AS_OF),
            point("margin_balance", 110.0, AS_OF),
            point("margin_buying", 15.0, AS_OF),
            point("valuation", 21.0, AS_OF),
            point("turnover_pct", 21.0, AS_OF),
            point("limit_down_pct", 2.0, AS_OF),
            point("shibor_3m", 2.3, AS_OF),
        )

        result = AShareFeatureStrategy().build(raw, history)

        self.assertEqual(result.features["breadth_up_pct"], 25.0)
        self.assertEqual(result.features["breadth_above_ma20_pct"], 30.0)
        self.assertAlmostEqual(result.features["margin_balance_growth_20d"], 0.1)
        self.assertTrue(result.features["margin_balance_contracting_from_high"])
        self.assertEqual(result.features["valuation_percentile_20d"], 1.0)
        self.assertEqual(result.features["turnover_percentile_20d"], 1.0)
        self.assertEqual(result.features["limit_down_pct"], 2.0)
        self.assertAlmostEqual(result.features["shibor_3m_change_20d"], 0.3)

    def test_missing_breadth_remains_none_and_is_auditable(self):
        raw = snapshot(
            point("index_price", 100.0, AS_OF),
            point("index_change_pct", 0.0, AS_OF),
        )

        result = AShareFeatureStrategy().build(raw, ())

        self.assertIsNone(result.features["breadth_up_pct"])
        evidence = next(item for item in result.evidence if item.evidence_id.split(":")[2] == "breadth_up_pct")
        self.assertIsNone(evidence.value)
        self.assertIn("unavailable", evidence.summary)


class USFeatureTests(TestCase):
    def test_us_relative_strength_volatility_and_transition_features(self):
        market = Market.US
        history = (
            *history_from_series("index_price", [100.0] * 20, market=market, symbol="HYG"),
            *history_from_series("index_price", [100.0] * 20, market=market, symbol="LQD"),
            *history_from_series("index_price", [100.0] * 20, market=market, symbol="SPX"),
            *history_from_series("index_price", [100.0] * 20, market=market, symbol="RUT"),
            *history_from_series("index_price", [100.0] * 20, market=market, symbol="NDX"),
            *history_from_series("index_price", [100.0] * 20, market=market, symbol="SOXX"),
            *history_from_series("vix", [20.0] * 20, market=market, symbol="VIX"),
        )
        raw = snapshot(
            point("index_price", 102.0, AS_OF, market=market, symbol="SPX"),
            point("index_change_pct", 2.0, AS_OF, market=market, symbol="SPX"),
            point("index_price", 98.0, AS_OF, market=market, symbol="HYG"),
            point("index_price", 101.0, AS_OF, market=market, symbol="LQD"),
            point("index_price", 99.0, AS_OF, market=market, symbol="RUT"),
            point("index_price", 103.0, AS_OF, market=market, symbol="NDX"),
            point("index_price", 97.0, AS_OF, market=market, symbol="SOXX"),
            point("vix", 25.0, AS_OF, market=market, symbol="VIX", source="vix_source"),
            point("vix3m", 24.0, AS_OF, market=market, symbol="VIX3M"),
            market=market,
            session_slot="intraday",
        )

        result = USFeatureStrategy().build(raw, history)

        self.assertAlmostEqual(result.features["hyg_lqd_relative_return_5d"], -0.03)
        self.assertAlmostEqual(result.features["vix_vix3m_ratio"], 25 / 24)
        self.assertAlmostEqual(result.features["vix_change_5d"], 0.25)
        self.assertAlmostEqual(result.features["russell_spx_relative_return_5d"], -0.03)
        self.assertAlmostEqual(result.features["nasdaq_spx_relative_return_5d"], 0.01)
        self.assertAlmostEqual(result.features["soxx_spx_relative_return_5d"], -0.05)
        self.assertTrue(result.features["credit_volatility_transition"])
        self.assertEqual(result.reliability_grade, "C")
        vix_evidence = next(item for item in result.evidence if item.evidence_id.split(":")[2] == "vix")
        self.assertEqual(vix_evidence.source, "vix_source")
        self.assertEqual(vix_evidence.as_of_time, AS_OF)

    def test_evidence_ids_and_source_times_are_stable_and_non_empty(self):
        at = AS_OF - timedelta(minutes=1)
        raw = snapshot(
            point("index_price", 100.0, at, market=Market.US, symbol="SPX", source="z"),
            point("index_change_pct", 0.0, at, market=Market.US, symbol="SPX", source="a"),
            market=Market.US,
        )

        first = USFeatureStrategy().build(raw, ())
        second = USFeatureStrategy().build(raw, ())

        self.assertTrue(first.source_times)
        self.assertEqual(first.source_times, {"z:first": at, "z:last": at})
        self.assertEqual(first.evidence_ids, second.evidence_ids)
        self.assertEqual(first.evidence_ids, tuple(sorted(first.evidence_ids)))
        self.assertTrue(all(item.evidence_id.startswith("us:market-warning-v1:") for item in first.evidence))

    def test_source_times_exclude_future_audit_metadata(self):
        raw = RawMarketSnapshot(
            market=Market.US,
            as_of_time=AS_OF,
            session_slot="intraday",
            points=(
                point("index_price", 100.0, AS_OF, market=Market.US, symbol="SPX"),
                point("index_change_pct", 0.0, AS_OF, market=Market.US, symbol="SPX"),
            ),
            source_times={"future_metadata": AS_OF + timedelta(seconds=1)},
        )

        result = USFeatureStrategy().build(raw, ())

        self.assertNotIn("future_metadata", result.source_times)
        self.assertTrue(all(value <= AS_OF for value in result.source_times.values()))

    def test_rolling_and_multi_leg_evidence_lists_all_input_sources(self):
        market = Market.US
        old = AS_OF - timedelta(days=1)
        history = (
            snapshot(point("index_price", 100.0, old, market=market, symbol="SPX", source="spx_history"), market=market, at=old),
            snapshot(point("index_price", 100.0, old, market=market, symbol="HYG", source="hyg_history"), market=market, at=old),
            snapshot(point("index_price", 100.0, old, market=market, symbol="LQD", source="lqd_history"), market=market, at=old),
        )
        raw = snapshot(
            point("index_price", 101.0, AS_OF, market=market, symbol="SPX", source="spx_current"),
            point("index_change_pct", 1.0, AS_OF, market=market, symbol="SPX"),
            point("index_price", 98.0, AS_OF, market=market, symbol="HYG", source="hyg_current"),
            point("index_price", 100.0, AS_OF, market=market, symbol="LQD", source="lqd_current"),
            market=market,
            session_slot="intraday",
        )

        result = USFeatureStrategy().build(raw, history)

        return_evidence = next(item for item in result.evidence if item.evidence_id.split(":")[2] == "return_1d")
        relative_evidence = next(item for item in result.evidence if item.evidence_id.split(":")[2] == "hyg_lqd_relative_return_5d")
        self.assertEqual(return_evidence.source, "spx_current+spx_history")
        self.assertEqual(relative_evidence.source, "hyg_current+hyg_history+lqd_current+lqd_history")
        self.assertEqual(relative_evidence.as_of_time, AS_OF)
        self.assertEqual(result.source_times["spx_history:first"], old)
        self.assertEqual(result.source_times["spx_history:last"], old)

    def test_return_evidence_excludes_remote_history_and_credit_lists_four_legs(self):
        market = Market.US
        history = tuple(
            snapshot(
                point("index_price", 100.0, AS_OF - timedelta(days=days), market=market, symbol="SPX", source=f"remote_{days}"),
                market=market,
                at=AS_OF - timedelta(days=days),
            )
            for days in range(2, 9)
        ) + (
            snapshot(point("index_price", 100.0, AS_OF - timedelta(days=1), market=market, symbol="SPX", source="near"), market=market, at=AS_OF - timedelta(days=1)),
        )
        raw = snapshot(
            point("index_price", 101.0, AS_OF, market=market, symbol="SPX", source="now"),
            point("index_change_pct", 1.0, AS_OF, market=market, symbol="SPX"),
            point("index_price", 98.0, AS_OF, market=market, symbol="HYG", source="hyg"),
            point("index_price", 100.0, AS_OF, market=market, symbol="LQD", source="lqd"),
            point("vix", 25.0, AS_OF, market=market, symbol="VIX", source="vix"),
            point("vix3m", 24.0, AS_OF, market=market, symbol="VIX3M", source="vix3m"),
            market=market,
            session_slot="intraday",
        )

        result = USFeatureStrategy().build(raw, history)

        return_evidence = next(item for item in result.evidence if item.evidence_id.split(":")[2] == "return_1d")
        credit_evidence = next(item for item in result.evidence if item.evidence_id.split(":")[2] == "credit_volatility_transition")
        self.assertEqual(return_evidence.source, "near+now")
        self.assertEqual(credit_evidence.source, "hyg+lqd+vix+vix3m")

    def test_intraday_alignment_accepts_120_seconds_and_rejects_121(self):
        market = Market.US
        def build(skew: int):
            return snapshot(
                point("index_price", 100.0, AS_OF, market=market, symbol="SPX"),
                point("index_change_pct", 0.0, AS_OF, market=market, symbol="SPX"),
                point("vix", 24.0, AS_OF, market=market, symbol="VIX"),
                point("vix3m", 23.0, AS_OF - timedelta(seconds=skew), market=market, symbol="VIX3M"),
                market=market,
                session_slot="intraday",
            )

        self.assertAlmostEqual(USFeatureStrategy().build(build(120), ()).features["vix_vix3m_ratio"], 24 / 23)
        self.assertIsNone(USFeatureStrategy().build(build(121), ()).features["vix_vix3m_ratio"])

    def test_source_times_excludes_visible_but_unused_raw_metadata(self):
        raw = RawMarketSnapshot(
            market=Market.US,
            as_of_time=AS_OF,
            session_slot="intraday",
            points=(
                point("index_price", 100.0, AS_OF, market=Market.US, symbol="SPX", source="used"),
                point("index_change_pct", 0.0, AS_OF, market=Market.US, symbol="SPX", source="quality_only"),
            ),
            source_times={"unused_adapter": AS_OF},
        )

        result = USFeatureStrategy().build(raw, ())

        self.assertNotIn("unused_adapter:first", result.source_times)
        self.assertNotIn("quality_only:first", result.source_times)
        self.assertEqual(result.source_times["used:first"], AS_OF)


if __name__ == "__main__":
    main()
