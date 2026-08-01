"""Hand-calculated contract tests for deterministic market-warning features."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import pstdev
import sys
from unittest import TestCase, main

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from tradingagents.harness.market_warning.domain import (
    DataStatus,
    Market,
    MarketDataPoint,
    MarketPhase,
    QuantRiskAssessment,
    RawMarketSnapshot,
    RiskLevel,
)
from tradingagents.harness.market_warning.features import (
    AShareFeatureStrategy,
    FEATURE_METADATA,
    FEATURE_VERSION,
    USFeatureStrategy,
    derive_market_phase,
)
from tradingagents.harness.market_warning.policy import baseline_level


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
    quality_status: DataStatus = DataStatus.FRESH,
) -> MarketDataPoint:
    return MarketDataPoint(
        market=market,
        symbol=symbol,
        field=field,
        value=value,
        data_time=at,
        fetched_at=at,
        source=source,
        quality_status=quality_status,
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
    def test_range_zscore_and_transition_metadata_have_auditable_missing_contracts(self):
        names = (
            "audited_ohlc_return_1d",
            "range_zscore_20d",
            "abnormal_range_weak_close_transition",
            "breadth_deterioration_transition",
            "equity_dispersion_transition",
        )
        for name in names:
            with self.subTest(name=name):
                self.assertIn(name, FEATURE_METADATA)
                self.assertEqual(
                    FEATURE_METADATA[name]["missing"],
                    "preserve as None; emit unavailable evidence",
                )

    def test_missing_inputs_preserve_new_transition_features_as_none(self):
        a_result = AShareFeatureStrategy().build(
            snapshot(point("index_price", 100.0, AS_OF), point("index_change_pct", 0.0, AS_OF)),
            (),
        )
        us_result = USFeatureStrategy().build(
            snapshot(
                point("index_price", 100.0, AS_OF, market=Market.US, symbol="SPX"),
                point("index_change_pct", 0.0, AS_OF, market=Market.US, symbol="SPX"),
                market=Market.US,
            ),
            (),
        )

        self.assertIsNone(a_result.features["range_zscore_20d"])
        self.assertIsNone(a_result.features["abnormal_range_weak_close_transition"])
        self.assertIsNone(a_result.features["breadth_deterioration_transition"])
        self.assertIsNone(us_result.features["equity_dispersion_transition"])

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
        self.assertEqual(FEATURE_VERSION, "market-warning-v2")
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
    def _credit_history(self, *, include_vix: bool = True):
        market = Market.US
        points = []
        for days in range(1, 6):
            at = AS_OF - timedelta(days=days)
            row = [
                point("index_price", 100.0, at, market=market, symbol="SPX", source="spx"),
                point("index_price", 100.0, at, market=market, symbol="HYG", source="hyg"),
                point("index_price", 100.0, at, market=market, symbol="LQD", source="lqd"),
            ]
            if include_vix:
                row.append(point("vix", 20.0, at, market=market, symbol="VIX", source="vix"))
            points.append(snapshot(*row, market=market, at=at, session_slot="intraday"))
        return tuple(points)

    def _credit_raw(self, *, vix_time: datetime | None = AS_OF, include_vix3m: bool = True):
        market = Market.US
        points = [
            point("index_price", 100.0, AS_OF, market=market, symbol="SPX", source="spx"),
            point("index_change_pct", 0.0, AS_OF, market=market, symbol="SPX"),
            point("index_price", 98.0, AS_OF, market=market, symbol="HYG", source="hyg"),
            point("index_price", 100.0, AS_OF, market=market, symbol="LQD", source="lqd"),
        ]
        if vix_time is not None:
            points.append(point("vix", 25.0, vix_time, market=market, symbol="VIX", source="vix"))
        if include_vix3m:
            points.append(point("vix3m", 24.0, AS_OF, market=market, symbol="VIX3M", source="vix3m"))
        return snapshot(*points, market=market, session_slot="intraday")

    def test_stale_vix_observation_cannot_confirm_valid_credit_weakness(self):
        result = USFeatureStrategy().build(
            self._credit_raw(vix_time=AS_OF - timedelta(days=1), include_vix3m=False),
            self._credit_history(),
        )

        self.assertIsNone(result.features["vix_change_5d"])
        self.assertIsNone(result.features["credit_volatility_transition"])

    def test_credit_transition_uses_vix_change_when_vix3m_is_missing(self):
        result = USFeatureStrategy().build(self._credit_raw(include_vix3m=False), self._credit_history())

        self.assertAlmostEqual(result.features["vix_change_5d"], 0.25)
        self.assertTrue(result.features["credit_volatility_transition"])
        evidence = next(item for item in result.evidence if item.evidence_id.split(":")[2] == "credit_volatility_transition")
        self.assertEqual(evidence.source, "hyg+lqd+vix")

    def test_credit_transition_uses_vix_ratio_when_vix_history_is_missing(self):
        result = USFeatureStrategy().build(self._credit_raw(), self._credit_history(include_vix=False))

        self.assertIsNone(result.features["vix_change_5d"])
        self.assertTrue(result.features["credit_volatility_transition"])
        evidence = next(item for item in result.evidence if item.evidence_id.split(":")[2] == "credit_volatility_transition")
        self.assertEqual(evidence.source, "hyg+lqd+vix+vix3m")

    def test_credit_transition_is_unavailable_without_either_volatility_confirmation(self):
        result = USFeatureStrategy().build(
            self._credit_raw(vix_time=None, include_vix3m=False), self._credit_history(include_vix=False)
        )

        self.assertIsNone(result.features["credit_volatility_transition"])

    def test_vix_metadata_matches_endpoint_and_or_path_contract(self):
        self.assertEqual(
            FEATURE_METADATA["vix_change_5d"]["availability"],
            "5-market-day span with two aligned endpoints visible by as_of",
        )
        self.assertEqual(
            FEATURE_METADATA["credit_volatility_transition"]["availability"],
            "aligned HYG/LQD credit weakness AND (aligned 5-day VIX change OR current aligned VIX/VIX3M ratio)",
        )

    def test_vix_change_evidence_uses_only_five_day_endpoints(self):
        market = Market.US
        history = tuple(
            snapshot(
                point("index_price", 100.0, AS_OF - timedelta(days=days), market=market, symbol="SPX"),
                point("vix", 20.0 + days, AS_OF - timedelta(days=days), market=market, symbol="VIX", source=f"vix_{days}"),
                market=market,
                at=AS_OF - timedelta(days=days),
                session_slot="intraday",
            )
            for days in range(1, 6)
        )
        raw = snapshot(
            point("index_price", 100.0, AS_OF, market=market, symbol="SPX"),
            point("index_change_pct", 0.0, AS_OF, market=market, symbol="SPX"),
            point("vix", 30.0, AS_OF, market=market, symbol="VIX", source="vix_now"),
            market=market,
            session_slot="intraday",
        )

        result = USFeatureStrategy().build(raw, history)

        evidence = next(item for item in result.evidence if item.evidence_id.split(":")[2] == "vix_change_5d")
        self.assertEqual(evidence.source, "vix_5+vix_now")

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
        self.assertTrue(all(item.evidence_id.startswith("us:market-warning-v2:") for item in first.evidence))

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


class PolicyReachabilityIntegrationTests(TestCase):
    @staticmethod
    def quant(probability: float) -> QuantRiskAssessment:
        return QuantRiskAssessment(
            crash_1d_probability=probability,
            crash_3d_probability=0.01,
            market_phase=MarketPhase.FIRST_SHOCK,
            base_rate_1d=0.01,
            base_rate_3d=0.01,
            reliability_grade="A",
            model_version="integration-model",
            calibration_version="integration-calibration",
            top_contributors=(),
        )

    def test_a_share_strategy_emits_reachable_orange_pressure_transition(self):
        old = AS_OF - timedelta(days=1)
        history = (
            snapshot(
                point("index_price", 100.0, old, source="index_history"),
                market=Market.A_SHARE,
                at=old,
            ),
        )
        raw = snapshot(
            point("index_price", 99.0, AS_OF, source="index_now"),
            point("index_change_pct", -1.0, AS_OF, source="index_now"),
            point("breadth_up_pct", 25.0, AS_OF, source="breadth_now"),
            point("industry_decline_pct", 80.0, AS_OF, source="industry_now"),
            point("limit_down_pct", 0.5, AS_OF, source="limit_now"),
        )

        result = AShareFeatureStrategy().build(raw, history)

        self.assertTrue(result.features["breadth_deterioration_transition"])
        self.assertEqual(baseline_level(self.quant(0.04), result), RiskLevel.ORANGE)
        evidence = next(
            item for item in result.evidence
            if item.evidence_id.split(":")[2] == "breadth_deterioration_transition"
        )
        self.assertEqual(evidence.source, "breadth_now+index_history+index_now+industry_now")
        self.assertEqual(evidence.as_of_time, AS_OF)
        self.assertIn("breadth_now:last", result.source_times)
        self.assertIn("index_history:first", result.source_times)

    def test_us_strategy_emits_non_credit_orange_transition(self):
        market = Market.US
        history = tuple(
            snapshot(
                *(
                    point("index_price", 100.0, AS_OF - timedelta(days=days), market=market, symbol=symbol, source=f"{symbol}_history")
                    for symbol in ("SPX", "RUT", "NDX", "SOXX")
                ),
                market=market,
                at=AS_OF - timedelta(days=days),
            )
            for days in range(5, 0, -1)
        )
        raw = snapshot(
            point("index_price", 99.0, AS_OF, market=market, symbol="SPX", source="SPX_now"),
            point("index_change_pct", -1.0, AS_OF, market=market, symbol="SPX", source="SPX_now"),
            point("index_price", 97.0, AS_OF, market=market, symbol="RUT", source="RUT_now"),
            point("index_price", 97.5, AS_OF, market=market, symbol="NDX", source="NDX_now"),
            point("index_price", 98.5, AS_OF, market=market, symbol="SOXX", source="SOXX_now"),
            market=market,
            session_slot="intraday",
        )

        result = USFeatureStrategy().build(raw, history)

        self.assertTrue(result.features["equity_dispersion_transition"])
        self.assertIsNone(result.features["credit_volatility_transition"])
        self.assertEqual(baseline_level(self.quant(0.04), result), RiskLevel.ORANGE)
        evidence = next(
            item for item in result.evidence
            if item.evidence_id.split(":")[2] == "equity_dispersion_transition"
        )
        self.assertNotEqual(evidence.source, "unavailable")
        self.assertEqual(evidence.as_of_time, AS_OF)

    def test_strategy_emitted_abnormal_range_reaches_hard_red(self):
        ranges = [0.02] * 19 + [0.20]
        history = tuple(
            snapshot(
                point("index_price", 100.0, AS_OF - timedelta(days=days), source="candle_history"),
                point("open", 100.0, AS_OF - timedelta(days=days), source="candle_history"),
                point("high", 101.0, AS_OF - timedelta(days=days), source="candle_history"),
                point("low", 99.0, AS_OF - timedelta(days=days), source="candle_history"),
                at=AS_OF - timedelta(days=days),
            )
            for days in range(19, 0, -1)
        )
        raw = snapshot(
            point("index_price", 92.0, AS_OF, source="candle_now"),
            point("index_change_pct", -8.0, AS_OF, source="candle_now"),
            point("open", 100.0, AS_OF, source="candle_now"),
            point("high", 110.0, AS_OF, source="candle_now"),
            point("low", 90.0, AS_OF, source="candle_now"),
        )

        result = AShareFeatureStrategy().build(raw, history)

        expected_zscore = (ranges[-1] - sum(ranges) / len(ranges)) / pstdev(ranges)
        self.assertAlmostEqual(result.features["range_zscore_20d"], expected_zscore)
        self.assertAlmostEqual(result.features["audited_ohlc_return_1d"], -0.08)
        self.assertTrue(result.features["abnormal_range_weak_close_transition"])
        self.assertEqual(baseline_level(self.quant(0.01), result), RiskLevel.RED)
        evidence = next(
            item for item in result.evidence if item.evidence_id.split(":")[2] == "range_zscore_20d"
        )
        self.assertEqual(evidence.source, "candle_history+candle_now")
        self.assertEqual(evidence.as_of_time, AS_OF)
        self.assertIn("candle_history:first", result.source_times)
        self.assertIn("candle_now:last", result.source_times)

    def test_unusable_newer_historical_close_cannot_pollute_returns_or_range_trigger(self):
        for status in (DataStatus.STALE, DataStatus.CONFLICTED, DataStatus.INSUFFICIENT):
            with self.subTest(status=status):
                history = []
                for days in range(19, 0, -1):
                    at = AS_OF - timedelta(days=days)
                    candle_at = at - timedelta(minutes=1) if days == 1 else at
                    points = [
                        point("index_price", 92.0, candle_at, source="coherent_history"),
                        point("open", 92.0, candle_at, source="coherent_history"),
                        point("high", 92.92, candle_at, source="coherent_history"),
                        point("low", 91.08, candle_at, source="coherent_history"),
                    ]
                    if days == 1:
                        points.append(
                            point(
                                "index_price",
                                120.0,
                                at,
                                source="bad_incomplete_close",
                                quality_status=status,
                            )
                        )
                    history.append(snapshot(*points, at=at))
                raw = snapshot(
                    point("index_price", 92.0, AS_OF, source="coherent_current"),
                    point("index_change_pct", 0.0, AS_OF, source="coherent_current"),
                    point("open", 100.0, AS_OF, source="coherent_current"),
                    point("high", 110.0, AS_OF, source="coherent_current"),
                    point("low", 90.0, AS_OF, source="coherent_current"),
                )

                result = AShareFeatureStrategy().build(raw, tuple(history))

                self.assertEqual(result.features["return_1d"], 0.0)
                self.assertEqual(result.features["audited_ohlc_return_1d"], 0.0)
                self.assertFalse(result.features["abnormal_range_weak_close_transition"])
                self.assertEqual(baseline_level(self.quant(0.01), result), RiskLevel.GREEN)
                audited = next(
                    item
                    for item in result.evidence
                    if item.evidence_id.split(":")[2] == "audited_ohlc_return_1d"
                )
                self.assertNotIn("bad_incomplete_close", audited.source)
                self.assertNotIn("bad_incomplete_close", audited.summary)

    def test_fresh_incomplete_close_cannot_hijack_audited_ohlc_return(self):
        history = []
        for days in range(19, 0, -1):
            at = AS_OF - timedelta(days=days)
            candle_at = at - timedelta(minutes=1) if days == 1 else at
            points = [
                point("index_price", 92.0, candle_at, source="coherent_history"),
                point("open", 92.0, candle_at, source="coherent_history"),
                point("high", 92.92, candle_at, source="coherent_history"),
                point("low", 91.08, candle_at, source="coherent_history"),
            ]
            if days == 1:
                points.append(point("index_price", 120.0, at, source="fresh_incomplete_close"))
            history.append(snapshot(*points, at=at))
        raw = snapshot(
            point("index_price", 92.0, AS_OF, source="coherent_current"),
            point("index_change_pct", 0.0, AS_OF, source="coherent_current"),
            point("open", 100.0, AS_OF, source="coherent_current"),
            point("high", 110.0, AS_OF, source="coherent_current"),
            point("low", 90.0, AS_OF, source="coherent_current"),
        )

        result = AShareFeatureStrategy().build(raw, tuple(history))

        self.assertAlmostEqual(result.features["return_1d"], 92.0 / 120.0 - 1.0)
        self.assertEqual(result.features["audited_ohlc_return_1d"], 0.0)
        self.assertFalse(result.features["abnormal_range_weak_close_transition"])
        self.assertEqual(baseline_level(self.quant(0.01), result), RiskLevel.GREEN)

    def test_missing_prior_coherent_ohlc_bar_disables_audited_return_and_hard_trigger(self):
        history = tuple(
            snapshot(
                point("index_price", 92.0, AS_OF - timedelta(days=days), source="coherent_history"),
                point("open", 92.0, AS_OF - timedelta(days=days), source="coherent_history"),
                point("high", 92.92, AS_OF - timedelta(days=days), source="coherent_history"),
                point("low", 91.08, AS_OF - timedelta(days=days), source="coherent_history"),
                at=AS_OF - timedelta(days=days),
            )
            for days in range(20, 1, -1)
        ) + (
            snapshot(
                point("index_price", 120.0, AS_OF - timedelta(days=1), source="incomplete"),
                at=AS_OF - timedelta(days=1),
            ),
        )
        raw = snapshot(
            point("index_price", 92.0, AS_OF, source="coherent_current"),
            point("index_change_pct", 0.0, AS_OF, source="coherent_current"),
            point("open", 100.0, AS_OF, source="coherent_current"),
            point("high", 110.0, AS_OF, source="coherent_current"),
            point("low", 90.0, AS_OF, source="coherent_current"),
        )

        result = AShareFeatureStrategy().build(raw, history)

        self.assertIsNone(result.features["audited_ohlc_return_1d"])
        self.assertIsNone(result.features["abnormal_range_weak_close_transition"])
        self.assertEqual(baseline_level(self.quant(0.01), result), RiskLevel.GREEN)

    def test_audited_return_evidence_names_exact_endpoint_and_current_ohlc_inputs(self):
        prior = AS_OF - timedelta(days=1)
        result = AShareFeatureStrategy().build(
            snapshot(
                point("index_price", 92.0, AS_OF, source="current_candle"),
                point("index_change_pct", -8.0, AS_OF, source="current_candle"),
                point("open", 100.0, AS_OF, source="current_candle"),
                point("high", 110.0, AS_OF, source="current_candle"),
                point("low", 90.0, AS_OF, source="current_candle"),
            ),
            (
                snapshot(
                    point("index_price", 100.0, prior, source="prior_candle"),
                    point("open", 100.0, prior, source="prior_candle"),
                    point("high", 101.0, prior, source="prior_candle"),
                    point("low", 99.0, prior, source="prior_candle"),
                    at=prior,
                ),
            ),
        )

        audited = next(
            item
            for item in result.evidence
            if item.evidence_id.split(":")[2] == "audited_ohlc_return_1d"
        )
        self.assertEqual(audited.source, "current_candle+prior_candle")
        for expected in (
            "prior_candle/INDEX/index_price@",
            "current_candle/INDEX/open@",
            "current_candle/INDEX/high@",
            "current_candle/INDEX/low@",
            "current_candle/INDEX/index_price@",
        ):
            self.assertIn(expected, audited.summary)
        self.assertIn("prior_candle:first", result.source_times)
        self.assertIn("current_candle:last", result.source_times)

    def test_reviewer_invalid_close_regression_is_unknown_not_red(self):
        history = tuple(
            snapshot(
                point("index_price", 100.0, AS_OF - timedelta(days=days), source="candle_history"),
                point("open", 100.0, AS_OF - timedelta(days=days), source="candle_history"),
                point("high", 101.0, AS_OF - timedelta(days=days), source="candle_history"),
                point("low", 99.0, AS_OF - timedelta(days=days), source="candle_history"),
                at=AS_OF - timedelta(days=days),
            )
            for days in range(19, 0, -1)
        )
        raw = snapshot(
            point("index_price", 80.0, AS_OF, source="candle_now"),
            point("index_change_pct", -20.0, AS_OF, source="candle_now"),
            point("open", 100.0, AS_OF, source="candle_now"),
            point("high", 110.0, AS_OF, source="candle_now"),
            point("low", 90.0, AS_OF, source="candle_now"),
        )

        result = AShareFeatureStrategy().build(raw, history)

        self.assertEqual(result.data_quality, DataStatus.CONFLICTED)
        self.assertEqual(result.reliability_grade, "UNAVAILABLE")
        for name in (
            "range_pct",
            "range_zscore_20d",
            "close_location",
            "abnormal_range_weak_close_transition",
        ):
            self.assertIsNone(result.features[name], name)
        self.assertEqual(baseline_level(self.quant(0.01), result), RiskLevel.UNKNOWN)

    def test_historical_invalid_ohlc_is_nan_and_cannot_create_range_zscore(self):
        history = tuple(
            snapshot(
                point("index_price", 100.0, AS_OF - timedelta(days=days)),
                point("open", 100.0, AS_OF - timedelta(days=days)),
                point("high", 99.0 if days == 10 else 101.0, AS_OF - timedelta(days=days)),
                point("low", 101.0 if days == 10 else 99.0, AS_OF - timedelta(days=days)),
                at=AS_OF - timedelta(days=days),
            )
            for days in range(19, 0, -1)
        )
        raw = snapshot(
            point("index_price", 92.0, AS_OF),
            point("index_change_pct", -8.0, AS_OF),
            point("open", 100.0, AS_OF),
            point("high", 110.0, AS_OF),
            point("low", 90.0, AS_OF),
        )

        result = AShareFeatureStrategy().build(raw, history)

        self.assertIsNone(result.features["range_zscore_20d"])
        self.assertIsNone(result.features["abnormal_range_weak_close_transition"])
        self.assertNotEqual(result.data_quality, DataStatus.CONFLICTED)
        self.assertEqual(baseline_level(self.quant(0.01), result), RiskLevel.GREEN)

    def test_stale_current_ohlc_is_unknown_and_cannot_trigger_red(self):
        history = tuple(
            snapshot(
                point("index_price", 100.0, AS_OF - timedelta(days=days), source="candle_history"),
                point("open", 100.0, AS_OF - timedelta(days=days), source="candle_history"),
                point("high", 101.0, AS_OF - timedelta(days=days), source="candle_history"),
                point("low", 99.0, AS_OF - timedelta(days=days), source="candle_history"),
                at=AS_OF - timedelta(days=days),
            )
            for days in range(19, 0, -1)
        )
        raw = snapshot(
            point("index_price", 92.0, AS_OF, source="candle_now"),
            point("index_change_pct", -8.0, AS_OF, source="candle_now"),
            point("open", 100.0, AS_OF, source="candle_now", quality_status=DataStatus.STALE),
            point("high", 110.0, AS_OF, source="candle_now", quality_status=DataStatus.STALE),
            point("low", 90.0, AS_OF, source="candle_now", quality_status=DataStatus.STALE),
        )

        result = AShareFeatureStrategy().build(raw, history)

        self.assertEqual(result.data_quality, DataStatus.STALE)
        self.assertEqual(result.reliability_grade, "UNAVAILABLE")
        for name in (
            "range_pct",
            "range_zscore_20d",
            "close_location",
            "abnormal_range_weak_close_transition",
        ):
            self.assertIsNone(result.features[name], name)
        self.assertEqual(baseline_level(self.quant(0.01), result), RiskLevel.UNKNOWN)

    def test_insufficient_current_ohlc_is_unknown_and_has_no_range_features(self):
        raw = snapshot(
            point("index_price", 100.0, AS_OF, source="candle"),
            point("index_change_pct", 0.0, AS_OF, source="candle"),
            point(
                "open",
                100.0,
                AS_OF,
                source="candle",
                quality_status=DataStatus.INSUFFICIENT,
            ),
            point("high", 101.0, AS_OF, source="candle"),
            point("low", 99.0, AS_OF, source="candle"),
        )

        result = AShareFeatureStrategy().build(raw, ())

        self.assertEqual(result.data_quality, DataStatus.INSUFFICIENT)
        self.assertEqual(result.reliability_grade, "UNAVAILABLE")
        for name in (
            "range_pct",
            "range_zscore_20d",
            "close_location",
            "abnormal_range_weak_close_transition",
        ):
            self.assertIsNone(result.features[name], name)
        self.assertEqual(baseline_level(self.quant(0.01), result), RiskLevel.UNKNOWN)

    def test_historical_stale_or_conflicted_ohlc_is_nan_without_fill(self):
        for status in (DataStatus.STALE, DataStatus.CONFLICTED):
            with self.subTest(status=status):
                history = tuple(
                    snapshot(
                        point("index_price", 100.0, AS_OF - timedelta(days=days), source="history"),
                        point(
                            "open",
                            100.0,
                            AS_OF - timedelta(days=days),
                            source="history",
                            quality_status=status if days == 10 else DataStatus.FRESH,
                        ),
                        point("high", 101.0, AS_OF - timedelta(days=days), source="history"),
                        point("low", 99.0, AS_OF - timedelta(days=days), source="history"),
                        at=AS_OF - timedelta(days=days),
                    )
                    for days in range(19, 0, -1)
                )
                raw = snapshot(
                    point("index_price", 92.0, AS_OF, source="current"),
                    point("index_change_pct", -8.0, AS_OF, source="current"),
                    point("open", 100.0, AS_OF, source="current"),
                    point("high", 110.0, AS_OF, source="current"),
                    point("low", 90.0, AS_OF, source="current"),
                )

                result = AShareFeatureStrategy().build(raw, history)

                self.assertIsNone(result.features["range_zscore_20d"])
                self.assertIsNone(result.features["abnormal_range_weak_close_transition"])
                self.assertEqual(baseline_level(self.quant(0.01), result), RiskLevel.GREEN)

    def test_two_valid_sources_do_not_mix_and_remain_usable(self):
        observation_time = AS_OF + timedelta(hours=1)
        points = []
        source_values = {
            "alpha": {"open": 100.0, "high": 110.0, "low": 90.0, "index_price": 100.0},
            "beta": {"open": 100.4, "high": 110.4, "low": 90.4, "index_price": 100.4},
        }
        skews = {
            "alpha": {"open": 20, "high": 0, "low": 20, "index_price": 0},
            "beta": {"open": 0, "high": 20, "low": 0, "index_price": 20},
        }
        for source, values in source_values.items():
            for field, value in values.items():
                points.append(
                    point(
                        field,
                        value,
                        observation_time - timedelta(seconds=skews[source][field]),
                        source=source,
                    )
                )
        points.append(point("index_change_pct", 0.0, observation_time, source="alpha"))

        history = (
            snapshot(
                point("index_price", 100.0, observation_time - timedelta(days=1), source="history"),
                at=observation_time - timedelta(days=1),
            ),
        )
        result = AShareFeatureStrategy().build(
            snapshot(*points, at=observation_time, session_slot="intraday"),
            history,
        )

        self.assertEqual(result.data_quality, DataStatus.FRESH)
        expected_ranges = {(110.0 - 90.0) / 100.0, (110.4 - 90.4) / 100.4}
        self.assertIn(result.features["range_pct"], expected_ranges)
        self.assertAlmostEqual(result.features["return_1d"], 100.4 / 100.0 - 1.0)
        evidence = next(item for item in result.evidence if item.evidence_id.split(":")[2] == "range_pct")
        self.assertEqual(len(evidence.source.split("+")), 1)

    def test_mixed_incomplete_sources_cannot_create_range_features(self):
        raw = snapshot(
            point("index_price", 100.0, AS_OF, source="close_source"),
            point("index_change_pct", 0.0, AS_OF, source="close_source"),
            point("open", 100.0, AS_OF, source="close_source"),
            point("high", 110.0, AS_OF, source="range_source"),
            point("low", 90.0, AS_OF, source="range_source"),
        )

        result = AShareFeatureStrategy().build(raw, ())

        self.assertNotEqual(result.data_quality, DataStatus.CONFLICTED)
        for name in (
            "range_pct",
            "range_zscore_20d",
            "close_location",
            "abnormal_range_weak_close_transition",
        ):
            self.assertIsNone(result.features[name], name)

    def test_materially_divergent_complete_sources_fail_closed(self):
        raw = snapshot(
            *(
                point(field, value, AS_OF, source="alpha")
                for field, value in (
                    ("index_price", 100.0), ("open", 100.0), ("high", 101.0), ("low", 99.0)
                )
            ),
            *(
                point(field, value, AS_OF, source="beta")
                for field, value in (
                    ("index_price", 110.0), ("open", 110.0), ("high", 111.0), ("low", 109.0)
                )
            ),
            point("index_change_pct", 0.0, AS_OF, source="alpha"),
        )

        result = AShareFeatureStrategy().build(raw, ())

        self.assertEqual(result.data_quality, DataStatus.CONFLICTED)
        self.assertEqual(result.reliability_grade, "UNAVAILABLE")
        self.assertIsNone(result.features["range_pct"])
        self.assertEqual(baseline_level(self.quant(0.01), result), RiskLevel.UNKNOWN)

    def test_valid_close_at_low_and_incomplete_ohlc_keep_exact_missing_semantics(self):
        old = AS_OF - timedelta(days=1)
        history = (snapshot(point("index_price", 100.0, old), at=old),)
        valid = AShareFeatureStrategy().build(
            snapshot(
                point("index_price", 90.0, AS_OF),
                point("index_change_pct", -10.0, AS_OF),
                point("open", 100.0, AS_OF),
                point("high", 110.0, AS_OF),
                point("low", 90.0, AS_OF),
            ),
            history,
        )
        incomplete = AShareFeatureStrategy().build(
            snapshot(
                point("index_price", 99.0, AS_OF),
                point("index_change_pct", -1.0, AS_OF),
                point("open", 100.0, AS_OF),
                point("high", 101.0, AS_OF),
            ),
            history,
        )

        self.assertAlmostEqual(valid.features["range_pct"], 0.20)
        self.assertEqual(valid.features["close_location"], 0.0)
        self.assertNotEqual(valid.data_quality, DataStatus.CONFLICTED)
        for name in (
            "range_pct",
            "range_zscore_20d",
            "close_location",
            "abnormal_range_weak_close_transition",
        ):
            self.assertIsNone(incomplete.features[name], name)
        self.assertNotEqual(incomplete.data_quality, DataStatus.CONFLICTED)


if __name__ == "__main__":
    main()
