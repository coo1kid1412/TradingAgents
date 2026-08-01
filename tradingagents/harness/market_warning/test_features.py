"""Hand-calculated contract tests for deterministic market-warning features."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
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
        self.assertEqual(derive_market_phase(None), MarketPhase.FIRST_SHOCK)
        self.assertEqual(FEATURE_VERSION, "market-warning-v1")
        for name, metadata in FEATURE_METADATA.items():
            with self.subTest(name=name):
                self.assertEqual(set(metadata), {"source", "availability", "missing", "direction", "unit", "version"})
                self.assertEqual(metadata["version"], FEATURE_VERSION)


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
            point("index_price", 100.0, AS_OF, market=market, symbol="SPX"),
            point("index_change_pct", -1.0, AS_OF, market=market, symbol="SPX"),
            point("index_price", 98.0, AS_OF, market=market, symbol="HYG"),
            point("index_price", 100.0, AS_OF, market=market, symbol="LQD"),
            point("index_price", 95.0, AS_OF, market=market, symbol="RUT"),
            point("index_price", 97.0, AS_OF, market=market, symbol="NDX"),
            point("index_price", 96.0, AS_OF, market=market, symbol="SOXX"),
            point("vix", 25.0, AS_OF, market=market, symbol="VIX", source="vix_source"),
            point("vix3m", 24.0, AS_OF, market=market, symbol="VIX3M"),
            market=market,
            session_slot="intraday",
        )

        result = USFeatureStrategy().build(raw, history)

        self.assertAlmostEqual(result.features["hyg_lqd_relative_return_5d"], -0.02)
        self.assertAlmostEqual(result.features["vix_vix3m_ratio"], 25 / 24)
        self.assertAlmostEqual(result.features["vix_change_5d"], 0.25)
        self.assertAlmostEqual(result.features["russell_spx_relative_return_5d"], -0.05)
        self.assertAlmostEqual(result.features["nasdaq_spx_relative_return_5d"], -0.03)
        self.assertAlmostEqual(result.features["soxx_spx_relative_return_5d"], -0.04)
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
        self.assertEqual(first.source_times, {"a": at, "z": at})
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


if __name__ == "__main__":
    main()
