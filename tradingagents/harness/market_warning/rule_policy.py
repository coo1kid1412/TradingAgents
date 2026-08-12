"""Versioned, deterministic A-share market-warning rules."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from math import isfinite
from numbers import Real
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from .domain import (
    DataStatus,
    FeatureSnapshot,
    Market,
    MarketPhase,
    RiskLevel,
    RuleLayer,
    RuleRiskAssessment,
    TriggeredRule,
)


_UNUSABLE_DATA = frozenset(
    {DataStatus.CONFLICTED, DataStatus.STALE, DataStatus.INSUFFICIENT}
)


@dataclass(frozen=True)
class RuleManifest:
    engine_version: str
    manifest_sha256: str
    market: Market
    score_cap: float
    layer_caps: Mapping[str, float]
    thresholds: Mapping[str, float]
    required_core_features: tuple[str, ...]
    optional_groups: Mapping[str, tuple[str, ...]]
    display_only_features: tuple[str, ...]
    units: Mapping[str, str]


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, Real):
        return None
    numeric = float(value)
    return numeric if isfinite(numeric) else None


def manifest_sha256(path: Path | str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_rule_manifest(path: Path | str) -> RuleManifest:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    market = Market(payload["market"])
    if market != Market.A_SHARE:
        raise ValueError("rule V1 manifest must target A shares")
    thresholds = {name: float(value) for name, value in payload["thresholds"].items()}
    layer_caps = {name: float(value) for name, value in payload["layer_caps"].items()}
    expected_layers = {item.value for item in RuleLayer}
    if set(layer_caps) != expected_layers:
        raise ValueError("layer_caps must define every rule layer")
    return RuleManifest(
        engine_version=str(payload["engine_version"]),
        manifest_sha256=manifest_sha256(path),
        market=market,
        score_cap=float(payload["score_cap"]),
        layer_caps=MappingProxyType(layer_caps),
        thresholds=MappingProxyType(thresholds),
        required_core_features=tuple(payload["required_core_features"]),
        optional_groups=MappingProxyType(
            {name: tuple(fields) for name, fields in payload["optional_groups"].items()}
        ),
        display_only_features=tuple(payload["display_only_features"]),
        units=MappingProxyType(dict(payload["units"])),
    )


def _phase(snapshot: FeatureSnapshot) -> MarketPhase:
    value = snapshot.features.get("market_phase")
    try:
        return MarketPhase(value)
    except (TypeError, ValueError):
        drawdown = _number(snapshot.features.get("drawdown_20d"))
        return (
            MarketPhase.FIRST_SHOCK
            if drawdown is not None and drawdown > -0.05
            else MarketPhase.CONTINUATION
        )


def _missing_optional_groups(
    snapshot: FeatureSnapshot, manifest: RuleManifest
) -> tuple[str, ...]:
    return tuple(
        name
        for name, fields in manifest.optional_groups.items()
        if any(snapshot.features.get(field) is None for field in fields)
    )


def _intraday_cross_section_valid(
    snapshot: FeatureSnapshot, manifest: RuleManifest
) -> bool:
    if "intraday" not in snapshot.session_slot.lower():
        return True
    coverage = _number(snapshot.features.get("realtime_breadth_coverage_pct"))
    staleness = _number(snapshot.features.get("realtime_breadth_staleness_minutes"))
    return (
        coverage is not None
        and staleness is not None
        and coverage >= manifest.thresholds["realtime_breadth_coverage_pct"]
        and staleness <= manifest.thresholds["realtime_breadth_staleness_minutes"]
    )


def _reliability(
    snapshot: FeatureSnapshot,
    missing_groups: tuple[str, ...],
    cross_section_valid: bool,
) -> str:
    if not cross_section_valid or "breadth" in missing_groups:
        return "C"
    if missing_groups and snapshot.reliability_grade == "A":
        return "B"
    return snapshot.reliability_grade


def _evidence_ids(snapshot: FeatureSnapshot, feature_names: tuple[str, ...]) -> tuple[str, ...]:
    ids = tuple(
        evidence.evidence_id
        for evidence in snapshot.evidence
        if any(name in evidence.evidence_id for name in feature_names)
    )
    if not ids:
        raise ValueError(f"triggered rule has no evidence for {feature_names!r}")
    return tuple(dict.fromkeys(ids))


def _unknown_assessment(
    snapshot: FeatureSnapshot,
    manifest: RuleManifest,
    missing_groups: tuple[str, ...],
) -> RuleRiskAssessment:
    return RuleRiskAssessment(
        market=snapshot.market,
        as_of_time=snapshot.as_of_time,
        engine_version=manifest.engine_version,
        manifest_sha256=manifest.manifest_sha256,
        risk_level=RiskLevel.UNKNOWN,
        risk_score=0.0,
        market_phase=_phase(snapshot),
        triggered_rules=(),
        missing_optional_groups=missing_groups,
        reliability_grade="UNAVAILABLE",
        evaluation_latency_ms=0.0,
    )


def evaluate_a_share_rules(
    snapshot: FeatureSnapshot,
    manifest: RuleManifest,
    previous_assessment: RuleRiskAssessment | None = None,
) -> RuleRiskAssessment:
    """Evaluate one snapshot without I/O, clocks, models, or environment access."""

    if snapshot.market != Market.A_SHARE or manifest.market != Market.A_SHARE:
        raise ValueError("A-share rule evaluator only accepts A-share inputs")
    missing_groups = _missing_optional_groups(snapshot, manifest)
    core_missing = any(
        _number(snapshot.features.get(name)) is None
        for name in manifest.required_core_features
    )
    if (
        snapshot.data_quality in _UNUSABLE_DATA
        or snapshot.reliability_grade == "UNAVAILABLE"
        or core_missing
    ):
        return _unknown_assessment(snapshot, manifest, missing_groups)

    thresholds = manifest.thresholds
    cross_section_valid = _intraday_cross_section_valid(snapshot, manifest)
    if not cross_section_valid and "breadth" not in missing_groups:
        missing_groups = tuple((*missing_groups, "breadth"))
    triggered: list[TriggeredRule] = []

    def add(
        rule_id: str,
        layer: RuleLayer,
        points: int,
        observed_value: Any,
        threshold_description: str,
        *features: str,
    ) -> None:
        triggered.append(
            TriggeredRule(
                rule_id=rule_id,
                layer=layer,
                severity_points=points,
                observed_value=observed_value,
                threshold_description=threshold_description,
                evidence_ids=_evidence_ids(snapshot, tuple(features)),
            )
        )

    return_60d = _number(snapshot.features.get("return_60d"))
    if return_60d is not None and return_60d >= thresholds["return_60d"]:
        add(
            "vulnerability.return_60d",
            RuleLayer.VULNERABILITY,
            1,
            return_60d,
            ">= 15%",
            "return_60d",
        )
    margin_growth = _number(snapshot.features.get("margin_balance_growth_20d"))
    if margin_growth is not None and margin_growth >= thresholds["margin_balance_growth_20d"]:
        add(
            "vulnerability.margin_growth_20d",
            RuleLayer.VULNERABILITY,
            1,
            margin_growth,
            ">= 8%",
            "margin_balance_growth_20d",
        )
    turnover = _number(snapshot.features.get("turnover_percentile_20d"))
    above_ma20 = _number(snapshot.features.get("breadth_above_ma20_pct"))
    if (
        cross_section_valid
        and turnover is not None
        and above_ma20 is not None
        and turnover >= thresholds["turnover_percentile_20d"]
        and above_ma20 <= thresholds["breadth_above_ma20_pct"]
    ):
        add(
            "vulnerability.turnover_narrow_breadth",
            RuleLayer.VULNERABILITY,
            1,
            {"turnover_percentile_20d": turnover, "breadth_above_ma20_pct": above_ma20},
            "turnover >= 0.80 and breadth above MA20 <= 50%",
            "turnover_percentile_20d",
            "breadth_above_ma20_pct",
        )
    return_5d = _number(snapshot.features.get("return_5d"))
    breadth_up = _number(snapshot.features.get("breadth_up_pct"))
    if (
        cross_section_valid
        and return_5d is not None
        and breadth_up is not None
        and return_5d >= 0.0
        and breadth_up <= thresholds["breadth_up_pct_divergence"]
    ):
        add(
            "vulnerability.index_breadth_divergence",
            RuleLayer.VULNERABILITY,
            1,
            {"return_5d": return_5d, "breadth_up_pct": breadth_up},
            "5d return >= 0 and breadth up <= 40%",
            "return_5d",
            "breadth_up_pct",
        )

    volatility_ratio = _number(snapshot.features.get("volatility_ratio_5d_20d"))
    if volatility_ratio is not None and volatility_ratio >= thresholds["volatility_ratio_5d_20d_pressure"]:
        add(
            "pressure.volatility_acceleration",
            RuleLayer.PRESSURE,
            2,
            volatility_ratio,
            ">= 1.50",
            "volatility_ratio_5d_20d",
        )
    abnormal_range = snapshot.features.get("abnormal_range_weak_close_transition") is True
    if abnormal_range:
        add(
            "pressure.abnormal_range_weak_close",
            RuleLayer.PRESSURE,
            2,
            True,
            "transition confirmed",
            "abnormal_range_weak_close_transition",
        )
    breadth_deterioration = (
        cross_section_valid
        and snapshot.features.get("breadth_deterioration_transition") is True
    )
    if breadth_deterioration:
        add(
            "pressure.breadth_deterioration",
            RuleLayer.PRESSURE,
            2,
            True,
            "transition confirmed with valid current breadth",
            "breadth_deterioration_transition",
        )

    drawdown = _number(snapshot.features.get("drawdown_20d"))
    ma20_distance = _number(snapshot.features.get("ma20_distance"))
    if (
        drawdown is not None
        and ma20_distance is not None
        and drawdown <= thresholds["drawdown_20d"]
        and ma20_distance < 0.0
    ):
        add(
            "continuation.drawdown_below_ma20",
            RuleLayer.CONTINUATION,
            1,
            {"drawdown_20d": drawdown, "ma20_distance": ma20_distance},
            "drawdown <= -5% and MA20 distance < 0",
            "drawdown_20d",
            "ma20_distance",
        )
    if (
        return_5d is not None
        and volatility_ratio is not None
        and return_5d < 0.0
        and volatility_ratio >= thresholds["volatility_ratio_5d_20d_continuation"]
    ):
        add(
            "continuation.negative_return_volatility",
            RuleLayer.CONTINUATION,
            1,
            {"return_5d": return_5d, "volatility_ratio_5d_20d": volatility_ratio},
            "5d return < 0 and volatility ratio >= 1.20",
            "return_5d",
            "volatility_ratio_5d_20d",
        )
    margin_contracting = snapshot.features.get("margin_balance_contracting_from_high") is True
    if margin_contracting and breadth_deterioration:
        add(
            "continuation.margin_contracting_breadth",
            RuleLayer.CONTINUATION,
            1,
            True,
            "margin contracts from high with breadth deterioration",
            "margin_balance_contracting_from_high",
            "breadth_deterioration_transition",
        )

    range_zscore = _number(snapshot.features.get("range_zscore_20d"))
    close_location = _number(snapshot.features.get("close_location"))
    audited_return = _number(snapshot.features.get("audited_ohlc_return_1d"))
    if (
        range_zscore is not None
        and close_location is not None
        and audited_return is not None
        and range_zscore >= thresholds["range_zscore_20d_hard"]
        and close_location <= thresholds["close_location_hard"]
        and audited_return <= thresholds["audited_ohlc_return_1d_hard"]
    ):
        add(
            "hard.range_weak_close",
            RuleLayer.HARD_TRIGGER,
            2,
            {
                "range_zscore_20d": range_zscore,
                "close_location": close_location,
                "audited_ohlc_return_1d": audited_return,
            },
            "range z >= 3, close location <= 0.15, audited return <= -2%",
            "range_zscore_20d",
            "close_location",
            "audited_ohlc_return_1d",
        )
    return_1d = _number(snapshot.features.get("return_1d"))
    declining_stock_pct = _number(snapshot.features.get("declining_stock_pct"))
    if declining_stock_pct is None and breadth_up is not None:
        declining_stock_pct = 100.0 - breadth_up
    if (
        cross_section_valid
        and return_1d is not None
        and return_1d < 0.0
        and declining_stock_pct is not None
        and declining_stock_pct >= thresholds["declining_stock_pct_hard"]
    ):
        add(
            "hard.declining_breadth",
            RuleLayer.HARD_TRIGGER,
            2,
            declining_stock_pct,
            "index negative and declining stocks >= 85%",
            "return_1d",
            "breadth_up_pct",
        )
    limit_down = _number(snapshot.features.get("limit_down_pct"))
    if (
        cross_section_valid
        and return_1d is not None
        and return_1d < 0.0
        and limit_down is not None
        and limit_down >= thresholds["limit_down_pct_hard"]
    ):
        add(
            "hard.limit_down",
            RuleLayer.HARD_TRIGGER,
            2,
            limit_down,
            "index negative and limit-down share >= 2%",
            "return_1d",
            "limit_down_pct",
        )

    by_layer = {
        layer: tuple(item for item in triggered if item.layer == layer)
        for layer in RuleLayer
    }
    layer_scores = {
        layer: min(
            manifest.layer_caps[layer.value],
            sum(item.severity_points for item in by_layer[layer]),
        )
        for layer in RuleLayer
    }
    risk_score = min(manifest.score_cap, sum(layer_scores.values()))
    vulnerability_count = len(by_layer[RuleLayer.VULNERABILITY])
    pressure_count = len(by_layer[RuleLayer.PRESSURE])
    continuation_count = len(by_layer[RuleLayer.CONTINUATION])
    hard_triggered = bool(by_layer[RuleLayer.HARD_TRIGGER])
    previous_pressure_ids = (
        {
            item.rule_id
            for item in previous_assessment.triggered_rules
            if item.layer == RuleLayer.PRESSURE
        }
        if previous_assessment is not None
        else set()
    )
    current_pressure_ids = {item.rule_id for item in by_layer[RuleLayer.PRESSURE]}
    orange_then_new_pressure = (
        previous_assessment is not None
        and previous_assessment.risk_level == RiskLevel.ORANGE
        and bool(current_pressure_ids - previous_pressure_ids)
        and continuation_count >= 1
    )

    if (
        hard_triggered
        or (
            pressure_count >= 2
            and continuation_count >= 1
            and return_1d is not None
            and return_1d < 0.0
        )
        or orange_then_new_pressure
    ):
        risk_level = RiskLevel.RED
    elif (
        abnormal_range
        or pressure_count >= 2
        or (pressure_count >= 1 and (vulnerability_count >= 1 or continuation_count >= 1))
    ):
        risk_level = RiskLevel.ORANGE
    elif vulnerability_count >= 2 or pressure_count == 1:
        risk_level = RiskLevel.YELLOW
    else:
        risk_level = RiskLevel.GREEN

    return RuleRiskAssessment(
        market=snapshot.market,
        as_of_time=snapshot.as_of_time,
        engine_version=manifest.engine_version,
        manifest_sha256=manifest.manifest_sha256,
        risk_level=risk_level,
        risk_score=risk_score,
        market_phase=_phase(snapshot),
        triggered_rules=tuple(triggered),
        missing_optional_groups=missing_groups,
        reliability_grade=_reliability(snapshot, missing_groups, cross_section_valid),
        evaluation_latency_ms=0.0,
    )
