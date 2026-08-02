"""Historical evaluation for the frozen A-share rule manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import pandas as pd

from .domain import DataStatus, Evidence, FeatureSnapshot, Market, RiskLevel
from .rule_policy import RuleManifest, evaluate_a_share_rules, load_rule_manifest


_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_DATASET = (
    _PROJECT_ROOT
    / "harness_data/market_warning/features/a-share-2000-2026-07-31.joblib"
)
_DEFAULT_OUTPUT = (
    _PROJECT_ROOT / "harness_data/models/market_warning/rule-v1-evaluation.json"
)
_NON_FEATURE_COLUMNS = frozenset(
    {
        "as_of_time",
        "feature_available_at",
        "close",
        "future_return_1d",
        "future_worst_return_3d",
        "label_1d",
        "label_3d",
        "label_end_1d",
        "label_end_3d",
        "partition",
        "crisis_period",
        "old_market_risk_alert",
        "old_market_risk_score",
    }
)


def _feature_columns(frame: pd.DataFrame) -> tuple[str, ...]:
    return tuple(
        column
        for column in frame.columns
        if column not in _NON_FEATURE_COLUMNS and not column.endswith("_available_at")
    )


def _validate_availability(frame: pd.DataFrame, features: tuple[str, ...]) -> pd.Series:
    if "as_of_time" not in frame:
        raise ValueError("frame must include as_of_time")
    times = pd.to_datetime(frame["as_of_time"], utc=True, errors="coerce")
    if times.isna().any() or not times.is_monotonic_increasing or times.duplicated().any():
        raise ValueError("as_of_time must be unique, valid, and chronological")
    proof = dict(frame.attrs.get("availability_proof", {}))
    for feature in features:
        proof_column = proof.get(feature) or proof.get("*")
        if not proof_column or proof_column not in frame:
            raise ValueError(f"missing availability proof for {feature}")
    proof_columns = tuple(dict.fromkeys(proof.get(feature) or proof.get("*") for feature in features))
    for column in proof_columns:
        available = pd.to_datetime(frame[column], utc=True, errors="coerce")
        if available.isna().any():
            raise ValueError(f"{column} must contain valid availability timestamps")
        if available.gt(times).any():
            raise ValueError(f"{column} contains future feature availability")
    return times


def _clean_value(value: Any) -> Any:
    if value is None:
        return None
    try:
        if bool(pd.isna(value)):
            return None
    except (TypeError, ValueError):
        pass
    if hasattr(value, "item"):
        try:
            return value.item()
        except ValueError:
            return value
    return value


def _snapshot_from_row(
    row: Mapping[str, Any],
    as_of_time: pd.Timestamp,
    features: tuple[str, ...],
) -> FeatureSnapshot:
    values = {name: _clean_value(row[name]) for name in features}
    evidence = tuple(
        Evidence(
            evidence_id=f"historical:{name}:{as_of_time.isoformat()}",
            group="historical_feature",
            summary=name,
            value=value,
            source="point_in_time_dataset",
            as_of_time=as_of_time.to_pydatetime(),
        )
        for name, value in values.items()
    )
    return FeatureSnapshot(
        market=Market.A_SHARE,
        as_of_time=as_of_time.to_pydatetime(),
        session_slot="historical-daily",
        feature_version="market-warning-v2",
        features=values,
        evidence=evidence,
        data_quality=DataStatus.FRESH,
        reliability_grade="A",
        source_times={"point_in_time_dataset": as_of_time.to_pydatetime()},
    )


def _partition(value: pd.Timestamp) -> str | None:
    year = value.year
    if 2000 <= year <= 2012:
        return "dev"
    if 2013 <= year <= 2019:
        return "validation"
    if 2020 <= year <= 2026:
        return "test"
    return None


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _metrics(frame: pd.DataFrame) -> dict[str, Any]:
    observations = len(frame)
    positives = int(frame["_positive"].sum()) if observations else 0
    entries = frame.loc[frame["_alert_entry"]]
    alert_entries = len(entries)
    true_positives = int(entries["_positive"].sum()) if alert_entries else 0
    precision = _ratio(true_positives, alert_entries)
    recall = _ratio(true_positives, positives)
    base_rate = _ratio(positives, observations)
    lift = (
        precision / base_rate
        if precision is not None and base_rate not in (None, 0.0)
        else None
    )
    if observations:
        months = frame["as_of_time"].dt.tz_localize(None).dt.to_period("M").nunique()
    else:
        months = 0
    alerts_per_month = alert_entries / months if months else 0.0

    crisis_counts = (
        entries.loc[entries["_positive"] & entries["crisis_period"].notna(), "crisis_period"]
        .astype(str)
        .value_counts()
    )
    crisis_total = int(crisis_counts.sum())
    crisis_contribution = {
        name: float(count / crisis_total)
        for name, count in sorted(crisis_counts.items())
    }
    max_crisis_contribution = max(crisis_contribution.values(), default=0.0)

    phase_breakdown = {}
    for phase in ("FIRST_SHOCK", "CONTINUATION"):
        phase_rows = frame.loc[frame["market_phase"] == phase]
        phase_entries = phase_rows.loc[phase_rows["_alert_entry"]]
        phase_positives = int(phase_rows["_positive"].sum())
        phase_true_positives = int(phase_entries["_positive"].sum())
        phase_breakdown[phase] = {
            "observations": len(phase_rows),
            "positives": phase_positives,
            "alert_entries": len(phase_entries),
            "true_positives": phase_true_positives,
            "precision": _ratio(phase_true_positives, len(phase_entries)),
            "recall": _ratio(phase_true_positives, phase_positives),
        }

    old_alerts = frame["old_market_risk_alert"].fillna(False).astype(bool)
    old_budget = int(old_alerts.sum())
    old_true_positives = int((old_alerts & frame["_positive"]).sum())
    old_recall = _ratio(old_true_positives, positives)
    if old_budget:
        selected = frame.sort_values(
            ["_risk_rank", "_risk_score", "as_of_time"],
            ascending=[False, False, True],
            kind="stable",
        ).head(old_budget)
        rule_recall_at_old_budget = _ratio(int(selected["_positive"].sum()), positives)
    else:
        rule_recall_at_old_budget = None

    return {
        "observations": observations,
        "positives": positives,
        "alert_entries": alert_entries,
        "true_positives": true_positives,
        "precision": precision,
        "recall": recall,
        "base_rate": base_rate,
        "lift": lift,
        "alerts_per_month": alerts_per_month,
        "phase_breakdown": phase_breakdown,
        "crisis_contribution": crisis_contribution,
        "max_crisis_contribution": max_crisis_contribution,
        "old_market_risk_recall": old_recall,
        "rule_recall_at_old_budget": rule_recall_at_old_budget,
    }


def evaluate_rule_frame(frame: pd.DataFrame, manifest: RuleManifest) -> dict[str, Any]:
    """Evaluate frozen rules without changing their thresholds."""

    values = frame.copy()
    features = _feature_columns(values)
    times = _validate_availability(values, features)
    for label in ("label_1d", "label_3d"):
        if label not in values:
            raise ValueError(f"frame must include {label}")
    values["as_of_time"] = times
    values["crisis_period"] = values.get("crisis_period")
    values["old_market_risk_alert"] = values.get("old_market_risk_alert", False)
    levels: list[str] = []
    scores: list[float] = []
    phases: list[str] = []
    previous = None
    for position, (_, row) in enumerate(values.iterrows()):
        snapshot = _snapshot_from_row(row, times.iloc[position], features)
        assessment = evaluate_a_share_rules(snapshot, manifest, previous_assessment=previous)
        levels.append(assessment.risk_level.value)
        scores.append(assessment.risk_score)
        phases.append(assessment.market_phase.value)
        previous = assessment
    values["market_phase"] = phases
    values["_risk_level"] = levels
    values["_risk_score"] = scores
    values["_risk_rank"] = values["_risk_level"].map(
        {RiskLevel.GREEN.value: 0, RiskLevel.YELLOW.value: 0, RiskLevel.ORANGE.value: 1, RiskLevel.RED.value: 2, RiskLevel.UNKNOWN.value: 0}
    )
    high = values["_risk_level"].isin((RiskLevel.ORANGE.value, RiskLevel.RED.value))
    values["_alert_entry"] = high & ~high.shift(1, fill_value=False)
    values["_positive"] = (
        values["label_1d"].fillna(False).astype(bool)
        | values["label_3d"].fillna(False).astype(bool)
    )
    values["_partition"] = values["as_of_time"].map(_partition)
    partitions = {
        name: _metrics(values.loc[values["_partition"] == name].copy())
        for name in ("dev", "validation", "test")
    }
    frozen = partitions["test"]
    return {
        "engine_version": manifest.engine_version,
        "manifest_sha256": manifest.manifest_sha256,
        "market": Market.A_SHARE.value,
        "previously_observed_holdout": True,
        "partitions": partitions,
        "production_gates": {
            "lift": frozen["lift"],
            "alerts_per_month": frozen["alerts_per_month"],
            "max_crisis_contribution": frozen["max_crisis_contribution"],
        },
    }


def _load_frame(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".joblib", ".pkl", ".pickle"}:
        value = joblib.load(path)
    elif suffix == ".parquet":
        value = pd.read_parquet(path)
    elif suffix == ".csv":
        value = pd.read_csv(path)
    else:
        raise ValueError(f"unsupported dataset format: {suffix}")
    if not isinstance(value, pd.DataFrame):
        raise ValueError("dataset must contain a pandas DataFrame")
    return value


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate frozen A-share warning rules.")
    parser.add_argument("--market", choices=("a_share",), default="a_share")
    parser.add_argument("--dataset", default=str(_DEFAULT_DATASET))
    parser.add_argument(
        "--manifest",
        default=str(Path(__file__).with_name("rule_manifest_v1.json")),
    )
    parser.add_argument("--output", default=str(_DEFAULT_OUTPUT))
    args = parser.parse_args(argv)
    result = evaluate_rule_frame(
        _load_frame(Path(args.dataset)),
        load_rule_manifest(args.manifest),
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(result["production_gates"], ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
