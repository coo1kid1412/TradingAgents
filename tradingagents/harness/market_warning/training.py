"""Leakage-safe labels, chronological model fitting, and evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import exchange_calendars as xcals
import joblib
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .domain import Market
from .features import FEATURE_VERSION


_CANONICAL_MODULE = "tradingagents.harness.market_warning.training"
if __name__ == "__main__":
    sys.modules[_CANONICAL_MODULE] = sys.modules[__name__]


MODEL_VERSION = "market-warning-logistic-v1"
CALIBRATION_VERSION = "platt-v1"
EMBARGO_TRADING_DAYS = 3
TEST_END = pd.Timestamp("2026-07-31", tz="UTC")
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_FEATURE_ROOT = _PROJECT_ROOT / "harness_data/market_warning/features"
_DEFAULT_DATASETS = {
    Market.A_SHARE: _DEFAULT_FEATURE_ROOT / "a-share-2000-2026-07-31.joblib",
    Market.US: _DEFAULT_FEATURE_ROOT / "us-2000-2026-07-31.joblib",
}
_DEFAULT_ARTIFACT_ROOT = _PROJECT_ROOT / "harness_data/models/market_warning"
_DEFAULT_EVALUATION_REPORT = (
    _PROJECT_ROOT / "reports/market_warning/model-evaluation/market-warning-v1.md"
)
_MODEL_VERSION_PATTERN = re.compile(
    r"[A-Za-z0-9](?:[A-Za-z0-9._-]{0,62}[A-Za-z0-9])?\Z"
)

_LABEL_THRESHOLDS = {
    Market.A_SHARE: {"1d": -0.04, "3d": -0.06},
    Market.US: {"1d": -0.03, "3d": -0.05},
}
_NON_FEATURE_COLUMNS = {
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


@dataclass(frozen=True)
class CalibrationBin:
    lower_bound: float
    upper_bound: float
    count: int
    mean_probability: float | None
    observed_rate: float | None


@dataclass(frozen=True)
class PhaseMetrics:
    observations: int
    positives: int
    alerts: int
    true_positives: int
    recall: float | None


@dataclass(frozen=True)
class EvaluationReport:
    market: Market
    horizon: str
    observations: int
    prevalence: float
    brier_score: float
    average_precision: float
    constant_base_rate_brier: float
    expected_calibration_error: float
    calibration_bins: tuple[CalibrationBin, ...]
    phase_breakdown: Mapping[str, PhaseMetrics]
    crisis_contribution: Mapping[str, float]
    monthly_alert_entries: float
    old_market_risk_recall: float | None
    model_recall_at_old_budget: float | None

    def __post_init__(self) -> None:
        object.__setattr__(self, "market", Market(self.market))
        object.__setattr__(self, "calibration_bins", tuple(self.calibration_bins))
        object.__setattr__(self, "phase_breakdown", MappingProxyType(dict(self.phase_breakdown)))
        object.__setattr__(self, "crisis_contribution", MappingProxyType(dict(self.crisis_contribution)))


@dataclass(frozen=True)
class ModelBundle:
    market: Market
    horizon: str
    feature_names: tuple[str, ...]
    feature_version: str
    model_version: str
    calibration_version: str
    calibration_method: str
    base_rate: float
    training_start: datetime
    training_end: datetime
    calibration_start: datetime
    calibration_end: datetime
    pipeline: Pipeline
    calibrator: LogisticRegression

    def predict_proba(self, frame: pd.DataFrame | Mapping[str, Any]) -> np.ndarray:
        values = _feature_matrix(frame, self.feature_names)
        raw_scores = np.asarray(self.pipeline.decision_function(values), dtype=float).reshape(-1, 1)
        return np.asarray(self.calibrator.predict_proba(raw_scores)[:, 1], dtype=float)

    def contributions(self, features: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
        values = _feature_matrix(features, self.feature_names)
        imputer: SimpleImputer = self.pipeline.named_steps["imputer"]
        scaler: StandardScaler = self.pipeline.named_steps["scaler"]
        classifier: LogisticRegression = self.pipeline.named_steps["classifier"]
        imputed = imputer.transform(values)
        transformed = np.asarray(scaler.transform(imputed), dtype=float)[0]
        coefficients = np.asarray(classifier.coef_[0], dtype=float)
        names = list(self.feature_names)
        indicator = getattr(imputer, "indicator_", None)
        if indicator is not None:
            names.extend(f"{self.feature_names[int(index)]}__missing" for index in indicator.features_)
        rows = [
            {"feature": name, "contribution": float(coefficient * value)}
            for name, coefficient, value in zip(names, coefficients, transformed, strict=True)
        ]
        rows.sort(key=lambda row: (-abs(row["contribution"]), row["feature"]))
        return tuple(rows)


if __name__ == "__main__":
    ModelBundle.__module__ = _CANONICAL_MODULE


def build_labels(index_frame: pd.DataFrame, market: Market) -> pd.DataFrame:
    """Attach frozen 1-day and worst cumulative 3-day labels to index rows."""

    market = Market(market)
    frame = index_frame.copy()
    times = _as_of_times(frame)
    if not times.is_monotonic_increasing or times.duplicated().any():
        raise ValueError("as_of_time must be unique and chronological")
    if "close" not in frame:
        raise ValueError("index_frame must include close")
    closes = pd.to_numeric(frame["close"], errors="coerce")
    if closes.isna().any() or (closes <= 0).any():
        raise ValueError("close must contain finite positive values")
    source_columns = tuple(
        column
        for column in frame
        if column != "as_of_time"
        and column not in _NON_FEATURE_COLUMNS - {"close"}
        and not column.endswith("_available_at")
    )
    _validate_availability_proof(frame, source_columns)

    future_1d = closes.shift(-1) / closes - 1.0
    future_returns = pd.concat(
        [closes.shift(-offset) / closes - 1.0 for offset in range(1, 4)], axis=1
    )
    future_worst_3d = future_returns.min(axis=1, skipna=False)
    frame["future_return_1d"] = future_1d
    frame["future_worst_return_3d"] = future_worst_3d
    frame["label_1d"] = future_1d.le(_LABEL_THRESHOLDS[market]["1d"]).where(future_1d.notna()).astype("boolean")
    frame["label_3d"] = future_worst_3d.le(_LABEL_THRESHOLDS[market]["3d"]).where(future_worst_3d.notna()).astype("boolean")
    frame["label_end_1d"] = times.shift(-1)
    frame["label_end_3d"] = times.shift(-3)
    frame.attrs.update(index_frame.attrs)
    frame.attrs["label_market"] = market.value
    frame.attrs["point_in_time_validated"] = True
    return frame


def time_partitions(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return frozen chronological splits with three-row boundary purging."""

    values = frame.copy()
    values["as_of_time"] = pd.to_datetime(_as_of_times(values), utc=True)
    values = values.sort_values("as_of_time", kind="stable")
    times = values["as_of_time"]
    definitions = (
        ("dev", pd.Timestamp("2000-01-01", tz="UTC"), pd.Timestamp("2012-12-31 23:59:59.999999", tz="UTC")),
        ("validation", pd.Timestamp("2013-01-01", tz="UTC"), pd.Timestamp("2019-12-31 23:59:59.999999", tz="UTC")),
        ("test", pd.Timestamp("2020-01-01", tz="UTC"), TEST_END + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)),
    )
    partitions: list[pd.DataFrame] = []
    if "label_end_3d" not in values:
        raise ValueError("frame must include label_end_3d for boundary-safe partitions")
    label_ends = pd.to_datetime(values["label_end_3d"], utc=True, errors="coerce")
    for name, start, end in definitions:
        mask = times.between(start, end, inclusive="both")
        part = values.loc[mask].copy()
        ends = label_ends.loc[part.index]
        part = part.loc[ends.notna() & ends.le(end)]
        part["partition"] = name
        part.attrs.update(frame.attrs)
        partitions.append(part.reset_index(drop=True))
    return tuple(partitions)  # type: ignore[return-value]


def production_partitions(
    frame: pd.DataFrame,
    market: Market,
    *,
    calibration_sessions: int = 252,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return a latest-date refit split after the frozen evaluation is complete."""

    market = Market(market)
    values = frame.copy()
    values["as_of_time"] = pd.to_datetime(_as_of_times(values), utc=True)
    values = values.sort_values("as_of_time", kind="stable")
    if "label_end_3d" not in values:
        raise ValueError("frame must include label_end_3d for production refit")
    label_ends = pd.to_datetime(values["label_end_3d"], utc=True, errors="coerce")
    eligible = values.loc[label_ends.notna()].copy()
    if len(eligible) < 80:
        raise ValueError("production refit requires at least 80 labeled sessions")
    calibration_size = min(calibration_sessions, max(40, len(eligible) // 5))
    calibration = eligible.tail(calibration_size).copy()
    calibration_start = calibration["as_of_time"].min()
    training = eligible.loc[
        eligible["as_of_time"].lt(calibration_start)
        & pd.to_datetime(eligible["label_end_3d"], utc=True, errors="coerce").lt(calibration_start)
    ].copy()
    while not training.empty and _embargo_sessions(
        market,
        training["as_of_time"].max(),
        calibration_start,
    ) < EMBARGO_TRADING_DAYS:
        training = training.iloc[:-1].copy()
    if training.empty:
        raise ValueError("production refit has no training rows after the embargo")
    training["partition"] = "production_train"
    calibration["partition"] = "production_calibration"
    training.attrs.update(frame.attrs)
    calibration.attrs.update(frame.attrs)
    return training.reset_index(drop=True), calibration.reset_index(drop=True)


def fit_model(
    train: pd.DataFrame,
    calibration: pd.DataFrame,
    market: Market,
    horizon: str,
    *,
    model_version: str = MODEL_VERSION,
) -> ModelBundle:
    """Fit one deterministic logistic pipeline and a later sigmoid calibrator."""

    market = Market(market)
    target = _target_column(horizon)
    if train.empty or calibration.empty:
        raise ValueError("training and calibration windows must not be empty")
    train_times = pd.to_datetime(_as_of_times(train), utc=True)
    calibration_times = pd.to_datetime(_as_of_times(calibration), utc=True)
    if target not in train or target not in calibration:
        raise ValueError(f"training and calibration frames must include {target}")
    train_labeled = train[target].notna()
    calibration_labeled = calibration[target].notna()
    labeled_train_times = train_times.loc[train_labeled]
    labeled_calibration_times = calibration_times.loc[calibration_labeled]
    if labeled_train_times.empty or labeled_calibration_times.empty:
        raise ValueError("training and calibration windows must contain labeled rows")
    if labeled_train_times.max() >= labeled_calibration_times.min():
        raise ValueError("calibration window must be later than the training window")
    if _embargo_sessions(
        market,
        labeled_train_times.max(),
        labeled_calibration_times.min(),
    ) < EMBARGO_TRADING_DAYS:
        raise ValueError("calibration window must follow a three-trading-day embargo")
    feature_version = str(train.attrs.get("feature_version", FEATURE_VERSION))
    calibration_feature_version = str(calibration.attrs.get("feature_version", feature_version))
    if calibration_feature_version != feature_version:
        raise ValueError("training and calibration feature versions must match")
    feature_names = tuple(
        sorted(
            column
            for column in train.columns
            if column not in _NON_FEATURE_COLUMNS and not column.endswith("_available_at")
        )
    )
    if not feature_names:
        raise ValueError("no model features are available")
    missing_calibration = set(feature_names).difference(calibration.columns)
    if missing_calibration:
        raise ValueError(f"calibration is missing features: {sorted(missing_calibration)}")
    _validate_model_frame(train, feature_names, "training")
    _validate_model_frame(calibration, feature_names, "calibration")

    x_train, y_train = _labeled_rows(train, feature_names, target)
    x_calibration, y_calibration = _labeled_rows(calibration, feature_names, target)
    _require_binary_classes(y_train, "training")
    _require_binary_classes(y_calibration, "calibration")
    pipeline = Pipeline(
        steps=(
            ("imputer", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=2000,
                    solver="lbfgs",
                    random_state=0,
                ),
            ),
        )
    )
    pipeline.fit(x_train, y_train)
    calibration_scores = np.asarray(pipeline.decision_function(x_calibration), dtype=float).reshape(-1, 1)
    calibrator = LogisticRegression(C=1_000_000.0, max_iter=2000, solver="lbfgs", random_state=0)
    calibrator.fit(calibration_scores, y_calibration)
    return ModelBundle(
        market=market,
        horizon=horizon,
        feature_names=feature_names,
        feature_version=feature_version,
        model_version=model_version,
        calibration_version=CALIBRATION_VERSION,
        calibration_method="platt",
        base_rate=float(y_train.mean()),
        training_start=labeled_train_times.min().to_pydatetime(),
        training_end=labeled_train_times.max().to_pydatetime(),
        calibration_start=labeled_calibration_times.min().to_pydatetime(),
        calibration_end=labeled_calibration_times.max().to_pydatetime(),
        pipeline=pipeline,
        calibrator=calibrator,
    )


def evaluate_model(bundle: ModelBundle, test: pd.DataFrame) -> EvaluationReport:
    """Evaluate one frozen bundle without tuning it on the supplied rows."""

    _validate_model_frame(test, bundle.feature_names, "test")
    target = _target_column(bundle.horizon)
    labeled = test.loc[test[target].notna()].copy()
    if labeled.empty:
        raise ValueError("test data has no labeled rows")
    y_true = labeled[target].astype(int).to_numpy()
    _require_binary_classes(y_true, "test")
    probabilities = bundle.predict_proba(labeled)
    prevalence = float(y_true.mean())
    bins = _calibration_bins(y_true, probabilities)
    ece = sum(
        item.count / len(y_true) * abs(float(item.mean_probability) - float(item.observed_rate))
        for item in bins
        if item.count
    )
    alert_threshold = min(1.0, bundle.base_rate * 4.0)
    alerts = probabilities >= alert_threshold
    times = pd.to_datetime(_as_of_times(labeled), utc=True)
    phases = _phase_breakdown(labeled, y_true, alerts)
    crises = _crisis_contribution(labeled, times, y_true, alerts)
    monthly_budget = _monthly_alert_entries(times, alerts)
    old_alerts = _old_alerts(labeled)
    old_recall = _recall(y_true, old_alerts) if old_alerts is not None else None
    model_at_old_budget = _recall_at_budget(y_true, probabilities, int(old_alerts.sum())) if old_alerts is not None else None
    return EvaluationReport(
        market=bundle.market,
        horizon=bundle.horizon,
        observations=len(y_true),
        prevalence=prevalence,
        brier_score=float(brier_score_loss(y_true, probabilities)),
        average_precision=float(average_precision_score(y_true, probabilities)),
        constant_base_rate_brier=float(brier_score_loss(y_true, np.full(len(y_true), bundle.base_rate))),
        expected_calibration_error=float(ece),
        calibration_bins=bins,
        phase_breakdown=phases,
        crisis_contribution=crises,
        monthly_alert_entries=monthly_budget,
        old_market_risk_recall=old_recall,
        model_recall_at_old_budget=model_at_old_budget,
    )


def _as_of_times(frame: pd.DataFrame) -> pd.Series:
    if "as_of_time" in frame:
        values = pd.Series(frame["as_of_time"], index=frame.index)
    elif isinstance(frame.index, pd.DatetimeIndex):
        values = pd.Series(frame.index, index=frame.index)
    else:
        raise ValueError("frame must include as_of_time or use a DatetimeIndex")
    for value in values:
        try:
            timestamp = pd.Timestamp(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("as_of_time must contain valid timestamps") from exc
        if timestamp.tzinfo is None or timestamp.utcoffset() is None:
            raise ValueError("as_of_time must be timezone-aware")
    return pd.to_datetime(values, errors="raise", utc=True)


def _target_column(horizon: str) -> str:
    if horizon not in {"1d", "3d"}:
        raise ValueError("horizon must be '1d' or '3d'")
    return f"label_{horizon}"


def _validate_model_frame(
    frame: pd.DataFrame,
    feature_names: Sequence[str],
    frame_name: str,
) -> None:
    if frame.attrs.get("point_in_time_validated") is not True:
        raise ValueError(f"{frame_name} frame must carry an explicit point-in-time validated marker")
    proof = frame.attrs.get("availability_proof")
    if not isinstance(proof, Mapping):
        raise ValueError(f"{frame_name} frame must document availability proof")
    label_source = "close" if "close" in proof else "label_source"
    _validate_availability_proof(frame, tuple(feature_names) + (label_source,))


def validate_model_version(model_version: str) -> str:
    """Return a filesystem-safe model version component."""

    if not isinstance(model_version, str) or not _MODEL_VERSION_PATTERN.fullmatch(model_version):
        raise ValueError(
            "model_version must be 1-64 ASCII letters, digits, dots, underscores, or hyphens "
            "and must begin and end with a letter or digit"
        )
    return model_version


def _validate_availability_proof(frame: pd.DataFrame, source_names: Sequence[str]) -> None:
    proof = frame.attrs.get("availability_proof")
    if not isinstance(proof, Mapping) or not proof:
        raise ValueError("frame must document availability proof for every source")
    as_of = pd.to_datetime(_as_of_times(frame), utc=True)
    for source_name in source_names:
        availability_column = proof.get(source_name, proof.get("*"))
        if not isinstance(availability_column, str) or not availability_column:
            raise ValueError(f"availability proof is missing source: {source_name}")
        if availability_column not in frame:
            raise ValueError(f"availability proof column is missing: {availability_column}")
        available = pd.to_datetime(frame[availability_column], utc=True, errors="coerce")
        if available.isna().any():
            raise ValueError(f"availability proof contains missing timestamps: {availability_column}")
        if (available > as_of).any():
            raise ValueError(f"point-in-time violation: {availability_column} is after as_of_time")


def _embargo_sessions(market: Market, training_end: pd.Timestamp, calibration_start: pd.Timestamp) -> int:
    calendar_name = "XSHG" if market == Market.A_SHARE else "XNYS"
    start = pd.Timestamp(training_end.date()) + pd.Timedelta(days=1)
    end = pd.Timestamp(calibration_start.date()) - pd.Timedelta(days=1)
    if start > end:
        return 0
    calendar = xcals.get_calendar(calendar_name)
    if start < calendar.first_session or end > calendar.last_session:
        return int(np.busday_count(start.date(), calibration_start.date()))
    return len(calendar.sessions_in_range(start, end))


def _feature_matrix(
    frame: pd.DataFrame | Mapping[str, Any], feature_names: Sequence[str]
) -> pd.DataFrame:
    if isinstance(frame, Mapping):
        values = pd.DataFrame([{name: frame.get(name) for name in feature_names}])
    else:
        missing = set(feature_names).difference(frame.columns)
        if missing:
            raise ValueError(f"frame is missing model features: {sorted(missing)}")
        values = frame.loc[:, list(feature_names)].copy()
    for name in feature_names:
        if name == "market_phase":
            values[name] = values[name].map({"FIRST_SHOCK": 0.0, "CONTINUATION": 1.0})
        elif pd.api.types.is_bool_dtype(values[name].dtype):
            values[name] = values[name].astype(float)
        else:
            values[name] = pd.to_numeric(values[name], errors="coerce")
    return values.astype(float)


def _labeled_rows(
    frame: pd.DataFrame, feature_names: Sequence[str], target: str
) -> tuple[pd.DataFrame, np.ndarray]:
    if target not in frame:
        raise ValueError(f"frame must include {target}")
    mask = frame[target].notna()
    return _feature_matrix(frame.loc[mask], feature_names), frame.loc[mask, target].astype(int).to_numpy()


def _require_binary_classes(labels: np.ndarray, window_name: str) -> None:
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError(f"{window_name} window must contain both label classes")


def _calibration_bins(labels: np.ndarray, probabilities: np.ndarray) -> tuple[CalibrationBin, ...]:
    rows = []
    edges = np.linspace(0.0, 1.0, 11)
    indices = np.minimum(np.floor(probabilities * 10).astype(int), 9)
    for index in range(10):
        selected = indices == index
        count = int(selected.sum())
        rows.append(
            CalibrationBin(
                lower_bound=float(edges[index]),
                upper_bound=float(edges[index + 1]),
                count=count,
                mean_probability=float(probabilities[selected].mean()) if count else None,
                observed_rate=float(labels[selected].mean()) if count else None,
            )
        )
    return tuple(rows)


def _phase_breakdown(
    frame: pd.DataFrame, labels: np.ndarray, alerts: np.ndarray
) -> Mapping[str, PhaseMetrics]:
    phase_values = frame.get("market_phase", pd.Series([None] * len(frame), index=frame.index)).astype("object")
    result = {}
    for phase in ("FIRST_SHOCK", "CONTINUATION"):
        selected = phase_values.eq(phase).to_numpy()
        positives = int(labels[selected].sum())
        true_positives = int((alerts[selected] & labels[selected].astype(bool)).sum())
        result[phase] = PhaseMetrics(
            observations=int(selected.sum()),
            positives=positives,
            alerts=int(alerts[selected].sum()),
            true_positives=true_positives,
            recall=None if positives == 0 else true_positives / positives,
        )
    return result


def _crisis_contribution(
    frame: pd.DataFrame,
    times: pd.Series,
    labels: np.ndarray,
    alerts: np.ndarray,
) -> Mapping[str, float]:
    if "crisis_period" in frame:
        crisis = frame["crisis_period"].fillna("non_crisis").astype(str).to_numpy()
    else:
        years = times.dt.year.to_numpy()
        crisis = np.where(np.isin(years, [2008, 2015, 2020, 2022]), years.astype(str), "non_crisis")
    true_positive = alerts & labels.astype(bool)
    total = int(true_positive.sum())
    names = sorted(set(crisis).union({"2008", "2015", "2020", "2022", "non_crisis"}))
    return {
        name: (0.0 if total == 0 else float((true_positive & (crisis == name)).sum()) / total)
        for name in names
    }


def _monthly_alert_entries(times: pd.Series, alerts: np.ndarray) -> float:
    if len(alerts) == 0:
        return 0.0
    entries = alerts & ~pd.Series(alerts).shift(1, fill_value=False).to_numpy()
    months = times.dt.tz_convert("UTC").dt.strftime("%Y-%m")
    counts = pd.Series(entries.astype(int)).groupby(months.reset_index(drop=True)).sum()
    return float(counts.mean()) if len(counts) else 0.0


def _old_alerts(frame: pd.DataFrame) -> np.ndarray | None:
    if "old_market_risk_alert" in frame:
        return frame["old_market_risk_alert"].fillna(False).astype(bool).to_numpy()
    if "old_market_risk_score" in frame:
        return pd.to_numeric(frame["old_market_risk_score"], errors="coerce").ge(4).to_numpy()
    return None


def _recall(labels: np.ndarray, alerts: np.ndarray) -> float | None:
    positives = int(labels.sum())
    return None if positives == 0 else float((alerts & labels.astype(bool)).sum()) / positives


def _recall_at_budget(labels: np.ndarray, probabilities: np.ndarray, budget: int) -> float | None:
    positives = int(labels.sum())
    if positives == 0:
        return None
    selected = np.zeros(len(labels), dtype=bool)
    if budget > 0:
        order = np.argsort(-probabilities, kind="stable")[: min(budget, len(labels))]
        selected[order] = True
    return float((selected & labels.astype(bool)).sum()) / positives


def _command_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Leakage-safe market-warning model operations")
    commands = parser.add_subparsers(dest="command", required=True)

    def add_frozen_window(command: argparse.ArgumentParser) -> None:
        command.add_argument("--start", default="2000-01-01")
        command.add_argument("--test-end", default="2026-07-31")
        command.add_argument("--version", default=MODEL_VERSION)

    backfill = commands.add_parser("backfill", help="Prepare point-in-time history (Task 12 gated)")
    add_frozen_window(backfill)
    backfill.add_argument("--market", choices=[market.value for market in Market])
    backfill.add_argument("--output")
    backfill.add_argument("--cache-root")

    train = commands.add_parser("train", help="Fit one local bundle or the frozen four-model set")
    add_frozen_window(train)
    train.add_argument("--dataset")
    train.add_argument("--market", choices=[market.value for market in Market])
    train.add_argument("--horizon", choices=("1d", "3d"))
    train.add_argument("--artifact-root")
    train.add_argument("--db")
    train.add_argument("--dataset-a-share")
    train.add_argument("--dataset-us")
    train.add_argument("--report")
    train.add_argument("--manifest")

    evaluate = commands.add_parser("evaluate", help="Evaluate one local bundle on the frozen test partition")
    add_frozen_window(evaluate)
    evaluate.add_argument("--dataset")
    evaluate.add_argument("--artifact")

    promote = commands.add_parser("promote", help="Verify and atomically promote four passing artifacts")
    add_frozen_window(promote)
    promote.add_argument("--db")
    promote.add_argument("--artifact-root")
    promote.add_argument("--manifest")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _command_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "backfill":
            _validate_backfill_window(args.start, args.test_end)
        else:
            _validate_cli_window(args.start, args.test_end)
        try:
            validate_model_version(args.version)
        except ValueError as exc:
            raise ValueError(f"--version is invalid: {exc}") from exc
        if args.command == "backfill":
            payload = _run_backfill_command(args)
            print(json.dumps(payload, sort_keys=True, separators=(",", ":")))
            return 0
        if args.command == "promote":
            payload = _run_promote_command(args)
            print(json.dumps(payload, sort_keys=True, separators=(",", ":")))
            return 0
        if args.command == "train":
            single_model = any((args.dataset, args.market, args.horizon))
            payload = _run_train_command(args) if single_model else _run_train_all_command(args)
        else:
            payload = _run_evaluate_command(args)
    except (FileNotFoundError, OSError, TypeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(payload, sort_keys=True, separators=(",", ":")))
    return 0


def _validate_cli_window(start: str, test_end: str) -> None:
    start_date = pd.Timestamp(start)
    end_date = pd.Timestamp(test_end)
    if start_date > pd.Timestamp("2000-01-01"):
        raise ValueError("--start must include the frozen development window beginning 2000-01-01")
    if end_date != TEST_END.tz_localize(None):
        raise ValueError("--test-end must remain frozen at 2026-07-31")


def _validate_backfill_window(start: str, test_end: str) -> None:
    try:
        start_date = pd.Timestamp(start)
        end_date = pd.Timestamp(test_end)
    except ValueError as exc:
        raise ValueError("backfill dates must use YYYY-MM-DD") from exc
    if start_date > end_date:
        raise ValueError("backfill --start must not be after --test-end")
    if end_date > TEST_END.tz_localize(None):
        raise ValueError("backfill --test-end must not pass the frozen 2026-07-31 cutoff")


def _calendar_resolvers(market: Market):
    from .calendars import session_resolvers

    return session_resolvers(market)


def _run_backfill_command(args: argparse.Namespace) -> dict[str, Any]:
    missing = [name for name in ("market", "output") if not getattr(args, name)]
    if missing:
        flags = ", ".join(f"--{name}" for name in missing)
        raise ValueError(f"Task 12 backfill requires {flags}")
    try:
        from dotenv import load_dotenv

        load_dotenv(_PROJECT_ROOT / ".env", override=False)
    except ImportError:
        pass
    from .adapters.data_cache import RawDataCache
    from .adapters.tushare_data import TushareAShareDataAdapter
    from .adapters.us_market_data import YahooUSDataAdapter
    from .backfill import run_backfill
    from .features import AShareFeatureStrategy, USFeatureStrategy

    market = Market(args.market)
    cache_root = Path(args.cache_root or _PROJECT_ROOT / "harness_data/market_warning/raw")
    cache = RawDataCache(cache_root)
    next_session, previous_session, calendar_version = _calendar_resolvers(market)
    if market == Market.A_SHARE:
        from tradingagents.dataflows.tushare_vendor import _get_tushare_api

        requests_per_minute = float(
            os.getenv("TUSHARE_BACKFILL_REQUESTS_PER_MINUTE", "450")
        )
        if not 1 <= requests_per_minute <= 500:
            raise ValueError(
                "TUSHARE_BACKFILL_REQUESTS_PER_MINUTE must be between 1 and 500"
            )
        adapter = TushareAShareDataAdapter(
            pro=_get_tushare_api(),
            cache=cache,
            next_trading_day=next_session,
            previous_session=previous_session,
            calendar_version=calendar_version,
            minimum_request_interval=60.0 / requests_per_minute,
            max_fetch_attempts=3,
        )
        strategy = AShareFeatureStrategy()
    else:
        adapter = YahooUSDataAdapter(
            cache=cache,
            previous_session=previous_session,
            calendar_version=calendar_version,
        )
        strategy = USFeatureStrategy()
    return run_backfill(
        market,
        args.start,
        args.test_end,
        adapter,
        strategy,
        Path(args.output),
    )


def _load_cli_dataset(path_value: str | None, start: str, test_end: str) -> pd.DataFrame:
    if not path_value:
        raise ValueError("Task 12 input is not ready: provide --dataset with a validated point-in-time joblib DataFrame")
    path = Path(path_value)
    if not path.is_file():
        raise FileNotFoundError(f"dataset does not exist: {path}")
    frame = joblib.load(path)
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("--dataset must contain a pandas DataFrame")
    times = pd.to_datetime(_as_of_times(frame), utc=True)
    start_time = pd.Timestamp(start, tz="UTC")
    end_time = pd.Timestamp(test_end, tz="UTC") + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
    selected = frame.loc[times.between(start_time, end_time, inclusive="both")].copy()
    selected.attrs.update(frame.attrs)
    if selected.empty:
        raise ValueError("--dataset contains no rows in the requested frozen window")
    return selected


def _run_train_command(args: argparse.Namespace) -> dict[str, Any]:
    dataset = _load_cli_dataset(args.dataset, args.start, args.test_end)
    missing = [
        name
        for name in ("market", "horizon", "artifact_root", "db")
        if not getattr(args, name)
    ]
    if missing:
        flags = ", ".join(f"--{name.replace('_', '-')}" for name in missing)
        raise ValueError(f"local training also requires {flags}")
    dev, validation, test = time_partitions(dataset)
    bundle = fit_model(
        dev,
        validation,
        Market(args.market),
        args.horizon,
        model_version=args.version,
    )
    report = evaluate_model(bundle, test)
    from .adapters.sqlite_repository import SQLiteWarningRepository
    from .probability import save_model_bundle

    record = save_model_bundle(
        bundle,
        Path(args.artifact_root),
        SQLiteWarningRepository(Path(args.db)),
        active=False,
    )
    return {
        "artifact_path": record["artifact_path"],
        "artifact_sha256": record["artifact_sha256"],
        "market": bundle.market.value,
        "horizon": bundle.horizon,
        "model_version": bundle.model_version,
        "evaluation": _evaluation_payload(report),
        "active": False,
    }


def _run_train_all_command(args: argparse.Namespace) -> dict[str, Any]:
    from .adapters.sqlite_repository import SQLiteWarningRepository
    from .backfill import promotion_failures
    from .probability import save_model_bundle

    dataset_paths = {
        Market.A_SHARE: Path(args.dataset_a_share or _DEFAULT_DATASETS[Market.A_SHARE]),
        Market.US: Path(args.dataset_us or _DEFAULT_DATASETS[Market.US]),
    }
    missing = [market.value for market, path in dataset_paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Task 12 input is not ready: missing --dataset-a-share/--dataset-us frozen "
            f"point-in-time datasets for {', '.join(missing)}"
        )
    artifact_root = Path(args.artifact_root or _DEFAULT_ARTIFACT_ROOT)
    repository = SQLiteWarningRepository(args.db)
    models: list[dict[str, Any]] = []
    for market in Market:
        dataset = _load_cli_dataset(
            str(dataset_paths[market]),
            args.start,
            args.test_end,
        )
        if str(dataset.attrs.get("label_market", market.value)) != market.value:
            raise ValueError(f"dataset market marker does not match {market.value}")
        dev, validation, test = time_partitions(dataset)
        partitions = _partition_payload(dev, validation, test)
        production_train, production_calibration = production_partitions(dataset, market)
        production_windows = _production_partition_payload(
            production_train,
            production_calibration,
        )
        for horizon in ("1d", "3d"):
            evaluation_bundle = fit_model(
                dev,
                validation,
                market,
                horizon,
                model_version=args.version,
            )
            evaluation = _evaluation_payload(evaluate_model(evaluation_bundle, test))
            production_bundle = fit_model(
                production_train,
                production_calibration,
                market,
                horizon,
                model_version=args.version,
            )
            record = save_model_bundle(
                production_bundle,
                artifact_root,
                repository,
                active=False,
            )
            record = dict(record)
            evaluation_windows = _bundle_window_payload(evaluation_bundle)
            record["metrics"] = {
                **dict(record["metrics"]),
                "evaluation": evaluation,
                "evaluation_bundle": evaluation_windows,
            }
            repository.register_model(record)
            models.append(
                {
                    **_model_record_payload(record),
                    "dataset": str(dataset_paths[market]),
                    "partitions": partitions,
                    "production_partitions": production_windows,
                    "evaluation_bundle": evaluation_windows,
                    "evaluation": evaluation,
                }
            )

    reports = [model["evaluation"] for model in models]
    failures = promotion_failures(reports)
    manifest_path = Path(
        args.manifest or artifact_root / args.version / "evaluation-manifest.json"
    )
    report_path = Path(args.report or _DEFAULT_EVALUATION_REPORT)
    manifest = {
        "model_version": args.version,
        "feature_version": FEATURE_VERSION,
        "frozen_start": args.start,
        "frozen_test_end": args.test_end,
        "eligible_for_promotion": not failures,
        "promotion_failures": list(failures),
        "models": models,
    }
    _atomic_write_text(
        manifest_path,
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
    )
    _atomic_write_text(report_path, _render_evaluation_report(manifest))
    return {
        "model_version": args.version,
        "models": models,
        "eligible_for_promotion": not failures,
        "promotion_failures": list(failures),
        "manifest": str(manifest_path),
        "report": str(report_path),
    }


def _run_promote_command(args: argparse.Namespace) -> dict[str, Any]:
    from .adapters.sqlite_repository import SQLiteWarningRepository
    from .backfill import promotion_failures

    artifact_root = Path(args.artifact_root or _DEFAULT_ARTIFACT_ROOT).resolve()
    manifest_path = Path(
        args.manifest or artifact_root / args.version / "evaluation-manifest.json"
    )
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Task 12 input is not ready: evaluation manifest does not exist: {manifest_path}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, Mapping) or manifest.get("model_version") != args.version:
        raise ValueError("evaluation manifest model_version does not match --version")
    models = manifest.get("models")
    if not isinstance(models, list):
        raise ValueError("evaluation manifest models must be a list")
    reports = [model.get("evaluation", {}) for model in models if isinstance(model, Mapping)]
    failures = promotion_failures(reports)
    if failures:
        raise ValueError("promotion gates failed: " + "; ".join(failures))

    expected = {(market.value, horizon) for market in Market for horizon in ("1d", "3d")}
    indexed = {
        (str(model.get("market")), str(model.get("horizon"))): model
        for model in models
        if isinstance(model, Mapping)
    }
    if set(indexed) != expected or len(models) != len(expected):
        raise ValueError("evaluation manifest must contain one complete four-model set")
    repository = SQLiteWarningRepository(args.db)
    registered = {
        (record["market"].value, str(record["horizon"])): record
        for record in repository.load_model_set(args.version)
    }
    if set(registered) != expected:
        raise ValueError("registry must contain one complete four-model set")
    version_root = (artifact_root / args.version).resolve()
    for key in sorted(expected):
        model = indexed[key]
        record = registered[key]
        artifact = Path(str(model.get("artifact_path", ""))).resolve()
        if not artifact.is_file() or not artifact.is_relative_to(version_root):
            raise ValueError(f"artifact is missing or outside the version root: {key[0]}/{key[1]}")
        checksum = _file_sha256(artifact)
        expected_checksum = str(model.get("artifact_sha256", ""))
        if checksum != expected_checksum or checksum != str(record["artifact_sha256"]):
            raise ValueError(f"artifact checksum mismatch: {key[0]}/{key[1]}")
        if Path(str(record["artifact_path"])).resolve() != artifact:
            raise ValueError(f"registry artifact path mismatch: {key[0]}/{key[1]}")
        if str(record["feature_version"]) != str(model.get("feature_version")):
            raise ValueError(f"registry feature version mismatch: {key[0]}/{key[1]}")
        if str(record["calibration_version"]) != str(model.get("calibration_version")):
            raise ValueError(f"registry calibration version mismatch: {key[0]}/{key[1]}")

    activated = repository.activate_model_set(args.version)
    return {
        "model_version": args.version,
        "activated_models": len(activated),
        "active": True,
        "manifest": str(manifest_path),
    }


def _run_evaluate_command(args: argparse.Namespace) -> dict[str, Any]:
    if not args.artifact:
        raise ValueError("evaluate requires --artifact from a completed local train command")
    dataset = _load_cli_dataset(args.dataset, args.start, args.test_end)
    artifact = Path(args.artifact)
    if not artifact.is_file():
        raise FileNotFoundError(f"artifact does not exist: {artifact}")
    bundle = joblib.load(artifact)
    if not isinstance(bundle, ModelBundle):
        raise TypeError("--artifact must contain a ModelBundle")
    if bundle.model_version != args.version:
        raise ValueError("--version does not match the model artifact")
    _, _, test = time_partitions(dataset)
    return _evaluation_payload(evaluate_model(bundle, test))


def _evaluation_payload(report: EvaluationReport) -> dict[str, Any]:
    return {
        "market": report.market.value,
        "horizon": report.horizon,
        "observations": report.observations,
        "prevalence": report.prevalence,
        "brier_score": report.brier_score,
        "average_precision": report.average_precision,
        "constant_base_rate_brier": report.constant_base_rate_brier,
        "expected_calibration_error": report.expected_calibration_error,
        "monthly_alert_entries": report.monthly_alert_entries,
        "old_market_risk_recall": report.old_market_risk_recall,
        "model_recall_at_old_budget": report.model_recall_at_old_budget,
        "calibration_bins": [
            {
                "lower_bound": item.lower_bound,
                "upper_bound": item.upper_bound,
                "count": item.count,
                "mean_probability": item.mean_probability,
                "observed_rate": item.observed_rate,
            }
            for item in report.calibration_bins
        ],
        "phase_breakdown": {
            name: {
                "observations": metrics.observations,
                "positives": metrics.positives,
                "alerts": metrics.alerts,
                "true_positives": metrics.true_positives,
                "recall": metrics.recall,
            }
            for name, metrics in report.phase_breakdown.items()
        },
        "crisis_contribution": dict(report.crisis_contribution),
    }


def _partition_payload(
    dev: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
) -> dict[str, Any]:
    result: dict[str, Any] = {"embargo_trading_days": EMBARGO_TRADING_DAYS}
    for name, frame in (("dev", dev), ("validation", validation), ("test", test)):
        times = pd.to_datetime(_as_of_times(frame), utc=True)
        result[name] = {
            "rows": len(frame),
            "start": times.min().isoformat() if len(times) else None,
            "end": times.max().isoformat() if len(times) else None,
        }
    return result


def _production_partition_payload(
    training: pd.DataFrame,
    calibration: pd.DataFrame,
) -> dict[str, Any]:
    result: dict[str, Any] = {"embargo_trading_days": EMBARGO_TRADING_DAYS}
    for name, frame in (("training", training), ("calibration", calibration)):
        times = pd.to_datetime(_as_of_times(frame), utc=True)
        result[name] = {
            "rows": len(frame),
            "start": times.min().isoformat(),
            "end": times.max().isoformat(),
        }
    return result


def _bundle_window_payload(bundle: ModelBundle) -> dict[str, str]:
    return {
        "training_start": bundle.training_start.isoformat(),
        "training_end": bundle.training_end.isoformat(),
        "calibration_start": bundle.calibration_start.isoformat(),
        "calibration_end": bundle.calibration_end.isoformat(),
    }


def _model_record_payload(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "model_version": str(record["model_version"]),
        "market": Market(record["market"]).value,
        "horizon": str(record["horizon"]),
        "feature_version": str(record["feature_version"]),
        "calibration_version": str(record["calibration_version"]),
        "training_cutoff": str(record["training_cutoff"]),
        "artifact_path": str(record["artifact_path"]),
        "artifact_sha256": str(record["artifact_sha256"]),
        "metrics": dict(record["metrics"]),
        "base_rate": float(record["base_rate"]),
        "active": bool(record.get("active", False)),
    }


def _render_evaluation_report(manifest: Mapping[str, Any]) -> str:
    models = tuple(manifest.get("models", ()))
    lines = [
        f"# 大盘骤跌预警模型评估：{manifest['model_version']}",
        "",
        "## 冻结窗口与结论",
        "",
        f"- 数据区间：{manifest['frozen_start']} 至 {manifest['frozen_test_end']}",
        f"- 特征版本：`{manifest['feature_version']}`",
        "- 切分：开发集 2000-2012、校准集 2013-2019、冻结测试集 2020-2026-07-31；边界保留 3 个交易日隔离。",
        f"- 晋级状态：**{'通过全部门槛' if manifest['eligible_for_promotion'] else '未通过，保持未激活'}**",
        "",
        "## 四模型汇总",
        "",
        "| 市场 | 周期 | Brier | 常数基线 | AUPRC | 流行率 | ECE | 月均升级次数 | 生产校准截止 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for model in models:
        evaluation = model["evaluation"]
        lines.append(
            "| {market} | {horizon} | {brier:.4f} | {constant:.4f} | {auprc:.4f} | "
            "{prevalence:.4f} | {ece:.4f} | {budget:.2f} | {cutoff} |".format(
                market=model["market"],
                horizon=model["horizon"],
                brier=evaluation["brier_score"],
                constant=evaluation["constant_base_rate_brier"],
                auprc=evaluation["average_precision"],
                prevalence=evaluation["prevalence"],
                ece=evaluation["expected_calibration_error"],
                budget=evaluation["monthly_alert_entries"],
                cutoff=model["training_cutoff"],
            )
        )
    if manifest.get("promotion_failures"):
        lines.extend(["", "## 未通过门槛", ""])
        lines.extend(f"- {failure}" for failure in manifest["promotion_failures"])
    lines.extend(["", "## 分阶段、校准与危机贡献", ""])
    for model in models:
        evaluation = model["evaluation"]
        lines.extend(
            [
                f"### {model['market']} / {model['horizon']}",
                "",
                "| 阶段 | 样本 | 正例 | 告警 | 命中 | 召回率 |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for phase in ("FIRST_SHOCK", "CONTINUATION"):
            metrics = evaluation["phase_breakdown"][phase]
            recall = "N/A" if metrics["recall"] is None else f"{metrics['recall']:.3f}"
            lines.append(
                f"| {phase} | {metrics['observations']} | {metrics['positives']} | "
                f"{metrics['alerts']} | {metrics['true_positives']} | {recall} |"
            )
        crisis = ", ".join(
            f"{name}={value:.1%}"
            for name, value in sorted(evaluation["crisis_contribution"].items())
        )
        nonempty_bins = sum(1 for item in evaluation["calibration_bins"] if item["count"])
        lines.extend(
            [
                "",
                f"- 危机贡献：{crisis}",
                f"- 校准分箱：10 组，其中 {nonempty_bins} 组有测试样本。",
                f"- 旧系统召回率：{evaluation['old_market_risk_recall']}",
                f"- 相同旧系统预算下模型召回率：{evaluation['model_recall_at_old_budget']}",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(content, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
