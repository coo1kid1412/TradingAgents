"""Leakage-safe labels, chronological model fitting, and evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import exchange_calendars as xcals
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .domain import Market
from .features import FEATURE_VERSION


MODEL_VERSION = "market-warning-logistic-v1"
CALIBRATION_VERSION = "platt-v1"
EMBARGO_TRADING_DAYS = 3
TEST_END = pd.Timestamp("2026-07-31", tz="UTC")

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
    availability_columns = [column for column in frame if column.endswith("_available_at")]
    for column in availability_columns:
        available = pd.to_datetime(frame[column], utc=True, errors="coerce")
        comparable_times = pd.to_datetime(times, utc=True)
        invalid = available.notna() & (available > comparable_times)
        if invalid.any():
            raise ValueError(f"point-in-time violation: {column} is after as_of_time")

    future_1d = closes.shift(-1) / closes - 1.0
    future_returns = pd.concat(
        [closes.shift(-offset) / closes - 1.0 for offset in range(1, 4)], axis=1
    )
    future_worst_3d = future_returns.min(axis=1, skipna=False)
    epsilon = 1e-12
    frame["future_return_1d"] = future_1d
    frame["future_worst_return_3d"] = future_worst_3d
    frame["label_1d"] = future_1d.le(_LABEL_THRESHOLDS[market]["1d"] + epsilon).where(future_1d.notna()).astype("boolean")
    frame["label_3d"] = future_worst_3d.le(_LABEL_THRESHOLDS[market]["3d"] + epsilon).where(future_worst_3d.notna()).astype("boolean")
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
    has_label_end = "label_end_3d" in values
    label_ends = pd.to_datetime(values["label_end_3d"], utc=True, errors="coerce") if has_label_end else None
    for name, start, end in definitions:
        mask = times.between(start, end, inclusive="both")
        part = values.loc[mask].copy()
        if name != "test":
            if has_label_end:
                ends = label_ends.loc[part.index]
                part = part.loc[ends.notna() & ends.le(end)]
            elif len(part):
                part = part.iloc[:-EMBARGO_TRADING_DAYS] if len(part) > EMBARGO_TRADING_DAYS else part.iloc[0:0]
        part["partition"] = name
        part.attrs.update(frame.attrs)
        partitions.append(part.reset_index(drop=True))
    return tuple(partitions)  # type: ignore[return-value]


def fit_model(
    train: pd.DataFrame,
    calibration: pd.DataFrame,
    market: Market,
    horizon: str,
) -> ModelBundle:
    """Fit one deterministic logistic pipeline and a later sigmoid calibrator."""

    market = Market(market)
    target = _target_column(horizon)
    train_times = pd.to_datetime(_as_of_times(train), utc=True)
    calibration_times = pd.to_datetime(_as_of_times(calibration), utc=True)
    if train.empty or calibration.empty:
        raise ValueError("training and calibration windows must not be empty")
    if train_times.max() >= calibration_times.min():
        raise ValueError("calibration window must be later than the training window")
    if _embargo_sessions(market, train_times.max(), calibration_times.min()) < EMBARGO_TRADING_DAYS:
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
        model_version=MODEL_VERSION,
        calibration_version=CALIBRATION_VERSION,
        calibration_method="platt",
        base_rate=float(y_train.mean()),
        training_start=train_times.min().to_pydatetime(),
        training_end=train_times.max().to_pydatetime(),
        calibration_start=calibration_times.min().to_pydatetime(),
        calibration_end=calibration_times.max().to_pydatetime(),
        pipeline=pipeline,
        calibrator=calibrator,
    )


def evaluate_model(bundle: ModelBundle, test: pd.DataFrame) -> EvaluationReport:
    """Evaluate one frozen bundle without tuning it on the supplied rows."""

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
        constant_base_rate_brier=float(brier_score_loss(y_true, np.full(len(y_true), prevalence))),
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
    parsed = pd.to_datetime(values, errors="raise")
    if getattr(parsed.dt, "tz", None) is None:
        raise ValueError("as_of_time must be timezone-aware")
    return parsed


def _target_column(horizon: str) -> str:
    if horizon not in {"1d", "3d"}:
        raise ValueError("horizon must be '1d' or '3d'")
    return f"label_{horizon}"


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
