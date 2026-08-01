"""Immutable model artifacts and fail-closed runtime probability inference."""

from __future__ import annotations

import hashlib
import hmac
import os
import tempfile
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol

import joblib

from .domain import DataStatus, FeatureSnapshot, Market, MarketPhase, QuantRiskAssessment
from .training import ModelBundle


class ModelRegistry(Protocol):
    def register_model(self, record: Mapping[str, Any]) -> None: ...

    def load_active_model(self, market: Market, horizon: str) -> dict[str, Any] | None: ...


def save_model_bundle(
    bundle: ModelBundle,
    artifact_root: Path | str,
    repository: ModelRegistry,
    *,
    active: bool = False,
) -> dict[str, Any]:
    """Persist one immutable joblib artifact and register its checksum metadata."""

    root = Path(artifact_root).resolve()
    target_dir = root / bundle.model_version
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"{bundle.market.value}-{bundle.horizon}.joblib"
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target_dir)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        joblib.dump(bundle, temporary)
        checksum = _sha256(temporary)
        if target.exists():
            if not hmac.compare_digest(_sha256(target), checksum):
                raise FileExistsError(f"immutable model artifact already exists: {target}")
        else:
            os.replace(temporary, target)
        record = {
            "model_version": bundle.model_version,
            "market": bundle.market,
            "horizon": bundle.horizon,
            "feature_version": bundle.feature_version,
            "calibration_version": bundle.calibration_version,
            "training_cutoff": bundle.calibration_end.date().isoformat(),
            "artifact_path": str(target),
            "artifact_sha256": checksum,
            "metrics": {
                "calibration_method": bundle.calibration_method,
                "training_start": bundle.training_start.isoformat(),
                "training_end": bundle.training_end.isoformat(),
                "calibration_start": bundle.calibration_start.isoformat(),
                "calibration_end": bundle.calibration_end.isoformat(),
            },
            "base_rate": bundle.base_rate,
            "active": active,
        }
        repository.register_model(record)
        return record
    finally:
        temporary.unlink(missing_ok=True)


class SklearnProbabilityModel:
    """Load both active horizon artifacts and infer without training on demand."""

    def __init__(
        self,
        repository: ModelRegistry,
        artifact_root: Path | str,
        *,
        max_model_age: timedelta = timedelta(days=550),
        now: Callable[[], datetime] | None = None,
    ) -> None:
        self._repository = repository
        self._artifact_root = Path(artifact_root).resolve()
        self._max_model_age = max_model_age
        self._now = now or (lambda: datetime.now(timezone.utc))

    def predict(self, snapshot: FeatureSnapshot) -> QuantRiskAssessment:
        if snapshot.data_quality in {DataStatus.CONFLICTED, DataStatus.STALE, DataStatus.INSUFFICIENT}:
            return _unavailable(snapshot, "snapshot_data_unusable")
        try:
            one_day = self._load_bundle(snapshot, "1d")
            three_day = self._load_bundle(snapshot, "3d")
            if one_day.model_version != three_day.model_version:
                return _unavailable(snapshot, "horizon_model_version_mismatch")
            if one_day.calibration_version != three_day.calibration_version:
                return _unavailable(snapshot, "horizon_calibration_version_mismatch")
            probability_1d = float(one_day.predict_proba(snapshot.features)[0])
            probability_3d = float(three_day.predict_proba(snapshot.features)[0])
            contributors = _contributors(snapshot, one_day, three_day)
            phase = _phase(snapshot)
            return QuantRiskAssessment(
                crash_1d_probability=probability_1d,
                crash_3d_probability=probability_3d,
                market_phase=phase,
                base_rate_1d=one_day.base_rate,
                base_rate_3d=three_day.base_rate,
                reliability_grade=snapshot.reliability_grade,
                model_version=one_day.model_version,
                calibration_version=one_day.calibration_version,
                top_contributors=contributors,
            )
        except (FileNotFoundError, OSError, ValueError, TypeError, EOFError):
            return _unavailable(snapshot, "model_artifact_unavailable")

    def _load_bundle(self, snapshot: FeatureSnapshot, horizon: str) -> ModelBundle:
        record = self._repository.load_active_model(snapshot.market, horizon)
        if record is None:
            raise FileNotFoundError(f"no active {horizon} model")
        if Market(record["market"]) != snapshot.market or str(record["horizon"]) != horizon:
            raise ValueError("registry market or horizon mismatch")
        if str(record["feature_version"]) != snapshot.feature_version:
            raise ValueError("registry feature version mismatch")
        cutoff = _date_value(record["training_cutoff"])
        now = self._now()
        if now.tzinfo is None or now.utcoffset() is None:
            raise ValueError("model clock must be timezone-aware")
        if cutoff > snapshot.as_of_time.date():
            raise ValueError("model training cutoff is after prediction time")
        if now.date() - cutoff > self._max_model_age:
            raise ValueError("model registry entry is stale")
        artifact = Path(str(record["artifact_path"])).resolve()
        if not artifact.is_relative_to(self._artifact_root):
            raise ValueError("artifact path escapes configured root")
        expected_checksum = str(record["artifact_sha256"])
        if len(expected_checksum) != 64 or not hmac.compare_digest(_sha256(artifact), expected_checksum):
            raise ValueError("artifact checksum mismatch")
        bundle = joblib.load(artifact)
        if not isinstance(bundle, ModelBundle):
            raise TypeError("artifact is not a ModelBundle")
        expected = (
            bundle.market == snapshot.market,
            bundle.horizon == horizon,
            bundle.feature_version == snapshot.feature_version,
            bundle.model_version == str(record["model_version"]),
            bundle.calibration_version == str(record["calibration_version"]),
            abs(bundle.base_rate - float(record["base_rate"])) <= 1e-12,
        )
        if not all(expected):
            raise ValueError("artifact metadata mismatch")
        return bundle


def _contributors(
    snapshot: FeatureSnapshot, one_day: ModelBundle, three_day: ModelBundle
) -> tuple[dict[str, Any], ...]:
    evidence_by_feature = {}
    for name in one_day.feature_names:
        marker = f":{name}:"
        evidence_by_feature[name] = next(
            (item.evidence_id for item in snapshot.evidence if marker in item.evidence_id),
            None,
        )
    rows = []
    for horizon, bundle in (("1d", one_day), ("3d", three_day)):
        for item in bundle.contributions(snapshot.features)[:3]:
            base_name = str(item["feature"]).removesuffix("__missing")
            rows.append(
                {
                    "horizon": horizon,
                    "feature": item["feature"],
                    "contribution": item["contribution"],
                    "evidence_id": evidence_by_feature.get(base_name),
                }
            )
    rows.sort(key=lambda item: (-abs(item["contribution"]), item["horizon"], item["feature"]))
    return tuple(rows)


def _phase(snapshot: FeatureSnapshot) -> MarketPhase:
    value = snapshot.features.get("market_phase")
    try:
        return MarketPhase(value)
    except (TypeError, ValueError):
        return MarketPhase.CONTINUATION


def _unavailable(snapshot: FeatureSnapshot, reason: str) -> QuantRiskAssessment:
    return QuantRiskAssessment(
        crash_1d_probability=0.0,
        crash_3d_probability=0.0,
        market_phase=_phase(snapshot),
        base_rate_1d=0.0,
        base_rate_3d=0.0,
        reliability_grade="UNAVAILABLE",
        model_version="unavailable",
        calibration_version="unavailable",
        top_contributors=({"unavailable_reason": reason},),
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _date_value(value: Any) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value)[:10])
