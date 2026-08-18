"""Finite-group calibration of predictive density superlevel sets.

Each independent physical object or acquisition session contributes one maximum
negative log-density score over its registered query endpoints.  The resulting
threshold calibrates a simultaneous density superlevel set without collapsing a
non-Gaussian predictive belief to a covariance matrix.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from numbers import Real
from pathlib import Path
from typing import Any, Final

import numpy as np

from .predictive_query_mixture import (
    SameMeanGaussianMixturePredictionV1,
    gaussian_mixture_negative_log_density,
)

QUERY_DENSITY_CALIBRATION_SCHEMA: Final = (
    "bayesian-phystwin-query-density-calibration-v1"
)
QUERY_DENSITY_CALIBRATION_VERSION: Final = 1
QUERY_DENSITY_SCORE: Final = "group-max-negative-log-density-v1"


def _canonical_string(value: object, *, name: str) -> str:
    if type(value) is not str or not value or value.strip() != value:
        raise ValueError(f"{name} must be a nonempty canonical string")
    if any(character in value for character in "\x00\r\n"):
        raise ValueError(f"{name} must be a single canonical line")
    return value


def _sha256(value: object, *, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _finite_real(
    value: object,
    *,
    name: str,
    minimum: float | None = None,
) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite real number")
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def _open_probability(value: object, *, name: str) -> float:
    result = _finite_real(value, name=name)
    if not 0.0 < result < 1.0:
        raise ValueError(f"{name} must lie strictly inside (0, 1)")
    return result


def _plain_json(value: object, *, name: str = "value") -> Any:
    if value is None or type(value) in (str, bool, int):
        return value
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (float, np.floating)):
        result = float(value)
        if not math.isfinite(result):
            raise ValueError(f"{name} must contain only finite JSON values")
        return result
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            canonical_key = _canonical_string(key, name=f"{name} key")
            if canonical_key in result:
                raise ValueError(f"{name} contains a duplicate key")
            result[canonical_key] = _plain_json(
                item,
                name=f"{name}.{canonical_key}",
            )
        return result
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [
            _plain_json(item, name=f"{name}[{index}]")
            for index, item in enumerate(value)
        ]
    raise ValueError(f"{name} must contain only finite JSON values")


def _content_id(values: Mapping[str, Any]) -> str:
    payload = json.dumps(
        _plain_json(values),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _immutable_float(value: object) -> np.ndarray:
    canonical = np.array(value, dtype=np.dtype("<f8"), copy=True, order="C")
    return np.frombuffer(
        canonical.tobytes(order="C"),
        dtype=np.dtype("<f8"),
    ).reshape(canonical.shape)


def _canonical_group_ids(value: object, *, count: int) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError("calibration_group_ids must be a sequence of strings")
    result = tuple(
        _canonical_string(item, name=f"calibration_group_ids[{index}]")
        for index, item in enumerate(tuple(value))
    )
    if len(result) != count:
        raise ValueError("calibration_group_ids length must match group scores")
    if len(set(result)) != len(result):
        raise ValueError("calibration_group_ids must be unique")
    return result


def _finite_sample_rank(group_count: int, nominal_coverage: float) -> int:
    rank = int(math.ceil((group_count + 1) * nominal_coverage))
    if rank > group_count:
        raise ValueError(
            "requested nominal coverage has no finite split-conformal rank for "
            f"{group_count} independent groups"
        )
    return rank


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def group_density_nonconformity(
    residual_m: object,
    prediction: SameMeanGaussianMixturePredictionV1,
) -> float:
    """Return one maximum negative log density for a physical group."""

    scores = gaussian_mixture_negative_log_density(residual_m, prediction)
    if scores.size == 0:
        raise ValueError("a calibration group must contain at least one endpoint")
    result = float(np.max(scores))
    if not math.isfinite(result):
        raise ValueError("group density nonconformity must be finite")
    return result


@dataclass(frozen=True, slots=True)
class QueryDensityCalibrationV1:
    """Content-addressed groupwise calibration of a fixed predictive density."""

    predictor_id: str
    query_set_id: str
    grouping_rule_id: str
    guard_id: str
    calibration_evidence_id: str
    calibration_group_ids: Sequence[str]
    calibration_group_scores: np.ndarray
    nominal_coverage: float
    finite_sample_rank: int
    density_score_threshold: float
    predictor_frozen_before_scores: bool
    calibration_outcomes_used_for_selection: bool
    calibration_groups_independent: bool
    metadata: Mapping[str, Any] = field(default_factory=dict)
    artifact_id: str | None = None

    def __post_init__(self) -> None:
        for name in (
            "predictor_id",
            "query_set_id",
            "grouping_rule_id",
            "guard_id",
            "calibration_evidence_id",
        ):
            object.__setattr__(self, name, _sha256(getattr(self, name), name=name))
        raw_scores = np.asarray(self.calibration_group_scores)
        if raw_scores.dtype.kind not in "iuf" or raw_scores.ndim != 1:
            raise ValueError(
                "calibration_group_scores must be a one-dimensional real array"
            )
        scores = np.array(raw_scores, dtype=np.float64, copy=True, order="C")
        if scores.size == 0 or not np.all(np.isfinite(scores)):
            raise ValueError("calibration_group_scores must be nonempty and finite")
        group_ids = _canonical_group_ids(
            self.calibration_group_ids,
            count=len(scores),
        )
        coverage = _open_probability(
            self.nominal_coverage,
            name="nominal_coverage",
        )
        expected_rank = _finite_sample_rank(len(scores), coverage)
        if isinstance(self.finite_sample_rank, bool) or not isinstance(
            self.finite_sample_rank,
            (int, np.integer),
        ):
            raise ValueError("finite_sample_rank must be an integer")
        rank = int(self.finite_sample_rank)
        if rank != expected_rank:
            raise ValueError(
                "finite_sample_rank must equal the finite-group conformal rank"
            )
        order = np.argsort(np.asarray(group_ids, dtype=object), kind="mergesort")
        group_ids = tuple(group_ids[int(index)] for index in order)
        scores = scores[order]
        expected_threshold = float(np.partition(scores, rank - 1)[rank - 1])
        threshold = _finite_real(
            self.density_score_threshold,
            name="density_score_threshold",
        )
        if threshold != expected_threshold:
            raise ValueError(
                "density_score_threshold must equal the declared order statistic"
            )
        if self.predictor_frozen_before_scores is not True:
            raise ValueError("the predictor must be frozen before calibration scores")
        if self.calibration_outcomes_used_for_selection is not False:
            raise ValueError(
                "calibration outcomes cannot select the calibrated predictor"
            )
        if self.calibration_groups_independent is not True:
            raise ValueError(
                "calibration groups must be independent physical objects or sessions"
            )
        object.__setattr__(self, "calibration_group_ids", group_ids)
        object.__setattr__(self, "calibration_group_scores", _immutable_float(scores))
        object.__setattr__(self, "nominal_coverage", coverage)
        object.__setattr__(self, "finite_sample_rank", rank)
        object.__setattr__(self, "density_score_threshold", threshold)
        object.__setattr__(
            self,
            "metadata",
            _plain_json(self.metadata, name="metadata"),
        )
        expected_id = _content_id(self.descriptor())
        if self.artifact_id is None:
            object.__setattr__(self, "artifact_id", expected_id)
        elif _sha256(self.artifact_id, name="artifact_id") != expected_id:
            raise ValueError("artifact_id does not match query density calibration")

    def descriptor(self) -> dict[str, Any]:
        return {
            "schema": QUERY_DENSITY_CALIBRATION_SCHEMA,
            "schema_version": QUERY_DENSITY_CALIBRATION_VERSION,
            "score": QUERY_DENSITY_SCORE,
            "predictor_id": self.predictor_id,
            "query_set_id": self.query_set_id,
            "grouping_rule_id": self.grouping_rule_id,
            "guard_id": self.guard_id,
            "calibration_evidence_id": self.calibration_evidence_id,
            "calibration_groups": [
                {"group_id": group_id, "score": float(score)}
                for group_id, score in zip(
                    self.calibration_group_ids,
                    self.calibration_group_scores,
                    strict=True,
                )
            ],
            "nominal_coverage": self.nominal_coverage,
            "finite_sample_rank": self.finite_sample_rank,
            "density_score_threshold": self.density_score_threshold,
            "predictor_frozen_before_scores": self.predictor_frozen_before_scores,
            "calibration_outcomes_used_for_selection": (
                self.calibration_outcomes_used_for_selection
            ),
            "calibration_groups_independent": self.calibration_groups_independent,
            "metadata": _plain_json(self.metadata, name="metadata"),
        }

    def as_dict(self) -> dict[str, Any]:
        result = self.descriptor()
        result["artifact_id"] = self.artifact_id
        return result

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> QueryDensityCalibrationV1:
        if not isinstance(values, Mapping):
            raise ValueError("query density calibration record must be a mapping")
        expected_fields = {
            "schema",
            "schema_version",
            "score",
            "artifact_id",
            "predictor_id",
            "query_set_id",
            "grouping_rule_id",
            "guard_id",
            "calibration_evidence_id",
            "calibration_groups",
            "nominal_coverage",
            "finite_sample_rank",
            "density_score_threshold",
            "predictor_frozen_before_scores",
            "calibration_outcomes_used_for_selection",
            "calibration_groups_independent",
            "metadata",
        }
        if set(values) != expected_fields:
            raise ValueError(
                "query density calibration record has missing or unknown fields"
            )
        if values["schema"] != QUERY_DENSITY_CALIBRATION_SCHEMA:
            raise ValueError("unsupported query density calibration schema")
        if values["schema_version"] != QUERY_DENSITY_CALIBRATION_VERSION:
            raise ValueError("unsupported query density calibration version")
        if values["score"] != QUERY_DENSITY_SCORE:
            raise ValueError("unsupported query density calibration score")
        raw_groups = values["calibration_groups"]
        if not isinstance(raw_groups, list) or not raw_groups:
            raise ValueError("calibration_groups must be a nonempty list")
        group_ids: list[str] = []
        group_scores: list[float] = []
        for group in raw_groups:
            if not isinstance(group, Mapping) or set(group) != {"group_id", "score"}:
                raise ValueError(
                    "each calibration group must contain group_id and score"
                )
            group_ids.append(group["group_id"])
            group_scores.append(group["score"])
        return cls(
            predictor_id=values["predictor_id"],
            query_set_id=values["query_set_id"],
            grouping_rule_id=values["grouping_rule_id"],
            guard_id=values["guard_id"],
            calibration_evidence_id=values["calibration_evidence_id"],
            calibration_group_ids=group_ids,
            calibration_group_scores=np.asarray(group_scores),
            nominal_coverage=values["nominal_coverage"],
            finite_sample_rank=values["finite_sample_rank"],
            density_score_threshold=values["density_score_threshold"],
            predictor_frozen_before_scores=values["predictor_frozen_before_scores"],
            calibration_outcomes_used_for_selection=values[
                "calibration_outcomes_used_for_selection"
            ],
            calibration_groups_independent=values["calibration_groups_independent"],
            metadata=values["metadata"],
            artifact_id=values["artifact_id"],
        )


def fit_query_density_calibration(
    *,
    calibration_group_ids: Sequence[str],
    residual_groups: Sequence[np.ndarray],
    prediction_groups: Sequence[SameMeanGaussianMixturePredictionV1],
    nominal_coverage: float,
    predictor_id: str,
    query_set_id: str,
    grouping_rule_id: str,
    guard_id: str,
    calibration_evidence_id: str,
    predictor_frozen_before_scores: bool = True,
    calibration_outcomes_used_for_selection: bool = False,
    calibration_groups_independent: bool = True,
    metadata: Mapping[str, Any] | None = None,
) -> QueryDensityCalibrationV1:
    """Fit one group-maximum density threshold on disjoint calibration units."""

    residual_values = tuple(residual_groups)
    prediction_values = tuple(prediction_groups)
    if not residual_values or len(residual_values) != len(prediction_values):
        raise ValueError(
            "residual_groups and prediction_groups must have equal nonzero length"
        )
    group_ids = _canonical_group_ids(
        calibration_group_ids,
        count=len(residual_values),
    )
    coverage = _open_probability(nominal_coverage, name="nominal_coverage")
    rank = _finite_sample_rank(len(group_ids), coverage)
    scores = np.asarray(
        [
            group_density_nonconformity(residual, prediction)
            for residual, prediction in zip(
                residual_values,
                prediction_values,
                strict=True,
            )
        ],
        dtype=np.float64,
    )
    threshold = float(np.partition(scores, rank - 1)[rank - 1])
    return QueryDensityCalibrationV1(
        predictor_id=predictor_id,
        query_set_id=query_set_id,
        grouping_rule_id=grouping_rule_id,
        guard_id=guard_id,
        calibration_evidence_id=calibration_evidence_id,
        calibration_group_ids=group_ids,
        calibration_group_scores=scores,
        nominal_coverage=coverage,
        finite_sample_rank=rank,
        density_score_threshold=threshold,
        predictor_frozen_before_scores=predictor_frozen_before_scores,
        calibration_outcomes_used_for_selection=(
            calibration_outcomes_used_for_selection
        ),
        calibration_groups_independent=calibration_groups_independent,
        metadata={} if metadata is None else metadata,
    )


def density_region_contains(
    residual_m: object,
    prediction: SameMeanGaussianMixturePredictionV1,
    calibration: QueryDensityCalibrationV1,
    *,
    predictor_id: str,
) -> np.ndarray:
    """Return endpoint-wise membership in the calibrated density superlevel set."""

    if not isinstance(calibration, QueryDensityCalibrationV1):
        raise TypeError("calibration must be a QueryDensityCalibrationV1")
    if _sha256(predictor_id, name="predictor_id") != calibration.predictor_id:
        raise ValueError("predictor_id does not match the calibration artifact")
    scores = gaussian_mixture_negative_log_density(residual_m, prediction)
    return scores <= calibration.density_score_threshold


def group_density_region_covered(
    residual_m: object,
    prediction: SameMeanGaussianMixturePredictionV1,
    calibration: QueryDensityCalibrationV1,
    *,
    predictor_id: str,
) -> bool:
    """Return whether every registered endpoint in one group is covered."""

    return bool(
        np.all(
            density_region_contains(
                residual_m,
                prediction,
                calibration,
                predictor_id=predictor_id,
            )
        )
    )


def save_query_density_calibration(
    calibration: QueryDensityCalibrationV1,
    path: str | Path,
) -> None:
    """Atomically publish one strict calibration artifact without replacement."""

    if not isinstance(calibration, QueryDensityCalibrationV1):
        raise TypeError("calibration must be a QueryDensityCalibrationV1")
    target = Path(path)
    if target.is_symlink():
        raise ValueError("query density calibration target cannot be a symlink")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if not target.is_file():
            raise ValueError("query density calibration target must be a file")
        existing = load_query_density_calibration(target)
        if existing.artifact_id != calibration.artifact_id:
            raise FileExistsError(
                "a different query density calibration already occupies the path"
            )
        return
    payload = json.dumps(
        calibration.as_dict(),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.",
        suffix=".tmp",
        dir=target.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, target)
        except FileExistsError as error:
            existing = load_query_density_calibration(target)
            if existing.artifact_id != calibration.artifact_id:
                raise FileExistsError(
                    "a different query density calibration was published concurrently"
                ) from error
        _fsync_directory(target.parent)
    finally:
        temporary.unlink(missing_ok=True)


def load_query_density_calibration(
    path: str | Path,
) -> QueryDensityCalibrationV1:
    """Load and revalidate a strict query-density calibration JSON artifact."""

    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise ValueError("query density calibration path must be a regular file")
    try:
        values = json.loads(
            source.read_text(encoding="utf-8"),
            object_pairs_hook=_unique_json_object,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"invalid JSON constant: {value}")
            ),
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("query density calibration JSON is unreadable") from error
    return QueryDensityCalibrationV1.from_dict(values)


__all__ = [
    "QUERY_DENSITY_CALIBRATION_SCHEMA",
    "QUERY_DENSITY_CALIBRATION_VERSION",
    "QueryDensityCalibrationV1",
    "density_region_contains",
    "fit_query_density_calibration",
    "group_density_nonconformity",
    "group_density_region_covered",
    "load_query_density_calibration",
    "save_query_density_calibration",
]
