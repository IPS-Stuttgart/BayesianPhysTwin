"""Finite-group calibration for vector-valued physical query covariances.

Each independent object or acquisition session contributes exactly one maximum
Mahalanobis nonconformity score.  The resulting content-addressed artifact can
inflate future query covariances without treating frames or endpoints as
independent calibration samples.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ._canonical_contracts import (
    canonical_string_tuple,
    genuine_integer,
    literal_lower_hex,
)
from .calibration import plan_finite_group_calibration

QUERY_CALIBRATION_SCHEMA = "bayesian_phystwin.query_calibration"
QUERY_CALIBRATION_VERSION = 1
QUERY_CALIBRATION_SCORE = "group_max_mahalanobis"


def _finite_real(
    value: object,
    *,
    name: str,
    positive: bool = False,
    nonnegative: bool = False,
) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a finite real number")
    raw = np.asarray(value)
    if raw.shape != () or raw.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be a finite real number")
    scalar = float(raw.item())
    if not np.isfinite(scalar):
        raise ValueError(f"{name} must be a finite real number")
    if positive and scalar <= 0.0:
        raise ValueError(f"{name} must be positive")
    if nonnegative and scalar < 0.0:
        raise ValueError(f"{name} must be nonnegative")
    return scalar


def _sha256(value: object, *, name: str) -> str:
    return literal_lower_hex(value, name=name, lengths={64})


def _canonical_json_bytes(values: Mapping[str, Any]) -> bytes:
    return json.dumps(
        dict(values),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _content_id(values: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json_bytes(values)).hexdigest()


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _query_arrays(
    residual: np.ndarray,
    covariance: np.ndarray,
    *,
    name: str,
) -> tuple[np.ndarray, np.ndarray]:
    raw_residual = np.asarray(residual)
    raw_covariance = np.asarray(covariance)
    if raw_residual.dtype.kind not in "iuf":
        raise ValueError(f"{name} residual must contain real numeric values")
    if raw_covariance.dtype.kind not in "iuf":
        raise ValueError(f"{name} covariance must contain real numeric values")

    residual_array = np.asarray(raw_residual, dtype=np.float64)
    covariance_array = np.asarray(raw_covariance, dtype=np.float64)
    if residual_array.ndim == 1:
        residual_array = residual_array[None, :]
    elif residual_array.ndim != 2:
        raise ValueError(f"{name} residual must have shape (d,) or (m, d)")
    if covariance_array.ndim == 2:
        covariance_array = covariance_array[None, :, :]
    elif covariance_array.ndim != 3:
        raise ValueError(
            f"{name} covariance must have shape (d, d) or (m, d, d)"
        )
    if residual_array.shape[0] == 0 or residual_array.shape[1] == 0:
        raise ValueError(f"{name} cannot be empty")
    expected_shape = (
        residual_array.shape[0],
        residual_array.shape[1],
        residual_array.shape[1],
    )
    if covariance_array.shape != expected_shape:
        raise ValueError(
            f"{name} covariance shape must match the residual endpoint dimensions"
        )
    if not np.all(np.isfinite(residual_array)):
        raise ValueError(f"{name} residual must be finite")
    if not np.all(np.isfinite(covariance_array)):
        raise ValueError(f"{name} covariance must be finite")
    return residual_array, covariance_array


def _transformed_covariance(
    covariance: np.ndarray,
    *,
    covariance_scale: float,
    isotropic_variance: float,
    name: str,
) -> tuple[np.ndarray, np.ndarray]:
    dimension = covariance.shape[0]
    transformed = covariance_scale * covariance + isotropic_variance * np.eye(
        dimension,
        dtype=np.float64,
    )
    if not np.allclose(transformed, transformed.T, rtol=1e-12, atol=1e-12):
        raise ValueError(f"{name} must be symmetric after the frozen transform")
    symmetric = 0.5 * (transformed + transformed.T)
    try:
        cholesky = np.linalg.cholesky(symmetric)
    except np.linalg.LinAlgError as error:
        raise ValueError(
            f"{name} must be positive definite after the frozen transform"
        ) from error
    return symmetric, cholesky


def group_mahalanobis_nonconformity(
    residual: np.ndarray,
    covariance: np.ndarray,
    *,
    covariance_scale: float = 1.0,
    isotropic_variance: float = 0.0,
) -> float:
    """Return one maximum Mahalanobis score for an independent group."""

    scale = _finite_real(
        covariance_scale,
        name="covariance_scale",
        positive=True,
    )
    nugget = _finite_real(
        isotropic_variance,
        name="isotropic_variance",
        nonnegative=True,
    )
    residual_array, covariance_array = _query_arrays(
        residual,
        covariance,
        name="query group",
    )
    maximum = 0.0
    for index in range(residual_array.shape[0]):
        _, cholesky = _transformed_covariance(
            covariance_array[index],
            covariance_scale=scale,
            isotropic_variance=nugget,
            name=f"query group covariance {index}",
        )
        whitened = np.linalg.solve(cholesky, residual_array[index])
        maximum = max(maximum, float(np.linalg.norm(whitened)))
    return maximum


@dataclass(frozen=True, slots=True)
class QueryCalibrationV1:
    """Content-addressed split-conformal calibration for physical queries."""

    predictor_id: str
    query_set_id: str
    grouping_rule_id: str
    guard_id: str
    calibration_evidence_id: str
    calibration_group_ids: tuple[str, ...]
    calibration_group_scores: np.ndarray
    nominal_coverage: float
    finite_sample_rank: int
    conformal_quantile: float
    covariance_scale: float
    isotropic_variance: float
    predictor_frozen_before_scores: bool
    calibration_outcomes_used_for_selection: bool

    def __post_init__(self) -> None:
        for name in (
            "predictor_id",
            "query_set_id",
            "grouping_rule_id",
            "guard_id",
            "calibration_evidence_id",
        ):
            object.__setattr__(self, name, _sha256(getattr(self, name), name=name))

        group_ids = canonical_string_tuple(
            self.calibration_group_ids,
            name="calibration_group_ids",
            allow_empty=False,
        )
        if len(set(group_ids)) != len(group_ids):
            raise ValueError("calibration_group_ids must be unique")

        raw_scores = np.asarray(self.calibration_group_scores)
        if raw_scores.dtype.kind not in "iuf" or raw_scores.ndim != 1:
            raise ValueError(
                "calibration_group_scores must be a one-dimensional real array"
            )
        scores = np.array(raw_scores, dtype=np.float64, copy=True, order="C")
        if scores.size != len(group_ids):
            raise ValueError(
                "calibration_group_scores must contain one score per group"
            )
        if not np.all(np.isfinite(scores)) or np.any(scores < 0.0):
            raise ValueError(
                "calibration_group_scores must be finite and nonnegative"
            )

        if type(self.predictor_frozen_before_scores) is not bool:
            raise ValueError("predictor_frozen_before_scores must be a boolean")
        if type(self.calibration_outcomes_used_for_selection) is not bool:
            raise ValueError(
                "calibration_outcomes_used_for_selection must be a boolean"
            )
        design = plan_finite_group_calibration(
            len(group_ids),
            self.nominal_coverage,
            pooling="pooled",
            predictor_frozen_before_scores=self.predictor_frozen_before_scores,
            calibration_outcomes_used_for_selection=(
                self.calibration_outcomes_used_for_selection
            ),
        )
        rank = genuine_integer(
            self.finite_sample_rank,
            name="finite_sample_rank",
            minimum=1,
        )
        if rank != design.finite_sample_rank:
            raise ValueError(
                "finite_sample_rank must equal the finite-group conformal rank"
            )

        pairs = sorted(zip(group_ids, scores.tolist(), strict=True))
        canonical_group_ids = tuple(group_id for group_id, _ in pairs)
        canonical_scores = np.asarray(
            [score for _, score in pairs],
            dtype=np.float64,
        )
        expected_quantile = float(
            np.partition(canonical_scores, rank - 1)[rank - 1]
        )
        quantile = _finite_real(
            self.conformal_quantile,
            name="conformal_quantile",
            nonnegative=True,
        )
        if quantile != expected_quantile:
            raise ValueError(
                "conformal_quantile must equal the declared finite-sample order "
                "statistic"
            )
        scale = _finite_real(
            self.covariance_scale,
            name="covariance_scale",
            positive=True,
        )
        nugget = _finite_real(
            self.isotropic_variance,
            name="isotropic_variance",
            nonnegative=True,
        )

        canonical_scores.setflags(write=False)
        object.__setattr__(self, "calibration_group_ids", canonical_group_ids)
        object.__setattr__(self, "calibration_group_scores", canonical_scores)
        object.__setattr__(self, "nominal_coverage", design.nominal_coverage)
        object.__setattr__(self, "finite_sample_rank", rank)
        object.__setattr__(self, "conformal_quantile", quantile)
        object.__setattr__(self, "covariance_scale", scale)
        object.__setattr__(self, "isotropic_variance", nugget)

    def descriptor(self) -> dict[str, Any]:
        return {
            "schema": QUERY_CALIBRATION_SCHEMA,
            "schema_version": QUERY_CALIBRATION_VERSION,
            "score": QUERY_CALIBRATION_SCORE,
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
            "conformal_quantile": self.conformal_quantile,
            "covariance_scale": self.covariance_scale,
            "isotropic_variance": self.isotropic_variance,
            "predictor_frozen_before_scores": (
                self.predictor_frozen_before_scores
            ),
            "calibration_outcomes_used_for_selection": (
                self.calibration_outcomes_used_for_selection
            ),
        }

    @property
    def artifact_id(self) -> str:
        return _content_id(self.descriptor())

    def as_dict(self) -> dict[str, Any]:
        result = self.descriptor()
        result["artifact_id"] = self.artifact_id
        return result

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> QueryCalibrationV1:
        if not isinstance(values, Mapping):
            raise ValueError("query calibration record must be a mapping")
        expected_keys = {
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
            "conformal_quantile",
            "covariance_scale",
            "isotropic_variance",
            "predictor_frozen_before_scores",
            "calibration_outcomes_used_for_selection",
        }
        if set(values) != expected_keys:
            raise ValueError("query calibration record has missing or unknown fields")
        if values["schema"] != QUERY_CALIBRATION_SCHEMA:
            raise ValueError("unsupported query calibration schema")
        schema_version = genuine_integer(
            values["schema_version"],
            name="schema_version",
            minimum=1,
        )
        if schema_version != QUERY_CALIBRATION_VERSION:
            raise ValueError("unsupported query calibration schema version")
        if values["score"] != QUERY_CALIBRATION_SCORE:
            raise ValueError("unsupported query calibration score")

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

        artifact = cls(
            predictor_id=values["predictor_id"],
            query_set_id=values["query_set_id"],
            grouping_rule_id=values["grouping_rule_id"],
            guard_id=values["guard_id"],
            calibration_evidence_id=values["calibration_evidence_id"],
            calibration_group_ids=tuple(group_ids),
            calibration_group_scores=np.asarray(group_scores),
            nominal_coverage=values["nominal_coverage"],
            finite_sample_rank=values["finite_sample_rank"],
            conformal_quantile=values["conformal_quantile"],
            covariance_scale=values["covariance_scale"],
            isotropic_variance=values["isotropic_variance"],
            predictor_frozen_before_scores=values[
                "predictor_frozen_before_scores"
            ],
            calibration_outcomes_used_for_selection=values[
                "calibration_outcomes_used_for_selection"
            ],
        )
        declared_id = _sha256(values["artifact_id"], name="artifact_id")
        if artifact.artifact_id != declared_id:
            raise ValueError("query calibration artifact_id does not match content")
        return artifact


def fit_query_calibration(
    calibration_group_ids: Sequence[str],
    residual_groups: Sequence[np.ndarray],
    covariance_groups: Sequence[np.ndarray],
    *,
    nominal_coverage: float,
    predictor_id: str,
    query_set_id: str,
    grouping_rule_id: str,
    guard_id: str,
    calibration_evidence_id: str,
    covariance_scale: float = 1.0,
    isotropic_variance: float = 0.0,
    predictor_frozen_before_scores: bool,
    calibration_outcomes_used_for_selection: bool,
) -> QueryCalibrationV1:
    """Fit a finite-group query calibration after validating information order.

    The finite-sample design is validated before any residual or covariance
    element is inspected, so an impossible requested coverage fails before
    calibration outcomes are consumed.
    """

    group_ids = canonical_string_tuple(
        calibration_group_ids,
        name="calibration_group_ids",
        allow_empty=False,
    )
    if len(set(group_ids)) != len(group_ids):
        raise ValueError("calibration_group_ids must be unique")
    for name, value in (
        ("predictor_id", predictor_id),
        ("query_set_id", query_set_id),
        ("grouping_rule_id", grouping_rule_id),
        ("guard_id", guard_id),
        ("calibration_evidence_id", calibration_evidence_id),
    ):
        _sha256(value, name=name)
    scale = _finite_real(
        covariance_scale,
        name="covariance_scale",
        positive=True,
    )
    nugget = _finite_real(
        isotropic_variance,
        name="isotropic_variance",
        nonnegative=True,
    )
    design = plan_finite_group_calibration(
        len(group_ids),
        nominal_coverage,
        pooling="pooled",
        predictor_frozen_before_scores=predictor_frozen_before_scores,
        calibration_outcomes_used_for_selection=(
            calibration_outcomes_used_for_selection
        ),
    )
    if len(residual_groups) != len(group_ids):
        raise ValueError("residual_groups must contain one entry per group")
    if len(covariance_groups) != len(group_ids):
        raise ValueError("covariance_groups must contain one entry per group")

    scores = np.asarray(
        [
            group_mahalanobis_nonconformity(
                residual,
                covariance,
                covariance_scale=scale,
                isotropic_variance=nugget,
            )
            for residual, covariance in zip(
                residual_groups,
                covariance_groups,
                strict=True,
            )
        ],
        dtype=np.float64,
    )
    quantile = float(
        np.partition(scores, design.finite_sample_rank - 1)[
            design.finite_sample_rank - 1
        ]
    )
    return QueryCalibrationV1(
        predictor_id=predictor_id,
        query_set_id=query_set_id,
        grouping_rule_id=grouping_rule_id,
        guard_id=guard_id,
        calibration_evidence_id=calibration_evidence_id,
        calibration_group_ids=group_ids,
        calibration_group_scores=scores,
        nominal_coverage=design.nominal_coverage,
        finite_sample_rank=design.finite_sample_rank,
        conformal_quantile=quantile,
        covariance_scale=scale,
        isotropic_variance=nugget,
        predictor_frozen_before_scores=predictor_frozen_before_scores,
        calibration_outcomes_used_for_selection=(
            calibration_outcomes_used_for_selection
        ),
    )


def calibrate_query_covariance(
    covariance: np.ndarray,
    calibration: QueryCalibrationV1,
) -> np.ndarray:
    """Apply the frozen affine transform and conformal inflation to covariance."""

    raw = np.asarray(covariance)
    if raw.dtype.kind not in "iuf":
        raise ValueError("covariance must contain real numeric values")
    covariance_array = np.asarray(raw, dtype=np.float64)
    if covariance_array.ndim < 2 or covariance_array.shape[-1] == 0:
        raise ValueError("covariance must contain one or more square matrices")
    if covariance_array.shape[-2] != covariance_array.shape[-1]:
        raise ValueError("covariance matrices must be square")
    if covariance_array.size == 0 or not np.all(np.isfinite(covariance_array)):
        raise ValueError("covariance must be nonempty and finite")

    dimension = covariance_array.shape[-1]
    flattened = covariance_array.reshape((-1, dimension, dimension))
    calibrated = np.empty_like(flattened)
    multiplier = calibration.conformal_quantile**2
    for index, matrix in enumerate(flattened):
        transformed, _ = _transformed_covariance(
            matrix,
            covariance_scale=calibration.covariance_scale,
            isotropic_variance=calibration.isotropic_variance,
            name=f"covariance {index}",
        )
        calibrated[index] = multiplier * transformed
    result = calibrated.reshape(covariance_array.shape)
    result.setflags(write=False)
    return result


def query_group_is_covered(
    residual: np.ndarray,
    covariance: np.ndarray,
    calibration: QueryCalibrationV1,
) -> bool:
    """Return whether every registered endpoint lies in its calibrated ellipsoid."""

    score = group_mahalanobis_nonconformity(
        residual,
        covariance,
        covariance_scale=calibration.covariance_scale,
        isotropic_variance=calibration.isotropic_variance,
    )
    return score <= calibration.conformal_quantile


def save_query_calibration(
    calibration: QueryCalibrationV1,
    path: str | Path,
) -> None:
    """Write a validated query calibration as deterministic JSON."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(
            calibration.as_dict(),
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def load_query_calibration(path: str | Path) -> QueryCalibrationV1:
    """Load and revalidate a content-addressed query calibration JSON file."""

    values = json.loads(
        Path(path).read_text(encoding="utf-8"),
        object_pairs_hook=_unique_json_object,
    )
    if not isinstance(values, Mapping):
        raise ValueError("query calibration file must contain one JSON object")
    return QueryCalibrationV1.from_dict(values)


__all__ = [
    "QUERY_CALIBRATION_SCHEMA",
    "QUERY_CALIBRATION_SCORE",
    "QUERY_CALIBRATION_VERSION",
    "QueryCalibrationV1",
    "calibrate_query_covariance",
    "fit_query_calibration",
    "group_mahalanobis_nonconformity",
    "load_query_calibration",
    "query_group_is_covered",
    "save_query_calibration",
]
