"""Regime-conditioned calibration of predictive covariance matrices.

The calibrator operates on independent object/session groups.  It selects a
regime-specific affine-plus-low-rank covariance transform by leave-one-group-out
Gaussian log score and then refits that transform on all source groups.  The
identity transform is mandatory, so the selected source cross-validation score
cannot be worse than the uncalibrated covariance merely because a more complex
candidate was available.

This module calibrates predictive covariance.  It does not decide whether a
candidate physical update is supported in a regime; that remains the
responsibility of a separate calibration-domain guard and exact-fallback policy.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ._canonical_contracts import (
    canonical_string_tuple,
    frozen_finite_json_mapping,
    genuine_boolean,
    genuine_integer,
    immutable_array,
    literal_lower_hex,
    plain_json,
)
from ._portable_contracts import (
    content_id,
    load_strict_json_object,
    require_exact_fields,
    write_atomic_json,
)
from .calibration_domain_guard import CalibrationDomainGuardCertificateV1
from .posterior_covariance_semantics import PosteriorCovarianceSemanticsV1

REGIME_COVARIANCE_CALIBRATION_SCHEMA = (
    "bayesian_phystwin.regime_covariance_calibration"
)
REGIME_COVARIANCE_CALIBRATION_VERSION = 1
REGIME_COVARIANCE_TRANSFORM_SCHEMA = (
    "bayesian_phystwin.regime_covariance_transform"
)
REGIME_COVARIANCE_TRANSFORM_VERSION = 1
REGIME_COVARIANCE_GROUP_SCHEMA = "bayesian_phystwin.covariance_calibration_group"
REGIME_COVARIANCE_GROUP_VERSION = 1

_DEFAULT_SCALES = (0.5, 1.0, 2.0, 4.0)
_DEFAULT_FLOOR_FRACTIONS = (0.0, 0.01, 0.05, 0.10)
_DEFAULT_SHRINKAGES = (0.0, 0.5, 1.0)
_LOG_2PI = float(np.log(2.0 * np.pi))


def _literal_string(value: object, *, name: str) -> str:
    if type(value) is not str or not value or value.strip() != value or "\x00" in value:
        raise ValueError(f"{name} must be a nonempty canonical literal string")
    return value


def _sha256(value: object, *, name: str) -> str:
    return literal_lower_hex(value, name=name, lengths={64})


def _finite_real(
    value: object,
    *,
    name: str,
    positive: bool = False,
    nonnegative: bool = False,
    maximum: float | None = None,
) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a finite real number")
    raw = np.asarray(value)
    if raw.shape != () or raw.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be a finite real number")
    result = float(raw.item())
    if not np.isfinite(result):
        raise ValueError(f"{name} must be a finite real number")
    if positive and result <= 0.0:
        raise ValueError(f"{name} must be positive")
    if nonnegative and result < 0.0:
        raise ValueError(f"{name} must be nonnegative")
    if maximum is not None and result > maximum:
        raise ValueError(f"{name} must not exceed {maximum}")
    return result


def _real_grid(
    values: Sequence[float],
    *,
    name: str,
    positive: bool = False,
    nonnegative: bool = False,
    maximum: float | None = None,
) -> tuple[float, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ValueError(f"{name} must be a sequence")
    result = tuple(
        _finite_real(
            value,
            name=f"{name} entry",
            positive=positive,
            nonnegative=nonnegative,
            maximum=maximum,
        )
        for value in values
    )
    if not result:
        raise ValueError(f"{name} must not be empty")
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must not contain duplicates")
    return tuple(sorted(result))


def _array_record(values: np.ndarray) -> dict[str, object]:
    array = np.asarray(values, dtype=np.dtype("<f8"), order="C")
    return {
        "dtype": "float64-le",
        "shape": list(array.shape),
        "sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
    }


def _validate_residual_covariance_arrays(
    residuals: object,
    covariances: object,
    *,
    name: str,
) -> tuple[np.ndarray, np.ndarray]:
    raw_residuals = np.asarray(residuals)
    raw_covariances = np.asarray(covariances)
    if raw_residuals.dtype.kind not in "iuf":
        raise ValueError(f"{name} residuals must contain real numeric values")
    if raw_covariances.dtype.kind not in "iuf":
        raise ValueError(f"{name} covariances must contain real numeric values")
    residual_array = np.asarray(raw_residuals, dtype=np.float64)
    covariance_array = np.asarray(raw_covariances, dtype=np.float64)
    if residual_array.ndim != 2 or 0 in residual_array.shape:
        raise ValueError(f"{name} residuals must have nonempty shape (m, d)")
    expected = (
        residual_array.shape[0],
        residual_array.shape[1],
        residual_array.shape[1],
    )
    if covariance_array.shape != expected:
        raise ValueError(
            f"{name} covariances must have shape (m, d, d) matching residuals"
        )
    if not np.all(np.isfinite(residual_array)):
        raise ValueError(f"{name} residuals must be finite")
    if not np.all(np.isfinite(covariance_array)):
        raise ValueError(f"{name} covariances must be finite")
    for index, covariance in enumerate(covariance_array):
        if not np.allclose(covariance, covariance.T, rtol=1e-11, atol=1e-12):
            raise ValueError(f"{name} covariance {index} must be symmetric")
        symmetric = 0.5 * (covariance + covariance.T)
        try:
            np.linalg.cholesky(symmetric)
        except np.linalg.LinAlgError as error:
            raise ValueError(
                f"{name} covariance {index} must be positive definite"
            ) from error
    return (
        immutable_array(residual_array, dtype=np.dtype("<f8")),
        immutable_array(covariance_array, dtype=np.dtype("<f8")),
    )


def _canonical_factor(values: object, *, dimension: int) -> np.ndarray:
    raw = np.asarray(values)
    if raw.dtype.kind not in "iuf":
        raise ValueError("additive_factor must contain real numeric values")
    factor = np.asarray(raw, dtype=np.float64)
    if factor.ndim != 2 or factor.shape[0] != dimension:
        raise ValueError("additive_factor must have shape (dimension, rank)")
    if not np.all(np.isfinite(factor)):
        raise ValueError("additive_factor must be finite")
    return immutable_array(factor, dtype=np.dtype("<f8"))


def _canonicalize_factor_signs(factor: np.ndarray) -> np.ndarray:
    canonical = np.array(factor, dtype=np.float64, copy=True, order="C")
    for column in range(canonical.shape[1]):
        vector = canonical[:, column]
        pivot = int(np.argmax(np.abs(vector)))
        if vector[pivot] < 0.0:
            canonical[:, column] *= -1.0
    return canonical


@dataclass(frozen=True, slots=True)
class CovarianceCalibrationGroupV1:
    """One independent source group for covariance calibration."""

    group_id: str
    regime_id: str
    residuals: np.ndarray
    covariances: np.ndarray
    metadata: Mapping[str, Any] = field(default_factory=dict)
    artifact_id: str | None = None

    def __post_init__(self) -> None:
        group_id = _literal_string(self.group_id, name="group_id")
        regime_id = _literal_string(self.regime_id, name="regime_id")
        residuals, covariances = _validate_residual_covariance_arrays(
            self.residuals,
            self.covariances,
            name=f"calibration group {group_id}",
        )
        metadata = frozen_finite_json_mapping(
            self.metadata,
            name="covariance calibration group metadata",
        )
        object.__setattr__(self, "group_id", group_id)
        object.__setattr__(self, "regime_id", regime_id)
        object.__setattr__(self, "residuals", residuals)
        object.__setattr__(self, "covariances", covariances)
        object.__setattr__(self, "metadata", metadata)
        expected_id = content_id(self.descriptor())
        if self.artifact_id is not None:
            supplied = _sha256(self.artifact_id, name="artifact_id")
            if supplied != expected_id:
                raise ValueError(
                    "covariance calibration group artifact_id does not match content"
                )
        object.__setattr__(self, "artifact_id", expected_id)

    @property
    def dimension(self) -> int:
        return int(self.residuals.shape[1])

    @property
    def row_count(self) -> int:
        return int(self.residuals.shape[0])

    def descriptor(self) -> dict[str, object]:
        return {
            "schema": REGIME_COVARIANCE_GROUP_SCHEMA,
            "schema_version": REGIME_COVARIANCE_GROUP_VERSION,
            "group_id": self.group_id,
            "regime_id": self.regime_id,
            "residuals": _array_record(self.residuals),
            "covariances": _array_record(self.covariances),
            "metadata": plain_json(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class RegimeCovarianceTransformV1:
    """One fitted source-only covariance transform for a declared regime."""

    regime_id: str
    dimension: int
    covariance_scale: float
    isotropic_variance: float
    additive_factor: np.ndarray
    selected_floor_fraction: float
    selected_shrinkage: float
    max_rank: int
    scale_grid: Sequence[float]
    floor_fraction_grid: Sequence[float]
    shrinkage_grid: Sequence[float]
    calibration_group_ids: Sequence[str]
    calibration_group_artifact_ids: Sequence[str]
    raw_loo_nll: float
    calibrated_loo_nll: float
    refit_nll: float
    raw_normalized_nees: float
    calibrated_refit_normalized_nees: float
    metadata: Mapping[str, Any] = field(default_factory=dict)
    artifact_id: str | None = None

    def __post_init__(self) -> None:
        regime_id = _literal_string(self.regime_id, name="regime_id")
        dimension = genuine_integer(self.dimension, name="dimension", minimum=1)
        scale = _finite_real(
            self.covariance_scale,
            name="covariance_scale",
            positive=True,
        )
        variance = _finite_real(
            self.isotropic_variance,
            name="isotropic_variance",
            nonnegative=True,
        )
        floor_fraction = _finite_real(
            self.selected_floor_fraction,
            name="selected_floor_fraction",
            nonnegative=True,
        )
        shrinkage = _finite_real(
            self.selected_shrinkage,
            name="selected_shrinkage",
            nonnegative=True,
            maximum=1.0,
        )
        max_rank = genuine_integer(self.max_rank, name="max_rank", minimum=0)
        if max_rank > dimension:
            raise ValueError("max_rank must not exceed dimension")
        factor = _canonical_factor(self.additive_factor, dimension=dimension)
        if factor.shape[1] > max_rank:
            raise ValueError("additive_factor rank must not exceed max_rank")
        scale_grid = _real_grid(self.scale_grid, name="scale_grid", positive=True)
        floor_grid = _real_grid(
            self.floor_fraction_grid,
            name="floor_fraction_grid",
            nonnegative=True,
        )
        shrinkage_grid = _real_grid(
            self.shrinkage_grid,
            name="shrinkage_grid",
            nonnegative=True,
            maximum=1.0,
        )
        if scale not in scale_grid:
            raise ValueError("covariance_scale must belong to scale_grid")
        if floor_fraction not in floor_grid:
            raise ValueError(
                "selected_floor_fraction must belong to floor_fraction_grid"
            )
        if shrinkage not in shrinkage_grid:
            raise ValueError("selected_shrinkage must belong to shrinkage_grid")
        group_ids = canonical_string_tuple(
            self.calibration_group_ids,
            name="calibration_group_ids",
            allow_empty=False,
        )
        artifact_ids = tuple(
            _sha256(value, name="calibration_group_artifact_id")
            for value in self.calibration_group_artifact_ids
        )
        if len(group_ids) != len(artifact_ids):
            raise ValueError(
                "calibration group IDs and artifact IDs must have equal length"
            )
        if len(set(group_ids)) != len(group_ids):
            raise ValueError("calibration_group_ids must be unique")
        pairs = sorted(zip(group_ids, artifact_ids, strict=True))
        canonical_group_ids = tuple(group_id for group_id, _ in pairs)
        canonical_artifact_ids = tuple(artifact_id for _, artifact_id in pairs)
        metrics = {
            "raw_loo_nll": _finite_real(self.raw_loo_nll, name="raw_loo_nll"),
            "calibrated_loo_nll": _finite_real(
                self.calibrated_loo_nll,
                name="calibrated_loo_nll",
            ),
            "refit_nll": _finite_real(self.refit_nll, name="refit_nll"),
            "raw_normalized_nees": _finite_real(
                self.raw_normalized_nees,
                name="raw_normalized_nees",
                nonnegative=True,
            ),
            "calibrated_refit_normalized_nees": _finite_real(
                self.calibrated_refit_normalized_nees,
                name="calibrated_refit_normalized_nees",
                nonnegative=True,
            ),
        }
        tolerance = 1e-12 * max(1.0, abs(metrics["raw_loo_nll"]))
        if metrics["calibrated_loo_nll"] > metrics["raw_loo_nll"] + tolerance:
            raise ValueError(
                "calibrated_loo_nll cannot exceed raw_loo_nll when identity is "
                "available"
            )
        metadata = frozen_finite_json_mapping(
            self.metadata,
            name="regime covariance transform metadata",
        )
        object.__setattr__(self, "regime_id", regime_id)
        object.__setattr__(self, "dimension", dimension)
        object.__setattr__(self, "covariance_scale", scale)
        object.__setattr__(self, "isotropic_variance", variance)
        object.__setattr__(self, "additive_factor", factor)
        object.__setattr__(self, "selected_floor_fraction", floor_fraction)
        object.__setattr__(self, "selected_shrinkage", shrinkage)
        object.__setattr__(self, "max_rank", max_rank)
        object.__setattr__(self, "scale_grid", scale_grid)
        object.__setattr__(self, "floor_fraction_grid", floor_grid)
        object.__setattr__(self, "shrinkage_grid", shrinkage_grid)
        object.__setattr__(self, "calibration_group_ids", canonical_group_ids)
        object.__setattr__(
            self,
            "calibration_group_artifact_ids",
            canonical_artifact_ids,
        )
        for name, value in metrics.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "metadata", metadata)
        expected_id = content_id(self.descriptor())
        if self.artifact_id is not None:
            supplied = _sha256(self.artifact_id, name="artifact_id")
            if supplied != expected_id:
                raise ValueError(
                    "regime covariance transform artifact_id does not match content"
                )
        object.__setattr__(self, "artifact_id", expected_id)

    @property
    def effective_rank(self) -> int:
        return int(self.additive_factor.shape[1])

    def descriptor(self) -> dict[str, object]:
        return {
            "schema": REGIME_COVARIANCE_TRANSFORM_SCHEMA,
            "schema_version": REGIME_COVARIANCE_TRANSFORM_VERSION,
            "regime_id": self.regime_id,
            "dimension": self.dimension,
            "covariance_scale": self.covariance_scale,
            "isotropic_variance": self.isotropic_variance,
            "additive_factor": self.additive_factor.tolist(),
            "selected_floor_fraction": self.selected_floor_fraction,
            "selected_shrinkage": self.selected_shrinkage,
            "max_rank": self.max_rank,
            "scale_grid": list(self.scale_grid),
            "floor_fraction_grid": list(self.floor_fraction_grid),
            "shrinkage_grid": list(self.shrinkage_grid),
            "calibration_groups": [
                {"group_id": group_id, "artifact_id": artifact_id}
                for group_id, artifact_id in zip(
                    self.calibration_group_ids,
                    self.calibration_group_artifact_ids,
                    strict=True,
                )
            ],
            "raw_loo_nll": self.raw_loo_nll,
            "calibrated_loo_nll": self.calibrated_loo_nll,
            "refit_nll": self.refit_nll,
            "raw_normalized_nees": self.raw_normalized_nees,
            "calibrated_refit_normalized_nees": (
                self.calibrated_refit_normalized_nees
            ),
            "metadata": plain_json(self.metadata),
        }

    def to_record(self) -> dict[str, object]:
        return {**self.descriptor(), "artifact_id": self.artifact_id}

    @classmethod
    def from_mapping(
        cls,
        values: object,
        *,
        name: str = "regime covariance transform",
    ) -> RegimeCovarianceTransformV1:
        if not isinstance(values, Mapping):
            raise ValueError(f"{name} must be a mapping")
        expected = frozenset(
            {
                "schema",
                "schema_version",
                "regime_id",
                "dimension",
                "covariance_scale",
                "isotropic_variance",
                "additive_factor",
                "selected_floor_fraction",
                "selected_shrinkage",
                "max_rank",
                "scale_grid",
                "floor_fraction_grid",
                "shrinkage_grid",
                "calibration_groups",
                "raw_loo_nll",
                "calibrated_loo_nll",
                "refit_nll",
                "raw_normalized_nees",
                "calibrated_refit_normalized_nees",
                "metadata",
                "artifact_id",
            }
        )
        require_exact_fields(values, expected=expected, name=name)
        if values["schema"] != REGIME_COVARIANCE_TRANSFORM_SCHEMA:
            raise ValueError(f"{name} schema changed")
        version = genuine_integer(
            values["schema_version"],
            name=f"{name} schema_version",
            minimum=1,
        )
        if version != REGIME_COVARIANCE_TRANSFORM_VERSION:
            raise ValueError(f"{name} version changed")
        raw_groups = values["calibration_groups"]
        if not isinstance(raw_groups, list) or not raw_groups:
            raise ValueError(f"{name} calibration_groups must be a nonempty list")
        group_ids: list[str] = []
        artifact_ids: list[str] = []
        for index, group in enumerate(raw_groups):
            if not isinstance(group, Mapping):
                raise ValueError(f"{name} calibration group {index} must be a mapping")
            require_exact_fields(
                group,
                expected=frozenset({"group_id", "artifact_id"}),
                name=f"{name} calibration group {index}",
            )
            group_ids.append(group["group_id"])
            artifact_ids.append(group["artifact_id"])
        return cls(
            regime_id=values["regime_id"],
            dimension=values["dimension"],
            covariance_scale=values["covariance_scale"],
            isotropic_variance=values["isotropic_variance"],
            additive_factor=np.asarray(values["additive_factor"]),
            selected_floor_fraction=values["selected_floor_fraction"],
            selected_shrinkage=values["selected_shrinkage"],
            max_rank=values["max_rank"],
            scale_grid=values["scale_grid"],
            floor_fraction_grid=values["floor_fraction_grid"],
            shrinkage_grid=values["shrinkage_grid"],
            calibration_group_ids=group_ids,
            calibration_group_artifact_ids=artifact_ids,
            raw_loo_nll=values["raw_loo_nll"],
            calibrated_loo_nll=values["calibrated_loo_nll"],
            refit_nll=values["refit_nll"],
            raw_normalized_nees=values["raw_normalized_nees"],
            calibrated_refit_normalized_nees=values[
                "calibrated_refit_normalized_nees"
            ],
            metadata=values["metadata"],
            artifact_id=values["artifact_id"],
        )


@dataclass(frozen=True, slots=True)
class RegimeCovarianceCalibrationV1:
    """Content-addressed collection of source-only regime transforms."""

    predictor_id: str
    query_set_id: str
    grouping_rule_id: str
    calibration_evidence_id: str
    transforms: Sequence[RegimeCovarianceTransformV1]
    predictor_frozen_before_calibration_outcomes: bool
    transform_family_frozen_before_calibration_outcomes: bool
    calibration_groups_independent: bool
    application_outcomes_used_for_fit: bool
    metadata: Mapping[str, Any] = field(default_factory=dict)
    artifact_id: str | None = None

    def __post_init__(self) -> None:
        for name in (
            "predictor_id",
            "query_set_id",
            "grouping_rule_id",
            "calibration_evidence_id",
        ):
            object.__setattr__(self, name, _sha256(getattr(self, name), name=name))
        if isinstance(self.transforms, (str, bytes)):
            raise ValueError("transforms must be a sequence")
        transforms = tuple(self.transforms)
        if not transforms or any(
            not isinstance(transform, RegimeCovarianceTransformV1)
            for transform in transforms
        ):
            raise ValueError(
                "transforms must contain one or more RegimeCovarianceTransformV1"
            )
        transforms = tuple(sorted(transforms, key=lambda item: item.regime_id))
        regime_ids = tuple(transform.regime_id for transform in transforms)
        if len(set(regime_ids)) != len(regime_ids):
            raise ValueError("transforms must contain one entry per regime")
        flags = {
            "predictor_frozen_before_calibration_outcomes": genuine_boolean(
                self.predictor_frozen_before_calibration_outcomes,
                name="predictor_frozen_before_calibration_outcomes",
            ),
            "transform_family_frozen_before_calibration_outcomes": genuine_boolean(
                self.transform_family_frozen_before_calibration_outcomes,
                name="transform_family_frozen_before_calibration_outcomes",
            ),
            "calibration_groups_independent": genuine_boolean(
                self.calibration_groups_independent,
                name="calibration_groups_independent",
            ),
            "application_outcomes_used_for_fit": genuine_boolean(
                self.application_outcomes_used_for_fit,
                name="application_outcomes_used_for_fit",
            ),
        }
        metadata = frozen_finite_json_mapping(
            self.metadata,
            name="regime covariance calibration metadata",
        )
        object.__setattr__(self, "transforms", transforms)
        for name, value in flags.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "metadata", metadata)
        expected_id = content_id(self.descriptor())
        if self.artifact_id is not None:
            supplied = _sha256(self.artifact_id, name="artifact_id")
            if supplied != expected_id:
                raise ValueError(
                    "regime covariance calibration artifact_id does not match content"
                )
        object.__setattr__(self, "artifact_id", expected_id)

    @property
    def deployment_admissible(self) -> bool:
        return (
            self.predictor_frozen_before_calibration_outcomes
            and self.transform_family_frozen_before_calibration_outcomes
            and self.calibration_groups_independent
            and not self.application_outcomes_used_for_fit
        )

    def transform_for(self, regime_id: str) -> RegimeCovarianceTransformV1:
        requested = _literal_string(regime_id, name="regime_id")
        for transform in self.transforms:
            if transform.regime_id == requested:
                return transform
        raise KeyError(f"unknown covariance-calibration regime {requested!r}")

    def descriptor(self) -> dict[str, object]:
        return {
            "schema": REGIME_COVARIANCE_CALIBRATION_SCHEMA,
            "schema_version": REGIME_COVARIANCE_CALIBRATION_VERSION,
            "predictor_id": self.predictor_id,
            "query_set_id": self.query_set_id,
            "grouping_rule_id": self.grouping_rule_id,
            "calibration_evidence_id": self.calibration_evidence_id,
            "transforms": [transform.to_record() for transform in self.transforms],
            "predictor_frozen_before_calibration_outcomes": (
                self.predictor_frozen_before_calibration_outcomes
            ),
            "transform_family_frozen_before_calibration_outcomes": (
                self.transform_family_frozen_before_calibration_outcomes
            ),
            "calibration_groups_independent": self.calibration_groups_independent,
            "application_outcomes_used_for_fit": (
                self.application_outcomes_used_for_fit
            ),
            "metadata": plain_json(self.metadata),
        }

    def to_record(self) -> dict[str, object]:
        return {
            **self.descriptor(),
            "deployment_admissible": self.deployment_admissible,
            "artifact_id": self.artifact_id,
        }

    @classmethod
    def from_mapping(
        cls,
        values: object,
        *,
        name: str = "regime covariance calibration",
    ) -> RegimeCovarianceCalibrationV1:
        if not isinstance(values, Mapping):
            raise ValueError(f"{name} must be a mapping")
        expected = frozenset(
            {
                "schema",
                "schema_version",
                "predictor_id",
                "query_set_id",
                "grouping_rule_id",
                "calibration_evidence_id",
                "transforms",
                "predictor_frozen_before_calibration_outcomes",
                "transform_family_frozen_before_calibration_outcomes",
                "calibration_groups_independent",
                "application_outcomes_used_for_fit",
                "metadata",
                "deployment_admissible",
                "artifact_id",
            }
        )
        require_exact_fields(values, expected=expected, name=name)
        if values["schema"] != REGIME_COVARIANCE_CALIBRATION_SCHEMA:
            raise ValueError(f"{name} schema changed")
        version = genuine_integer(
            values["schema_version"],
            name=f"{name} schema_version",
            minimum=1,
        )
        if version != REGIME_COVARIANCE_CALIBRATION_VERSION:
            raise ValueError(f"{name} version changed")
        raw_transforms = values["transforms"]
        if not isinstance(raw_transforms, list) or not raw_transforms:
            raise ValueError(f"{name} transforms must be a nonempty list")
        artifact = cls(
            predictor_id=values["predictor_id"],
            query_set_id=values["query_set_id"],
            grouping_rule_id=values["grouping_rule_id"],
            calibration_evidence_id=values["calibration_evidence_id"],
            transforms=tuple(
                RegimeCovarianceTransformV1.from_mapping(
                    item,
                    name=f"{name} transform {index}",
                )
                for index, item in enumerate(raw_transforms)
            ),
            predictor_frozen_before_calibration_outcomes=values[
                "predictor_frozen_before_calibration_outcomes"
            ],
            transform_family_frozen_before_calibration_outcomes=values[
                "transform_family_frozen_before_calibration_outcomes"
            ],
            calibration_groups_independent=values[
                "calibration_groups_independent"
            ],
            application_outcomes_used_for_fit=values[
                "application_outcomes_used_for_fit"
            ],
            metadata=values["metadata"],
            artifact_id=values["artifact_id"],
        )
        declared_admissible = genuine_boolean(
            values["deployment_admissible"],
            name=f"{name} deployment_admissible",
        )
        if declared_admissible != artifact.deployment_admissible:
            raise ValueError(f"{name} deployment_admissible contradicts flags")
        return artifact


@dataclass(frozen=True, slots=True)
class _Candidate:
    scale: float
    floor_fraction: float
    shrinkage: float


def _group_moments(
    groups: Sequence[CovarianceCalibrationGroupV1],
) -> tuple[np.ndarray, np.ndarray, float]:
    dimension = groups[0].dimension
    second_moments = []
    mean_covariances = []
    for group in groups:
        second_moments.append(
            np.einsum("ni,nj->ij", group.residuals, group.residuals)
            / group.row_count
        )
        mean_covariances.append(np.mean(group.covariances, axis=0))
    residual_second_moment = np.mean(np.stack(second_moments), axis=0)
    mean_covariance = np.mean(np.stack(mean_covariances), axis=0)
    reference_variance = max(
        float(np.trace(residual_second_moment) / dimension),
        np.finfo(np.float64).tiny,
    )
    return residual_second_moment, mean_covariance, reference_variance


def _fit_factor(
    groups: Sequence[CovarianceCalibrationGroupV1],
    *,
    candidate: _Candidate,
    max_rank: int,
) -> tuple[float, np.ndarray]:
    second_moment, mean_covariance, reference_variance = _group_moments(groups)
    dimension = groups[0].dimension
    isotropic_variance = candidate.floor_fraction * reference_variance
    gap = (
        second_moment
        - candidate.scale * mean_covariance
        - isotropic_variance * np.eye(dimension, dtype=np.float64)
    )
    eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (gap + gap.T))
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    tolerance = 1e-12 * max(1.0, float(np.max(np.abs(eigenvalues))))
    positive = np.flatnonzero(eigenvalues > tolerance)[:max_rank]
    if candidate.shrinkage == 0.0 or positive.size == 0:
        factor = np.empty((dimension, 0), dtype=np.float64)
    else:
        retained = candidate.shrinkage * eigenvalues[positive]
        factor = eigenvectors[:, positive] * np.sqrt(retained)[None, :]
        factor = _canonicalize_factor_signs(factor)
    return isotropic_variance, factor


def _transform_matrix(
    covariance: np.ndarray,
    *,
    scale: float,
    isotropic_variance: float,
    factor: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    dimension = covariance.shape[0]
    transformed = (
        scale * covariance
        + factor @ factor.T
        + isotropic_variance * np.eye(dimension, dtype=np.float64)
    )
    if not np.all(np.isfinite(transformed)):
        raise ArithmeticError("calibrated covariance is not finite")
    transformed = 0.5 * (transformed + transformed.T)
    try:
        cholesky = np.linalg.cholesky(transformed)
    except np.linalg.LinAlgError as error:
        raise ArithmeticError(
            "calibrated covariance is not positive definite"
        ) from error
    return transformed, cholesky


def _group_score(
    group: CovarianceCalibrationGroupV1,
    *,
    scale: float,
    isotropic_variance: float,
    factor: np.ndarray,
) -> tuple[float, float]:
    total_nll = 0.0
    total_nees = 0.0
    dimension = group.dimension
    for residual, covariance in zip(
        group.residuals,
        group.covariances,
        strict=True,
    ):
        transformed, cholesky = _transform_matrix(
            covariance,
            scale=scale,
            isotropic_variance=isotropic_variance,
            factor=factor,
        )
        whitened = np.linalg.solve(cholesky, residual)
        mahalanobis = float(whitened @ whitened)
        log_determinant = 2.0 * float(np.sum(np.log(np.diag(cholesky))))
        total_nll += 0.5 * (dimension * _LOG_2PI + log_determinant + mahalanobis)
        total_nees += mahalanobis / dimension
        del transformed
    return total_nll / group.row_count, total_nees / group.row_count


def _raw_scores(
    groups: Sequence[CovarianceCalibrationGroupV1],
) -> tuple[float, float]:
    dimension = groups[0].dimension
    empty_factor = np.empty((dimension, 0), dtype=np.float64)
    scores = [
        _group_score(
            group,
            scale=1.0,
            isotropic_variance=0.0,
            factor=empty_factor,
        )
        for group in groups
    ]
    return (
        float(np.mean([score[0] for score in scores])),
        float(np.mean([score[1] for score in scores])),
    )


def _candidate_key(candidate: _Candidate, score: float) -> tuple[float, ...]:
    return (
        score,
        float(candidate.shrinkage != 0.0),
        candidate.shrinkage,
        candidate.floor_fraction,
        abs(float(np.log(candidate.scale))),
        candidate.scale,
    )


def _fit_regime_transform(
    groups: Sequence[CovarianceCalibrationGroupV1],
    *,
    scales: tuple[float, ...],
    floor_fractions: tuple[float, ...],
    shrinkages: tuple[float, ...],
    max_rank: int,
    metadata: Mapping[str, Any],
) -> RegimeCovarianceTransformV1:
    groups = tuple(sorted(groups, key=lambda group: group.group_id))
    if len(groups) < 3:
        raise ValueError("each regime requires at least three independent groups")
    dimensions = {group.dimension for group in groups}
    if len(dimensions) != 1:
        raise ValueError("all groups in a regime must have the same dimension")
    dimension = dimensions.pop()
    effective_max_rank = min(max_rank, dimension)
    candidates = tuple(
        _Candidate(scale, floor_fraction, shrinkage)
        for scale in scales
        for floor_fraction in floor_fractions
        for shrinkage in shrinkages
    )
    candidate_scores: list[tuple[_Candidate, float]] = []
    for candidate in candidates:
        heldout_scores: list[float] = []
        for heldout_index, heldout in enumerate(groups):
            training = groups[:heldout_index] + groups[heldout_index + 1 :]
            isotropic_variance, factor = _fit_factor(
                training,
                candidate=candidate,
                max_rank=effective_max_rank,
            )
            score, _ = _group_score(
                heldout,
                scale=candidate.scale,
                isotropic_variance=isotropic_variance,
                factor=factor,
            )
            heldout_scores.append(score)
        candidate_scores.append((candidate, float(np.mean(heldout_scores))))
    selected, selected_cv_nll = min(
        candidate_scores,
        key=lambda item: _candidate_key(item[0], item[1]),
    )
    raw_nll, raw_nees = _raw_scores(groups)
    isotropic_variance, factor = _fit_factor(
        groups,
        candidate=selected,
        max_rank=effective_max_rank,
    )
    refit_scores = [
        _group_score(
            group,
            scale=selected.scale,
            isotropic_variance=isotropic_variance,
            factor=factor,
        )
        for group in groups
    ]
    return RegimeCovarianceTransformV1(
        regime_id=groups[0].regime_id,
        dimension=dimension,
        covariance_scale=selected.scale,
        isotropic_variance=isotropic_variance,
        additive_factor=factor,
        selected_floor_fraction=selected.floor_fraction,
        selected_shrinkage=selected.shrinkage,
        max_rank=effective_max_rank,
        scale_grid=scales,
        floor_fraction_grid=floor_fractions,
        shrinkage_grid=shrinkages,
        calibration_group_ids=tuple(group.group_id for group in groups),
        calibration_group_artifact_ids=tuple(
            group.artifact_id for group in groups if group.artifact_id is not None
        ),
        raw_loo_nll=raw_nll,
        calibrated_loo_nll=selected_cv_nll,
        refit_nll=float(np.mean([score[0] for score in refit_scores])),
        raw_normalized_nees=raw_nees,
        calibrated_refit_normalized_nees=float(
            np.mean([score[1] for score in refit_scores])
        ),
        metadata=metadata,
    )


def fit_regime_covariance_calibration(
    groups: Sequence[CovarianceCalibrationGroupV1],
    *,
    predictor_id: str,
    query_set_id: str,
    grouping_rule_id: str,
    calibration_evidence_id: str,
    scales: Sequence[float] = _DEFAULT_SCALES,
    floor_fractions: Sequence[float] = _DEFAULT_FLOOR_FRACTIONS,
    shrinkages: Sequence[float] = _DEFAULT_SHRINKAGES,
    max_rank: int = 3,
    predictor_frozen_before_calibration_outcomes: bool,
    transform_family_frozen_before_calibration_outcomes: bool,
    calibration_groups_independent: bool,
    application_outcomes_used_for_fit: bool,
    metadata: Mapping[str, Any] | None = None,
) -> RegimeCovarianceCalibrationV1:
    """Fit one leave-one-group-out covariance transform per regime."""

    for name, value in (
        ("predictor_id", predictor_id),
        ("query_set_id", query_set_id),
        ("grouping_rule_id", grouping_rule_id),
        ("calibration_evidence_id", calibration_evidence_id),
    ):
        _sha256(value, name=name)
    if isinstance(groups, (str, bytes)) or not isinstance(groups, Sequence):
        raise ValueError("groups must be a sequence")
    validated_groups = tuple(groups)
    if not validated_groups or any(
        not isinstance(group, CovarianceCalibrationGroupV1)
        for group in validated_groups
    ):
        raise ValueError("groups must contain CovarianceCalibrationGroupV1 values")
    group_ids = tuple(group.group_id for group in validated_groups)
    if len(set(group_ids) != len(group_ids):
        raise ValueError("calibration group_ids must be globally unique")
    scale_grid = _real_grid(scales, name="scales", positive=True)
    floor_grid = _real_grid(
        floor_fractions,
        name="floor_fractions",
        nonnegative=True,
    )
    shrinkage_grid = _real_grid(
        shrinkages,
        name="shrinkages",
        nonnegative=True,
        maximum=1.0,
    )
    if 1.0 not in scale_grid or 0.0 not in floor_grid or 0.0 not in shrinkage_grid:
        raise ValueError(
            "candidate grids must include the (scale=1, floor=0, shrinkage=0) identity"
        )
    effective_max_rank = genuine_integer(max_rank, name="max_rank", minimum=0)
    flags = {
        "predictor_frozen_before_calibration_outcomes": genuine_boolean(
            predictor_frozen_before_calibration_outcomes,
            name="predictor_frozen_before_calibration_outcomes",
        ),
        "transform_family_frozen_before_calibration_outcomes": genuine_boolean(
            transform_family_frozen_before_calibration_outcomes,
            name="transform_family_frozen_before_calibration_outcomes",
        ),
        "calibration_groups_independent": genuine_boolean(
            calibration_groups_independent,
            name="calibration_groups_independent",
        ),
        "application_outcomes_used_for_fit": genuine_boolean(
            application_outcomes_used_for_fit,
            name="application_outcomes_used_for_fit",
        ),
    }
    by_regime: dict[str, list[CovarianceCalibrationGroupV1]] = {}
    for group in validated_groups:
        by_regime.setdefault(group.regime_id, []).append(group)
    transforms = tuple(
        _fit_regime_transform(
            by_regime[regime_id],
            scales=scale_grid,
            floor_fractions=floor_grid,
            shrinkages=shrinkage_grid,
            max_rank=effective_max_rank,
            metadata={"group_balanced": True},
        )
        for regime_id in sorted(by_regime)
    )
    artifact = RegimeCovarianceCalibrationV1(
        predictor_id=predictor_id,
        query_set_id=query_set_id,
        grouping_rule_id=grouping_rule_id,
        calibration_evidence_id=calibration_evidence_id,
        transforms=transforms,
        **flags,
        metadata={} if metadata is None else metadata,
    )
    if require_deployment_admissible and not artifact.deployment_admissible:
        raise PermissionError(
            "regime covariance calibration is diagnostic and not deployment-admissible"
        )
    return artifact


def calibrate_regime_covariance(
    covariance: np.ndarray,
    *,
    regime_id: str,
    calibration: RegimeCovarianceCalibrationV1,
    require_deployment_admissible: bool = True,
) -> np.ndarray:
    """Apply the frozen transform to a strict SPD covariance or batch."""

    if not isinstance(calibration, RegimeCovarianceCalibrationV1):
        raise TypeError("calibration must be a RegimeCovarianceCalibrationV1")
    if require_deployment_admissible and not calibration.deployment_admissible:
        raise PermissionError(
            "regime covariance calibration is diagnostic and not deployment-admissible"
        )
    transform = calibration.transform_for(regime_id)
    raw = np.asarray(covariance)
    if raw.dtype.kind not in "iuf":
        raise ValueError("covariance must contain real numeric values")
    covariance_array = np.asarray(raw, dtype=np.float64)
    if covariance_array.ndim < 2 or covariance_array.shape[-2:] != (
        transform.dimension,
        transform.dimension,
    ):
        raise ValueError(
            "covariance matrices must match the selected transform dimension"
        )
    if covariance_array.size == 0 or not np.all(np.isfinite(covariance_array)):
        raise ValueError("covariance must be nonempty and finite")
    flattened = covariance_array.reshape(
        (-1, transform.dimension, transform.dimension)
    )
    output = np.empty_like(flattened)
    for index, matrix in enumerate(flattened):
        if not np.allclose(matrix, matrix.T, rtol=1e-11, atol=1e-12):
            raise ValueError(f"covariance {index} must be symmetric")
        symmetric = 0.5 * (matrix + matrix.T)
        try:
            np.linalg.cholesky(symmetric)
        except np.linalg.LinAlgError as error:
            raise ValueError(
                f"covariance {index} must be positive definite"
            ) from error
        calibrated, _ = _transform_matrix(
            matrix,
            scale=transform.covariance_scale,
            isotropic_variance=transform.isotropic_variance,
            factor=transform.additive_factor,
        )
        output[index] = calibrated
    return immutable_array(
        output.reshape(covariance_array.shape),
        dtype=np.dtype("<f8"),
    )


def _validated_guard_support(
    certificate: CalibrationDomainGuardCertificateV1,
    *,
    regime_id: str,
    transform: RegimeCovarianceTransformV1,
) -> tuple[str, str]:
    if not isinstance(certificate, CalibrationDomainGuardCertificateV1):
        raise TypeError(
            "certificate must be a CalibrationDomainGuardCertificateV1"
        )
    if not certificate.deployment_admissible:
        raise PermissionError(
            "calibration-domain guard is diagnostic and not deployment-admissible"
        )
    decision = certificate.decision_for_domain(regime_id)
    if decision is None:
        raise KeyError(f"unknown calibration-domain guard regime {regime_id!r}")
    if not decision.calibration_supported:
        raise PermissionError(
            f"calibration-domain guard rejects regime {regime_id!r}"
        )
    if tuple(decision.group_ids) != tuple(transform.calibration_group_ids):
        raise ValueError(
            "covariance calibration and domain guard use different group rosters"
        )
    if certificate.artifact_id is None or decision.artifact_id is None:
        raise AssertionError("validated guard records must be content-addressed")
    return certificate.artifact_id, decision.artifact_id


def calibrate_guarded_regime_covariance(
    covariance: np.ndarray,
    *,
    regime_id: str,
    calibration: RegimeCovarianceCalibrationV1,
    certificate: CalibrationDomainGuardCertificateV1
et_id", query_set_id),
        ("grouping_rule_id", grouping_rule_id),
        ("calibration_evidence_id", calibration_evidence_id),
    ):
        _sha256(value, name=name)
    flags = {
        "predictor_frozen_before_calibration_outcomes": genuine_boolean(
            predictor_frozen_before_calibration_outcomes,
            name="predictor_frozen_before_calibration_outcomes",
        ),
        "transform_family_frozen_before_calibration_outcomes": genuine_boolean(
            transform_family_frozen_before_calibration_outcomes,
            name="transform_family_frozen_before_calibration_outcomes",
        ),
        "calibration_groups_independent": genuine_boolean(
            calibration_groups_independent,
            name="calibration_groups_independent",
        ),
        "application_outcomes_used_for_fit": genuine_boolean(
            application_outcomes_used_for_fit,
            name="application_outcomes_used_for_fit",
        ),
    }
    scale_grid = _real_grid(scales, name="scales", positive=True)
    floor_grid = _real_grid(
        floor_fractions,
        name="floor_fractions",
        nonnegative=True,
    )
    shrinkage_grid = _real_grid(
        shrinkages,
        name="shrinkages",
        nonnegative=True,
        maximum=1.0,
    )
    if 1.0 not in scale_grid or 0.0 not in floor_grid or 0.0 not in shrinkage_grid:
        raise ValueError(
            "candidate grids must include the identity transform (1.0, 0.0, 0.0)"
        )
    requested_max_rank = genuine_integer(max_rank, name="max_rank", minimum=0)
    if isinstance(groups, (str, bytes)) or not isinstance(groups, Sequence):
        raise ValueError("groups must be a sequence")
    validated_groups = tuple(groups)
    if not validated_groups or any(
        not isinstance(group, CovarianceCalibrationGroupV1)
        for group in validated_groups
    ):
        raise ValueError(
            "groups must contain one or more CovarianceCalibrationGroupV1 entries"
        )
    group_ids = tuple(group.group_id for group in validated_groups)
    if len(set(group_ids)) != len(group_ids):
        raise ValueError("calibration group IDs must be globally unique")
    by_regime: dict[str, list[CovarianceCalibrationGroupV1]] = {}
    for group in validated_groups:
        by_regime.setdefault(group.regime_id, []).append(group)
    frozen_metadata = frozen_finite_json_mapping(
        metadata,
        name="regime covariance calibration metadata",
    )
    transforms = tuple(
        _fit_regime_transform(
            tuple(regime_groups),
            scales=scale_grid,
            floor_fractions=floor_grid,
            shrinkages=shrinkage_grid,
            max_rank=requested_max_rank,
            metadata={
                "fit_unit": "independent-group",
                "selection": "leave-one-group-out-gaussian-nll",
            },
        )
        for _, regime_groups in sorted(by_regime.items())
    )
    return RegimeCovarianceCalibrationV1(
        predictor_id=predictor_id,
        query_set_id=query_set_id,
        grouping_rule_id=grouping_rule_id,
        calibration_evidence_id=calibration_evidence_id,
        transforms=transforms,
        predictor_frozen_before_calibration_outcomes=flags[
            "predictor_frozen_before_calibration_outcomes"
        ],
        transform_family_frozen_before_calibration_outcomes=flags[
            "transform_family_frozen_before_calibration_outcomes"
        ],
        calibration_groups_independent=flags["calibration_groups_independent"],
        application_outcomes_used_for_fit=flags[
            "application_outcomes_used_for_fit"
        ],
        metadata=frozen_metadata,
    )


def calibrate_regime_covariance(
    covariance: np.ndarray,
    *,
    regime_id: str,
    calibration: RegimeCovarianceCalibrationV1,
    require_deployment_admissible: bool = True,
) -> np.ndarray:
    """Apply one frozen regime transform to one or more covariance matrices."""

    if not isinstance(calibration, RegimeCovarianceCalibrationV1):
        raise TypeError("calibration must be a RegimeCovarianceCalibrationV1")
    require_admissible = genuine_boolean(
        require_deployment_admissible,
        name="require_deployment_admissible",
    )
    if require_admissible and not calibration.deployment_admissible:
        raise PermissionError(
            "regime covariance calibration is diagnostic and not deployment-admissible"
        )
    transform = calibration.transform_for(regime_id)
    raw = np.asarray(covariance)
    if raw.dtype.kind not in "iuf":
        raise ValueError("covariance must contain real numeric values")
    covariance_array = np.asarray(raw, dtype=np.float64)
    if covariance_array.ndim < 2 or covariance_array.shape[-2:] != (
        transform.dimension,
        transform.dimension,
    ):
        raise ValueError(
            "covariance matrices must match the selected transform dimension"
        )
    if covariance_array.size == 0 or not np.all(np.isfinite(covariance_array)):
        raise ValueError("covariance must be nonempty and finite")
    flattened = covariance_array.reshape(
        (-1, transform.dimension, transform.dimension)
    )
    output = np.empty_like(flattened)
    for index, matrix in enumerate(flattened):
        if not np.allclose(matrix, matrix.T, rtol=1e-11, atol=1e-12):
            raise ValueError(f"covariance {index} must be symmetric")
        symmetric = 0.5 * (matrix + matrix.T)
        try:
            np.linalg.cholesky(symmetric)
        except np.linalg.LinAlgError as error:
            raise ValueError(
                f"covariance {index} must be positive definite"
            ) from error
        calibrated, _ = _transform_matrix(
            matrix,
            scale=transform.covariance_scale,
            isotropic_variance=transform.isotropic_variance,
            factor=transform.additive_factor,
        )
        output[index] = calibrated
    return immutable_array(
        output.reshape(covariance_array.shape),
        dtype=np.dtype("<f8"),
    )


def _validated_guard_support(
    certificate: CalibrationDomainGuardCertificateV1,
    *,
    regime_id: str,
    transform: RegimeCovarianceTransformV1,
) -> tuple[str, str]:
    if not isinstance(certificate, CalibrationDomainGuardCertificateV1):
        raise TypeError(
            "certificate must be a CalibrationDomainGuardCertificateV1"
        )
    if not certificate.deployment_admissible:
        raise PermissionError(
            "calibration-domain guard is diagnostic and not deployment-admissible"
        )
    decision = certificate.decision_for_domain(regime_id)
    if decision is None:
        raise KeyError(f"unknown calibration-domain guard regime {regime_id!r}")
    if not decision.calibration_supported:
        raise PermissionError(
            f"calibration-domain guard rejects regime {regime_id!r}"
        )
    if tuple(decision.group_ids) != tuple(transform.calibration_group_ids):
        raise ValueError(
            "covariance calibration and domain guard use different group rosters"
        )
    if certificate.artifact_id is None or decision.artifact_id is None:
        raise AssertionError("validated guard records must be content-addressed")
    return certificate.artifact_id, decision.artifact_id


def calibrate_guarded_regime_covariance(
    covariance: np.ndarray,
    *,
    regime_id: str,
    calibration: RegimeCovarianceCalibrationV1,
    certificate: CalibrationDomainGuardCertificateV1,
) -> np.ndarray:
    """Apply a regime transform only after the merged domain guard authorizes it."""

    if not isinstance(calibration, RegimeCovarianceCalibrationV1):
        raise TypeError("calibration must be a RegimeCovarianceCalibrationV1")
    transform = calibration.transform_for(regime_id)
    _validated_guard_support(
        certificate,
        regime_id=transform.regime_id,
        transform=transform,
    )
    return calibrate_regime_covariance(
        covariance,
        regime_id=transform.regime_id,
        calibration=calibration,
        require_deployment_admissible=True,
    )


def calibrated_covariance_semantics(
    semantics: PosteriorCovarianceSemanticsV1,
    *,
    regime_id: str,
    calibration: RegimeCovarianceCalibrationV1,
    certificate: CalibrationDomainGuardCertificateV1 | None = None,
) -> PosteriorCovarianceSemanticsV1:
    """Bind a calibrated covariance to its source calibration artifact."""

    if not isinstance(semantics, PosteriorCovarianceSemanticsV1):
        raise TypeError("semantics must be PosteriorCovarianceSemanticsV1")
    if not isinstance(calibration, RegimeCovarianceCalibrationV1):
        raise TypeError("calibration must be a RegimeCovarianceCalibrationV1")
    if semantics.method == "exact_prior_fallback":
        raise ValueError("exact prior fallback covariance must remain uncalibrated")
    if semantics.calibrated:
        raise ValueError("covariance semantics are already calibrated")
    if not calibration.deployment_admissible:
        raise PermissionError(
            "diagnostic covariance calibration cannot mark deployment semantics"
        )
    transform = calibration.transform_for(regime_id)
    if semantics.dimension != transform.dimension:
        raise ValueError("covariance semantics dimension does not match transform")
    metadata = dict(semantics.metadata)
    metadata.update(
        {
            "regime_covariance_calibration_id": calibration.artifact_id,
            "regime_covariance_transform_id": transform.artifact_id,
            "regime_id": transform.regime_id,
        }
    )
    if certificate is not None:
        certificate_id, decision_id = _validated_guard_support(
            certificate,
            regime_id=transform.regime_id,
            transform=transform,
        )
        metadata.update(
            {
                "calibration_domain_guard_certificate_id": certificate_id,
                "calibration_domain_guard_decision_id": decision_id,
            }
        )
    return PosteriorCovarianceSemanticsV1(
        method=semantics.method,
        dimension=semantics.dimension,
        likelihood_power_semantics=semantics.likelihood_power_semantics,
        prior_included=semantics.prior_included,
        generalized_bayes=semantics.generalized_bayes,
        mixture_curvature_exact=semantics.mixture_curvature_exact,
        group_score_correction=semantics.group_score_correction,
        calibrated=True,
        calibration_artifact_id=calibration.artifact_id,
        metadata=metadata,
    )


def save_regime_covariance_calibration(
    calibration: RegimeCovarianceCalibrationV1,
    path: str | Path,
    *,
    overwrite: bool = False,
) -> None:
    """Atomically write one validated calibration artifact."""

    if not isinstance(calibration, RegimeCovarianceCalibrationV1):
        raise TypeError("calibration must be a RegimeCovarianceCalibrationV1")
    write_atomic_json(calibration.to_record(), path, overwrite=overwrite)


def load_regime_covariance_calibration(
    path: str | Path,
) -> RegimeCovarianceCalibrationV1:
    """Load and revalidate one strict covariance-calibration JSON artifact."""

    values = load_strict_json_object(
        path,
        label="regime covariance calibration",
    )
    return RegimeCovarianceCalibrationV1.from_mapping(values)


__all__ = [
    "REGIME_COVARIANCE_CALIBRATION_SCHEMA",
    "REGIME_COVARIANCE_CALIBRATION_VERSION",
    "CovarianceCalibrationGroupV1",
    "RegimeCovarianceCalibrationV1",
    "RegimeCovarianceTransformV1",
    "calibrate_guarded_regime_covariance",
    "calibrate_regime_covariance",
    "calibrated_covariance_semantics",
    "fit_regime_covariance_calibration",
    "load_regime_covariance_calibration",
    "save_regime_covariance_calibration",
]
