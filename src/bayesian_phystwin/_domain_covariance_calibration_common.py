"""Shared contracts and canonical helpers for domain covariance calibration.

The fitted transform is intentionally conservative::

    covariance_calibrated = scale * covariance_raw + floor_variance * I

``scale`` is constrained to be at least one and ``floor_variance`` is
nonnegative. Parameters are selected by group-balanced Gaussian negative log
likelihood. Domain support is then decided from leave-one-group-out predictive
loss ratios through :mod:`bayesian_phystwin.calibration_domain_guard`.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from ._canonical_contracts import genuine_integer
from ._portable_contracts import content_id

DOMAIN_COVARIANCE_CALIBRATION_SCHEMA = "bayesian_phystwin.domain_covariance_calibration"
DOMAIN_COVARIANCE_CALIBRATION_VERSION = 1
DOMAIN_COVARIANCE_TRANSFORM_SCHEMA = "bayesian_phystwin.domain_covariance_transform"
DOMAIN_COVARIANCE_TRANSFORM_VERSION = 1
DOMAIN_COVARIANCE_FOLD_SCHEMA = "bayesian_phystwin.domain_covariance_fold"
DOMAIN_COVARIANCE_FOLD_VERSION = 1
DOMAIN_COVARIANCE_DATA_SCHEMA = "bayesian_phystwin.domain_covariance_data"
DOMAIN_COVARIANCE_DATA_VERSION = 1
DOMAIN_COVARIANCE_APPLICATION_SCHEMA = "bayesian_phystwin.domain_covariance_application"
DOMAIN_COVARIANCE_APPLICATION_VERSION = 1
DOMAIN_COVARIANCE_GUARD_METRIC = "leave-one-group-out-geometric-gaussian-loss-ratio"

_GAUSSIAN_CONSTANT = math.log(2.0 * math.pi)
_COORDINATE_COVERAGE_Z90 = 1.6448536269514722


def _canonical_string(value: object, *, name: str) -> str:
    if type(value) is not str or not value or value.strip() != value:
        raise ValueError(f"{name} must be a nonempty canonical string")
    return value


def _canonical_strings(
    values: Sequence[str],
    *,
    name: str,
    allow_empty: bool = False,
) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{name} must be a sequence of strings")
    try:
        source = tuple(values)
    except TypeError as error:
        raise ValueError(f"{name} must be a sequence of strings") from error
    if not allow_empty and not source:
        raise ValueError(f"{name} must not be empty")
    result = tuple(
        _canonical_string(value, name=f"{name}[{index}]")
        for index, value in enumerate(source)
    )
    return result


def _finite_float(value: object, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a finite number")
    raw = np.asarray(value)
    if raw.ndim != 0 or raw.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be a finite number")
    result = float(raw.item())
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


def _bounded_float(
    value: object,
    *,
    name: str,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    result = _finite_float(value, name=name)
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    if maximum is not None and result > maximum:
        raise ValueError(f"{name} must be <= {maximum}")
    return result


def _canonical_float_array(
    value: object,
    *,
    name: str,
    ndim: int,
) -> np.ndarray:
    raw = np.asarray(value)
    if raw.ndim != ndim or raw.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be a {ndim}-dimensional numeric array")
    result = np.array(raw, dtype=np.dtype("<f8"), copy=True, order="C")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values")
    return result


def _array_record(array: np.ndarray) -> dict[str, object]:
    canonical = np.array(array, dtype=np.dtype("<f8"), copy=True, order="C")
    return {
        "dtype": "<f8",
        "shape": list(canonical.shape),
        "sha256": hashlib.sha256(canonical.tobytes(order="C")).hexdigest(),
    }


def _array_id(array: np.ndarray) -> str:
    return content_id(
        {
            "schema": "bayesian_phystwin.canonical_float64_array",
            "schema_version": 1,
            **_array_record(array),
        }
    )


def _matches_frozen_grid(value: float, grid: Sequence[float]) -> bool:
    return any(
        math.isclose(value, candidate, rel_tol=1e-12, abs_tol=1e-15)
        for candidate in grid
    )


@dataclass(frozen=True, slots=True)
class DomainCovarianceCalibrationConfigV1:
    """Deterministic conservative scale-plus-floor calibration grid."""

    minimum_scale: float = 1.0
    maximum_scale: float = 4096.0
    scale_grid_size: int = 49
    minimum_positive_floor_ratio: float = 1e-6
    maximum_floor_ratio: float = 4.0
    floor_grid_size: int = 17
    symmetry_tolerance: float = 1e-10
    minimum_eigenvalue: float = 0.0
    score_tolerance: float = 1e-3
    log_loss_ratio_clip: float = 50.0

    def __post_init__(self) -> None:
        minimum_scale = _bounded_float(
            self.minimum_scale,
            name="minimum_scale",
            minimum=1.0,
        )
        if minimum_scale != 1.0:
            raise ValueError(
                "minimum_scale must equal 1.0 so the identity transform is present"
            )
        maximum_scale = _bounded_float(
            self.maximum_scale,
            name="maximum_scale",
            minimum=minimum_scale,
        )
        scale_grid_size = genuine_integer(
            self.scale_grid_size,
            name="scale_grid_size",
            minimum=1,
        )
        minimum_floor = _bounded_float(
            self.minimum_positive_floor_ratio,
            name="minimum_positive_floor_ratio",
            minimum=0.0,
        )
        maximum_floor = _bounded_float(
            self.maximum_floor_ratio,
            name="maximum_floor_ratio",
            minimum=minimum_floor,
        )
        floor_grid_size = genuine_integer(
            self.floor_grid_size,
            name="floor_grid_size",
            minimum=0,
        )
        if floor_grid_size > 0 and minimum_floor <= 0.0:
            raise ValueError(
                "minimum_positive_floor_ratio must be positive when the "
                "floor grid is enabled"
            )
        symmetry_tolerance = _bounded_float(
            self.symmetry_tolerance,
            name="symmetry_tolerance",
            minimum=0.0,
        )
        minimum_eigenvalue = _bounded_float(
            self.minimum_eigenvalue,
            name="minimum_eigenvalue",
            minimum=0.0,
        )
        score_tolerance = _bounded_float(
            self.score_tolerance,
            name="score_tolerance",
            minimum=0.0,
        )
        log_loss_ratio_clip = _bounded_float(
            self.log_loss_ratio_clip,
            name="log_loss_ratio_clip",
            minimum=np.finfo(np.float64).eps,
        )
        object.__setattr__(self, "minimum_scale", minimum_scale)
        object.__setattr__(self, "maximum_scale", maximum_scale)
        object.__setattr__(self, "scale_grid_size", scale_grid_size)
        object.__setattr__(
            self,
            "minimum_positive_floor_ratio",
            minimum_floor,
        )
        object.__setattr__(self, "maximum_floor_ratio", maximum_floor)
        object.__setattr__(self, "floor_grid_size", floor_grid_size)
        object.__setattr__(self, "symmetry_tolerance", symmetry_tolerance)
        object.__setattr__(self, "minimum_eigenvalue", minimum_eigenvalue)
        object.__setattr__(self, "score_tolerance", score_tolerance)
        object.__setattr__(self, "log_loss_ratio_clip", log_loss_ratio_clip)

    def scale_grid(self) -> tuple[float, ...]:
        if self.scale_grid_size == 1:
            return (self.minimum_scale,)
        values = np.geomspace(
            self.minimum_scale,
            self.maximum_scale,
            self.scale_grid_size,
            dtype=np.float64,
        )
        return tuple(float(value) for value in np.unique(values))

    def floor_ratio_grid(self) -> tuple[float, ...]:
        if self.floor_grid_size == 0:
            return (0.0,)
        if self.floor_grid_size == 1:
            positive = (self.minimum_positive_floor_ratio,)
        else:
            values = np.geomspace(
                self.minimum_positive_floor_ratio,
                self.maximum_floor_ratio,
                self.floor_grid_size,
                dtype=np.float64,
            )
            positive = tuple(float(value) for value in np.unique(values))
        return (0.0, *positive)

    def descriptor(self) -> dict[str, object]:
        return {
            "minimum_scale": self.minimum_scale,
            "maximum_scale": self.maximum_scale,
            "scale_grid_size": self.scale_grid_size,
            "minimum_positive_floor_ratio": (self.minimum_positive_floor_ratio),
            "maximum_floor_ratio": self.maximum_floor_ratio,
            "floor_grid_size": self.floor_grid_size,
            "symmetry_tolerance": self.symmetry_tolerance,
            "minimum_eigenvalue": self.minimum_eigenvalue,
            "score_tolerance": self.score_tolerance,
            "log_loss_ratio_clip": self.log_loss_ratio_clip,
        }
