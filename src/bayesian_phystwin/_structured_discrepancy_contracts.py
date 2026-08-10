"""Immutable contracts for structured deformable-object discrepancy beliefs."""

from __future__ import annotations

from dataclasses import dataclass, field
from numbers import Real
from typing import TypeAlias

import numpy as np

from .endpoint_model_average import ModelAveragedEndpointConfigV1

STRUCTURED_DISCREPANCY_CONTRACT_VERSION = 1
STRUCTURED_DISCREPANCY_EVIDENCE_SEMANTICS = (
    "mean-cumulative-track-marginal-mixture-log-score-v1"
)
STRUCTURED_DISCREPANCY_COVARIANCE_SEMANTICS = (
    "diagonal-local-plus-shared-basis-plus-model-disagreement-v1"
)
STRUCTURED_DISCREPANCY_CLAIM_BOUNDARY = (
    "Development interface only. The belief describes an observable discrepancy "
    "field, not an identified simulator-state correction. Raw covariance still "
    "requires independent object- or session-level calibration."
)


def _readonly(
    value: np.ndarray,
    *,
    dtype: np.dtype | type = np.float64,
) -> np.ndarray:
    result = np.array(value, dtype=dtype, copy=True, order="C")
    result.setflags(write=False)
    return result


def _require_real_scalar(
    value: object,
    *,
    name: str,
    positive: bool = False,
) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if positive and result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


@dataclass(frozen=True, slots=True)
class StructuredDiscrepancyConfigV1:
    """Robust component family and numerical contract for a spatial basis."""

    endpoint_config: ModelAveragedEndpointConfigV1 = field(
        default_factory=ModelAveragedEndpointConfigV1
    )
    basis_orthonormal_atol: float = 1e-10

    def __post_init__(self) -> None:
        if not isinstance(self.endpoint_config, ModelAveragedEndpointConfigV1):
            raise TypeError("endpoint_config must be a ModelAveragedEndpointConfigV1")
        tolerance = _require_real_scalar(
            self.basis_orthonormal_atol,
            name="basis_orthonormal_atol",
            positive=True,
        )
        object.__setattr__(self, "basis_orthonormal_atol", tolerance)


def _validated_basis(
    spatial_basis: np.ndarray,
    *,
    track_count: int,
    tolerance: float,
) -> np.ndarray:
    raw = np.asarray(spatial_basis)
    if raw.dtype.kind in {"b", "O", "U", "S"}:
        raise TypeError("spatial_basis must be a numeric matrix")
    basis = np.asarray(raw, dtype=np.float64)
    if basis.ndim != 2 or basis.shape[0] != track_count:
        raise ValueError("spatial_basis must have shape (track_count, rank)")
    if not 1 <= basis.shape[1] <= track_count:
        raise ValueError("spatial_basis rank must lie in [1, track_count]")
    if not np.all(np.isfinite(basis)):
        raise ValueError("spatial_basis must contain only finite values")
    gram = basis.T @ basis
    if not np.allclose(
        gram,
        np.eye(basis.shape[1]),
        atol=tolerance,
        rtol=tolerance,
    ):
        raise ValueError("spatial_basis columns must be orthonormal")
    return basis


def _numeric_array(value: object, *, name: str) -> np.ndarray:
    raw = np.asarray(value)
    if raw.dtype.kind not in {"i", "u", "f"}:
        raise TypeError(f"{name} must be real numeric")
    result = np.asarray(raw, dtype=np.float64)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values")
    return result


def _validate_component_state(
    spatial_basis: np.ndarray,
    component_coefficient_mean_m: np.ndarray,
    component_coefficient_covariance_m2: np.ndarray,
    component_local_variance_m2: np.ndarray,
    component_weights: np.ndarray,
    component_process_variance_m2: np.ndarray,
    *,
    basis_tolerance: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    raw_basis = np.asarray(spatial_basis)
    if raw_basis.ndim != 2 or raw_basis.shape[0] < 1:
        raise ValueError("spatial_basis must be a nonempty matrix")
    basis = _validated_basis(
        raw_basis,
        track_count=raw_basis.shape[0],
        tolerance=basis_tolerance,
    )
    coefficient_mean = _numeric_array(
        component_coefficient_mean_m,
        name="component_coefficient_mean_m",
    )
    coefficient_covariance = _numeric_array(
        component_coefficient_covariance_m2,
        name="component_coefficient_covariance_m2",
    )
    local_variance = _numeric_array(
        component_local_variance_m2,
        name="component_local_variance_m2",
    )
    weights = _numeric_array(component_weights, name="component_weights")
    process_variance = _numeric_array(
        component_process_variance_m2,
        name="component_process_variance_m2",
    )
    track_count, rank = basis.shape
    if coefficient_mean.ndim != 3 or coefficient_mean.shape[1:] != (rank, 3):
        raise ValueError(
            "component_coefficient_mean_m must have shape (components, rank, 3)"
        )
    component_count = len(coefficient_mean)
    expected_covariance_shape = (component_count, rank, rank)
    if coefficient_covariance.shape != expected_covariance_shape:
        raise ValueError("component_coefficient_covariance_m2 shape changed")
    if local_variance.shape != (component_count, track_count):
        raise ValueError("component_local_variance_m2 shape changed")
    if weights.shape != (component_count,):
        raise ValueError("component_weights shape changed")
    if process_variance.shape != (component_count,):
        raise ValueError("component_process_variance_m2 shape changed")
    if np.any(local_variance < 0.0) or np.any(process_variance < 0.0):
        raise ValueError("structured discrepancy variances must be nonnegative")
    if np.any(weights < 0.0) or not np.isclose(
        np.sum(weights),
        1.0,
        atol=1e-12,
        rtol=1e-12,
    ):
        raise ValueError("component_weights must be normalized")
    if not np.allclose(
        coefficient_covariance,
        coefficient_covariance.transpose(0, 2, 1),
        atol=1e-10,
        rtol=1e-10,
    ):
        raise ValueError("component coefficient covariance must be symmetric")
    eigenvalues = np.linalg.eigvalsh(coefficient_covariance)
    if np.min(eigenvalues, initial=0.0) < -1e-12:
        raise ValueError(
            "component coefficient covariance must be positive semidefinite"
        )
    return (
        basis,
        coefficient_mean,
        coefficient_covariance,
        local_variance,
        weights,
        process_variance,
    )


def _derived_field_moments(
    basis: np.ndarray,
    coefficient_mean: np.ndarray,
    coefficient_covariance: np.ndarray,
    local_variance: np.ndarray,
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    component_mean = np.einsum("nr,krc->knc", basis, coefficient_mean)
    mean = np.einsum("k,knc->nc", weights, component_mean)
    represented_variance = np.einsum(
        "nr,krs,ns->kn",
        basis,
        coefficient_covariance,
        basis,
    )
    scalar_variance = np.maximum(represented_variance + local_variance, 0.0)
    centered = component_mean - mean[None, :, :]
    between = centered[:, :, :, None] * centered[:, :, None, :]
    within = scalar_variance[:, :, None, None] * np.eye(3, dtype=np.float64)
    marginal = np.einsum("k,knij->nij", weights, within + between)
    marginal = 0.5 * (marginal + marginal.transpose(0, 2, 1))
    return component_mean, mean, marginal


@dataclass(frozen=True, slots=True)
class StructuredDiscrepancyPosteriorV1:
    """Factorized endpoint belief with shared spatial modes and local variance."""

    spatial_basis: np.ndarray
    component_coefficient_mean_m: np.ndarray
    component_coefficient_covariance_m2: np.ndarray
    component_local_variance_m2: np.ndarray
    component_weights: np.ndarray
    component_log_score: np.ndarray
    component_final_nominal_probability: np.ndarray
    update_count: np.ndarray
    component_process_variance_m2: np.ndarray
    config: StructuredDiscrepancyConfigV1
    end_frame: int
    component_mean_m: np.ndarray = field(init=False, repr=False)
    mean_m: np.ndarray = field(init=False)
    marginal_covariance_m2: np.ndarray = field(init=False, repr=False)
    final_nominal_probability: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.config, StructuredDiscrepancyConfigV1):
            raise TypeError("config must be a StructuredDiscrepancyConfigV1")
        if isinstance(self.end_frame, (bool, np.bool_)):
            raise TypeError("end_frame must be an integer")
        end_frame = int(self.end_frame)
        if end_frame != self.end_frame or end_frame < 1:
            raise ValueError("end_frame must be a positive integer")
        (
            basis,
            coefficient_mean,
            coefficient_covariance,
            local_variance,
            weights,
            process_variance,
        ) = _validate_component_state(
            self.spatial_basis,
            self.component_coefficient_mean_m,
            self.component_coefficient_covariance_m2,
            self.component_local_variance_m2,
            self.component_weights,
            self.component_process_variance_m2,
            basis_tolerance=self.config.basis_orthonormal_atol,
        )
        component_count = len(weights)
        track_count = len(basis)
        expected_component_count = len(self.config.endpoint_config.components)
        if component_count != expected_component_count:
            raise ValueError("component count differs from endpoint_config")
        expected_process_variance = np.asarray(
            [
                component.process_std_m**2
                for component in self.config.endpoint_config.components
            ],
            dtype=np.float64,
        )
        if not np.allclose(
            process_variance,
            expected_process_variance,
            atol=0.0,
            rtol=0.0,
        ):
            raise ValueError("component process variance differs from config")
        scores = np.asarray(self.component_log_score, dtype=np.float64)
        probability = np.asarray(
            self.component_final_nominal_probability,
            dtype=np.float64,
        )
        raw_count = np.asarray(self.update_count)
        if scores.shape != (component_count,) or not np.all(np.isfinite(scores)):
            raise ValueError("component_log_score must be a finite component vector")
        prior = np.asarray(
            self.config.endpoint_config.component_prior_probability,
            dtype=np.float64,
        )
        expected_log_weight = np.log(prior) + scores
        expected_log_weight = expected_log_weight - np.max(expected_log_weight)
        expected_weights = np.exp(expected_log_weight)
        expected_weights = expected_weights / np.sum(expected_weights)
        if not np.allclose(weights, expected_weights, atol=1e-12, rtol=1e-12):
            raise ValueError("component_weights disagree with scores and prior")
        if probability.shape != (component_count, track_count):
            raise ValueError("component_final_nominal_probability shape changed")
        if not np.all(np.isfinite(probability)) or np.any(
            (probability < 0.0) | (probability > 1.0)
        ):
            raise ValueError("component_final_nominal_probability must lie in [0, 1]")
        if not np.issubdtype(raw_count.dtype, np.integer):
            raise ValueError("update_count must contain integers")
        update_count = np.asarray(raw_count, dtype=np.int64)
        if update_count.shape != (track_count,) or np.any(update_count < 0):
            raise ValueError("update_count must be a nonnegative track vector")
        component_mean, mean, marginal = _derived_field_moments(
            basis,
            coefficient_mean,
            coefficient_covariance,
            local_variance,
            weights,
        )
        final_probability = np.einsum("k,kn->n", weights, probability)
        for name, value, dtype in (
            ("spatial_basis", basis, np.float64),
            ("component_coefficient_mean_m", coefficient_mean, np.float64),
            (
                "component_coefficient_covariance_m2",
                coefficient_covariance,
                np.float64,
            ),
            ("component_local_variance_m2", local_variance, np.float64),
            ("component_weights", weights, np.float64),
            ("component_log_score", scores, np.float64),
            (
                "component_final_nominal_probability",
                probability,
                np.float64,
            ),
            ("update_count", update_count, np.int64),
            (
                "component_process_variance_m2",
                process_variance,
                np.float64,
            ),
            ("component_mean_m", component_mean, np.float64),
            ("mean_m", mean, np.float64),
            ("marginal_covariance_m2", marginal, np.float64),
            ("final_nominal_probability", final_probability, np.float64),
        ):
            object.__setattr__(self, name, _readonly(value, dtype=dtype))
        object.__setattr__(self, "end_frame", end_frame)

    @property
    def updated_mask(self) -> np.ndarray:
        result = self.update_count > 0
        result.setflags(write=False)
        return result


@dataclass(frozen=True, slots=True)
class StructuredDiscrepancyPredictionV1:
    """Horizon-propagated structured discrepancy moments."""

    spatial_basis: np.ndarray
    component_coefficient_mean_m: np.ndarray
    component_coefficient_covariance_m2: np.ndarray
    component_local_variance_m2: np.ndarray
    component_weights: np.ndarray
    component_process_variance_m2: np.ndarray
    config: StructuredDiscrepancyConfigV1
    source_end_frame: int
    horizon_steps: int
    component_mean_m: np.ndarray = field(init=False, repr=False)
    mean_m: np.ndarray = field(init=False)
    marginal_covariance_m2: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.config, StructuredDiscrepancyConfigV1):
            raise TypeError("config must be a StructuredDiscrepancyConfigV1")
        for raw_value, field_name, minimum in (
            (self.source_end_frame, "source_end_frame", 1),
            (self.horizon_steps, "horizon_steps", 0),
        ):
            if isinstance(raw_value, (bool, np.bool_)):
                raise TypeError(f"{field_name} must be an integer")
            integer = int(raw_value)
            if integer != raw_value or integer < minimum:
                raise ValueError(
                    f"{field_name} must be an integer of at least {minimum}"
                )
            object.__setattr__(self, field_name, integer)
        (
            basis,
            coefficient_mean,
            coefficient_covariance,
            local_variance,
            weights,
            process_variance,
        ) = _validate_component_state(
            self.spatial_basis,
            self.component_coefficient_mean_m,
            self.component_coefficient_covariance_m2,
            self.component_local_variance_m2,
            self.component_weights,
            self.component_process_variance_m2,
            basis_tolerance=self.config.basis_orthonormal_atol,
        )
        expected_component_count = len(self.config.endpoint_config.components)
        if len(weights) != expected_component_count:
            raise ValueError("component count differs from endpoint_config")
        expected_process_variance = np.asarray(
            [
                component.process_std_m**2
                for component in self.config.endpoint_config.components
            ],
            dtype=np.float64,
        )
        if not np.allclose(
            process_variance,
            expected_process_variance,
            atol=0.0,
            rtol=0.0,
        ):
            raise ValueError("component process variance differs from config")
        component_mean, mean, marginal = _derived_field_moments(
            basis,
            coefficient_mean,
            coefficient_covariance,
            local_variance,
            weights,
        )
        for field_name, array_value in (
            ("spatial_basis", basis),
            ("component_coefficient_mean_m", coefficient_mean),
            ("component_coefficient_covariance_m2", coefficient_covariance),
            ("component_local_variance_m2", local_variance),
            ("component_weights", weights),
            ("component_process_variance_m2", process_variance),
            ("component_mean_m", component_mean),
            ("mean_m", mean),
            ("marginal_covariance_m2", marginal),
        ):
            object.__setattr__(self, field_name, _readonly(array_value))


StructuredDiscrepancyBeliefV1: TypeAlias = (
    StructuredDiscrepancyPosteriorV1 | StructuredDiscrepancyPredictionV1
)


@dataclass(frozen=True, slots=True)
class StructuredDiscrepancyQueryMomentsV1:
    """Moments of an arbitrary linear query of the discrepancy field."""

    mean: np.ndarray
    covariance: np.ndarray

    def __post_init__(self) -> None:
        mean = np.asarray(self.mean, dtype=np.float64)
        covariance = np.asarray(self.covariance, dtype=np.float64)
        if mean.ndim != 1 or len(mean) < 1:
            raise ValueError("query mean must be a nonempty vector")
        if covariance.shape != (len(mean), len(mean)):
            raise ValueError("query covariance shape changed")
        if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(covariance)):
            raise ValueError("query moments must contain only finite values")
        if not np.allclose(covariance, covariance.T, atol=1e-10, rtol=1e-10):
            raise ValueError("query covariance must be symmetric")
        if np.min(np.linalg.eigvalsh(covariance), initial=0.0) < -1e-10:
            raise ValueError("query covariance must be positive semidefinite")
        object.__setattr__(self, "mean", _readonly(mean))
        object.__setattr__(self, "covariance", _readonly(covariance))


__all__ = [
    "STRUCTURED_DISCREPANCY_CLAIM_BOUNDARY",
    "STRUCTURED_DISCREPANCY_CONTRACT_VERSION",
    "STRUCTURED_DISCREPANCY_COVARIANCE_SEMANTICS",
    "STRUCTURED_DISCREPANCY_EVIDENCE_SEMANTICS",
    "StructuredDiscrepancyBeliefV1",
    "StructuredDiscrepancyConfigV1",
    "StructuredDiscrepancyPosteriorV1",
    "StructuredDiscrepancyPredictionV1",
    "StructuredDiscrepancyQueryMomentsV1",
]
