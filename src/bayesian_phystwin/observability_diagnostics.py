"""Nuisance-marginalized observability diagnostics for physical queries.

The functions compare already constructed Gaussian information states after all
declared nuisance variables have been marginalized. An optional full-row-rank
query Jacobian restricts the report to the physical quantity that will actually
be deployed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


class MarginalInformationState(Protocol):
    """Structural interface required by the diagnostics."""

    def marginal_state_precision(self) -> np.ndarray:
        """Return the nuisance-marginalized physical-state precision."""
        ...


def _require(condition: bool | np.bool_, message: str) -> None:
    if not bool(condition):
        raise ValueError(message)


def _positive_scalar(value: object, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a finite positive number")
    raw = np.asarray(value)
    if raw.shape != () or raw.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be a finite positive number")
    scalar = float(raw.item())
    _require(
        np.isfinite(scalar) and scalar > 0.0,
        f"{name} must be a finite positive number",
    )
    return scalar


def _matrix(value: object, *, name: str) -> np.ndarray:
    raw = np.asarray(value)
    _require(raw.dtype.kind in "iuf", f"{name} must contain real numbers")
    result = np.array(raw, dtype=np.float64, copy=True, order="C")
    _require(result.ndim == 2, f"{name} must be a matrix")
    _require(np.all(np.isfinite(result)), f"{name} must be finite")
    return result


def _immutable_array(value: object) -> np.ndarray:
    canonical = np.array(
        value,
        dtype=np.dtype("<f8"),
        copy=True,
        order="C",
    )
    frozen = np.frombuffer(
        canonical.tobytes(order="C"),
        dtype=np.dtype("<f8"),
    )
    return frozen.reshape(canonical.shape)


def _spd(value: object, *, name: str) -> np.ndarray:
    result = _matrix(value, name=name)
    _require(result.shape[0] == result.shape[1], f"{name} must be square")
    _require(result.shape[0] > 0, f"{name} must be nonempty")
    _require(
        np.allclose(result, result.T, rtol=1e-10, atol=1e-12),
        f"{name} must be symmetric",
    )
    result = 0.5 * (result + result.T)
    try:
        np.linalg.cholesky(result)
    except np.linalg.LinAlgError as error:
        raise ValueError(f"{name} must be positive definite") from error
    return result


def _inverse_spd(value: np.ndarray, *, name: str) -> np.ndarray:
    try:
        cholesky = np.linalg.cholesky(value)
        identity = np.eye(value.shape[0], dtype=np.float64)
        lower = np.linalg.solve(cholesky, identity)
        result = np.linalg.solve(cholesky.T, lower)
    except np.linalg.LinAlgError as error:
        raise ValueError(f"{name} is not numerically positive definite") from error
    result = 0.5 * (result + result.T)
    _require(np.all(np.isfinite(result)), f"{name} inverse is non-finite")
    return result


def _query(
    value: object | None,
    *,
    state_dimension: int,
    relative_tolerance: float,
    absolute_tolerance: float,
) -> np.ndarray:
    result = (
        np.eye(state_dimension, dtype=np.float64)
        if value is None
        else _matrix(value, name="query_jacobian")
    )
    _require(result.shape[1] == state_dimension, "query state dimension changed")
    _require(result.shape[0] > 0, "query_jacobian must have at least one row")
    singular_values = np.linalg.svd(result, compute_uv=False)
    threshold = max(
        absolute_tolerance,
        relative_tolerance * float(singular_values[0]),
    )
    _require(
        int(np.sum(singular_values > threshold)) == result.shape[0],
        "query_jacobian rows must be numerically independent",
    )
    return _immutable_array(result)


def _query_moments(
    state: MarginalInformationState,
    *,
    query_jacobian: object | None,
    relative_tolerance: float,
    absolute_tolerance: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    state_precision = _spd(
        state.marginal_state_precision(),
        name="marginal_state_precision",
    )
    query = _query(
        query_jacobian,
        state_dimension=state_precision.shape[0],
        relative_tolerance=relative_tolerance,
        absolute_tolerance=absolute_tolerance,
    )
    query_covariance = (
        query
        @ _inverse_spd(
            state_precision,
            name="marginal_state_precision",
        )
        @ query.T
    )
    query_covariance = _spd(query_covariance, name="query_covariance")
    query_precision = _inverse_spd(query_covariance, name="query_covariance")
    return query, query_covariance, query_precision


@dataclass(frozen=True)
class MarginalObservabilitySummary:
    """Spectrum and uncertainty of one nuisance-marginalized query."""

    state_dimension: int
    query_dimension: int
    numerical_rank: int
    effective_rank: float
    precision_eigenvalues: np.ndarray
    marginal_variances: np.ndarray
    log_determinant_precision: float
    trace_precision: float
    condition_number: float
    weakest_direction_variance: float
    query_jacobian: np.ndarray

    def __post_init__(self) -> None:
        eigenvalues = np.asarray(self.precision_eigenvalues, dtype=np.float64).copy()
        variances = np.asarray(self.marginal_variances, dtype=np.float64).copy()
        query = np.asarray(self.query_jacobian, dtype=np.float64).copy()
        _require(
            eigenvalues.shape == (self.query_dimension,),
            "precision_eigenvalues shape changed",
        )
        _require(
            variances.shape == (self.query_dimension,),
            "marginal_variances shape changed",
        )
        _require(
            query.shape == (self.query_dimension, self.state_dimension),
            "query_jacobian shape changed",
        )
        _require(
            np.all(np.isfinite(eigenvalues)) and np.all(eigenvalues > 0.0),
            "precision_eigenvalues must be finite and positive",
        )
        _require(
            np.all(np.isfinite(variances)) and np.all(variances > 0.0),
            "marginal_variances must be finite and positive",
        )
        _require(
            0 <= self.numerical_rank <= self.query_dimension,
            "numerical_rank is outside the query dimension",
        )
        scalars = (
            self.effective_rank,
            self.log_determinant_precision,
            self.trace_precision,
            self.condition_number,
            self.weakest_direction_variance,
        )
        _require(
            all(np.isfinite(value) for value in scalars),
            "observability summary contains a non-finite scalar",
        )
        _require(
            1.0 - 1e-12 <= self.effective_rank <= self.query_dimension + 1e-12,
            "effective_rank is outside the query dimension",
        )
        _require(self.trace_precision > 0.0, "trace_precision must be positive")
        _require(
            self.condition_number >= 1.0 - 1e-12,
            "condition_number must be at least one",
        )
        _require(
            self.weakest_direction_variance > 0.0,
            "weakest_direction_variance must be positive",
        )
        for name, array in (
            ("precision_eigenvalues", eigenvalues),
            ("marginal_variances", variances),
            ("query_jacobian", query),
        ):
            object.__setattr__(self, name, _immutable_array(array))

    @property
    def minimum_precision_eigenvalue(self) -> float:
        return float(self.precision_eigenvalues[0])

    @property
    def maximum_precision_eigenvalue(self) -> float:
        return float(self.precision_eigenvalues[-1])

    def to_record(self) -> dict[str, object]:
        """Return a JSON-compatible diagnostic record."""

        return {
            "state_dimension": self.state_dimension,
            "query_dimension": self.query_dimension,
            "numerical_rank": self.numerical_rank,
            "effective_rank": self.effective_rank,
            "precision_eigenvalues": self.precision_eigenvalues.tolist(),
            "marginal_variances": self.marginal_variances.tolist(),
            "log_determinant_precision": self.log_determinant_precision,
            "trace_precision": self.trace_precision,
            "minimum_precision_eigenvalue": self.minimum_precision_eigenvalue,
            "maximum_precision_eigenvalue": self.maximum_precision_eigenvalue,
            "condition_number": self.condition_number,
            "weakest_direction_variance": self.weakest_direction_variance,
            "query_jacobian": self.query_jacobian.tolist(),
        }


@dataclass(frozen=True)
class MarginalObservabilityComparison:
    """Incremental query information from a candidate observation family."""

    reference: MarginalObservabilitySummary
    candidate: MarginalObservabilitySummary
    information_increment_eigenvalues: np.ndarray
    log_determinant_gain: float
    mutual_information_gain_nats: float
    trace_precision_gain: float
    weakest_direction_precision_ratio: float
    mean_variance_reduction_fraction: float
    maximum_variance_reduction_fraction: float
    numerical_rank_gain: int
    effective_rank_gain: float

    def __post_init__(self) -> None:
        _require(
            self.reference.state_dimension == self.candidate.state_dimension
            and self.reference.query_dimension == self.candidate.query_dimension,
            "comparison dimensions differ",
        )
        _require(
            np.array_equal(
                self.reference.query_jacobian,
                self.candidate.query_jacobian,
            ),
            "comparison query Jacobians differ",
        )
        increments = np.asarray(
            self.information_increment_eigenvalues,
            dtype=np.float64,
        ).copy()
        _require(
            increments.shape == (self.reference.query_dimension,),
            "information_increment_eigenvalues shape changed",
        )
        _require(
            np.all(np.isfinite(increments)) and np.all(increments >= 0.0),
            "information_increment_eigenvalues must be nonnegative",
        )
        _require(
            np.isclose(
                self.mutual_information_gain_nats,
                0.5 * self.log_determinant_gain,
            ),
            "mutual information and log-determinant gain differ",
        )
        _require(
            self.log_determinant_gain >= 0.0
            and self.trace_precision_gain >= 0.0
            and self.weakest_direction_precision_ratio >= 1.0,
            "candidate information gain must be nonnegative",
        )
        for value, name in (
            (
                self.mean_variance_reduction_fraction,
                "mean_variance_reduction_fraction",
            ),
            (
                self.maximum_variance_reduction_fraction,
                "maximum_variance_reduction_fraction",
            ),
        ):
            _require(0.0 <= value <= 1.0, f"{name} must lie in [0, 1]")
        object.__setattr__(
            self,
            "information_increment_eigenvalues",
            _immutable_array(increments),
        )

    def to_record(self) -> dict[str, object]:
        """Return a JSON-compatible comparison record."""

        return {
            "reference": self.reference.to_record(),
            "candidate": self.candidate.to_record(),
            "information_increment_eigenvalues": (
                self.information_increment_eigenvalues.tolist()
            ),
            "log_determinant_gain": self.log_determinant_gain,
            "mutual_information_gain_nats": self.mutual_information_gain_nats,
            "trace_precision_gain": self.trace_precision_gain,
            "weakest_direction_precision_ratio": (
                self.weakest_direction_precision_ratio
            ),
            "mean_variance_reduction_fraction": (
                self.mean_variance_reduction_fraction
            ),
            "maximum_variance_reduction_fraction": (
                self.maximum_variance_reduction_fraction
            ),
            "numerical_rank_gain": self.numerical_rank_gain,
            "effective_rank_gain": self.effective_rank_gain,
        }


def summarize_marginal_observability(
    state: MarginalInformationState,
    *,
    query_jacobian: object | None = None,
    relative_rank_tolerance: object = 1e-9,
    absolute_rank_tolerance: object = 1e-12,
) -> MarginalObservabilitySummary:
    """Summarize nuisance-marginalized information for one physical query."""

    relative_tolerance = _positive_scalar(
        relative_rank_tolerance,
        name="relative_rank_tolerance",
    )
    absolute_tolerance = _positive_scalar(
        absolute_rank_tolerance,
        name="absolute_rank_tolerance",
    )
    query, covariance, precision = _query_moments(
        state,
        query_jacobian=query_jacobian,
        relative_tolerance=relative_tolerance,
        absolute_tolerance=absolute_tolerance,
    )
    eigenvalues = np.linalg.eigvalsh(precision)
    _require(
        np.all(np.isfinite(eigenvalues)) and np.all(eigenvalues > 0.0),
        "query precision spectrum must be finite and positive",
    )
    threshold = max(
        absolute_tolerance,
        relative_tolerance * float(eigenvalues[-1]),
    )
    weights = eigenvalues / np.sum(eigenvalues)
    effective_rank = float(np.exp(-np.sum(weights * np.log(weights))))
    minimum = float(eigenvalues[0])
    maximum = float(eigenvalues[-1])
    return MarginalObservabilitySummary(
        state_dimension=query.shape[1],
        query_dimension=query.shape[0],
        numerical_rank=int(np.sum(eigenvalues > threshold)),
        effective_rank=effective_rank,
        precision_eigenvalues=eigenvalues,
        marginal_variances=np.diag(covariance),
        log_determinant_precision=float(np.sum(np.log(eigenvalues))),
        trace_precision=float(np.sum(eigenvalues)),
        condition_number=float(maximum / minimum),
        weakest_direction_variance=float(1.0 / minimum),
        query_jacobian=query,
    )


def compare_marginal_observability(
    reference_state: MarginalInformationState,
    candidate_state: MarginalInformationState,
    *,
    query_jacobian: object | None = None,
    relative_rank_tolerance: object = 1e-9,
    absolute_rank_tolerance: object = 1e-12,
) -> MarginalObservabilityComparison:
    """Compare a candidate observation family with a frozen reference state.

    The comparison fails closed if the candidate reduces query information
    beyond the declared tolerance. This normally indicates mismatched priors,
    nuisance domains, query definitions, or evidence order.
    """

    relative_tolerance = _positive_scalar(
        relative_rank_tolerance,
        name="relative_rank_tolerance",
    )
    absolute_tolerance = _positive_scalar(
        absolute_rank_tolerance,
        name="absolute_rank_tolerance",
    )
    reference = summarize_marginal_observability(
        reference_state,
        query_jacobian=query_jacobian,
        relative_rank_tolerance=relative_tolerance,
        absolute_rank_tolerance=absolute_tolerance,
    )
    candidate = summarize_marginal_observability(
        candidate_state,
        query_jacobian=query_jacobian,
        relative_rank_tolerance=relative_tolerance,
        absolute_rank_tolerance=absolute_tolerance,
    )
    _, _, reference_precision = _query_moments(
        reference_state,
        query_jacobian=reference.query_jacobian,
        relative_tolerance=relative_tolerance,
        absolute_tolerance=absolute_tolerance,
    )
    _, _, candidate_precision = _query_moments(
        candidate_state,
        query_jacobian=reference.query_jacobian,
        relative_tolerance=relative_tolerance,
        absolute_tolerance=absolute_tolerance,
    )
    increment = candidate_precision - reference_precision
    increment = 0.5 * (increment + increment.T)
    increment_eigenvalues = np.linalg.eigvalsh(increment)
    scale = max(
        1.0,
        float(np.max(np.abs(reference_precision))),
        float(np.max(np.abs(candidate_precision))),
    )
    tolerance = absolute_tolerance + relative_tolerance * scale
    unitless_tolerance = absolute_tolerance + relative_tolerance
    _require(
        float(increment_eigenvalues[0]) >= -tolerance,
        "candidate nuisance-marginalized information is lower than the reference",
    )
    increment_eigenvalues = np.maximum(increment_eigenvalues, 0.0)

    log_gain = candidate.log_determinant_precision - reference.log_determinant_precision
    trace_gain = candidate.trace_precision - reference.trace_precision
    _require(
        log_gain >= -unitless_tolerance and trace_gain >= -tolerance,
        "candidate aggregate information is lower than the reference",
    )
    log_gain = max(log_gain, 0.0)
    trace_gain = max(trace_gain, 0.0)

    variance_reduction = 1.0 - (
        candidate.marginal_variances / reference.marginal_variances
    )
    _require(
        float(np.min(variance_reduction)) >= -unitless_tolerance,
        "candidate marginal query variance is larger than the reference",
    )
    variance_reduction = np.clip(variance_reduction, 0.0, 1.0)
    weakest_ratio = (
        candidate.minimum_precision_eigenvalue
        / reference.minimum_precision_eigenvalue
    )
    if weakest_ratio < 1.0 and weakest_ratio >= 1.0 - unitless_tolerance:
        weakest_ratio = 1.0
    _require(
        weakest_ratio >= 1.0,
        "candidate weakest-direction precision is lower than the reference",
    )

    return MarginalObservabilityComparison(
        reference=reference,
        candidate=candidate,
        information_increment_eigenvalues=increment_eigenvalues,
        log_determinant_gain=log_gain,
        mutual_information_gain_nats=0.5 * log_gain,
        trace_precision_gain=trace_gain,
        weakest_direction_precision_ratio=float(weakest_ratio),
        mean_variance_reduction_fraction=float(np.mean(variance_reduction)),
        maximum_variance_reduction_fraction=float(np.max(variance_reduction)),
        numerical_rank_gain=(candidate.numerical_rank - reference.numerical_rank),
        effective_rank_gain=float(candidate.effective_rank - reference.effective_rank),
    )


__all__ = [
    "MarginalInformationState",
    "MarginalObservabilityComparison",
    "MarginalObservabilitySummary",
    "compare_marginal_observability",
    "summarize_marginal_observability",
]
