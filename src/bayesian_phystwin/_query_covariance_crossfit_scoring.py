"""Proper-score and sharpness diagnostics for structured query covariance."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ._canonical_contracts import genuine_integer
from ._query_covariance_crossfit_common import (
    StructuredQueryCovarianceTransformV1,
    _cholesky,
    _finite_real,
    _numeric_array,
    _query_group_arrays,
    apply_structured_query_covariance,
)


@dataclass(frozen=True, slots=True)
class QueryCovarianceGroupDiagnosticsV1:
    """Proper-score, coverage, sharpness, and conditioning diagnostics."""

    endpoint_count: int
    squared_ellipsoid_radius: float
    mean_gaussian_nll: float
    mean_mahalanobis_squared: float
    ellipsoid_coverage: float
    mean_log_sqrt_determinant: float
    mean_effective_rank: float
    maximum_condition_number: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "endpoint_count",
            genuine_integer(self.endpoint_count, name="endpoint_count", minimum=1),
        )
        object.__setattr__(
            self,
            "squared_ellipsoid_radius",
            _finite_real(
                self.squared_ellipsoid_radius,
                name="squared_ellipsoid_radius",
                minimum=0.0,
            ),
        )
        for name in (
            "mean_gaussian_nll",
            "mean_mahalanobis_squared",
            "mean_log_sqrt_determinant",
            "mean_effective_rank",
            "maximum_condition_number",
        ):
            object.__setattr__(
                self,
                name,
                _finite_real(getattr(self, name), name=name),
            )
        coverage = _finite_real(
            self.ellipsoid_coverage,
            name="ellipsoid_coverage",
            minimum=0.0,
            maximum=1.0,
        )
        object.__setattr__(self, "ellipsoid_coverage", coverage)


def score_query_covariance_group(
    residual: object,
    covariance: object,
    transform: StructuredQueryCovarianceTransformV1,
    *,
    squared_ellipsoid_radius: float,
) -> QueryCovarianceGroupDiagnosticsV1:
    """Score one independent group without treating endpoints as independent groups."""

    if not isinstance(transform, StructuredQueryCovarianceTransformV1):
        raise TypeError("transform must be StructuredQueryCovarianceTransformV1")
    radius = _finite_real(
        squared_ellipsoid_radius,
        name="squared_ellipsoid_radius",
        minimum=0.0,
    )
    errors, raw_covariance = _query_group_arrays(
        residual,
        covariance,
        name="query group",
        dimension=transform.dimension,
    )
    transformed = apply_structured_query_covariance(raw_covariance, transform)
    nll: list[float] = []
    mahalanobis: list[float] = []
    log_sqrt_determinant: list[float] = []
    effective_rank: list[float] = []
    condition_number: list[float] = []
    constant = transform.dimension * np.log(2.0 * np.pi)
    for index, (error, matrix) in enumerate(
        zip(errors, transformed, strict=True)
    ):
        factor = _cholesky(matrix, name=f"transformed covariance {index}")
        whitened = np.linalg.solve(factor, error)
        squared = float(whitened @ whitened)
        log_det = 2.0 * float(np.sum(np.log(np.diag(factor))))
        nll.append(0.5 * (constant + log_det + squared))
        mahalanobis.append(squared)
        log_sqrt_determinant.append(0.5 * log_det)
        eigenvalues = np.linalg.eigvalsh(matrix)
        normalized = eigenvalues / float(np.sum(eigenvalues))
        entropy = -float(np.sum(normalized * np.log(normalized)))
        effective_rank.append(float(np.exp(entropy)))
        condition_number.append(float(eigenvalues[-1] / eigenvalues[0]))
    mahalanobis_array = np.asarray(mahalanobis)
    return QueryCovarianceGroupDiagnosticsV1(
        endpoint_count=len(errors),
        squared_ellipsoid_radius=radius,
        mean_gaussian_nll=float(np.mean(nll)),
        mean_mahalanobis_squared=float(np.mean(mahalanobis_array)),
        ellipsoid_coverage=float(np.mean(mahalanobis_array <= radius)),
        mean_log_sqrt_determinant=float(np.mean(log_sqrt_determinant)),
        mean_effective_rank=float(np.mean(effective_rank)),
        maximum_condition_number=float(np.max(condition_number)),
    )


def group_gaussian_energy_score(
    residual: object,
    covariance: object,
    transform: StructuredQueryCovarianceTransformV1,
    *,
    standard_normal_sample_pairs: object,
) -> float:
    """Return a deterministic paired Monte Carlo Gaussian energy score.

    ``standard_normal_sample_pairs`` must have shape ``(2, s>=1, d)`` and be
    frozen independently of the scored outcomes.  Axis zero supplies independent
    sample sets for the two expectations in the energy-score estimator.
    """

    if not isinstance(transform, StructuredQueryCovarianceTransformV1):
        raise TypeError("transform must be StructuredQueryCovarianceTransformV1")
    errors, raw_covariance = _query_group_arrays(
        residual,
        covariance,
        name="query group",
        dimension=transform.dimension,
    )
    samples = _numeric_array(
        standard_normal_sample_pairs,
        name="standard_normal_sample_pairs",
    )
    if (
        samples.ndim != 3
        or samples.shape[0] != 2
        or samples.shape[1] < 1
        or samples.shape[2] != transform.dimension
    ):
        raise ValueError(
            "standard_normal_sample_pairs must have shape (2, s>=1, dimension)"
        )
    transformed = apply_structured_query_covariance(raw_covariance, transform)
    scores: list[float] = []
    for index, (error, matrix) in enumerate(
        zip(errors, transformed, strict=True)
    ):
        factor = _cholesky(matrix, name=f"transformed covariance {index}")
        first = samples[0] @ factor.T
        second = samples[1] @ factor.T
        score = float(
            np.mean(np.linalg.norm(first - error[None, :], axis=1))
            - 0.5 * np.mean(np.linalg.norm(first - second, axis=1))
        )
        if not np.isfinite(score):
            raise ValueError("Gaussian energy score must be finite")
        scores.append(score)
    return float(np.mean(scores))


__all__ = [
    "QueryCovarianceGroupDiagnosticsV1",
    "group_gaussian_energy_score",
    "score_query_covariance_group",
]
