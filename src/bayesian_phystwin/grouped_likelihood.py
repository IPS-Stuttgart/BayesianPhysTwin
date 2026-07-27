"""Correlation-aware robust likelihood for observation-belief artifacts."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .observation_belief import ObservationBeliefV1


@dataclass(frozen=True)
class GroupedStudentTLikelihoodConfig:
    """Settings for a nominal/broad multivariate Student-t mixture."""

    degrees_of_freedom: float = 5.0
    outlier_covariance_multiplier: float = 25.0
    model_discrepancy_variance_m2: float = 0.0
    probability_floor: float = 1e-6
    covariance_jitter_m2: float = 1e-12

    def __post_init__(self) -> None:
        if not np.isfinite(self.degrees_of_freedom) or (self.degrees_of_freedom <= 2.0):
            raise ValueError("degrees_of_freedom must exceed two")
        if not np.isfinite(self.outlier_covariance_multiplier) or (
            self.outlier_covariance_multiplier <= 1.0
        ):
            raise ValueError("outlier_covariance_multiplier must exceed one")
        if not np.isfinite(self.model_discrepancy_variance_m2) or (
            self.model_discrepancy_variance_m2 < 0.0
        ):
            raise ValueError("model_discrepancy_variance_m2 must be nonnegative")
        if not 0.0 < self.probability_floor < 0.5:
            raise ValueError("probability_floor must lie in (0, 0.5)")
        if not np.isfinite(self.covariance_jitter_m2) or (
            self.covariance_jitter_m2 <= 0.0
        ):
            raise ValueError("covariance_jitter_m2 must be positive")


@dataclass(frozen=True)
class GroupedStudentTLikelihoodResult:
    """Per-group robust evidence and posterior nominal responsibilities."""

    group_ids: np.ndarray
    dimensions: np.ndarray
    negative_log_likelihood: np.ndarray
    weighted_negative_log_likelihood: np.ndarray
    posterior_nominal_probability: np.ndarray
    prior_nominal_probability: np.ndarray
    composite_weight: np.ndarray
    log_nominal_density: np.ndarray
    log_outlier_density: np.ndarray
    mean_association_probability: np.ndarray
    covariance_log_determinant_m2: np.ndarray
    covariance_mahalanobis_squared: np.ndarray

    def __post_init__(self) -> None:
        group_ids = np.asarray(self.group_ids, dtype=np.int64).copy()
        dimensions = np.asarray(self.dimensions, dtype=np.int64).copy()
        arrays = {
            "negative_log_likelihood": self.negative_log_likelihood,
            "weighted_negative_log_likelihood": (self.weighted_negative_log_likelihood),
            "posterior_nominal_probability": (self.posterior_nominal_probability),
            "prior_nominal_probability": self.prior_nominal_probability,
            "composite_weight": self.composite_weight,
            "log_nominal_density": self.log_nominal_density,
            "log_outlier_density": self.log_outlier_density,
            "mean_association_probability": (self.mean_association_probability),
            "covariance_log_determinant_m2": (self.covariance_log_determinant_m2),
            "covariance_mahalanobis_squared": (self.covariance_mahalanobis_squared),
        }
        count = len(group_ids)
        if dimensions.shape != (count,) or np.any(dimensions < 1):
            raise ValueError("dimensions must identify every nonempty group")
        group_ids.setflags(write=False)
        dimensions.setflags(write=False)
        object.__setattr__(self, "group_ids", group_ids)
        object.__setattr__(self, "dimensions", dimensions)
        for name, values in arrays.items():
            array = np.asarray(values, dtype=np.float64).copy()
            if array.shape != (count,) or not np.all(np.isfinite(array)):
                raise ValueError(f"{name} must be a finite group vector")
            array.setflags(write=False)
            object.__setattr__(self, name, array)

    @property
    def total_negative_log_likelihood(self) -> float:
        return float(np.sum(self.weighted_negative_log_likelihood))

    @property
    def mean_posterior_nominal_probability(self) -> float:
        weights = self.composite_weight
        return float(
            np.sum(weights * self.posterior_nominal_probability) / np.sum(weights)
        )


def _covariance_statistics(
    residual: np.ndarray,
    local_covariance: np.ndarray,
    low_rank_factor: np.ndarray,
    factor_group_ids: np.ndarray,
    *,
    model_discrepancy_variance_m2: float,
    covariance_jitter_m2: float,
) -> tuple[float, float]:
    """Return log determinant and Mahalanobis square via Woodbury."""

    identity = np.eye(3, dtype=np.float64)
    blocks = np.asarray(local_covariance, dtype=np.float64).copy()
    blocks += (model_discrepancy_variance_m2 + covariance_jitter_m2) * identity
    try:
        cholesky = np.linalg.cholesky(blocks)
    except np.linalg.LinAlgError as error:
        raise ValueError("local covariance lost positive definiteness") from error

    whitened_residual = np.linalg.solve(
        cholesky,
        residual[..., None],
    )[..., 0]
    diagonal = np.diagonal(cholesky, axis1=1, axis2=2)
    log_determinant = float(2.0 * np.sum(np.log(diagonal)))
    mahalanobis = float(np.sum(np.square(whitened_residual)))

    rank = low_rank_factor.shape[2]
    if rank == 0 or not np.any(low_rank_factor):
        return log_determinant, max(mahalanobis, 0.0)

    correction = 0.0
    for factor_group in np.unique(factor_group_ids):
        selected = factor_group_ids == factor_group
        whitened_factor = np.linalg.solve(
            cholesky[selected],
            low_rank_factor[selected],
        )
        gram = np.eye(rank, dtype=np.float64) + np.einsum(
            "ncr,ncs->rs",
            whitened_factor,
            whitened_factor,
        )
        try:
            gram_cholesky = np.linalg.cholesky(gram)
        except np.linalg.LinAlgError as error:
            raise ValueError(
                "low-rank covariance update is not positive definite"
            ) from error
        log_determinant += float(2.0 * np.sum(np.log(np.diag(gram_cholesky))))
        projection = np.einsum(
            "ncr,nc->r",
            whitened_factor,
            whitened_residual[selected],
        )
        whitened_projection = np.linalg.solve(
            gram_cholesky,
            projection,
        )
        correction += float(np.sum(np.square(whitened_projection)))

    return (
        log_determinant,
        max(mahalanobis - correction, 0.0),
    )


def _student_t_log_density_from_statistics(
    *,
    dimension: int,
    degrees_of_freedom: float,
    covariance_log_determinant: float,
    covariance_mahalanobis_squared: float,
    covariance_multiplier: float,
) -> float:
    """Log density when the requested covariance is a scalar multiple of C."""

    # For nu > 2, Psi = (nu - 2) / nu * C gives covariance C.
    scale_multiplier = (
        (degrees_of_freedom - 2.0) / degrees_of_freedom * covariance_multiplier
    )
    log_scale_determinant = covariance_log_determinant + dimension * math.log(
        scale_multiplier
    )
    quadratic = covariance_mahalanobis_squared / scale_multiplier
    return (
        math.lgamma(0.5 * (degrees_of_freedom + dimension))
        - math.lgamma(0.5 * degrees_of_freedom)
        - 0.5
        * (dimension * math.log(degrees_of_freedom * math.pi) + log_scale_determinant)
        - 0.5
        * (degrees_of_freedom + dimension)
        * math.log1p(quadratic / degrees_of_freedom)
    )


def grouped_student_t_mixture_likelihood(
    belief: ObservationBeliefV1,
    predicted_xyz_m: np.ndarray,
    *,
    config: GroupedStudentTLikelihoodConfig | None = None,
) -> GroupedStudentTLikelihoodResult:
    """Evaluate the paper's grouped robust factor without double residual use.

    The prior nominal probability comes directly from the artifact and is never
    recomputed from the innovation. Association support is reported separately
    and does not become an additional reliability factor.
    """

    settings = config or GroupedStudentTLikelihoodConfig()
    predicted = np.asarray(predicted_xyz_m, dtype=np.float64)
    if predicted.shape != belief.mean_xyz_m.shape:
        raise ValueError("predicted_xyz_m must match the observation mean shape")
    if not np.all(np.isfinite(predicted)):
        raise ValueError("predicted_xyz_m must be finite")

    group_count = len(belief.group_ids)
    dimension: NDArray[np.int64] = np.empty(group_count, dtype=np.int64)
    nll: NDArray[np.float64] = np.empty(group_count, dtype=np.float64)
    weighted_nll: NDArray[np.float64] = np.empty(group_count, dtype=np.float64)
    posterior_nominal: NDArray[np.float64] = np.empty(
        group_count,
        dtype=np.float64,
    )
    log_nominal: NDArray[np.float64] = np.empty(group_count, dtype=np.float64)
    log_outlier: NDArray[np.float64] = np.empty(group_count, dtype=np.float64)
    mean_association: NDArray[np.float64] = np.empty(
        group_count,
        dtype=np.float64,
    )
    covariance_logdet: NDArray[np.float64] = np.empty(
        group_count,
        dtype=np.float64,
    )
    covariance_mahalanobis: NDArray[np.float64] = np.empty(
        group_count,
        dtype=np.float64,
    )

    residual_all = belief.mean_xyz_m - predicted
    prior = np.clip(
        belief.group_prior_nominal_probability,
        settings.probability_floor,
        1.0 - settings.probability_floor,
    )
    for position, group_id in enumerate(belief.group_ids):
        selected = belief.correlation_group_ids == group_id
        residual = residual_all[selected]
        covariance_logdet[position], covariance_mahalanobis[position] = (
            _covariance_statistics(
                residual,
                belief.local_covariance_m2[selected],
                belief.low_rank_factor_m[selected],
                belief.factor_group_ids[selected],
                model_discrepancy_variance_m2=(settings.model_discrepancy_variance_m2),
                covariance_jitter_m2=settings.covariance_jitter_m2,
            )
        )
        dimension[position] = residual.size
        log_nominal[position] = _student_t_log_density_from_statistics(
            dimension=int(dimension[position]),
            degrees_of_freedom=settings.degrees_of_freedom,
            covariance_log_determinant=float(covariance_logdet[position]),
            covariance_mahalanobis_squared=float(covariance_mahalanobis[position]),
            covariance_multiplier=1.0,
        )
        log_outlier[position] = _student_t_log_density_from_statistics(
            dimension=int(dimension[position]),
            degrees_of_freedom=settings.degrees_of_freedom,
            covariance_log_determinant=float(covariance_logdet[position]),
            covariance_mahalanobis_squared=float(covariance_mahalanobis[position]),
            covariance_multiplier=(settings.outlier_covariance_multiplier),
        )
        log_nominal_component = math.log(prior[position]) + (log_nominal[position])
        log_outlier_component = math.log1p(-prior[position]) + (log_outlier[position])
        log_mixture = float(np.logaddexp(log_nominal_component, log_outlier_component))
        nll[position] = -log_mixture
        weighted_nll[position] = belief.group_composite_weight[position] * nll[position]
        posterior_nominal[position] = math.exp(log_nominal_component - log_mixture)
        mean_association[position] = float(
            np.mean(belief.association_probability[selected])
        )

    return GroupedStudentTLikelihoodResult(
        group_ids=belief.group_ids,
        dimensions=dimension,
        negative_log_likelihood=nll,
        weighted_negative_log_likelihood=weighted_nll,
        posterior_nominal_probability=posterior_nominal,
        prior_nominal_probability=prior,
        composite_weight=belief.group_composite_weight,
        log_nominal_density=log_nominal,
        log_outlier_density=log_outlier,
        mean_association_probability=mean_association,
        covariance_log_determinant_m2=covariance_logdet,
        covariance_mahalanobis_squared=covariance_mahalanobis,
    )


__all__ = [
    "GroupedStudentTLikelihoodConfig",
    "GroupedStudentTLikelihoodResult",
    "grouped_student_t_mixture_likelihood",
]
