"""Correlation-aware robust scores for observation-belief artifacts."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ._gauge_aware_contracts import (
    COMPOSITE_WEIGHT_MODE_CONSUMER_CAP,
    COMPOSITE_WEIGHT_MODE_PROVIDER_FINAL,
)
from ._prior_aware_gauge_math import (
    PriorAwareGaugeConfigV1,
    _student_t_mixture_statistics,
)
from .observation_belief import ObservationBeliefV1

COVARIANCE_MARGINAL_SCORE_SEMANTICS = (
    "covariance-marginalized-student-t-score-v1"
)
CONDITIONAL_GROUP_OBJECTIVE_SEMANTICS = (
    "conditional-reliability-weighted-student-t-objective-v1"
)
PROB4D_FINAL_COMPOSITE_WEIGHT_SEMANTICS = "final-per-row-effective-sample-cap-v1"
_PROB4D_REPOSITORY_IDENTITIES = frozenset(
    {"FlorianPfaff/Prob4D", "IPS-Stuttgart/Prob4D"}
)
_COMPOSITE_WEIGHT_MODES = frozenset(
    {
        COMPOSITE_WEIGHT_MODE_CONSUMER_CAP,
        COMPOSITE_WEIGHT_MODE_PROVIDER_FINAL,
    }
)


@dataclass(frozen=True)
class GroupedStudentTLikelihoodConfig:
    """Settings for the covariance-marginalized robust diagnostic score."""

    degrees_of_freedom: float = 5.0
    outlier_covariance_multiplier: float = 25.0
    model_discrepancy_variance_m2: float = 0.0
    probability_floor: float = 1e-6
    covariance_jitter_m2: float = 1e-12

    def __post_init__(self) -> None:
        if not np.isfinite(self.degrees_of_freedom) or self.degrees_of_freedom <= 2.0:
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
class ConditionalGroupedStudentTObjectiveConfig:
    """Settings for the exact grouped objective used by prior-aware inference."""

    degrees_of_freedom: float = 5.0
    outlier_covariance_multiplier: float = 25.0
    probability_floor: float = 1e-6
    effective_samples_per_correlation_group: float = 64.0
    composite_weight_mode: str | None = None

    def __post_init__(self) -> None:
        if not np.isfinite(self.degrees_of_freedom) or self.degrees_of_freedom <= 2.0:
            raise ValueError("degrees_of_freedom must exceed two")
        if not np.isfinite(self.outlier_covariance_multiplier) or (
            self.outlier_covariance_multiplier <= 1.0
        ):
            raise ValueError("outlier_covariance_multiplier must exceed one")
        if not 0.0 < self.probability_floor < 0.5:
            raise ValueError("probability_floor must lie in (0, 0.5)")
        if not np.isfinite(self.effective_samples_per_correlation_group) or (
            self.effective_samples_per_correlation_group <= 0.0
        ):
            raise ValueError(
                "effective_samples_per_correlation_group must be positive"
            )
        if (
            self.composite_weight_mode is not None
            and self.composite_weight_mode not in _COMPOSITE_WEIGHT_MODES
        ):
            raise ValueError(
                "composite_weight_mode must be a supported information-power mode"
            )


@dataclass(frozen=True)
class GroupedStudentTLikelihoodResult:
    """Per-group covariance-marginalized score and mixture responsibilities."""

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
            "weighted_negative_log_likelihood": self.weighted_negative_log_likelihood,
            "posterior_nominal_probability": self.posterior_nominal_probability,
            "prior_nominal_probability": self.prior_nominal_probability,
            "composite_weight": self.composite_weight,
            "log_nominal_density": self.log_nominal_density,
            "log_outlier_density": self.log_outlier_density,
            "mean_association_probability": self.mean_association_probability,
            "covariance_log_determinant_m2": self.covariance_log_determinant_m2,
            "covariance_mahalanobis_squared": self.covariance_mahalanobis_squared,
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

    @property
    def semantics(self) -> str:
        """Machine-readable scientific interpretation of this score."""

        return COVARIANCE_MARGINAL_SCORE_SEMANTICS

    @property
    def prior_reliability_used(self) -> bool:
        """Whether row reliability enters this covariance-marginalized score."""

        return False


@dataclass(frozen=True)
class ConditionalGroupedStudentTObjectiveResult:
    """Exact conditional grouped objective used by prior-aware inference."""

    group_ids: np.ndarray
    active_dimensions: np.ndarray
    negative_log_objective: np.ndarray
    weighted_negative_log_objective: np.ndarray
    posterior_nominal_probability: np.ndarray
    prior_nominal_probability: np.ndarray
    group_power: np.ndarray
    log_nominal_density: np.ndarray
    log_outlier_density: np.ndarray
    mean_association_probability: np.ndarray
    reliability_weighted_mahalanobis_squared: np.ndarray
    composite_weight_mode: str

    def __post_init__(self) -> None:
        group_ids = np.asarray(self.group_ids, dtype=np.int64).copy()
        dimensions = np.asarray(self.active_dimensions, dtype=np.int64).copy()
        arrays = {
            "negative_log_objective": self.negative_log_objective,
            "weighted_negative_log_objective": self.weighted_negative_log_objective,
            "posterior_nominal_probability": self.posterior_nominal_probability,
            "prior_nominal_probability": self.prior_nominal_probability,
            "group_power": self.group_power,
            "log_nominal_density": self.log_nominal_density,
            "log_outlier_density": self.log_outlier_density,
            "mean_association_probability": self.mean_association_probability,
            "reliability_weighted_mahalanobis_squared": (
                self.reliability_weighted_mahalanobis_squared
            ),
        }
        count = len(group_ids)
        if dimensions.shape != (count,) or np.any(dimensions < 0):
            raise ValueError("active_dimensions must identify every group")
        if self.composite_weight_mode not in _COMPOSITE_WEIGHT_MODES:
            raise ValueError("unsupported composite_weight_mode")
        group_ids.setflags(write=False)
        dimensions.setflags(write=False)
        object.__setattr__(self, "group_ids", group_ids)
        object.__setattr__(self, "active_dimensions", dimensions)
        for name, values in arrays.items():
            array = np.asarray(values, dtype=np.float64).copy()
            if array.shape != (count,) or not np.all(np.isfinite(array)):
                raise ValueError(f"{name} must be a finite group vector")
            array.setflags(write=False)
            object.__setattr__(self, name, array)

    @property
    def total_negative_log_objective(self) -> float:
        return float(np.sum(self.weighted_negative_log_objective))

    @property
    def semantics(self) -> str:
        return CONDITIONAL_GROUP_OBJECTIVE_SEMANTICS

    @property
    def prior_reliability_used(self) -> bool:
        return True


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

    return log_determinant, max(mahalanobis - correction, 0.0)


def _mixture_config(
    *,
    degrees_of_freedom: float,
    outlier_covariance_multiplier: float,
    probability_floor: float,
) -> PriorAwareGaugeConfigV1:
    return PriorAwareGaugeConfigV1(
        degrees_of_freedom=degrees_of_freedom,
        outlier_covariance_multiplier=outlier_covariance_multiplier,
        probability_floor=probability_floor,
        minimum_robust_precision=0.0,
    )


def _resolved_composite_weight_mode(
    belief: ObservationBeliefV1,
    explicit_mode: str | None,
) -> str:
    if explicit_mode is not None:
        return explicit_mode
    semantics = belief.metadata.get("group_composite_weight_semantics")
    if semantics == PROB4D_FINAL_COMPOSITE_WEIGHT_SEMANTICS:
        return COMPOSITE_WEIGHT_MODE_PROVIDER_FINAL
    if belief.source_repository in _PROB4D_REPOSITORY_IDENTITIES:
        if semantics is not None:
            raise ValueError(
                f"unsupported Prob4D group_composite_weight_semantics {semantics!r}"
            )
        if "effective_samples_per_group" in belief.metadata:
            return COMPOSITE_WEIGHT_MODE_PROVIDER_FINAL
    return COMPOSITE_WEIGHT_MODE_CONSUMER_CAP


def _conditional_group_statistics(
    belief: ObservationBeliefV1,
    residual_all: np.ndarray,
    selected: np.ndarray,
) -> tuple[int, float, float]:
    active = selected & (belief.prior_reliability > 0.0)
    active_count = int(np.sum(active))
    if active_count == 0:
        return 0, 0.0, 0.0
    residual = residual_all[active]
    covariance = belief.local_covariance_m2[active]
    reliability = belief.prior_reliability[active]
    try:
        cholesky = np.linalg.cholesky(covariance)
    except np.linalg.LinAlgError as error:
        raise ValueError("local covariance lost positive definiteness") from error
    whitened = np.linalg.solve(cholesky, residual[..., None])[..., 0]
    squared_mahalanobis = float(
        np.sum(reliability * np.sum(np.square(whitened), axis=1))
    )
    mean_association = float(np.mean(belief.association_probability[active]))
    return 3 * active_count, squared_mahalanobis, mean_association


def grouped_student_t_mixture_likelihood(
    belief: ObservationBeliefV1,
    predicted_xyz_m: np.ndarray,
    *,
    config: GroupedStudentTLikelihoodConfig | None = None,
) -> GroupedStudentTLikelihoodResult:
    """Evaluate the frozen covariance-marginalized robust diagnostic.

    This score integrates the declared low-rank factors into a covariance through
    Woodbury identities. It intentionally preserves the historical behavior and
    does not use ``prior_reliability``. It is therefore not the conditional
    generalized-Bayes objective optimized by ``update_prior_aware_gauge_belief``.
    Use :func:`conditional_grouped_student_t_mixture_objective` for that objective.
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
    posterior_nominal: NDArray[np.float64] = np.empty(group_count, dtype=np.float64)
    log_nominal: NDArray[np.float64] = np.empty(group_count, dtype=np.float64)
    log_outlier: NDArray[np.float64] = np.empty(group_count, dtype=np.float64)
    mean_association: NDArray[np.float64] = np.empty(group_count, dtype=np.float64)
    covariance_logdet: NDArray[np.float64] = np.empty(group_count, dtype=np.float64)
    covariance_mahalanobis: NDArray[np.float64] = np.empty(
        group_count, dtype=np.float64
    )

    residual_all = belief.mean_xyz_m - predicted
    prior = np.clip(
        belief.group_prior_nominal_probability,
        settings.probability_floor,
        1.0 - settings.probability_floor,
    )
    mixture_config = _mixture_config(
        degrees_of_freedom=settings.degrees_of_freedom,
        outlier_covariance_multiplier=settings.outlier_covariance_multiplier,
        probability_floor=settings.probability_floor,
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
        statistics = _student_t_mixture_statistics(
            float(covariance_mahalanobis[position]),
            int(dimension[position]),
            float(prior[position]),
            mixture_config,
        )
        determinant_term = 0.5 * float(covariance_logdet[position])
        log_nominal[position] = statistics.log_nominal_density - determinant_term
        log_outlier[position] = statistics.log_outlier_density - determinant_term
        nll[position] = -(statistics.log_mixture_density - determinant_term)
        weighted_nll[position] = belief.group_composite_weight[position] * nll[position]
        posterior_nominal[position] = statistics.posterior_nominal_probability
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


def conditional_grouped_student_t_mixture_objective(
    belief: ObservationBeliefV1,
    conditional_prediction_xyz_m: np.ndarray,
    *,
    config: ConditionalGroupedStudentTObjectiveConfig | None = None,
) -> ConditionalGroupedStudentTObjectiveResult:
    """Evaluate the exact conditional grouped objective used by the solver.

    ``conditional_prediction_xyz_m`` must already include the evaluated physical
    state and nuisance contribution. Local covariance remains conditional, row
    reliability enters the Mahalanobis distance once, zero-reliability rows are
    inert, and the group power follows the same consumer-cap or provider-final
    semantics as prior-aware inference.
    """

    settings = config or ConditionalGroupedStudentTObjectiveConfig()
    predicted = np.asarray(conditional_prediction_xyz_m, dtype=np.float64)
    if predicted.shape != belief.mean_xyz_m.shape:
        raise ValueError(
            "conditional_prediction_xyz_m must match the observation mean shape"
        )
    if not np.all(np.isfinite(predicted)):
        raise ValueError("conditional_prediction_xyz_m must be finite")

    mode = _resolved_composite_weight_mode(belief, settings.composite_weight_mode)
    group_count = len(belief.group_ids)
    dimensions = np.zeros(group_count, dtype=np.int64)
    nll = np.zeros(group_count, dtype=np.float64)
    weighted_nll = np.zeros(group_count, dtype=np.float64)
    posterior_nominal = np.empty(group_count, dtype=np.float64)
    log_nominal = np.zeros(group_count, dtype=np.float64)
    log_outlier = np.zeros(group_count, dtype=np.float64)
    mean_association = np.zeros(group_count, dtype=np.float64)
    mahalanobis = np.zeros(group_count, dtype=np.float64)
    group_power = np.zeros(group_count, dtype=np.float64)
    prior = np.clip(
        belief.group_prior_nominal_probability,
        settings.probability_floor,
        1.0 - settings.probability_floor,
    )
    mixture_config = _mixture_config(
        degrees_of_freedom=settings.degrees_of_freedom,
        outlier_covariance_multiplier=settings.outlier_covariance_multiplier,
        probability_floor=settings.probability_floor,
    )
    residual_all = belief.mean_xyz_m - predicted

    for position, group_id in enumerate(belief.group_ids):
        selected = belief.correlation_group_ids == group_id
        dimensions[position], mahalanobis[position], mean_association[position] = (
            _conditional_group_statistics(belief, residual_all, selected)
        )
        active_count = int(dimensions[position] // 3)
        if active_count == 0:
            posterior_nominal[position] = prior[position]
            continue
        raw_composite = float(belief.group_composite_weight[position])
        if mode == COMPOSITE_WEIGHT_MODE_PROVIDER_FINAL:
            group_power[position] = raw_composite
        else:
            cap = settings.effective_samples_per_correlation_group
            group_power[position] = raw_composite * min(cap, active_count) / active_count
        statistics = _student_t_mixture_statistics(
            float(mahalanobis[position]),
            int(dimensions[position]),
            float(prior[position]),
            mixture_config,
        )
        log_nominal[position] = statistics.log_nominal_density
        log_outlier[position] = statistics.log_outlier_density
        nll[position] = -statistics.log_mixture_density
        weighted_nll[position] = group_power[position] * nll[position]
        posterior_nominal[position] = statistics.posterior_nominal_probability

    return ConditionalGroupedStudentTObjectiveResult(
        group_ids=belief.group_ids,
        active_dimensions=dimensions,
        negative_log_objective=nll,
        weighted_negative_log_objective=weighted_nll,
        posterior_nominal_probability=posterior_nominal,
        prior_nominal_probability=prior,
        group_power=group_power,
        log_nominal_density=log_nominal,
        log_outlier_density=log_outlier,
        mean_association_probability=mean_association,
        reliability_weighted_mahalanobis_squared=mahalanobis,
        composite_weight_mode=mode,
    )


__all__ = [
    "CONDITIONAL_GROUP_OBJECTIVE_SEMANTICS",
    "COVARIANCE_MARGINAL_SCORE_SEMANTICS",
    "ConditionalGroupedStudentTObjectiveConfig",
    "ConditionalGroupedStudentTObjectiveResult",
    "GroupedStudentTLikelihoodConfig",
    "GroupedStudentTLikelihoodResult",
    "conditional_grouped_student_t_mixture_objective",
    "grouped_student_t_mixture_likelihood",
]
