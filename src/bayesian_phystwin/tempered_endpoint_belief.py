"""Source-tempered endpoint beliefs, candidate guards, and group calibration.

Version 1 of the endpoint model average accumulates predictive log evidence over
every supported prefix observation.  On temporally correlated PhysTwin residuals,
that evidence can behave as though many more independent observations were
available and collapse the nominal finite mixture onto one component.  This
module is additive: it consumes an immutable V1 posterior, tempers only the
component evidence, and leaves every V1 contract and frozen result unchanged.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np

from ._canonical_contracts import canonical_string_tuple
from .bias_aware_belief import (
    GuardedUpdateDecision,
    SourceRegretCertificate,
    apply_regret_guard,
    fit_source_regret_certificate,
)
from .endpoint_model_average import (
    ModelAveragedEndpointConfigV1,
    ModelAveragedEndpointPosteriorV1,
    infer_model_averaged_endpoint,
)
from .grouped_conformal import (
    ConformalScore,
    finite_group_conformal_quantile,
    group_max_nonconformity_scores,
)

TEMPERED_ENDPOINT_CONTRACT_VERSION = 2
ENDPOINT_REGRET_GUARD_FEATURE_NAMES = (
    "validation_relative_improvement",
    "mean_component_entropy_nats",
    "median_effective_component_count",
    "mean_predictive_std_m",
    "correction_rms_m",
    "correction_saturated_fraction",
    "normalized_horizon",
)
EndpointPriorMode = Literal["base", "source"]


def _canonical_sha256(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _number(
    value: object,
    *,
    name: str,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value,
        (int, float, np.integer, np.floating),
    ):
        raise TypeError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    if maximum is not None and result > maximum:
        raise ValueError(f"{name} must be at most {maximum}")
    return result


def _readonly(
    value: object,
    *,
    dtype: np.dtype | type = np.float64,
) -> np.ndarray:
    result = np.array(value, dtype=dtype, copy=True, order="C")
    result.setflags(write=False)
    return result


def _validate_covariance(covariance: np.ndarray, *, name: str) -> None:
    if covariance.ndim != 3 or covariance.shape[1:] != (3, 3):
        raise ValueError(f"{name} must have shape (N, 3, 3)")
    if not np.all(np.isfinite(covariance)):
        raise ValueError(f"{name} must contain finite values")
    if not np.allclose(covariance, covariance.transpose(0, 2, 1)):
        raise ValueError(f"{name} must be symmetric")
    if np.min(np.linalg.eigvalsh(covariance), initial=0.0) < -1e-12:
        raise ValueError(f"{name} must be positive semidefinite")


def _normalized_prior(value: object, *, component_count: int) -> np.ndarray:
    prior = np.asarray(value, dtype=np.float64)
    if prior.shape != (component_count,):
        raise ValueError("component prior must match the component count")
    if not np.all(np.isfinite(prior)) or np.any(prior <= 0.0):
        raise ValueError("component prior must be finite and positive")
    prior = prior / np.sum(prior)
    return prior


@dataclass(frozen=True, slots=True)
class TemperedEndpointConfigV2:
    """Evidence tempering and source-frozen covariance inflation settings."""

    evidence_temperature: float = 1.0
    maximum_effective_observations: float | None = 8.0
    component_prior_probability: tuple[float, ...] | None = None
    covariance_scale: float = 1.0
    isotropic_floor_std_m: float = 0.0

    def __post_init__(self) -> None:
        temperature = _number(
            self.evidence_temperature,
            name="evidence_temperature",
            minimum=1.0,
        )
        cap = self.maximum_effective_observations
        if cap is not None:
            cap = _number(
                cap,
                name="maximum_effective_observations",
                minimum=1.0,
            )
        scale = _number(
            self.covariance_scale,
            name="covariance_scale",
            minimum=1.0,
        )
        floor = _number(
            self.isotropic_floor_std_m,
            name="isotropic_floor_std_m",
            minimum=0.0,
        )
        prior = self.component_prior_probability
        if prior is not None:
            values = tuple(prior)
            if not values:
                raise ValueError("component_prior_probability must not be empty")
            normalized = _normalized_prior(values, component_count=len(values))
            prior = tuple(float(value) for value in normalized)
        object.__setattr__(self, "evidence_temperature", temperature)
        object.__setattr__(self, "maximum_effective_observations", cap)
        object.__setattr__(self, "component_prior_probability", prior)
        object.__setattr__(self, "covariance_scale", scale)
        object.__setattr__(self, "isotropic_floor_std_m", floor)

    @property
    def config_id(self) -> str:
        return _canonical_sha256(
            {
                "schema": "bayesian_phystwin.tempered_endpoint_config",
                "schema_version": TEMPERED_ENDPOINT_CONTRACT_VERSION,
                "evidence_temperature": self.evidence_temperature,
                "maximum_effective_observations": (
                    self.maximum_effective_observations
                ),
                "component_prior_probability": (
                    None
                    if self.component_prior_probability is None
                    else list(self.component_prior_probability)
                ),
                "covariance_scale": self.covariance_scale,
                "isotropic_floor_std_m": self.isotropic_floor_std_m,
            }
        )


DEFAULT_TEMPERED_ENDPOINT_CONFIG_V2 = TemperedEndpointConfigV2()


@dataclass(frozen=True, slots=True)
class SourceComponentPriorV1:
    """Equal-group source score aggregation with explicit uniform shrinkage."""

    probability: np.ndarray
    group_log_score: np.ndarray
    group_ids: tuple[str, ...]
    score_temperature: float
    uniform_pseudocount: float

    def __post_init__(self) -> None:
        groups = canonical_string_tuple(
            self.group_ids,
            name="group_ids",
            allow_empty=False,
        )
        if len(groups) < 2 or len(set(groups)) != len(groups):
            raise ValueError("source component prior requires unique source groups")
        scores = np.asarray(self.group_log_score, dtype=np.float64)
        if scores.ndim != 2 or scores.shape[0] != len(groups) or scores.shape[1] < 1:
            raise ValueError(
                "group_log_score must have shape (source groups, components)"
            )
        if not np.all(np.isfinite(scores)):
            raise ValueError("group_log_score must contain finite values")
        probability = _normalized_prior(
            self.probability,
            component_count=scores.shape[1],
        )
        temperature = _number(
            self.score_temperature,
            name="score_temperature",
            minimum=1.0,
        )
        pseudocount = _number(
            self.uniform_pseudocount,
            name="uniform_pseudocount",
            minimum=0.0,
        )
        object.__setattr__(self, "probability", _readonly(probability))
        object.__setattr__(self, "group_log_score", _readonly(scores))
        object.__setattr__(self, "group_ids", groups)
        object.__setattr__(self, "score_temperature", temperature)
        object.__setattr__(self, "uniform_pseudocount", pseudocount)

    @property
    def prior_id(self) -> str:
        return _canonical_sha256(
            {
                "schema": "bayesian_phystwin.source_component_prior",
                "schema_version": 1,
                "probability": self.probability.tolist(),
                "group_log_score": self.group_log_score.tolist(),
                "group_ids": list(self.group_ids),
                "score_temperature": self.score_temperature,
                "uniform_pseudocount": self.uniform_pseudocount,
            }
        )


def fit_source_component_prior(
    group_log_score: np.ndarray,
    group_ids: Sequence[str],
    *,
    score_temperature: float = 1.0,
    uniform_pseudocount: float = 1.0,
) -> SourceComponentPriorV1:
    """Fit a component prior with equal weight for each independent source group.

    ``group_log_score[g, k]`` is a source-only proper log score for component
    ``k`` on group ``g``; larger values are better. Scores are centered within
    each group before averaging, so groups with different endpoint counts or
    additive likelihood constants receive equal influence.
    """

    scores = np.asarray(group_log_score, dtype=np.float64)
    groups = canonical_string_tuple(
        group_ids,
        name="group_ids",
        allow_empty=False,
    )
    if scores.ndim != 2 or scores.shape[0] != len(groups) or scores.shape[1] < 1:
        raise ValueError("group_log_score must have shape (groups, components)")
    if len(groups) < 2 or len(set(groups)) != len(groups):
        raise ValueError("at least two unique source groups are required")
    if not np.all(np.isfinite(scores)):
        raise ValueError("group_log_score must contain finite values")
    temperature = _number(
        score_temperature,
        name="score_temperature",
        minimum=1.0,
    )
    pseudocount = _number(
        uniform_pseudocount,
        name="uniform_pseudocount",
        minimum=0.0,
    )
    centered = scores - np.max(scores, axis=1, keepdims=True)
    aggregate = np.mean(centered, axis=0) / temperature
    aggregate -= np.max(aggregate)
    evidence_probability = np.exp(aggregate)
    evidence_probability /= np.sum(evidence_probability)
    uniform = np.full(scores.shape[1], 1.0 / scores.shape[1])
    probability = (
        len(groups) * evidence_probability + pseudocount * uniform
    ) / (len(groups) + pseudocount)
    return SourceComponentPriorV1(
        probability=probability,
        group_log_score=scores,
        group_ids=groups,
        score_temperature=temperature,
        uniform_pseudocount=pseudocount,
    )


@dataclass(frozen=True, slots=True)
class TemperedEndpointPosteriorV2:
    """A V1 component bank recombined with source-tempered evidence weights."""

    base_posterior: ModelAveragedEndpointPosteriorV1
    config: TemperedEndpointConfigV2
    mean_m: np.ndarray
    covariance_m2: np.ndarray
    final_nominal_probability: np.ndarray
    component_weights: np.ndarray
    component_prior_probability: np.ndarray
    temperature_by_track: np.ndarray
    between_model_covariance_fraction: np.ndarray

    def __post_init__(self) -> None:
        if not isinstance(self.base_posterior, ModelAveragedEndpointPosteriorV1):
            raise TypeError(
                "base_posterior must be a ModelAveragedEndpointPosteriorV1"
            )
        if not isinstance(self.config, TemperedEndpointConfigV2):
            raise TypeError("config must be a TemperedEndpointConfigV2")
        track_count = len(self.base_posterior.mean_m)
        component_count = len(self.base_posterior.config.components)
        mean = np.asarray(self.mean_m, dtype=np.float64)
        covariance = np.asarray(self.covariance_m2, dtype=np.float64)
        probability = np.asarray(
            self.final_nominal_probability,
            dtype=np.float64,
        )
        weights = np.asarray(self.component_weights, dtype=np.float64)
        prior = np.asarray(
            self.component_prior_probability,
            dtype=np.float64,
        )
        temperature = np.asarray(self.temperature_by_track, dtype=np.float64)
        between = np.asarray(
            self.between_model_covariance_fraction,
            dtype=np.float64,
        )
        if mean.shape != (track_count, 3):
            raise ValueError("mean_m shape changed")
        _validate_covariance(covariance, name="covariance_m2")
        if covariance.shape[0] != track_count:
            raise ValueError("covariance_m2 track count changed")
        if probability.shape != (track_count,) or np.any(
            (probability < 0.0) | (probability > 1.0)
        ):
            raise ValueError(
                "final_nominal_probability must be a track vector in [0, 1]"
            )
        if weights.shape != (track_count, component_count):
            raise ValueError("component_weights shape changed")
        if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
            raise ValueError("component_weights must be finite and nonnegative")
        if not np.allclose(
            np.sum(weights, axis=1),
            1.0,
            atol=1e-12,
            rtol=1e-12,
        ):
            raise ValueError("component_weights must be row-normalized")
        prior = _normalized_prior(prior, component_count=component_count)
        if temperature.shape != (track_count,) or not np.all(
            np.isfinite(temperature)
        ):
            raise ValueError("temperature_by_track shape or values changed")
        if np.any(temperature < 1.0):
            raise ValueError("temperature_by_track must be at least one")
        if between.shape != (track_count,) or not np.all(np.isfinite(between)):
            raise ValueError(
                "between_model_covariance_fraction shape or values changed"
            )
        if np.any((between < 0.0) | (between > 1.0 + 1e-12)):
            raise ValueError(
                "between_model_covariance_fraction must lie in [0, 1]"
            )
        finite_arrays = (mean, covariance, probability)
        if not all(np.all(np.isfinite(value)) for value in finite_arrays):
            raise ValueError("tempered endpoint posterior contains non-finite values")
        object.__setattr__(self, "mean_m", _readonly(mean))
        object.__setattr__(self, "covariance_m2", _readonly(covariance))
        object.__setattr__(
            self,
            "final_nominal_probability",
            _readonly(probability),
        )
        object.__setattr__(self, "component_weights", _readonly(weights))
        object.__setattr__(
            self,
            "component_prior_probability",
            _readonly(prior),
        )
        object.__setattr__(
            self,
            "temperature_by_track",
            _readonly(temperature),
        )
        object.__setattr__(
            self,
            "between_model_covariance_fraction",
            _readonly(np.clip(between, 0.0, 1.0)),
        )

    @property
    def update_count(self) -> np.ndarray:
        return self.base_posterior.update_count

    @property
    def updated_mask(self) -> np.ndarray:
        return self.base_posterior.updated_mask

    @property
    def component_entropy_nats(self) -> np.ndarray:
        entropy = -np.sum(
            self.component_weights
            * np.log(np.maximum(self.component_weights, 1e-300)),
            axis=1,
        )
        entropy.setflags(write=False)
        return entropy

    @property
    def effective_component_count(self) -> np.ndarray:
        effective = 1.0 / np.sum(np.square(self.component_weights), axis=1)
        effective.setflags(write=False)
        return effective


@dataclass(frozen=True, slots=True)
class TemperedEndpointPredictionV2:
    """Horizon propagation of a tempered endpoint posterior."""

    mean_m: np.ndarray
    covariance_m2: np.ndarray
    component_weights: np.ndarray
    horizon_steps: int
    config_id: str

    def __post_init__(self) -> None:
        mean = np.asarray(self.mean_m, dtype=np.float64)
        covariance = np.asarray(self.covariance_m2, dtype=np.float64)
        weights = np.asarray(self.component_weights, dtype=np.float64)
        if mean.ndim != 2 or mean.shape[1] != 3 or len(mean) < 1:
            raise ValueError("mean_m must have shape (N>=1, 3)")
        _validate_covariance(covariance, name="covariance_m2")
        if len(covariance) != len(mean):
            raise ValueError("covariance_m2 track count changed")
        if weights.ndim != 2 or weights.shape[0] != len(mean):
            raise ValueError("component_weights shape changed")
        if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(weights)):
            raise ValueError("tempered endpoint prediction contains non-finite values")
        if np.any(weights < 0.0) or not np.allclose(
            np.sum(weights, axis=1),
            1.0,
            atol=1e-12,
            rtol=1e-12,
        ):
            raise ValueError("component_weights must be row-normalized")
        raw_horizon = self.horizon_steps
        if (
            isinstance(raw_horizon, (bool, np.bool_))
            or not isinstance(raw_horizon, (int, np.integer))
            or raw_horizon < 0
        ):
            raise ValueError("horizon_steps must be a nonnegative integer")
        if (
            not isinstance(self.config_id, str)
            or len(self.config_id) != 64
            or any(character not in "0123456789abcdef" for character in self.config_id)
        ):
            raise ValueError("config_id must be a lowercase SHA-256 digest")
        object.__setattr__(self, "mean_m", _readonly(mean))
        object.__setattr__(self, "covariance_m2", _readonly(covariance))
        object.__setattr__(self, "component_weights", _readonly(weights))
        object.__setattr__(self, "horizon_steps", int(raw_horizon))


def _resolved_prior(
    posterior: ModelAveragedEndpointPosteriorV1,
    config: TemperedEndpointConfigV2,
) -> np.ndarray:
    supplied = config.component_prior_probability
    if supplied is None:
        supplied = posterior.config.component_prior_probability
    return _normalized_prior(
        supplied,
        component_count=len(posterior.config.components),
    )


def _mixture_moments(
    component_mean_m: np.ndarray,
    component_variance_m2: np.ndarray,
    weights: np.ndarray,
    *,
    covariance_scale: float,
    isotropic_floor_std_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = np.einsum("nk,knc->nc", weights, component_mean_m)
    centered = component_mean_m - mean[None, :, :]
    outer = centered[:, :, :, None] * centered[:, :, None, :]
    identity = np.eye(3, dtype=np.float64)
    within = component_variance_m2[:, :, None, None] * identity
    raw_covariance = np.einsum("nk,knij->nij", weights, within + outer)
    raw_covariance = 0.5 * (
        raw_covariance + raw_covariance.transpose(0, 2, 1)
    )
    within_trace = 3.0 * np.einsum(
        "nk,kn->n",
        weights,
        component_variance_m2,
    )
    total_trace = np.trace(raw_covariance, axis1=1, axis2=2)
    between_fraction = np.maximum(
        0.0,
        total_trace - within_trace,
    ) / np.maximum(total_trace, 1e-30)
    covariance = covariance_scale * raw_covariance
    if isotropic_floor_std_m > 0.0:
        covariance = (
            covariance
            + isotropic_floor_std_m**2 * identity[None, :, :]
        )
    covariance = 0.5 * (covariance + covariance.transpose(0, 2, 1))
    return mean, covariance, between_fraction


def temper_model_averaged_endpoint(
    posterior: ModelAveragedEndpointPosteriorV1,
    *,
    config: TemperedEndpointConfigV2 | None = None,
) -> TemperedEndpointPosteriorV2:
    """Recombine a V1 component bank with an effective-evidence cap."""

    if not isinstance(posterior, ModelAveragedEndpointPosteriorV1):
        raise TypeError(
            "posterior must be a ModelAveragedEndpointPosteriorV1"
        )
    settings = (
        DEFAULT_TEMPERED_ENDPOINT_CONFIG_V2 if config is None else config
    )
    if not isinstance(settings, TemperedEndpointConfigV2):
        raise TypeError("config must be a TemperedEndpointConfigV2")
    prior = _resolved_prior(posterior, settings)
    update_count = np.asarray(posterior.update_count, dtype=np.float64)
    temperature = np.full(
        len(update_count),
        settings.evidence_temperature,
        dtype=np.float64,
    )
    if settings.maximum_effective_observations is not None:
        temperature = np.maximum(
            temperature,
            update_count / settings.maximum_effective_observations,
        )
    logits = (
        posterior.component_log_evidence / temperature[:, None]
        + np.log(prior)[None, :]
    )
    logits -= np.max(logits, axis=1, keepdims=True)
    weights = np.exp(logits)
    weights /= np.sum(weights, axis=1, keepdims=True)
    mean, covariance, between_fraction = _mixture_moments(
        posterior.component_mean_m,
        posterior.component_variance_m2,
        weights,
        covariance_scale=settings.covariance_scale,
        isotropic_floor_std_m=settings.isotropic_floor_std_m,
    )
    # V1 exposes the weighted terminal nominal probability, not its
    # per-component decomposition. Preserve that diagnostic unchanged rather
    # than pretending it can be reweighted exactly.
    final_probability = posterior.final_nominal_probability
    return TemperedEndpointPosteriorV2(
        base_posterior=posterior,
        config=settings,
        mean_m=mean,
        covariance_m2=covariance,
        final_nominal_probability=final_probability,
        component_weights=weights,
        component_prior_probability=prior,
        temperature_by_track=temperature,
        between_model_covariance_fraction=between_fraction,
    )


def infer_tempered_endpoint(
    residual_m: np.ndarray,
    valid: np.ndarray,
    *,
    end_frame: int,
    base_config: ModelAveragedEndpointConfigV1 | None = None,
    config: TemperedEndpointConfigV2 | None = None,
) -> TemperedEndpointPosteriorV2:
    """Infer V1 components from a causal prefix and temper their evidence."""

    base = infer_model_averaged_endpoint(
        residual_m,
        valid,
        end_frame=end_frame,
        config=base_config,
    )
    return temper_model_averaged_endpoint(base, config=config)


def predict_tempered_endpoint(
    posterior: TemperedEndpointPosteriorV2,
    *,
    horizon_steps: int,
) -> TemperedEndpointPredictionV2:
    """Propagate a tempered component bank without future observations."""

    if not isinstance(posterior, TemperedEndpointPosteriorV2):
        raise TypeError("posterior must be a TemperedEndpointPosteriorV2")
    raw_horizon = horizon_steps
    if (
        isinstance(raw_horizon, (bool, np.bool_))
        or not isinstance(raw_horizon, (int, np.integer))
        or raw_horizon < 0
    ):
        raise ValueError("horizon_steps must be a nonnegative integer")
    horizon = int(raw_horizon)
    base = posterior.base_posterior
    component_variance = (
        base.component_variance_m2
        + horizon * base.component_process_variance_m2[:, None]
    )
    mean, covariance, _ = _mixture_moments(
        base.component_mean_m,
        component_variance,
        posterior.component_weights,
        covariance_scale=posterior.config.covariance_scale,
        isotropic_floor_std_m=posterior.config.isotropic_floor_std_m,
    )
    return TemperedEndpointPredictionV2(
        mean_m=mean,
        covariance_m2=covariance,
        component_weights=posterior.component_weights,
        horizon_steps=horizon,
        config_id=posterior.config.config_id,
    )


@dataclass(frozen=True, slots=True)
class EndpointRegretGuardFeaturesV1:
    """Pre-target endpoint diagnostics used by a source-only regret guard."""

    validation_relative_improvement: float
    mean_component_entropy_nats: float
    median_effective_component_count: float
    mean_predictive_std_m: float
    correction_rms_m: float
    correction_saturated_fraction: float
    normalized_horizon: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "validation_relative_improvement",
            _number(
                self.validation_relative_improvement,
                name="validation_relative_improvement",
            ),
        )
        object.__setattr__(
            self,
            "mean_component_entropy_nats",
            _number(
                self.mean_component_entropy_nats,
                name="mean_component_entropy_nats",
                minimum=0.0,
            ),
        )
        object.__setattr__(
            self,
            "median_effective_component_count",
            _number(
                self.median_effective_component_count,
                name="median_effective_component_count",
                minimum=1.0,
            ),
        )
        for name in (
            "mean_predictive_std_m",
            "correction_rms_m",
        ):
            object.__setattr__(
                self,
                name,
                _number(getattr(self, name), name=name, minimum=0.0),
            )
        for name in (
            "correction_saturated_fraction",
            "normalized_horizon",
        ):
            object.__setattr__(
                self,
                name,
                _number(
                    getattr(self, name),
                    name=name,
                    minimum=0.0,
                    maximum=1.0,
                ),
            )

    def as_array(self) -> np.ndarray:
        values = np.asarray(
            [
                self.validation_relative_improvement,
                self.mean_component_entropy_nats,
                self.median_effective_component_count,
                self.mean_predictive_std_m,
                self.correction_rms_m,
                self.correction_saturated_fraction,
                self.normalized_horizon,
            ],
            dtype=np.float64,
        )
        values.setflags(write=False)
        return values


@dataclass(frozen=True, slots=True)
class EndpointRegretGuardV1:
    """A model-average-specific source regret certificate."""

    certificate: SourceRegretCertificate
    candidate_name: str
    fallback_name: str
    loss_name: str
    source_group_ids: tuple[str, ...]
    feature_names: tuple[str, ...] = ENDPOINT_REGRET_GUARD_FEATURE_NAMES

    def __post_init__(self) -> None:
        if not isinstance(self.certificate, SourceRegretCertificate):
            raise TypeError("certificate must be a SourceRegretCertificate")
        for name in ("candidate_name", "fallback_name", "loss_name"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a nonempty string")
        groups = canonical_string_tuple(
            self.source_group_ids,
            name="source_group_ids",
            allow_empty=False,
        )
        if len(groups) < 3 or len(set(groups)) != len(groups):
            raise ValueError("guard requires at least three unique source groups")
        features = canonical_string_tuple(
            self.feature_names,
            name="feature_names",
            allow_empty=False,
        )
        if features != ENDPOINT_REGRET_GUARD_FEATURE_NAMES:
            raise ValueError("endpoint guard feature ordering changed")
        if len(self.certificate.feature_center) != len(features):
            raise ValueError("guard certificate feature count changed")
        object.__setattr__(self, "source_group_ids", groups)
        object.__setattr__(self, "feature_names", features)

    @property
    def guard_id(self) -> str:
        return _canonical_sha256(
            {
                "schema": "bayesian_phystwin.endpoint_regret_guard",
                "schema_version": 1,
                "candidate_name": self.candidate_name,
                "fallback_name": self.fallback_name,
                "loss_name": self.loss_name,
                "source_group_ids": list(self.source_group_ids),
                "feature_names": list(self.feature_names),
                "certificate": {
                    "feature_center": self.certificate.feature_center.tolist(),
                    "feature_scale": self.certificate.feature_scale.tolist(),
                    "standardized_feature_lower": (
                        self.certificate.standardized_feature_lower.tolist()
                    ),
                    "standardized_feature_upper": (
                        self.certificate.standardized_feature_upper.tolist()
                    ),
                    "coefficients": self.certificate.coefficients.tolist(),
                    "upper_residual_quantile": (
                        self.certificate.upper_residual_quantile
                    ),
                    "nominal_coverage": self.certificate.nominal_coverage,
                    "minimum_improvement": self.certificate.minimum_improvement,
                    "ridge_penalty": self.certificate.ridge_penalty,
                    "support_margin_std": self.certificate.support_margin_std,
                    "source_group_count": self.certificate.source_group_count,
                    "finite_sample_rank": self.certificate.finite_sample_rank,
                    "finite_sample_coverage": (
                        self.certificate.finite_sample_coverage
                    ),
                },
            }
        )


def fit_endpoint_regret_guard(
    feature_rows: Sequence[EndpointRegretGuardFeaturesV1],
    candidate_loss: np.ndarray,
    fallback_loss: np.ndarray,
    group_ids: Sequence[str],
    *,
    candidate_name: str = "tempered_endpoint",
    fallback_name: str = "fallback",
    loss_name: str = "loss",
    nominal_coverage: float = 0.75,
    within_group_coverage: float = 1.0,
    minimum_improvement: float = 0.0,
    ridge_penalty: float = 10.0,
    support_margin_std: float = 0.0,
) -> EndpointRegretGuardV1:
    """Fit a candidate-specific regret UCB from source groups only."""

    rows = tuple(feature_rows)
    if not rows or any(
        not isinstance(row, EndpointRegretGuardFeaturesV1) for row in rows
    ):
        raise TypeError(
            "feature_rows must contain EndpointRegretGuardFeaturesV1 values"
        )
    features = np.vstack([row.as_array() for row in rows])
    candidate = np.asarray(candidate_loss, dtype=np.float64)
    fallback = np.asarray(fallback_loss, dtype=np.float64)
    groups = canonical_string_tuple(
        group_ids,
        name="group_ids",
        allow_empty=False,
    )
    if candidate.shape != (len(rows),) or fallback.shape != (len(rows),):
        raise ValueError("candidate and fallback losses must match feature rows")
    if not np.all(np.isfinite(candidate)) or not np.all(np.isfinite(fallback)):
        raise ValueError("candidate and fallback losses must be finite")
    if np.any(candidate < 0.0) or np.any(fallback < 0.0):
        raise ValueError("candidate and fallback losses must be nonnegative")
    if len(groups) != len(rows):
        raise ValueError("group_ids must match feature rows")
    certificate = fit_source_regret_certificate(
        features,
        candidate - fallback,
        groups,
        nominal_coverage=nominal_coverage,
        within_group_coverage=within_group_coverage,
        minimum_improvement=minimum_improvement,
        ridge_penalty=ridge_penalty,
        support_margin_std=support_margin_std,
    )
    unique_groups = tuple(sorted(set(groups)))
    return EndpointRegretGuardV1(
        certificate=certificate,
        candidate_name=candidate_name,
        fallback_name=fallback_name,
        loss_name=loss_name,
        source_group_ids=unique_groups,
    )


def apply_endpoint_regret_guard(
    fallback_value: np.ndarray,
    candidate_value: np.ndarray,
    features: EndpointRegretGuardFeaturesV1,
    guard: EndpointRegretGuardV1,
) -> GuardedUpdateDecision:
    """Apply a frozen endpoint guard with exact fallback identity."""

    if not isinstance(features, EndpointRegretGuardFeaturesV1):
        raise TypeError("features must be EndpointRegretGuardFeaturesV1")
    if not isinstance(guard, EndpointRegretGuardV1):
        raise TypeError("guard must be EndpointRegretGuardV1")
    return apply_regret_guard(
        fallback_value,
        candidate_value,
        features.as_array(),
        guard.certificate,
    )


def minimum_groups_for_finite_conformal_quantile(coverage: float) -> int:
    """Return the smallest group count with a finite split-conformal rank."""

    nominal = _number(
        coverage,
        name="coverage",
        minimum=0.0,
        maximum=1.0,
    )
    if nominal in {0.0, 1.0}:
        raise ValueError("coverage must lie strictly inside (0, 1)")
    group_count = 1
    while math.ceil((group_count + 1) * nominal) > group_count:
        group_count += 1
    return group_count


@dataclass(frozen=True, slots=True)
class EndpointGroupedCalibrationV1:
    """Reusable group-balanced calibration for predictive residual radii."""

    calibration_group_scores: np.ndarray
    calibration_group_ids: tuple[str, ...]
    quantile: float
    finite_sample_rank: int
    nominal_coverage: float
    score: ConformalScore

    def __post_init__(self) -> None:
        groups = canonical_string_tuple(
            self.calibration_group_ids,
            name="calibration_group_ids",
            allow_empty=False,
        )
        if len(set(groups)) != len(groups):
            raise ValueError("calibration_group_ids must be unique")
        scores = np.asarray(self.calibration_group_scores, dtype=np.float64)
        if scores.shape != (len(groups),) or not np.all(np.isfinite(scores)):
            raise ValueError(
                "calibration_group_scores must be one finite value per group"
            )
        raw_rank = self.finite_sample_rank
        if (
            isinstance(raw_rank, (bool, np.bool_))
            or not isinstance(raw_rank, (int, np.integer))
            or raw_rank < 1
        ):
            raise ValueError("finite_sample_rank must be a positive integer")
        rank = int(raw_rank)
        nominal = _number(
            self.nominal_coverage,
            name="nominal_coverage",
            minimum=0.0,
            maximum=1.0,
        )
        if nominal in {0.0, 1.0}:
            raise ValueError("nominal_coverage must lie strictly inside (0, 1)")
        if self.score not in {"scaled", "additive"}:
            raise ValueError("score must be 'scaled' or 'additive'")
        quantile = float(self.quantile)
        if math.isnan(quantile):
            raise ValueError("quantile cannot be NaN")
        if rank > len(groups):
            if quantile != math.inf:
                raise ValueError(
                    "an impossible finite-sample rank requires infinite quantile"
                )
        elif not math.isfinite(quantile):
            raise ValueError("a feasible finite-sample rank requires finite quantile")
        object.__setattr__(
            self,
            "calibration_group_scores",
            _readonly(scores),
        )
        object.__setattr__(self, "calibration_group_ids", groups)
        object.__setattr__(self, "finite_sample_rank", rank)
        object.__setattr__(self, "nominal_coverage", nominal)
        object.__setattr__(self, "quantile", quantile)

    @property
    def finite(self) -> bool:
        return math.isfinite(self.quantile)

    @property
    def required_group_count_for_finite_quantile(self) -> int:
        return minimum_groups_for_finite_conformal_quantile(
            self.nominal_coverage
        )

    @property
    def calibration_id(self) -> str:
        return _canonical_sha256(
            {
                "schema": "bayesian_phystwin.endpoint_grouped_calibration",
                "schema_version": 1,
                "calibration_group_scores": (
                    self.calibration_group_scores.tolist()
                ),
                "calibration_group_ids": list(self.calibration_group_ids),
                "quantile": (
                    "infinity" if math.isinf(self.quantile) else self.quantile
                ),
                "finite_sample_rank": self.finite_sample_rank,
                "nominal_coverage": self.nominal_coverage,
                "score": self.score,
            }
        )

    def upper_bound(self, predictive_radius_m: np.ndarray) -> np.ndarray:
        radius = np.asarray(predictive_radius_m, dtype=np.float64)
        if radius.size == 0 or not np.all(np.isfinite(radius)):
            raise ValueError("predictive_radius_m must be nonempty and finite")
        if np.any(radius < 0.0):
            raise ValueError("predictive_radius_m must be nonnegative")
        if self.score == "scaled" and np.any(radius <= 0.0):
            raise ValueError("scaled predictive radii must be positive")
        if math.isinf(self.quantile):
            result = np.full(radius.shape, math.inf, dtype=np.float64)
        elif self.score == "scaled":
            result = self.quantile * radius
        else:
            result = radius + self.quantile
        result = np.maximum(result, 0.0)
        result.setflags(write=False)
        return result


def fit_endpoint_grouped_calibration(
    calibration_error_norm_groups: Sequence[np.ndarray],
    calibration_predictive_radius_groups: Sequence[np.ndarray],
    calibration_group_ids: Sequence[str],
    *,
    coverage: float = 0.90,
    score: ConformalScore = "scaled",
) -> EndpointGroupedCalibrationV1:
    """Fit one simultaneous nonconformity score per independent source group."""

    groups = canonical_string_tuple(
        calibration_group_ids,
        name="calibration_group_ids",
        allow_empty=False,
    )
    if len(groups) != len(calibration_error_norm_groups):
        raise ValueError("calibration_group_ids must match calibration groups")
    if len(set(groups)) != len(groups):
        raise ValueError("calibration_group_ids must be unique")
    group_scores = group_max_nonconformity_scores(
        calibration_error_norm_groups,
        calibration_predictive_radius_groups,
        score=score,
    )
    quantile, rank = finite_group_conformal_quantile(group_scores, coverage)
    return EndpointGroupedCalibrationV1(
        calibration_group_scores=group_scores,
        calibration_group_ids=groups,
        quantile=quantile,
        finite_sample_rank=rank,
        nominal_coverage=coverage,
        score=score,
    )


__all__ = [
    "DEFAULT_TEMPERED_ENDPOINT_CONFIG_V2",
    "ENDPOINT_REGRET_GUARD_FEATURE_NAMES",
    "TEMPERED_ENDPOINT_CONTRACT_VERSION",
    "EndpointGroupedCalibrationV1",
    "EndpointRegretGuardFeaturesV1",
    "EndpointRegretGuardV1",
    "SourceComponentPriorV1",
    "TemperedEndpointConfigV2",
    "TemperedEndpointPosteriorV2",
    "TemperedEndpointPredictionV2",
    "apply_endpoint_regret_guard",
    "fit_endpoint_grouped_calibration",
    "fit_endpoint_regret_guard",
    "fit_source_component_prior",
    "infer_tempered_endpoint",
    "minimum_groups_for_finite_conformal_quantile",
    "predict_tempered_endpoint",
    "temper_model_averaged_endpoint",
]
