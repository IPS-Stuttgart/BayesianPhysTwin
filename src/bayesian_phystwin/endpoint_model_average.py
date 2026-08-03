"""Evidence-weighted endpoint uncertainty for robust PhysTwin discrepancy."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .contracts.fixed_anchor import FixedBayesianAnchorConfigV1

MODEL_AVERAGED_ENDPOINT_CONTRACT_VERSION = 1


def _historical_component_grid() -> tuple[FixedBayesianAnchorConfigV1, ...]:
    return tuple(
        FixedBayesianAnchorConfigV1(
            process_std_m=process_std,
            observation_std_m=observation_std,
        )
        for process_std in (0.0, 0.0005, 0.001, 0.0025, 0.005)
        for observation_std in (0.001, 0.0025, 0.005)
    )


@dataclass(frozen=True, slots=True)
class ModelAveragedEndpointConfigV1:
    """Finite robust random-walk family with source-frozen model priors."""

    components: tuple[FixedBayesianAnchorConfigV1, ...] = field(
        default_factory=_historical_component_grid
    )
    component_prior_probability: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        components = tuple(self.components)
        if not components:
            raise ValueError("at least one endpoint component is required")
        if not all(
            isinstance(component, FixedBayesianAnchorConfigV1)
            for component in components
        ):
            raise TypeError(
                "components must contain FixedBayesianAnchorConfigV1 values"
            )
        identities = {
            (
                component.process_std_m,
                component.observation_std_m,
                component.initial_std_m,
                component.inlier_prior,
                component.outlier_variance_multiplier,
            )
            for component in components
        }
        if len(identities) != len(components):
            raise ValueError("endpoint components must be unique")
        if self.component_prior_probability is None:
            prior = np.full(len(components), 1.0 / len(components))
        else:
            prior = np.asarray(
                self.component_prior_probability,
                dtype=np.float64,
            )
            if prior.shape != (len(components),):
                raise ValueError(
                    "component_prior_probability must match the component count"
                )
            if not np.all(np.isfinite(prior)) or np.any(prior <= 0.0):
                raise ValueError(
                    "component_prior_probability must be finite and positive"
                )
            prior = prior / np.sum(prior)
        object.__setattr__(self, "components", components)
        object.__setattr__(
            self,
            "component_prior_probability",
            tuple(float(value) for value in prior),
        )


DEFAULT_MODEL_AVERAGED_ENDPOINT_CONFIG_V1 = ModelAveragedEndpointConfigV1()


def _readonly(value: np.ndarray, *, dtype: np.dtype | type = np.float64) -> np.ndarray:
    result = np.array(value, dtype=dtype, copy=True, order="C")
    result.setflags(write=False)
    return result


def _validate_covariance(covariance: np.ndarray, *, name: str) -> None:
    if covariance.ndim != 3 or covariance.shape[1:] != (3, 3):
        raise ValueError(f"{name} must have shape (N, 3, 3)")
    if not np.all(np.isfinite(covariance)):
        raise ValueError(f"{name} must contain only finite values")
    if not np.allclose(covariance, covariance.transpose(0, 2, 1)):
        raise ValueError(f"{name} must be symmetric")
    eigenvalues = np.linalg.eigvalsh(covariance)
    if np.min(eigenvalues, initial=0.0) < -1e-12:
        raise ValueError(f"{name} must be positive semidefinite")


@dataclass(frozen=True, slots=True)
class ModelAveragedEndpointPosteriorV1:
    """Per-track model-averaged endpoint posterior and component evidence."""

    mean_m: np.ndarray
    covariance_m2: np.ndarray
    final_nominal_probability: np.ndarray
    update_count: np.ndarray
    component_weights: np.ndarray
    component_log_evidence: np.ndarray
    component_mean_m: np.ndarray
    component_variance_m2: np.ndarray
    component_process_variance_m2: np.ndarray
    config: ModelAveragedEndpointConfigV1
    end_frame: int

    def __post_init__(self) -> None:
        if not isinstance(self.config, ModelAveragedEndpointConfigV1):
            raise TypeError("config must be a ModelAveragedEndpointConfigV1")
        if isinstance(self.end_frame, bool) or int(self.end_frame) != self.end_frame:
            raise ValueError("end_frame must be an integer")
        if self.end_frame < 1:
            raise ValueError("end_frame must be positive")
        mean = np.asarray(self.mean_m, dtype=np.float64)
        covariance = np.asarray(self.covariance_m2, dtype=np.float64)
        probability = np.asarray(
            self.final_nominal_probability,
            dtype=np.float64,
        )
        raw_count = np.asarray(self.update_count)
        weights = np.asarray(self.component_weights, dtype=np.float64)
        evidence = np.asarray(self.component_log_evidence, dtype=np.float64)
        component_mean = np.asarray(self.component_mean_m, dtype=np.float64)
        component_variance = np.asarray(
            self.component_variance_m2,
            dtype=np.float64,
        )
        process_variance = np.asarray(
            self.component_process_variance_m2,
            dtype=np.float64,
        )
        if mean.ndim != 2 or mean.shape[1] != 3 or len(mean) < 1:
            raise ValueError("mean_m must have shape (N>=1, 3)")
        track_count = len(mean)
        component_count = len(self.config.components)
        _validate_covariance(covariance, name="covariance_m2")
        if len(covariance) != track_count:
            raise ValueError("covariance_m2 track count changed")
        if probability.shape != (track_count,):
            raise ValueError("final_nominal_probability shape changed")
        if not np.issubdtype(raw_count.dtype, np.integer):
            raise ValueError("update_count must contain integers")
        count = np.asarray(raw_count, dtype=np.int64)
        if count.shape != (track_count,) or np.any(count < 0):
            raise ValueError("update_count must be a nonnegative track vector")
        expected_weights = (track_count, component_count)
        if weights.shape != expected_weights or evidence.shape != expected_weights:
            raise ValueError("component weight/evidence shape changed")
        if component_mean.shape != (component_count, track_count, 3):
            raise ValueError("component_mean_m shape changed")
        if component_variance.shape != (component_count, track_count):
            raise ValueError("component_variance_m2 shape changed")
        if process_variance.shape != (component_count,):
            raise ValueError("component_process_variance_m2 shape changed")
        finite_values = (
            mean,
            probability,
            weights,
            evidence,
            component_mean,
            component_variance,
            process_variance,
        )
        if not all(np.all(np.isfinite(value)) for value in finite_values):
            raise ValueError("model-averaged posterior contains non-finite values")
        if np.any((probability < 0.0) | (probability > 1.0)):
            raise ValueError("final_nominal_probability must lie in [0, 1]")
        if np.any(weights < 0.0) or not np.allclose(
            np.sum(weights, axis=1),
            1.0,
            atol=1e-12,
            rtol=1e-12,
        ):
            raise ValueError("component_weights must be row-normalized")
        if np.any(component_variance < 0.0) or np.any(process_variance < 0.0):
            raise ValueError("component variances must be nonnegative")
        for name, value, dtype in (
            ("mean_m", mean, np.float64),
            ("covariance_m2", covariance, np.float64),
            ("final_nominal_probability", probability, np.float64),
            ("update_count", count, np.int64),
            ("component_weights", weights, np.float64),
            ("component_log_evidence", evidence, np.float64),
            ("component_mean_m", component_mean, np.float64),
            ("component_variance_m2", component_variance, np.float64),
            (
                "component_process_variance_m2",
                process_variance,
                np.float64,
            ),
        ):
            object.__setattr__(self, name, _readonly(value, dtype=dtype))
        object.__setattr__(self, "end_frame", int(self.end_frame))

    @property
    def updated_mask(self) -> np.ndarray:
        updated = self.update_count > 0
        updated.setflags(write=False)
        return updated


@dataclass(frozen=True, slots=True)
class ModelAveragedEndpointPredictionV1:
    """Horizon-propagated model-averaged endpoint moments."""

    mean_m: np.ndarray
    covariance_m2: np.ndarray
    component_weights: np.ndarray
    horizon_steps: int

    def __post_init__(self) -> None:
        if (
            isinstance(self.horizon_steps, bool)
            or int(self.horizon_steps) != self.horizon_steps
            or self.horizon_steps < 0
        ):
            raise ValueError("horizon_steps must be a nonnegative integer")
        mean = np.asarray(self.mean_m, dtype=np.float64)
        covariance = np.asarray(self.covariance_m2, dtype=np.float64)
        weights = np.asarray(self.component_weights, dtype=np.float64)
        if mean.ndim != 2 or mean.shape[1] != 3 or len(mean) < 1:
            raise ValueError("mean_m must have shape (N>=1, 3)")
        _validate_covariance(covariance, name="covariance_m2")
        if len(covariance) != len(mean):
            raise ValueError("prediction covariance track count changed")
        if weights.ndim != 2 or weights.shape[0] != len(mean):
            raise ValueError("component_weights shape changed")
        if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(weights)):
            raise ValueError("model-averaged prediction contains non-finite values")
        if np.any(weights < 0.0) or not np.allclose(
            np.sum(weights, axis=1),
            1.0,
            atol=1e-12,
            rtol=1e-12,
        ):
            raise ValueError("component_weights must be row-normalized")
        object.__setattr__(self, "mean_m", _readonly(mean))
        object.__setattr__(self, "covariance_m2", _readonly(covariance))
        object.__setattr__(self, "component_weights", _readonly(weights))
        object.__setattr__(self, "horizon_steps", int(self.horizon_steps))


def _validated_inputs(
    residual_m: np.ndarray,
    valid: np.ndarray,
    *,
    end_frame: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    residual = np.asarray(residual_m, dtype=np.float64)
    validity = np.asarray(valid, dtype=bool)
    if residual.ndim != 3 or residual.shape[2:] != (3,) or residual.shape[1] < 1:
        raise ValueError("residual_m must have shape (T, N>=1, 3)")
    if validity.shape != residual.shape[:2]:
        raise ValueError("valid must match the residual frame and track dimensions")
    if not np.all(np.isfinite(residual)):
        raise ValueError("residual_m must contain only finite values")
    frame_stop = int(end_frame)
    if isinstance(end_frame, bool) or frame_stop != end_frame:
        raise ValueError("end_frame must be an integer")
    if not 0 < frame_stop <= len(residual):
        raise ValueError("end_frame must lie inside the residual sequence")
    return residual, validity, frame_stop


def _filter_component(
    residual: np.ndarray,
    validity: np.ndarray,
    *,
    end_frame: int,
    config: FixedBayesianAnchorConfigV1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    process_variance = config.process_std_m**2
    observation_variance = config.observation_std_m**2
    initial_variance = config.initial_std_m**2
    track_count = residual.shape[1]
    mean = np.zeros((track_count, 3), dtype=np.float64)
    variance = np.full(track_count, initial_variance, dtype=np.float64)
    final_probability = np.zeros(track_count, dtype=np.float64)
    update_count = np.zeros(track_count, dtype=np.int64)
    log_evidence = np.zeros(track_count, dtype=np.float64)
    log_prior = np.log(config.inlier_prior)
    log_outlier_prior = np.log1p(-config.inlier_prior)
    for frame in range(end_frame):
        predicted_variance = variance + process_variance
        mask = validity[frame]
        variance = predicted_variance
        if not np.any(mask):
            continue
        innovation = residual[frame, mask] - mean[mask]
        predicted = predicted_variance[mask]
        inlier_innovation_variance = predicted + observation_variance
        outlier_innovation_variance = (
            predicted
            + observation_variance * config.outlier_variance_multiplier
        )
        squared_norm = np.sum(np.square(innovation), axis=1)
        log_inlier = log_prior - 0.5 * (
            3.0 * np.log(2.0 * np.pi * inlier_innovation_variance)
            + squared_norm / inlier_innovation_variance
        )
        log_outlier = log_outlier_prior - 0.5 * (
            3.0 * np.log(2.0 * np.pi * outlier_innovation_variance)
            + squared_norm / outlier_innovation_variance
        )
        log_mixture = np.logaddexp(log_inlier, log_outlier)
        probability = np.exp(log_inlier - log_mixture)
        log_evidence[mask] += log_mixture
        inlier_gain = predicted / inlier_innovation_variance
        outlier_gain = predicted / outlier_innovation_variance
        inlier_mean = mean[mask] + inlier_gain[:, None] * innovation
        outlier_mean = mean[mask] + outlier_gain[:, None] * innovation
        updated_mean = (
            probability[:, None] * inlier_mean
            + (1.0 - probability)[:, None] * outlier_mean
        )
        inlier_variance = (1.0 - inlier_gain) * predicted
        outlier_variance = (1.0 - outlier_gain) * predicted
        inlier_spread = np.mean(np.square(inlier_mean - updated_mean), axis=1)
        outlier_spread = np.mean(np.square(outlier_mean - updated_mean), axis=1)
        updated_variance = probability * (inlier_variance + inlier_spread) + (
            1.0 - probability
        ) * (outlier_variance + outlier_spread)
        mean[mask] = updated_mean
        variance[mask] = np.maximum(updated_variance, 0.0)
        final_probability[mask] = probability
        update_count[mask] += 1
    return mean, variance, final_probability, update_count, log_evidence


def infer_model_averaged_endpoint(
    residual_m: np.ndarray,
    valid: np.ndarray,
    *,
    end_frame: int,
    config: ModelAveragedEndpointConfigV1 | None = None,
) -> ModelAveragedEndpointPosteriorV1:
    """Average robust endpoint components by per-track predictive evidence."""

    residual, validity, frame_stop = _validated_inputs(
        residual_m,
        valid,
        end_frame=end_frame,
    )
    settings = DEFAULT_MODEL_AVERAGED_ENDPOINT_CONFIG_V1 if config is None else config
    if not isinstance(settings, ModelAveragedEndpointConfigV1):
        raise TypeError("config must be a ModelAveragedEndpointConfigV1")
    component_count = len(settings.components)
    track_count = residual.shape[1]
    component_mean = np.empty((component_count, track_count, 3))
    component_variance = np.empty((component_count, track_count))
    component_probability = np.empty((component_count, track_count))
    component_evidence = np.empty((track_count, component_count))
    component_process_variance = np.empty(component_count)
    common_update_count: np.ndarray | None = None
    for index, component in enumerate(settings.components):
        (
            component_mean[index],
            component_variance[index],
            component_probability[index],
            update_count,
            component_evidence[:, index],
        ) = _filter_component(
            residual,
            validity,
            end_frame=frame_stop,
            config=component,
        )
        component_process_variance[index] = component.process_std_m**2
        if common_update_count is None:
            common_update_count = update_count
        elif not np.array_equal(common_update_count, update_count):
            raise AssertionError("endpoint components used different observations")
    assert common_update_count is not None
    log_prior = np.log(
        np.asarray(settings.component_prior_probability, dtype=np.float64)
    )
    unnormalized = component_evidence + log_prior[None, :]
    normalizer = np.logaddexp.reduce(unnormalized, axis=1)
    weights = np.exp(unnormalized - normalizer[:, None])
    mean = np.einsum("nk,knc->nc", weights, component_mean)
    centered = component_mean - mean[None, :, :]
    outer = centered[:, :, :, None] * centered[:, :, None, :]
    identity = np.eye(3, dtype=np.float64)
    within = component_variance[:, :, None, None] * identity
    covariance = np.einsum("nk,knij->nij", weights, within + outer)
    covariance = 0.5 * (covariance + covariance.transpose(0, 2, 1))
    final_probability = np.einsum(
        "nk,kn->n",
        weights,
        component_probability,
    )
    return ModelAveragedEndpointPosteriorV1(
        mean_m=mean,
        covariance_m2=covariance,
        final_nominal_probability=final_probability,
        update_count=common_update_count,
        component_weights=weights,
        component_log_evidence=component_evidence,
        component_mean_m=component_mean,
        component_variance_m2=component_variance,
        component_process_variance_m2=component_process_variance,
        config=settings,
        end_frame=frame_stop,
    )


def predict_model_averaged_endpoint(
    posterior: ModelAveragedEndpointPosteriorV1,
    *,
    horizon_steps: int,
) -> ModelAveragedEndpointPredictionV1:
    """Propagate endpoint uncertainty without future observations."""

    if not isinstance(posterior, ModelAveragedEndpointPosteriorV1):
        raise TypeError("posterior must be a ModelAveragedEndpointPosteriorV1")
    if (
        isinstance(horizon_steps, bool)
        or int(horizon_steps) != horizon_steps
        or horizon_steps < 0
    ):
        raise ValueError("horizon_steps must be a nonnegative integer")
    horizon = int(horizon_steps)
    component_variance = (
        posterior.component_variance_m2
        + horizon * posterior.component_process_variance_m2[:, None]
    )
    centered = posterior.component_mean_m - posterior.mean_m[None, :, :]
    outer = centered[:, :, :, None] * centered[:, :, None, :]
    within = component_variance[:, :, None, None] * np.eye(3)
    covariance = np.einsum(
        "nk,knij->nij",
        posterior.component_weights,
        within + outer,
    )
    covariance = 0.5 * (covariance + covariance.transpose(0, 2, 1))
    return ModelAveragedEndpointPredictionV1(
        mean_m=posterior.mean_m,
        covariance_m2=covariance,
        component_weights=posterior.component_weights,
        horizon_steps=horizon,
    )


__all__ = [
    "DEFAULT_MODEL_AVERAGED_ENDPOINT_CONFIG_V1",
    "MODEL_AVERAGED_ENDPOINT_CONTRACT_VERSION",
    "ModelAveragedEndpointConfigV1",
    "ModelAveragedEndpointPosteriorV1",
    "ModelAveragedEndpointPredictionV1",
    "infer_model_averaged_endpoint",
    "predict_model_averaged_endpoint",
]
