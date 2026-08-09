from __future__ import annotations

import numpy as np

from bayesian_phystwin.contracts.fixed_anchor import FixedBayesianAnchorConfigV1
from bayesian_phystwin.dynamic_endpoint_model_average import (
    DampedTrendEndpointComponentV2,
    DynamicEndpointModelAverageConfigV2,
    PersistenceEndpointComponentV2,
    infer_dynamic_endpoint_model_average,
)


def _single(
    component: object,
    *,
    pooling: str = "per_track",
) -> DynamicEndpointModelAverageConfigV2:
    return DynamicEndpointModelAverageConfigV2(
        components=(component,),  # type: ignore[arg-type]
        evidence_pooling=pooling,  # type: ignore[arg-type]
    )


def _reference_local_level(
    residual: np.ndarray,
    valid: np.ndarray,
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
    evidence = np.zeros(track_count, dtype=np.float64)
    log_prior = np.log(config.inlier_prior)
    log_outlier_prior = np.log1p(-config.inlier_prior)
    for frame in range(end_frame):
        predicted_variance = variance + process_variance
        variance = predicted_variance
        mask = valid[frame]
        if not np.any(mask):
            continue
        innovation = residual[frame, mask] - mean[mask]
        predicted = predicted_variance[mask]
        inlier_s = predicted + observation_variance
        outlier_s = (
            predicted
            + observation_variance * config.outlier_variance_multiplier
        )
        squared_norm = np.sum(np.square(innovation), axis=1)
        log_inlier = log_prior - 0.5 * (
            3.0 * np.log(2.0 * np.pi * inlier_s) + squared_norm / inlier_s
        )
        log_outlier = log_outlier_prior - 0.5 * (
            3.0 * np.log(2.0 * np.pi * outlier_s) + squared_norm / outlier_s
        )
        log_mixture = np.logaddexp(log_inlier, log_outlier)
        probability = np.exp(log_inlier - log_mixture)
        evidence[mask] += log_mixture
        inlier_gain = predicted / inlier_s
        outlier_gain = predicted / outlier_s
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
    return mean, variance, final_probability, update_count, evidence


def test_single_local_level_matches_historical_filter() -> None:
    residual = np.array(
        [
            [[0.01, 0.0, 0.0], [0.0, 0.01, 0.0]],
            [[0.012, -0.001, 0.0], [0.0, 0.008, 0.0]],
            [[0.04, 0.0, 0.0], [0.0, 0.009, 0.0]],
        ]
    )
    valid = np.array([[True, True], [True, False], [True, True]])
    component = FixedBayesianAnchorConfigV1(
        process_std_m=0.001,
        observation_std_m=0.0025,
    )
    posterior = infer_dynamic_endpoint_model_average(
        residual,
        valid,
        end_frame=3,
        config=_single(component),
    )
    expected = _reference_local_level(
        residual,
        valid,
        end_frame=3,
        config=component,
    )
    mean, variance, probability, count, evidence = expected

    assert np.allclose(posterior.mean_m, mean)
    assert np.allclose(posterior.covariance_m2, variance[:, None, None] * np.eye(3))
    assert np.allclose(posterior.final_nominal_probability, probability)
    assert np.array_equal(posterior.update_count, count)
    assert np.allclose(posterior.component_log_evidence[:, 0], evidence)
    assert np.allclose(posterior.component_weights, 1.0)
    assert np.allclose(posterior.component_state_mean[0, :, 1, :], 0.0)


def test_persistence_is_exact_last_valid_residual() -> None:
    residual = np.array(
        [
            [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]],
            [[5.0, 0.0, 0.0], [0.0, 7.0, 0.0]],
            [[3.0, 4.0, 0.0], [0.0, 9.0, 1.0]],
        ]
    )
    valid = np.array([[True, True], [False, True], [True, False]])
    posterior = infer_dynamic_endpoint_model_average(
        residual,
        valid,
        end_frame=3,
        config=_single(PersistenceEndpointComponentV2(process_std_m=0.0)),
    )

    assert np.array_equal(posterior.mean_m[0], residual[2, 0])
    assert np.array_equal(posterior.mean_m[1], residual[1, 1])
    assert np.array_equal(posterior.update_count, [2, 2])
    assert np.allclose(posterior.component_state_mean[0, :, 1, :], 0.0)


def test_object_evidence_pooling_shares_weights_across_tracks() -> None:
    residual = np.zeros((4, 2, 3))
    residual[:, 0, 0] = [0.0, 0.01, 0.02, 0.03]
    residual[:, 1, 0] = [0.0, 0.0, 0.0, 0.0]
    components = (
        PersistenceEndpointComponentV2(),
        DampedTrendEndpointComponentV2(observation_std_m=0.001),
    )
    per_track = infer_dynamic_endpoint_model_average(
        residual,
        np.ones((4, 2), dtype=bool),
        end_frame=4,
        config=DynamicEndpointModelAverageConfigV2(
            components=components,
            evidence_pooling="per_track",
        ),
    )
    pooled = infer_dynamic_endpoint_model_average(
        residual,
        np.ones((4, 2), dtype=bool),
        end_frame=4,
        config=DynamicEndpointModelAverageConfigV2(
            components=components,
            evidence_pooling="object",
        ),
    )

    assert not np.allclose(
        per_track.component_weights[0],
        per_track.component_weights[1],
    )
    assert np.array_equal(pooled.component_weights[0], pooled.component_weights[1])


def test_object_pooling_without_observations_preserves_priors() -> None:
    config = DynamicEndpointModelAverageConfigV2(
        components=(
            PersistenceEndpointComponentV2(),
            FixedBayesianAnchorConfigV1(),
        ),
        component_prior_probability=(0.2, 0.8),
        evidence_pooling="object",
    )
    posterior = infer_dynamic_endpoint_model_average(
        np.zeros((3, 2, 3)),
        np.zeros((3, 2), dtype=bool),
        end_frame=3,
        config=config,
    )
    assert np.allclose(posterior.component_weights, [[0.2, 0.8], [0.2, 0.8]])
    assert not np.any(posterior.updated_mask)


def test_between_model_disagreement_is_included() -> None:
    components = (
        PersistenceEndpointComponentV2(observation_std_m=0.001),
        FixedBayesianAnchorConfigV1(
            process_std_m=0.0,
            observation_std_m=0.02,
        ),
    )
    posterior = infer_dynamic_endpoint_model_average(
        np.array([[[0.0, 0.0, 0.0]], [[0.03, 0.0, 0.0]]]),
        np.ones((2, 1), dtype=bool),
        end_frame=2,
        config=DynamicEndpointModelAverageConfigV2(components=components),
    )
    weighted_within = float(
        np.sum(
            posterior.component_weights[0]
            * posterior.component_state_covariance[:, 0, 0, 0]
        )
    )
    assert posterior.covariance_m2[0, 0, 0] > weighted_within
    assert posterior.covariance_m2[0, 0, 0] > posterior.covariance_m2[0, 1, 1]
