import numpy as np
import pytest

from bayesian_phystwin.causal4d_belief_provider_v2 import (
    CAUSAL4D_BELIEF_PROVIDER_V2_API_VERSION,
    causal4d_belief_provider_v2_manifest,
    infer_model_averaged_bayesian_anchor_endpoint,
)
from bayesian_phystwin.contracts.fixed_anchor import FixedBayesianAnchorConfigV1
from bayesian_phystwin.endpoint_model_average import (
    ModelAveragedEndpointConfigV1,
    infer_model_averaged_endpoint,
    predict_model_averaged_endpoint,
)
from bayesian_phystwin.phystwin_bayesian_anchor import robust_random_walk_endpoint


def _single_config(
    component: FixedBayesianAnchorConfigV1,
) -> ModelAveragedEndpointConfigV1:
    return ModelAveragedEndpointConfigV1(components=(component,))


def test_single_component_matches_fixed_endpoint_exactly() -> None:
    residual = np.array(
        [
            [[0.01, 0.0, 0.0], [0.0, 0.01, 0.0]],
            [[0.012, 0.0, 0.0], [0.0, 0.008, 0.0]],
            [[0.011, 0.0, 0.0], [0.0, 0.009, 0.0]],
        ]
    )
    valid = np.array([[True, True], [True, False], [True, True]])
    component = FixedBayesianAnchorConfigV1(
        process_std_m=0.001,
        observation_std_m=0.0025,
    )
    posterior = infer_model_averaged_endpoint(
        residual,
        valid,
        end_frame=3,
        config=_single_config(component),
    )
    reference = robust_random_walk_endpoint(
        residual,
        valid,
        end_frame=3,
        process_variance=component.process_std_m**2,
        observation_variance=component.observation_std_m**2,
        initial_variance=component.initial_std_m**2,
        inlier_prior=component.inlier_prior,
        outlier_variance_multiplier=component.outlier_variance_multiplier,
    )

    assert np.array_equal(posterior.update_count, reference.update_count)
    assert np.allclose(posterior.mean_m, reference.mean)
    assert np.allclose(
        posterior.component_variance_m2[0],
        reference.variance,
    )
    assert np.allclose(
        posterior.final_nominal_probability,
        reference.final_inlier_probability,
    )
    assert np.allclose(posterior.component_weights, 1.0)
    expected = reference.variance[:, None, None] * np.eye(3)
    assert np.allclose(posterior.covariance_m2, expected)


def test_between_model_disagreement_increases_covariance() -> None:
    residual = np.array(
        [
            [[0.0, 0.0, 0.0]],
            [[0.01, 0.0, 0.0]],
            [[0.02, 0.0, 0.0]],
        ]
    )
    valid = np.ones((3, 1), dtype=bool)
    config = ModelAveragedEndpointConfigV1(
        components=(
            FixedBayesianAnchorConfigV1(
                process_std_m=0.0,
                observation_std_m=0.001,
            ),
            FixedBayesianAnchorConfigV1(
                process_std_m=0.01,
                observation_std_m=0.01,
            ),
        )
    )
    posterior = infer_model_averaged_endpoint(
        residual,
        valid,
        end_frame=3,
        config=config,
    )

    weighted_within_x = float(
        np.sum(
            posterior.component_weights[0]
            * posterior.component_variance_m2[:, 0]
        )
    )
    assert posterior.covariance_m2[0, 0, 0] >= weighted_within_x
    assert posterior.covariance_m2[0, 0, 0] > posterior.covariance_m2[0, 1, 1]


def test_no_observations_preserve_model_prior_weights() -> None:
    config = ModelAveragedEndpointConfigV1(
        components=(
            FixedBayesianAnchorConfigV1(process_std_m=0.0),
            FixedBayesianAnchorConfigV1(process_std_m=0.001),
        ),
        component_prior_probability=(0.25, 0.75),
    )
    posterior = infer_model_averaged_endpoint(
        np.zeros((4, 2, 3)),
        np.zeros((4, 2), dtype=bool),
        end_frame=4,
        config=config,
    )

    assert np.allclose(posterior.component_weights, [[0.25, 0.75]] * 2)
    assert np.allclose(posterior.mean_m, 0.0)
    assert not np.any(posterior.updated_mask)


def test_horizon_prediction_increases_covariance_trace() -> None:
    config = ModelAveragedEndpointConfigV1(
        components=(
            FixedBayesianAnchorConfigV1(process_std_m=0.001),
            FixedBayesianAnchorConfigV1(process_std_m=0.005),
        )
    )
    posterior = infer_model_averaged_endpoint(
        np.zeros((2, 1, 3)),
        np.ones((2, 1), dtype=bool),
        end_frame=2,
        config=config,
    )
    now = predict_model_averaged_endpoint(posterior, horizon_steps=0)
    future = predict_model_averaged_endpoint(posterior, horizon_steps=20)

    assert np.allclose(now.mean_m, future.mean_m)
    assert np.trace(future.covariance_m2[0]) > np.trace(now.covariance_m2[0])


def test_result_arrays_are_read_only() -> None:
    posterior = infer_model_averaged_endpoint(
        np.zeros((1, 1, 3)),
        np.ones((1, 1), dtype=bool),
        end_frame=1,
        config=_single_config(FixedBayesianAnchorConfigV1()),
    )
    assert not posterior.mean_m.flags.writeable
    assert not posterior.covariance_m2.flags.writeable
    assert not posterior.component_weights.flags.writeable
    with pytest.raises(ValueError):
        posterior.mean_m[0, 0] = 1.0


def test_default_configuration_uses_historical_noise_grid() -> None:
    config = ModelAveragedEndpointConfigV1()
    assert len(config.components) == 15
    assert np.isclose(sum(config.component_prior_probability), 1.0)


def test_invalid_configurations_fail() -> None:
    with pytest.raises(ValueError, match="at least one"):
        ModelAveragedEndpointConfigV1(components=())
    with pytest.raises(ValueError, match="unique"):
        ModelAveragedEndpointConfigV1(
            components=(FixedBayesianAnchorConfigV1(),) * 2
        )
    with pytest.raises(ValueError, match="component count"):
        ModelAveragedEndpointConfigV1(
            components=(FixedBayesianAnchorConfigV1(),),
            component_prior_probability=(0.5, 0.5),
        )


def test_provider_v2_wraps_endpoint_and_declares_calibration_boundary() -> None:
    posterior = infer_model_averaged_bayesian_anchor_endpoint(
        np.zeros((2, 1, 3)),
        np.ones((2, 1), dtype=bool),
        end_frame=2,
        config=_single_config(FixedBayesianAnchorConfigV1()),
    )
    manifest = causal4d_belief_provider_v2_manifest(
        provider_revision="revision-test"
    )

    assert posterior.update_count[0] == 2
    assert manifest["schema_version"] == CAUSAL4D_BELIEF_PROVIDER_V2_API_VERSION
    assert "horizon_dependent_predictive_covariance" in manifest["capabilities"]
    assert "calibration" in manifest["metadata"]["raw_covariance_claim"]


def test_invalid_inputs_fail_closed() -> None:
    with pytest.raises(ValueError, match="shape"):
        infer_model_averaged_endpoint(
            np.zeros((2, 3)),
            np.zeros(2, dtype=bool),
            end_frame=1,
        )
    with pytest.raises(ValueError, match="integer"):
        infer_model_averaged_endpoint(
            np.zeros((2, 1, 3)),
            np.zeros((2, 1), dtype=bool),
            end_frame=1.5,
        )
    posterior = infer_model_averaged_endpoint(
        np.zeros((2, 1, 3)),
        np.zeros((2, 1), dtype=bool),
        end_frame=1,
        config=_single_config(FixedBayesianAnchorConfigV1()),
    )
    with pytest.raises(ValueError, match="nonnegative integer"):
        predict_model_averaged_endpoint(posterior, horizon_steps=-1)
