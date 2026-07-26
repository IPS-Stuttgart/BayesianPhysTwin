import numpy as np
import pytest

from bayesian_phystwin.gauge_aware_belief import GaugeAwareBeliefConfig
from bayesian_phystwin.observation_belief import ObservationBeliefV1
from bayesian_phystwin.recursive_gauge_rbf_belief import (
    RecursiveGaugeRbfConfig,
    decode_recursive_gauge_rbf_belief,
    initialize_recursive_gauge_rbf_belief,
    predict_recursive_gauge_rbf_belief,
    recursive_gauge_rbf_state_jacobian,
    select_recursive_gauge_rbf_candidate,
    update_recursive_gauge_rbf_belief,
)


def _config(*, local_blend: float = 1.0) -> RecursiveGaugeRbfConfig:
    return RecursiveGaugeRbfConfig(
        length_scale_fraction=0.05,
        local_blend=local_blend,
        global_prior_std_m=0.05,
        local_prior_std_m=0.05,
        global_process_std_m_per_sqrt_frame=0.001,
        local_process_std_m_per_sqrt_frame=0.001,
        gauge_update=GaugeAwareBeliefConfig(
            state_prior_std_m=0.05,
            shared_bias_prior_std_m=0.05,
            view_bias_prior_std_m=0.02,
            effective_samples_per_correlation_group=8.0,
            maximum_state_update_m=0.08,
            maximum_update_to_physical_response_ratio=4.0,
        ),
    )


def _geometry() -> tuple[np.ndarray, np.ndarray]:
    centers = np.asarray([[-0.5, 0.0, 1.0], [0.5, 0.0, 1.0]])
    rows = centers[[0, 0, 1, 1]]
    return centers, rows


def _belief(
    mean_xyz_m: np.ndarray,
    *,
    frame: int,
    reliability: float = 1.0,
) -> ObservationBeliefV1:
    count = len(mean_xyz_m)
    return ObservationBeliefV1(
        case_id="synthetic",
        stream_id=f"two-view-frame-{frame}",
        causal_frame_stop=frame + 1,
        view_names=("camera-0", "camera-1"),
        window_names=("window",),
        factor_names=(),
        source_repository="synthetic",
        source_revision="a" * 40,
        source_artifact_sha256="b" * 64,
        declared_frame_ids=np.asarray([frame]),
        mean_xyz_m=mean_xyz_m,
        frame_ids=np.full(count, frame),
        entity_ids=np.asarray([0, 0, 1, 1]),
        view_indices=np.asarray([0, 1, 0, 1]),
        window_indices=np.zeros(count, dtype=int),
        correlation_group_ids=np.asarray([0, 1, 0, 1]),
        factor_group_ids=np.zeros(count, dtype=int),
        prior_reliability=np.full(count, reliability),
        association_probability=np.full(count, 0.5),
        local_covariance_m2=np.tile(
            np.eye(3)[None] * 1e-6,
            (count, 1, 1),
        ),
        low_rank_factor_m=np.zeros((count, 3, 0)),
        group_ids=np.asarray([0, 1]),
        group_prior_nominal_probability=np.ones(2),
        group_composite_weight=np.ones(2),
    )


def _initial(config: RecursiveGaugeRbfConfig):
    centers, _ = _geometry()
    return initialize_recursive_gauge_rbf_belief(
        np.asarray([0, 1]),
        centers,
        centers,
        config=config,
    )


def test_local_deformation_is_separated_from_shared_camera_bias() -> None:
    config = _config()
    centers, physical = _geometry()
    prior = _initial(config)
    jacobian = recursive_gauge_rbf_state_jacobian(
        prior,
        physical,
        config=config,
    )
    true_state = np.zeros(prior.state_dimension)
    true_state[3] = 0.012
    true_state[6] = -0.012
    physical_correction = np.einsum(
        "nci,i->nc",
        jacobian,
        true_state,
    )
    shared_bias = np.asarray([0.030, -0.010, 0.0])
    observed = physical + physical_correction + shared_bias

    update = update_recursive_gauge_rbf_belief(
        prior,
        frame_index=1,
        center_positions_m=centers,
        observation_belief=_belief(observed, frame=1),
        physical_prediction_xyz_m=physical,
        query_positions_m=centers,
        physical_response_scale_m=0.05,
        config=config,
    )

    assert update.accepted
    np.testing.assert_allclose(
        update.query_prediction.mean_m[:, 0],
        [0.012, -0.012],
        atol=1.5e-3,
    )
    assert update.gauge_result.shared_bias_coefficients[0] == pytest.approx(
        0.030,
        abs=2e-3,
    )
    assert (
        update.adapter_summary["association_used_as_prior_reliability"]
        is False
    )


def test_unanchored_global_translation_abstains_bit_exact() -> None:
    config = _config(local_blend=0.0)
    centers, physical = _geometry()
    prior = _initial(config)
    observed = physical + np.asarray([0.030, 0.0, 0.0])

    update = update_recursive_gauge_rbf_belief(
        prior,
        frame_index=1,
        center_positions_m=centers,
        observation_belief=_belief(observed, frame=1),
        physical_prediction_xyz_m=physical,
        query_positions_m=centers,
        physical_response_scale_m=0.05,
        config=config,
    )
    baseline = centers.astype(np.float32)
    selected = select_recursive_gauge_rbf_candidate(baseline, update)

    assert not update.accepted
    assert update.reason == "no-identifiable-query-state"
    assert selected.dtype == baseline.dtype
    assert selected.tobytes() == baseline.tobytes()


def test_independent_anchor_identifies_global_translation() -> None:
    config = _config(local_blend=0.0)
    centers, physical = _geometry()
    prior = _initial(config)
    observed = physical + np.asarray([0.030, 0.0, 0.0])
    anchor_position = centers[:1]

    update = update_recursive_gauge_rbf_belief(
        prior,
        frame_index=1,
        center_positions_m=centers,
        observation_belief=_belief(observed, frame=1),
        physical_prediction_xyz_m=physical,
        query_positions_m=centers,
        physical_response_scale_m=0.05,
        config=config,
        anchor_observation_xyz_m=anchor_position
        + np.asarray([0.010, 0.0, 0.0]),
        anchor_physical_prediction_xyz_m=anchor_position,
        anchor_positions_m=anchor_position,
        anchor_covariance_m2=np.asarray([np.eye(3) * 1e-7]),
    )

    assert update.accepted
    np.testing.assert_allclose(
        update.query_prediction.mean_m[:, 0],
        0.010,
        atol=8e-4,
    )
    assert update.gauge_result.shared_bias_coefficients[0] == pytest.approx(
        0.020,
        abs=1.5e-3,
    )


def test_action_transition_reverses_retained_local_state() -> None:
    config = _config()
    centers, physical = _geometry()
    prior = _initial(config)
    jacobian = recursive_gauge_rbf_state_jacobian(
        prior,
        physical,
        config=config,
    )
    true_state = np.zeros(prior.state_dimension)
    true_state[3] = 0.010
    true_state[6] = -0.010
    observed = physical + np.einsum("nci,i->nc", jacobian, true_state)
    first = update_recursive_gauge_rbf_belief(
        prior,
        frame_index=1,
        center_positions_m=centers,
        observation_belief=_belief(observed, frame=1),
        physical_prediction_xyz_m=physical,
        query_positions_m=centers,
        physical_response_scale_m=0.05,
        config=config,
    )
    assert first.accepted

    transition = np.eye(prior.state_dimension)
    transition[3:, 3:] *= -1.0
    reversed_snapshot = predict_recursive_gauge_rbf_belief(
        first.posterior_snapshot,
        frame_index=2,
        center_positions_m=centers,
        config=config,
        state_transition=transition,
        process_covariance_m2=np.zeros(
            (prior.state_dimension, prior.state_dimension)
        ),
    )
    reversed_prediction = decode_recursive_gauge_rbf_belief(
        reversed_snapshot,
        centers,
        config=config,
    )

    np.testing.assert_allclose(
        reversed_prediction.mean_m,
        -first.query_prediction.mean_m,
        atol=1e-12,
    )


def test_rejected_second_update_retains_predicted_belief_but_selects_baseline() -> None:
    config = _config()
    centers, physical = _geometry()
    prior = _initial(config)
    jacobian = recursive_gauge_rbf_state_jacobian(
        prior,
        physical,
        config=config,
    )
    state = np.zeros(prior.state_dimension)
    state[3] = 0.008
    state[6] = -0.008
    observed = physical + np.einsum("nci,i->nc", jacobian, state)
    first = update_recursive_gauge_rbf_belief(
        prior,
        frame_index=1,
        center_positions_m=centers,
        observation_belief=_belief(observed, frame=1),
        physical_prediction_xyz_m=physical,
        query_positions_m=centers,
        physical_response_scale_m=0.05,
        config=config,
    )
    second = update_recursive_gauge_rbf_belief(
        first.posterior_snapshot,
        frame_index=3,
        center_positions_m=centers,
        observation_belief=_belief(
            physical,
            frame=3,
            reliability=0.0,
        ),
        physical_prediction_xyz_m=physical,
        query_positions_m=centers,
        physical_response_scale_m=0.05,
        config=config,
    )
    baseline = np.asarray([[1.0, -0.0, 2.0], [3.0, 4.0, 5.0]], np.float32)
    selected = select_recursive_gauge_rbf_candidate(baseline, second)

    assert not second.accepted
    assert second.reason == "no-observation-support"
    assert selected.tobytes() == baseline.tobytes()
    np.testing.assert_allclose(
        second.posterior_snapshot.coefficient_mean_m,
        second.predicted_snapshot.coefficient_mean_m,
    )
    assert np.trace(second.posterior_snapshot.coefficient_covariance_m2) > (
        np.trace(first.posterior_snapshot.coefficient_covariance_m2)
    )


def test_full_covariance_propagates_through_nonorthogonal_transition() -> None:
    config = _config()
    centers, _ = _geometry()
    prior = _initial(config)
    dimension = prior.state_dimension
    transition = np.eye(dimension)
    transition[0, 3] = 0.5
    process = np.eye(dimension) * 2e-6

    predicted = predict_recursive_gauge_rbf_belief(
        prior,
        frame_index=1,
        center_positions_m=centers,
        config=config,
        state_transition=transition,
        process_covariance_m2=process,
    )
    expected = (
        transition
        @ prior.coefficient_covariance_m2
        @ transition.T
        + process
    )

    np.testing.assert_allclose(
        predicted.coefficient_covariance_m2,
        expected,
    )
    assert predicted.coefficient_covariance_m2[0, 3] != 0.0
