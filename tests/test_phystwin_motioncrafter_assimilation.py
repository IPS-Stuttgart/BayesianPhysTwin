import numpy as np

from bayesian_phystwin.phystwin_motioncrafter_assimilation import (
    AnonymousSceneFlowConfig,
    FramewiseGraphObservations,
    _covariance_intersection_pair,
    associate_anonymous_scene_flow,
    combine_framewise_graph_observations,
    graph_regularized_state_observations,
)
from bayesian_phystwin.phystwin_motioncrafter_association import (
    MotionCrafterPrediction,
)


def _prediction(
    point_map: np.ndarray,
    *,
    valid_mask: np.ndarray | None = None,
    scene_flow: np.ndarray | None = None,
    deform_mask: np.ndarray | None = None,
    point_covariance_m2: np.ndarray | None = None,
    flow_covariance_m2: np.ndarray | None = None,
    contributors: np.ndarray | None = None,
) -> MotionCrafterPrediction:
    shape = point_map.shape[:3]
    return MotionCrafterPrediction(
        point_map=np.asarray(point_map, dtype=np.float32),
        valid_mask=(
            np.ones(shape, dtype=bool)
            if valid_mask is None
            else np.asarray(valid_mask, dtype=bool)
        ),
        scene_flow=(
            np.zeros_like(point_map, dtype=np.float32)
            if scene_flow is None
            else np.asarray(scene_flow, dtype=np.float32)
        ),
        deform_mask=(
            np.ones(shape, dtype=bool)
            if deform_mask is None
            else np.asarray(deform_mask, dtype=bool)
        ),
        point_covariance_m2=point_covariance_m2,
        flow_covariance_m2=flow_covariance_m2,
        contributors=contributors,
    )


def test_framewise_association_recovers_after_an_occluded_frame() -> None:
    graph = np.array(
        [
            [[0.00, 0.0, 0.0], [0.02, 0.0, 0.0]],
            [[0.01, 0.0, 0.0], [0.03, 0.0, 0.0]],
            [[0.02, 0.0, 0.0], [0.04, 0.0, 0.0]],
        ]
    )
    point_map = graph.reshape(3, 1, 2, 3).copy()
    valid = np.ones((3, 1, 2), dtype=bool)
    valid[1, 0, 0] = False
    observations = associate_anonymous_scene_flow(
        _prediction(point_map, valid_mask=valid),
        np.ones((3, 1, 2), dtype=bool),
        graph,
        config=AnonymousSceneFlowConfig(
            measurement_stride_pixels=1,
            candidate_count=1,
            minimum_observation_mass=0.1,
        ),
    )

    assert observations.valid[0, 0]
    assert not observations.valid[1, 0]
    assert observations.valid[2, 0]
    np.testing.assert_allclose(observations.positions[2, 0], graph[2, 0])


def test_scene_flow_disambiguates_nearby_graph_vertices() -> None:
    graph = np.array(
        [
            [[0.000, 0.0, 0.0], [0.002, 0.0, 0.0]],
            [[0.020, 0.0, 0.0], [-0.018, 0.0, 0.0]],
        ]
    )
    point_map = np.array(
        [
            [[[0.001, 0.0, 0.0]]],
            [[[0.020, 0.0, 0.0]]],
        ]
    )
    scene_flow = np.zeros_like(point_map)
    scene_flow[0, 0, 0, 0] = 0.019
    prediction = _prediction(point_map, scene_flow=scene_flow)
    common = dict(
        measurement_stride_pixels=1,
        candidate_count=2,
        position_scale_m=0.01,
        flow_scale_m=0.002,
        maximum_position_error_m=0.1,
        maximum_flow_endpoint_error_m=0.1,
        minimum_observation_mass=0.05,
        entropy_strength=0.0,
    )

    position_only = associate_anonymous_scene_flow(
        prediction,
        np.ones((2, 1, 1), dtype=bool),
        graph,
        config=AnonymousSceneFlowConfig(flow_strength=0.0, **common),
    )
    position_flow = associate_anonymous_scene_flow(
        prediction,
        np.ones((2, 1, 1), dtype=bool),
        graph,
        config=AnonymousSceneFlowConfig(flow_strength=1.0, **common),
    )

    np.testing.assert_allclose(position_only.measurement_mass[0], [0.5, 0.5], atol=1e-6)
    assert position_flow.measurement_mass[0, 0] > 0.999
    assert position_flow.measurement_mass[0, 1] < 0.001


def test_inconsistent_flow_is_rejected_as_an_outlier() -> None:
    graph = np.array([[[0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0]]])
    point_map = graph.reshape(2, 1, 1, 3).copy()
    scene_flow = np.zeros_like(point_map)
    scene_flow[0, 0, 0, 0] = 0.2
    prediction = _prediction(point_map, scene_flow=scene_flow)
    common = dict(
        measurement_stride_pixels=1,
        candidate_count=1,
        maximum_flow_endpoint_error_m=0.02,
        minimum_observation_mass=0.1,
    )

    position_only = associate_anonymous_scene_flow(
        prediction,
        np.ones((2, 1, 1), dtype=bool),
        graph,
        config=AnonymousSceneFlowConfig(flow_strength=0.0, **common),
    )
    position_flow = associate_anonymous_scene_flow(
        prediction,
        np.ones((2, 1, 1), dtype=bool),
        graph,
        config=AnonymousSceneFlowConfig(flow_strength=1.0, **common),
    )

    assert position_only.valid[0, 0]
    assert not position_flow.valid[0, 0]
    assert position_flow.accepted_measurement_count[0] == 0


def test_multiview_fusion_suppresses_a_disagreeing_camera() -> None:
    def observation(x_position: float) -> FramewiseGraphObservations:
        position = np.array([[[x_position, 0.0, 0.0]]], dtype=np.float32)
        valid = np.ones((1, 1), dtype=bool)
        reliability = np.full((1, 1), 0.9, dtype=np.float32)
        diagnostic = np.zeros((1, 1), dtype=np.float32)
        count = np.ones(1, dtype=np.int32)
        return FramewiseGraphObservations(
            positions=position,
            flow_endpoints=position.copy(),
            valid=valid,
            flow_valid=valid.copy(),
            reliability=reliability,
            flow_reliability=reliability.copy(),
            measurement_mass=np.ones((1, 1), dtype=np.float32),
            flow_measurement_mass=np.ones((1, 1), dtype=np.float32),
            normalized_entropy=diagnostic.copy(),
            position_error_m=diagnostic.copy(),
            flow_endpoint_error_m=diagnostic.copy(),
            sampled_measurement_count=count.copy(),
            accepted_measurement_count=count.copy(),
        )

    fused = combine_framewise_graph_observations(
        {0: observation(0.0), 1: observation(0.001), 2: observation(0.1)},
        consistency_scale_m=0.015,
    )

    assert fused.valid[0, 0]
    assert abs(float(fused.positions[0, 0, 0]) - 0.0005) < 1e-4
    assert fused.reliability[0, 0] > 0.9


def test_graph_regularization_propagates_a_constant_innovation() -> None:
    graph = np.zeros((1, 3, 3), dtype=float)
    point_map = np.array([[[[0.001, 0.0, 0.0], [0.001, 0.0, 0.0]]]])
    graph[0, :, 0] = [0.0, 0.01, 0.02]
    point_map[0, 0, :, 0] += [0.0, 0.02]
    observations = associate_anonymous_scene_flow(
        _prediction(point_map),
        np.ones((1, 1, 2), dtype=bool),
        graph,
        config=AnonymousSceneFlowConfig(
            measurement_stride_pixels=1,
            candidate_count=1,
            maximum_position_error_m=0.005,
            minimum_observation_mass=0.1,
        ),
    )

    regularized = graph_regularized_state_observations(
        graph,
        observations,
        np.array([[0, 1], [1, 2]]),
        prior_strength=10.0,
    )

    assert np.all(regularized.valid)
    np.testing.assert_allclose(regularized.correction[0, :, 0], 0.001, atol=2e-6)
    np.testing.assert_allclose(regularized.correction[0, :, 1:], 0.0, atol=1e-9)


def test_decoupled_prior_reliability_does_not_reuse_state_residual() -> None:
    point_map = np.zeros((2, 1, 1, 3), dtype=float)
    masks = np.ones((2, 1, 1), dtype=bool)
    near_graph = np.zeros((2, 1, 3), dtype=float)
    far_graph = near_graph.copy()
    far_graph[..., 0] = 0.03
    config = AnonymousSceneFlowConfig(
        reliability_mode="decoupled_robust",
        measurement_stride_pixels=1,
        candidate_count=1,
        maximum_position_error_m=0.1,
        minimum_observation_mass=0.1,
        boundary_reliability_floor=1.0,
        observation_variance_floor_m2=1e-8,
    )

    near = associate_anonymous_scene_flow(
        _prediction(point_map), masks, near_graph, config=config
    )
    far = associate_anonymous_scene_flow(
        _prediction(point_map), masks, far_graph, config=config
    )

    np.testing.assert_allclose(near.prior_reliability, far.prior_reliability)
    assert float(far.reliability[0, 0]) < float(near.reliability[0, 0])


def test_duplicate_pixels_in_one_correlation_block_do_not_inflate_confidence() -> None:
    graph = np.zeros((1, 1, 3), dtype=float)
    masks = np.ones((1, 4, 4), dtype=bool)
    sparse_valid = np.zeros((1, 4, 4), dtype=bool)
    sparse_valid[0, 0, 0] = True
    dense_valid = np.ones((1, 4, 4), dtype=bool)
    point_map = np.zeros((1, 4, 4, 3), dtype=float)
    covariance = np.broadcast_to(1e-4 * np.eye(3), (1, 4, 4, 3, 3)).copy()
    config = AnonymousSceneFlowConfig(
        reliability_mode="decoupled_robust",
        measurement_stride_pixels=1,
        candidate_count=1,
        minimum_observation_mass=0.1,
        correlation_block_pixels=16,
        boundary_reliability_floor=1.0,
        observation_variance_floor_m2=1e-12,
    )

    sparse = associate_anonymous_scene_flow(
        _prediction(
            point_map,
            valid_mask=sparse_valid,
            point_covariance_m2=covariance,
        ),
        masks,
        graph,
        config=config,
    )
    duplicated = associate_anonymous_scene_flow(
        _prediction(
            point_map,
            valid_mask=dense_valid,
            point_covariance_m2=covariance,
        ),
        masks,
        graph,
        config=config,
    )

    np.testing.assert_allclose(sparse.effective_sample_size, 1.0)
    np.testing.assert_allclose(duplicated.effective_sample_size, 1.0)
    np.testing.assert_allclose(sparse.prior_reliability, duplicated.prior_reliability)
    np.testing.assert_allclose(sparse.reliability, duplicated.reliability, atol=1e-7)
    np.testing.assert_allclose(
        sparse.observation_covariance_m2,
        duplicated.observation_covariance_m2,
        atol=1e-10,
    )


def test_assignment_mixture_spread_remains_in_metric_covariance() -> None:
    graph = np.array([[[-0.01, 0.0, 0.0], [0.01, 0.0, 0.0]]])
    point_map = np.zeros((1, 1, 1, 3), dtype=float)
    base_config = dict(
        reliability_mode="decoupled_robust",
        measurement_stride_pixels=1,
        maximum_position_error_m=0.1,
        minimum_observation_mass=0.1,
        position_scale_m=0.01,
        boundary_reliability_floor=1.0,
        observation_variance_floor_m2=1e-8,
    )
    ambiguous = associate_anonymous_scene_flow(
        _prediction(point_map),
        np.ones((1, 1, 1), dtype=bool),
        graph,
        config=AnonymousSceneFlowConfig(candidate_count=2, **base_config),
    )
    unambiguous = associate_anonymous_scene_flow(
        _prediction(point_map),
        np.ones((1, 1, 1), dtype=bool),
        graph,
        config=AnonymousSceneFlowConfig(candidate_count=1, **base_config),
    )

    ambiguous_covariance = ambiguous.observation_covariance_m2[0, ambiguous.valid[0]]
    unambiguous_covariance = unambiguous.observation_covariance_m2[
        0, unambiguous.valid[0]
    ]
    assert len(ambiguous_covariance) == 2
    assert len(unambiguous_covariance) == 1
    assert np.all(
        ambiguous_covariance[:, 0, 0] > unambiguous_covariance[0, 0, 0] + 5e-5
    )
    np.testing.assert_allclose(
        ambiguous_covariance[:, 1, 1],
        unambiguous_covariance[0, 1, 1],
        rtol=1e-5,
    )


def test_covariance_intersection_is_not_as_confident_as_independence() -> None:
    covariance = np.eye(3)
    _, fused_covariance, _ = _covariance_intersection_pair(
        np.zeros(3), covariance, np.zeros(3), covariance
    )
    naive_independent = np.linalg.inv(
        np.linalg.inv(covariance) + np.linalg.inv(covariance)
    )

    assert np.all(np.linalg.eigvalsh(fused_covariance - naive_independent) >= -1e-10)


def test_duplicate_camera_does_not_create_independent_confidence() -> None:
    position = np.zeros((1, 1, 3), dtype=np.float32)
    valid = np.ones((1, 1), dtype=bool)
    reliability = np.full((1, 1), 0.9, dtype=np.float32)
    covariance = np.broadcast_to(1e-4 * np.eye(3), (1, 1, 3, 3)).astype(np.float32)
    diagnostic = np.zeros((1, 1), dtype=np.float32)
    count = np.ones(1, dtype=np.int32)
    observation = FramewiseGraphObservations(
        positions=position,
        flow_endpoints=position.copy(),
        valid=valid,
        flow_valid=valid.copy(),
        reliability=reliability,
        flow_reliability=reliability.copy(),
        measurement_mass=np.ones((1, 1), dtype=np.float32),
        flow_measurement_mass=np.ones((1, 1), dtype=np.float32),
        normalized_entropy=diagnostic.copy(),
        position_error_m=diagnostic.copy(),
        flow_endpoint_error_m=diagnostic.copy(),
        sampled_measurement_count=count.copy(),
        accepted_measurement_count=count.copy(),
        prior_reliability=reliability.copy(),
        flow_prior_reliability=reliability.copy(),
        observation_covariance_m2=covariance,
        flow_observation_covariance_m2=covariance.copy(),
        effective_sample_size=np.ones((1, 1), dtype=np.float32),
        flow_effective_sample_size=np.ones((1, 1), dtype=np.float32),
    )

    fused = combine_framewise_graph_observations(
        {0: observation, 1: observation},
        fusion_mode="covariance_intersection",
    )

    assert fused.reliability[0, 0] <= 0.9 + 1e-6
    assert fused.prior_reliability[0, 0] <= 0.9 + 1e-6
    assert fused.observation_covariance_m2[0, 0, 0, 0] >= (
        covariance[0, 0, 0, 0] * (1.0 - 1e-6)
    )


def test_metric_variance_controls_graph_smoothing_weight() -> None:
    graph = np.zeros((1, 3, 3), dtype=float)
    positions = graph.copy()
    positions[0, 0, 0] = 0.001
    positions[0, 2, 0] = 0.02
    valid = np.array([[True, False, True]])
    covariance = np.broadcast_to(1e-6 * np.eye(3), (1, 3, 3, 3)).copy()
    covariance[0, 2] *= 1000.0
    diagnostic = np.zeros((1, 3), dtype=np.float32)
    observations = FramewiseGraphObservations(
        positions=positions.astype(np.float32),
        flow_endpoints=positions.astype(np.float32),
        valid=valid,
        flow_valid=valid.copy(),
        reliability=np.ones((1, 3), dtype=np.float32),
        flow_reliability=np.ones((1, 3), dtype=np.float32),
        measurement_mass=np.ones((1, 3), dtype=np.float32),
        flow_measurement_mass=np.ones((1, 3), dtype=np.float32),
        normalized_entropy=diagnostic.copy(),
        position_error_m=diagnostic.copy(),
        flow_endpoint_error_m=diagnostic.copy(),
        sampled_measurement_count=np.ones(1, dtype=np.int32),
        accepted_measurement_count=np.ones(1, dtype=np.int32),
        observation_covariance_m2=covariance,
    )

    regularized = graph_regularized_state_observations(
        graph,
        observations,
        np.array([[0, 1], [1, 2]]),
        prior_strength=10.0,
        covariance_probes=32,
    )

    assert regularized.correction[0, 1, 0] < 0.005
    assert regularized.marginal_variance_m2 is not None
    assert np.all(regularized.marginal_variance_m2 >= 0.0)


def test_selected_graph_covariance_is_exposed_without_random_probes() -> None:
    graph = np.zeros((1, 2, 3), dtype=float)
    graph[0, 1, 0] = 1.0
    observations = FramewiseGraphObservations(
        positions=graph.astype(np.float32),
        flow_endpoints=graph.astype(np.float32),
        valid=np.ones((1, 2), dtype=bool),
        flow_valid=np.ones((1, 2), dtype=bool),
        reliability=np.ones((1, 2), dtype=np.float32),
        flow_reliability=np.ones((1, 2), dtype=np.float32),
        measurement_mass=np.ones((1, 2), dtype=np.float32),
        flow_measurement_mass=np.ones((1, 2), dtype=np.float32),
        normalized_entropy=np.zeros((1, 2), dtype=np.float32),
        position_error_m=np.zeros((1, 2), dtype=np.float32),
        flow_endpoint_error_m=np.zeros((1, 2), dtype=np.float32),
        sampled_measurement_count=np.ones(1, dtype=np.int32),
        accepted_measurement_count=np.ones(1, dtype=np.int32),
        observation_covariance_m2=np.broadcast_to(
            np.eye(3, dtype=np.float32), (1, 2, 3, 3)
        ).copy(),
    )

    result = graph_regularized_state_observations(
        graph,
        observations,
        np.array([[0, 1]], dtype=np.int64),
        prior_strength=0.2,
        covariance_node_indices=np.array([1]),
    )

    assert result.marginal_variance_m2 is not None
    assert np.isnan(result.marginal_variance_m2[0, 0])
    assert np.isfinite(result.marginal_variance_m2[0, 1])
