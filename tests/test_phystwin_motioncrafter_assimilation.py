import numpy as np

from bayesian_phystwin.phystwin_motioncrafter_assimilation import (
    AnonymousSceneFlowConfig,
    FramewiseGraphObservations,
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
