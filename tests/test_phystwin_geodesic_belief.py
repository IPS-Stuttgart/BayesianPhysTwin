import numpy as np

from bayesian_phystwin.deform360_geodesic_diagnostic import (
    evaluate_geodesic_decoder_arrays,
)
from bayesian_phystwin.phystwin_geodesic_belief import (
    MaterialGeodesicGraph,
    build_reference_knn_geodesic_graph,
    decode_recursive_geodesic_rbf_belief,
    deterministic_geodesic_farthest_point_ids,
    geodesic_distances_to_centers_m,
)
from bayesian_phystwin.phystwin_online_belief import (
    RecursiveRbfBeliefConfig,
    RecursiveRbfBeliefSnapshot,
    decode_recursive_rbf_belief,
    initialize_recursive_rbf_belief,
    update_recursive_rbf_belief,
)


def _chain_graph(count: int) -> MaterialGeodesicGraph:
    reference = np.stack(
        (np.linspace(0.0, 1.0, count), np.zeros(count), np.zeros(count)), axis=1
    )
    edges = np.column_stack((np.arange(count - 1), np.arange(1, count)))
    return MaterialGeodesicGraph(reference, edges)


def _local_belief(
    graph: MaterialGeodesicGraph,
    center_ids: np.ndarray,
    center_positions_m: np.ndarray,
    local_x: np.ndarray,
) -> RecursiveRbfBeliefSnapshot:
    return RecursiveRbfBeliefSnapshot(
        center_ids=center_ids,
        center_positions_m=center_positions_m,
        global_mean_m=np.zeros(3),
        global_variance_m2=np.full(3, 1e-6),
        local_mean_m=np.column_stack(
            (local_x, np.zeros(len(local_x)), np.zeros(len(local_x)))
        ),
        local_variance_m2=np.full((len(center_ids), 3), 1e-6),
        update_count=np.ones(len(center_ids), dtype=np.int64),
        last_update_frame=1,
        object_scale_m=1.0,
    )


def test_geodesic_distances_follow_material_chain_not_chord() -> None:
    graph = _chain_graph(11)

    distance = geodesic_distances_to_centers_m(graph, np.asarray([0, 5]))

    np.testing.assert_allclose(distance[10], np.asarray([1.0, 0.5]))
    np.testing.assert_allclose(distance[2], np.asarray([0.2, 0.3]))


def test_geodesic_decoder_blocks_local_leak_across_a_fold() -> None:
    graph = _chain_graph(11)
    # A hairpin places material endpoint 10 next to endpoint 0, although they
    # remain one full arc length apart on the chain.
    current = np.asarray(
        [
            [0.0, 0.00, 0.0],
            [0.1, 0.00, 0.0],
            [0.2, 0.00, 0.0],
            [0.3, 0.00, 0.0],
            [0.4, 0.00, 0.0],
            [0.5, 0.00, 0.0],
            [0.4, 0.01, 0.0],
            [0.3, 0.01, 0.0],
            [0.2, 0.01, 0.0],
            [0.1, 0.01, 0.0],
            [0.0, 0.01, 0.0],
        ]
    )
    center_ids = np.asarray([0, 5])
    belief = _local_belief(
        graph,
        center_ids,
        current[center_ids],
        np.asarray([1.0, 0.0]),
    )
    config = RecursiveRbfBeliefConfig(
        length_scale_fraction=0.15,
        local_blend=1.0,
        maximum_correction_m=2.0,
    )

    euclidean = decode_recursive_rbf_belief(
        belief, current, forecast_frames=0, config=config
    )
    geodesic = decode_recursive_geodesic_rbf_belief(
        belief,
        graph,
        np.arange(len(current)),
        forecast_frames=0,
        config=config,
    )

    assert euclidean.mean_m[10, 0] > 0.99
    assert geodesic.mean_m[10, 0] < 1e-6
    assert geodesic.mean_m[1, 0] > 0.95


def test_geodesic_decoder_blocks_local_leak_at_a_self_crossing() -> None:
    graph = _chain_graph(13)
    # Material IDs 2 and 10 occupy the same embedded position at a crossing.
    # A second active centre at ID 7 anchors the far branch to zero.
    current = graph.reference_positions_m.copy()
    current[10] = current[2]
    center_ids = np.asarray([2, 7])
    belief = _local_belief(
        graph,
        center_ids,
        current[center_ids],
        np.asarray([1.0, 0.0]),
    )
    config = RecursiveRbfBeliefConfig(
        length_scale_fraction=0.20,
        local_blend=1.0,
        maximum_correction_m=2.0,
    )

    euclidean = decode_recursive_rbf_belief(
        belief, current, forecast_frames=0, config=config
    )
    geodesic = decode_recursive_geodesic_rbf_belief(
        belief,
        graph,
        np.arange(len(current)),
        forecast_frames=0,
        config=config,
    )

    assert euclidean.mean_m[10, 0] > 0.85
    assert geodesic.mean_m[10, 0] < 0.01
    assert geodesic.mean_m[2, 0] > 0.85


def test_knn_proxy_is_order_stable_for_distance_ties() -> None:
    points = np.asarray(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    )

    graph = build_reference_knn_geodesic_graph(points, neighbor_count=1)

    np.testing.assert_array_equal(
        graph.edges,
        np.asarray([[0, 1], [0, 2], [0, 3]]),
    )
    assert graph.construction == "frame_zero_symmetric_union_1nn"


def test_geodesic_fps_spans_disconnected_components_before_refining() -> None:
    positions = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [2.0, 1.0, 0.0],
        ]
    )
    graph = MaterialGeodesicGraph(
        positions,
        np.asarray([[0, 1], [1, 2], [3, 4], [4, 5]]),
    )

    selected = deterministic_geodesic_farthest_point_ids(
        graph, np.asarray([5, 4, 3, 2, 1, 0]), 3
    )

    np.testing.assert_array_equal(selected, np.asarray([0, 3, 2]))


def test_deform_diagnostic_selects_observation_closer_backbone() -> None:
    point_count = 20
    frame_count = 76
    frame_zero = np.stack(
        (
            np.linspace(0.0, 0.19, point_count),
            np.zeros(point_count),
            np.zeros(point_count),
        ),
        axis=1,
    )
    prior = np.repeat(frame_zero[None], frame_count, axis=0).astype(np.float32)
    persistence = prior.copy()
    persistence[1:, :, 1] += 0.006
    target = prior.copy()
    target[1:, :, 1] += 0.010
    visible = np.ones((frame_count, point_count), dtype=bool)
    validity = visible.copy()
    center_ids = np.arange(16)
    updates = [
        {
            "frame": frame,
            "available_center_count": 16,
            "accepted": True,
            "causal_continuation_selected": True,
        }
        for frame in (19, 38, 57)
    ]

    report, trajectories = evaluate_geodesic_decoder_arrays(
        prior,
        persistence,
        target,
        visible,
        validity,
        center_ids=center_ids,
        update_records=updates,
        belief_config=RecursiveRbfBeliefConfig(local_blend=1.0),
        neighbor_count=2,
        scored_frames=tuple(range(20, frame_count)),
    )

    assert report["observed_backbone_selector"]["selected_by_update"] == [
        "persistence",
        "persistence",
        "persistence",
    ]
    selected = trajectories["selected_backbone_euclidean_rbf_ungated"]
    assert np.mean(np.abs(selected[20:, :, 1] - target[20:, :, 1])) < 0.001


def test_deform_diagnostic_keeps_separate_beliefs_when_backbone_switches() -> None:
    point_count = 20
    frame_count = 76
    frame_zero = np.stack(
        (
            np.linspace(0.0, 0.19, point_count),
            np.zeros(point_count),
            np.zeros(point_count),
        ),
        axis=1,
    )
    prior = np.repeat(frame_zero[None], frame_count, axis=0).astype(np.float32)
    persistence = prior.copy()
    persistence[1:, :, 1] = 0.020
    target = prior.copy()
    target[19, :, 1] = 0.001
    target[38:57, :, 1] = 0.019
    target[57:, :, 1] = 0.001
    visible = np.ones((frame_count, point_count), dtype=bool)
    centers = np.arange(16)
    updates = [
        {
            "frame": frame,
            "available_center_count": 16,
            "accepted": True,
            "causal_continuation_selected": True,
        }
        for frame in (19, 38, 57)
    ]
    config = RecursiveRbfBeliefConfig(local_blend=1.0)

    report, trajectories = evaluate_geodesic_decoder_arrays(
        prior,
        persistence,
        target,
        visible,
        visible,
        center_ids=centers,
        update_records=updates,
        belief_config=config,
        neighbor_count=2,
        scored_frames=tuple(range(20, frame_count)),
    )

    assert report["observed_backbone_selector"]["selected_by_update"] == [
        "physical_prior",
        "persistence",
        "physical_prior",
    ]
    persistence_belief = initialize_recursive_rbf_belief(
        centers,
        persistence[0, centers],
        persistence[0],
        config=config,
    )
    available = np.ones(len(centers), dtype=bool)
    for frame in (19, 38):
        residual = target[frame, centers] - persistence[frame, centers]
        persistence_belief, _ = update_recursive_rbf_belief(
            persistence_belief,
            frame,
            persistence[frame, centers],
            residual,
            available,
            config=config,
        )
    decoded = decode_recursive_rbf_belief(
        persistence_belief,
        persistence[38],
        forecast_frames=1,
        config=config,
    )
    expected = (persistence[39] + decoded.mean_m).astype(np.float32)
    np.testing.assert_array_equal(
        trajectories["selected_backbone_euclidean_rbf_ungated"][39],
        expected,
    )
