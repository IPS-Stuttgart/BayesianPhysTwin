from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from bayesian_phystwin.phystwin_graph import PhysTwinSpringGraphConfig
from causal4d_public.deform360_reusable_graph import (
    ReusableGraphRegistrationConfig,
    build_canonical_deform360_graph,
    build_registered_phystwin_graph,
    canonical_reference_registration,
    deterministic_farthest_point_indices,
    episode_registration_summary,
    register_canonical_graph_to_episode,
    registered_episode_data,
)


SPRINGS = PhysTwinSpringGraphConfig(
    object_radius=0.05,
    object_max_neighbours=8,
    controller_radius=0.04,
    controller_max_neighbours=1,
)


def _reference() -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(
        [
            [0.00, 0.00, 0.00],
            [0.02, 0.00, 0.00],
            [0.04, 0.01, 0.00],
            [0.06, 0.01, 0.01],
            [0.08, 0.02, 0.01],
            [0.10, 0.03, 0.02],
            [0.01, 0.01, 0.01],
            [0.03, 0.02, 0.00],
            [0.05, 0.03, 0.01],
            [0.07, 0.04, 0.02],
        ]
    )
    colors = np.column_stack(
        (
            np.linspace(0.05, 0.95, len(points)),
            np.linspace(0.90, 0.10, len(points)),
            np.linspace(0.20, 0.70, len(points)),
        )
    )
    return points, colors


def _config(count: int = 8) -> ReusableGraphRegistrationConfig:
    return ReusableGraphRegistrationConfig(
        canonical_node_count=count,
        geometry_sigma_m=0.01,
        color_sigma=0.10,
        color_cost_weight=1.0,
        assignment_temperature=0.25,
        maximum_match_distance_m=0.01,
        minimum_match_fraction=0.95,
        minimum_effective_reliable_fraction=0.70,
        icp_iterations=5,
        trim_fraction=1.0,
    )


def test_farthest_point_selection_is_deterministic_and_unique() -> None:
    points, _ = _reference()
    first = deterministic_farthest_point_indices(points, 7)
    second = deterministic_farthest_point_indices(points.copy(), 7)
    np.testing.assert_array_equal(first, second)
    assert len(np.unique(first)) == 7


def test_rigid_appearance_registration_recovers_material_indices() -> None:
    points, colors = _reference()
    canonical = build_canonical_deform360_graph(
        points,
        colors,
        registration_config=_config(),
        spring_config=SPRINGS,
    )
    adjacency = coo_matrix(
        (
            np.ones(2 * len(canonical.springs)),
            (
                np.concatenate((canonical.springs[:, 0], canonical.springs[:, 1])),
                np.concatenate((canonical.springs[:, 1], canonical.springs[:, 0])),
            ),
        ),
        shape=(len(canonical.vertices), len(canonical.vertices)),
    )
    assert connected_components(adjacency, directed=False)[0] == 1
    angle = np.deg2rad(23.0)
    rotation = np.asarray(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    permutation = np.asarray([7, 1, 9, 3, 0, 8, 2, 6, 4, 5])
    target = points[permutation] @ rotation.T + np.asarray([0.03, -0.02, 0.01])
    target_colors = colors[permutation]
    result = register_canonical_graph_to_episode(
        canonical,
        target,
        target_colors,
        config=_config(),
    )
    inverse = np.empty(len(permutation), dtype=np.int64)
    inverse[permutation] = np.arange(len(permutation))
    np.testing.assert_array_equal(
        result.target_indices, inverse[canonical.source_indices]
    )
    assert result.passed
    assert np.max(result.geometric_error_m) < 1e-8
    assert np.all(np.linalg.eigvalsh(result.observation_covariance_m2) >= -1e-12)


def test_reference_registration_uses_exact_source_material_indices() -> None:
    points, colors = _reference()
    canonical = build_canonical_deform360_graph(
        points,
        colors,
        registration_config=_config(),
        spring_config=SPRINGS,
    )
    candidate_reliability = np.linspace(0.8, 1.0, len(points))
    result = canonical_reference_registration(
        canonical,
        config=_config(),
        candidate_reliability=candidate_reliability,
    )
    np.testing.assert_array_equal(result.target_indices, canonical.source_indices)
    np.testing.assert_array_equal(
        result.prior_reliability,
        candidate_reliability[canonical.source_indices],
    )
    assert result.passed


def test_assignment_ambiguity_remains_in_covariance_and_reliability() -> None:
    points, colors = _reference()
    canonical = build_canonical_deform360_graph(
        points,
        colors,
        registration_config=_config(count=6),
        spring_config=SPRINGS,
    )
    target = np.repeat(canonical.vertices, 2, axis=0)
    target_colors = np.repeat(canonical.colors, 2, axis=0)
    result = register_canonical_graph_to_episode(
        canonical,
        target,
        target_colors,
        config=_config(count=6),
    )
    assert np.all(result.assignment_probability <= 0.5 + 1e-8)
    assert np.all(result.assignment_entropy > 0.0)
    assert np.all(result.prior_reliability < 1.0)
    assert np.all(np.trace(result.observation_covariance_m2, axis1=1, axis2=2) >= 12e-6)


def test_registered_graph_preserves_object_topology_and_rest_lengths() -> None:
    points, colors = _reference()
    canonical = build_canonical_deform360_graph(
        points,
        colors,
        registration_config=_config(),
        spring_config=SPRINGS,
    )
    shifted = canonical.vertices + np.asarray([0.01, -0.005, 0.002])
    controller = shifted[:1] + np.asarray([[0.0, 0.0, 0.005]])
    graph = build_registered_phystwin_graph(
        canonical,
        shifted,
        controller,
        spring_config=SPRINGS,
    )
    assert graph.num_object_springs == len(canonical.springs)
    np.testing.assert_array_equal(
        graph.springs[: graph.num_object_springs],
        canonical.springs,
    )
    np.testing.assert_array_equal(
        graph.rest_lengths[: graph.num_object_springs],
        canonical.rest_lengths,
    )
    assert len(graph.springs) > graph.num_object_springs


def test_occlusion_gaps_receive_latent_bending_chain_nodes() -> None:
    first = np.column_stack((np.linspace(0.0, 0.03, 5), np.zeros(5), np.zeros(5)))
    second = np.column_stack((np.linspace(0.10, 0.13, 5), np.zeros(5), np.zeros(5)))
    points = np.concatenate((first, second), axis=0)
    colors = np.tile(np.asarray([[0.2, 0.5, 0.7]]), (len(points), 1))
    canonical = build_canonical_deform360_graph(
        points,
        colors,
        registration_config=ReusableGraphRegistrationConfig(
            canonical_node_count=10,
        ),
        spring_config=PhysTwinSpringGraphConfig(
            object_radius=0.015,
            object_max_neighbours=4,
            controller_radius=0.02,
            controller_max_neighbours=1,
        ),
    )
    assert canonical.observed_node_count == 10
    assert canonical.latent_node_count >= 5
    assert canonical.bridge_spring_count == canonical.latent_node_count + 1
    assert np.all(canonical.source_indices[-canonical.latent_node_count :] < 0)
    assert np.max(canonical.rest_lengths[-canonical.bridge_spring_count :]) <= 0.0075


def test_grasp_occlusion_receives_a_registered_contact_chain() -> None:
    points, colors = _reference()
    controller = np.tile(np.asarray([[-0.05, 0.0, 0.0]]), (4, 1))
    canonical = build_canonical_deform360_graph(
        points,
        colors,
        registration_config=_config(),
        spring_config=SPRINGS,
        reference_controller_points=controller,
        controller_group_size=4,
    )
    assert len(canonical.contact_anchor_indices) == 1
    assert canonical.contact_chain_spring_count > 1
    anchor = canonical.contact_anchor_indices[0]
    np.testing.assert_allclose(
        np.linalg.norm(canonical.vertices[anchor] - controller[0]),
        0.002,
        atol=1e-8,
    )
    assert canonical.source_indices[anchor] < 0

    graph = build_registered_phystwin_graph(
        canonical,
        canonical.vertices,
        controller,
        spring_config=SPRINGS,
    )
    controller_springs = graph.springs[graph.num_object_springs :]
    assert controller_springs.shape == (1, 2)
    assert controller_springs[0, 0] == len(canonical.vertices)
    assert controller_springs[0, 1] == anchor
    np.testing.assert_allclose(
        graph.rest_lengths[graph.num_object_springs :],
        0.002,
        atol=1e-8,
    )


def test_existing_contact_anchor_needs_no_latent_chain() -> None:
    points, colors = _reference()
    controller = np.tile(points[:1], (4, 1))
    canonical = build_canonical_deform360_graph(
        points,
        colors,
        registration_config=_config(),
        spring_config=SPRINGS,
        reference_controller_points=controller,
        controller_group_size=4,
    )

    assert canonical.contact_chain_spring_count == 0
    assert canonical.contact_anchor_indices.shape == (1,)
    np.testing.assert_allclose(
        canonical.vertices[canonical.contact_anchor_indices[0]],
        controller[0],
        atol=1e-8,
    )


def test_registered_graph_supports_opt_in_distributed_contact_patch() -> None:
    points, colors = _reference()
    controller = np.tile(points[:1], (4, 1))
    canonical = build_canonical_deform360_graph(
        points,
        colors,
        registration_config=_config(),
        spring_config=SPRINGS,
        reference_controller_points=controller,
        controller_group_size=4,
    )

    graph = build_registered_phystwin_graph(
        canonical,
        canonical.vertices,
        controller,
        spring_config=SPRINGS,
        controller_patch_size=3,
    )

    controller_springs = graph.springs[graph.num_object_springs :]
    assert controller_springs.shape == (3, 2)
    assert canonical.contact_anchor_indices[0] in controller_springs[:, 1]
    assert len(np.unique(controller_springs[:, 1])) == 3
    assert np.all(
        graph.rest_lengths[graph.num_object_springs :] <= SPRINGS.controller_radius
    )


def test_registered_graph_builds_dynamic_episode_contact_patches() -> None:
    points, colors = _reference()
    canonical = build_canonical_deform360_graph(
        points,
        colors,
        registration_config=_config(),
        spring_config=SPRINGS,
    )
    assert len(canonical.contact_anchor_indices) == 0
    first = np.tile(canonical.vertices[:1], (4, 1))
    second = np.tile(canonical.vertices[-1:], (4, 1))
    controller = np.concatenate((first, second), axis=0)

    graph = build_registered_phystwin_graph(
        canonical,
        canonical.vertices,
        controller,
        spring_config=SPRINGS,
        controller_patch_size=3,
        controller_group_size=4,
    )

    controller_springs = graph.springs[graph.num_object_springs :]
    assert controller_springs.shape == (6, 2)
    assert len(np.unique(controller_springs[:3, 1])) == 3
    assert len(np.unique(controller_springs[3:, 1])) == 3
    assert np.all(controller_springs[:3, 0] < len(canonical.vertices) + 4)
    assert np.all(controller_springs[3:, 0] >= len(canonical.vertices) + 4)
    np.testing.assert_array_equal(
        graph.springs[: graph.num_object_springs], canonical.springs
    )
    np.testing.assert_array_equal(
        graph.rest_lengths[: graph.num_object_springs], canonical.rest_lengths
    )


def test_registration_reorders_tracks_and_enforces_information_boundary() -> None:
    points, colors = _reference()
    canonical = build_canonical_deform360_graph(
        points,
        colors,
        registration_config=_config(),
        spring_config=SPRINGS,
    )
    result = register_canonical_graph_to_episode(
        canonical,
        points,
        colors,
        config=_config(),
    )
    frames = 3
    data = {
        "object_points": np.repeat(points[None], frames, axis=0),
        "object_colors": np.repeat(colors[None], frames, axis=0),
        "object_visibilities": np.ones((frames, len(points)), dtype=bool),
        "object_motions_valid": np.ones((frames, len(points)), dtype=bool),
        "controller_points": np.zeros((frames, 1, 3)),
        "surface_points": np.empty((0, 3)),
        "interior_points": np.empty((0, 3)),
    }
    registered = registered_episode_data(
        data,
        result,
        canonical_graph_sha256=canonical.sha256,
    )
    assert registered["object_points"].shape == (frames, 8, 3)
    np.testing.assert_array_equal(
        registered["object_points"][0],
        points[result.target_indices],
    )
    assert registered["surface_points"].shape == (0, 3)
    assert (
        registered["reusable_graph_registration"]["canonical_graph_sha256"]
        == canonical.sha256
    )
    summary = episode_registration_summary(
        result,
        canonical_graph_sha256=canonical.sha256,
        information_boundary={
            "observed_prefix_frame_count": 1,
            "simulator_residual_used": False,
            "future_object_frames_used": False,
        },
    )
    assert summary["passed"] is True
    assert len(summary["result_sha256"]) == 64
