import numpy as np

from bayesian_phystwin.pokeflex_bayesian_registration import (
    PokeFlexActionGuardConfig,
    PokeFlexBayesianRegistrationConfig,
    crop_points_to_template,
    depth_image_to_world_points,
    pokeflex_action_contact_fields,
    pokeflex_correction_field_variants,
    register_pokeflex_graph_posterior,
    voxel_cluster_centroids,
)


def _grid() -> np.ndarray:
    x, y, z = np.meshgrid(
        np.linspace(-0.03, 0.03, 5),
        np.linspace(-0.03, 0.03, 5),
        np.linspace(-0.03, 0.03, 5),
        indexing="ij",
    )
    return np.column_stack((x.ravel(), y.ravel(), z.ravel()))


def _config(**changes: object) -> PokeFlexBayesianRegistrationConfig:
    values = {
        "control_node_count": 24,
        "graph_neighbors": 4,
        "interpolation_neighbors": 3,
        "assignment_candidates": 3,
        "minimum_points_per_view": 8,
        "effective_samples_per_view": 32.0,
    }
    values.update(changes)
    return PokeFlexBayesianRegistrationConfig(**values)


def test_depth_projection_matches_world_to_camera_convention() -> None:
    depth = np.array([[1000, 0], [0, 0]], dtype=np.uint16)
    intrinsics = np.array([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 1.0]])
    world_to_camera = np.eye(4)
    world_to_camera[0, 3] = 0.25

    points = depth_image_to_world_points(depth, intrinsics, world_to_camera)

    np.testing.assert_allclose(points, [[-0.25, 0.0, 1.0]])


def test_template_crop_uses_metric_bounds() -> None:
    template = np.array([[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]])
    points = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, -0.95, 0.0]])

    cropped = crop_points_to_template(
        points, template, scale=1.0, minimum_vertical_offset_m=0.1
    )

    np.testing.assert_array_equal(cropped, [[0.0, 0.0, 0.0]])


def test_voxel_clustering_makes_exact_duplicates_idempotent() -> None:
    points = np.array([[0.0, 0.0, 0.0], [0.001, 0.0, 0.0], [0.02, 0.0, 0.0]])
    duplicated = np.repeat(points, 20, axis=0)

    first = voxel_cluster_centroids(points, 0.005)
    second = voxel_cluster_centroids(duplicated, 0.005)

    np.testing.assert_allclose(first, second)


def test_two_view_common_mode_shift_falls_back_without_action_support() -> None:
    prior = _grid()
    biased = prior + np.array([0.01, 0.0, 0.0])

    result = register_pokeflex_graph_posterior(
        prior,
        (biased, biased.copy()),
        action_supported=False,
        config=_config(),
    )

    assert result.accepted is False
    assert result.reason == "insufficient-independent-support"
    assert result.posterior_vertices_m is prior


def test_action_support_allows_two_view_update_without_changing_reliability() -> None:
    prior = _grid()
    shifted = prior + np.array([0.006, 0.0, 0.0])
    config = _config(association_radius_m=0.02)

    small = register_pokeflex_graph_posterior(
        prior, (shifted, shifted.copy()), action_supported=True, config=config
    )
    larger = register_pokeflex_graph_posterior(
        prior,
        (shifted + [0.006, 0.0, 0.0], shifted + [0.006, 0.0, 0.0]),
        action_supported=True,
        config=config,
    )

    assert small.accepted and larger.accepted
    assert small.diagnostics["source_reliabilities"] == [1.0, 1.0]
    assert larger.diagnostics["source_reliabilities"] == [1.0, 1.0]


def test_duplicated_correlated_camera_does_not_increase_information_mass() -> None:
    prior = _grid()
    view = prior + np.array([0.004, 0.0, 0.0])
    config = _config()

    original = register_pokeflex_graph_posterior(
        prior, (view, view.copy()), action_supported=True, config=config
    )
    duplicated = register_pokeflex_graph_posterior(
        prior,
        (np.repeat(view, 10, axis=0), np.repeat(view, 10, axis=0)),
        action_supported=True,
        config=config,
    )

    np.testing.assert_allclose(
        original.diagnostics["effective_information_mass"],
        duplicated.diagnostics["effective_information_mass"],
    )


def test_robust_innovation_downweights_gross_associated_outlier() -> None:
    prior = _grid()
    clean = prior + np.array([0.003, 0.0, 0.0])
    contaminated = clean.copy()
    contaminated[:20] += np.array([0.012, 0.011, 0.013])

    result = register_pokeflex_graph_posterior(
        prior,
        (contaminated, contaminated.copy()),
        action_supported=True,
        config=_config(association_radius_m=0.03, huber_scale_m=0.001),
    )

    assert result.accepted
    assert result.diagnostics["minimum_robust_weight"] < 1.0
    assert result.diagnostics["downweighted_fraction"] > 0.0
    assert result.diagnostics["innovation_uses_prior_reliability"] is False


def test_ambiguous_assignments_use_mixture_mean_and_graph_weights() -> None:
    prior = _grid()
    interior = prior[prior[:, 0] < prior[:, 0].max()]
    halfway = interior + np.array([0.0075, 0.0, 0.0])

    result = register_pokeflex_graph_posterior(
        prior,
        (halfway, halfway.copy()),
        action_supported=True,
        config=_config(
            assignment_candidates=2,
            association_radius_m=0.02,
            camera_bias_variance_m2=1e-12,
        ),
    )

    assert result.accepted
    assert result.diagnostics["assignment_variance_m2_mean"] > 0.0
    assert result.diagnostics["rms_update_m"] < 0.0005


def test_point_to_plane_update_rejects_tangential_surface_drift() -> None:
    x, y = np.meshgrid(
        np.linspace(-0.03, 0.03, 5),
        np.linspace(-0.03, 0.03, 5),
        indexing="ij",
    )
    prior = np.column_stack((x.ravel(), y.ravel(), np.zeros(x.size)))
    faces = []
    for row in range(4):
        for column in range(4):
            lower = row * 5 + column
            faces.append((lower, lower + 1, lower + 5))
            faces.append((lower + 1, lower + 6, lower + 5))
    tangential = prior + np.array([0.003, 0.0, 0.0])

    result = register_pokeflex_graph_posterior(
        prior,
        (tangential, tangential.copy()),
        action_supported=True,
        prior_faces=np.asarray(faces, dtype=np.int64),
        config=_config(
            assignment_candidates=1,
            residual_geometry="point_to_plane",
        ),
    )

    assert result.accepted
    assert result.diagnostics["residual_geometry"] == "point_to_plane"
    assert result.diagnostics["rms_update_m"] < 1e-8


def test_temporal_correction_fields_use_only_past_innovations() -> None:
    source = _grid()
    target = source + np.array([0.001, 0.0, 0.0])
    current = np.broadcast_to(np.array([0.004, 0.002, 0.0]), source.shape).copy()
    previous = np.broadcast_to(np.array([0.002, 0.002, 0.0]), source.shape).copy()

    fields = pokeflex_correction_field_variants(
        source,
        target,
        current,
        previous_correction_m=previous,
    )

    np.testing.assert_allclose(fields["temporal_linear"], 2.0 * current - previous)
    np.testing.assert_allclose(fields["temporal_mean"], 0.5 * (current + previous))
    assert np.all(np.linalg.norm(fields["temporal_shared"], axis=1) > 0.0)


def test_first_temporal_correction_is_exact_raw_fallback() -> None:
    source = _grid()
    correction = np.full_like(source, 0.003)

    fields = pokeflex_correction_field_variants(
        source,
        source.copy(),
        correction,
    )

    np.testing.assert_array_equal(fields["temporal_linear"], correction)
    np.testing.assert_array_equal(fields["temporal_mean"], correction)
    np.testing.assert_allclose(fields["temporal_shared"], correction)


def test_action_contact_field_uses_only_measured_history() -> None:
    source = _grid()
    target = source.copy()
    correction = np.zeros_like(source)
    tool = np.array([[0.0, -0.05, 0.0], [0.0, -0.048, 0.0]])
    end_effector = tool + np.array([0.0, -0.1, 0.0])

    fields = pokeflex_action_contact_fields(
        source,
        target,
        correction,
        tool,
        end_effector,
        influence_radius_m=0.02,
    )

    peak = int(np.argmax(np.linalg.norm(fields["action_velocity"], axis=1)))
    np.testing.assert_allclose(
        fields["action_velocity"][peak],
        tool[-1] - tool[-2],
        atol=1e-12,
    )
    assert np.linalg.norm(fields["action_velocity"], axis=1).min() < 0.5 * np.linalg.norm(
        fields["action_velocity"], axis=1
    ).max()


def test_action_guard_has_exact_unsupported_fallback() -> None:
    guard = PokeFlexActionGuardConfig()

    assert guard.selected_scale(
        30.0, observation_update_accepted=False, action_supported=True
    ) == 0.0
    assert guard.selected_scale(
        30.0, observation_update_accepted=True, action_supported=False
    ) == 0.0
    assert guard.selected_scale(
        10.0, observation_update_accepted=True, action_supported=True
    ) == 0.125
    assert guard.selected_scale(
        20.0, observation_update_accepted=True, action_supported=True
    ) == 0.5


def test_relative_contact_radius_controls_spatial_support() -> None:
    source = _grid()
    tool = np.array([[0.0, -0.05, 0.0], [0.0, -0.048, 0.0]])
    end_effector = tool + np.array([0.0, -0.1, 0.0])

    narrow = pokeflex_action_contact_fields(
        source,
        source.copy(),
        np.zeros_like(source),
        tool,
        end_effector,
        influence_radius_m=0.01,
    )["action_local_state"]
    broad = pokeflex_action_contact_fields(
        source,
        source.copy(),
        np.zeros_like(source),
        tool,
        end_effector,
        influence_radius_m=0.05,
    )["action_local_state"]

    assert np.sum(np.linalg.norm(narrow, axis=1)) < np.sum(
        np.linalg.norm(broad, axis=1)
    )
