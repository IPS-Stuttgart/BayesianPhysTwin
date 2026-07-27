from __future__ import annotations

from dataclasses import replace

import numpy as np

from bayesian_phystwin.phystwin_graph_spectral_residual import (
    GraphSpectralResidualConfig,
    GraphSpectralSeries,
    blend_with_endpoint_persistence,
    build_knn_laplacian_basis,
    compose_dense_endpoint_with_anchor_dynamics,
    controller_action_field,
    default_mode_groups,
    deterministic_farthest_point_sample,
    endpoint_persistence,
    fit_graph_spectral_transition,
    inverse_distance_map,
    rollout_graph_spectral_transition,
)


def _ring(count: int = 24) -> np.ndarray:
    angle = np.linspace(0.0, 2.0 * np.pi, count, endpoint=False)
    return np.column_stack((np.cos(angle), np.sin(angle), 0.1 * np.sin(2 * angle)))


def _series(
    *,
    rotation: np.ndarray | None = None,
    sign: np.ndarray | None = None,
) -> tuple[GraphSpectralSeries, GraphSpectralResidualConfig]:
    config = GraphSpectralResidualConfig(
        rank=6,
        neighbor_count=4,
        mode_group_count=3,
        ridge_fraction=1.0e-8,
        minimum_group_samples=3,
        maximum_residual_m=10.0,
    )
    points = _ring()
    basis, eigenvalues = build_knn_laplacian_basis(
        points,
        rank=config.rank,
        neighbor_count=config.neighbor_count,
    )
    if sign is not None:
        basis = basis * sign[None]
    groups = default_mode_groups(config.rank, config.mode_group_count)
    rng = np.random.default_rng(14)
    action = np.zeros((18, config.rank, 3), dtype=float)
    action[2:] = 0.01 * rng.normal(size=(16, config.rank, 3))
    coefficients = np.zeros_like(action)
    retention = np.array([0.2, 0.5, 0.8])
    current = np.array([0.4, -0.2, 0.1])
    change = np.array([0.05, 0.1, -0.1])
    velocity = np.zeros((config.rank, 3), dtype=float)
    for frame in range(1, len(coefficients)):
        velocity = (
            retention[groups, None] * velocity
            + current[groups, None] * action[frame]
            + change[groups, None] * (action[frame] - action[frame - 1])
        )
        coefficients[frame] = coefficients[frame - 1] + velocity
    if rotation is not None:
        coefficients = coefficients @ rotation.T
        action = action @ rotation.T
    return (
        GraphSpectralSeries(
            basis=basis,
            eigenvalues=eigenvalues,
            mode_groups=groups,
            residual_coefficients=coefficients,
            action_coefficients=action,
            object_scale_m=1.0,
        ),
        config,
    )


def test_laplacian_basis_is_orthonormal_and_deterministic() -> None:
    points = _ring()
    first_basis, first_values = build_knn_laplacian_basis(
        points,
        rank=6,
        neighbor_count=4,
    )
    second_basis, second_values = build_knn_laplacian_basis(
        points,
        rank=6,
        neighbor_count=4,
    )
    assert np.allclose(first_basis.T @ first_basis, np.eye(6), atol=1.0e-8)
    assert np.allclose(first_values, second_values)
    assert np.allclose(first_basis, second_basis)


def test_transition_recovers_synthetic_group_dynamics() -> None:
    series, config = _series()
    fitted = fit_graph_spectral_transition([series], config=config)
    assert np.allclose(fitted.velocity_retention, [0.2, 0.5, 0.8], atol=2.0e-3)
    assert np.allclose(fitted.action_current, [0.4, -0.2, 0.1], atol=2.0e-3)


def test_prefix_adaptation_shrinks_toward_source_transition() -> None:
    series, config = _series()
    source = fit_graph_spectral_transition([series], config=config)
    short = replace(
        series,
        residual_coefficients=series.residual_coefficients[:5],
        action_coefficients=series.action_coefficients[:5],
    )
    adapted = fit_graph_spectral_transition(
        [short],
        config=config,
        prior=source,
        prior_strength=100.0,
    )
    assert np.linalg.norm(
        adapted.velocity_retention - source.velocity_retention
    ) < 0.05


def test_rollout_is_rotation_equivariant() -> None:
    series, config = _series()
    angle = 0.4
    rotation = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    rotated, _ = _series(rotation=rotation)
    transition = fit_graph_spectral_transition([series], config=config)
    first = rollout_graph_spectral_transition(
        series,
        transition,
        start_frame=12,
        end_frame=18,
        config=config,
    )
    second = rollout_graph_spectral_transition(
        rotated,
        transition,
        start_frame=12,
        end_frame=18,
        config=config,
    )
    assert np.allclose(second, first @ rotation.T, atol=1.0e-9)


def test_rollout_is_invariant_to_graph_eigenvector_sign() -> None:
    series, config = _series()
    sign = np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0])
    signed, _ = _series(sign=sign)
    signed = replace(
        signed,
        residual_coefficients=signed.residual_coefficients * sign[None, :, None],
        action_coefficients=signed.action_coefficients * sign[None, :, None],
    )
    transition = fit_graph_spectral_transition([series], config=config)
    first = rollout_graph_spectral_transition(
        series,
        transition,
        start_frame=12,
        end_frame=18,
        config=config,
    )
    second = rollout_graph_spectral_transition(
        signed,
        transition,
        start_frame=12,
        end_frame=18,
        config=config,
    )
    assert np.allclose(first, second, atol=1.0e-9)


def test_zero_blend_is_exact_persistence() -> None:
    series, config = _series()
    persistence = endpoint_persistence(
        series,
        start_frame=12,
        end_frame=18,
        config=config,
    )
    dynamic = persistence + 1.0
    result = blend_with_endpoint_persistence(dynamic, persistence, 0.0)
    assert np.array_equal(result, persistence)


def test_rollout_uses_known_actions_without_future_residuals() -> None:
    series, config = _series()
    truncated = replace(
        series,
        residual_coefficients=series.residual_coefficients[:12],
    )
    transition = fit_graph_spectral_transition([truncated], config=config)
    dynamic = rollout_graph_spectral_transition(
        truncated,
        transition,
        start_frame=12,
        end_frame=18,
        config=config,
    )
    persistence = endpoint_persistence(
        truncated,
        start_frame=12,
        end_frame=18,
        config=config,
    )
    assert dynamic.shape == persistence.shape == (6, len(_ring()), 3)


def test_action_field_rotates_with_geometry_and_controls() -> None:
    points = _ring(12)
    baseline = np.repeat(points[None], 5, axis=0)
    controllers = np.zeros((5, 1, 3), dtype=float)
    controllers[:, 0, 0] = np.arange(5) * 0.1
    first = controller_action_field(
        baseline,
        controllers,
        object_scale_m=1.0,
        kernel_fraction=0.5,
    )
    rotation = np.array(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    )
    second = controller_action_field(
        baseline @ rotation.T,
        controllers @ rotation.T,
        object_scale_m=1.0,
        kernel_fraction=0.5,
    )
    assert np.allclose(second, first @ rotation.T)


def test_anchor_lift_preserves_dense_endpoint_and_zero_change() -> None:
    points = _ring(12)
    candidates = np.arange(len(points), dtype=np.int64)
    anchors = deterministic_farthest_point_sample(points, candidates, 5)
    interpolation_indices, interpolation_weights = inverse_distance_map(
        points[anchors],
        points,
        neighbor_count=3,
    )
    endpoint = np.linspace(0.0, 0.005, len(points))[:, None] * np.ones((1, 3))
    anchor_dynamic = np.broadcast_to(
        endpoint[anchors],
        (4, len(anchors), 3),
    )
    result = compose_dense_endpoint_with_anchor_dynamics(
        anchor_dynamic,
        endpoint,
        anchors,
        interpolation_indices,
        interpolation_weights,
        maximum_residual_m=0.01,
    )
    assert np.allclose(result, endpoint[None])
    assert len(np.unique(anchors)) == len(anchors)
