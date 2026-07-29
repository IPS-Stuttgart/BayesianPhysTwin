from __future__ import annotations

import numpy as np

from bayesian_phystwin.phystwin_cross_provider_guarded_field import (
    CrossProviderGuardedFieldConfig,
    build_guarded_dense_field,
    estimate_relative_provider_bias,
    fit_correlated_graph_field,
    residual_independent_dense_reliability,
)


def _orthonormal_basis(node_count: int, rank: int) -> np.ndarray:
    coordinate = np.linspace(-1.0, 1.0, node_count)
    design = np.column_stack([coordinate**power for power in range(rank)])
    basis, _ = np.linalg.qr(design)
    return basis


def test_dense_reliability_does_not_accept_state_innovation() -> None:
    config = CrossProviderGuardedFieldConfig()
    quality = np.array([0.8, 0.8])
    reprojection = np.array([1.0, 4.0])
    cameras = np.array([2, 3])
    valid = np.array([True, True])

    first = residual_independent_dense_reliability(
        quality,
        reprojection,
        cameras,
        valid,
        config=config,
    )
    second = residual_independent_dense_reliability(
        quality.copy(),
        reprojection.copy(),
        cameras.copy(),
        valid.copy(),
        config=config,
    )

    np.testing.assert_array_equal(first, second)
    assert first[0] > 0.0
    assert first[1] == 0.0


def test_duplicate_correlated_block_does_not_increase_information_mass() -> None:
    basis = _orthonormal_basis(6, 2)
    innovation = basis @ np.array([[0.004, 0.0, 0.0], [0.002, 0.0, 0.0]])
    available = np.ones(6, dtype=bool)
    reliability = np.ones(6)
    field, coefficients, diagnostics = fit_correlated_graph_field(
        basis,
        innovation,
        available,
        reliability,
        robust_scale_m=0.010,
        robust_iterations=2,
        projection_ridge=1e-8,
        maximum_correction_m=0.100,
    )
    duplicated_field, duplicated_coefficients, duplicated = (
        fit_correlated_graph_field(
            np.repeat(basis, 3, axis=0),
            np.repeat(innovation, 3, axis=0),
            np.ones(18, dtype=bool),
            np.ones(18),
            robust_scale_m=0.010,
            robust_iterations=2,
            projection_ridge=1e-8,
            maximum_correction_m=0.100,
        )
    )

    np.testing.assert_allclose(duplicated_coefficients, coefficients, atol=1e-12)
    np.testing.assert_allclose(
        duplicated_field.reshape(6, 3, 3)[:, 0],
        field,
        atol=1e-12,
    )
    assert diagnostics["information_mass"] == duplicated["information_mass"] == 2.0


def test_relative_provider_bias_ignores_absolute_gauge() -> None:
    frame_count = 6
    provider = np.zeros((frame_count, 2, 3))
    provider[:, :, 0] = np.arange(frame_count)[:, None] * 0.001
    dense = provider + np.array([0.2, -0.1, 0.05])
    dense[:, :, 1] += np.arange(frame_count)[:, None] * 0.002
    support = np.ones((frame_count, 2), dtype=bool)
    code = np.ones((frame_count, 2), dtype=np.int8)

    bias, diagnostics = estimate_relative_provider_bias(
        provider,
        support,
        code,
        dense,
        support,
        history_frames=4,
    )

    np.testing.assert_allclose(bias, [0.0, 0.007, 0.0], atol=1e-12)
    assert diagnostics["absolute_gauge_used"] is False


def _guard_fixture(
    *,
    validation_matches_field: bool,
) -> tuple[dict[str, np.ndarray], CrossProviderGuardedFieldConfig]:
    node_count = 10
    rank = 3
    total_frames = 10
    source_frames = 4
    basis = _orthonormal_basis(node_count, rank)
    coefficients = np.array(
        [
            [0.010, 0.0, 0.0],
            [0.012, 0.0, 0.0],
            [-0.008, 0.0, 0.0],
        ]
    )
    field = basis @ coefficients
    baseline = np.zeros((total_frames, node_count, 3), dtype=np.float32)
    dense = np.zeros((source_frames, node_count, 3))
    dense[-1] = field
    dense_valid = np.ones((source_frames, node_count), dtype=bool)
    camera_count = np.full((source_frames, node_count), 3)
    reprojection = np.ones((source_frames, node_count))
    quality = np.full((source_frames, node_count), 0.9)

    provider_nodes = np.array([0, 1])
    provider_ids = np.array([10, 11])
    provider = np.zeros((source_frames, 2, 3))
    provider[-1] = field[provider_nodes]
    provider_support = np.ones((source_frames, 2), dtype=bool)
    provider_code = np.ones((source_frames, 2), dtype=np.int8)
    local_dense = provider.copy()

    validation_nodes = np.array([7, 9])
    validation_ids = np.array([20, 21])
    validation = np.repeat(
        (
            field[validation_nodes]
            if validation_matches_field
            else np.zeros((2, 3))
        )[None],
        4,
        axis=0,
    )
    config = CrossProviderGuardedFieldConfig(
        source_frame_start=0,
        source_frame_end_exclusive=source_frames,
        apply_frame_start=source_frames,
        validation_frame_end_exclusive=8,
        minimum_provider_count=2,
        bias_history_frames=3,
        projection_ridge=1e-8,
        maximum_correction_m=0.100,
        minimum_validation_improvement_fraction=0.01,
        minimum_validation_improvement_m=1e-6,
    )
    return (
        {
            "baseline_trajectory_m": baseline,
            "graph_basis": basis,
            "dense_points_world_m": dense,
            "dense_point_valid": dense_valid,
            "dense_camera_count": camera_count,
            "dense_reprojection_error_px": reprojection,
            "dense_quality_probability": quality,
            "provider_trajectory_m": provider,
            "provider_support": provider_support,
            "provider_code": provider_code,
            "provider_identity_ids": provider_ids,
            "provider_node_ids": provider_nodes,
            "dense_local_trajectory_m": local_dense,
            "dense_local_available": provider_support,
            "validation_tracks_world_m": validation,
            "validation_identity_ids": validation_ids,
            "validation_node_ids": validation_nodes,
        },
        config,
    )


def test_disjoint_prefix_gate_accepts_transferable_dense_field() -> None:
    inputs, config = _guard_fixture(validation_matches_field=True)
    result = build_guarded_dense_field(**inputs, config=config)

    assert result["accepted"]
    assert result["reason"] == "prefix-disjoint-validation-passed"
    assert result["diagnostics"]["validation"]["passed"]
    assert result["diagnostics"]["information_boundary"][
        "state_innovation_used_in_prior_reliability"
    ] is False


def test_rejected_field_is_bit_exact_baseline_fallback() -> None:
    inputs, config = _guard_fixture(validation_matches_field=False)
    result = build_guarded_dense_field(**inputs, config=config)

    assert not result["accepted"]
    np.testing.assert_array_equal(
        result["candidate_trajectory_m"],
        inputs["baseline_trajectory_m"],
    )
    assert result["diagnostics"]["information_boundary"][
        "rejection_is_bit_exact_baseline"
    ]


def test_missing_cross_provider_history_falls_back_exactly() -> None:
    inputs, config = _guard_fixture(validation_matches_field=True)
    inputs["provider_code"] = np.zeros_like(inputs["provider_code"])
    result = build_guarded_dense_field(**inputs, config=config)

    assert not result["accepted"]
    assert result["reason"] == "insufficient-provider-support"
    assert not result["diagnostics"]["relative_provider_bias"][
        "estimate_available"
    ]
    np.testing.assert_array_equal(
        result["candidate_trajectory_m"],
        inputs["baseline_trajectory_m"],
    )


def test_provider_and_validation_identity_overlap_is_rejected() -> None:
    inputs, config = _guard_fixture(validation_matches_field=True)
    inputs["validation_identity_ids"] = np.array([10, 21])

    with np.testing.assert_raises_regex(
        ValueError,
        "provider and validation identities overlap",
    ):
        build_guarded_dense_field(**inputs, config=config)
