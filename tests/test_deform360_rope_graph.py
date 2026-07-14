from __future__ import annotations

import numpy as np

from causal4d_public.deform360_rope_graph import (
    RopeCenterlineConfig,
    extract_rope_centerline,
    initialize_rope_centerline_pca,
    rope_chain_edges,
)


def _tube_around_curve(curve: np.ndarray, *, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    repeated = np.repeat(curve, 12, axis=0)
    return repeated + rng.normal(scale=0.0025, size=repeated.shape)


def test_centerline_recovers_a_curved_rope_and_equal_spacing() -> None:
    parameter = np.linspace(-1.0, 1.0, 120)
    curve = np.column_stack(
        (
            0.22 * parameter,
            0.08 * parameter**2,
            0.04 * np.sin(np.pi * parameter),
        )
    )
    points = _tube_around_curve(curve, seed=5)
    config = RopeCenterlineConfig(node_count=21, refinement_iterations=10)

    centerline, diagnostics = extract_rope_centerline(points, config=config)

    edge_lengths = np.linalg.norm(np.diff(centerline, axis=0), axis=1)
    assert centerline.shape == (21, 3)
    assert np.std(edge_lengths) / np.mean(edge_lengths) < 0.03
    assert diagnostics["point_to_centerline_node_distance_m"]["p95"] < 0.02
    assert 0.45 < diagnostics["centerline_length_m"] < 0.60


def test_reference_centerline_fixes_material_orientation() -> None:
    parameter = np.linspace(0.0, 1.0, 80)
    reference = np.column_stack((0.4 * parameter, 0.03 * parameter**2, 0.0 * parameter))
    points = _tube_around_curve(reference[::-1], seed=12)
    config = RopeCenterlineConfig(
        node_count=len(reference),
        density_keep_quantile=1.0,
        refinement_iterations=4,
    )

    centerline, diagnostics = extract_rope_centerline(
        points,
        config=config,
        reference_centerline_m=reference,
    )

    assert np.linalg.norm(centerline[0] - reference[0]) < np.linalg.norm(
        centerline[-1] - reference[0]
    )
    assert diagnostics["orientation"].startswith("reference-")


def test_provided_initial_centerline_avoids_a_volumetric_graph_detour() -> None:
    parameter = np.linspace(0.0, 1.0, 80)
    reference = np.column_stack((0.4 * parameter, 0.03 * parameter**2, 0.0 * parameter))
    points = _tube_around_curve(reference, seed=22)
    config = RopeCenterlineConfig(
        node_count=len(reference),
        density_keep_quantile=1.0,
        refinement_iterations=4,
    )

    centerline, diagnostics = extract_rope_centerline(
        points,
        config=config,
        initial_centerline_m=reference,
        reference_centerline_m=reference,
    )

    assert diagnostics["initialization"] == "provided-centerline"
    assert diagnostics["graph_diameter_length_m"] is None
    assert np.median(np.linalg.norm(centerline - reference, axis=1)) < 0.01
    assert np.isclose(
        diagnostics["centerline_length_m"],
        np.linalg.norm(np.diff(reference, axis=0), axis=1).sum(),
        rtol=1e-4,
    )


def test_robust_pca_initializer_ignores_tube_winding_paths() -> None:
    parameter = np.linspace(-1.0, 1.0, 100)
    reference = np.column_stack((0.18 * parameter, 0.0 * parameter, 0.0 * parameter))
    points = _tube_around_curve(reference, seed=31)

    centerline, diagnostics = initialize_rope_centerline_pca(points)

    assert diagnostics["initialization"] == "robust-principal-axis"
    assert diagnostics["principal_variance_fraction"] > 0.98
    assert 0.34 < diagnostics["centerline_length_m"] < 0.38
    assert np.max(np.abs(centerline[:, 1:])) < 0.01


def test_rope_chain_edges_are_ordered_and_open() -> None:
    np.testing.assert_array_equal(
        rope_chain_edges(5),
        np.asarray([[0, 1], [1, 2], [2, 3], [3, 4]], dtype=np.int32),
    )
