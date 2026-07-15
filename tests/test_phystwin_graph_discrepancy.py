import pickle
from pathlib import Path

import numpy as np

from bayesian_phystwin.phystwin_graph_anchor_comparison import (
    apply_graph_anchor_variants,
    graph_method_id,
)
from bayesian_phystwin.phystwin_graph_discrepancy import (
    graph_discrepancy_diagnostics,
    graph_smoothed_discrepancy_posterior,
    normalized_spring_laplacian,
)


def test_normalized_spring_laplacian_matches_path_graph() -> None:
    laplacian = normalized_spring_laplacian(
        3,
        np.array([[0, 1], [1, 2]]),
    ).toarray()
    expected = np.array(
        [
            [1.0, -1.0, 0.0],
            [-0.5, 1.0, -0.5],
            [0.0, -1.0, 1.0],
        ]
    )

    np.testing.assert_allclose(laplacian, expected, atol=1e-12)
    np.testing.assert_allclose(laplacian @ np.ones(3), 0.0, atol=1e-12)


def test_graph_posterior_reduces_laplacian_energy() -> None:
    springs = np.array([[0, 1], [1, 2]])
    laplacian = normalized_spring_laplacian(3, springs)
    mean = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    posterior = graph_smoothed_discrepancy_posterior(
        mean,
        np.ones(3),
        np.ones(3, dtype=bool),
        laplacian,
        prior_strength=1.0,
    )

    raw = graph_discrepancy_diagnostics(mean, springs, laplacian)
    smoothed = graph_discrepancy_diagnostics(posterior.mean, springs, laplacian)
    assert (
        smoothed["laplacian_energy_m2_per_node"] < raw["laplacian_energy_m2_per_node"]
    )
    assert 0.0 < posterior.mean[1, 0] < 1.0
    assert posterior.solve_relative_residuals[0] < 1e-6


def test_graph_posterior_computes_exact_selected_marginal_variance() -> None:
    laplacian = normalized_spring_laplacian(3, np.array([[0, 1], [1, 2]]))
    variance = np.array([1e-4, 2e-4, 4e-4])
    posterior = graph_smoothed_discrepancy_posterior(
        np.zeros((3, 3)),
        variance,
        np.ones(3, dtype=bool),
        laplacian,
        prior_strength=0.3,
        ridge=1e-8,
        covariance_indices=np.array([0, 2]),
    )
    reference = np.median(variance)
    weights = reference / variance
    precision = (
        np.diag(weights) + 0.6 * (laplacian.T @ laplacian).toarray() + 1e-8 * np.eye(3)
    )
    expected = reference * np.diag(np.linalg.inv(precision))

    assert posterior.marginal_variance is not None
    np.testing.assert_allclose(
        posterior.marginal_variance[[0, 2]], expected[[0, 2]], rtol=1e-7
    )
    assert np.isnan(posterior.marginal_variance[1])


def test_graph_anchor_variants_share_the_endpoint_posterior(tmp_path: Path) -> None:
    frame_count = 6
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.01, 0.0, 0.0],
            [0.01, 0.01, 0.0],
            [0.0, 0.01, 0.0],
        ]
    )
    baseline = np.repeat(points[None], frame_count, axis=0)
    observed = baseline.copy()
    observed[1:, 0, 2] += 0.004
    data = {
        "object_points": observed.astype(np.float32),
        "object_visibilities": np.ones((frame_count, 4), dtype=bool),
        "object_motions_valid": np.ones((frame_count, 4), dtype=bool),
        "controller_points": np.zeros((frame_count, 1, 3), dtype=np.float32),
        "surface_points": np.empty((0, 3), dtype=np.float32),
        "interior_points": np.empty((0, 3), dtype=np.float32),
    }
    optimal = {
        "object_radius": 0.011,
        "object_max_neighbours": 3,
        "controller_radius": 0.02,
        "controller_max_neighbours": 2,
    }
    paths = {
        "final": tmp_path / "final.pkl",
        "baseline": tmp_path / "baseline.pkl",
        "optimal": tmp_path / "optimal.pkl",
    }
    for key, value in (("final", data), ("baseline", baseline), ("optimal", optimal)):
        with paths[key].open("wb") as handle:
            pickle.dump(value, handle)

    summary = apply_graph_anchor_variants(
        paths["final"],
        paths["baseline"],
        paths["optimal"],
        tmp_path / "output",
        train_end_frame=4,
        prior_strengths=(0.3,),
    )

    method = graph_method_id(0.3)
    assert tuple(summary["methods"]) == ("raw_per_point", "knn_lifted", method)
    assert summary["raw_knn_identical"] is True
    assert summary["graph"]["spring_count"] == 4
    assert (
        summary["methods"][method]["correction"]["laplacian_energy_m2_per_node"]
        < summary["methods"]["raw_per_point"]["correction"][
            "laplacian_energy_m2_per_node"
        ]
    )
