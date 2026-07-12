import numpy as np
import pytest

from causal4d.graph_temporal_discrepancy import (
    fit_graph_temporal_discrepancy,
    forecast_graph_temporal_discrepancy,
    graph_laplacian_basis,
    project_graph_coefficients,
)


pytest.importorskip("scipy")


def _path_graph(node_count: int) -> np.ndarray:
    return np.column_stack((np.arange(node_count - 1), np.arange(1, node_count)))


def _residual_sequence() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    node_count = 24
    basis, eigenvalues = graph_laplacian_basis(
        node_count,
        _path_graph(node_count),
        rank=6,
    )
    coefficients = np.zeros((30, 6, 3), dtype=float)
    coefficients[0, 0, 0] = 0.02
    coefficients[0, 1, 1] = -0.01
    for frame in range(1, len(coefficients)):
        coefficients[frame] = 0.92 * coefficients[frame - 1]
    residual = np.einsum("nr,trc->tnc", basis, coefficients)
    valid = np.ones(residual.shape[:2], dtype=bool)
    return residual, valid, basis, eigenvalues


def test_graph_basis_and_projection_recover_smooth_residuals() -> None:
    residual, valid, basis, _ = _residual_sequence()
    coefficients = project_graph_coefficients(
        residual,
        valid,
        basis,
        ridge=1e-10,
    )
    reconstructed = np.einsum("nr,trc->tnc", basis, coefficients)
    np.testing.assert_allclose(reconstructed, residual, atol=1e-8)


def test_graph_temporal_model_is_stable_and_forecasts_correlated_variance() -> None:
    residual, valid, basis, eigenvalues = _residual_sequence()
    model = fit_graph_temporal_discrepancy(
        residual[:24],
        valid[:24],
        basis,
        eigenvalues,
        rank_candidates=(2, 4, 6),
        projection_ridge=1e-8,
        dynamics_ridge=1e-8,
    )
    assert model.selected_rank in {2, 4, 6}
    assert model.spectral_radius <= 0.995 + 1e-12
    assert len(model.candidate_validation_rmse_m) == 3

    mean, variance = forecast_graph_temporal_discrepancy(
        model,
        residual[20:24],
        valid[20:24],
        total_frame_count=10,
    )
    assert mean.shape == (10, 24, 3)
    assert variance.shape == mean.shape
    assert np.all(variance >= 0.0)
    assert np.linalg.norm(mean[4]) < np.linalg.norm(mean[3])


def test_persistence_and_learned_dynamics_are_distinct() -> None:
    residual, valid, basis, eigenvalues = _residual_sequence()
    model = fit_graph_temporal_discrepancy(
        residual[:24],
        valid[:24],
        basis,
        eigenvalues,
        rank_candidates=(2,),
        projection_ridge=1e-8,
        dynamics_ridge=1e-8,
    )
    learned, _ = forecast_graph_temporal_discrepancy(
        model,
        residual[20:24],
        valid[20:24],
        total_frame_count=8,
        dynamics="learned",
    )
    persistent, _ = forecast_graph_temporal_discrepancy(
        model,
        residual[20:24],
        valid[20:24],
        total_frame_count=8,
        dynamics="persistence",
    )
    assert not np.allclose(learned[4:], persistent[4:])
    np.testing.assert_allclose(persistent[4], persistent[3], atol=1e-7)
