import numpy as np
import pytest

from bayesian_phystwin.phystwin_geodesic_belief import MaterialGeodesicGraph
from bayesian_phystwin.phystwin_graph_residual_mapping import (
    GraphResidualMappingConfig,
    fit_graph_residual_mapping,
)


def _chain(count: int = 12) -> MaterialGeodesicGraph:
    positions = np.column_stack(
        (np.linspace(0.0, 1.0, count), np.zeros(count), np.zeros(count))
    )
    edges = np.column_stack((np.arange(count - 1), np.arange(1, count)))
    return MaterialGeodesicGraph(positions, edges)


def test_graph_residual_mapping_recovers_global_translation() -> None:
    graph = _chain()
    centers = np.asarray([0, 3, 6, 9, 11])
    residual = np.repeat(np.asarray([[0.02, -0.01, 0.005]]), len(centers), axis=0)

    result = fit_graph_residual_mapping(
        graph.reference_positions_m,
        centers,
        residual,
        np.ones(len(centers), dtype=bool),
        graph=graph,
    )

    np.testing.assert_allclose(
        result.correction_m,
        np.repeat(residual[:1], graph.node_count, axis=0),
        atol=1e-6,
    )
    assert result.selected_regularization == 100.0
    assert result.leave_one_out_rmse_m < 1e-6
    assert result.observation_count == len(centers)
    assert result.clipped_point_count == 0


def test_graph_residual_mapping_interpolates_linear_displacement() -> None:
    graph = _chain()
    centers = np.asarray([0, 3, 6, 9, 11])
    residual = np.column_stack(
        (
            0.01 * graph.reference_positions_m[centers, 0],
            np.zeros(len(centers)),
            np.zeros(len(centers)),
        )
    )

    result = fit_graph_residual_mapping(
        graph.reference_positions_m,
        centers,
        residual,
        np.ones(len(centers), dtype=bool),
        graph=graph,
    )

    np.testing.assert_allclose(
        result.correction_m[:, 0],
        0.01 * graph.reference_positions_m[:, 0],
        atol=2e-5,
    )
    np.testing.assert_array_equal(result.correction_m[:, 1:], 0.0)


def test_graph_residual_mapping_ignores_unavailable_nonfinite_measurements() -> None:
    graph = _chain()
    centers = np.asarray([0, 3, 6, 9, 11])
    residual = np.repeat(np.asarray([[0.01, 0.0, 0.0]]), len(centers), axis=0)
    residual[1] = np.nan
    available = np.ones(len(centers), dtype=bool)

    result = fit_graph_residual_mapping(
        graph.reference_positions_m,
        centers,
        residual,
        available,
        graph=graph,
    )

    assert result.observation_count == 4
    np.testing.assert_allclose(result.correction_m[:, 0], 0.01, atol=1e-6)


def test_graph_residual_mapping_clips_large_point_corrections() -> None:
    graph = _chain()
    centers = np.asarray([0, 3, 6, 9, 11])
    residual = np.repeat(np.asarray([[0.5, 0.0, 0.0]]), len(centers), axis=0)

    result = fit_graph_residual_mapping(
        graph.reference_positions_m,
        centers,
        residual,
        np.ones(len(centers), dtype=bool),
        config=GraphResidualMappingConfig(maximum_correction_m=0.1),
        graph=graph,
    )

    np.testing.assert_allclose(np.linalg.norm(result.correction_m, axis=1), 0.1)
    assert result.clipped_point_count == graph.node_count


def test_graph_residual_mapping_rejects_insufficient_support() -> None:
    graph = _chain()
    centers = np.asarray([0, 3, 6, 9, 11])
    available = np.asarray([True, True, False, False, False])

    with pytest.raises(ValueError, match="too few finite observations"):
        fit_graph_residual_mapping(
            graph.reference_positions_m,
            centers,
            np.zeros((len(centers), 3)),
            available,
            graph=graph,
        )


def test_graph_residual_mapping_config_grid_must_be_frozen_ordered() -> None:
    with pytest.raises(ValueError, match="unique and increasing"):
        GraphResidualMappingConfig(regularization_grid=(1.0, 0.1, 1.0))
