from __future__ import annotations

import numpy as np

from bayesian_phystwin.phystwin_sparse_state_update import (
    closure_gate_passed,
    fixed_identity_node_association,
    low_frequency_scalar_graph_basis,
    nonlinear_closure_diagnostics,
    prefix_persistence_correction,
)


def _cycle_edges(node_count: int) -> np.ndarray:
    return np.column_stack(
        (
            np.arange(node_count, dtype=np.int64),
            np.roll(np.arange(node_count, dtype=np.int64), -1),
        )
    )


def test_low_frequency_graph_basis_is_deterministic_and_orthonormal() -> None:
    first_basis, first_values, diagnostics = low_frequency_scalar_graph_basis(
        12,
        _cycle_edges(12),
        rank=4,
    )
    second_basis, second_values, _ = low_frequency_scalar_graph_basis(
        12,
        _cycle_edges(12),
        rank=4,
    )

    np.testing.assert_array_equal(first_basis, second_basis)
    np.testing.assert_array_equal(first_values, second_values)
    np.testing.assert_allclose(first_basis.T @ first_basis, np.eye(4), atol=1e-12)
    assert first_values[0] < 1e-12
    assert diagnostics["eigenpair_residual_norms"][0] < 1e-10


def test_identity_association_uses_only_frame_zero_geometry() -> None:
    state = np.column_stack((np.arange(6), np.zeros((6, 2)))).astype(float)
    tracks = np.asarray(((0.1, 0.0, 0.0), (4.8, 0.0, 0.0), (2.2, 0.0, 0.0)))

    nodes, distances, diagnostics = fixed_identity_node_association(
        state,
        tracks,
        np.asarray((0, 2), dtype=np.int64),
    )

    np.testing.assert_array_equal(nodes, np.asarray((0, 2)))
    np.testing.assert_allclose(distances, np.asarray((0.1, 0.2)))
    assert diagnostics["association_uses_frame_zero_only"]


def test_prefix_persistence_is_limited_over_the_full_graph() -> None:
    coordinate = np.linspace(-1.0, 1.0, 10)
    full_basis = np.linalg.qr(
        np.column_stack(
            (
                np.ones(10),
                coordinate,
                coordinate**2,
                coordinate**3,
            )
        )
    )[0]
    observed = np.asarray((0, 3, 6, 9))
    observed_basis = full_basis[observed]
    coefficients = np.asarray(
        (
            (0.02, 0.0, 0.0),
            (0.0, 0.01, 0.0),
            (0.0, 0.0, 0.01),
            (0.0, 0.0, 0.0),
        )
    )
    innovation = np.repeat(
        (observed_basis @ coefficients)[None],
        4,
        axis=0,
    )

    field, _, diagnostics = prefix_persistence_correction(
        innovation,
        np.ones((4, 4), dtype=bool),
        observed_basis,
        full_basis,
        fit_frame_count=4,
        ridge=1e-8,
        maximum_node_norm_m=0.003,
    )

    assert np.max(np.linalg.norm(field, axis=1)) <= 0.003 + 1e-12
    assert diagnostics["field_limit"]["limit_applied"]


def test_nonlinear_closure_gate_rejects_large_model_error() -> None:
    response = np.zeros((3, 2, 3, 1), dtype=float)
    response[:, :, 0, 0] = 0.002
    weights = np.asarray((1.0,))
    nonlinear = np.einsum("tncp,p->tnc", response, weights)
    good = nonlinear_closure_diagnostics(
        response,
        weights,
        nonlinear,
        np.ones((3, 2), dtype=bool),
    )
    bad_nonlinear = nonlinear.copy()
    bad_nonlinear[:, :, 1] = 0.004
    bad = nonlinear_closure_diagnostics(
        response,
        weights,
        bad_nonlinear,
        np.ones((3, 2), dtype=bool),
    )

    assert closure_gate_passed(
        good,
        maximum_vector_rmse_m=0.001,
        maximum_relative_vector_rmse=0.25,
    )
    assert not closure_gate_passed(
        bad,
        maximum_vector_rmse_m=0.001,
        maximum_relative_vector_rmse=0.25,
    )
