"""Source-safe helpers for sparse-identity PhysTwin state updates."""

from __future__ import annotations

from typing import Any

import numpy as np

from .dynamic_discrepancy import (
    project_prefix_graph_coefficients,
    scale_coefficients_to_field_limit,
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _canonical_sign(values: np.ndarray) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float64).copy()
    pivot = int(np.argmax(np.abs(vector)))
    if vector[pivot] < 0.0:
        vector *= -1.0
    return vector


def low_frequency_scalar_graph_basis(
    node_count: int,
    springs: np.ndarray,
    *,
    rank: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Return deterministic low-frequency modes of the spring graph.

    The symmetric normalized Laplacian is used here because its eigenvectors
    are orthonormal in the Euclidean inner product expected by the propagated
    state artifact. The null mode is retained: global low-frequency state
    error is a live hypothesis in the released single-lift diagnostic.
    """

    try:
        from scipy import sparse
        from scipy.sparse.linalg import eigsh
    except (ImportError, OSError) as error:
        raise RuntimeError("sparse state modes require scipy") from error

    edges = np.asarray(springs, dtype=np.int64)
    _require(node_count >= 2, "node_count must be at least two")
    _require(1 <= rank < node_count, "rank must lie in [1, node_count)")
    _require(
        edges.ndim == 2 and edges.shape[1] == 2 and len(edges) >= 1,
        "springs must have nonempty shape (S, 2)",
    )
    _require(
        not np.any(edges < 0) and not np.any(edges >= node_count),
        "spring endpoint exceeds node_count",
    )
    _require(not np.any(edges[:, 0] == edges[:, 1]), "self springs are unsupported")

    rows = np.concatenate((edges[:, 0], edges[:, 1]))
    columns = np.concatenate((edges[:, 1], edges[:, 0]))
    adjacency = sparse.coo_matrix(
        (np.ones(len(rows), dtype=np.float64), (rows, columns)),
        shape=(node_count, node_count),
    ).tocsr()
    adjacency.sum_duplicates()
    adjacency.data[:] = 1.0
    degree = np.asarray(adjacency.sum(axis=1)).reshape(-1)
    _require(np.all(degree > 0.0), "every state node must belong to the spring graph")
    inverse_root = 1.0 / np.sqrt(degree)
    scaling = sparse.diags(inverse_root, format="csr")
    laplacian = sparse.eye(node_count, format="csr") - scaling @ adjacency @ scaling

    if node_count <= 256:
        eigenvalues, basis = np.linalg.eigh(laplacian.toarray())
        eigenvalues = eigenvalues[:rank]
        basis = basis[:, :rank]
        solver = "dense_eigh"
    else:
        coordinate = np.arange(node_count, dtype=np.float64)
        initial = np.cos((coordinate + 0.5) * np.sqrt(2.0))
        eigenvalues, basis = eigsh(
            laplacian,
            k=rank,
            which="SM",
            v0=initial,
            tol=1e-10,
        )
        order = np.argsort(eigenvalues, kind="mergesort")
        eigenvalues = eigenvalues[order]
        basis = basis[:, order]
        solver = "sparse_eigsh"

    eigenvalues = np.maximum(np.asarray(eigenvalues, dtype=np.float64), 0.0)
    basis = np.column_stack(
        [_canonical_sign(basis[:, mode]) for mode in range(rank)]
    )
    orthonormality_error = float(
        np.max(np.abs(basis.T @ basis - np.eye(rank)), initial=0.0)
    )
    residual = laplacian @ basis - basis * eigenvalues[None]
    residual_norms = np.linalg.norm(residual, axis=0)
    _require(orthonormality_error <= 1e-7, "graph modes are not orthonormal")
    _require(
        np.max(residual_norms, initial=0.0) <= 1e-6,
        "graph eigenpair residual is too large",
    )
    return (
        basis,
        eigenvalues,
        {
            "solver": solver,
            "rank": rank,
            "node_count": node_count,
            "spring_count": int(len(edges)),
            "orthonormality_maximum_error": orthonormality_error,
            "eigenpair_residual_norms": residual_norms.tolist(),
        },
    )


def fixed_identity_node_association(
    frame_zero_state_m: np.ndarray,
    frame_zero_tracks_m: np.ndarray,
    identity_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Associate selected identities once, using frame-zero geometry only."""

    state = np.asarray(frame_zero_state_m, dtype=np.float64)
    tracks = np.asarray(frame_zero_tracks_m, dtype=np.float64)
    identities = np.asarray(identity_indices, dtype=np.int64)
    _require(
        state.ndim == 2 and state.shape[1] == 3 and np.all(np.isfinite(state)),
        "frame-zero state must have finite shape (N, 3)",
    )
    _require(
        tracks.ndim == 2 and tracks.shape[1] == 3,
        "frame-zero tracks must have shape (K, 3)",
    )
    _require(
        identities.ndim == 1
        and len(identities) >= 1
        and not np.any(identities < 0)
        and not np.any(identities >= len(tracks)),
        "identity indices exceed frame-zero tracks",
    )
    selected = tracks[identities]
    _require(np.all(np.isfinite(selected)), "selected frame-zero track is non-finite")
    squared = np.sum(
        np.square(state[:, None, :] - selected[None, :, :]),
        axis=2,
    )
    nodes = np.argmin(squared, axis=0).astype(np.int64)
    distances = np.sqrt(squared[nodes, np.arange(len(nodes))])
    return (
        nodes,
        distances,
        {
            "association_uses_frame_zero_only": True,
            "identity_count": int(len(identities)),
            "unique_node_count": int(len(np.unique(nodes))),
            "maximum_initial_distance_m": float(np.max(distances, initial=0.0)),
            "mean_initial_distance_m": float(np.mean(distances)),
        },
    )


def prefix_persistence_correction(
    innovation_m: np.ndarray,
    available: np.ndarray,
    observed_graph_basis: np.ndarray,
    full_graph_basis: np.ndarray,
    *,
    fit_frame_count: int,
    ridge: float,
    maximum_node_norm_m: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Fit the information-matched persistent readout comparator."""

    innovation = np.asarray(innovation_m, dtype=np.float64)
    mask = np.asarray(available, dtype=bool)
    observed_basis = np.asarray(observed_graph_basis, dtype=np.float64)
    full_basis = np.asarray(full_graph_basis, dtype=np.float64)
    _require(
        1 <= fit_frame_count <= len(innovation),
        "fit_frame_count exceeds the prefix",
    )
    _require(
        observed_basis.shape[1] == full_basis.shape[1],
        "observed and full graph basis ranks differ",
    )
    history = project_prefix_graph_coefficients(
        innovation[:fit_frame_count],
        mask[:fit_frame_count],
        observed_basis,
        ridge=ridge,
    )
    coefficients, limit = scale_coefficients_to_field_limit(
        full_basis,
        history[-1],
        maximum_node_norm=maximum_node_norm_m,
    )
    return (
        full_basis @ coefficients,
        coefficients,
        {
            "fit_frame_count": fit_frame_count,
            "projection_ridge": ridge,
            "field_limit": limit,
        },
    )


def nonlinear_closure_diagnostics(
    state_response_at_step_m: np.ndarray,
    state_weights: np.ndarray,
    nonlinear_state_displacement_m: np.ndarray,
    available: np.ndarray,
) -> dict[str, Any]:
    """Compare the accepted linearized update with its nonlinear Warp replay."""

    response = np.asarray(state_response_at_step_m, dtype=np.float64)
    weights = np.asarray(state_weights, dtype=np.float64)
    nonlinear = np.asarray(nonlinear_state_displacement_m, dtype=np.float64)
    mask = np.asarray(available, dtype=bool)
    _require(
        response.ndim == 4 and response.shape[:3] == nonlinear.shape,
        "closure response and nonlinear displacement shapes differ",
    )
    _require(
        weights.shape == (response.shape[3],),
        "closure state weight shape changed",
    )
    _require(mask.shape == nonlinear.shape[:2], "closure availability shape changed")
    linear = np.einsum("tncp,p->tnc", response, weights)
    finite = mask & np.all(np.isfinite(nonlinear), axis=2)
    _require(np.any(finite), "closure interval has no valid observations")
    difference = nonlinear[finite] - linear[finite]
    linear_selected = linear[finite]
    vector_rmse = float(
        np.sqrt(np.mean(np.sum(np.square(difference), axis=1)))
    )
    coordinate_rmse = float(np.sqrt(np.mean(np.square(difference))))
    reference_rms = float(
        np.sqrt(np.mean(np.sum(np.square(linear_selected), axis=1)))
    )
    if reference_rms > np.finfo(np.float64).tiny:
        relative_rmse = vector_rmse / reference_rms
    elif vector_rmse <= np.finfo(np.float64).tiny:
        relative_rmse = 0.0
    else:
        relative_rmse = float(np.finfo(np.float64).max)
    return {
        "valid_point_frame_count": int(np.sum(finite)),
        "coordinate_rmse_m": coordinate_rmse,
        "vector_rmse_m": vector_rmse,
        "linearized_displacement_rms_m": reference_rms,
        "relative_vector_rmse": relative_rmse,
    }


def closure_gate_passed(
    diagnostics: dict[str, Any],
    *,
    maximum_vector_rmse_m: float,
    maximum_relative_vector_rmse: float,
) -> bool:
    """Apply a predeclared prefix-only nonlinear-closure gate."""

    _require(maximum_vector_rmse_m > 0.0, "closure RMSE limit must be positive")
    _require(
        maximum_relative_vector_rmse > 0.0,
        "relative closure limit must be positive",
    )
    return bool(
        float(diagnostics["vector_rmse_m"]) <= maximum_vector_rmse_m
        and float(diagnostics["relative_vector_rmse"])
        <= maximum_relative_vector_rmse
    )


__all__ = [
    "closure_gate_passed",
    "fixed_identity_node_association",
    "low_frequency_scalar_graph_basis",
    "nonlinear_closure_diagnostics",
    "prefix_persistence_correction",
]
