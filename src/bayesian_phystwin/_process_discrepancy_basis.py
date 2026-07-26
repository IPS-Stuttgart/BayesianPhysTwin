"""Constrained graph force basis for process discrepancy."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ._process_discrepancy_common import cross_matrix, readonly


def _expanded_graph_operator(
    graph_basis: np.ndarray,
    support_weights: np.ndarray,
) -> np.ndarray:
    node_count, graph_rank = graph_basis.shape
    operator = np.zeros((3 * node_count, 3 * graph_rank), dtype=float)
    weighted_basis = support_weights[:, None] * graph_basis
    for mode in range(graph_rank):
        for coordinate in range(3):
            operator[coordinate::3, 3 * mode + coordinate] = weighted_basis[:, mode]
    return operator


def _constraint_matrix(
    node_positions_m: np.ndarray,
    balanced_nodes: np.ndarray,
    *,
    enforce_zero_net_force: bool,
    enforce_zero_net_torque: bool,
) -> np.ndarray:
    node_count = len(node_positions_m)
    selected = np.flatnonzero(balanced_nodes)
    rows: list[np.ndarray] = []
    if enforce_zero_net_force and len(selected):
        for coordinate in range(3):
            row = np.zeros(3 * node_count, dtype=float)
            row[3 * selected + coordinate] = 1.0
            rows.append(row)
    if enforce_zero_net_torque and len(selected):
        center = np.mean(node_positions_m[selected], axis=0)
        torque_rows = np.zeros((3, 3 * node_count), dtype=float)
        for node_index in selected:
            relative_position = node_positions_m[node_index] - center
            torque_rows[:, 3 * node_index : 3 * node_index + 3] = cross_matrix(
                relative_position
            )
        rows.extend(torque_rows)
    if not rows:
        return np.empty((0, 3 * node_count), dtype=float)
    return np.stack(rows)


def _nullspace(matrix: np.ndarray, *, relative_tolerance: float) -> np.ndarray:
    column_count = matrix.shape[1]
    if matrix.shape[0] == 0:
        return np.eye(column_count)
    _, singular_values, right_vectors = np.linalg.svd(matrix, full_matrices=True)
    scale = max(float(singular_values[0]) if len(singular_values) else 0.0, 1.0)
    threshold = relative_tolerance * max(matrix.shape) * scale
    rank = int(np.sum(singular_values > threshold))
    return right_vectors[rank:].T.copy()


@dataclass(frozen=True)
class ProcessDiscrepancyBasisV1:
    """Constrained low-rank map from latent coefficients to nodal force."""

    graph_basis: np.ndarray
    graph_eigenvalues: np.ndarray
    node_positions_m: np.ndarray
    support_weights: np.ndarray
    externally_supported: np.ndarray
    force_operator: np.ndarray
    latent_to_graph_coefficients: np.ndarray
    constraint_matrix: np.ndarray
    latent_graph_roughness: np.ndarray
    enforce_zero_net_force: bool
    enforce_zero_net_torque: bool

    def __post_init__(self) -> None:
        graph_basis = readonly(self.graph_basis)
        graph_eigenvalues = readonly(self.graph_eigenvalues)
        node_positions = readonly(self.node_positions_m)
        support_weights = readonly(self.support_weights)
        externally_supported = readonly(self.externally_supported, dtype=bool)
        force_operator = readonly(self.force_operator)
        latent_to_graph = readonly(self.latent_to_graph_coefficients)
        constraints = readonly(self.constraint_matrix)
        roughness = readonly(self.latent_graph_roughness)

        if graph_basis.ndim != 2 or graph_basis.shape[1] < 1:
            raise ValueError("graph_basis must have shape (node, rank) with rank > 0")
        node_count, graph_rank = graph_basis.shape
        if node_positions.shape != (node_count, 3):
            raise ValueError("node_positions_m must have shape (node, 3)")
        if graph_eigenvalues.shape != (graph_rank,):
            raise ValueError("graph_eigenvalues must match graph basis rank")
        if support_weights.shape != (node_count,):
            raise ValueError("support_weights must match graph node count")
        if externally_supported.shape != (node_count,):
            raise ValueError("externally_supported must match graph node count")
        if not np.all(np.isfinite(graph_basis)) or not np.all(
            np.isfinite(node_positions)
        ):
            raise ValueError("graph basis and node positions must be finite")
        if not np.all(np.isfinite(graph_eigenvalues)) or np.any(
            graph_eigenvalues < 0.0
        ):
            raise ValueError("graph_eigenvalues must be finite and nonnegative")
        if not np.all(np.isfinite(support_weights)) or np.any(support_weights < 0.0):
            raise ValueError("support_weights must be finite and nonnegative")
        if np.any(support_weights > 1.0):
            raise ValueError("support_weights must not exceed one")
        if not np.allclose(
            graph_basis.T @ graph_basis,
            np.eye(graph_rank),
            atol=1e-7,
            rtol=1e-7,
        ):
            raise ValueError("graph_basis columns must be orthonormal")

        if force_operator.ndim != 2 or force_operator.shape[0] != 3 * node_count:
            raise ValueError("force_operator must have shape (3 * node, latent)")
        latent_dimension = force_operator.shape[1]
        if latent_dimension < 1:
            raise ValueError("force_operator must retain at least one latent direction")
        if latent_to_graph.shape != (3 * graph_rank, latent_dimension):
            raise ValueError(
                "latent_to_graph_coefficients must have shape "
                "(3 * graph_rank, latent)"
            )
        if constraints.ndim != 2 or constraints.shape[1] != 3 * node_count:
            raise ValueError("constraint_matrix must have shape (constraint, 3 * node)")
        if roughness.shape != (latent_dimension,):
            raise ValueError("latent_graph_roughness must match latent dimension")
        arrays = (force_operator, latent_to_graph, constraints, roughness)
        if not all(np.all(np.isfinite(values)) for values in arrays):
            raise ValueError(
                "latent mappings, constraints, and roughness must be finite"
            )
        if np.any(roughness < -1e-12):
            raise ValueError("latent_graph_roughness must be nonnegative")
        if not np.allclose(
            force_operator.T @ force_operator,
            np.eye(latent_dimension),
            atol=1e-7,
            rtol=1e-7,
        ):
            raise ValueError("force_operator columns must be orthonormal")
        raw_operator = _expanded_graph_operator(graph_basis, support_weights)
        if not np.allclose(
            raw_operator @ latent_to_graph,
            force_operator,
            atol=1e-7,
            rtol=1e-7,
        ):
            raise ValueError("latent-to-graph map does not reproduce force_operator")
        if constraints.shape[0] and not np.allclose(
            constraints @ force_operator,
            0.0,
            atol=1e-8,
            rtol=1e-8,
        ):
            raise ValueError("force_operator violates declared hard constraints")

        object.__setattr__(self, "graph_basis", graph_basis)
        object.__setattr__(self, "graph_eigenvalues", graph_eigenvalues)
        object.__setattr__(self, "node_positions_m", node_positions)
        object.__setattr__(self, "support_weights", support_weights)
        object.__setattr__(self, "externally_supported", externally_supported)
        object.__setattr__(self, "force_operator", force_operator)
        object.__setattr__(self, "latent_to_graph_coefficients", latent_to_graph)
        object.__setattr__(self, "constraint_matrix", constraints)
        object.__setattr__(self, "latent_graph_roughness", np.maximum(roughness, 0.0))

    @property
    def node_count(self) -> int:
        return self.graph_basis.shape[0]

    @property
    def graph_rank(self) -> int:
        return self.graph_basis.shape[1]

    @property
    def latent_dimension(self) -> int:
        return self.force_operator.shape[1]

    @property
    def balanced_nodes(self) -> np.ndarray:
        values = (~self.externally_supported) & (self.support_weights > 0.0)
        values.setflags(write=False)
        return values

    def force_from_coefficients(self, coefficients_n: np.ndarray) -> np.ndarray:
        coefficients = np.asarray(coefficients_n, dtype=float)
        if coefficients.shape != (self.latent_dimension,):
            raise ValueError("coefficients_n must match latent dimension")
        if not np.all(np.isfinite(coefficients)):
            raise ValueError("coefficients_n must be finite")
        return (self.force_operator @ coefficients).reshape(self.node_count, 3)

    def graph_coefficients_from_latent(
        self,
        coefficients_n: np.ndarray,
    ) -> np.ndarray:
        coefficients = np.asarray(coefficients_n, dtype=float)
        if coefficients.shape != (self.latent_dimension,):
            raise ValueError("coefficients_n must match latent dimension")
        return (self.latent_to_graph_coefficients @ coefficients).reshape(
            self.graph_rank,
            3,
        )

    def constraint_residual(self, force_n: np.ndarray) -> np.ndarray:
        force = np.asarray(force_n, dtype=float)
        if force.shape != (self.node_count, 3):
            raise ValueError("force_n must have shape (node, 3)")
        return self.constraint_matrix @ force.reshape(-1)

    def marginal_force_variance_n2(self, covariance_n2: np.ndarray) -> np.ndarray:
        covariance = np.asarray(covariance_n2, dtype=float)
        if covariance.shape != (self.latent_dimension, self.latent_dimension):
            raise ValueError("covariance_n2 must match latent dimension")
        projected = self.force_operator @ covariance
        diagonal = np.einsum("ij,ij->i", projected, self.force_operator)
        return np.maximum(diagonal, 0.0).reshape(self.node_count, 3)


def build_process_discrepancy_basis(
    graph_basis: np.ndarray,
    node_positions_m: np.ndarray,
    *,
    graph_eigenvalues: np.ndarray | None = None,
    support_weights: np.ndarray | None = None,
    externally_supported: np.ndarray | None = None,
    enforce_zero_net_force: bool = True,
    enforce_zero_net_torque: bool = True,
    relative_singular_value_tolerance: float = 1e-10,
) -> ProcessDiscrepancyBasisV1:
    """Build a contact-aware force basis with hard momentum constraints."""

    basis = np.asarray(graph_basis, dtype=float)
    positions = np.asarray(node_positions_m, dtype=float)
    if basis.ndim != 2 or basis.shape[1] < 1:
        raise ValueError("graph_basis must have shape (node, rank) with rank > 0")
    node_count, graph_rank = basis.shape
    if positions.shape != (node_count, 3):
        raise ValueError("node_positions_m must have shape (node, 3)")
    if relative_singular_value_tolerance <= 0.0:
        raise ValueError("relative_singular_value_tolerance must be positive")
    eigenvalues = (
        np.zeros(graph_rank, dtype=float)
        if graph_eigenvalues is None
        else np.asarray(graph_eigenvalues, dtype=float)
    )
    weights = (
        np.ones(node_count, dtype=float)
        if support_weights is None
        else np.asarray(support_weights, dtype=float)
    )
    external = (
        np.zeros(node_count, dtype=bool)
        if externally_supported is None
        else np.asarray(externally_supported, dtype=bool)
    )
    if eigenvalues.shape != (graph_rank,):
        raise ValueError("graph_eigenvalues must match graph basis rank")
    if weights.shape != (node_count,):
        raise ValueError("support_weights must match graph node count")
    if external.shape != (node_count,):
        raise ValueError("externally_supported must match graph node count")
    if not np.all(np.isfinite(basis)) or not np.all(np.isfinite(positions)):
        raise ValueError("graph basis and node positions must be finite")
    if not np.all(np.isfinite(eigenvalues)) or np.any(eigenvalues < 0.0):
        raise ValueError("graph_eigenvalues must be finite and nonnegative")
    if not np.all(np.isfinite(weights)) or np.any(weights < 0.0) or np.any(
        weights > 1.0
    ):
        raise ValueError("support_weights must be finite values in [0, 1]")
    if not np.any(weights > 0.0):
        raise ValueError("at least one node must have nonzero support weight")
    if not np.allclose(
        basis.T @ basis,
        np.eye(graph_rank),
        atol=1e-7,
        rtol=1e-7,
    ):
        raise ValueError("graph_basis columns must be orthonormal")

    raw_operator = _expanded_graph_operator(basis, weights)
    balanced_nodes = (~external) & (weights > 0.0)
    constraints = _constraint_matrix(
        positions,
        balanced_nodes,
        enforce_zero_net_force=enforce_zero_net_force,
        enforce_zero_net_torque=enforce_zero_net_torque,
    )
    coefficient_nullspace = _nullspace(
        constraints @ raw_operator,
        relative_tolerance=relative_singular_value_tolerance,
    )
    if coefficient_nullspace.shape[1] == 0:
        raise ValueError("hard constraints remove the entire graph force span")
    constrained_operator = raw_operator @ coefficient_nullspace
    left_vectors, singular_values, right_vectors = np.linalg.svd(
        constrained_operator,
        full_matrices=False,
    )
    if not len(singular_values):
        raise ValueError("graph force span is empty")
    threshold = (
        relative_singular_value_tolerance
        * max(constrained_operator.shape)
        * max(float(singular_values[0]), 1.0)
    )
    retained = singular_values > threshold
    if not np.any(retained):
        raise ValueError("graph force span is numerically rank deficient")
    force_operator = left_vectors[:, retained]
    latent_to_graph = coefficient_nullspace @ (
        right_vectors[retained].T / singular_values[retained][None, :]
    )
    repeated_eigenvalues = np.repeat(eigenvalues, 3)
    roughness = np.einsum(
        "ij,i,ij->j",
        latent_to_graph,
        repeated_eigenvalues,
        latent_to_graph,
    )
    return ProcessDiscrepancyBasisV1(
        graph_basis=basis,
        graph_eigenvalues=eigenvalues,
        node_positions_m=positions,
        support_weights=weights,
        externally_supported=external,
        force_operator=force_operator,
        latent_to_graph_coefficients=latent_to_graph,
        constraint_matrix=constraints,
        latent_graph_roughness=np.maximum(roughness, 0.0),
        enforce_zero_net_force=bool(enforce_zero_net_force),
        enforce_zero_net_torque=bool(enforce_zero_net_torque),
    )
