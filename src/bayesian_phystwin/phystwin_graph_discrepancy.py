"""Sparse Bayesian smoothing of PhysTwin endpoint discrepancy on its graph."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GraphDiscrepancyPosterior:
    """Posterior mean and approximate marginal variance on graph vertices."""

    mean: np.ndarray
    marginal_variance: np.ndarray | None
    observation_weight: np.ndarray
    reference_variance: float
    solve_iterations: tuple[int, ...]
    solve_relative_residuals: tuple[float, ...]
    covariance_negative_fraction: float | None


def normalized_spring_laplacian(
    node_count: int,
    springs: np.ndarray,
):
    """Return the random-walk normalized Laplacian of object springs.

    This dimensionless form keeps a constant displacement in the nullspace,
    including on graphs whose vertices have unequal degree.
    """

    try:
        from scipy import sparse
    except (ImportError, OSError) as error:
        raise RuntimeError("graph discrepancy smoothing requires scipy") from error
    edges = np.asarray(springs, dtype=np.int64)
    if node_count < 1:
        raise ValueError("node_count must be positive")
    if edges.ndim != 2 or edges.shape[1] != 2 or len(edges) < 1:
        raise ValueError("springs must have nonempty shape (S, 2)")
    if np.any(edges < 0) or np.any(edges >= node_count):
        raise ValueError("spring endpoint exceeds node_count")
    if np.any(edges[:, 0] == edges[:, 1]):
        raise ValueError("self springs are not supported")
    rows = np.concatenate((edges[:, 0], edges[:, 1]))
    columns = np.concatenate((edges[:, 1], edges[:, 0]))
    adjacency = sparse.coo_matrix(
        (np.ones(len(rows), dtype=float), (rows, columns)),
        shape=(node_count, node_count),
    ).tocsr()
    adjacency.sum_duplicates()
    adjacency.data[:] = 1.0
    degree = np.asarray(adjacency.sum(axis=1)).ravel()
    active = degree > 0.0
    inverse_degree = np.zeros(node_count, dtype=float)
    inverse_degree[active] = 1.0 / degree[active]
    scaling = sparse.diags(inverse_degree, format="csr")
    normalized_adjacency = scaling @ adjacency
    laplacian = sparse.diags(active.astype(float), format="csr")
    laplacian -= normalized_adjacency
    return laplacian.tocsr()


def _precision_solver(
    laplacian,
    observation_weight: np.ndarray,
    *,
    prior_strength: float,
    ridge: float,
    relative_tolerance: float,
    maximum_iterations: int,
):
    try:
        from scipy.sparse.linalg import LinearOperator, cg
    except (ImportError, OSError) as error:
        raise RuntimeError("graph discrepancy smoothing requires scipy") from error
    if prior_strength <= 0.0:
        raise ValueError("prior_strength must be positive")
    if ridge <= 0.0 or relative_tolerance <= 0.0 or maximum_iterations < 1:
        raise ValueError("solver settings must be positive")
    weights = np.asarray(observation_weight, dtype=float)
    if weights.ndim != 1 or len(weights) != laplacian.shape[0]:
        raise ValueError("observation_weight must match the Laplacian")
    if np.any(weights < 0.0) or not np.all(np.isfinite(weights)):
        raise ValueError("observation weights must be finite and nonnegative")
    prior_scale = 2.0 * prior_strength

    def precision_product(vector: np.ndarray) -> np.ndarray:
        values = np.asarray(vector, dtype=float)
        return (
            weights * values
            + prior_scale * (laplacian.T @ (laplacian @ values))
            + ridge * values
        )

    size = len(weights)
    precision = LinearOperator(
        (size, size),
        matvec=precision_product,
        rmatvec=precision_product,
        dtype=float,
    )
    laplacian_square_diagonal = np.asarray(laplacian.power(2).sum(axis=0)).ravel()
    diagonal = weights + prior_scale * laplacian_square_diagonal + ridge
    preconditioner = LinearOperator(
        (size, size),
        matvec=lambda vector: np.asarray(vector, dtype=float) / diagonal,
        rmatvec=lambda vector: np.asarray(vector, dtype=float) / diagonal,
        dtype=float,
    )

    def solve(right_hand_side: np.ndarray) -> tuple[np.ndarray, int, float]:
        rhs = np.asarray(right_hand_side, dtype=float)
        if rhs.shape != (size,):
            raise ValueError("right_hand_side must match graph node count")
        iteration_count = 0

        def count_iteration(_: np.ndarray) -> None:
            nonlocal iteration_count
            iteration_count += 1

        solution, info = cg(
            precision,
            rhs,
            M=preconditioner,
            rtol=relative_tolerance,
            atol=0.0,
            maxiter=maximum_iterations,
            callback=count_iteration,
        )
        residual = precision_product(solution) - rhs
        relative_residual = float(
            np.linalg.norm(residual) / max(np.linalg.norm(rhs), 1e-15)
        )
        if info != 0 or not np.all(np.isfinite(solution)):
            raise RuntimeError(
                "graph posterior conjugate-gradient solve did not converge: "
                f"info={info}, residual={relative_residual:.3e}"
            )
        return solution, iteration_count, relative_residual

    return solve


def graph_smoothed_discrepancy_posterior(
    observed_mean: np.ndarray,
    observed_variance: np.ndarray,
    observed: np.ndarray,
    laplacian,
    *,
    prior_strength: float,
    ridge: float = 1e-8,
    relative_tolerance: float = 1e-7,
    maximum_iterations: int = 2000,
    covariance_probes: int = 0,
    covariance_seed: int = 20260711,
    covariance_indices: np.ndarray | None = None,
) -> GraphDiscrepancyPosterior:
    """Condition a Laplacian discrepancy prior on robust endpoint posteriors.

    The variance-whitened negative log posterior is

    ``0.5 * ||W^(1/2) (b - m)||^2 + lambda * ||L b||^2``.

    Its covariance is ``v_ref * (W + 2 lambda L.T L)^-1`` up to the documented
    numerical ridge. Hutchinson probing estimates its marginal diagonal.
    """

    means = np.asarray(observed_mean, dtype=float)
    variances = np.asarray(observed_variance, dtype=float)
    observation_mask = np.asarray(observed, dtype=bool)
    if means.ndim != 2 or means.shape[1] != 3:
        raise ValueError("observed_mean must have shape (N, 3)")
    if variances.shape != (len(means),) or observation_mask.shape != (len(means),):
        raise ValueError("variance and observation mask must match observed_mean")
    if not np.any(observation_mask):
        raise ValueError("at least one graph observation is required")
    if not np.all(np.isfinite(means)) or not np.all(
        np.isfinite(variances[observation_mask])
    ):
        raise ValueError("observed posterior values must be finite")
    if np.any(variances[observation_mask] <= 0.0):
        raise ValueError("observed variances must be positive")
    node_count = int(laplacian.shape[0])
    if laplacian.shape != (node_count, node_count) or len(means) > node_count:
        raise ValueError("Laplacian does not cover observed nodes")
    if covariance_probes < 0:
        raise ValueError("covariance_probes must be nonnegative")
    selected_covariance_indices = (
        np.empty(0, dtype=np.int64)
        if covariance_indices is None
        else np.unique(np.asarray(covariance_indices, dtype=np.int64))
    )
    if np.any(selected_covariance_indices < 0) or np.any(
        selected_covariance_indices >= node_count
    ):
        raise ValueError("covariance_indices exceed the graph")
    if covariance_probes and len(selected_covariance_indices):
        raise ValueError("choose covariance probes or exact covariance indices")

    reference_variance = float(np.median(variances[observation_mask]))
    weights = np.zeros(node_count, dtype=float)
    observed_indices = np.flatnonzero(observation_mask)
    weights[observed_indices] = reference_variance / variances[observed_indices]
    full_mean = np.zeros((node_count, 3), dtype=float)
    full_mean[: len(means)] = means
    right_hand_side = weights[:, None] * full_mean
    solve = _precision_solver(
        laplacian,
        weights,
        prior_strength=prior_strength,
        ridge=ridge,
        relative_tolerance=relative_tolerance,
        maximum_iterations=maximum_iterations,
    )
    posterior_mean = np.empty_like(full_mean)
    iterations: list[int] = []
    residuals: list[float] = []
    for coordinate in range(3):
        solution, count, residual = solve(right_hand_side[:, coordinate])
        posterior_mean[:, coordinate] = solution
        iterations.append(count)
        residuals.append(residual)

    marginal_variance = None
    negative_fraction = None
    if len(selected_covariance_indices):
        diagonal = np.full(node_count, np.nan, dtype=float)
        for index in selected_covariance_indices:
            unit = np.zeros(node_count, dtype=float)
            unit[index] = 1.0
            inverse_unit, count, residual = solve(unit)
            diagonal[index] = reference_variance * inverse_unit[index]
            iterations.append(count)
            residuals.append(residual)
        selected_diagonal = diagonal[selected_covariance_indices]
        negative_fraction = float(np.mean(selected_diagonal < 0.0))
        diagonal[selected_covariance_indices] = np.maximum(selected_diagonal, 0.0)
        marginal_variance = diagonal
    elif covariance_probes:
        rng = np.random.default_rng(covariance_seed)
        diagonal = np.zeros(node_count, dtype=float)
        for _ in range(covariance_probes):
            probe = rng.choice(np.array([-1.0, 1.0]), size=node_count)
            inverse_probe, count, residual = solve(probe)
            diagonal += probe * inverse_probe
            iterations.append(count)
            residuals.append(residual)
        diagonal *= reference_variance / covariance_probes
        negative_fraction = float(np.mean(diagonal < 0.0))
        marginal_variance = np.maximum(diagonal, 0.0)

    return GraphDiscrepancyPosterior(
        mean=posterior_mean,
        marginal_variance=marginal_variance,
        observation_weight=weights,
        reference_variance=reference_variance,
        solve_iterations=tuple(iterations),
        solve_relative_residuals=tuple(residuals),
        covariance_negative_fraction=negative_fraction,
    )


def graph_discrepancy_diagnostics(
    correction: np.ndarray,
    springs: np.ndarray,
    laplacian,
) -> dict[str, float]:
    """Measure correction roughness on the reconstructed object graph."""

    values = np.asarray(correction, dtype=float)
    edges = np.asarray(springs, dtype=np.int64)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError("correction must have shape (N, 3)")
    if np.any(edges < 0) or np.any(edges >= len(values)):
        raise ValueError("spring endpoint exceeds correction")
    edge_delta = values[edges[:, 0]] - values[edges[:, 1]]
    laplacian_values = laplacian @ values
    return {
        "edge_difference_rms_m": float(
            np.sqrt(np.mean(np.sum(np.square(edge_delta), axis=1)))
        ),
        "laplacian_energy_m2_per_node": float(
            np.mean(np.sum(np.square(laplacian_values), axis=1))
        ),
    }
