"""Bias-aware belief updates over action-propagated simulator responses.

The state design supplied here is a finite-difference response from the
physical simulator, not a free observation-space basis.  A persistent graph
bias is inferred alongside it so a coherent offset is not automatically
laundered into the physical state.  Observation reliability is fixed before
the innovation is evaluated; the innovation enters once through the robust
Student-t likelihood below.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _readonly(value: np.ndarray, *, dtype: object = np.float64) -> np.ndarray:
    result = np.asarray(value, dtype=dtype).copy()
    _require(np.all(np.isfinite(result)), "array contains non-finite values")
    result.setflags(write=False)
    return result


def _positive_definite_cholesky(value: np.ndarray, name: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    _require(
        matrix.ndim == 2 and matrix.shape[0] == matrix.shape[1],
        f"{name} must be square",
    )
    _require(np.all(np.isfinite(matrix)), f"{name} contains non-finite values")
    _require(np.allclose(matrix, matrix.T), f"{name} must be symmetric")
    try:
        return np.linalg.cholesky(matrix)
    except np.linalg.LinAlgError as error:
        raise ValueError(f"{name} must be positive definite") from error


def _cholesky_solve(cholesky: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.linalg.solve(cholesky.T, np.linalg.solve(cholesky, right))


def _positive_definite_inverse(value: np.ndarray, name: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    cholesky = _positive_definite_cholesky(matrix, name)
    inverse = _cholesky_solve(cholesky, np.eye(len(matrix), dtype=np.float64))
    return 0.5 * (inverse + inverse.T)


def _orthonormal_column_space(value: np.ndarray) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if not matrix.size or matrix.shape[1] == 0:
        return np.zeros((len(matrix), 0), dtype=np.float64)
    left, singular_values, _ = np.linalg.svd(matrix, full_matrices=False)
    if not len(singular_values) or singular_values[0] == 0.0:
        return np.zeros((len(matrix), 0), dtype=np.float64)
    tolerance = max(matrix.shape) * np.finfo(np.float64).eps * singular_values[0]
    return left[:, singular_values > tolerance]


def _subspace_overlap(first: np.ndarray, second: np.ndarray) -> float:
    first_space = _orthonormal_column_space(first)
    second_space = _orthonormal_column_space(second)
    if first_space.shape[1] == 0 or second_space.shape[1] == 0:
        return 0.0
    return float(np.linalg.svd(first_space.T @ second_space, compute_uv=False)[0])


@dataclass(frozen=True)
class PropagatedStateBeliefConfig:
    """Priors and numerical controls for one propagated-state update."""

    observation_std_m: float = 0.005
    state_weight_prior_std: float = 1.0
    shared_bias_prior_std_m: float = 0.020
    effective_samples_per_frame: float = 64.0
    effective_frame_count: float = 4.0
    degrees_of_freedom: float = 4.0
    minimum_robust_weight: float = 0.02
    maximum_iterations: int = 8
    convergence_tolerance: float = 1e-9
    maximum_condition_number: float = 1e12
    ambiguous_subspace_cosine: float = 0.999
    reject_unidentifiable_state: bool = True

    def __post_init__(self) -> None:
        positive = (
            self.observation_std_m,
            self.state_weight_prior_std,
            self.shared_bias_prior_std_m,
            self.effective_samples_per_frame,
            self.effective_frame_count,
            self.degrees_of_freedom,
            self.convergence_tolerance,
            self.maximum_condition_number,
        )
        _require(
            all(np.isfinite(value) and value > 0.0 for value in positive),
            "propagated-state scales must be positive",
        )
        _require(
            0.0 < self.minimum_robust_weight <= 1.0,
            "minimum robust weight must lie in (0, 1]",
        )
        _require(self.maximum_iterations >= 1, "maximum iterations must be positive")
        _require(
            0.0 <= self.ambiguous_subspace_cosine <= 1.0,
            "ambiguity cosine must lie in [0, 1]",
        )


@dataclass(frozen=True)
class PropagatedStateBeliefResult:
    """Posterior state/bias coefficients and the exact-fallback decision."""

    accepted: bool
    reason: str
    state_weights: np.ndarray
    shared_bias_coefficients_m: np.ndarray
    posterior_covariance: np.ndarray
    prior_reliability: np.ndarray
    robust_weights: np.ndarray
    diagnostics: dict[str, object]

    def __post_init__(self) -> None:
        state = _readonly(self.state_weights)
        bias = _readonly(self.shared_bias_coefficients_m)
        covariance = _readonly(self.posterior_covariance)
        reliability = _readonly(self.prior_reliability)
        robust = _readonly(self.robust_weights)
        _require(state.ndim == 1, "state weights must be a vector")
        _require(bias.ndim == 2 and bias.shape[1] == 3, "bias must have shape (B, 3)")
        dimension = len(state) + 3 * len(bias)
        _require(
            covariance.shape == (dimension, dimension),
            "posterior covariance shape changed",
        )
        _require(reliability.ndim == 2, "prior reliability must have shape (T, N)")
        _require(robust.shape == reliability.shape, "robust weight shape changed")
        object.__setattr__(self, "state_weights", state)
        object.__setattr__(self, "shared_bias_coefficients_m", bias)
        object.__setattr__(self, "posterior_covariance", covariance)
        object.__setattr__(self, "prior_reliability", reliability)
        object.__setattr__(self, "robust_weights", robust)
        object.__setattr__(self, "diagnostics", dict(self.diagnostics))


def _fallback(
    state_count: int,
    bias_count: int,
    reliability: np.ndarray,
    reason: str,
    diagnostics: dict[str, object],
) -> PropagatedStateBeliefResult:
    dimension = state_count + 3 * bias_count
    return PropagatedStateBeliefResult(
        accepted=False,
        reason=reason,
        state_weights=np.zeros(state_count, dtype=np.float64),
        shared_bias_coefficients_m=np.zeros((bias_count, 3), dtype=np.float64),
        posterior_covariance=np.zeros((dimension, dimension), dtype=np.float64),
        prior_reliability=reliability,
        robust_weights=np.zeros_like(reliability),
        diagnostics=diagnostics,
    )


def _observation_variance(
    supplied: np.ndarray | None,
    shape: tuple[int, int],
    default_std_m: float,
) -> np.ndarray:
    if supplied is None:
        return np.full(shape, default_std_m**2, dtype=np.float64)
    variance = np.asarray(supplied, dtype=np.float64)
    if variance.shape == (*shape, 3):
        variance = np.mean(variance, axis=2)
    _require(variance.shape == shape, "observation variance shape changed")
    _require(
        np.all(np.isfinite(variance)) and np.all(variance > 0.0),
        "observation variance must be positive",
    )
    return variance.copy()


def _state_prior(
    state_count: int,
    mean: np.ndarray | None,
    covariance: np.ndarray | None,
    default_std: float,
) -> tuple[np.ndarray, np.ndarray]:
    if mean is None:
        prior_mean = np.zeros(state_count, dtype=np.float64)
    else:
        prior_mean = np.asarray(mean, dtype=np.float64)
        _require(prior_mean.shape == (state_count,), "state prior mean shape changed")
        _require(np.all(np.isfinite(prior_mean)), "state prior mean is non-finite")
    if covariance is None:
        precision = np.eye(state_count, dtype=np.float64) / default_std**2
    else:
        supplied = np.asarray(covariance, dtype=np.float64)
        _require(
            supplied.shape == (state_count, state_count),
            "state prior covariance shape changed",
        )
        precision = _positive_definite_inverse(supplied, "state prior covariance")
    return prior_mean.copy(), precision


def infer_propagated_state_belief(
    innovation_m: np.ndarray,
    available: np.ndarray,
    state_response_at_step_m: np.ndarray,
    shared_bias_basis: np.ndarray,
    *,
    prior_reliability: np.ndarray | None = None,
    observation_variance_m2: np.ndarray | None = None,
    state_prior_mean: np.ndarray | None = None,
    state_prior_covariance: np.ndarray | None = None,
    config: PropagatedStateBeliefConfig | None = None,
) -> PropagatedStateBeliefResult:
    """Infer state-response and persistent-bias coefficients from a prefix.

    ``state_response_at_step_m[t, n, xyz, k]`` is the displacement produced by
    one predeclared simulator perturbation step for state parameter ``k``.
    The returned state weights are therefore dimensionless multipliers of
    those steps.  The state response may rotate or leak between spatial modes;
    no coordinate-separable linearization is assumed.
    """

    cfg = config or PropagatedStateBeliefConfig()
    innovation = np.asarray(innovation_m, dtype=np.float64)
    mask = np.asarray(available, dtype=bool)
    response = np.asarray(state_response_at_step_m, dtype=np.float64)
    bias_basis = np.asarray(shared_bias_basis, dtype=np.float64)
    _require(
        innovation.ndim == 3 and innovation.shape[2] == 3,
        "innovation must have shape (T, N, 3)",
    )
    frame_count, point_count, _ = innovation.shape
    _require(mask.shape == (frame_count, point_count), "availability shape changed")
    _require(
        response.ndim == 4 and response.shape[:3] == innovation.shape,
        "state response must have shape (T, N, 3, K)",
    )
    _require(
        bias_basis.ndim == 2 and bias_basis.shape[0] == point_count,
        "shared bias basis must have shape (N, B)",
    )
    _require(
        np.all(np.isfinite(response)) and np.all(np.isfinite(bias_basis)),
        "response and bias basis must be finite",
    )
    state_count = response.shape[3]
    bias_count = bias_basis.shape[1]
    _require(state_count >= 1, "state response is empty")

    if prior_reliability is None:
        reliability = np.ones(mask.shape, dtype=np.float64)
    else:
        reliability = np.asarray(prior_reliability, dtype=np.float64).copy()
        _require(reliability.shape == mask.shape, "prior reliability shape changed")
        _require(
            np.all(np.isfinite(reliability))
            and np.all((reliability >= 0.0) & (reliability <= 1.0)),
            "prior reliability must lie in [0, 1]",
        )
    finite = np.all(np.isfinite(innovation), axis=2)
    usable = mask & finite & (reliability > 0.0)
    reliability[~usable] = 0.0
    variance = _observation_variance(
        observation_variance_m2,
        mask.shape,
        cfg.observation_std_m,
    )
    active_frames = np.flatnonzero(np.any(usable, axis=1))
    diagnostics: dict[str, object] = {
        "frame_count": frame_count,
        "active_frame_count": len(active_frames),
        "usable_node_frame_count": int(np.sum(usable)),
        "state_parameter_count": state_count,
        "shared_bias_mode_count": bias_count,
        "prior_reliability_uses_innovation": False,
        "innovation_likelihood_count": 1,
        "correlation_treatment": (
            "effective node samples within each frame and capped effective "
            "frame count across the prefix"
        ),
    }
    if not len(active_frames):
        return _fallback(
            state_count, bias_count, reliability, "no-observation-support", diagnostics
        )

    node_rows = np.argwhere(usable)
    dimension = state_count + 3 * bias_count
    design = np.zeros((3 * len(node_rows), dimension), dtype=np.float64)
    target = np.zeros(3 * len(node_rows), dtype=np.float64)
    row_variance = np.zeros(3 * len(node_rows), dtype=np.float64)
    node_base_weight = np.zeros(len(node_rows), dtype=np.float64)
    frame_factor = min(cfg.effective_frame_count, float(len(active_frames))) / len(
        active_frames
    )
    for node_row, (frame, point) in enumerate(node_rows):
        row_slice = slice(3 * node_row, 3 * node_row + 3)
        design[row_slice, :state_count] = response[frame, point]
        for coordinate in range(3):
            bias_start = state_count + coordinate * bias_count
            design[3 * node_row + coordinate, bias_start : bias_start + bias_count] = (
                bias_basis[point]
            )
        target[row_slice] = innovation[frame, point]
        row_variance[row_slice] = variance[frame, point]
        count = int(np.sum(usable[frame]))
        within_frame = min(cfg.effective_samples_per_frame, float(count)) / count
        node_base_weight[node_row] = (
            reliability[frame, point] * within_frame * frame_factor
        )

    coordinate_base_weight = np.repeat(node_base_weight, 3)
    weighted_scale = np.sqrt(coordinate_base_weight / row_variance)[:, None]
    overlap = _subspace_overlap(
        weighted_scale * design[:, :state_count],
        weighted_scale * design[:, state_count:],
    )
    diagnostics["state_bias_subspace_cosine"] = overlap
    if cfg.reject_unidentifiable_state and overlap >= cfg.ambiguous_subspace_cosine:
        return _fallback(
            state_count,
            bias_count,
            reliability,
            "state-response-confounded-with-persistent-bias",
            diagnostics,
        )

    prior_mean, state_precision = _state_prior(
        state_count,
        state_prior_mean,
        state_prior_covariance,
        cfg.state_weight_prior_std,
    )
    prior_precision = np.zeros((dimension, dimension), dtype=np.float64)
    prior_precision[:state_count, :state_count] = state_precision
    if bias_count:
        prior_precision[state_count:, state_count:] = np.eye(3 * bias_count) / (
            cfg.shared_bias_prior_std_m**2
        )
    prior_right = np.zeros(dimension, dtype=np.float64)
    prior_right[:state_count] = state_precision @ prior_mean

    def posterior_system(node_robust: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        coordinate_weight = (
            coordinate_base_weight * np.repeat(node_robust, 3) / row_variance
        )
        normal = prior_precision + design.T @ (coordinate_weight[:, None] * design)
        normal = 0.5 * (normal + normal.T)
        right = prior_right + design.T @ (coordinate_weight * target)
        return normal, right

    def solve_posterior(
        normal: np.ndarray,
        right: np.ndarray,
    ) -> tuple[np.ndarray | None, np.ndarray | None, float, str | None]:
        condition_number = float(np.linalg.cond(normal))
        if not np.isfinite(condition_number) or condition_number > (
            cfg.maximum_condition_number
        ):
            return None, None, condition_number, "ill-conditioned-posterior"
        try:
            cholesky = np.linalg.cholesky(normal)
        except np.linalg.LinAlgError:
            return None, None, condition_number, "singular-posterior"
        return _cholesky_solve(cholesky, right), cholesky, condition_number, None

    solution = np.zeros(dimension, dtype=np.float64)
    solution[:state_count] = prior_mean
    robust = np.ones(len(node_rows), dtype=np.float64)
    normal = prior_precision.copy()
    for iteration in range(cfg.maximum_iterations):
        iterations = iteration + 1
        previous = solution.copy()
        normal, right = posterior_system(robust)
        solved, _, condition_number, failure = solve_posterior(normal, right)
        if failure is not None or solved is None:
            diagnostics["condition_number"] = condition_number
            return _fallback(
                state_count,
                bias_count,
                reliability,
                failure or "singular-posterior",
                diagnostics,
            )
        solution = solved
        residual = (target - design @ solution).reshape(-1, 3)
        squared_radius = np.sum(np.square(residual), axis=1) / np.asarray(
            [variance[frame, point] for frame, point in node_rows]
        )
        robust = np.clip(
            (cfg.degrees_of_freedom + 3.0) / (cfg.degrees_of_freedom + squared_radius),
            cfg.minimum_robust_weight,
            1.0,
        )
        if (
            np.max(np.abs(solution - previous), initial=0.0)
            <= cfg.convergence_tolerance
        ):
            break

    normal, right = posterior_system(robust)
    solved, cholesky, condition_number, failure = solve_posterior(normal, right)
    if failure is not None or solved is None or cholesky is None:
        diagnostics["condition_number"] = condition_number
        return _fallback(
            state_count,
            bias_count,
            reliability,
            failure or "singular-posterior",
            diagnostics,
        )
    solution = solved
    posterior_covariance = _cholesky_solve(
        cholesky,
        np.eye(dimension, dtype=np.float64),
    )
    posterior_covariance = 0.5 * (posterior_covariance + posterior_covariance.T)
    predicted = (design @ solution).reshape(-1, 3)
    observed = target.reshape(-1, 3)
    diagnostics.update(
        {
            "condition_number": condition_number,
            "iterations": iterations,
            "prefix_rmse_m": float(np.sqrt(np.mean(np.square(predicted - observed)))),
            "state_weight_norm": float(np.linalg.norm(solution[:state_count])),
            "shared_bias_rms_m": float(
                np.sqrt(np.mean(np.square(solution[state_count:])))
                if bias_count
                else 0.0
            ),
            "mean_robust_weight": float(np.mean(robust)),
            "posterior_solver": "cholesky",
            "final_system_uses_returned_robust_weights": True,
            "posterior_covariance_is_raw_not_calibrated": True,
        }
    )
    robust_matrix = np.zeros(mask.shape, dtype=np.float64)
    for weight, (frame, point) in zip(robust, node_rows, strict=True):
        robust_matrix[frame, point] = weight
    bias_coefficients = solution[state_count:].reshape(3, bias_count).T
    return PropagatedStateBeliefResult(
        accepted=True,
        reason="accepted",
        state_weights=solution[:state_count],
        shared_bias_coefficients_m=bias_coefficients,
        posterior_covariance=posterior_covariance,
        prior_reliability=reliability,
        robust_weights=robust_matrix,
        diagnostics=diagnostics,
    )


def propagated_state_readout(
    state_response_at_step_m: np.ndarray,
    state_weights: np.ndarray,
    shared_bias_basis: np.ndarray,
    shared_bias_coefficients_m: np.ndarray,
) -> np.ndarray:
    """Evaluate the inferred propagated-state plus persistent-bias readout."""

    response = np.asarray(state_response_at_step_m, dtype=np.float64)
    weights = np.asarray(state_weights, dtype=np.float64)
    basis = np.asarray(shared_bias_basis, dtype=np.float64)
    coefficients = np.asarray(shared_bias_coefficients_m, dtype=np.float64)
    _require(response.ndim == 4 and response.shape[2] == 3, "response shape changed")
    _require(weights.shape == (response.shape[3],), "state weight shape changed")
    _require(
        basis.shape == (response.shape[1], coefficients.shape[0]),
        "bias basis shape changed",
    )
    _require(
        coefficients.ndim == 2 and coefficients.shape[1] == 3,
        "bias coefficients changed",
    )
    _require(
        np.all(np.isfinite(response))
        and np.all(np.isfinite(weights))
        and np.all(np.isfinite(basis))
        and np.all(np.isfinite(coefficients)),
        "propagated readout inputs must be finite",
    )
    state = np.einsum("tnck,k->tnc", response, weights)
    bias = basis @ coefficients
    return state + bias[None]


__all__ = [
    "PropagatedStateBeliefConfig",
    "PropagatedStateBeliefResult",
    "infer_propagated_state_belief",
    "propagated_state_readout",
]
