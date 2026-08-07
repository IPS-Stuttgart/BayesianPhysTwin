"""Prospective SPD bias-aware state update.

This module is a versioned successor to the frozen implementation in
``bias_aware_belief.py``. Historical protocols and result artifacts continue to
identify and execute v1. New protocols may opt into v2 explicitly after freezing
this implementation and its numerical settings before target access.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final

import numpy as np

from .bias_aware_belief import (
    BiasAwareStateUpdateConfig,
    BiasAwareStateUpdateResult,
    _finite_matrix,
    _require,
    _student_t_weight,
    _subspace_overlap,
)
from .spd_system import (
    SPD_SYSTEM_SCHEMA,
    SPD_SYSTEM_VERSION,
    SPDConditionError,
    SPDSystem,
    SPDSystemError,
    SPDValidationError,
)

BIAS_AWARE_BELIEF_V2_SCHEMA: Final = "bayesian_phystwin.bias_aware_belief"
BIAS_AWARE_BELIEF_V2_VERSION: Final = 2
BIAS_AWARE_BELIEF_V2_IMPLEMENTATION: Final = "bias-aware-linear-student-t-spd-v2"
BIAS_AWARE_BELIEF_V2_CLAIM_BOUNDARY: Final = (
    "Prospective numerical implementation only. Results produced by v2 are not "
    "interchangeable with historical v1 evidence and require a separately frozen "
    "protocol, calibration, guard, and target-access record."
)


@dataclass(frozen=True)
class BiasAwareStateUpdateConfigV2(BiasAwareStateUpdateConfig):
    """Version-2 numerical settings in addition to the frozen model settings."""

    symmetry_absolute_tolerance: float = 1e-12
    symmetry_relative_tolerance: float = 1e-10
    solve_residual_tolerance: float = 1e-10
    inverse_residual_tolerance: float = 1e-9

    def __post_init__(self) -> None:
        super().__post_init__()
        values = (
            self.symmetry_absolute_tolerance,
            self.symmetry_relative_tolerance,
            self.solve_residual_tolerance,
            self.inverse_residual_tolerance,
        )
        _require(
            all(np.isfinite(value) and value >= 0.0 for value in values[:2]),
            "symmetry tolerances must be finite and nonnegative",
        )
        _require(
            all(np.isfinite(value) and value > 0.0 for value in values[2:]),
            "residual tolerances must be finite and positive",
        )


@dataclass(frozen=True)
class BiasAwareStateUpdateResultV2(BiasAwareStateUpdateResult):
    """Bias-aware result explicitly bound to the prospective v2 implementation."""

    implementation_schema: str = field(
        init=False,
        default=BIAS_AWARE_BELIEF_V2_SCHEMA,
    )
    implementation_version: int = field(
        init=False,
        default=BIAS_AWARE_BELIEF_V2_VERSION,
    )
    implementation_id: str = field(
        init=False,
        default=BIAS_AWARE_BELIEF_V2_IMPLEMENTATION,
    )
    numerical_backend_schema: str = field(
        init=False,
        default=SPD_SYSTEM_SCHEMA,
    )
    numerical_backend_version: int = field(
        init=False,
        default=SPD_SYSTEM_VERSION,
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        _require(
            self.diagnostics.get("implementation_id") == self.implementation_id,
            "v2 result diagnostics do not identify the v2 implementation",
        )
        _require(
            self.diagnostics.get("numerical_backend_schema")
            == self.numerical_backend_schema,
            "v2 result diagnostics do not identify the SPD backend",
        )


def _tag_diagnostics(diagnostics: dict[str, object]) -> dict[str, object]:
    tagged = dict(diagnostics)
    tagged.update(
        {
            "implementation_schema": BIAS_AWARE_BELIEF_V2_SCHEMA,
            "implementation_version": BIAS_AWARE_BELIEF_V2_VERSION,
            "implementation_id": BIAS_AWARE_BELIEF_V2_IMPLEMENTATION,
            "numerical_backend_schema": SPD_SYSTEM_SCHEMA,
            "numerical_backend_version": SPD_SYSTEM_VERSION,
            "implicit_jitter": False,
            "eigenvalue_clipping": False,
            "pseudoinverse_fallback": False,
            "claim_boundary": BIAS_AWARE_BELIEF_V2_CLAIM_BOUNDARY,
        }
    )
    return tagged


def _fallback_result_v2(
    state_count: int,
    shared_bias_count: int,
    view_count: int,
    prior_reliability: np.ndarray,
    reason: str,
    diagnostics: dict[str, object],
    anchor_count: int = 0,
) -> BiasAwareStateUpdateResultV2:
    dimension = state_count + shared_bias_count + view_count
    return BiasAwareStateUpdateResultV2(
        accepted=False,
        reason=reason,
        state_coefficients_m=np.zeros((state_count, 3), dtype=np.float64),
        shared_bias_coefficients_m=np.zeros(
            (shared_bias_count, 3),
            dtype=np.float64,
        ),
        camera_biases_m=np.zeros((view_count, 3), dtype=np.float64),
        posterior_covariance_m2=np.zeros((dimension, dimension), dtype=np.float64),
        prior_reliability=prior_reliability,
        robust_weights=np.zeros_like(prior_reliability),
        anchor_robust_weights=np.zeros(anchor_count, dtype=np.float64),
        diagnostics=_tag_diagnostics(diagnostics),
    )


def _factor_system(
    matrix: np.ndarray,
    *,
    name: str,
    config: BiasAwareStateUpdateConfigV2,
) -> SPDSystem:
    return SPDSystem.from_matrix(
        matrix,
        name=name,
        maximum_condition_number=config.maximum_condition_number,
        symmetry_absolute_tolerance=config.symmetry_absolute_tolerance,
        symmetry_relative_tolerance=config.symmetry_relative_tolerance,
        solve_residual_tolerance=config.solve_residual_tolerance,
    )


def _posterior_failure(
    error: SPDSystemError,
    *,
    diagnostics: dict[str, object],
) -> str:
    diagnostics["numerical_failure_type"] = type(error).__name__
    diagnostics["numerical_failure"] = str(error)
    if isinstance(error, SPDConditionError):
        return "ill-conditioned-posterior"
    if isinstance(error, SPDValidationError):
        return "non-positive-definite-posterior"
    return "unstable-posterior-solve"


def update_bias_aware_state_v2(
    camera_innovation_m: np.ndarray,
    camera_available: np.ndarray,
    state_basis: np.ndarray,
    shared_bias_basis: np.ndarray,
    *,
    prior_reliability: np.ndarray | None = None,
    observation_variance_m2: np.ndarray | None = None,
    anchor_innovation_m: np.ndarray | None = None,
    anchor_state_basis: np.ndarray | None = None,
    anchor_variance_m2: np.ndarray | None = None,
    state_prior_covariance_m2: np.ndarray | None = None,
    config: BiasAwareStateUpdateConfigV2 | None = None,
) -> BiasAwareStateUpdateResultV2:
    """Run the prospective robust update through the versioned SPD backend.

    The statistical model, reliability semantics, robust-weight update, and
    exact physical fallback match v1 on well-conditioned inputs. The numerical
    path differs deliberately: every admitted normal system is deterministically
    symmetrized, Cholesky-factorized once, solved through its triangular factors,
    and checked against explicit condition and residual limits. The final
    covariance is reconstructed only for the exported result contract.
    """

    if config is not None and not isinstance(config, BiasAwareStateUpdateConfigV2):
        raise TypeError("config must be a BiasAwareStateUpdateConfigV2")
    cfg = config or BiasAwareStateUpdateConfigV2()
    innovation = np.asarray(camera_innovation_m, dtype=np.float64)
    available = np.asarray(camera_available, dtype=bool)
    _require(
        innovation.ndim == 3 and innovation.shape[2] == 3,
        "camera innovation must have shape (V, N, 3)",
    )
    view_count, point_count, _ = innovation.shape
    _require(available.shape == (view_count, point_count), "camera mask changed")
    state = _finite_matrix(state_basis, "state basis")
    shared_bias = _finite_matrix(shared_bias_basis, "shared bias basis")
    _require(state.shape[0] == point_count, "state basis point count changed")
    _require(
        shared_bias.shape[0] == point_count,
        "shared bias basis point count changed",
    )
    state_count = state.shape[1]
    shared_bias_count = shared_bias.shape[1]
    _require(state_count >= 1, "state basis is empty")

    if prior_reliability is None:
        reliability = np.ones((view_count, point_count), dtype=np.float64)
    else:
        reliability = np.asarray(prior_reliability, dtype=np.float64).copy()
        _require(reliability.shape == available.shape, "prior reliability changed")
        _require(
            np.all(np.isfinite(reliability))
            and np.all((reliability >= 0.0) & (reliability <= 1.0)),
            "prior reliability must lie in [0, 1]",
        )
    reliability[~available] = 0.0
    finite_camera = np.all(np.isfinite(innovation), axis=2)
    reliability[~finite_camera] = 0.0
    usable = available & finite_camera & (reliability > 0.0)

    if observation_variance_m2 is None:
        camera_variance = np.full(
            available.shape,
            cfg.observation_std_m**2,
            dtype=np.float64,
        )
    else:
        camera_variance = np.asarray(
            observation_variance_m2,
            dtype=np.float64,
        ).copy()
        _require(camera_variance.shape == available.shape, "camera variance changed")
        _require(
            np.all(np.isfinite(camera_variance)) and np.all(camera_variance > 0.0),
            "camera variance must be positive",
        )

    active_views = np.flatnonzero(np.any(usable, axis=1))
    diagnostics: dict[str, object] = {
        "view_count": view_count,
        "active_view_count": len(active_views),
        "usable_camera_observation_count": int(np.sum(usable)),
        "state_mode_count": state_count,
        "shared_bias_mode_count": shared_bias_count,
        "correlation_treatment": (
            "effective samples within view and equal-weight covariance "
            "intersection across views"
        ),
        "prior_reliability_uses_innovation": False,
    }
    camera_rows = np.argwhere(usable)
    camera_design = np.zeros(
        (len(camera_rows), state_count + shared_bias_count + view_count),
        dtype=np.float64,
    )
    camera_target = np.empty((len(camera_rows), 3), dtype=np.float64)
    camera_row_variance = np.empty(len(camera_rows), dtype=np.float64)
    camera_base_weight = np.empty(len(camera_rows), dtype=np.float64)
    for row_index, (view_index, point_index) in enumerate(camera_rows):
        camera_design[row_index, :state_count] = state[point_index]
        camera_design[
            row_index,
            state_count : state_count + shared_bias_count,
        ] = shared_bias[point_index]
        camera_design[
            row_index,
            state_count + shared_bias_count + view_index,
        ] = 1.0
        camera_target[row_index] = innovation[view_index, point_index]
        camera_row_variance[row_index] = camera_variance[view_index, point_index]
        count = int(np.sum(usable[view_index]))
        within_view = min(cfg.effective_samples_per_view, float(count)) / count
        camera_base_weight[row_index] = (
            reliability[view_index, point_index] * within_view / len(active_views)
        )

    has_anchor = anchor_innovation_m is not None
    if has_anchor:
        _require(anchor_state_basis is not None, "anchor state basis is missing")
        anchor_target = np.asarray(anchor_innovation_m, dtype=np.float64)
        anchor_design_state = _finite_matrix(anchor_state_basis, "anchor state basis")
        _require(
            anchor_target.ndim == 2 and anchor_target.shape[1] == 3,
            "anchor innovation must have shape (A, 3)",
        )
        _require(
            anchor_design_state.shape == (len(anchor_target), state_count),
            "anchor state basis changed",
        )
        anchor_finite = np.all(np.isfinite(anchor_target), axis=1)
        anchor_target = anchor_target[anchor_finite]
        anchor_design_state = anchor_design_state[anchor_finite]
        anchor_design = np.zeros(
            (len(anchor_target), camera_design.shape[1]),
            dtype=np.float64,
        )
        anchor_design[:, :state_count] = anchor_design_state
        if anchor_variance_m2 is None:
            anchor_variance = np.full(
                len(anchor_target),
                cfg.anchor_std_m**2,
                dtype=np.float64,
            )
        else:
            supplied_anchor_variance = np.asarray(
                anchor_variance_m2,
                dtype=np.float64,
            )
            _require(
                supplied_anchor_variance.shape == (len(anchor_finite),),
                "anchor variance changed",
            )
            anchor_variance = supplied_anchor_variance[anchor_finite]
            _require(
                np.all(np.isfinite(anchor_variance)) and np.all(anchor_variance > 0.0),
                "anchor variance must be positive",
            )
    else:
        _require(anchor_state_basis is None, "anchor innovation is missing")
        _require(anchor_variance_m2 is None, "anchor innovation is missing")
        anchor_target = np.zeros((0, 3), dtype=np.float64)
        anchor_design = np.zeros((0, camera_design.shape[1]), dtype=np.float64)
        anchor_variance = np.zeros(0, dtype=np.float64)

    if not len(camera_rows) and not len(anchor_target):
        return _fallback_result_v2(
            state_count,
            shared_bias_count,
            view_count,
            reliability,
            "no-observation-support",
            diagnostics,
        )

    if state_prior_covariance_m2 is None:
        state_precision = np.eye(state_count) / cfg.state_prior_std_m**2
        diagnostics["state_prior_numerical_path"] = "diagonal-analytic"
    else:
        supplied = np.asarray(state_prior_covariance_m2, dtype=np.float64)
        _require(
            supplied.shape == (state_count, state_count),
            "state prior covariance changed",
        )
        try:
            state_prior_system = _factor_system(
                supplied,
                name="state prior covariance",
                config=cfg,
            )
            state_information_root = state_prior_system.whiten(
                np.eye(state_count, dtype=np.float64)
            )
            state_precision = state_information_root.T @ state_information_root
            state_precision = 0.5 * (state_precision + state_precision.T)
        except SPDSystemError as error:
            diagnostics["numerical_failure_type"] = type(error).__name__
            diagnostics["numerical_failure"] = str(error)
            return _fallback_result_v2(
                state_count,
                shared_bias_count,
                view_count,
                reliability,
                "invalid-state-prior-covariance",
                diagnostics,
                anchor_count=len(anchor_target),
            )
        diagnostics["state_prior_numerical_path"] = "cholesky"
        diagnostics["state_prior_spd"] = state_prior_system.diagnostics()

    dimension = camera_design.shape[1]
    prior_precision = np.zeros((dimension, dimension), dtype=np.float64)
    prior_precision[:state_count, :state_count] = state_precision
    shared_slice = slice(state_count, state_count + shared_bias_count)
    prior_precision[shared_slice, shared_slice] = (
        np.eye(shared_bias_count) / cfg.shared_bias_prior_std_m**2
    )
    camera_slice = slice(state_count + shared_bias_count, dimension)
    camera_bias_precision = np.full(
        view_count,
        1.0 / cfg.camera_bias_prior_std_m**2,
        dtype=np.float64,
    )
    if len(active_views):
        camera_bias_precision[active_views] /= len(active_views)
    prior_precision[camera_slice, camera_slice] = np.diag(camera_bias_precision)
    diagnostics["per_camera_bias_prior_ci_scaled"] = True

    camera_bias_design = camera_design[:, state_count:]
    camera_overlap_weight = np.sqrt(camera_base_weight / camera_row_variance)[:, None]
    overlap = _subspace_overlap(
        camera_overlap_weight * camera_design[:, :state_count],
        camera_overlap_weight * camera_bias_design,
    )
    diagnostics["state_bias_subspace_cosine"] = overlap
    diagnostics["independent_anchor_count"] = len(anchor_target)
    if (
        cfg.reject_unanchored_ambiguity
        and not len(anchor_target)
        and overlap >= cfg.ambiguous_subspace_cosine
    ):
        return _fallback_result_v2(
            state_count,
            shared_bias_count,
            view_count,
            reliability,
            "unanchored-common-mode-ambiguity",
            diagnostics,
            anchor_count=len(anchor_target),
        )

    solution = np.zeros((dimension, 3), dtype=np.float64)
    camera_robust = np.ones(len(camera_rows), dtype=np.float64)
    anchor_robust = np.ones(len(anchor_target), dtype=np.float64)

    def posterior_system() -> tuple[np.ndarray, np.ndarray]:
        camera_precision_weight = (
            camera_base_weight * camera_robust / camera_row_variance
        )
        posterior_normal = prior_precision + camera_design.T @ (
            camera_precision_weight[:, None] * camera_design
        )
        posterior_right = camera_design.T @ (
            camera_precision_weight[:, None] * camera_target
        )
        if len(anchor_target):
            anchor_precision_weight = anchor_robust / anchor_variance
            posterior_normal += anchor_design.T @ (
                anchor_precision_weight[:, None] * anchor_design
            )
            posterior_right += anchor_design.T @ (
                anchor_precision_weight[:, None] * anchor_target
            )
        return posterior_normal, posterior_right

    for iteration in range(cfg.maximum_iterations):
        previous = solution.copy()
        normal, right = posterior_system()
        try:
            system = _factor_system(
                normal,
                name=f"posterior normal iteration {iteration + 1}",
                config=cfg,
            )
            solution = system.solve(right)
        except SPDSystemError as error:
            reason = _posterior_failure(error, diagnostics=diagnostics)
            return _fallback_result_v2(
                state_count,
                shared_bias_count,
                view_count,
                reliability,
                reason,
                diagnostics,
                anchor_count=len(anchor_target),
            )
        camera_residual = camera_target - camera_design @ solution
        camera_robust = _student_t_weight(
            camera_residual,
            camera_row_variance,
            cfg.degrees_of_freedom,
            cfg.minimum_robust_weight,
        )
        if len(anchor_target):
            anchor_residual = anchor_target - anchor_design @ solution
            anchor_robust = _student_t_weight(
                anchor_residual,
                anchor_variance,
                cfg.degrees_of_freedom,
                cfg.minimum_robust_weight,
            )
        if np.linalg.norm(solution - previous) <= cfg.convergence_tolerance:
            break

    normal, right = posterior_system()
    try:
        final_system = _factor_system(
            normal,
            name="final posterior normal",
            config=cfg,
        )
        solution = final_system.solve(right)
    except SPDSystemError as error:
        reason = _posterior_failure(error, diagnostics=diagnostics)
        return _fallback_result_v2(
            state_count,
            shared_bias_count,
            view_count,
            reliability,
            reason,
            diagnostics,
            anchor_count=len(anchor_target),
        )

    state_coefficients = solution[:state_count]
    state_update = state @ state_coefficients
    maximum_update = float(np.max(np.linalg.norm(state_update, axis=1)))
    final_solve_residual = final_system.relative_residual(right, solution)
    diagnostics.update(
        {
            "iterations": iteration + 1,
            "condition_number": final_system.condition_number,
            "final_spd_system": final_system.diagnostics(),
            "final_solve_relative_residual": final_solve_residual,
            "effective_camera_information_mass": float(np.sum(camera_base_weight)),
            "minimum_camera_robust_weight": (
                float(np.min(camera_robust)) if len(camera_robust) else 1.0
            ),
            "downweighted_camera_fraction": (
                float(np.mean(camera_robust < 1.0)) if len(camera_robust) else 0.0
            ),
            "maximum_state_update_m": maximum_update,
        }
    )
    if not np.all(np.isfinite(solution)) or maximum_update > cfg.maximum_state_update_m:
        return _fallback_result_v2(
            state_count,
            shared_bias_count,
            view_count,
            reliability,
            "implausible-state-update",
            diagnostics,
            anchor_count=len(anchor_target),
        )

    try:
        covariance = final_system.reconstruct_inverse()
        np.linalg.cholesky(covariance)
    except (SPDSystemError, np.linalg.LinAlgError) as error:
        diagnostics["numerical_failure_type"] = type(error).__name__
        diagnostics["numerical_failure"] = str(error)
        return _fallback_result_v2(
            state_count,
            shared_bias_count,
            view_count,
            reliability,
            "unstable-posterior-covariance",
            diagnostics,
            anchor_count=len(anchor_target),
        )
    identity = np.eye(dimension, dtype=np.float64)
    inverse_residual = normal @ covariance - identity
    inverse_relative_residual = float(
        np.linalg.norm(inverse_residual, ord=np.inf)
        / max(
            1.0,
            float(np.linalg.norm(normal, ord=np.inf))
            * float(np.linalg.norm(covariance, ord=np.inf)),
        )
    )
    diagnostics["inverse_relative_residual"] = inverse_relative_residual
    diagnostics["exported_covariance_positive_definite"] = True
    if (
        not np.isfinite(inverse_relative_residual)
        or inverse_relative_residual > cfg.inverse_residual_tolerance
    ):
        return _fallback_result_v2(
            state_count,
            shared_bias_count,
            view_count,
            reliability,
            "unstable-posterior-covariance",
            diagnostics,
            anchor_count=len(anchor_target),
        )

    bias_indices = np.arange(state_count, dimension)
    state_indices = np.arange(state_count)
    if len(bias_indices):
        cross = covariance[np.ix_(state_indices, bias_indices)]
        scale = np.sqrt(
            covariance[state_indices, state_indices][:, None]
            * covariance[bias_indices, bias_indices][None, :]
        )
        maximum_cross_correlation = float(
            np.max(np.abs(cross / np.maximum(scale, 1e-15)))
        )
    else:
        maximum_cross_correlation = 0.0
    diagnostics["maximum_state_bias_posterior_correlation"] = maximum_cross_correlation

    robust_map = np.zeros((view_count, point_count), dtype=np.float64)
    robust_map[camera_rows[:, 0], camera_rows[:, 1]] = camera_robust
    return BiasAwareStateUpdateResultV2(
        accepted=True,
        reason="accepted",
        state_coefficients_m=state_coefficients,
        shared_bias_coefficients_m=solution[shared_slice],
        camera_biases_m=solution[camera_slice],
        posterior_covariance_m2=covariance,
        prior_reliability=reliability,
        robust_weights=robust_map,
        anchor_robust_weights=anchor_robust,
        diagnostics=_tag_diagnostics(diagnostics),
    )


__all__ = [
    "BIAS_AWARE_BELIEF_V2_CLAIM_BOUNDARY",
    "BIAS_AWARE_BELIEF_V2_IMPLEMENTATION",
    "BIAS_AWARE_BELIEF_V2_SCHEMA",
    "BIAS_AWARE_BELIEF_V2_VERSION",
    "BiasAwareStateUpdateConfigV2",
    "BiasAwareStateUpdateResultV2",
    "update_bias_aware_state_v2",
]
