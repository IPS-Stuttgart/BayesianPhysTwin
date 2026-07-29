"""Sparse-witness guard for a dense causal PhysTwin discrepancy field.

The dense camera block proposes one graph-smooth displacement correction.  Its
rows are treated as a single correlated block: reliability changes their
relative weighting, but the block receives a fixed total information mass.
Sparse material tracks from a different tracker estimate a relative gauge
bias.  Separate prefix identities then decide whether the field may be used.

The physical-state innovation is never used to construct prior perception
reliability.  It enters once, through robust graph-field fitting.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from .dynamic_discrepancy import scale_coefficients_to_field_limit


def _require(condition: bool | np.bool_, message: str) -> None:
    if not bool(condition):
        raise ValueError(message)


@dataclass(frozen=True)
class CrossProviderGuardedFieldConfig:
    """Frozen choices for one displacement-anchored source smoke."""

    source_frame_start: int = 68
    source_frame_end_exclusive: int = 88
    apply_frame_start: int = 88
    validation_frame_end_exclusive: int = 121
    minimum_camera_count: int = 2
    maximum_reprojection_error_px: float = 3.0
    minimum_quality_probability: float = 0.1
    minimum_dense_support_fraction: float = 0.10
    minimum_provider_count: int = 3
    bias_history_frames: int = 5
    robust_scale_m: float = 0.010
    robust_iterations: int = 4
    projection_ridge: float = 1e-5
    maximum_correction_m: float = 0.010
    minimum_validation_improvement_fraction: float = 0.05
    minimum_validation_improvement_m: float = 0.00025

    def __post_init__(self) -> None:
        _require(
            self.source_frame_end_exclusive > self.source_frame_start + 1,
            "source window is too short",
        )
        _require(
            self.apply_frame_start == self.source_frame_end_exclusive,
            "field must branch immediately after the source window",
        )
        _require(
            self.validation_frame_end_exclusive > self.apply_frame_start,
            "validation interval is empty",
        )
        _require(self.minimum_camera_count >= 2, "at least two cameras are required")
        positive = (
            self.maximum_reprojection_error_px,
            self.minimum_quality_probability,
            self.minimum_dense_support_fraction,
            self.robust_scale_m,
            self.projection_ridge,
            self.maximum_correction_m,
            self.minimum_validation_improvement_m,
        )
        _require(all(value > 0.0 for value in positive), "scales must be positive")
        _require(self.minimum_provider_count >= 2, "provider support is too small")
        _require(self.bias_history_frames >= 2, "bias history is too short")
        _require(self.robust_iterations >= 1, "robust iterations must be positive")
        _require(
            0.0 < self.minimum_dense_support_fraction <= 1.0,
            "dense support fraction must lie in (0, 1]",
        )
        _require(
            0.0 <= self.minimum_validation_improvement_fraction < 1.0,
            "validation improvement must lie in [0, 1)",
        )

    @property
    def source_frame_count(self) -> int:
        return self.source_frame_end_exclusive - self.source_frame_start


def residual_independent_dense_reliability(
    quality_probability: np.ndarray,
    reprojection_error_px: np.ndarray,
    camera_count: np.ndarray,
    valid: np.ndarray,
    *,
    config: CrossProviderGuardedFieldConfig,
) -> np.ndarray:
    """Return camera-only prior reliability for one source endpoint pair."""

    quality = np.asarray(quality_probability, dtype=np.float64)
    reprojection = np.asarray(reprojection_error_px, dtype=np.float64)
    cameras = np.asarray(camera_count)
    mask = np.asarray(valid, dtype=bool)
    _require(
        quality.shape == reprojection.shape == cameras.shape == mask.shape,
        "dense cue shapes differ",
    )
    accepted = (
        mask
        & np.isfinite(quality)
        & np.isfinite(reprojection)
        & (quality >= config.minimum_quality_probability)
        & (reprojection <= config.maximum_reprojection_error_px)
        & (cameras >= config.minimum_camera_count)
    )
    reliability = np.zeros_like(quality)
    reliability[accepted] = quality[accepted] * np.exp(
        -0.5
        * np.square(
            reprojection[accepted] / config.maximum_reprojection_error_px
        )
    )
    return reliability


def estimate_relative_provider_bias(
    provider_trajectory_m: np.ndarray,
    provider_support: np.ndarray,
    provider_code: np.ndarray,
    dense_local_trajectory_m: np.ndarray,
    dense_local_available: np.ndarray,
    *,
    history_frames: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Estimate tracker-relative displacement bias from prior common rows.

    Only rows supplied directly by the primary provider (code 1) are used.
    Both trajectories are differenced from their first common row per identity,
    so an absolute world-frame offset cannot masquerade as displacement bias.
    """

    provider = np.asarray(provider_trajectory_m, dtype=np.float64)
    support = np.asarray(provider_support, dtype=bool)
    code = np.asarray(provider_code)
    dense = np.asarray(dense_local_trajectory_m, dtype=np.float64)
    dense_available = np.asarray(dense_local_available, dtype=bool)
    _require(
        provider.ndim == 3
        and provider.shape[2] == 3
        and support.shape == code.shape == dense_available.shape == provider.shape[:2]
        and dense.shape == provider.shape,
        "provider history shapes differ",
    )
    differences: list[np.ndarray] = []
    identity_count = 0
    for identity in range(provider.shape[1]):
        common = (
            support[:, identity]
            & (code[:, identity] == 1)
            & dense_available[:, identity]
            & np.all(np.isfinite(provider[:, identity]), axis=1)
            & np.all(np.isfinite(dense[:, identity]), axis=1)
        )
        rows = np.flatnonzero(common)
        if len(rows) < 2:
            continue
        anchor = int(rows[0])
        selected = rows[-history_frames:]
        provider_displacement = provider[selected, identity] - provider[anchor, identity]
        dense_displacement = dense[selected, identity] - dense[anchor, identity]
        differences.extend(dense_displacement - provider_displacement)
        identity_count += 1
    if not differences:
        bias = np.zeros(3, dtype=np.float64)
        return bias, {
            "identity_count": 0,
            "row_count": 0,
            "bias_m": bias.tolist(),
            "radial_rmse_m": None,
            "estimate_available": False,
            "absolute_gauge_used": False,
        }
    stacked = np.asarray(differences, dtype=np.float64)
    bias = np.median(stacked, axis=0)
    centered = stacked - bias
    return bias, {
        "identity_count": identity_count,
        "row_count": len(stacked),
        "bias_m": bias.tolist(),
        "radial_rmse_m": float(
            np.sqrt(np.mean(np.sum(np.square(centered), axis=1)))
        ),
        "estimate_available": True,
        "absolute_gauge_used": False,
    }


def fit_correlated_graph_field(
    graph_basis: np.ndarray,
    innovation_m: np.ndarray,
    available: np.ndarray,
    prior_reliability: np.ndarray,
    *,
    robust_scale_m: float,
    robust_iterations: int,
    projection_ridge: float,
    maximum_correction_m: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Fit one robust graph field with fixed correlated-block information mass."""

    basis = np.asarray(graph_basis, dtype=np.float64)
    innovation = np.asarray(innovation_m, dtype=np.float64)
    mask = np.asarray(available, dtype=bool).copy()
    reliability = np.asarray(prior_reliability, dtype=np.float64)
    _require(
        basis.ndim == 2
        and basis.shape[1] >= 1
        and innovation.shape == (len(basis), 3)
        and mask.shape == reliability.shape == (len(basis),),
        "graph-field inputs changed shape",
    )
    _require(
        np.all(np.isfinite(basis))
        and np.all(np.isfinite(reliability))
        and np.all((reliability >= 0.0) & (reliability <= 1.0)),
        "basis or reliability is invalid",
    )
    mask &= np.all(np.isfinite(innovation), axis=1) & (reliability > 0.0)
    selected_count = int(np.sum(mask))
    _require(selected_count >= 2, "insufficient graph-field support")
    design = basis[mask]
    target = innovation[mask]
    base = reliability[mask].copy()
    reliability_sum = float(np.sum(base))
    information_mass = float(min(basis.shape[1], selected_count))
    base *= information_mass / reliability_sum
    weights = base.copy()
    coefficients = np.zeros((basis.shape[1], 3), dtype=np.float64)
    ridge = projection_ridge * np.eye(basis.shape[1], dtype=np.float64)
    for _ in range(robust_iterations):
        normal = design.T @ (weights[:, None] * design) + ridge
        right = design.T @ (weights[:, None] * target)
        coefficients = np.linalg.solve(normal, right)
        residual_norm = np.linalg.norm(target - design @ coefficients, axis=1)
        robust = np.minimum(
            1.0,
            robust_scale_m / np.maximum(residual_norm, 1e-12),
        )
        weights = base * robust
        weight_sum = float(np.sum(weights))
        _require(weight_sum > 0.0, "robust graph field lost all support")
        weights *= information_mass / weight_sum
    coefficients, limit = scale_coefficients_to_field_limit(
        basis,
        coefficients,
        maximum_node_norm=maximum_correction_m,
    )
    field = basis @ coefficients
    residual = target - design @ coefficients
    return field, coefficients, {
        "selected_count": selected_count,
        "information_mass": information_mass,
        "prior_reliability_sum_before_normalization": reliability_sum,
        "effective_weight_sum": float(np.sum(weights)),
        "robust_downweighted_fraction": float(np.mean(weights < base)),
        "weighted_vector_rmse_m": float(
            np.sqrt(
                np.sum(weights * np.sum(np.square(residual), axis=1))
                / np.sum(weights)
            )
        ),
        "field_limit": limit,
    }


def _vector_rmse(
    prediction_m: np.ndarray,
    target_m: np.ndarray,
    available: np.ndarray,
) -> float:
    prediction = np.asarray(prediction_m, dtype=np.float64)
    target = np.asarray(target_m, dtype=np.float64)
    mask = np.asarray(available, dtype=bool).copy()
    _require(
        prediction.shape == target.shape
        and prediction.ndim == 3
        and prediction.shape[2] == 3
        and mask.shape == prediction.shape[:2],
        "validation shapes differ",
    )
    mask &= np.all(np.isfinite(prediction), axis=2) & np.all(
        np.isfinite(target), axis=2
    )
    _require(np.any(mask), "validation has no finite support")
    return float(
        np.sqrt(np.mean(np.sum(np.square(prediction[mask] - target[mask]), axis=1)))
    )


def build_guarded_dense_field(
    baseline_trajectory_m: np.ndarray,
    graph_basis: np.ndarray,
    dense_points_world_m: np.ndarray,
    dense_point_valid: np.ndarray,
    dense_camera_count: np.ndarray,
    dense_reprojection_error_px: np.ndarray,
    dense_quality_probability: np.ndarray,
    provider_trajectory_m: np.ndarray,
    provider_support: np.ndarray,
    provider_code: np.ndarray,
    provider_identity_ids: np.ndarray,
    provider_node_ids: np.ndarray,
    dense_local_trajectory_m: np.ndarray,
    dense_local_available: np.ndarray,
    validation_tracks_world_m: np.ndarray,
    validation_identity_ids: np.ndarray,
    validation_node_ids: np.ndarray,
    *,
    config: CrossProviderGuardedFieldConfig | None = None,
) -> dict[str, Any]:
    """Build and prefix-guard a dense displacement field.

    ``validation_tracks_world_m`` contains only identities that are disjoint
    from ``provider_identity_ids`` and from the later future-scoring set.
    """

    cfg = config or CrossProviderGuardedFieldConfig()
    baseline_input = np.asarray(baseline_trajectory_m)
    baseline = np.asarray(baseline_input, dtype=np.float64)
    basis = np.asarray(graph_basis, dtype=np.float64)
    dense = np.asarray(dense_points_world_m, dtype=np.float64)
    dense_valid = np.asarray(dense_point_valid, dtype=bool)
    camera_count = np.asarray(dense_camera_count)
    reprojection = np.asarray(dense_reprojection_error_px, dtype=np.float64)
    quality = np.asarray(dense_quality_probability, dtype=np.float64)
    provider = np.asarray(provider_trajectory_m, dtype=np.float64)
    provider_mask = np.asarray(provider_support, dtype=bool)
    provider_kind = np.asarray(provider_code)
    provider_ids = np.asarray(provider_identity_ids, dtype=np.int64)
    provider_nodes = np.asarray(provider_node_ids, dtype=np.int64)
    local_dense = np.asarray(dense_local_trajectory_m, dtype=np.float64)
    local_available = np.asarray(dense_local_available, dtype=bool)
    validation = np.asarray(validation_tracks_world_m, dtype=np.float64)
    validation_ids = np.asarray(validation_identity_ids, dtype=np.int64)
    validation_nodes = np.asarray(validation_node_ids, dtype=np.int64)
    frame_count, node_count, coordinate_count = baseline.shape
    _require(
        coordinate_count == 3
        and basis.shape[0] == node_count
        and basis.ndim == 2,
        "baseline and graph basis differ",
    )
    source_count = cfg.source_frame_count
    _require(
        dense.shape == (source_count, node_count, 3)
        and dense_valid.shape == camera_count.shape == reprojection.shape
        == quality.shape == (source_count, node_count),
        "dense source arrays differ from the frozen window",
    )
    _require(
        provider.shape == local_dense.shape
        and provider.shape[0] == source_count
        and provider.shape[2] == 3
        and provider_mask.shape == provider_kind.shape
        == local_available.shape == provider.shape[:2],
        "provider arrays differ from the frozen window",
    )
    _require(
        provider_ids.shape == provider_nodes.shape == (provider.shape[1],)
        and len(np.unique(provider_ids)) == len(provider_ids)
        and len(np.unique(provider_nodes)) == len(provider_nodes),
        "provider identities or graph nodes are invalid",
    )
    _require(
        np.all((provider_nodes >= 0) & (provider_nodes < node_count)),
        "provider graph nodes are out of range",
    )
    _require(
        validation.ndim == 3
        and validation_ids.shape == validation_nodes.shape
        and validation.shape[1:] == (len(validation_ids), 3)
        and len(validation) == cfg.validation_frame_end_exclusive
        - cfg.apply_frame_start,
        "validation arrays differ from the frozen interval",
    )
    _require(
        len(validation_ids) >= 2
        and len(np.unique(validation_ids)) == len(validation_ids)
        and len(np.unique(validation_nodes)) == len(validation_nodes)
        and np.all((validation_nodes >= 0) & (validation_nodes < node_count)),
        "validation arrays differ from the frozen interval",
    )
    _require(
        set(map(int, provider_ids)).isdisjoint(map(int, validation_ids)),
        "provider and validation identities overlap",
    )
    _require(
        cfg.validation_frame_end_exclusive <= frame_count,
        "baseline is shorter than validation",
    )

    start = 0
    endpoint = source_count - 1
    start_reliability = residual_independent_dense_reliability(
        quality[start],
        reprojection[start],
        camera_count[start],
        dense_valid[start],
        config=cfg,
    )
    endpoint_reliability = residual_independent_dense_reliability(
        quality[endpoint],
        reprojection[endpoint],
        camera_count[endpoint],
        dense_valid[endpoint],
        config=cfg,
    )
    reliability = np.sqrt(start_reliability * endpoint_reliability)
    available = reliability > 0.0
    dense_support_fraction = float(np.mean(available))

    bias_m, bias_diagnostics = estimate_relative_provider_bias(
        provider,
        provider_mask,
        provider_kind,
        local_dense,
        local_available,
        history_frames=cfg.bias_history_frames,
    )
    source_start = cfg.source_frame_start
    source_endpoint = cfg.source_frame_end_exclusive - 1
    dense_displacement = dense[endpoint] - dense[start]
    physical_displacement = (
        baseline[source_endpoint] - baseline[source_start]
    )
    innovation = dense_displacement - physical_displacement - bias_m

    raw_field = np.zeros((node_count, 3), dtype=np.float64)
    dense_coefficients = np.zeros((basis.shape[1], 3), dtype=np.float64)
    dense_fit: dict[str, Any] = {"selected_count": int(np.sum(available))}
    dense_support_passed = (
        dense_support_fraction >= cfg.minimum_dense_support_fraction
        and int(np.sum(available)) >= basis.shape[1]
    )
    if dense_support_passed:
        raw_field, dense_coefficients, dense_fit = fit_correlated_graph_field(
            basis,
            innovation,
            available,
            reliability,
            robust_scale_m=cfg.robust_scale_m,
            robust_iterations=cfg.robust_iterations,
            projection_ridge=cfg.projection_ridge,
            maximum_correction_m=cfg.maximum_correction_m,
        )

    provider_endpoint_available = (
        provider_mask[start]
        & provider_mask[endpoint]
        & np.all(np.isfinite(provider[endpoint]), axis=1)
        & np.all(np.isfinite(provider[start]), axis=1)
    )
    sparse_available_count = int(np.sum(provider_endpoint_available))
    sparse_reliability = np.zeros(node_count, dtype=np.float64)
    sparse_innovation = np.zeros((node_count, 3), dtype=np.float64)
    selected_provider_nodes = provider_nodes[provider_endpoint_available]
    if sparse_available_count:
        sparse_reliability[selected_provider_nodes] = 1.0
        sparse_innovation[selected_provider_nodes] = (
            provider[endpoint, provider_endpoint_available]
            - provider[start, provider_endpoint_available]
            - (
                baseline[source_endpoint, selected_provider_nodes]
                - baseline[source_start, selected_provider_nodes]
            )
        )
    sparse_field = np.zeros_like(raw_field)
    sparse_coefficients = np.zeros_like(dense_coefficients)
    sparse_fit: dict[str, Any] = {"selected_count": sparse_available_count}
    provider_support_passed = (
        sparse_available_count >= cfg.minimum_provider_count
        and int(bias_diagnostics["identity_count"]) >= cfg.minimum_provider_count
    )
    if provider_support_passed:
        sparse_field, sparse_coefficients, sparse_fit = fit_correlated_graph_field(
            basis,
            sparse_innovation,
            sparse_reliability > 0.0,
            sparse_reliability,
            robust_scale_m=cfg.robust_scale_m,
            robust_iterations=cfg.robust_iterations,
            projection_ridge=cfg.projection_ridge,
            maximum_correction_m=cfg.maximum_correction_m,
        )

    raw_candidate = baseline_input.copy()
    sparse_candidate = baseline_input.copy()
    raw_candidate[cfg.apply_frame_start :] = (
        baseline[cfg.apply_frame_start :] + raw_field[None]
    ).astype(baseline_input.dtype, copy=False)
    sparse_candidate[cfg.apply_frame_start :] = (
        baseline[cfg.apply_frame_start :] + sparse_field[None]
    ).astype(baseline_input.dtype, copy=False)
    validation_slice = slice(
        cfg.apply_frame_start,
        cfg.validation_frame_end_exclusive,
    )
    validation_available = np.all(np.isfinite(validation), axis=2)
    baseline_rmse = _vector_rmse(
        baseline[validation_slice, validation_nodes],
        validation,
        validation_available,
    )
    dense_rmse = _vector_rmse(
        raw_candidate[validation_slice, validation_nodes],
        validation,
        validation_available,
    )
    sparse_rmse = _vector_rmse(
        sparse_candidate[validation_slice, validation_nodes],
        validation,
        validation_available,
    )
    relative_gain_baseline = (
        1.0 - dense_rmse / baseline_rmse if baseline_rmse > 0.0 else 0.0
    )
    relative_gain_sparse = (
        1.0 - dense_rmse / sparse_rmse if sparse_rmse > 0.0 else 0.0
    )
    absolute_gain_baseline = baseline_rmse - dense_rmse
    absolute_gain_sparse = sparse_rmse - dense_rmse
    validation_passed = (
        relative_gain_baseline >= cfg.minimum_validation_improvement_fraction
        and relative_gain_sparse >= cfg.minimum_validation_improvement_fraction
        and absolute_gain_baseline >= cfg.minimum_validation_improvement_m
        and absolute_gain_sparse >= cfg.minimum_validation_improvement_m
    )
    accepted = dense_support_passed and provider_support_passed and validation_passed
    candidate = raw_candidate if accepted else baseline_input.copy()
    _require(
        accepted or np.array_equal(candidate, baseline_input),
        "rejected field did not preserve the baseline byte-for-byte",
    )
    return {
        "accepted": accepted,
        "reason": (
            "prefix-disjoint-validation-passed"
            if accepted
            else (
                "insufficient-dense-support"
                if not dense_support_passed
                else (
                    "insufficient-provider-support"
                    if not provider_support_passed
                    else "prefix-disjoint-validation-regret-guard"
                )
            )
        ),
        "candidate_trajectory_m": candidate,
        "raw_candidate_trajectory_m": raw_candidate,
        "sparse_comparator_trajectory_m": sparse_candidate,
        "dense_field_m": raw_field,
        "sparse_field_m": sparse_field,
        "dense_coefficients_m": dense_coefficients,
        "sparse_coefficients_m": sparse_coefficients,
        "diagnostics": {
            "config": asdict(cfg),
            "dense_support_fraction": dense_support_fraction,
            "dense_support_passed": dense_support_passed,
            "provider_endpoint_count": sparse_available_count,
            "provider_support_passed": provider_support_passed,
            "relative_provider_bias": bias_diagnostics,
            "dense_fit": dense_fit,
            "sparse_fit": sparse_fit,
            "validation": {
                "baseline_vector_rmse_m": baseline_rmse,
                "dense_vector_rmse_m": dense_rmse,
                "sparse_vector_rmse_m": sparse_rmse,
                "dense_gain_over_baseline_fraction": relative_gain_baseline,
                "dense_gain_over_sparse_fraction": relative_gain_sparse,
                "dense_gain_over_baseline_m": absolute_gain_baseline,
                "dense_gain_over_sparse_m": absolute_gain_sparse,
                "passed": validation_passed,
            },
            "information_boundary": {
                "future_object_observation_read": False,
                "future_scoring_identity_read": False,
                "state_innovation_used_in_prior_reliability": False,
                "dense_rows_treated_as_independent_samples": False,
                "dense_block_information_mass": int(basis.shape[1]),
                "rejection_is_bit_exact_baseline": bool(
                    not accepted and np.array_equal(candidate, baseline_input)
                ),
            },
        },
    }


__all__ = [
    "CrossProviderGuardedFieldConfig",
    "build_guarded_dense_field",
    "estimate_relative_provider_bias",
    "fit_correlated_graph_field",
    "residual_independent_dense_reliability",
]
