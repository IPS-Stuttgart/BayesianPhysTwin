"""Observation-side diagnostics for graph-persistent PhysTwin residuals."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from .dynamic_discrepancy import project_prefix_graph_coefficients


_PER_VIEW_POINT_KEYS = (
    "object_points_by_camera",
    "object_points_by_view",
    "per_camera_object_points",
)
_PER_VIEW_VALID_KEYS = (
    "object_valid_by_camera",
    "object_valid_by_view",
    "per_camera_object_valid",
)


def cross_view_residual_audit(
    observed_by_view_m: np.ndarray,
    valid_by_view: np.ndarray,
    baseline_m: np.ndarray,
    graph_basis: np.ndarray,
    *,
    ridge: float,
) -> dict[str, Any]:
    """Test whether prefix residual fields transfer across calibrated views."""

    observed = np.asarray(observed_by_view_m, dtype=float)
    valid = np.asarray(valid_by_view, dtype=bool)
    baseline = np.asarray(baseline_m, dtype=float)
    basis = np.asarray(graph_basis, dtype=float)
    if observed.ndim != 4 or observed.shape[3] != 3:
        raise ValueError("observed_by_view_m must have shape (V, T, N, 3)")
    if valid.shape != observed.shape[:3]:
        raise ValueError("valid_by_view must have shape (V, T, N)")
    if baseline.shape != observed.shape[1:]:
        raise ValueError("baseline_m must have shape (T, N, 3)")
    if observed.shape[0] < 2:
        raise ValueError("cross-view auditing requires at least two views")
    if observed.shape[2] > basis.shape[0]:
        raise ValueError("graph basis does not cover observed nodes")
    coefficients = []
    per_view = []
    for view in range(len(observed)):
        residual = observed[view] - baseline
        history = project_prefix_graph_coefficients(
            residual,
            valid[view],
            basis,
            ridge=ridge,
        )
        coefficient = history[-1]
        coefficients.append(coefficient)
        selected = valid[view] & np.all(np.isfinite(residual), axis=2)
        corrected = residual - (basis[: residual.shape[1]] @ coefficient)[None]
        per_view.append(
            {
                "view_index": view,
                "baseline_rmse_m": float(
                    np.sqrt(np.mean(np.square(residual[selected])))
                ),
                "own_field_rmse_m": float(
                    np.sqrt(np.mean(np.square(corrected[selected])))
                ),
            }
        )
    coefficient_array = np.stack(coefficients)
    leave_one_out = []
    for held_out in range(len(observed)):
        source = np.delete(coefficient_array, held_out, axis=0)
        correction = basis[: observed.shape[2]] @ np.mean(source, axis=0)
        residual = observed[held_out] - baseline
        selected = valid[held_out] & np.all(np.isfinite(residual), axis=2)
        baseline_rmse = float(np.sqrt(np.mean(np.square(residual[selected]))))
        corrected_rmse = float(
            np.sqrt(np.mean(np.square((residual - correction[None])[selected])))
        )
        leave_one_out.append(
            {
                "held_out_view_index": held_out,
                "baseline_rmse_m": baseline_rmse,
                "cross_view_rmse_m": corrected_rmse,
                "cross_view_error_ratio": corrected_rmse / baseline_rmse,
            }
        )
    centered = coefficient_array - np.mean(coefficient_array, axis=0, keepdims=True)
    denominator = max(
        float(np.linalg.norm(np.mean(coefficient_array, axis=0))),
        np.finfo(float).tiny,
    )
    return {
        "status": "available",
        "view_count": len(observed),
        "common_metric_frame_required": True,
        "per_view": per_view,
        "leave_one_view_out": leave_one_out,
        "mean_cross_view_error_ratio": float(
            np.mean([value["cross_view_error_ratio"] for value in leave_one_out])
        ),
        "relative_coefficient_dispersion": float(
            np.sqrt(np.mean(np.square(centered))) / denominator
        ),
    }


def released_observation_capability_audit(
    data: Mapping[str, Any],
) -> dict[str, Any]:
    """Declare which observation-localization questions released data support."""

    point_key = next((key for key in _PER_VIEW_POINT_KEYS if key in data), None)
    valid_key = next((key for key in _PER_VIEW_VALID_KEYS if key in data), None)
    per_view_available = point_key is not None and valid_key is not None
    confidence_available = any(
        key in data
        for key in (
            "object_track_confidence",
            "cotracker_confidence",
            "forward_backward_track_error",
        )
    )
    object_frame_available = any(
        key in data
        for key in (
            "object_frame_transforms",
            "material_frame_transforms",
            "canonical_object_transforms",
        )
    )
    point_to_plane_available = any(
        key in data for key in ("surface_normals", "observed_surface_normals")
    )
    return {
        "status": "available" if per_view_available else "partially_unavailable",
        "cross_view_residual_fields": {
            "available": per_view_available,
            "point_key": point_key,
            "valid_key": valid_key,
            "reason": (
                None
                if per_view_available
                else "released artifact has fused 3D tracks without per-view identities"
            ),
        },
        "object_frame_consistency": {
            "available": object_frame_available,
            "reason": (
                None
                if object_frame_available
                else "released artifact has no time-indexed material-frame transform"
            ),
        },
        "visibility_confidence_regression": {
            "available": confidence_available,
            "reason": (
                None
                if confidence_available
                else "released artifact retains validity masks but no continuous confidence"
            ),
        },
        "point_to_plane_metric": {
            "available": point_to_plane_available,
            "reason": (
                None
                if point_to_plane_available
                else "released artifact has no matched observed surface normals"
            ),
        },
        "manual_track_and_chamfer_agreement": {"available": True},
    }


def metric_agreement_audit(
    chamfer_by_frame_m: np.ndarray,
    track_by_frame_m: np.ndarray,
) -> dict[str, float | None]:
    """Report whether readout gains agree across geometric and track metrics."""

    chamfer = np.asarray(chamfer_by_frame_m, dtype=float)
    track = np.asarray(track_by_frame_m, dtype=float)
    if chamfer.ndim != 1 or track.shape != chamfer.shape:
        raise ValueError("metric series must be matching vectors")
    finite = np.isfinite(chamfer) & np.isfinite(track)
    if np.sum(finite) < 3:
        return {"pearson_correlation": None, "frame_count": int(np.sum(finite))}
    correlation = float(np.corrcoef(chamfer[finite], track[finite])[0, 1])
    return {
        "pearson_correlation": correlation if np.isfinite(correlation) else None,
        "frame_count": int(np.sum(finite)),
    }
