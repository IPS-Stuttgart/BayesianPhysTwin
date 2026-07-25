"""Direct 2-D cross-view validation of physical-response state updates.

Unlike the triangulate-then-update diagnostic, this method does not require
three inlier cameras for every material point inside each split.  It estimates
state coefficients directly from tracked pixel residuals, constrained to a
causal PhysTwin response basis.  Per-camera constant image offsets are removed
as nuisance terms and every camera receives fixed total information mass, so
duplicating tracked pixels cannot create arbitrary confidence.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

import numpy as np

from .bias_aware_belief import build_physical_response_basis
from .deform360_bias_aware_belief_development import (
    Deform360BiasAwareDevelopmentConfig,
)
from .deform360_crossview_guard import (
    CrossViewGuardConfig,
    _projection_jacobian,
    _require,
    _validate_inputs,
    _validation_passes,
    correction_agreement_diagnostic,
    deterministic_camera_halves,
    heldout_reprojection_diagnostic,
)
from .deform360_raw_camera_observation import project_world_points


PROTOCOL_ID = "deform360-direct-2d-crossview-guard-v1-postopen-development"


@dataclass(frozen=True)
class DirectCrossView2DConfig:
    """Frozen proposal-fit settings for direct camera-space cross-validation."""

    minimum_fit_camera_count: int = 3
    minimum_centers_per_camera: int = 4
    minimum_total_observation_count: int = 18
    observation_std_px: float = 2.0
    state_prior_std_m: float = 0.020
    effective_observations_per_camera: float = 8.0
    degrees_of_freedom: float = 4.0
    minimum_robust_weight: float = 0.02
    maximum_iterations: int = 8
    minimum_design_singular_fraction: float = 0.02
    maximum_state_correction_m: float = 0.050

    def __post_init__(self) -> None:
        _require(
            self.minimum_fit_camera_count >= 2,
            "direct fit needs at least two cameras",
        )
        _require(
            self.minimum_centers_per_camera >= 2,
            "direct fit needs at least two centres per camera",
        )
        _require(
            self.minimum_total_observation_count >= 1,
            "direct fit observation count must be positive",
        )
        positive = (
            self.observation_std_px,
            self.state_prior_std_m,
            self.effective_observations_per_camera,
            self.degrees_of_freedom,
            self.maximum_state_correction_m,
        )
        _require(
            all(np.isfinite(value) and value > 0.0 for value in positive),
            "direct 2-D fit scales must be positive",
        )
        _require(
            0.0 < self.minimum_robust_weight <= 1.0,
            "minimum robust weight must lie in (0, 1]",
        )
        _require(self.maximum_iterations >= 1, "iteration count must be positive")
        _require(
            0.0 < self.minimum_design_singular_fraction <= 1.0,
            "design singular fraction must lie in (0, 1]",
        )


def camera_balanced_pair_weights(
    camera_ids: np.ndarray, *, effective_observations_per_camera: float
) -> np.ndarray:
    """Assign fixed total information mass to every represented camera."""

    identifiers = np.asarray(camera_ids, dtype=np.int64)
    _require(identifiers.ndim == 1 and len(identifiers), "camera IDs are empty")
    _require(
        np.isfinite(effective_observations_per_camera)
        and effective_observations_per_camera > 0.0,
        "effective camera information is invalid",
    )
    result = np.empty(len(identifiers), dtype=np.float64)
    for camera_id in np.unique(identifiers):
        selected = identifiers == camera_id
        result[selected] = effective_observations_per_camera / np.sum(selected)
    return result


def _camera_space_design(
    baseline_points_m: np.ndarray,
    physical_basis: np.ndarray,
    supplement: Mapping[str, np.ndarray],
    update_index: int,
    view_indices: np.ndarray,
    *,
    config: DirectCrossView2DConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    centers = np.asarray(supplement["center_ids"], dtype=np.int64)
    camera_names = tuple(str(value) for value in supplement["selected_cameras"])
    tracks = np.asarray(supplement["track_pixels_xy"], dtype=np.float64)
    visibility = np.asarray(supplement["track_visibility"], dtype=bool)
    intrinsics = np.asarray(supplement["intrinsics"], dtype=np.float64)
    camera_to_world = np.asarray(
        supplement["camera_to_world"], dtype=np.float64
    )
    baseline = np.asarray(baseline_points_m, dtype=np.float64)
    basis = np.asarray(physical_basis, dtype=np.float64)
    _require(basis.shape[0] == len(baseline), "physical basis point count changed")
    design_blocks: list[np.ndarray] = []
    residual_blocks: list[np.ndarray] = []
    camera_blocks: list[np.ndarray] = []
    camera_records: list[dict[str, Any]] = []

    for local_camera_id, view_index in enumerate(view_indices):
        baseline_pixels, depth = project_world_points(
            baseline[centers],
            intrinsics[view_index],
            camera_to_world[view_index],
        )
        observed = tracks[update_index, view_index]
        usable = (
            visibility[update_index, view_index]
            & np.all(np.isfinite(observed), axis=1)
            & np.all(np.isfinite(baseline_pixels), axis=1)
            & (depth > 0.0)
        )
        selected = np.flatnonzero(usable)
        if len(selected) < config.minimum_centers_per_camera:
            camera_records.append(
                {
                    "camera": camera_names[view_index],
                    "observation_count": len(selected),
                    "used": False,
                }
            )
            continue
        projection = intrinsics[view_index] @ np.linalg.inv(
            camera_to_world[view_index]
        )[:3]
        camera_design: list[np.ndarray] = []
        camera_residual: list[np.ndarray] = []
        for center_index in selected:
            center_id = centers[center_index]
            jacobian = _projection_jacobian(baseline[center_id], projection)
            block = np.einsum(
                "ad,r->ard", jacobian, basis[center_id]
            ).reshape(2, -1)
            camera_design.append(block)
            camera_residual.append(
                observed[center_index] - baseline_pixels[center_index]
            )
        design = np.stack(camera_design)
        residual = np.stack(camera_residual)
        # Constant per-camera pixel offsets are nuisance variables.  Removing
        # their means before fitting prevents them from entering state modes.
        design -= np.mean(design, axis=0, keepdims=True)
        residual -= np.mean(residual, axis=0, keepdims=True)
        design_blocks.append(design)
        residual_blocks.append(residual)
        camera_blocks.append(
            np.full(len(selected), local_camera_id, dtype=np.int64)
        )
        camera_records.append(
            {
                "camera": camera_names[view_index],
                "observation_count": len(selected),
                "used": True,
            }
        )
    used_camera_count = len(design_blocks)
    _require(
        used_camera_count >= config.minimum_fit_camera_count,
        "too few cameras retain direct 2-D support",
    )
    design = np.concatenate(design_blocks, axis=0)
    residual = np.concatenate(residual_blocks, axis=0)
    camera_ids = np.concatenate(camera_blocks)
    _require(
        len(design) >= config.minimum_total_observation_count,
        "too few direct 2-D observations",
    )
    pair_weights = camera_balanced_pair_weights(
        camera_ids,
        effective_observations_per_camera=(
            config.effective_observations_per_camera
        ),
    )
    return design, residual, pair_weights, {
        "fit_cameras": [camera_names[index] for index in view_indices],
        "used_camera_count": used_camera_count,
        "observation_count": len(design),
        "cameras": camera_records,
        "camera_information_treatment": (
            "constant per-camera pixel offsets removed; each camera has fixed "
            "total information mass independent of tracked point count"
        ),
    }


def fit_direct_crossview_correction(
    baseline_points_m: np.ndarray,
    physical_basis: np.ndarray,
    supplement: Mapping[str, np.ndarray],
    update_index: int,
    view_indices: np.ndarray,
    *,
    config: DirectCrossView2DConfig | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit one robust physical-basis correction from a camera subset."""

    cfg = config or DirectCrossView2DConfig()
    basis = np.asarray(physical_basis, dtype=np.float64)
    design, residual, base_weights, report = _camera_space_design(
        baseline_points_m,
        basis,
        supplement,
        update_index,
        view_indices,
        config=cfg,
    )
    weighted = design.reshape(-1, design.shape[-1]) * np.repeat(
        np.sqrt(base_weights), 2
    )[:, None]
    _, singular_values, right = np.linalg.svd(weighted, full_matrices=False)
    _require(len(singular_values) and singular_values[0] > 0.0, "2-D design is empty")
    retained = singular_values >= (
        singular_values[0] * cfg.minimum_design_singular_fraction
    )
    _require(np.any(retained), "2-D design has no identifiable state mode")
    transform = right[retained].T
    reduced_design = np.einsum("oap,pr->oar", design, transform)
    robust = np.ones(len(design), dtype=np.float64)
    coefficients = np.zeros(transform.shape[1], dtype=np.float64)
    for _ in range(cfg.maximum_iterations):
        pair_weight = base_weights * robust / cfg.observation_std_px**2
        row_weight = np.repeat(pair_weight, 2)
        matrix = reduced_design.reshape(-1, transform.shape[1])
        target = residual.reshape(-1)
        lhs = matrix.T @ (row_weight[:, None] * matrix)
        lhs += np.eye(len(coefficients)) / cfg.state_prior_std_m**2
        rhs = matrix.T @ (row_weight * target)
        updated = np.linalg.solve(lhs, rhs)
        fitted = np.einsum("oar,r->oa", reduced_design, updated)
        radial = np.linalg.norm(residual - fitted, axis=1)
        standardized = radial / cfg.observation_std_px
        updated_robust = np.maximum(
            cfg.minimum_robust_weight,
            (cfg.degrees_of_freedom + 2.0)
            / (cfg.degrees_of_freedom + np.square(standardized)),
        )
        if np.max(np.abs(updated - coefficients)) <= 1e-10:
            coefficients = updated
            robust = updated_robust
            break
        coefficients = updated
        robust = updated_robust
    full_coefficients = transform @ coefficients
    correction = basis @ full_coefficients.reshape(basis.shape[1], 3)
    maximum = float(np.max(np.linalg.norm(correction, axis=1)))
    _require(
        maximum <= cfg.maximum_state_correction_m,
        "direct 2-D state correction exceeds the declared cap",
    )
    fitted = np.einsum(
        "oap,p->oa", design, full_coefficients
    )
    baseline_loss = float(np.mean(np.linalg.norm(residual, axis=1)))
    fitted_loss = float(np.mean(np.linalg.norm(residual - fitted, axis=1)))
    report.update(
        {
            "physical_basis_rank": int(basis.shape[1]),
            "identifiable_parameter_rank": int(transform.shape[1]),
            "maximum_state_correction_m": maximum,
            "correction_rms_m": float(np.sqrt(np.mean(np.square(correction)))),
            "fit_mean_pixel_error_before_px": baseline_loss,
            "fit_mean_pixel_error_after_px": fitted_loss,
            "fit_error_improvement_fraction": (
                (baseline_loss - fitted_loss) / max(baseline_loss, 1e-12)
            ),
            "minimum_robust_weight": float(np.min(robust)),
            "maximum_robust_weight": float(np.max(robust)),
            "design_singular_values": singular_values.tolist(),
        }
    )
    return correction, report


def predict_direct_crossview_guarded_candidate_arrays(
    baseline_m: np.ndarray,
    physical_response_m: np.ndarray,
    frame_zero_points_m: np.ndarray,
    action_support: np.ndarray,
    supplement: Mapping[str, np.ndarray],
    *,
    development_config: Deform360BiasAwareDevelopmentConfig | None = None,
    fit_config: DirectCrossView2DConfig | None = None,
    guard_config: CrossViewGuardConfig | None = None,
) -> tuple[dict[str, Any], np.ndarray]:
    """Return direct 2-D guarded updates or an exact baseline fallback."""

    development = development_config or Deform360BiasAwareDevelopmentConfig()
    fit = fit_config or DirectCrossView2DConfig()
    guard = guard_config or CrossViewGuardConfig()
    baseline_input = np.asarray(baseline_m)
    baseline = np.asarray(baseline_input, dtype=np.float64)
    response = np.asarray(physical_response_m, dtype=np.float64)
    frame_zero = np.asarray(frame_zero_points_m, dtype=np.float64)
    support = np.asarray(action_support, dtype=np.float64)
    _validate_inputs(
        baseline,
        response,
        frame_zero,
        support,
        supplement,
        development.update_frames,
    )
    camera_names = tuple(str(value) for value in supplement["selected_cameras"])
    first_views, second_views = deterministic_camera_halves(camera_names)
    guarded = baseline_input.copy()
    updates: list[dict[str, Any]] = []
    accepted_count = 0

    for update_index, update_frame in enumerate(development.update_frames):
        stop = (
            development.update_frames[update_index + 1]
            if update_index + 1 < len(development.update_frames)
            else len(baseline)
        )
        record: dict[str, Any] = {
            "frame": update_frame,
            "interval_end_exclusive": stop,
            "candidate_available_in_both_groups": False,
            "accepted": False,
            "decision": "direct-fit-unavailable-exact-baseline-fallback",
            "bit_exact_baseline_fallback": True,
        }
        try:
            physical_basis = build_physical_response_basis(
                response[: update_frame + 1],
                action_support=support,
                rank=development.physical_response_rank,
                minimum_response_m=development.minimum_physical_response_m,
            )
            first_correction, first_fit = fit_direct_crossview_correction(
                baseline[update_frame],
                physical_basis.basis,
                supplement,
                update_index,
                first_views,
                config=fit,
            )
            second_correction, second_fit = fit_direct_crossview_correction(
                baseline[update_frame],
                physical_basis.basis,
                supplement,
                update_index,
                second_views,
                config=fit,
            )
            record["candidate_available_in_both_groups"] = True
            first_validation = heldout_reprojection_diagnostic(
                baseline[update_frame],
                first_correction,
                supplement,
                update_index,
                second_views,
                huber_delta_px=guard.heldout_huber_delta_px,
            )
            second_validation = heldout_reprojection_diagnostic(
                baseline[update_frame],
                second_correction,
                supplement,
                update_index,
                first_views,
                huber_delta_px=guard.heldout_huber_delta_px,
            )
            agreement = correction_agreement_diagnostic(
                first_correction, second_correction
            )
            first_pass = _validation_passes(first_validation, guard)
            second_pass = _validation_passes(second_validation, guard)
            agreement_pass = bool(
                agreement["cosine"] >= guard.minimum_correction_cosine
                and agreement["disagreement_ratio"]
                <= guard.maximum_correction_disagreement_ratio
            )
            record.update(
                {
                    "fit_group_a": first_fit,
                    "fit_group_b": second_fit,
                    "fit_a_validated_on_b": first_validation,
                    "fit_b_validated_on_a": second_validation,
                    "fit_a_validation_passed": first_pass,
                    "fit_b_validation_passed": second_pass,
                    "correction_agreement": agreement,
                    "correction_agreement_passed": agreement_pass,
                }
            )
            if first_pass and second_pass and agreement_pass:
                correction = 0.5 * (first_correction + second_correction)
                guarded[update_frame + 1 : stop] = (
                    baseline[update_frame + 1 : stop] + correction[None]
                ).astype(baseline_input.dtype, copy=False)
                record.update(
                    {
                        "accepted": True,
                        "decision": "direct-2d-bidirectional-crossview-accepted",
                        "bit_exact_baseline_fallback": False,
                        "combined_correction_rms_m": float(
                            np.sqrt(np.mean(np.square(correction)))
                        ),
                        "maximum_combined_correction_m": float(
                            np.max(np.linalg.norm(correction, axis=1))
                        ),
                    }
                )
                accepted_count += 1
            else:
                record["decision"] = "direct-2d-validation-exact-baseline-fallback"
        except (ValueError, np.linalg.LinAlgError) as error:
            record["fallback_reason"] = f"{type(error).__name__}: {error}"
        if not record["accepted"] and not np.array_equal(
            guarded[update_frame + 1 : stop],
            baseline_input[update_frame + 1 : stop],
        ):
            raise AssertionError("direct cross-view fallback changed the baseline")
        updates.append(record)

    report = {
        "protocol_id": PROTOCOL_ID,
        "arm": "bias_aware_direct_2d_crossview_guard",
        "development_config": asdict(development),
        "direct_2d_fit_config": asdict(fit),
        "crossview_guard_config": asdict(guard),
        "camera_split": {
            "rule": "lexicographically sorted camera identities, interleaved",
            "group_a_indices": first_views.tolist(),
            "group_b_indices": second_views.tolist(),
            "group_a_cameras": [camera_names[index] for index in first_views],
            "group_b_cameras": [camera_names[index] for index in second_views],
        },
        "updates": updates,
        "accepted_count": accepted_count,
        "exact_fallback_count": len(updates) - accepted_count,
        "information_boundary": {
            "target_argument_accepted": False,
            "outcome_argument_accepted": False,
            "future_observation_read": False,
            "camera_fit_and_validation_groups_disjoint": True,
            "correlated_points_do_not_accumulate_camera_information": True,
            "state_support_derived_from_causal_physical_response": True,
        },
        "claim_boundary": (
            "Direct cross-view validation cannot identify a bias shared "
            "coherently by all cameras. This is post-open method development."
        ),
    }
    return report, guarded


__all__ = [
    "PROTOCOL_ID",
    "DirectCrossView2DConfig",
    "camera_balanced_pair_weights",
    "fit_direct_crossview_correction",
    "predict_direct_crossview_guarded_candidate_arrays",
]
