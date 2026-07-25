"""Disjoint-camera validation for target-free Deform360 state updates.

This module is a post-open method diagnostic.  It preserves the frozen
triangulated observation path and consumes the separate camera-track
supplement.  Two disjoint camera groups independently propose a correction;
each proposal must improve reprojection on the cameras it did not use.  The
unchanged baseline is returned byte-for-byte whenever either direction lacks
support, fails validation, or yields an incompatible correction.

Cross-view validation cannot identify a bias shared coherently by every
camera.  It is therefore evidence against view-specific tracking error, not a
proof that an accepted update is physical.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from .deform360_bias_aware_belief_development import (
    Deform360BiasAwareDevelopmentConfig,
    predict_bias_aware_candidate_arrays,
)
from .deform360_raw_camera_observation import (
    RawCameraObservationConfig,
    project_world_points,
    triangulate_observation_ransac,
)


PROTOCOL_ID = "deform360-disjoint-crossview-guard-v1-postopen-development"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


@dataclass(frozen=True)
class CrossViewGuardConfig:
    """Pre-outcome choices for one disjoint-camera validation diagnostic."""

    minimum_fit_inlier_view_count: int = 3
    minimum_heldout_center_count: int = 6
    tracking_noise_std_px: float = 2.0
    reprojection_reliability_scale_px: float = 3.0
    heldout_huber_delta_px: float = 3.0
    minimum_heldout_improvement_fraction: float = 0.05
    minimum_heldout_mean_error_reduction_px: float = 0.05
    minimum_correction_cosine: float = 0.50
    maximum_correction_disagreement_ratio: float = 1.0

    def __post_init__(self) -> None:
        _require(
            self.minimum_fit_inlier_view_count >= 3,
            "cross-view fitting requires at least three inlier cameras",
        )
        _require(
            self.minimum_heldout_center_count >= 1,
            "held-out centre count must be positive",
        )
        positive = (
            self.tracking_noise_std_px,
            self.reprojection_reliability_scale_px,
            self.heldout_huber_delta_px,
            self.minimum_heldout_mean_error_reduction_px,
            self.maximum_correction_disagreement_ratio,
        )
        _require(
            all(np.isfinite(value) and value > 0.0 for value in positive),
            "cross-view scales must be positive",
        )
        _require(
            0.0 <= self.minimum_heldout_improvement_fraction < 1.0,
            "held-out improvement fraction must lie in [0, 1)",
        )
        _require(
            -1.0 <= self.minimum_correction_cosine <= 1.0,
            "correction cosine must lie in [-1, 1]",
        )


def deterministic_camera_halves(
    camera_names: Sequence[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Split sorted camera identities into interleaved, disjoint halves."""

    names = tuple(str(name) for name in camera_names)
    _require(len(names) >= 6, "at least six cameras are required")
    _require(len(set(names)) == len(names), "camera identities are not unique")
    order = np.asarray(sorted(range(len(names)), key=lambda index: names[index]))
    first = order[::2]
    second = order[1::2]
    _require(len(first) >= 3 and len(second) >= 3, "camera halves are too small")
    return first.astype(np.int64), second.astype(np.int64)


def _projection_matrix(
    intrinsic: np.ndarray, camera_to_world: np.ndarray
) -> np.ndarray:
    return np.asarray(intrinsic, dtype=np.float64) @ np.linalg.inv(
        np.asarray(camera_to_world, dtype=np.float64)
    )[:3]


def _projection_jacobian(
    point_m: np.ndarray, projection_matrix: np.ndarray
) -> np.ndarray:
    point = np.append(np.asarray(point_m, dtype=np.float64), 1.0)
    matrix = np.asarray(projection_matrix, dtype=np.float64)
    homogeneous = matrix @ point
    depth = float(homogeneous[2])
    _require(depth > 1e-9, "point is behind a triangulation camera")
    return np.stack(
        (
            (matrix[0, :3] * depth - homogeneous[0] * matrix[2, :3])
            / depth**2,
            (matrix[1, :3] * depth - homogeneous[1] * matrix[2, :3])
            / depth**2,
        )
    )


def conservative_triangulation_variance_m2(
    point_m: np.ndarray,
    projection_matrices: Sequence[np.ndarray],
    *,
    pixel_std: float,
) -> float:
    """Return a worst-axis variance without independent-view accumulation.

    The per-camera information matrices are averaged rather than summed.
    Duplicating an identical camera block therefore cannot make the estimate
    arbitrarily confident when cross-view error correlation is unknown.
    """

    matrices = tuple(np.asarray(matrix, dtype=np.float64) for matrix in projection_matrices)
    _require(len(matrices) >= 2, "triangulation covariance needs two cameras")
    _require(np.isfinite(pixel_std) and pixel_std > 0.0, "pixel noise is invalid")
    jacobians = np.stack(
        [_projection_jacobian(point_m, matrix) for matrix in matrices]
    )
    information = np.mean(
        np.einsum("vki,vkj->vij", jacobians, jacobians), axis=0
    )
    eigenvalues = np.linalg.eigvalsh(information)
    _require(
        eigenvalues[0] > max(1e-12, 1e-10 * eigenvalues[-1]),
        "triangulation covariance is singular",
    )
    return float(pixel_std**2 / eigenvalues[0])


def _validate_inputs(
    baseline_m: np.ndarray,
    physical_response_m: np.ndarray,
    frame_zero_points_m: np.ndarray,
    action_support: np.ndarray,
    supplement: Mapping[str, np.ndarray],
    update_frames: tuple[int, ...],
) -> None:
    baseline = np.asarray(baseline_m)
    _require(
        baseline.ndim == 3 and baseline.shape[2] == 3,
        "baseline must have shape (T, N, 3)",
    )
    _require(
        np.asarray(physical_response_m).shape == baseline.shape,
        "physical response shape changed",
    )
    _require(
        np.asarray(frame_zero_points_m).shape == baseline.shape[1:],
        "frame-zero point shape changed",
    )
    _require(
        np.asarray(action_support).shape == (baseline.shape[1],),
        "action support shape changed",
    )
    centers = np.asarray(supplement["center_ids"], dtype=np.int64)
    cameras = np.asarray(supplement["selected_cameras"])
    updates = np.asarray(supplement["update_frames"], dtype=np.int64)
    tracks = np.asarray(supplement["track_pixels_xy"])
    visibility = np.asarray(supplement["track_visibility"])
    _require(
        tuple(int(value) for value in updates) == update_frames,
        "supplement update frames differ from the predictor",
    )
    _require(
        tracks.shape == (len(updates), len(cameras), len(centers), 2),
        "camera-track shape changed",
    )
    _require(
        visibility.shape == tracks.shape[:-1],
        "camera visibility shape changed",
    )
    _require(
        np.asarray(supplement["intrinsics"]).shape == (len(cameras), 3, 3)
        and np.asarray(supplement["camera_to_world"]).shape
        == (len(cameras), 4, 4),
        "camera calibration shape changed",
    )
    _require(
        np.all((centers >= 0) & (centers < baseline.shape[1])),
        "centre ID exceeds the trajectory",
    )
    _require(
        np.allclose(
            np.asarray(supplement["center_frame_zero_points_m"]),
            np.asarray(frame_zero_points_m)[centers],
            atol=1e-6,
            rtol=0.0,
        ),
        "supplement material identities differ from the prediction",
    )


def _camera_maps(
    supplement: Mapping[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    names = tuple(str(value) for value in supplement["selected_cameras"])
    intrinsics = np.asarray(supplement["intrinsics"], dtype=np.float64)
    camera_to_world = np.asarray(
        supplement["camera_to_world"], dtype=np.float64
    )
    projections = {
        name: _projection_matrix(intrinsics[index], camera_to_world[index])
        for index, name in enumerate(names)
    }
    origins = {
        name: camera_to_world[index, :3, 3]
        for index, name in enumerate(names)
    }
    return projections, origins


def _triangulate_group_measurement(
    baseline_shape: tuple[int, int, int],
    frame_zero_points_m: np.ndarray,
    supplement: Mapping[str, np.ndarray],
    view_indices: np.ndarray,
    *,
    raw_config: RawCameraObservationConfig,
    guard_config: CrossViewGuardConfig,
    variance_floor_m2: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    centers = np.asarray(supplement["center_ids"], dtype=np.int64)
    camera_names = tuple(str(value) for value in supplement["selected_cameras"])
    update_frames = tuple(int(value) for value in supplement["update_frames"])
    tracks = np.asarray(supplement["track_pixels_xy"], dtype=np.float64)
    track_visibility = np.asarray(supplement["track_visibility"], dtype=bool)
    projections, origins = _camera_maps(supplement)
    measurement = np.full(baseline_shape, np.nan, dtype=np.float64)
    visibility = np.zeros(baseline_shape[:2], dtype=bool)
    validity = np.zeros_like(visibility)
    measurement[0, centers] = np.asarray(frame_zero_points_m)[centers]
    visibility[0, centers] = True
    validity[0, centers] = True
    reliability = np.zeros((len(update_frames), len(centers)), dtype=np.float64)
    variance = np.full_like(reliability, variance_floor_m2)
    update_records: list[dict[str, Any]] = []

    for update_index, update_frame in enumerate(update_frames):
        accepted = 0
        diagnostics: list[dict[str, Any]] = []
        for center_index, center_id in enumerate(centers):
            observations = {
                camera_names[view_index]: tracks[
                    update_index, view_index, center_index
                ]
                for view_index in view_indices
                if track_visibility[update_index, view_index, center_index]
                and np.all(
                    np.isfinite(
                        tracks[update_index, view_index, center_index]
                    )
                )
            }
            point, diagnostic = triangulate_observation_ransac(
                observations,
                projections,
                origins,
                np.asarray(frame_zero_points_m)[center_id],
                config=raw_config,
            )
            diagnostic = dict(diagnostic)
            diagnostic["center_id"] = int(center_id)
            if (
                point is None
                or int(diagnostic.get("inlier_view_count", 0))
                < guard_config.minimum_fit_inlier_view_count
            ):
                if point is not None:
                    diagnostic["accepted"] = False
                    diagnostic["decision"] = "independent_view_support_failure"
                diagnostics.append(diagnostic)
                continue
            inlier_names = tuple(str(name) for name in diagnostic["inlier_cameras"])
            pixel_std = max(
                guard_config.tracking_noise_std_px,
                float(diagnostic["median_reprojection_error_px"]),
            )
            try:
                point_variance = conservative_triangulation_variance_m2(
                    point,
                    [projections[name] for name in inlier_names],
                    pixel_std=pixel_std,
                )
            except ValueError as error:
                diagnostic["accepted"] = False
                diagnostic["decision"] = "covariance_failure"
                diagnostic["covariance_failure"] = str(error)
                diagnostics.append(diagnostic)
                continue
            measurement[update_frame, center_id] = point
            visibility[update_frame, center_id] = True
            validity[update_frame, center_id] = True
            reliability[update_index, center_index] = float(
                np.exp(
                    -0.5
                    * (
                        float(diagnostic["median_reprojection_error_px"])
                        / guard_config.reprojection_reliability_scale_px
                    )
                    ** 2
                )
            )
            variance[update_index, center_index] = max(
                point_variance, variance_floor_m2
            )
            diagnostic["conservative_variance_m2"] = float(
                variance[update_index, center_index]
            )
            diagnostics.append(diagnostic)
            accepted += 1
        update_records.append(
            {
                "frame": update_frame,
                "fit_camera_indices": view_indices.tolist(),
                "fit_cameras": [camera_names[index] for index in view_indices],
                "accepted_center_count": accepted,
                "centers": diagnostics,
            }
        )
    return measurement, visibility, validity, reliability, variance, {
        "updates": update_records,
        "minimum_fit_inlier_view_count": guard_config.minimum_fit_inlier_view_count,
        "correlation_treatment": (
            "per-camera information averaged, not summed; camera count does "
            "not independently multiply precision"
        ),
    }


def _huber_loss(value: np.ndarray, delta: float) -> np.ndarray:
    magnitude = np.asarray(value, dtype=np.float64)
    return np.where(
        magnitude <= delta,
        0.5 * np.square(magnitude),
        delta * (magnitude - 0.5 * delta),
    )


def heldout_reprojection_diagnostic(
    baseline_points_m: np.ndarray,
    correction_m: np.ndarray,
    supplement: Mapping[str, np.ndarray],
    update_index: int,
    heldout_view_indices: np.ndarray,
    *,
    huber_delta_px: float,
) -> dict[str, Any]:
    """Score one correction only on cameras excluded from its fit."""

    centers = np.asarray(supplement["center_ids"], dtype=np.int64)
    cameras = tuple(str(value) for value in supplement["selected_cameras"])
    tracks = np.asarray(supplement["track_pixels_xy"], dtype=np.float64)
    visible = np.asarray(supplement["track_visibility"], dtype=bool)
    intrinsics = np.asarray(supplement["intrinsics"], dtype=np.float64)
    camera_to_world = np.asarray(
        supplement["camera_to_world"], dtype=np.float64
    )
    baseline = np.asarray(baseline_points_m, dtype=np.float64)[centers]
    corrected = baseline + np.asarray(correction_m, dtype=np.float64)[centers]
    camera_records: list[dict[str, Any]] = []
    all_baseline_errors: list[np.ndarray] = []
    all_candidate_errors: list[np.ndarray] = []
    observed_centers: set[int] = set()

    for view_index in heldout_view_indices:
        observed_pixels = tracks[update_index, view_index]
        baseline_pixels, baseline_depth = project_world_points(
            baseline,
            intrinsics[view_index],
            camera_to_world[view_index],
        )
        candidate_pixels, candidate_depth = project_world_points(
            corrected,
            intrinsics[view_index],
            camera_to_world[view_index],
        )
        usable = (
            visible[update_index, view_index]
            & np.all(np.isfinite(observed_pixels), axis=1)
            & np.all(np.isfinite(baseline_pixels), axis=1)
            & np.all(np.isfinite(candidate_pixels), axis=1)
            & (baseline_depth > 0.0)
            & (candidate_depth > 0.0)
        )
        baseline_error = np.linalg.norm(
            baseline_pixels[usable] - observed_pixels[usable], axis=1
        )
        candidate_error = np.linalg.norm(
            candidate_pixels[usable] - observed_pixels[usable], axis=1
        )
        all_baseline_errors.append(baseline_error)
        all_candidate_errors.append(candidate_error)
        observed_centers.update(int(value) for value in np.flatnonzero(usable))
        camera_records.append(
            {
                "camera": cameras[view_index],
                "observation_count": int(np.sum(usable)),
                "baseline_mean_error_px": (
                    float(np.mean(baseline_error)) if len(baseline_error) else None
                ),
                "candidate_mean_error_px": (
                    float(np.mean(candidate_error)) if len(candidate_error) else None
                ),
                "baseline_mean_huber_loss_px2": (
                    float(np.mean(_huber_loss(baseline_error, huber_delta_px)))
                    if len(baseline_error)
                    else None
                ),
                "candidate_mean_huber_loss_px2": (
                    float(np.mean(_huber_loss(candidate_error, huber_delta_px)))
                    if len(candidate_error)
                    else None
                ),
            }
        )
    populated = [record for record in camera_records if record["observation_count"]]
    if not populated:
        return {
            "valid": False,
            "reason": "no-heldout-observations",
            "heldout_cameras": [cameras[index] for index in heldout_view_indices],
            "heldout_center_count": 0,
            "cameras": camera_records,
        }
    baseline_huber = float(
        np.mean([record["baseline_mean_huber_loss_px2"] for record in populated])
    )
    candidate_huber = float(
        np.mean([record["candidate_mean_huber_loss_px2"] for record in populated])
    )
    baseline_mean = float(
        np.mean([record["baseline_mean_error_px"] for record in populated])
    )
    candidate_mean = float(
        np.mean([record["candidate_mean_error_px"] for record in populated])
    )
    return {
        "valid": True,
        "reason": "scored",
        "heldout_cameras": [cameras[index] for index in heldout_view_indices],
        "heldout_center_count": len(observed_centers),
        "observation_count": int(sum(len(value) for value in all_baseline_errors)),
        "baseline_mean_error_px": baseline_mean,
        "candidate_mean_error_px": candidate_mean,
        "mean_error_reduction_px": baseline_mean - candidate_mean,
        "baseline_mean_huber_loss_px2": baseline_huber,
        "candidate_mean_huber_loss_px2": candidate_huber,
        "huber_improvement_fraction": (
            (baseline_huber - candidate_huber) / max(baseline_huber, 1e-12)
        ),
        "cameras": camera_records,
    }


def correction_agreement_diagnostic(
    first_correction_m: np.ndarray, second_correction_m: np.ndarray
) -> dict[str, float]:
    """Measure signed field agreement between independently fitted updates."""

    first = np.asarray(first_correction_m, dtype=np.float64).reshape(-1)
    second = np.asarray(second_correction_m, dtype=np.float64).reshape(-1)
    _require(first.shape == second.shape, "correction field shape changed")
    first_norm = float(np.linalg.norm(first))
    second_norm = float(np.linalg.norm(second))
    denominator = first_norm * second_norm
    cosine = float(np.dot(first, second) / denominator) if denominator > 0.0 else -1.0
    first_rms = float(np.sqrt(np.mean(np.square(first))))
    second_rms = float(np.sqrt(np.mean(np.square(second))))
    disagreement_rms = float(np.sqrt(np.mean(np.square(first - second))))
    reference_rms = max(0.5 * (first_rms + second_rms), 1e-12)
    return {
        "cosine": cosine,
        "first_rms_m": first_rms,
        "second_rms_m": second_rms,
        "disagreement_rms_m": disagreement_rms,
        "disagreement_ratio": disagreement_rms / reference_rms,
    }


def _validation_passes(
    diagnostic: Mapping[str, Any], config: CrossViewGuardConfig
) -> bool:
    return bool(
        diagnostic.get("valid")
        and int(diagnostic["heldout_center_count"])
        >= config.minimum_heldout_center_count
        and float(diagnostic["huber_improvement_fraction"])
        >= config.minimum_heldout_improvement_fraction
        and float(diagnostic["mean_error_reduction_px"])
        >= config.minimum_heldout_mean_error_reduction_px
    )


def predict_crossview_guarded_candidate_arrays(
    baseline_m: np.ndarray,
    physical_response_m: np.ndarray,
    frame_zero_points_m: np.ndarray,
    action_support: np.ndarray,
    supplement: Mapping[str, np.ndarray],
    *,
    development_config: Deform360BiasAwareDevelopmentConfig | None = None,
    raw_config: RawCameraObservationConfig | None = None,
    guard_config: CrossViewGuardConfig | None = None,
) -> tuple[dict[str, Any], np.ndarray]:
    """Build a target-free update accepted only by disjoint camera evidence."""

    development = development_config or Deform360BiasAwareDevelopmentConfig()
    raw = raw_config or RawCameraObservationConfig(
        selected_camera_count=len(supplement["selected_cameras"]),
        update_frames=development.update_frames,
    )
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
    _require(
        raw.update_frames == development.update_frames,
        "raw and development update frames differ",
    )
    camera_names = tuple(str(value) for value in supplement["selected_cameras"])
    first_views, second_views = deterministic_camera_halves(camera_names)
    group_inputs: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    group_triangulation: list[dict[str, Any]] = []
    for views in (first_views, second_views):
        measurement, visibility, validity, reliability, variance, diagnostic = (
            _triangulate_group_measurement(
                baseline.shape,
                frame_zero,
                supplement,
                views,
                raw_config=raw,
                guard_config=guard,
                variance_floor_m2=development.observation_variance_floor_m2,
            )
        )
        group_inputs.append(
            (measurement, visibility, validity, reliability, variance)
        )
        group_triangulation.append(diagnostic)

    group_reports: list[dict[str, Any]] = []
    group_candidates: list[np.ndarray] = []
    for measurement, visibility, validity, reliability, variance in group_inputs:
        report, candidate = predict_bias_aware_candidate_arrays(
            baseline_input,
            response,
            frame_zero,
            support,
            measurement,
            visibility,
            validity,
            center_ids=np.asarray(supplement["center_ids"], dtype=np.int64),
            prior_reliability=reliability,
            observation_variance_m2=variance,
            config=development,
        )
        group_reports.append(report)
        group_candidates.append(candidate)

    guarded = baseline_input.copy()
    update_records: list[dict[str, Any]] = []
    accepted_count = 0
    for update_index, update_frame in enumerate(development.update_frames):
        stop = (
            development.update_frames[update_index + 1]
            if update_index + 1 < len(development.update_frames)
            else len(baseline)
        )
        first_record = group_reports[0]["updates"][update_index]
        second_record = group_reports[1]["updates"][update_index]
        record: dict[str, Any] = {
            "frame": update_frame,
            "interval_end_exclusive": stop,
            "candidate_available_in_both_groups": False,
            "accepted": False,
            "decision": "candidate-unavailable-exact-baseline-fallback",
            "bit_exact_baseline_fallback": True,
            "fit_group_a": first_views.tolist(),
            "fit_group_b": second_views.tolist(),
        }
        if first_record["candidate_available"] and second_record["candidate_available"]:
            record["candidate_available_in_both_groups"] = True
            first_correction = (
                np.asarray(group_candidates[0][update_frame + 1], dtype=np.float64)
                - baseline[update_frame + 1]
            )
            second_correction = (
                np.asarray(group_candidates[1][update_frame + 1], dtype=np.float64)
                - baseline[update_frame + 1]
            )
            validate_first_on_second = heldout_reprojection_diagnostic(
                baseline[update_frame],
                first_correction,
                supplement,
                update_index,
                second_views,
                huber_delta_px=guard.heldout_huber_delta_px,
            )
            validate_second_on_first = heldout_reprojection_diagnostic(
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
            first_pass = _validation_passes(validate_first_on_second, guard)
            second_pass = _validation_passes(validate_second_on_first, guard)
            agreement_pass = bool(
                agreement["cosine"] >= guard.minimum_correction_cosine
                and agreement["disagreement_ratio"]
                <= guard.maximum_correction_disagreement_ratio
            )
            record.update(
                {
                    "fit_a_validated_on_b": validate_first_on_second,
                    "fit_b_validated_on_a": validate_second_on_first,
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
                        "decision": "bidirectional-heldout-crossview-accepted",
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
                record["decision"] = "crossview-validation-exact-baseline-fallback"
        if not record["accepted"] and not np.array_equal(
            guarded[update_frame + 1 : stop],
            baseline_input[update_frame + 1 : stop],
        ):
            raise AssertionError("cross-view fallback changed the exact baseline")
        update_records.append(record)

    report = {
        "protocol_id": PROTOCOL_ID,
        "arm": "bias_aware_disjoint_crossview_guard",
        "development_config": asdict(development),
        "raw_camera_config": asdict(raw),
        "crossview_guard_config": asdict(guard),
        "camera_split": {
            "rule": "lexicographically sorted camera identities, interleaved",
            "group_a_indices": first_views.tolist(),
            "group_b_indices": second_views.tolist(),
            "group_a_cameras": [camera_names[index] for index in first_views],
            "group_b_cameras": [camera_names[index] for index in second_views],
        },
        "group_triangulation": group_triangulation,
        "group_candidate_reports": group_reports,
        "updates": update_records,
        "accepted_count": accepted_count,
        "exact_fallback_count": len(update_records) - accepted_count,
        "information_boundary": {
            "target_argument_accepted": False,
            "outcome_argument_accepted": False,
            "future_observation_read": False,
            "camera_fit_and_validation_groups_disjoint": True,
            "state_innovation_processed_once_per_fit_group": True,
            "prior_reliability_uses_state_innovation": False,
        },
        "claim_boundary": (
            "Cross-view validation can reject view-specific inconsistency but "
            "cannot identify bias coherent across every camera. This result is "
            "post-open method-development evidence only."
        ),
    }
    return report, guarded


__all__ = [
    "PROTOCOL_ID",
    "CrossViewGuardConfig",
    "conservative_triangulation_variance_m2",
    "correction_agreement_diagnostic",
    "deterministic_camera_halves",
    "heldout_reprojection_diagnostic",
    "predict_crossview_guarded_candidate_arrays",
]
