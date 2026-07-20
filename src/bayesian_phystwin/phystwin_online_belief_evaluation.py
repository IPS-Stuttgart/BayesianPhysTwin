"""Causal evaluation of recursive belief fields on PhysTwin trajectories."""

from __future__ import annotations

from dataclasses import asdict, replace
import hashlib
import json
import pickle
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .phystwin_official_evaluation import _nearest_distances
from .phystwin_online_belief import (
    RecursiveRbfBeliefConfig,
    decode_recursive_rbf_belief,
    deterministic_farthest_point_ids,
    finite_sample_absolute_residual_quantile_m,
    initialize_recursive_rbf_belief,
    robust_huber_continuation_gain,
    update_recursive_rbf_belief,
)


CONFORMAL_STYLE_COVERAGES = (0.50, 0.90, 0.95)
CONTINUATION_GAIN_THRESHOLD = 0.25


def _load_pickle(path: str | Path) -> Any:
    with Path(path).open("rb") as handle:
        return pickle.load(handle)


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _mean_or_none(values: list[float]) -> float | None:
    return None if not values else float(np.mean(values))


def select_stable_geometry_centers(
    object_points_m: np.ndarray,
    visibility: np.ndarray,
    motion_valid: np.ndarray,
    *,
    train_end_frame: int,
    center_count: int,
    minimum_training_availability_fraction: float,
    fallback_candidate_count: int,
) -> tuple[np.ndarray, dict[str, object]]:
    """Select centres from pre-test availability and frame-zero geometry only."""

    points = np.asarray(object_points_m, dtype=float)
    visible = np.asarray(visibility, dtype=bool)
    valid_motion = np.asarray(motion_valid, dtype=bool)
    if points.ndim != 3 or points.shape[2] != 3:
        raise ValueError("object_points_m must have shape (T, N, 3)")
    if visible.shape != points.shape[:2] or valid_motion.shape != points.shape[:2]:
        raise ValueError("visibility and motion_valid must match object points")
    if not 1 <= train_end_frame <= len(points):
        raise ValueError("train_end_frame must lie inside object points")
    if center_count < 1 or fallback_candidate_count < center_count:
        raise ValueError("fallback candidates must cover the centre count")
    if not 0.0 <= minimum_training_availability_fraction <= 1.0:
        raise ValueError("minimum availability must lie in [0, 1]")

    finite = np.all(np.isfinite(points), axis=2)
    finite_zero = finite[0]
    availability = np.mean(
        visible[:train_end_frame]
        & valid_motion[:train_end_frame]
        & finite[:train_end_frame],
        axis=0,
    )
    eligible = np.flatnonzero(
        finite_zero & (availability >= minimum_training_availability_fraction)
    )
    used_fallback = len(eligible) < center_count
    if used_fallback:
        candidates = np.flatnonzero(finite_zero)
        order = np.lexsort((candidates, -availability[candidates]))
        eligible = np.sort(
            candidates[order[: min(fallback_candidate_count, len(candidates))]]
        )
    if len(eligible) < center_count:
        raise ValueError("too few finite frame-zero points for centre selection")
    centers = deterministic_farthest_point_ids(points[0], eligible, center_count)
    return centers, {
        "candidate_count": int(len(eligible)),
        "used_fallback": used_fallback,
        "center_training_availability": availability[centers].tolist(),
        "minimum_center_training_availability": float(np.min(availability[centers])),
        "mean_center_training_availability": float(np.mean(availability[centers])),
    }


def _update_frames(
    train_end_frame: int,
    test_end_frame: int,
    fractions: tuple[float, ...],
) -> tuple[int, ...]:
    horizon = test_end_frame - train_end_frame
    if horizon < 3:
        raise ValueError("test horizon is too short for an online update")
    if any(not 0.0 < value < 1.0 for value in fractions):
        raise ValueError("update fractions must lie in (0, 1)")
    frames = sorted(
        {train_end_frame + max(1, round(horizon * fraction)) for fraction in fractions}
    )
    frames = [frame for frame in frames if train_end_frame < frame < test_end_frame - 1]
    if not frames:
        raise ValueError("update fractions leave no scored future frame")
    return tuple(frames)


def _score_frames(
    update_frames: tuple[int, ...], test_end_frame: int
) -> tuple[int, ...]:
    scored: list[int] = []
    for index, update in enumerate(update_frames):
        stop = (
            update_frames[index + 1]
            if index + 1 < len(update_frames)
            else test_end_frame
        )
        scored.extend(range(update + 1, stop))
    if not scored:
        raise ValueError("online protocol has no scored frame")
    return tuple(scored)


def _radial_residuals_m(residual_m: np.ndarray) -> np.ndarray:
    """Return radial residuals around the coordinate-wise median."""

    residual = np.asarray(residual_m, dtype=float)
    if residual.ndim != 2 or residual.shape[1] != 3 or not len(residual):
        raise ValueError("residual_m must have nonempty shape (N, 3)")
    if not np.all(np.isfinite(residual)):
        raise ValueError("residual_m must be finite")
    location = np.median(residual, axis=0)
    return np.linalg.norm(residual - location, axis=1)


def _residual_dispersion_m(residual_m: np.ndarray) -> float:
    """Return a robust radial spread around the coordinate-wise median."""

    return float(np.median(_radial_residuals_m(residual_m)))


def _manual_track_indices(
    initial_prediction: np.ndarray,
    manual_tracks: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray] | None:
    if manual_tracks is None:
        return None
    tracks = np.asarray(manual_tracks, dtype=float)
    if tracks.ndim != 3 or tracks.shape[2] != 3:
        raise ValueError("manual tracks must have shape (T, K, 3)")
    initially_valid = np.all(np.isfinite(tracks[0]), axis=1)
    _, indices = _nearest_distances(
        initial_prediction,
        tracks[0, initially_valid],
        p=2,
    )
    return initially_valid, indices


def _score_trajectory(
    trajectory_m: np.ndarray,
    object_points_m: np.ndarray,
    visibility: np.ndarray,
    motion_valid: np.ndarray,
    manual_tracks_m: np.ndarray | None,
    *,
    surface_point_count: int,
    center_ids: np.ndarray,
    scored_frames: tuple[int, ...],
) -> dict[str, object]:
    trajectory = np.asarray(trajectory_m, dtype=float)
    points = np.asarray(object_points_m, dtype=float)
    visible = np.asarray(visibility, dtype=bool)
    valid_motion = np.asarray(motion_valid, dtype=bool)
    if trajectory.ndim != 3 or trajectory.shape[2] != 3:
        raise ValueError("trajectory_m must have shape (T, N, 3)")
    if points.ndim != 3 or points.shape[2] != 3:
        raise ValueError("object_points_m must have shape (T, M, 3)")
    if trajectory.shape[1] < points.shape[1]:
        raise ValueError("trajectory does not cover tracked object-point identities")
    if surface_point_count > trajectory.shape[1]:
        raise ValueError("surface point count exceeds trajectory")

    noncenter = np.ones(points.shape[1], dtype=bool)
    noncenter[np.asarray(center_ids, dtype=np.int64)] = False
    predicted_surface_ids = np.arange(surface_point_count, dtype=np.int64)
    predicted_surface_ids = predicted_surface_ids[
        ~np.isin(predicted_surface_ids, np.asarray(center_ids, dtype=np.int64))
    ]
    if not len(predicted_surface_ids):
        raise ValueError("centre exclusion leaves no predicted surface point")
    track_mapping = _manual_track_indices(trajectory[0], manual_tracks_m)
    tracks = (
        None if manual_tracks_m is None else np.asarray(manual_tracks_m, dtype=float)
    )
    chamfer_by_frame: list[float] = []
    noncenter_by_frame: list[float] = []
    noncenter_frames: list[int] = []
    track_by_frame: list[float] = []
    valid_noncenter_counts: list[int] = []

    for frame in scored_frames:
        observed_mask = (
            noncenter & visible[frame] & np.all(np.isfinite(points[frame]), axis=1)
        )
        observed = points[frame, observed_mask]
        predicted_surface = trajectory[frame, predicted_surface_ids]
        predicted_surface = predicted_surface[
            np.all(np.isfinite(predicted_surface), axis=1)
        ]
        if not len(observed) or not len(predicted_surface):
            raise ValueError(f"no hidden Chamfer support at frame {frame}")
        distance, _ = _nearest_distances(predicted_surface, observed, p=1)
        chamfer_by_frame.append(float(np.mean(distance)))

        identity_mask = (
            noncenter
            & visible[frame]
            & valid_motion[frame]
            & np.all(np.isfinite(points[frame]), axis=1)
        )
        identity_residual = (
            trajectory[frame, : points.shape[1]][identity_mask]
            - points[frame, identity_mask]
        )
        valid_noncenter_counts.append(int(np.sum(identity_mask)))
        if len(identity_residual):
            noncenter_frames.append(frame)
            noncenter_by_frame.append(
                float(np.mean(np.linalg.norm(identity_residual, axis=1)))
            )

        if track_mapping is not None and tracks is not None:
            initially_valid, indices = track_mapping
            current = tracks[frame, initially_valid]
            current_valid = np.all(np.isfinite(current), axis=1) & ~np.isin(
                indices,
                np.asarray(center_ids, dtype=np.int64),
            )
            track_by_frame.append(
                0.0
                if not np.any(current_valid)
                else float(
                    np.mean(
                        np.linalg.norm(
                            trajectory[frame, indices][current_valid]
                            - current[current_valid],
                            axis=1,
                        )
                    )
                )
            )

    if not noncenter_by_frame:
        raise ValueError("case has no supported non-centre identity frame")
    late_start = (2 * len(scored_frames)) // 3
    late_frame_threshold = scored_frames[late_start]
    late_noncenter = [
        value
        for frame, value in zip(noncenter_frames, noncenter_by_frame, strict=True)
        if frame >= late_frame_threshold
    ]
    if not late_noncenter:
        raise ValueError("case has no supported late non-centre identity frame")
    return {
        "frame_count": len(scored_frames),
        "scored_frames": list(scored_frames),
        "identity_frame_count": len(noncenter_frames),
        "identity_scored_frames": noncenter_frames,
        "future_chamfer_distance_m": float(np.mean(chamfer_by_frame)),
        "future_noncenter_point_error_m": float(np.mean(noncenter_by_frame)),
        "late_horizon_chamfer_distance_m": float(
            np.mean(chamfer_by_frame[late_start:])
        ),
        "late_horizon_noncenter_point_error_m": float(np.mean(late_noncenter)),
        "manual_track_error_m": _mean_or_none(track_by_frame),
        "valid_noncenter_points_per_frame": {
            "minimum": int(np.min(valid_noncenter_counts)),
            "mean": float(np.mean(valid_noncenter_counts)),
            "maximum": int(np.max(valid_noncenter_counts)),
        },
        "by_frame": {
            "chamfer_distance_m": chamfer_by_frame,
            "noncenter_point_error_m": noncenter_by_frame,
            "manual_track_error_m": track_by_frame or None,
        },
    }


def evaluate_online_belief_case(
    case_dir: str | Path,
    *,
    baseline_filename: str,
    measurement_policy: Mapping[str, object],
    belief_config: RecursiveRbfBeliefConfig,
    manual_track_filename: str = "gt_track_3d.pkl",
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    """Evaluate one case without exposing any post-update target to prediction."""

    directory = Path(case_dir)
    final_data_path = directory / "final_data.pkl"
    split_path = directory / "split.json"
    baseline_path = directory / baseline_filename
    for path in (final_data_path, split_path, baseline_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    data = _load_pickle(final_data_path)
    baseline = np.asarray(_load_pickle(baseline_path), dtype=float)
    split = json.loads(split_path.read_text(encoding="utf-8"))
    points = np.asarray(data["object_points"], dtype=float)
    visible = np.asarray(data["object_visibilities"], dtype=bool)
    motion_valid = np.asarray(data["object_motions_valid"], dtype=bool)
    train_end = int(split["test"][0])
    test_end = min(int(split["test"][1]), len(points), len(baseline))
    baseline = baseline[:test_end]
    points = points[:test_end]
    visible = visible[:test_end]
    motion_valid = motion_valid[:test_end]
    if baseline.shape[1] < points.shape[1]:
        raise ValueError("baseline has fewer nodes than tracked object points")

    centers, selection = select_stable_geometry_centers(
        points,
        visible,
        motion_valid,
        train_end_frame=train_end,
        center_count=int(measurement_policy["center_count"]),
        minimum_training_availability_fraction=float(
            measurement_policy["minimum_training_availability_fraction"]
        ),
        fallback_candidate_count=int(measurement_policy["fallback_candidate_count"]),
    )
    fractions = tuple(map(float, measurement_policy["update_fractions"]))
    updates = _update_frames(train_end, test_end, fractions)
    scored = _score_frames(updates, test_end)
    minimum_update_center_count = int(
        measurement_policy.get("minimum_update_center_count", 0)
    )
    if not 0 <= minimum_update_center_count <= len(centers):
        raise ValueError(
            "minimum_update_center_count must lie between zero and center_count"
        )
    dispersion_policy = measurement_policy.get("residual_dispersion_gate")
    dispersion_threshold_m: float | None = None
    minimum_current_inlier_count: int | None = None
    history_dispersion_reference_m: float | None = None
    history_dispersion_frame_count = 0
    if dispersion_policy is not None:
        minimum_history_center_count = int(
            dispersion_policy["minimum_history_center_count"]
        )
        history_quantile = float(dispersion_policy["history_quantile"])
        history_multiplier = float(dispersion_policy["history_multiplier"])
        minimum_threshold_m = float(dispersion_policy["minimum_threshold_m"])
        raw_minimum_current_inliers = dispersion_policy.get(
            "minimum_current_inlier_count"
        )
        minimum_current_inlier_count = (
            None
            if raw_minimum_current_inliers is None
            else int(raw_minimum_current_inliers)
        )
        if not 1 <= minimum_history_center_count <= len(centers):
            raise ValueError(
                "minimum_history_center_count must lie between one and center_count"
            )
        if not 0.0 <= history_quantile <= 1.0:
            raise ValueError("history_quantile must lie in [0, 1]")
        if not np.isfinite(history_multiplier) or history_multiplier <= 0.0:
            raise ValueError("history_multiplier must be positive")
        if not np.isfinite(minimum_threshold_m) or minimum_threshold_m <= 0.0:
            raise ValueError("minimum_threshold_m must be positive")
        if minimum_current_inlier_count is not None and not (
            1 <= minimum_current_inlier_count <= len(centers)
        ):
            raise ValueError(
                "minimum_current_inlier_count must lie between one and center_count"
            )
        history_dispersions: list[float] = []
        for frame in range(train_end):
            history_available = (
                visible[frame, centers]
                & motion_valid[frame, centers]
                & np.all(np.isfinite(points[frame, centers]), axis=1)
                & np.all(np.isfinite(baseline[frame, centers]), axis=1)
            )
            if int(np.sum(history_available)) < minimum_history_center_count:
                continue
            history_residual = (
                points[frame, centers[history_available]]
                - baseline[frame, centers[history_available]]
            )
            history_dispersions.append(_residual_dispersion_m(history_residual))
        if not history_dispersions:
            raise ValueError("no supported pre-test residual-dispersion frame")
        history_dispersion_frame_count = len(history_dispersions)
        history_dispersion_reference_m = float(
            np.quantile(history_dispersions, history_quantile)
        )
        dispersion_threshold_m = max(
            minimum_threshold_m,
            history_multiplier * history_dispersion_reference_m,
        )

    belief = initialize_recursive_rbf_belief(
        centers,
        baseline[0, centers],
        points[0],
        config=belief_config,
    )
    field_trajectory = baseline.copy()
    global_trajectory = baseline.copy()
    causal_continuation_trajectory = baseline.copy()
    frozen_current_trajectory = baseline.copy()
    field_variance = np.full_like(field_trajectory, np.nan, dtype=float)
    conformal_half_width = {
        coverage: np.full(len(field_trajectory), np.nan, dtype=np.float32)
        for coverage in CONFORMAL_STYLE_COVERAGES
    }
    update_records: list[dict[str, object]] = []
    global_config = replace(belief_config, local_blend=0.0)
    last_trusted_observation_frame = train_end - 1

    for update_index, update in enumerate(updates):
        available = (
            visible[update, centers]
            & motion_valid[update, centers]
            & np.all(np.isfinite(points[update, centers]), axis=1)
            & np.all(np.isfinite(baseline[update, centers]), axis=1)
        )
        available_center_count = int(np.sum(available))
        residual = np.full((len(centers), 3), np.nan, dtype=float)
        residual[available] = (
            points[update, centers[available]] - baseline[update, centers[available]]
        )
        residual_dispersion_m = (
            None
            if not available_center_count
            else _residual_dispersion_m(residual[available])
        )
        residual_radial_m = (
            np.empty(0, dtype=float)
            if not available_center_count
            else _radial_residuals_m(residual[available])
        )
        current_inlier_count = (
            None
            if dispersion_threshold_m is None
            else int(np.sum(residual_radial_m <= dispersion_threshold_m))
        )
        has_support = available_center_count >= minimum_update_center_count
        has_median_coherence = dispersion_threshold_m is None or (
            residual_dispersion_m is not None
            and residual_dispersion_m <= dispersion_threshold_m
        )
        has_inlier_support = minimum_current_inlier_count is None or (
            current_inlier_count is not None
            and current_inlier_count >= minimum_current_inlier_count
        )
        has_coherent_residual = has_median_coherence and has_inlier_support
        accepted = has_support and has_coherent_residual
        stop = (
            updates[update_index + 1] if update_index + 1 < len(updates) else test_end
        )
        reliability = np.zeros(len(centers), dtype=float)
        interval_half_widths: dict[str, float] | None = None
        continuation_gain: float | None = None
        continuation_selected: bool | None = None
        continuation_support_count = 0
        previous_trusted_observation_frame: int | None = None
        if accepted:
            previous_trusted_observation_frame = last_trusted_observation_frame
            continuation_support = (
                available
                & visible[last_trusted_observation_frame, centers]
                & motion_valid[last_trusted_observation_frame, centers]
                & np.all(
                    np.isfinite(points[last_trusted_observation_frame, centers]),
                    axis=1,
                )
                & np.all(
                    np.isfinite(baseline[last_trusted_observation_frame, centers]),
                    axis=1,
                )
            )
            continuation_support_count = int(np.sum(continuation_support))
            continuation_ids = centers[continuation_support]
            continuation_gain = robust_huber_continuation_gain(
                baseline[update, continuation_ids]
                - baseline[last_trusted_observation_frame, continuation_ids],
                points[update, continuation_ids]
                - points[last_trusted_observation_frame, continuation_ids],
                fallback=0.0,
            )
            continuation_selected = continuation_gain > CONTINUATION_GAIN_THRESHOLD
            if available_center_count:
                interval_half_widths = {
                    f"{coverage:.2f}": finite_sample_absolute_residual_quantile_m(
                        residual,
                        available,
                        coverage,
                    )
                    for coverage in CONFORMAL_STYLE_COVERAGES
                }
            belief, reliability = update_recursive_rbf_belief(
                belief,
                update,
                baseline[update, centers],
                residual,
                available,
                config=belief_config,
            )
            for frame in range(update + 1, stop):
                field = decode_recursive_rbf_belief(
                    belief,
                    baseline[update],
                    forecast_frames=frame - update,
                    config=belief_config,
                )
                global_field = decode_recursive_rbf_belief(
                    belief,
                    baseline[update],
                    forecast_frames=frame - update,
                    config=global_config,
                )
                field_trajectory[frame] += field.mean_m
                global_trajectory[frame] += global_field.mean_m
                corrected_current = baseline[update] + field.mean_m
                frozen_current_trajectory[frame] = corrected_current
                causal_continuation_trajectory[frame] = (
                    corrected_current + baseline[frame] - baseline[update]
                    if continuation_selected
                    else corrected_current
                )
                field_variance[frame] = field.variance_m2
                if interval_half_widths is not None:
                    for coverage in CONFORMAL_STYLE_COVERAGES:
                        conformal_half_width[coverage][frame] = interval_half_widths[
                            f"{coverage:.2f}"
                        ]
            last_trusted_observation_frame = update
        else:
            for name, trajectory in {
                "recursive_rbf_belief": field_trajectory,
                "recursive_global_translation": global_trajectory,
                "recursive_rbf_causal_continuation": (causal_continuation_trajectory),
                "risk_limited_frozen_current_state": frozen_current_trajectory,
            }.items():
                if not np.array_equal(
                    trajectory[update + 1 : stop], baseline[update + 1 : stop]
                ):
                    raise AssertionError(f"{name} violated exact open-loop fallback")
        selected_reliability = reliability[available]
        update_records.append(
            {
                "frame": update,
                "available_center_count": available_center_count,
                "available_center_ids": centers[available].tolist(),
                "minimum_update_center_count": minimum_update_center_count,
                "residual_dispersion_m": residual_dispersion_m,
                "maximum_residual_dispersion_m": dispersion_threshold_m,
                "current_residual_inlier_count": current_inlier_count,
                "minimum_current_residual_inlier_count": (minimum_current_inlier_count),
                "accepted": accepted,
                "conformal_style_absolute_residual_half_width_m": (
                    interval_half_widths
                ),
                "previous_trusted_observation_frame": (
                    previous_trusted_observation_frame
                ),
                "continuation_projection_center_count": (continuation_support_count),
                "causal_huber_continuation_gain": continuation_gain,
                "causal_continuation_gain_threshold": (CONTINUATION_GAIN_THRESHOLD),
                "causal_continuation_selected": continuation_selected,
                "decision": (
                    "accepted"
                    if accepted
                    else (
                        "insufficient_support_exact_fallback"
                        if not has_support
                        else "incoherent_residual_exact_fallback"
                    )
                ),
                "mean_reliability": (
                    None
                    if not accepted or not len(selected_reliability)
                    else float(np.mean(selected_reliability))
                ),
                "minimum_reliability": (
                    None
                    if not accepted or not len(selected_reliability)
                    else float(np.min(selected_reliability))
                ),
                "global_mean_m": belief.global_mean_m.tolist(),
                "global_std_m": np.sqrt(belief.global_variance_m2).tolist(),
            }
        )

    manual_track_path = directory / manual_track_filename
    manual_tracks = (
        np.asarray(_load_pickle(manual_track_path), dtype=float)[:test_end]
        if manual_track_path.is_file()
        else None
    )
    surface_count = points.shape[1] + len(np.asarray(data["surface_points"]))
    trajectories = {
        "open_loop": baseline,
        "recursive_global_translation": global_trajectory,
        "recursive_rbf_belief": field_trajectory,
        "recursive_rbf_causal_continuation": causal_continuation_trajectory,
        "risk_limited_frozen_current_state": frozen_current_trajectory,
    }
    scores = {
        name: _score_trajectory(
            trajectory,
            points,
            visible,
            motion_valid,
            manual_tracks,
            surface_point_count=surface_count,
            center_ids=centers,
            scored_frames=scored,
        )
        for name, trajectory in trajectories.items()
    }
    report: dict[str, object] = {
        "case": directory.name,
        "split": {"train_end_frame": train_end, "test_end_frame": test_end},
        "center_ids": centers.tolist(),
        "selection": selection,
        "risk_gate": {
            "minimum_update_center_count": minimum_update_center_count,
            "residual_dispersion": (
                None
                if dispersion_policy is None
                else {
                    "policy": dict(dispersion_policy),
                    "history_frame_count": history_dispersion_frame_count,
                    "history_reference_m": history_dispersion_reference_m,
                    "maximum_update_dispersion_m": dispersion_threshold_m,
                    "minimum_current_inlier_count": minimum_current_inlier_count,
                }
            ),
            "rejected_update_behavior": "exact open-loop fallback for the interval",
        },
        "update_frames": list(updates),
        "updates": update_records,
        "scores": scores,
        "metric_contract": {
            "assimilation_center_exclusion": (
                "all selected centre identities are excluded from identity error, "
                "manual-track error, and both the observed and predicted Chamfer sets"
            )
        },
        "causal_continuation_contract": {
            "hybrid": (
                "corrected current state plus selected physical-prior future "
                "displacement"
            ),
            "gain": (
                "Huber-IRLS scalar projection of currently observed sparse "
                "displacement onto physical-prior displacement since the previous "
                "accepted observation"
            ),
            "threshold": CONTINUATION_GAIN_THRESHOLD,
            "decision": "continue iff gain is strictly greater than threshold",
            "insufficient_support_gain": 0.0,
            "status": (
                "post-hoc development arm; threshold requires newly held-out transfer"
            ),
        },
        "uncertainty_contract": {
            "raw_variance": (
                "coordinate-wise marginal decoder variance; not calibrated"
            ),
            "conformal_style_interval": (
                "recursive mean plus/minus the finite-sample absolute-residual "
                "quantile from currently available centre coordinates"
            ),
            "finite_sample_rank": "min(n, ceil((n + 1) * nominal_coverage))",
            "nominal_coverages": list(CONFORMAL_STYLE_COVERAGES),
            "dependence_warning": (
                "coordinates share frames and material identities; this is a "
                "conformal-style diagnostic, not a formal iid coverage guarantee"
            ),
            "rejected_interval": "no interval; exact open-loop fallback",
        },
        "inputs": {
            "final_data": {
                "path": str(final_data_path),
                "sha256": _sha256(final_data_path),
            },
            "split": {"path": str(split_path), "sha256": _sha256(split_path)},
            "baseline": {"path": str(baseline_path), "sha256": _sha256(baseline_path)},
            "manual_tracks": (
                None
                if not manual_track_path.is_file()
                else {
                    "path": str(manual_track_path),
                    "sha256": _sha256(manual_track_path),
                }
            ),
        },
    }
    arrays = {
        "center_ids": centers,
        "field_trajectory_m": field_trajectory.astype(np.float32),
        "global_trajectory_m": global_trajectory.astype(np.float32),
        "causal_continuation_trajectory_m": causal_continuation_trajectory.astype(
            np.float32
        ),
        "risk_limited_frozen_current_state_m": frozen_current_trajectory.astype(
            np.float32
        ),
        "field_variance_m2": field_variance.astype(np.float32),
        "field_conformal_q50_half_width_m": conformal_half_width[0.50],
        "field_conformal_q90_half_width_m": conformal_half_width[0.90],
        "field_conformal_q95_half_width_m": conformal_half_width[0.95],
    }
    return report, arrays


def _paired_cluster_bootstrap(
    differences: Mapping[str, float],
    groups: Mapping[str, str],
    *,
    draws: int,
    seed: int,
) -> dict[str, float]:
    group_names = tuple(sorted(set(groups.values())))
    cases_by_group = {
        group: tuple(case for case, value in groups.items() if value == group)
        for group in group_names
    }
    rng = np.random.default_rng(seed)
    samples = np.empty(draws, dtype=float)
    for draw in range(draws):
        selected_groups = rng.choice(group_names, size=len(group_names), replace=True)
        selected_cases = [
            case for group in selected_groups for case in cases_by_group[str(group)]
        ]
        samples[draw] = np.mean([differences[case] for case in selected_cases])
    return {
        "difference": float(np.mean(list(differences.values()))),
        "lower_95": float(np.quantile(samples, 0.025)),
        "upper_95": float(np.quantile(samples, 0.975)),
        "probability_improved": float(np.mean(samples < 0.0)),
    }


def evaluate_online_belief_cohort(
    protocol_path: str | Path,
    output_dir: str | Path,
) -> dict[str, object]:
    """Run the frozen cohort and write checksummed per-case artifacts."""

    protocol_file = Path(protocol_path)
    protocol = json.loads(protocol_file.read_text(encoding="utf-8"))
    cohort = protocol["confirmation_cohort"]
    root = Path(cohort["case_root"])
    groups = dict(cohort["physical_object_groups"])
    belief_config = RecursiveRbfBeliefConfig(**protocol["belief"])
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    case_reports: list[dict[str, object]] = []
    artifact_records: list[dict[str, object]] = []
    for case in sorted(groups):
        report, arrays = evaluate_online_belief_case(
            root / case,
            baseline_filename=str(cohort["baseline_filename"]),
            measurement_policy=protocol["measurement_policy"],
            belief_config=belief_config,
        )
        case_json = output / f"{case}.json"
        case_npz = output / f"{case}.npz"
        case_json.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        np.savez_compressed(case_npz, **arrays)
        artifact_records.append(
            {
                "case": case,
                "report_sha256": _sha256(case_json),
                "arrays_sha256": _sha256(case_npz),
            }
        )
        case_reports.append(report)

    arms = tuple(protocol["fixed_arms"])
    metrics = tuple(protocol["primary_metrics"])
    aggregate: dict[str, dict[str, float]] = {}
    for arm in arms:
        aggregate[arm] = {
            metric: float(
                np.mean([case["scores"][arm][metric] for case in case_reports])
            )
            for metric in metrics
        }
    comparisons: dict[str, object] = {}
    bootstrap = protocol["aggregation"]
    candidate_comparators: dict[str, tuple[str, ...]] = {
        "recursive_rbf_belief": ("open_loop", "recursive_global_translation")
    }
    if "recursive_rbf_causal_continuation" in arms:
        candidate_comparators["recursive_rbf_causal_continuation"] = tuple(
            comparator
            for comparator in (
                "open_loop",
                "recursive_global_translation",
                "recursive_rbf_belief",
                "risk_limited_frozen_current_state",
            )
            if comparator in arms
        )
    for candidate, comparators in candidate_comparators.items():
        for comparator in comparators:
            comparison_metrics: dict[str, object] = {}
            for metric in metrics:
                differences = {
                    case["case"]: float(
                        case["scores"][candidate][metric]
                        - case["scores"][comparator][metric]
                    )
                    for case in case_reports
                }
                interval = _paired_cluster_bootstrap(
                    differences,
                    groups,
                    draws=int(bootstrap["bootstrap_draws"]),
                    seed=int(bootstrap["bootstrap_seed"]),
                )
                baseline_mean = aggregate[comparator][metric]
                candidate_mean = aggregate[candidate][metric]
                interval["relative_change"] = candidate_mean / baseline_mean - 1.0
                interval["case_wins"] = int(
                    np.sum(np.asarray(list(differences.values())) < 0.0)
                )
                interval["maximum_case_relative_regression"] = float(
                    max(
                        (
                            case["scores"][candidate][metric]
                            / case["scores"][comparator][metric]
                            - 1.0
                        )
                        for case in case_reports
                    )
                )
                comparison_metrics[metric] = interval
            comparisons[f"{candidate}_vs_{comparator}"] = comparison_metrics

    gate_config = protocol["confirmation_gate"]
    candidate_arm = str(gate_config.get("candidate_arm", "recursive_rbf_belief"))
    if candidate_arm not in arms:
        raise ValueError(
            f"confirmation-gate candidate is not a fixed arm: {candidate_arm}"
        )
    primary_key = f"{candidate_arm}_vs_open_loop"
    global_key = f"{candidate_arm}_vs_recursive_global_translation"
    if primary_key not in comparisons or global_key not in comparisons:
        raise ValueError(
            "confirmation-gate candidate lacks required open-loop/global comparisons"
        )
    primary = comparisons[primary_key]
    field_vs_global = comparisons[global_key]
    joint_wins = sum(
        all(
            case["scores"][candidate_arm][metric] < case["scores"]["open_loop"][metric]
            for metric in metrics
        )
        for case in case_reports
    )
    conditions = {
        "minimum_primary_improvements": all(
            primary[metric]["relative_change"]
            <= -float(
                gate_config[
                    "minimum_relative_improvement_over_open_loop_each_primary_metric"
                ]
            )
            for metric in metrics
        ),
        "minimum_joint_case_wins": joint_wins
        >= int(gate_config["minimum_two_metric_case_wins"]),
        "maximum_case_regression": all(
            primary[metric]["maximum_case_relative_regression"]
            <= float(gate_config["maximum_case_regression_each_primary_metric"])
            for metric in metrics
        ),
        "paired_intervals_exclude_zero": all(
            primary[metric]["upper_95"] < 0.0 for metric in metrics
        ),
        "field_beats_global_noncenter": field_vs_global[
            "future_noncenter_point_error_m"
        ]["relative_change"]
        <= -float(gate_config["minimum_noncenter_improvement_over_global_translation"]),
    }
    summary: dict[str, object] = {
        "schema_version": 1,
        "protocol_id": protocol["protocol_id"],
        "protocol": {
            "path": str(protocol_file.resolve()),
            "sha256": _sha256(protocol_file),
        },
        "belief_config": asdict(belief_config),
        "case_count": len(case_reports),
        "physical_object_group_count": len(set(groups.values())),
        "aggregate": aggregate,
        "comparisons": comparisons,
        "joint_two_metric_case_wins": joint_wins,
        "gate": {
            "candidate_arm": candidate_arm,
            "passed": all(conditions.values()),
            "conditions": conditions,
        },
        "artifacts": artifact_records,
        "claim_boundary": protocol["claim_boundary"],
    }
    summary_path = output / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary
