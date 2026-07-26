#!/usr/bin/env python3
"""Post-open headroom audit for sparse-prefix updates on raw MatPhys replays.

This script is deliberately not a confirmation runner. It evaluates an already
opened cohort to answer one model-class question: can a causally selected
MatPhys spring replay and a larger graph discrepancy posterior jointly cross
the published MatPhys CD/track operating point? Manual prefix tracks are an
explicit observation-quality upper bound and are never presented as a
deployable input.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from bayesian_phystwin.phystwin_additional_bayesian_confirmation import (
    FIXED_INITIAL_STD_M,
    FIXED_INLIER_PRIOR,
    FIXED_OBSERVATION_STD_M,
    FIXED_OUTLIER_VARIANCE_MULTIPLIER,
    FIXED_PROCESS_STD_M,
)
from bayesian_phystwin.phystwin_bayesian_anchor import (
    RobustEndpointPosterior,
    robust_random_walk_endpoint,
)
from bayesian_phystwin.phystwin_bias_aware_ray import (
    decide_prefix_admission,
    remove_affine_ray_nuisance,
)
from bayesian_phystwin.phystwin_comparison import official_metrics_by_frame
from bayesian_phystwin.phystwin_cotracker3_cues import (
    infer_cotracker3_ray_discrepancy,
    load_cotracker3_multiview_depth_observations,
    load_cotracker3_multiview_observations,
)
from bayesian_phystwin.phystwin_directional_endpoint import (
    DirectionalEndpointPosterior,
    robust_directional_endpoint,
)
from bayesian_phystwin.phystwin_graph import (
    PhysTwinSpringGraphConfig,
    build_phystwin_spring_graph,
)
from bayesian_phystwin.phystwin_graph_discrepancy import (
    graph_smoothed_discrepancy_posterior,
    normalized_spring_laplacian,
)
from bayesian_phystwin.phystwin_multiview_tangent_fusion import (
    fuse_source_normal_multiview_tangent,
    local_surface_tangent_projectors,
)
from bayesian_phystwin.phystwin_official_evaluation import _nearest_distances
from bayesian_phystwin.phystwin_online_belief import (
    RecursiveRbfBeliefConfig,
    decode_recursive_rbf_belief,
    deterministic_farthest_point_ids,
    initialize_recursive_rbf_belief,
    update_recursive_rbf_belief,
)
from bayesian_phystwin.phystwin_planar_discrepancy import (
    fit_canonical_planar_discrepancy,
)
from bayesian_phystwin.phystwin_residual_dynamics import (
    _clip_residual,
    _target_validity,
)


@dataclass(frozen=True)
class HeadroomConfig:
    baseline_kind: str = "raw_matphys_replay"
    observation_source: str = "final_data"
    manual_prefix_override: bool = True
    cotracker_minimum_quality: float = 0.5
    cotracker_maximum_cycle_error_px: float = 5.0
    cotracker_maximum_reprojection_error_px: float = 3.0
    cotracker_maximum_view_disagreement_m: float = 0.01
    cotracker_minimum_camera_count: int = 3
    multiview_priority_minimum_availability_fraction: float = 0.10
    multiview_tangent_neighbor_count: int = 16
    ray_window_frames: int = 31
    ray_minimum_camera_count: int = 3
    ray_pixel_noise_std: float = 2.0
    ray_prior_std_m: float = 0.05
    ray_degrees_of_freedom: float = 4.0
    ray_robust_iterations: int = 3
    ray_graph_prior_strength: float = 1e-4
    ray_correction_scale: float = 1.0
    ray_maximum_residual_m: float = 0.02
    ray_minimum_observed_fraction: float = 0.02
    ray_minimum_inlier_probability: float = 0.20
    ray_minimum_absolute_prefix_improvement_m: float = 0.0001
    ray_minimum_relative_prefix_improvement: float = 0.01
    prior_strengths: tuple[float, ...] = (0.003, 0.01, 0.03, 0.1)
    maximum_residuals_m: tuple[float, ...] = (0.02, 0.04, 0.06, 0.1)
    dense_correction_scales: tuple[float, ...] = (1.0,)
    nearest_cloud_windows: tuple[int, ...] = ()
    relative_cap_quantile: float = 0.95
    relative_cap_multipliers: tuple[float, ...] = (1.0, 2.0, 4.0, 8.0)
    temporal_gamma_candidates: tuple[float, ...] = (0.0, 0.25, 0.5, 1.0)
    rbf_center_counts: tuple[int, ...] = (16, 32, 64)
    rbf_minimum_availability_fraction: float = 0.5
    sparse_graph_center_counts: tuple[int, ...] = (8, 16, 32, 64)
    sparse_graph_minimum_availability_fraction: float = 0.25
    planar_degrees: tuple[int, ...] = (0, 1, 2)
    planar_ridge_strength: float = 1e-3
    process_std_m: float = FIXED_PROCESS_STD_M
    observation_std_m: float = FIXED_OBSERVATION_STD_M
    initial_std_m: float = FIXED_INITIAL_STD_M
    inlier_prior: float = FIXED_INLIER_PRIOR
    outlier_variance_multiplier: float = FIXED_OUTLIER_VARIANCE_MULTIPLIER
    published_sota_chamfer_m: float = 0.008
    published_sota_track_m: float = 0.015


def _load_pickle(path: Path) -> Any:
    with path.open("rb") as handle:
        return pickle.load(handle)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _metric_means(metrics: dict[str, np.ndarray]) -> dict[str, float]:
    return {name: float(np.mean(values)) for name, values in metrics.items()}


def _candidate_id(
    method: str,
    *,
    prior_strength: float | None = None,
    maximum_residual_m: float,
    correction_scale: float | None = None,
) -> str:
    cap_mm = int(round(1000.0 * maximum_residual_m))
    scale = (
        ""
        if correction_scale is None
        else "__scale_" + format(correction_scale, ".12g").replace(".", "p")
    )
    if prior_strength is None:
        return f"{method}{scale}__cap_{cap_mm:03d}mm"
    strength = format(prior_strength, ".12g").replace(".", "p")
    return f"{method}__lambda_{strength}{scale}__cap_{cap_mm:03d}mm"


def _endpoint(
    residual: np.ndarray,
    valid: np.ndarray,
    *,
    end_frame: int,
    config: HeadroomConfig,
) -> RobustEndpointPosterior:
    return robust_random_walk_endpoint(
        residual,
        valid,
        end_frame=end_frame,
        process_variance=config.process_std_m**2,
        observation_variance=config.observation_std_m**2,
        initial_variance=config.initial_std_m**2,
        inlier_prior=config.inlier_prior,
        outlier_variance_multiplier=config.outlier_variance_multiplier,
    )


def _merge_manual_endpoint(
    means: np.ndarray,
    variances: np.ndarray,
    observed: np.ndarray,
    manual_indices: np.ndarray,
    manual: RobustEndpointPosterior,
) -> None:
    for vertex in np.unique(manual_indices):
        rows = (manual_indices == vertex) & (manual.update_count > 0)
        if not np.any(rows):
            continue
        precision = 1.0 / np.maximum(manual.variance[rows], 1e-12)
        means[vertex] = np.average(manual.mean[rows], axis=0, weights=precision)
        variances[vertex] = 1.0 / float(np.sum(precision))
        observed[vertex] = True


def _override_manual_nodes(
    correction: np.ndarray,
    manual_indices: np.ndarray,
    manual: RobustEndpointPosterior,
) -> np.ndarray:
    result = np.asarray(correction, dtype=float).copy()
    for vertex in np.unique(manual_indices):
        rows = (manual_indices == vertex) & (manual.update_count > 0)
        if not np.any(rows):
            continue
        precision = 1.0 / np.maximum(manual.variance[rows], 1e-12)
        result[vertex] = np.average(
            manual.mean[rows],
            axis=0,
            weights=precision,
        )
    return result


def _nearest_cloud_corrections(
    baseline: np.ndarray,
    observed: np.ndarray,
    visible: np.ndarray,
    *,
    end_frame: int,
    windows: tuple[int, ...],
    num_surface_points: int,
) -> dict[int, np.ndarray]:
    if not windows:
        return {}
    maximum_window = max(windows)
    start_frame = max(0, end_frame - maximum_window)
    by_frame: list[np.ndarray] = []
    for frame in range(start_frame, end_frame):
        target = observed[frame, visible[frame]]
        if len(target) == 0:
            continue
        _, nearest = _nearest_distances(
            target,
            baseline[frame, :num_surface_points],
            p=1,
        )
        by_frame.append(
            target[nearest] - baseline[frame, :num_surface_points]
        )
    if not by_frame:
        raise ValueError("nearest-cloud endpoint has no visible prefix frame")
    stacked = np.stack(by_frame)
    result: dict[int, np.ndarray] = {}
    for window in windows:
        correction = np.zeros((baseline.shape[1], 3), dtype=float)
        correction[:num_surface_points] = np.median(
            stacked[-min(window, len(stacked)) :],
            axis=0,
        )
        result[window] = correction
    return result


def _lift_cotracker_source_depth(
    cues_path: Path,
    raw_case_dir: Path,
    initial_world_points: np.ndarray,
    *,
    train_end: int,
    minimum_quality: float,
    maximum_cycle_error_px: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    with np.load(cues_path) as archive:
        cues = {name: np.asarray(archive[name]) for name in archive.files}
    tracks = np.asarray(cues["source_tracks_xy"][:train_end], dtype=float)
    source_camera = np.asarray(cues["source_camera"], dtype=np.int64)
    quality_key = (
        "source_quality_probability"
        if "source_quality_probability" in cues
        else "cotracker_quality_probability"
    )
    quality = np.asarray(cues[quality_key][:train_end], dtype=float)
    cycle_error = np.asarray(
        cues["forward_backward_error_px"][:train_end],
        dtype=float,
    )
    cycle_valid = np.asarray(
        cues["forward_backward_valid"][:train_end],
        dtype=bool,
    )
    boundary = np.asarray(cues["boundary_distance"][:train_end], dtype=float)
    available = np.asarray(cues["cue_available"][:train_end], dtype=bool)
    if tracks.shape[:2] != quality.shape or tracks.shape[2] != 2:
        raise ValueError("CoTracker cue shapes are inconsistent")
    if len(source_camera) != tracks.shape[1]:
        raise ValueError("source_camera does not match CoTracker tracks")
    metadata = json.loads(
        (raw_case_dir / "metadata.json").read_text(encoding="utf-8")
    )
    intrinsics = np.asarray(metadata["intrinsics"], dtype=float)
    camera_to_world = np.asarray(
        _load_pickle(raw_case_dir / "calibrate.pkl"),
        dtype=float,
    )
    world = np.full((train_end, tracks.shape[1], 3), np.nan, dtype=float)
    depth_valid = np.zeros(tracks.shape[:2], dtype=bool)
    for camera in range(len(intrinsics)):
        selected = np.flatnonzero(source_camera == camera)
        if len(selected) == 0:
            continue
        inverse_intrinsic = np.linalg.inv(intrinsics[camera])
        for frame in range(train_end):
            depth = np.asarray(
                np.load(raw_case_dir / "depth" / str(camera) / f"{frame}.npy"),
                dtype=float,
            )
            xy = tracks[frame, selected]
            pixels = np.rint(xy).astype(np.int64)
            inside = (
                np.isfinite(xy).all(axis=1)
                & (pixels[:, 0] >= 0)
                & (pixels[:, 0] < depth.shape[1])
                & (pixels[:, 1] >= 0)
                & (pixels[:, 1] < depth.shape[0])
            )
            local = np.flatnonzero(inside)
            if len(local) == 0:
                continue
            z = depth[pixels[local, 1], pixels[local, 0]] / 1000.0
            positive = np.isfinite(z) & (z > 0.0)
            local = local[positive]
            z = z[positive]
            if len(local) == 0:
                continue
            homogeneous_pixels = np.column_stack(
                (xy[local], np.ones(len(local)))
            )
            camera_points = (
                homogeneous_pixels @ inverse_intrinsic.T
            ) * z[:, None]
            homogeneous_camera = np.column_stack(
                (camera_points, np.ones(len(camera_points)))
            )
            world_points = homogeneous_camera @ camera_to_world[camera].T
            target = selected[local]
            world[frame, target] = world_points[:, :3]
            depth_valid[frame, target] = True

    initial_valid = depth_valid[0] & np.isfinite(world[0]).all(axis=1)
    anchored = np.zeros_like(world)
    anchored[:, initial_valid] = (
        initial_world_points[initial_valid][None]
        + world[:, initial_valid]
        - world[0, initial_valid][None]
    )
    valid = (
        depth_valid
        & initial_valid[None]
        & available
        & (quality >= minimum_quality)
        & cycle_valid
        & (cycle_error <= maximum_cycle_error_px)
        & (boundary > 0.0)
        & np.isfinite(anchored).all(axis=2)
    )
    anchored[~valid] = 0.0
    return anchored, valid, {
        "quality_field": quality_key,
        "initial_depth_valid_fraction": float(np.mean(initial_valid)),
        "prefix_valid_fraction": float(np.mean(valid)),
        "median_quality": float(np.median(quality[valid])),
        "median_cycle_error_px": float(np.median(cycle_error[valid])),
    }


def _load_cotracker_multiview(
    cues_path: Path,
    initial_world_points: np.ndarray,
    *,
    train_end: int,
    minimum_quality: float,
    maximum_cycle_error_px: float,
    maximum_reprojection_error_px: float,
    minimum_camera_count: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    observations = load_cotracker3_multiview_observations(
        cues_path,
        initial_world_points,
        train_end_frame=train_end,
        minimum_view_quality=minimum_quality,
        maximum_reprojection_error_px=maximum_reprojection_error_px,
        maximum_cycle_error_px=maximum_cycle_error_px,
        minimum_camera_count=minimum_camera_count,
    )
    valid = observations.valid
    points = np.where(valid[:, :, None], observations.points_world_m, 0.0)
    summary = {
        "prefix_valid_fraction": float(np.mean(valid)),
        "minimum_camera_count": float(minimum_camera_count),
        "median_camera_count": float(np.median(observations.camera_count[valid])),
        "median_minimum_view_quality": float(
            np.median(observations.minimum_view_quality[valid])
        ),
        "median_reprojection_error_px": float(
            np.median(observations.reprojection_error_px[valid])
        ),
    }
    return points, valid, summary


def _load_cotracker_multiview_depth(
    cues_path: Path,
    raw_case_dir: Path,
    initial_world_points: np.ndarray,
    *,
    train_end: int,
    minimum_quality: float,
    maximum_cycle_error_px: float,
    maximum_view_disagreement_m: float,
    minimum_camera_count: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    observations = load_cotracker3_multiview_depth_observations(
        cues_path,
        raw_case_dir,
        initial_world_points,
        train_end_frame=train_end,
        minimum_view_quality=minimum_quality,
        maximum_view_disagreement_m=maximum_view_disagreement_m,
        maximum_cycle_error_px=maximum_cycle_error_px,
        minimum_camera_count=minimum_camera_count,
    )
    valid = observations.valid
    points = np.where(valid[:, :, None], observations.points_world_m, 0.0)
    summary = {
        "prefix_valid_fraction": float(np.mean(valid)),
        "minimum_camera_count": float(minimum_camera_count),
        "maximum_view_disagreement_m": maximum_view_disagreement_m,
        "median_camera_count": float(np.median(observations.camera_count[valid])),
        "median_minimum_view_quality": float(
            np.median(observations.minimum_view_quality[valid])
        ),
        "median_view_disagreement_m": float(
            np.median(observations.view_disagreement_m[valid])
        ),
    }
    return points, valid, summary


class InsufficientRbfCentersError(ValueError):
    """Raised when the frozen availability gate leaves too few RBF centers."""


def _recursive_rbf_correction(
    observation_points: np.ndarray,
    observation_valid: np.ndarray,
    baseline: np.ndarray,
    initial_structure_points: np.ndarray,
    *,
    fit_end_frame: int,
    query_start_frame: int,
    query_end_frame: int,
    original_count: int,
    center_count: int,
    minimum_availability_fraction: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not 0.0 <= minimum_availability_fraction <= 1.0:
        raise ValueError("RBF minimum availability must lie in [0, 1]")
    availability = np.mean(
        observation_valid[:fit_end_frame, :original_count],
        axis=0,
    )
    eligible = np.flatnonzero(
        np.all(np.isfinite(initial_structure_points[:original_count]), axis=1)
        & (availability >= minimum_availability_fraction)
    )
    if len(eligible) < center_count:
        raise InsufficientRbfCentersError(
            f"only {len(eligible)} RBF centers meet the availability gate"
        )
    center_ids = deterministic_farthest_point_ids(
        initial_structure_points[:original_count],
        eligible,
        center_count,
    )
    belief_config = RecursiveRbfBeliefConfig()
    belief = initialize_recursive_rbf_belief(
        center_ids,
        initial_structure_points[center_ids],
        initial_structure_points,
        config=belief_config,
    )
    for frame in range(fit_end_frame):
        available = observation_valid[frame, center_ids]
        center_positions = np.where(
            available[:, None],
            observation_points[frame, center_ids],
            belief.center_positions_m,
        )
        measured_residual = np.where(
            available[:, None],
            observation_points[frame, center_ids] - baseline[frame, center_ids],
            0.0,
        )
        belief, _ = update_recursive_rbf_belief(
            belief,
            frame,
            center_positions,
            measured_residual,
            available,
            config=belief_config,
        )
    correction = np.empty(
        (query_end_frame - query_start_frame, baseline.shape[1], 3),
        dtype=float,
    )
    for output_index, frame in enumerate(
        range(query_start_frame, query_end_frame)
    ):
        correction[output_index] = decode_recursive_rbf_belief(
            belief,
            baseline[frame],
            forecast_frames=max(0, frame - fit_end_frame + 1),
            config=belief_config,
        ).mean_m
    return correction, center_ids, availability[center_ids]


def _recursive_rbf_refit_or_fallback(
    observation_points: np.ndarray,
    observation_valid: np.ndarray,
    baseline: np.ndarray,
    initial_structure_points: np.ndarray,
    *,
    fit_end_frame: int,
    query_start_frame: int,
    query_end_frame: int,
    original_count: int,
    center_count: int,
    minimum_availability_fraction: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str | None]:
    try:
        correction, center_ids, availability = _recursive_rbf_correction(
            observation_points,
            observation_valid,
            baseline,
            initial_structure_points,
            fit_end_frame=fit_end_frame,
            query_start_frame=query_start_frame,
            query_end_frame=query_end_frame,
            original_count=original_count,
            center_count=center_count,
            minimum_availability_fraction=minimum_availability_fraction,
        )
    except InsufficientRbfCentersError as error:
        return (
            np.zeros(
                (
                    query_end_frame - query_start_frame,
                    baseline.shape[1],
                    3,
                ),
                dtype=float,
            ),
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=float),
            str(error),
        )
    return correction, center_ids, availability, None


def _canonical_planar_correction(
    structure_points: np.ndarray,
    endpoint: RobustEndpointPosterior,
    *,
    original_count: int,
    degree: int,
    ridge_strength: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    model = fit_canonical_planar_discrepancy(
        structure_points,
        np.arange(original_count, dtype=np.int64),
        endpoint.mean,
        endpoint.variance,
        endpoint.update_count > 0,
        degree=degree,
        ridge_strength=ridge_strength,
    )
    return model.predict(structure_points), {
        "degree": degree,
        "fit_count": model.fit_count,
        "robust_scale_m": model.robust_scale_m,
        "basis": model.basis.tolist(),
        "coordinate_scale_m": model.coordinate_scale.tolist(),
    }


def _sparse_graph_correction(
    structure_points: np.ndarray,
    endpoint: RobustEndpointPosterior,
    laplacian: Any,
    *,
    original_count: int,
    end_frame: int,
    center_count: int,
    minimum_availability_fraction: float,
    prior_strength: float,
    initial_variance: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if end_frame < 1:
        raise ValueError("end_frame must be positive")
    if not 0.0 <= minimum_availability_fraction <= 1.0:
        raise ValueError("minimum availability must lie in [0, 1]")
    availability = endpoint.update_count / end_frame
    eligible = np.flatnonzero(
        np.all(np.isfinite(structure_points[:original_count]), axis=1)
        & (availability >= minimum_availability_fraction)
    )
    if len(eligible) < center_count:
        raise ValueError(
            f"only {len(eligible)} graph identities meet the availability gate"
        )
    center_ids = deterministic_farthest_point_ids(
        structure_points[:original_count],
        eligible,
        center_count,
    )
    mean = np.zeros((len(structure_points), 3), dtype=float)
    variance = np.full(len(structure_points), initial_variance, dtype=float)
    observed = np.zeros(len(structure_points), dtype=bool)
    mean[center_ids] = endpoint.mean[center_ids]
    variance[center_ids] = endpoint.variance[center_ids]
    observed[center_ids] = True
    correction = graph_smoothed_discrepancy_posterior(
        mean,
        variance,
        observed,
        laplacian,
        prior_strength=prior_strength,
    ).mean
    return correction, center_ids, availability[center_ids]


def _evaluate(
    baseline: np.ndarray,
    correction: np.ndarray,
    *,
    start_frame: int,
    end_frame: int,
    observed: np.ndarray,
    visible: np.ndarray,
    manual_tracks: np.ndarray,
    num_surface_points: int,
) -> dict[str, float]:
    candidate = np.asarray(baseline, dtype=float).copy()
    values = np.asarray(correction, dtype=float)
    if values.ndim == 2:
        candidate[start_frame:end_frame] += values[None]
    elif values.shape == candidate[start_frame:end_frame].shape:
        candidate[start_frame:end_frame] += values
    else:
        raise ValueError("correction must be static (N, 3) or match the interval")
    return _metric_means(
        official_metrics_by_frame(
            candidate,
            observed,
            visible,
            manual_tracks,
            num_surface_points=num_surface_points,
            start_frame=start_frame,
            end_frame=end_frame,
        )
    )


def _evaluate_chamfer_only(
    baseline: np.ndarray,
    correction: np.ndarray,
    *,
    start_frame: int,
    end_frame: int,
    observed: np.ndarray,
    visible: np.ndarray,
    num_surface_points: int,
) -> float:
    candidate = np.asarray(baseline, dtype=float).copy()
    values = np.asarray(correction, dtype=float)
    if values.ndim == 2:
        candidate[start_frame:end_frame] += values[None]
    elif values.shape == candidate[start_frame:end_frame].shape:
        candidate[start_frame:end_frame] += values
    else:
        raise ValueError("correction must be static (N, 3) or match the interval")
    by_frame = []
    for frame in range(start_frame, end_frame):
        target = observed[frame, visible[frame]]
        distance, _ = _nearest_distances(
            candidate[frame, :num_surface_points],
            target,
            p=1,
        )
        by_frame.append(float(np.mean(distance)))
    return float(np.mean(by_frame))


def _selection_score(
    candidate: dict[str, float],
    baseline: dict[str, float],
    *,
    chamfer_weight: float = 0.5,
) -> float:
    if not 0.0 <= chamfer_weight <= 1.0:
        raise ValueError("chamfer_weight must lie in [0, 1]")
    if chamfer_weight == 1.0:
        return candidate["chamfer_distance_m"] / baseline["chamfer_distance_m"]
    if chamfer_weight == 0.0:
        return candidate["track_error_m"] / baseline["track_error_m"]
    return (
        chamfer_weight
        * candidate["chamfer_distance_m"]
        / baseline["chamfer_distance_m"]
        + (1.0 - chamfer_weight)
        * candidate["track_error_m"]
        / baseline["track_error_m"]
    )


def _full_endpoint_arrays(
    endpoint: RobustEndpointPosterior | DirectionalEndpointPosterior,
    *,
    state_count: int,
    original_count: int,
    initial_variance: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = np.zeros((state_count, 3), dtype=float)
    variance = np.full(state_count, initial_variance, dtype=float)
    observed = np.zeros(state_count, dtype=bool)
    mean[:original_count] = endpoint.mean
    variance[:original_count] = endpoint.variance
    observed[:original_count] = endpoint.update_count > 0
    return mean, variance, observed


def _selected_raw_entry(
    case: str,
    selection: dict[str, Any],
    family_entries: dict[str, dict[str, dict[str, Any]]],
) -> tuple[str, dict[str, Any]]:
    selected_family = str(selection["case_results"][case]["selected_family"])
    try:
        entry = family_entries[selected_family][case]
    except KeyError as error:
        raise ValueError(
            f"selected raw family entry is unavailable for {case}: {selected_family}"
        ) from error
    return selected_family, entry


def _run_case(job: tuple[Any, ...]) -> tuple[str, dict[str, Any]]:
    (
        case,
        case_root,
        selected_family,
        baseline_entry,
        selected_within_family_method,
        fit_end,
        cues_path,
        raw_case_dir,
        config,
    ) = job
    case_dir = Path(case_root)
    baseline_path = Path(str(baseline_entry["trajectory"]))
    expected_hash = str(baseline_entry["sha256"])
    actual_hash = _sha256(baseline_path)
    if actual_hash != expected_hash:
        raise ValueError(f"MatPhys baseline hash mismatch for {case}")

    split = json.loads((case_dir / "split.json").read_text(encoding="utf-8"))
    train_end = int(split["train"][1])
    future_end = int(split["test"][1])
    data = _load_pickle(case_dir / "final_data.pkl")
    baseline = np.asarray(_load_pickle(baseline_path), dtype=float)[:future_end]
    observed_points = np.asarray(data["object_points"], dtype=float)[:future_end]
    visible = np.asarray(data["object_visibilities"], dtype=bool)[:future_end]
    motion_valid = np.asarray(data["object_motions_valid"], dtype=bool)[
        : max(future_end - 1, 0)
    ]
    manual_tracks = np.asarray(
        _load_pickle(case_dir / "gt_track_3d.pkl"),
        dtype=float,
    )[:future_end]
    original_count = observed_points.shape[1]
    surface_points = np.asarray(data["surface_points"], dtype=float)
    interior_points = np.asarray(data["interior_points"], dtype=float)
    structure_points = np.concatenate(
        (observed_points[0], surface_points, interior_points),
        axis=0,
    )
    if baseline.shape != (future_end, len(structure_points), 3):
        raise ValueError(
            f"raw MatPhys replay shape disagrees with {case} structure: "
            f"{baseline.shape} versus {(future_end, len(structure_points), 3)}"
        )

    optimal = _load_pickle(case_dir / "optimal_params.pkl")
    graph = build_phystwin_spring_graph(
        structure_points,
        None,
        config=PhysTwinSpringGraphConfig(
            object_radius=float(optimal["object_radius"]),
            object_max_neighbours=int(optimal["object_max_neighbours"]),
            controller_radius=float(optimal["controller_radius"]),
            controller_max_neighbours=int(optimal["controller_max_neighbours"]),
        ),
    )
    springs = graph.springs[: graph.num_object_springs]
    laplacian = normalized_spring_laplacian(len(structure_points), springs)

    cotracker_summary = None
    directional_endpoint_inputs = None
    ray_bias_aware = (
        config.observation_source
        == "alltracker_multiview_ray_bias_aware"
    )
    ray_endpoint_diagnostics: dict[int, dict[str, Any]] = {}
    if config.observation_source == "final_data":
        inference_points = observed_points
        dense_valid = _target_validity(visible, motion_valid)
    elif config.observation_source in {
        "cotracker3_source_depth",
        "alltracker_source_depth",
    }:
        inference_prefix, valid_prefix, cotracker_summary = (
            _lift_cotracker_source_depth(
                Path(cues_path),
                Path(raw_case_dir),
                observed_points[0],
                train_end=train_end,
                minimum_quality=config.cotracker_minimum_quality,
                maximum_cycle_error_px=config.cotracker_maximum_cycle_error_px,
            )
        )
        inference_points = np.zeros_like(observed_points)
        inference_points[:train_end] = inference_prefix
        dense_valid = np.zeros_like(visible)
        dense_valid[:train_end] = valid_prefix
    elif config.observation_source == "cotracker3_multiview":
        inference_prefix, valid_prefix, cotracker_summary = (
            _load_cotracker_multiview(
                Path(cues_path),
                observed_points[0],
                train_end=train_end,
                minimum_quality=config.cotracker_minimum_quality,
                maximum_cycle_error_px=(
                    config.cotracker_maximum_cycle_error_px
                ),
                maximum_reprojection_error_px=(
                    config.cotracker_maximum_reprojection_error_px
                ),
                minimum_camera_count=config.cotracker_minimum_camera_count,
            )
        )
        inference_points = np.zeros_like(observed_points)
        inference_points[:train_end] = inference_prefix
        dense_valid = np.zeros_like(visible)
        dense_valid[:train_end] = valid_prefix
    elif config.observation_source == "cotracker3_multiview_depth":
        inference_prefix, valid_prefix, cotracker_summary = (
            _load_cotracker_multiview_depth(
                Path(cues_path),
                Path(raw_case_dir),
                observed_points[0],
                train_end=train_end,
                minimum_quality=config.cotracker_minimum_quality,
                maximum_cycle_error_px=(
                    config.cotracker_maximum_cycle_error_px
                ),
                maximum_view_disagreement_m=(
                    config.cotracker_maximum_view_disagreement_m
                ),
                minimum_camera_count=config.cotracker_minimum_camera_count,
            )
        )
        inference_points = np.zeros_like(observed_points)
        inference_points[:train_end] = inference_prefix
        dense_valid = np.zeros_like(visible)
        dense_valid[:train_end] = valid_prefix
    elif config.observation_source == "cotracker3_hybrid":
        source_points, source_valid, source_summary = _lift_cotracker_source_depth(
            Path(cues_path),
            Path(raw_case_dir),
            observed_points[0],
            train_end=train_end,
            minimum_quality=config.cotracker_minimum_quality,
            maximum_cycle_error_px=config.cotracker_maximum_cycle_error_px,
        )
        multiview_points, multiview_valid, multiview_summary = (
            _load_cotracker_multiview(
                Path(cues_path),
                observed_points[0],
                train_end=train_end,
                minimum_quality=config.cotracker_minimum_quality,
                maximum_cycle_error_px=(
                    config.cotracker_maximum_cycle_error_px
                ),
                maximum_reprojection_error_px=(
                    config.cotracker_maximum_reprojection_error_px
                ),
                minimum_camera_count=config.cotracker_minimum_camera_count,
            )
        )
        inference_prefix = source_points.copy()
        inference_prefix[multiview_valid] = multiview_points[multiview_valid]
        valid_prefix = source_valid | multiview_valid
        inference_points = np.zeros_like(observed_points)
        inference_points[:train_end] = inference_prefix
        dense_valid = np.zeros_like(visible)
        dense_valid[:train_end] = valid_prefix
        cotracker_summary = {
            "source_depth": source_summary,
            "multiview": multiview_summary,
            "multiview_replacement_fraction": float(
                np.mean(multiview_valid & source_valid)
            ),
            "multiview_added_fraction": float(
                np.mean(multiview_valid & ~source_valid)
            ),
        }
    elif config.observation_source == "cotracker3_multiview_priority":
        source_points, source_valid, source_summary = _lift_cotracker_source_depth(
            Path(cues_path),
            Path(raw_case_dir),
            observed_points[0],
            train_end=train_end,
            minimum_quality=config.cotracker_minimum_quality,
            maximum_cycle_error_px=config.cotracker_maximum_cycle_error_px,
        )
        multiview_points, multiview_valid, multiview_summary = (
            _load_cotracker_multiview(
                Path(cues_path),
                observed_points[0],
                train_end=train_end,
                minimum_quality=config.cotracker_minimum_quality,
                maximum_cycle_error_px=(
                    config.cotracker_maximum_cycle_error_px
                ),
                maximum_reprojection_error_px=(
                    config.cotracker_maximum_reprojection_error_px
                ),
                minimum_camera_count=config.cotracker_minimum_camera_count,
            )
        )
        multiview_availability = np.mean(multiview_valid, axis=0)
        priority_identities = (
            multiview_availability
            >= config.multiview_priority_minimum_availability_fraction
        )
        inference_prefix = source_points.copy()
        valid_prefix = source_valid.copy()
        inference_prefix[:, priority_identities] = multiview_points[
            :, priority_identities
        ]
        valid_prefix[:, priority_identities] = multiview_valid[
            :, priority_identities
        ]
        inference_points = np.zeros_like(observed_points)
        inference_points[:train_end] = inference_prefix
        dense_valid = np.zeros_like(visible)
        dense_valid[:train_end] = valid_prefix
        cotracker_summary = {
            "source_depth": source_summary,
            "multiview": multiview_summary,
            "priority_rule": (
                "an identity with sufficient three-view prefix availability "
                "uses only three-view observations; all others exactly retain "
                "the source-depth channel"
            ),
            "minimum_multiview_availability_fraction": (
                config.multiview_priority_minimum_availability_fraction
            ),
            "priority_identity_count": int(np.sum(priority_identities)),
            "priority_identity_fraction": float(
                np.mean(priority_identities)
            ),
            "priority_availability_median": (
                None
                if not np.any(priority_identities)
                else float(
                    np.median(
                        multiview_availability[priority_identities]
                    )
                )
            ),
        }
    elif (
        config.observation_source
        == "cotracker3_multiview_tangent_priority"
    ):
        source_points, source_valid, source_summary = _lift_cotracker_source_depth(
            Path(cues_path),
            Path(raw_case_dir),
            observed_points[0],
            train_end=train_end,
            minimum_quality=config.cotracker_minimum_quality,
            maximum_cycle_error_px=config.cotracker_maximum_cycle_error_px,
        )
        multiview_points, multiview_valid, multiview_summary = (
            _load_cotracker_multiview(
                Path(cues_path),
                observed_points[0],
                train_end=train_end,
                minimum_quality=config.cotracker_minimum_quality,
                maximum_cycle_error_px=(
                    config.cotracker_maximum_cycle_error_px
                ),
                maximum_reprojection_error_px=(
                    config.cotracker_maximum_reprojection_error_px
                ),
                minimum_camera_count=config.cotracker_minimum_camera_count,
            )
        )
        tangent_fusion = fuse_source_normal_multiview_tangent(
            source_points,
            source_valid,
            multiview_points,
            multiview_valid,
            observed_points[0],
            minimum_multiview_availability_fraction=(
                config.multiview_priority_minimum_availability_fraction
            ),
            neighbor_count=config.multiview_tangent_neighbor_count,
        )
        inference_prefix = tangent_fusion.points_world_m
        valid_prefix = tangent_fusion.valid
        inference_points = np.zeros_like(observed_points)
        inference_points[:train_end] = inference_prefix
        dense_valid = np.zeros_like(visible)
        dense_valid[:train_end] = valid_prefix
        cotracker_summary = {
            "source_depth": source_summary,
            "multiview": multiview_summary,
            "priority_rule": (
                "identities with sufficient three-view prefix availability "
                "admit only the multiview correction tangent to the initial "
                "surface; source depth retains normal authority and exact "
                "observation support"
            ),
            "minimum_multiview_availability_fraction": (
                config.multiview_priority_minimum_availability_fraction
            ),
            "tangent_neighbor_count": (
                config.multiview_tangent_neighbor_count
            ),
            "priority_identity_count": int(
                np.sum(tangent_fusion.priority_identities)
            ),
            "priority_identity_fraction": float(
                np.mean(tangent_fusion.priority_identities)
            ),
            "fused_update_count": int(
                np.sum(tangent_fusion.fused_update)
            ),
            "fused_update_fraction_of_source_support": float(
                np.sum(tangent_fusion.fused_update)
                / max(np.sum(source_valid), 1)
            ),
            "source_support_preserved_exactly": bool(
                np.array_equal(valid_prefix, source_valid)
            ),
        }
    elif (
        config.observation_source
        == "cotracker3_multiview_directional_priority"
    ):
        source_points, source_valid, source_summary = _lift_cotracker_source_depth(
            Path(cues_path),
            Path(raw_case_dir),
            observed_points[0],
            train_end=train_end,
            minimum_quality=config.cotracker_minimum_quality,
            maximum_cycle_error_px=config.cotracker_maximum_cycle_error_px,
        )
        multiview_points, multiview_valid, multiview_summary = (
            _load_cotracker_multiview(
                Path(cues_path),
                observed_points[0],
                train_end=train_end,
                minimum_quality=config.cotracker_minimum_quality,
                maximum_cycle_error_px=(
                    config.cotracker_maximum_cycle_error_px
                ),
                maximum_reprojection_error_px=(
                    config.cotracker_maximum_reprojection_error_px
                ),
                minimum_camera_count=config.cotracker_minimum_camera_count,
            )
        )
        multiview_availability = np.mean(multiview_valid, axis=0)
        priority_identities = (
            multiview_availability
            >= config.multiview_priority_minimum_availability_fraction
        )
        tangent_projectors = local_surface_tangent_projectors(
            observed_points[0],
            neighbor_count=config.multiview_tangent_neighbor_count,
        )
        directional_endpoint_inputs = {
            "source_residual": (
                source_points - baseline[:train_end, :original_count]
            ),
            "source_valid": source_valid,
            "multiview_residual": (
                multiview_points - baseline[:train_end, :original_count]
            ),
            "multiview_valid": multiview_valid,
            "tangent_projectors": tangent_projectors,
            "priority_identities": priority_identities,
        }
        # Non-endpoint diagnostics retain the exact source channel. The primary
        # dense endpoint correction below uses the directional posterior.
        inference_points = np.zeros_like(observed_points)
        inference_points[:train_end] = source_points
        dense_valid = np.zeros_like(visible)
        dense_valid[:train_end] = source_valid
        cotracker_summary = {
            "source_depth": source_summary,
            "multiview": multiview_summary,
            "priority_rule": (
                "nonpriority identities use the existing full source update; "
                "priority identities use source normal and redundant-view "
                "tangent innovations as orthogonal robust measurements"
            ),
            "minimum_multiview_availability_fraction": (
                config.multiview_priority_minimum_availability_fraction
            ),
            "tangent_neighbor_count": (
                config.multiview_tangent_neighbor_count
            ),
            "priority_identity_count": int(np.sum(priority_identities)),
            "priority_identity_fraction": float(
                np.mean(priority_identities)
            ),
            "source_update_count": int(np.sum(source_valid)),
            "multiview_tangent_update_count": int(
                np.sum(multiview_valid[:, priority_identities])
            ),
            "multiview_tangent_updates_without_source_count": int(
                np.sum(
                    multiview_valid[:, priority_identities]
                    & ~source_valid[:, priority_identities]
                )
            ),
            "scalar_graph_variance_rule": (
                "largest endpoint covariance eigenvalue in square metres"
            ),
        }
    elif ray_bias_aware:
        inference_points = np.zeros_like(observed_points)
        dense_valid = np.zeros_like(visible)
        cotracker_summary = {
            "method": (
                "three-view robust ray posterior relative to the physical "
                "baseline; shared affine field is retained as nuisance "
                "variance rather than applied as a state correction"
            ),
            "window_frames": config.ray_window_frames,
            "minimum_camera_count": config.ray_minimum_camera_count,
            "minimum_view_quality": config.cotracker_minimum_quality,
            "maximum_cycle_error_px": (
                config.cotracker_maximum_cycle_error_px
            ),
            "pixel_noise_std": config.ray_pixel_noise_std,
            "prior_std_m": config.ray_prior_std_m,
            "degrees_of_freedom": config.ray_degrees_of_freedom,
            "robust_iterations": config.ray_robust_iterations,
            "unknown_correlation_rule": (
                "temporal effective sample size and conservative cross-view "
                "precision averaging inside the ray posterior"
            ),
        }
    else:
        raise ValueError(
            f"unsupported observation source: {config.observation_source}"
        )
    dense_residual = (
        inference_points - baseline[:, :original_count]
    )

    def endpoint_at(end_frame: int):
        if ray_bias_aware:
            ray = infer_cotracker3_ray_discrepancy(
                Path(cues_path),
                baseline[:train_end, :original_count],
                end_frame=end_frame,
                window_frames=config.ray_window_frames,
                minimum_view_quality=config.cotracker_minimum_quality,
                maximum_cycle_error_px=(
                    config.cotracker_maximum_cycle_error_px
                ),
                minimum_camera_count=config.ray_minimum_camera_count,
                pixel_noise_std=config.ray_pixel_noise_std,
                prior_std_m=config.ray_prior_std_m,
                degrees_of_freedom=config.ray_degrees_of_freedom,
                robust_iterations=config.ray_robust_iterations,
            )
            bias_aware = remove_affine_ray_nuisance(
                ray,
                baseline[end_frame - 1, :original_count],
                unobserved_variance_m2=config.ray_prior_std_m**2,
            )
            diagnostic = asdict(bias_aware.diagnostics)
            diagnostic["coefficients"] = (
                bias_aware.diagnostics.coefficients.tolist()
            )
            diagnostic["observed_fraction"] = float(
                np.mean(ray.observed)
            )
            diagnostic["median_inlier_probability"] = (
                0.0
                if not np.any(ray.observed)
                else float(
                    np.median(
                        ray.final_inlier_probability[ray.observed]
                    )
                )
            )
            ray_endpoint_diagnostics[end_frame] = diagnostic
            return bias_aware.posterior
        if directional_endpoint_inputs is None:
            return _endpoint(
                dense_residual,
                dense_valid,
                end_frame=end_frame,
                config=config,
            )
        return robust_directional_endpoint(
            **directional_endpoint_inputs,
            end_frame=end_frame,
            process_variance=config.process_std_m**2,
            observation_variance=config.observation_std_m**2,
            initial_variance=config.initial_std_m**2,
            inlier_prior=config.inlier_prior,
            outlier_variance_multiplier=(
                config.outlier_variance_multiplier
            ),
        )

    dense_endpoint = endpoint_at(train_end)
    full_mean, full_variance, full_observed = _full_endpoint_arrays(
        dense_endpoint,
        state_count=len(structure_points),
        original_count=original_count,
        initial_variance=config.initial_std_m**2,
    )
    if not 2 < fit_end < train_end:
        raise ValueError(f"invalid fit/validation split for {case}")

    fit_dense_endpoint = endpoint_at(fit_end)
    fit_full_mean, fit_full_variance, fit_full_observed = (
        _full_endpoint_arrays(
            fit_dense_endpoint,
            state_count=len(structure_points),
            original_count=original_count,
            initial_variance=config.initial_std_m**2,
        )
    )
    inner_end = max(3, fit_end - (train_end - fit_end))
    if not inner_end < fit_end:
        raise ValueError(f"no inner temporal-selection interval for {case}")
    inner_dense_endpoint = endpoint_at(inner_end)
    inner_full_mean, _, _ = _full_endpoint_arrays(
        inner_dense_endpoint,
        state_count=len(structure_points),
        original_count=original_count,
        initial_variance=config.initial_std_m**2,
    )

    if config.manual_prefix_override:
        manual_initial = np.isfinite(manual_tracks[0]).all(axis=1)
        initial_match_m, manual_indices = _nearest_distances(
            baseline[0],
            manual_tracks[0, manual_initial],
            p=2,
        )
        manual_values = manual_tracks[:, manual_initial]
        manual_valid = np.isfinite(manual_values).all(axis=2)
        manual_residual = np.zeros_like(manual_values)
        baseline_at_manual = baseline[:, manual_indices]
        manual_residual[manual_valid] = (
            manual_values[manual_valid] - baseline_at_manual[manual_valid]
        )
    else:
        manual_initial = np.zeros(manual_tracks.shape[1], dtype=bool)
        initial_match_m = np.empty(0, dtype=float)
        manual_indices = np.empty(0, dtype=np.int64)
        manual_valid = np.zeros((future_end, 0), dtype=bool)
        manual_residual = np.zeros((future_end, 0, 3), dtype=float)
    manual_endpoint = _endpoint(
        manual_residual,
        manual_valid,
        end_frame=train_end,
        config=config,
    )
    fit_manual_endpoint = _endpoint(
        manual_residual,
        manual_valid,
        end_frame=fit_end,
        config=config,
    )

    combined_mean = full_mean.copy()
    combined_variance = full_variance.copy()
    combined_observed = full_observed.copy()
    _merge_manual_endpoint(
        combined_mean,
        combined_variance,
        combined_observed,
        manual_indices,
        manual_endpoint,
    )

    num_surface_points = original_count + len(surface_points)
    baseline_metrics = _metric_means(
        official_metrics_by_frame(
            baseline,
            observed_points,
            visible,
            manual_tracks,
            num_surface_points=num_surface_points,
            start_frame=train_end,
            end_frame=future_end,
        )
    )
    candidate_metrics: dict[str, dict[str, float]] = {}
    baseline_validation = _metric_means(
        official_metrics_by_frame(
            baseline,
            observed_points,
            visible,
            manual_tracks,
            num_surface_points=num_surface_points,
            start_frame=fit_end,
            end_frame=train_end,
        )
    )
    validation_candidates: list[dict[str, Any]] = []
    for cap in config.maximum_residuals_m:
        for scale in config.dense_correction_scales:
            fit_correction = _override_manual_nodes(
                scale * fit_full_mean,
                manual_indices,
                fit_manual_endpoint,
            )
            fit_correction = _clip_residual(fit_correction[None], cap)[0]
            metrics = _evaluate(
                baseline,
                fit_correction,
                start_frame=fit_end,
                end_frame=train_end,
                observed=observed_points,
                visible=visible,
                manual_tracks=manual_tracks,
                num_surface_points=num_surface_points,
            )
            validation_candidates.append(
                {
                    "candidate": _candidate_id(
                        "dense_raw_manual_override",
                        maximum_residual_m=cap,
                        correction_scale=None if scale == 1.0 else scale,
                    ),
                    "correction_scale": scale,
                    "maximum_residual_m": cap,
                    "metrics": metrics,
                    "selection_score": _selection_score(
                        metrics,
                        baseline_validation,
                    ),
                    "no_metric_regression": all(
                        metrics[name] <= baseline_validation[name]
                        for name in baseline_validation
                    ),
                }
            )
    fit_updated = fit_dense_endpoint.update_count > 0
    full_updated = dense_endpoint.update_count > 0
    fit_reference_norm_m = float(
        np.quantile(
            np.linalg.norm(fit_full_mean[:original_count][fit_updated], axis=1),
            config.relative_cap_quantile,
        )
    )
    full_reference_norm_m = float(
        np.quantile(
            np.linalg.norm(full_mean[:original_count][full_updated], axis=1),
            config.relative_cap_quantile,
        )
    )
    relative_validation_candidates: list[dict[str, Any]] = []
    for multiplier in config.relative_cap_multipliers:
        fit_cap = max(multiplier * fit_reference_norm_m, 1e-6)
        for scale in config.dense_correction_scales:
            fit_correction = _clip_residual(
                (scale * fit_full_mean)[None],
                fit_cap,
            )[0]
            fit_correction = _override_manual_nodes(
                fit_correction,
                manual_indices,
                fit_manual_endpoint,
            )
            metrics = _evaluate(
                baseline,
                fit_correction,
                start_frame=fit_end,
                end_frame=train_end,
                observed=observed_points,
                visible=visible,
                manual_tracks=manual_tracks,
                num_surface_points=num_surface_points,
            )
            relative_validation_candidates.append(
                {
                    "correction_scale": scale,
                    "relative_cap_multiplier": multiplier,
                    "fit_cap_m": fit_cap,
                    "metrics": metrics,
                    "selection_score": _selection_score(
                        metrics,
                        baseline_validation,
                    ),
                }
            )
    selectors: dict[str, dict[str, Any]] = {}
    if ray_bias_aware:
        fit_graph = graph_smoothed_discrepancy_posterior(
            fit_full_mean,
            fit_full_variance,
            fit_full_observed,
            laplacian,
            prior_strength=config.ray_graph_prior_strength,
        ).mean
        fit_ray_correction = _clip_residual(
            (config.ray_correction_scale * fit_graph)[None],
            config.ray_maximum_residual_m,
        )[0]
        validation_mid = (fit_end + train_end) // 2
        fit_all_chamfer = _evaluate_chamfer_only(
            baseline,
            fit_ray_correction,
            start_frame=fit_end,
            end_frame=train_end,
            observed=observed_points,
            visible=visible,
            num_surface_points=num_surface_points,
        )
        baseline_early_chamfer = _evaluate_chamfer_only(
            baseline,
            np.zeros_like(fit_ray_correction),
            start_frame=fit_end,
            end_frame=validation_mid,
            observed=observed_points,
            visible=visible,
            num_surface_points=num_surface_points,
        )
        fit_early_chamfer = _evaluate_chamfer_only(
            baseline,
            fit_ray_correction,
            start_frame=fit_end,
            end_frame=validation_mid,
            observed=observed_points,
            visible=visible,
            num_surface_points=num_surface_points,
        )
        baseline_late_chamfer = _evaluate_chamfer_only(
            baseline,
            np.zeros_like(fit_ray_correction),
            start_frame=validation_mid,
            end_frame=train_end,
            observed=observed_points,
            visible=visible,
            num_surface_points=num_surface_points,
        )
        fit_late_chamfer = _evaluate_chamfer_only(
            baseline,
            fit_ray_correction,
            start_frame=validation_mid,
            end_frame=train_end,
            observed=observed_points,
            visible=visible,
            num_surface_points=num_surface_points,
        )
        fit_updated = fit_dense_endpoint.update_count > 0
        fit_inlier = (
            0.0
            if not np.any(fit_updated)
            else float(
                np.median(
                    fit_dense_endpoint.final_inlier_probability[fit_updated]
                )
            )
        )
        ray_admission = decide_prefix_admission(
            baseline_all_m=baseline_validation["chamfer_distance_m"],
            candidate_all_m=fit_all_chamfer,
            baseline_early_m=baseline_early_chamfer,
            candidate_early_m=fit_early_chamfer,
            baseline_late_m=baseline_late_chamfer,
            candidate_late_m=fit_late_chamfer,
            observed_fraction=float(np.mean(fit_updated)),
            median_inlier_probability=fit_inlier,
            minimum_observed_fraction=(
                config.ray_minimum_observed_fraction
            ),
            minimum_inlier_probability=(
                config.ray_minimum_inlier_probability
            ),
            minimum_absolute_improvement_m=(
                config.ray_minimum_absolute_prefix_improvement_m
            ),
            minimum_relative_improvement=(
                config.ray_minimum_relative_prefix_improvement
            ),
        )
        if ray_admission.accepted:
            full_graph = graph_smoothed_discrepancy_posterior(
                full_mean,
                full_variance,
                full_observed,
                laplacian,
                prior_strength=config.ray_graph_prior_strength,
            ).mean
            ray_future_correction = _clip_residual(
                (config.ray_correction_scale * full_graph)[None],
                config.ray_maximum_residual_m,
            )[0]
        else:
            ray_future_correction = np.zeros_like(full_mean)
        ray_selector = (
            "causal_selected_alltracker_ray_bias_aware_graph"
        )
        candidate_metrics[ray_selector] = _evaluate(
            baseline,
            ray_future_correction,
            start_frame=train_end,
            end_frame=future_end,
            observed=observed_points,
            visible=visible,
            manual_tracks=manual_tracks,
            num_surface_points=num_surface_points,
        )
        selectors[ray_selector] = {
            "accepted": ray_admission.accepted,
            "admission": asdict(ray_admission),
            "graph_prior_strength": config.ray_graph_prior_strength,
            "correction_scale": config.ray_correction_scale,
            "maximum_residual_m": config.ray_maximum_residual_m,
            "fallback": (
                "bit-exact zero correction relative to the unchanged baseline"
            ),
            "fallback_applied": not ray_admission.accepted,
            "fallback_is_exact": (
                None
                if ray_admission.accepted
                else bool(
                    np.array_equal(
                    ray_future_correction,
                    np.zeros_like(ray_future_correction),
                    )
                )
            ),
            "future_metrics": candidate_metrics[ray_selector],
        }
    for selector, chamfer_weight, require_no_regression in (
        ("causal_selected_dense_scale_balanced", 0.5, False),
        ("causal_selected_dense_scale_cd75", 0.75, False),
        ("causal_selected_dense_scale_cd90", 0.9, False),
        ("causal_selected_dense_scale_cd_only", 1.0, False),
        ("causal_selected_dense_scale_no_regression", 0.5, True),
    ):
        eligible = [
            candidate
            for candidate in validation_candidates
            if not require_no_regression or candidate["no_metric_regression"]
        ]
        selected = min(
            eligible,
            key=lambda item: (
                _selection_score(
                    item["metrics"],
                    baseline_validation,
                    chamfer_weight=chamfer_weight,
                ),
                abs(np.log(item["correction_scale"])),
                item["maximum_residual_m"],
                item["correction_scale"],
            ),
            default=None,
        )
        selector_score = (
            None
            if selected is None
            else _selection_score(
                selected["metrics"],
                baseline_validation,
                chamfer_weight=chamfer_weight,
            )
        )
        accepted = (
            selected is not None and float(selector_score) < 1.0
        )
        if accepted:
            future_correction = _override_manual_nodes(
                float(selected["correction_scale"]) * full_mean,
                manual_indices,
                manual_endpoint,
            )
            future_correction = _clip_residual(
                future_correction[None],
                float(selected["maximum_residual_m"]),
            )[0]
        else:
            future_correction = np.zeros_like(full_mean)
        candidate_metrics[selector] = _evaluate(
            baseline,
            future_correction,
            start_frame=train_end,
            end_frame=future_end,
            observed=observed_points,
            visible=visible,
            manual_tracks=manual_tracks,
            num_surface_points=num_surface_points,
        )
        selectors[selector] = {
            "accepted": accepted,
            "chamfer_weight": chamfer_weight,
            "selected_candidate": (
                None
                if selected is None
                else {**selected, "selector_score": selector_score}
            ),
            "future_metrics": candidate_metrics[selector],
        }

    selection_chamfer_weight = (
        0.5 if config.manual_prefix_override else 1.0
    )
    relative_selected = min(
        relative_validation_candidates,
        key=lambda item: (
            _selection_score(
                item["metrics"],
                baseline_validation,
                chamfer_weight=selection_chamfer_weight,
            ),
            abs(np.log(item["correction_scale"])),
            item["relative_cap_multiplier"],
            item["correction_scale"],
        ),
    )
    relative_selector_score = _selection_score(
        relative_selected["metrics"],
        baseline_validation,
        chamfer_weight=selection_chamfer_weight,
    )
    relative_accepted = relative_selector_score < 1.0
    if relative_accepted:
        future_cap = max(
            float(relative_selected["relative_cap_multiplier"])
            * full_reference_norm_m,
            1e-6,
        )
        relative_future_correction = _clip_residual(
            (
                float(relative_selected["correction_scale"])
                * full_mean
            )[None],
            future_cap,
        )[0]
        relative_future_correction = _override_manual_nodes(
            relative_future_correction,
            manual_indices,
            manual_endpoint,
        )
    else:
        future_cap = None
        relative_future_correction = np.zeros_like(full_mean)
    relative_selector = "causal_selected_dense_relative_cap"
    candidate_metrics[relative_selector] = _evaluate(
        baseline,
        relative_future_correction,
        start_frame=train_end,
        end_frame=future_end,
        observed=observed_points,
        visible=visible,
        manual_tracks=manual_tracks,
        num_surface_points=num_surface_points,
    )
    selectors[relative_selector] = {
        "accepted": relative_accepted,
        "relative_cap_quantile": config.relative_cap_quantile,
        "fit_reference_norm_m": fit_reference_norm_m,
        "full_reference_norm_m": full_reference_norm_m,
        "future_cap_m": future_cap,
        "selected_candidate": relative_selected,
        "selector_chamfer_weight": selection_chamfer_weight,
        "selector_score": relative_selector_score,
        "future_metrics": candidate_metrics[relative_selector],
    }

    def temporal_correction(
        endpoint_mean: np.ndarray,
        previous_mean: np.ndarray,
        manual: RobustEndpointPosterior,
        *,
        gamma: float,
        interval_count: int,
        reference_count: int,
        correction_scale: float,
        cap_multiplier: float,
        observed_mask: np.ndarray,
    ) -> np.ndarray:
        delta = endpoint_mean - previous_mean
        corrections = np.empty(
            (interval_count, len(endpoint_mean), 3),
            dtype=float,
        )
        for frame in range(interval_count):
            horizon_ratio = (frame + 1) / max(reference_count, 1)
            predicted = endpoint_mean + gamma * horizon_ratio * delta
            reference = float(
                np.quantile(
                    np.linalg.norm(
                        predicted[:original_count][observed_mask],
                        axis=1,
                    ),
                    config.relative_cap_quantile,
                )
            )
            cap = max(cap_multiplier * reference, 1e-6)
            current = _clip_residual(
                (correction_scale * predicted)[None],
                cap,
            )[0]
            corrections[frame] = _override_manual_nodes(
                current,
                manual_indices,
                manual,
            )
        return corrections

    temporal_candidates: list[dict[str, Any]] = []
    relative_scale = float(relative_selected["correction_scale"])
    relative_multiplier = float(relative_selected["relative_cap_multiplier"])
    validation_count = train_end - fit_end
    temporal_reference_count = fit_end - inner_end
    for gamma in config.temporal_gamma_candidates:
        temporal_validation_correction = temporal_correction(
            fit_full_mean,
            inner_full_mean,
            fit_manual_endpoint,
            gamma=gamma,
            interval_count=validation_count,
            reference_count=temporal_reference_count,
            correction_scale=relative_scale,
            cap_multiplier=relative_multiplier,
            observed_mask=fit_updated,
        )
        metrics = _evaluate(
            baseline,
            temporal_validation_correction,
            start_frame=fit_end,
            end_frame=train_end,
            observed=observed_points,
            visible=visible,
            manual_tracks=manual_tracks,
            num_surface_points=num_surface_points,
        )
        temporal_candidates.append(
            {
                "gamma": gamma,
                "metrics": metrics,
                "selection_score": _selection_score(
                    metrics,
                    baseline_validation,
                    chamfer_weight=selection_chamfer_weight,
                ),
            }
        )
    temporal_selected = min(
        temporal_candidates,
        key=lambda item: (item["selection_score"], item["gamma"]),
    )
    temporal_accepted = (
        relative_accepted
        and float(temporal_selected["selection_score"]) < 1.0
    )
    if temporal_accepted:
        temporal_future_correction = temporal_correction(
            full_mean,
            fit_full_mean,
            manual_endpoint,
            gamma=float(temporal_selected["gamma"]),
            interval_count=future_end - train_end,
            reference_count=train_end - fit_end,
            correction_scale=relative_scale,
            cap_multiplier=relative_multiplier,
            observed_mask=full_updated,
        )
    else:
        temporal_future_correction = np.zeros(
            (future_end - train_end, len(full_mean), 3),
            dtype=float,
        )
    temporal_selector = "causal_selected_dense_relative_cap_temporal"
    candidate_metrics[temporal_selector] = _evaluate(
        baseline,
        temporal_future_correction,
        start_frame=train_end,
        end_frame=future_end,
        observed=observed_points,
        visible=visible,
        manual_tracks=manual_tracks,
        num_surface_points=num_surface_points,
    )
    selectors[temporal_selector] = {
        "accepted": temporal_accepted,
        "inner_end_frame_exclusive": inner_end,
        "reference_interval_frames": temporal_reference_count,
        "base_relative_candidate": relative_selected,
        "gamma_candidates": temporal_candidates,
        "selected_candidate": temporal_selected,
        "future_metrics": candidate_metrics[temporal_selector],
    }

    baseline_validation_chamfer = _evaluate_chamfer_only(
        baseline,
        np.zeros((len(structure_points), 3), dtype=float),
        start_frame=fit_end,
        end_frame=train_end,
        observed=observed_points,
        visible=visible,
        num_surface_points=num_surface_points,
    )
    sparse_graph_validation_candidates: list[dict[str, Any]] = []
    for center_count in config.sparse_graph_center_counts:
        for prior_strength in config.prior_strengths:
            try:
                sparse_field, sparse_ids, sparse_availability = (
                    _sparse_graph_correction(
                        structure_points,
                        fit_dense_endpoint,
                        laplacian,
                        original_count=original_count,
                        end_frame=fit_end,
                        center_count=center_count,
                        minimum_availability_fraction=(
                            config.sparse_graph_minimum_availability_fraction
                        ),
                        prior_strength=prior_strength,
                        initial_variance=config.initial_std_m**2,
                    )
                )
                sparse_reference_m = float(
                    np.quantile(
                        np.linalg.norm(sparse_field[sparse_ids], axis=1),
                        config.relative_cap_quantile,
                    )
                )
                for multiplier in config.relative_cap_multipliers:
                    cap = max(multiplier * sparse_reference_m, 1e-6)
                    for scale in config.dense_correction_scales:
                        correction = _clip_residual(
                            (scale * sparse_field)[None],
                            cap,
                        )[0]
                        validation_chamfer = _evaluate_chamfer_only(
                            baseline,
                            correction,
                            start_frame=fit_end,
                            end_frame=train_end,
                            observed=observed_points,
                            visible=visible,
                            num_surface_points=num_surface_points,
                        )
                        sparse_graph_validation_candidates.append(
                            {
                                "available": True,
                                "center_count": center_count,
                                "center_ids": sparse_ids.tolist(),
                                "minimum_center_availability": float(
                                    np.min(sparse_availability)
                                ),
                                "mean_center_availability": float(
                                    np.mean(sparse_availability)
                                ),
                                "prior_strength": prior_strength,
                                "correction_scale": scale,
                                "relative_cap_multiplier": multiplier,
                                "fit_reference_norm_m": sparse_reference_m,
                                "fit_cap_m": cap,
                                "validation_chamfer_distance_m": (
                                    validation_chamfer
                                ),
                                "selection_score": (
                                    validation_chamfer
                                    / baseline_validation_chamfer
                                ),
                            }
                        )
            except ValueError as error:
                sparse_graph_validation_candidates.append(
                    {
                        "available": False,
                        "center_count": center_count,
                        "prior_strength": prior_strength,
                        "reason": str(error),
                    }
                )
    available_sparse_graph = [
        item
        for item in sparse_graph_validation_candidates
        if item["available"]
    ]
    selected_sparse_graph = min(
        available_sparse_graph,
        key=lambda item: (
            item["selection_score"],
            item["center_count"],
            item["prior_strength"],
            abs(np.log(item["correction_scale"])),
            item["relative_cap_multiplier"],
        ),
        default=None,
    )
    sparse_graph_accepted = (
        selected_sparse_graph is not None
        and float(selected_sparse_graph["selection_score"]) < 1.0
    )
    if sparse_graph_accepted:
        (
            full_sparse_graph_field,
            full_sparse_graph_ids,
            full_sparse_graph_availability,
        ) = _sparse_graph_correction(
            structure_points,
            dense_endpoint,
            laplacian,
            original_count=original_count,
            end_frame=train_end,
            center_count=int(selected_sparse_graph["center_count"]),
            minimum_availability_fraction=(
                config.sparse_graph_minimum_availability_fraction
            ),
            prior_strength=float(selected_sparse_graph["prior_strength"]),
            initial_variance=config.initial_std_m**2,
        )
        full_sparse_graph_reference_m = float(
            np.quantile(
                np.linalg.norm(
                    full_sparse_graph_field[full_sparse_graph_ids],
                    axis=1,
                ),
                config.relative_cap_quantile,
            )
        )
        sparse_graph_future_cap_m = max(
            float(selected_sparse_graph["relative_cap_multiplier"])
            * full_sparse_graph_reference_m,
            1e-6,
        )
        sparse_graph_future_correction = _clip_residual(
            (
                float(selected_sparse_graph["correction_scale"])
                * full_sparse_graph_field
            )[None],
            sparse_graph_future_cap_m,
        )[0]
    else:
        full_sparse_graph_ids = np.empty(0, dtype=np.int64)
        full_sparse_graph_availability = np.empty(0, dtype=float)
        full_sparse_graph_reference_m = None
        sparse_graph_future_cap_m = None
        sparse_graph_future_correction = np.zeros_like(full_mean)
    sparse_graph_selector = "causal_selected_sparse_graph_support"
    candidate_metrics[sparse_graph_selector] = _evaluate(
        baseline,
        sparse_graph_future_correction,
        start_frame=train_end,
        end_frame=future_end,
        observed=observed_points,
        visible=visible,
        manual_tracks=manual_tracks,
        num_surface_points=num_surface_points,
    )
    selectors[sparse_graph_selector] = {
        "accepted": sparse_graph_accepted,
        "support_rule": (
            "prefix availability gate followed by deterministic farthest-point "
            "selection on exact initial graph identities"
        ),
        "minimum_availability_fraction": (
            config.sparse_graph_minimum_availability_fraction
        ),
        "baseline_validation_chamfer_distance_m": (
            baseline_validation_chamfer
        ),
        "validation_candidates": sparse_graph_validation_candidates,
        "selected_candidate": selected_sparse_graph,
        "refit_center_ids": full_sparse_graph_ids.tolist(),
        "refit_minimum_center_availability": (
            None
            if not len(full_sparse_graph_availability)
            else float(np.min(full_sparse_graph_availability))
        ),
        "refit_mean_center_availability": (
            None
            if not len(full_sparse_graph_availability)
            else float(np.mean(full_sparse_graph_availability))
        ),
        "full_reference_norm_m": full_sparse_graph_reference_m,
        "future_cap_m": sparse_graph_future_cap_m,
        "future_metrics": candidate_metrics[sparse_graph_selector],
    }

    planar_validation_candidates: list[dict[str, Any]] = []
    for degree in config.planar_degrees:
        try:
            planar_field, planar_fit = _canonical_planar_correction(
                structure_points,
                fit_dense_endpoint,
                original_count=original_count,
                degree=degree,
                ridge_strength=config.planar_ridge_strength,
            )
            fit_planar_norm = np.linalg.norm(
                planar_field[:original_count][fit_updated],
                axis=1,
            )
            fit_planar_reference_m = float(
                np.quantile(
                    fit_planar_norm,
                    config.relative_cap_quantile,
                )
            )
            for multiplier in config.relative_cap_multipliers:
                cap = max(multiplier * fit_planar_reference_m, 1e-6)
                for scale in config.dense_correction_scales:
                    correction = _clip_residual(
                        (scale * planar_field)[None],
                        cap,
                    )[0]
                    validation_chamfer = _evaluate_chamfer_only(
                        baseline,
                        correction,
                        start_frame=fit_end,
                        end_frame=train_end,
                        observed=observed_points,
                        visible=visible,
                        num_surface_points=num_surface_points,
                    )
                    planar_validation_candidates.append(
                        {
                            "available": True,
                            **planar_fit,
                            "correction_scale": scale,
                            "relative_cap_multiplier": multiplier,
                            "fit_reference_norm_m": fit_planar_reference_m,
                            "fit_cap_m": cap,
                            "validation_chamfer_distance_m": (
                                validation_chamfer
                            ),
                            "selection_score": (
                                validation_chamfer
                                / baseline_validation_chamfer
                            ),
                        }
                    )
        except ValueError as error:
            planar_validation_candidates.append(
                {
                    "available": False,
                    "degree": degree,
                    "reason": str(error),
                }
            )
    available_planar = [
        item for item in planar_validation_candidates if item["available"]
    ]
    selected_planar = min(
        available_planar,
        key=lambda item: (
            item["selection_score"],
            item["degree"],
            abs(np.log(item["correction_scale"])),
            item["relative_cap_multiplier"],
        ),
        default=None,
    )
    planar_accepted = (
        selected_planar is not None
        and float(selected_planar["selection_score"]) < 1.0
    )
    if planar_accepted:
        full_planar_field, full_planar_fit = _canonical_planar_correction(
            structure_points,
            dense_endpoint,
            original_count=original_count,
            degree=int(selected_planar["degree"]),
            ridge_strength=config.planar_ridge_strength,
        )
        full_planar_reference_m = float(
            np.quantile(
                np.linalg.norm(
                    full_planar_field[:original_count][full_updated],
                    axis=1,
                ),
                config.relative_cap_quantile,
            )
        )
        planar_future_cap_m = max(
            float(selected_planar["relative_cap_multiplier"])
            * full_planar_reference_m,
            1e-6,
        )
        planar_future_correction = _clip_residual(
            (
                float(selected_planar["correction_scale"])
                * full_planar_field
            )[None],
            planar_future_cap_m,
        )[0]
    else:
        full_planar_fit = None
        full_planar_reference_m = None
        planar_future_cap_m = None
        planar_future_correction = np.zeros_like(full_mean)
    planar_selector = "causal_selected_canonical_planar"
    candidate_metrics[planar_selector] = _evaluate(
        baseline,
        planar_future_correction,
        start_frame=train_end,
        end_frame=future_end,
        observed=observed_points,
        visible=visible,
        manual_tracks=manual_tracks,
        num_surface_points=num_surface_points,
    )
    selectors[planar_selector] = {
        "accepted": planar_accepted,
        "coordinate_system": (
            "deterministic PCA plane of initial PhysTwin geometry; "
            "canonical proxy, not recovered material UV"
        ),
        "ridge_strength": config.planar_ridge_strength,
        "baseline_validation_chamfer_distance_m": (
            baseline_validation_chamfer
        ),
        "validation_candidates": planar_validation_candidates,
        "selected_candidate": selected_planar,
        "refit": full_planar_fit,
        "full_reference_norm_m": full_planar_reference_m,
        "future_cap_m": planar_future_cap_m,
        "future_metrics": candidate_metrics[planar_selector],
    }

    rbf_validation_candidates: list[dict[str, Any]] = []
    for center_count in config.rbf_center_counts:
        try:
            validation_correction, fit_centers, fit_availability = (
                _recursive_rbf_correction(
                    inference_points,
                    dense_valid,
                    baseline,
                    structure_points,
                    fit_end_frame=fit_end,
                    query_start_frame=fit_end,
                    query_end_frame=train_end,
                    original_count=original_count,
                    center_count=center_count,
                    minimum_availability_fraction=(
                        config.rbf_minimum_availability_fraction
                    ),
                )
            )
            validation_chamfer = _evaluate_chamfer_only(
                baseline,
                validation_correction,
                start_frame=fit_end,
                end_frame=train_end,
                observed=observed_points,
                visible=visible,
                num_surface_points=num_surface_points,
            )
            rbf_validation_candidates.append(
                {
                    "available": True,
                    "center_count": center_count,
                    "center_ids": fit_centers.tolist(),
                    "minimum_center_availability": float(
                        np.min(fit_availability)
                    ),
                    "mean_center_availability": float(
                        np.mean(fit_availability)
                    ),
                    "validation_chamfer_distance_m": validation_chamfer,
                    "selection_score": (
                        validation_chamfer / baseline_validation_chamfer
                    ),
                }
            )
        except ValueError as error:
            rbf_validation_candidates.append(
                {
                    "available": False,
                    "center_count": center_count,
                    "reason": str(error),
                }
            )
    available_rbf = [
        item for item in rbf_validation_candidates if item["available"]
    ]
    selected_rbf = min(
        available_rbf,
        key=lambda item: (item["selection_score"], item["center_count"]),
        default=None,
    )
    rbf_accepted = (
        selected_rbf is not None
        and float(selected_rbf["selection_score"]) < 1.0
    )
    if rbf_accepted:
        (
            rbf_future_correction,
            full_centers,
            full_center_availability,
            rbf_refit_rejection_reason,
        ) = _recursive_rbf_refit_or_fallback(
            inference_points,
            dense_valid,
            baseline,
            structure_points,
            fit_end_frame=train_end,
            query_start_frame=train_end,
            query_end_frame=future_end,
            original_count=original_count,
            center_count=int(selected_rbf["center_count"]),
            minimum_availability_fraction=(
                config.rbf_minimum_availability_fraction
            ),
        )
        if rbf_refit_rejection_reason is not None:
            rbf_accepted = False
    else:
        rbf_future_correction = np.zeros(
            (future_end - train_end, len(structure_points), 3),
            dtype=float,
        )
        full_centers = np.empty(0, dtype=np.int64)
        full_center_availability = np.empty(0, dtype=float)
        rbf_refit_rejection_reason = None
    rbf_selector = "causal_selected_recursive_rbf"
    candidate_metrics[rbf_selector] = _evaluate(
        baseline,
        rbf_future_correction,
        start_frame=train_end,
        end_frame=future_end,
        observed=observed_points,
        visible=visible,
        manual_tracks=manual_tracks,
        num_surface_points=num_surface_points,
    )
    selectors[rbf_selector] = {
        "accepted": rbf_accepted,
        "minimum_availability_fraction": (
            config.rbf_minimum_availability_fraction
        ),
        "baseline_validation_chamfer_distance_m": (
            baseline_validation_chamfer
        ),
        "validation_candidates": rbf_validation_candidates,
        "selected_candidate": selected_rbf,
        "refit_rejection_reason": rbf_refit_rejection_reason,
        "refit_center_ids": full_centers.tolist(),
        "refit_minimum_center_availability": (
            None
            if not len(full_center_availability)
            else float(np.min(full_center_availability))
        ),
        "future_metrics": candidate_metrics[rbf_selector],
    }

    nearest_cloud = _nearest_cloud_corrections(
        baseline,
        observed_points,
        visible,
        end_frame=train_end,
        windows=config.nearest_cloud_windows,
        num_surface_points=num_surface_points,
    )
    for window, correction in nearest_cloud.items():
        for cap in config.maximum_residuals_m:
            for scale in config.dense_correction_scales:
                scaled = _override_manual_nodes(
                    scale * correction,
                    manual_indices,
                    manual_endpoint,
                )
                clipped = _clip_residual(scaled[None], cap)[0]
                candidate_metrics[
                    _candidate_id(
                        f"nearest_cloud_window_{window:03d}",
                        maximum_residual_m=cap,
                        correction_scale=None if scale == 1.0 else scale,
                    )
                ] = _evaluate(
                    baseline,
                    clipped,
                    start_frame=train_end,
                    end_frame=future_end,
                    observed=observed_points,
                    visible=visible,
                    manual_tracks=manual_tracks,
                    num_surface_points=num_surface_points,
                )

    manual_only = np.zeros_like(full_mean)
    manual_only = _override_manual_nodes(
        manual_only,
        manual_indices,
        manual_endpoint,
    )
    dense_raw_manual_override = _override_manual_nodes(
        full_mean,
        manual_indices,
        manual_endpoint,
    )
    for cap in config.maximum_residuals_m:
        for method, correction in (
            ("manual_prefix_nodes_only", manual_only),
            ("dense_raw", full_mean),
            ("dense_raw_manual_override", dense_raw_manual_override),
        ):
            clipped = _clip_residual(correction[None], cap)[0]
            candidate_metrics[
                _candidate_id(method, maximum_residual_m=cap)
            ] = _evaluate(
                baseline,
                clipped,
                start_frame=train_end,
                end_frame=future_end,
                observed=observed_points,
                visible=visible,
                manual_tracks=manual_tracks,
                num_surface_points=num_surface_points,
            )
        for scale in config.dense_correction_scales:
            if scale == 1.0:
                continue
            scaled = _override_manual_nodes(
                scale * full_mean,
                manual_indices,
                manual_endpoint,
            )
            clipped = _clip_residual(scaled[None], cap)[0]
            candidate_metrics[
                _candidate_id(
                    "dense_raw_manual_override",
                    maximum_residual_m=cap,
                    correction_scale=scale,
                )
            ] = _evaluate(
                baseline,
                clipped,
                start_frame=train_end,
                end_frame=future_end,
                observed=observed_points,
                visible=visible,
                manual_tracks=manual_tracks,
                num_surface_points=num_surface_points,
            )

    for strength in config.prior_strengths:
        dense_graph = graph_smoothed_discrepancy_posterior(
            full_mean,
            full_variance,
            full_observed,
            laplacian,
            prior_strength=strength,
        ).mean
        combined_graph = graph_smoothed_discrepancy_posterior(
            combined_mean,
            combined_variance,
            combined_observed,
            laplacian,
            prior_strength=strength,
        ).mean
        variants = {
            "dense_graph": dense_graph,
            "dense_graph_manual_override": _override_manual_nodes(
                dense_graph,
                manual_indices,
                manual_endpoint,
            ),
            "combined_graph": combined_graph,
            "combined_graph_manual_override": _override_manual_nodes(
                combined_graph,
                manual_indices,
                manual_endpoint,
            ),
        }
        for cap in config.maximum_residuals_m:
            for method, correction in variants.items():
                clipped = _clip_residual(correction[None], cap)[0]
                candidate_metrics[
                    _candidate_id(
                        method,
                        prior_strength=strength,
                        maximum_residual_m=cap,
                    )
                ] = _evaluate(
                    baseline,
                    clipped,
                    start_frame=train_end,
                    end_frame=future_end,
                    observed=observed_points,
                    visible=visible,
                    manual_tracks=manual_tracks,
                    num_surface_points=num_surface_points,
                )

    if ray_bias_aware:
        cotracker_summary["endpoint_diagnostics"] = {
            str(frame): diagnostic
            for frame, diagnostic in sorted(
                ray_endpoint_diagnostics.items()
            )
        }
    return case, {
        "selected_raw_family": selected_family,
        "selected_within_family_method": selected_within_family_method,
        "baseline_kind": config.baseline_kind,
        "baseline_trajectory": {
            "path": str(baseline_path),
            "sha256": actual_hash,
        },
        "train_end_frame_exclusive": train_end,
        "fit_end_frame_exclusive": fit_end,
        "future_end_frame_exclusive": future_end,
        "manual_prefix_track_count": int(np.sum(manual_initial)),
        "manual_initial_match_max_m": float(np.max(initial_match_m, initial=0.0)),
        "observation_source": config.observation_source,
        "cotracker_depth_lift": cotracker_summary,
        "baseline": baseline_metrics,
        "causal_selection": {
            "baseline_validation": baseline_validation,
            "candidates": validation_candidates,
            "relative_cap_candidates": relative_validation_candidates,
            "selectors": selectors,
        },
        "candidates": candidate_metrics,
    }


def _aggregate(
    case_results: dict[str, dict[str, Any]],
    config: HeadroomConfig,
) -> dict[str, Any]:
    candidate_ids = tuple(next(iter(case_results.values()))["candidates"])
    if any(tuple(result["candidates"]) != candidate_ids for result in case_results.values()):
        raise ValueError("candidate grids differ across cases")
    baseline = {
        metric: float(
            np.mean([result["baseline"][metric] for result in case_results.values()])
        )
        for metric in ("chamfer_distance_m", "track_error_m")
    }
    candidates: dict[str, dict[str, Any]] = {}
    for candidate in candidate_ids:
        metrics = {
            metric: float(
                np.mean(
                    [
                        result["candidates"][candidate][metric]
                        for result in case_results.values()
                    ]
                )
            )
            for metric in ("chamfer_distance_m", "track_error_m")
        }
        candidates[candidate] = {
            **metrics,
            "beats_published_sota_chamfer": (
                metrics["chamfer_distance_m"] < config.published_sota_chamfer_m
            ),
            "beats_published_sota_track": (
                metrics["track_error_m"] < config.published_sota_track_m
            ),
            "beats_both_published_sota_thresholds": (
                metrics["chamfer_distance_m"] < config.published_sota_chamfer_m
                and metrics["track_error_m"] < config.published_sota_track_m
            ),
        }
    ordered = sorted(
        candidates,
        key=lambda name: (
            max(
                candidates[name]["chamfer_distance_m"]
                / config.published_sota_chamfer_m,
                candidates[name]["track_error_m"]
                / config.published_sota_track_m,
            ),
            candidates[name]["chamfer_distance_m"],
            candidates[name]["track_error_m"],
            name,
        ),
    )
    per_case_oracle = {
        metric: float(
            np.mean(
                [
                    min(
                        result["candidates"][candidate][metric]
                        for candidate in candidate_ids
                    )
                    for result in case_results.values()
                ]
            )
        )
        for metric in ("chamfer_distance_m", "track_error_m")
    }
    return {
        "baseline": baseline,
        "candidate_count": len(candidate_ids),
        "best_joint_candidate": {
            "name": ordered[0],
            **candidates[ordered[0]],
        },
        "candidates": candidates,
        "per_case_metric_oracle": per_case_oracle,
    }


def _load_family_entries(
    selection: dict[str, Any],
    *,
    strength_root: Path,
) -> dict[str, dict[str, dict[str, Any]]]:
    families = {
        str(result["selected_family"])
        for result in selection["case_results"].values()
    }
    result: dict[str, dict[str, dict[str, Any]]] = {}
    for family in sorted(families):
        manifest_path = strength_root / "families" / family / "external_backbone_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        by_case: dict[str, dict[str, Any]] = {}
        for entry in manifest["cases"]:
            if str(entry["name"]) not in selection["case_results"]:
                continue
            item = dict(entry)
            item["selected_within_family_method"] = str(
                selection["case_results"][str(entry["name"])][
                    "selected_within_family_method"
                ]
            )
            by_case[str(entry["name"])] = item
        result[family] = by_case
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--strength-root", type=Path, required=True)
    parser.add_argument("--family-selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--baseline-kind",
        choices=("raw_matphys_replay", "selected_overlay_sequential"),
        default="raw_matphys_replay",
    )
    parser.add_argument(
        "--observation-source",
        choices=(
            "final_data",
            "cotracker3_source_depth",
            "alltracker_source_depth",
            "cotracker3_multiview",
            "cotracker3_multiview_depth",
            "cotracker3_hybrid",
            "cotracker3_multiview_priority",
            "cotracker3_multiview_tangent_priority",
            "cotracker3_multiview_directional_priority",
            "alltracker_multiview_ray_bias_aware",
        ),
        default="final_data",
    )
    parser.add_argument("--cotracker-cues-root", type=Path)
    parser.add_argument("--raw-case-root", type=Path)
    parser.add_argument("--no-manual-prefix-override", action="store_true")
    parser.add_argument("--cotracker-minimum-quality", type=float, default=0.5)
    parser.add_argument(
        "--cotracker-maximum-cycle-error-px",
        type=float,
        default=5.0,
    )
    parser.add_argument(
        "--cotracker-maximum-reprojection-error-px",
        type=float,
        default=3.0,
    )
    parser.add_argument(
        "--cotracker-minimum-camera-count",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--cotracker-maximum-view-disagreement-m",
        type=float,
        default=0.01,
    )
    parser.add_argument(
        "--multiview-priority-minimum-availability-fraction",
        type=float,
        default=(
            HeadroomConfig.multiview_priority_minimum_availability_fraction
        ),
    )
    parser.add_argument(
        "--multiview-tangent-neighbor-count",
        type=int,
        default=HeadroomConfig.multiview_tangent_neighbor_count,
    )
    parser.add_argument(
        "--prior-strengths",
        type=float,
        nargs="+",
        default=list(HeadroomConfig.prior_strengths),
    )
    parser.add_argument(
        "--maximum-residuals-m",
        type=float,
        nargs="+",
        default=list(HeadroomConfig.maximum_residuals_m),
    )
    parser.add_argument(
        "--dense-correction-scales",
        type=float,
        nargs="+",
        default=list(HeadroomConfig.dense_correction_scales),
    )
    parser.add_argument(
        "--nearest-cloud-windows",
        type=int,
        nargs="*",
        default=list(HeadroomConfig.nearest_cloud_windows),
    )
    parser.add_argument(
        "--relative-cap-quantile",
        type=float,
        default=HeadroomConfig.relative_cap_quantile,
    )
    parser.add_argument(
        "--relative-cap-multipliers",
        type=float,
        nargs="+",
        default=list(HeadroomConfig.relative_cap_multipliers),
    )
    parser.add_argument(
        "--temporal-gamma-candidates",
        type=float,
        nargs="+",
        default=list(HeadroomConfig.temporal_gamma_candidates),
    )
    parser.add_argument(
        "--rbf-center-counts",
        type=int,
        nargs="+",
        default=list(HeadroomConfig.rbf_center_counts),
    )
    parser.add_argument(
        "--rbf-minimum-availability-fraction",
        type=float,
        default=HeadroomConfig.rbf_minimum_availability_fraction,
    )
    parser.add_argument(
        "--sparse-graph-center-counts",
        type=int,
        nargs="+",
        default=list(HeadroomConfig.sparse_graph_center_counts),
    )
    parser.add_argument(
        "--sparse-graph-minimum-availability-fraction",
        type=float,
        default=(
            HeadroomConfig.sparse_graph_minimum_availability_fraction
        ),
    )
    parser.add_argument(
        "--planar-degrees",
        type=int,
        nargs="+",
        default=list(HeadroomConfig.planar_degrees),
    )
    parser.add_argument(
        "--planar-ridge-strength",
        type=float,
        default=HeadroomConfig.planar_ridge_strength,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.workers < 1:
        raise ValueError("workers must be positive")
    if not (
        0.0
        <= args.multiview_priority_minimum_availability_fraction
        <= 1.0
    ):
        raise ValueError(
            "multiview priority availability must lie in [0, 1]"
        )
    if args.multiview_tangent_neighbor_count < 3:
        raise ValueError("multiview tangent neighbor count must be at least three")
    config = HeadroomConfig(
        baseline_kind=args.baseline_kind,
        observation_source=args.observation_source,
        manual_prefix_override=not args.no_manual_prefix_override,
        cotracker_minimum_quality=args.cotracker_minimum_quality,
        cotracker_maximum_cycle_error_px=(
            args.cotracker_maximum_cycle_error_px
        ),
        cotracker_maximum_reprojection_error_px=(
            args.cotracker_maximum_reprojection_error_px
        ),
        cotracker_minimum_camera_count=args.cotracker_minimum_camera_count,
        cotracker_maximum_view_disagreement_m=(
            args.cotracker_maximum_view_disagreement_m
        ),
        multiview_priority_minimum_availability_fraction=(
            args.multiview_priority_minimum_availability_fraction
        ),
        multiview_tangent_neighbor_count=(
            args.multiview_tangent_neighbor_count
        ),
        prior_strengths=tuple(args.prior_strengths),
        maximum_residuals_m=tuple(args.maximum_residuals_m),
        dense_correction_scales=tuple(args.dense_correction_scales),
        nearest_cloud_windows=tuple(args.nearest_cloud_windows),
        relative_cap_quantile=args.relative_cap_quantile,
        relative_cap_multipliers=tuple(args.relative_cap_multipliers),
        temporal_gamma_candidates=tuple(args.temporal_gamma_candidates),
        rbf_center_counts=tuple(args.rbf_center_counts),
        rbf_minimum_availability_fraction=(
            args.rbf_minimum_availability_fraction
        ),
        sparse_graph_center_counts=tuple(args.sparse_graph_center_counts),
        sparse_graph_minimum_availability_fraction=(
            args.sparse_graph_minimum_availability_fraction
        ),
        planar_degrees=tuple(args.planar_degrees),
        planar_ridge_strength=args.planar_ridge_strength,
    )
    selection_path = args.family_selection.resolve()
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if not selection.get("case_results"):
        raise ValueError("family selection has no cases")
    family_entries = _load_family_entries(
        selection,
        strength_root=args.strength_root.resolve(),
    )
    if config.observation_source != "final_data":
        if args.cotracker_cues_root is None:
            raise ValueError("tracker observations require a cues root")
    if config.observation_source in {
        "cotracker3_source_depth",
        "alltracker_source_depth",
        "cotracker3_multiview_depth",
        "cotracker3_hybrid",
        "cotracker3_multiview_priority",
        "cotracker3_multiview_tangent_priority",
        "cotracker3_multiview_directional_priority",
    }:
        if args.raw_case_root is None:
            raise ValueError(
                "depth observations require a raw-case root"
            )
    jobs = []
    for case, selected in selection["case_results"].items():
        selected_family, raw_entry = _selected_raw_entry(
            case,
            selection,
            family_entries,
        )
        if config.baseline_kind == "raw_matphys_replay":
            baseline_entry = raw_entry
        else:
            baseline_entry = {
                "trajectory": str(selected["output"]["path"]),
                "sha256": str(selected["output"]["sha256"]),
            }
        jobs.append(
            (
                case,
                args.data_root.resolve() / case,
                selected_family,
                baseline_entry,
                str(selected["selected_within_family_method"]),
                int(selected["fit_end_frame_exclusive"]),
                (
                    None
                    if args.cotracker_cues_root is None
                    else args.cotracker_cues_root.resolve() / case / "cues.npz"
                ),
                (
                    None
                    if args.raw_case_root is None
                    else args.raw_case_root.resolve() / case
                ),
                config,
            )
        )
    if args.workers == 1:
        fitted = map(_run_case, jobs)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            fitted = executor.map(_run_case, jobs)
    case_results = dict(fitted)
    report = {
        "schema_version": 1,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "post-open model-class headroom diagnostic; not SOTA evidence",
        "config": asdict(config),
        "information_boundary": {
            "cohort": "previously opened PhysTwin 22-case exploratory cohort",
            "raw_backbone_selection": (
                "object-disjoint MatPhys spring strength selected before future opening"
            ),
            "baseline_kind": config.baseline_kind,
            "sequential_overlay_caveat": (
                "selected_overlay_sequential reuses a validation-selected prefix "
                "overlay as its baseline and then fits the remaining prefix residual; "
                "it is a headroom arm, not yet a coherent independent-evidence update"
                if config.baseline_kind == "selected_overlay_sequential"
                else None
            ),
            "correction_fit": (
                "object pseudo-tracks and optional released manual 3D tracks strictly "
                "before each train_end_frame"
            ),
            "manual_prefix_role": (
                "label-assisted sparse-observation upper bound; not a deployable input"
                if config.manual_prefix_override
                else "disabled; manual tracks are evaluation-only"
            ),
            "observation_source": config.observation_source,
            "future_inputs_used_for_prediction": False,
            "future_metrics_read_after_predictions_fixed": True,
            "claim": (
                "headroom and bottleneck localization only; an independent fresh-object "
                "evaluation remains required"
            ),
        },
        "inputs": {
            "family_selection": {
                "path": str(selection_path),
                "sha256": _sha256(selection_path),
            },
            "data_root": str(args.data_root.resolve()),
            "strength_root": str(args.strength_root.resolve()),
        },
        "case_results": case_results,
        "aggregate": _aggregate(case_results, config),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    best = report["aggregate"]["best_joint_candidate"]
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "case_count": len(case_results),
                "baseline": report["aggregate"]["baseline"],
                "best_joint_candidate": best,
                "per_case_metric_oracle": report["aggregate"][
                    "per_case_metric_oracle"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
