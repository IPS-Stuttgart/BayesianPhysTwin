#!/usr/bin/env python3
"""Validate explicit-gauge Prob4D state updates on real PhysTwin cameras.

This is a retrospective source-cohort experiment. It opens exactly two complete
MotionCrafter windows per case, writes a prediction artifact before query truth
is scored, and never describes the result as prospective confirmation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import pickle
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from bayesian_phystwin._gauge_aware_contracts import (
    COMPOSITE_WEIGHT_MODE_PROVIDER_FINAL,
    GaugeAwareObservationBatch,
)
from bayesian_phystwin._prior_aware_gauge_math import PriorAwareGaugeConfigV1
from bayesian_phystwin.observation_belief_gauge_adapter import (
    global_translation_bias_jacobian,
)
from bayesian_phystwin.phystwin_graph import (
    PhysTwinSpringGraphConfig,
    build_phystwin_spring_graph,
)
from bayesian_phystwin.phystwin_motioncrafter_association import (
    load_phystwin_world_point_grid,
    resample_cover_grid,
    robust_similarity_transform,
)
from bayesian_phystwin.prior_aware_gauge_belief import (
    update_prior_aware_gauge_belief,
)
from bayesian_phystwin.structural_artifact import build_rigid_free_graph_basis

PROTOCOL_SCHEMA = "bayesian-phystwin-prob4d-real-camera-validation"
REPORT_SCHEMA = "bayesian-phystwin-prob4d-real-camera-validation-report"
METHODS = (
    "B0_physical_fallback",
    "P1_marginal_gauge_persistent",
    "P2_explicit_gauge_framewise",
    "P3_explicit_gauge_persistent",
)
PRIMARY_METHOD = "P3_explicit_gauge_persistent"
CHI_SQUARE_3_90 = 6.251388631170325


@dataclass(frozen=True)
class GraphAssociation:
    """One fixed source-geometry association to physical graph nodes."""

    node_indices: np.ndarray
    weights: np.ndarray
    probability: float
    nearest_distance_m: float
    normalized_entropy: float


@dataclass(frozen=True)
class Candidate:
    """Truth-free physical-state candidate and its uncertainty."""

    method_id: str
    inference_admissible: bool
    reason: str
    raw_correction_m: np.ndarray
    covariance_m2: np.ndarray
    risk_score: float
    guard_accepted: bool
    deployed_correction_m: np.ndarray
    diagnostics: Mapping[str, Any]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--case-data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repository-revision", required=True)
    parser.add_argument("--prob4d-revision", required=True)
    parser.add_argument("--case")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _require(condition: bool | np.bool_, message: str) -> None:
    if not bool(condition):
        raise ValueError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(values))
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(json.dumps(array.shape, separators=(",", ":")).encode("ascii"))
    digest.update(array.view(np.uint8))
    return digest.hexdigest()


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_value(value), indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def _load_pickle(path: Path) -> Any:
    with path.open("rb") as handle:
        return pickle.load(handle)


def source_only_prior_reliability(
    parallel_disagreement: np.ndarray,
    lateral_disagreement: np.ndarray,
    parallel_variance: np.ndarray,
    lateral_variance: np.ndarray,
    overlap_count: np.ndarray,
    *,
    minimum: float,
) -> np.ndarray:
    """Compute reliability without consulting the physical innovation."""

    if not 0.0 < minimum <= 1.0:
        raise ValueError("minimum reliability must lie in (0, 1]")
    normalized = (
        np.asarray(parallel_disagreement, dtype=np.float64)
        / np.maximum(np.asarray(parallel_variance, dtype=np.float64), 1e-12)
        + np.asarray(lateral_disagreement, dtype=np.float64)
        / np.maximum(np.asarray(lateral_variance, dtype=np.float64), 1e-12)
    )
    reliability = np.exp(-0.5 * np.minimum(normalized, 50.0))
    reliability = np.where(np.asarray(overlap_count) > 0.0, reliability, 1.0)
    return np.clip(reliability, minimum, 1.0)


def deterministic_farthest_point_indices(
    points_m: np.ndarray,
    eligible_indices: np.ndarray,
    count: int,
) -> np.ndarray:
    """Select a spatially spread identity subset using frame-zero positions only."""

    points = np.asarray(points_m, dtype=np.float64)
    eligible = np.asarray(eligible_indices, dtype=np.int64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points_m must have shape (N, 3)")
    if count < 1 or count > len(eligible):
        raise ValueError("count must select a nonempty eligible subset")
    if len(np.unique(eligible)) != len(eligible):
        raise ValueError("eligible_indices must be unique")
    selected = [int(np.min(eligible))]
    squared = np.sum(np.square(points[eligible] - points[selected[0]]), axis=1)
    while len(selected) < count:
        best_value = float(np.max(squared))
        candidates = eligible[np.isclose(squared, best_value, rtol=0.0, atol=1e-15)]
        choice = int(np.min(candidates))
        selected.append(choice)
        candidate_distance = np.sum(
            np.square(points[eligible] - points[choice]), axis=1
        )
        squared = np.minimum(squared, candidate_distance)
    return np.asarray(selected, dtype=np.int64)


def soft_graph_association_from_neighbours(
    neighbour_indices: np.ndarray,
    neighbour_distances_m: np.ndarray,
    *,
    distance_scale_m: float,
    maximum_distance_m: float,
) -> GraphAssociation:
    """Turn source-geometry candidates into a fixed soft graph association."""

    nodes = np.asarray(neighbour_indices, dtype=np.int64)
    distances = np.asarray(neighbour_distances_m, dtype=np.float64)
    if nodes.ndim != 1 or distances.shape != nodes.shape or not len(nodes):
        raise ValueError("neighbour arrays must share one nonempty vector shape")
    if np.any(nodes < 0) or np.any(distances < 0.0) or not np.all(
        np.isfinite(distances)
    ):
        raise ValueError("neighbour candidates are invalid")
    if distance_scale_m <= 0.0 or maximum_distance_m <= 0.0:
        raise ValueError("association distance scales must be positive")
    logits = -0.5 * np.square(distances / distance_scale_m)
    logits -= np.max(logits)
    weights = np.exp(logits)
    weights /= np.sum(weights)
    entropy = float(-np.sum(weights * np.log(np.maximum(weights, 1e-15))))
    normalized_entropy = (
        0.0 if len(weights) == 1 else entropy / math.log(float(len(weights)))
    )
    distance_probability = math.exp(
        -0.5 * (float(distances[0]) / maximum_distance_m) ** 2
    )
    probability = float(np.clip(distance_probability * np.max(weights), 0.0, 1.0))
    return GraphAssociation(
        node_indices=nodes.copy(),
        weights=weights,
        probability=probability,
        nearest_distance_m=float(distances[0]),
        normalized_entropy=normalized_entropy,
    )


def assignment_mixture_covariance(
    candidate_points_m: np.ndarray,
    weights: np.ndarray,
    *,
    floor_m2: float,
) -> np.ndarray:
    """Return metric assignment spread so ambiguous identities stay uncertain."""

    points = np.asarray(candidate_points_m, dtype=np.float64)
    probabilities = np.asarray(weights, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or probabilities.shape != (
        len(points),
    ):
        raise ValueError("candidate points and weights disagree")
    if floor_m2 <= 0.0 or np.any(probabilities < 0.0):
        raise ValueError("assignment covariance inputs are invalid")
    probabilities = probabilities / np.sum(probabilities)
    mean = np.sum(probabilities[:, None] * points, axis=0)
    residual = points - mean
    covariance = np.einsum(
        "n,ni,nj->ij", probabilities, residual, residual, optimize=True
    )
    return 0.5 * (covariance + covariance.T) + np.eye(3) * floor_m2


def group_capped_composite_weights(
    correlation_groups: Sequence[str],
    association_probability: np.ndarray,
    *,
    effective_samples_per_group: float,
) -> np.ndarray:
    """Cap dense correlated evidence and keep association as likelihood power."""

    groups = np.asarray(tuple(map(str, correlation_groups)), dtype=object)
    association = np.asarray(association_probability, dtype=np.float64)
    if groups.shape != association.shape or not len(groups):
        raise ValueError("groups and association probability disagree")
    if effective_samples_per_group <= 0.0 or np.any(
        (association <= 0.0) | (association > 1.0)
    ):
        raise ValueError("composite-weight inputs are invalid")
    result = np.empty(len(groups), dtype=np.float64)
    for group in dict.fromkeys(groups.tolist()):
        selected = np.flatnonzero(groups == group)
        cap = min(effective_samples_per_group, float(len(selected))) / len(selected)
        group_association = float(np.mean(association[selected]))
        result[selected] = cap * group_association
    return result


def query_covariance(
    query_jacobian: np.ndarray,
    state_covariance: np.ndarray,
) -> np.ndarray:
    return np.einsum(
        "nci,ij,ndj->ncd",
        np.asarray(query_jacobian, dtype=np.float64),
        np.asarray(state_covariance, dtype=np.float64),
        np.asarray(query_jacobian, dtype=np.float64),
        optimize=True,
    )


def exact_fallback_selection(
    raw_correction_m: np.ndarray,
    *,
    inference_admissible: bool,
    risk_score: float,
    risk_threshold: float,
) -> tuple[bool, np.ndarray]:
    """Apply the frozen guard with an exact all-zero fallback."""

    raw = np.asarray(raw_correction_m, dtype=np.float64)
    accepted = bool(inference_admissible and risk_score <= risk_threshold)
    return accepted, raw.copy() if accepted else np.zeros_like(raw)


def _basis_with_metric_coefficients(
    graph_initial_m: np.ndarray,
    surface_springs: np.ndarray,
    *,
    rank: int,
) -> tuple[np.ndarray, np.ndarray, Mapping[str, Any]]:
    basis, frequencies, diagnostics = build_rigid_free_graph_basis(
        graph_initial_m,
        surface_springs,
        rank=rank,
    )
    maximum = np.max(np.linalg.norm(basis, axis=1), axis=0)
    _require(np.all(maximum > 0.0), "graph basis contains a zero mode")
    return basis / maximum[None, None, :], frequencies, diagnostics


def _nearest_manual_nodes(
    graph_initial_m: np.ndarray,
    manual_frame_zero_m: np.ndarray,
) -> np.ndarray:
    graph = np.asarray(graph_initial_m, dtype=np.float64)
    tracks = np.asarray(manual_frame_zero_m, dtype=np.float64)
    difference = tracks[:, None, :] - graph[None, :, :]
    return np.argmin(np.sum(np.square(difference), axis=2), axis=1).astype(np.int64)


def _metric_anchor_covariance(
    alignment_rmse_m: float,
    object_radius_m: float,
) -> np.ndarray:
    translation_std = max(float(alignment_rmse_m), 0.003)
    angular_std = float(
        np.clip(translation_std / max(float(object_radius_m), 0.05), 0.02, 0.25)
    )
    standard_deviation = np.asarray(
        [angular_std, angular_std, angular_std, angular_std]
        + [translation_std] * 3,
        dtype=np.float64,
    )
    return np.diag(np.square(standard_deviation))


def _risk_score(
    result: Any,
    covariance_m2: np.ndarray,
    *,
    physical_response_scale_m: float,
) -> tuple[float, Mapping[str, Any]]:
    width = float(
        np.sqrt(np.mean(np.trace(covariance_m2, axis1=1, axis2=2)))
    )
    diagnostics = result.diagnostics
    nominal_values = diagnostics.get(
        "observation_group_posterior_nominal_probability", []
    )
    nominal = float(np.mean(nominal_values)) if nominal_values else 0.0
    identifiable = (
        float(np.min(result.identifiable_fractions))
        if len(result.identifiable_fractions)
        else 0.0
    )
    sensitivity = (
        float(np.min(result.query_sensitivity_fractions))
        if len(result.query_sensitivity_fractions)
        else 0.0
    )
    converged = bool(diagnostics.get("mixture_fixed_point_converged", False))
    risk = (
        width / max(physical_response_scale_m, 1e-12)
        + (1.0 - nominal)
        + 0.50 * (1.0 - identifiable)
        + 0.25 * (1.0 - sensitivity)
        + (0.0 if converged else 0.35)
    )
    return risk, {
        "posterior_query_width_rms_m": width,
        "posterior_nominal_probability": nominal,
        "minimum_identifiable_fraction": identifiable,
        "minimum_query_sensitivity_fraction": sensitivity,
        "mixture_fixed_point_converged": converged,
    }


def _slice_window(window: Any, start: int, stop: int, *, window_id: str) -> Any:
    from prob4d.data import PredictionWindow

    return PredictionWindow(
        window_id=window_id,
        frame_indices=window.frame_indices[start:stop],
        point_map=window.point_map[start:stop],
        valid_mask=window.valid_mask[start:stop],
        scene_flow=window.scene_flow[start:stop],
        deform_mask=window.deform_mask[start:stop],
        ray_directions=(
            None
            if window.ray_directions is None
            else window.ray_directions[start:stop]
        ),
        dense_storage_dtype="float32",
    )


def _load_selected_windows(
    manifest_path: Path,
    *,
    count: int,
) -> tuple[dict[str, Any], list[Any], list[dict[str, Any]]]:
    from prob4d.data import PredictionWindow

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = manifest.get("overlap_windows")
    _require(isinstance(entries, list) and len(entries) >= count, "too few windows")
    selected_entries = entries[:count]
    cutoff = int(selected_entries[-1]["stop_frame"])
    windows: list[Any] = []
    lineage: list[dict[str, Any]] = []
    for entry in selected_entries:
        path = (manifest_path.parent / str(entry["path"])).resolve()
        _require(path.is_relative_to(manifest_path.parent.resolve()), "window escapes root")
        _require(int(entry["stop_frame"]) <= cutoff, "selected window crosses cutoff")
        payload_sha256 = _sha256(path)
        window = PredictionWindow.from_npz(
            path,
            start_frame=int(entry["start_frame"]),
            window_id=str(entry["window_id"]),
            dense_storage_dtype="float32",
        )
        _require(
            int(window.frame_indices[-1]) < cutoff,
            "selected prediction crosses causal cutoff",
        )
        windows.append(window)
        lineage.append(
            {
                "window_id": window.window_id,
                "source_frame_start": int(entry["start_frame"]),
                "source_frame_stop_exclusive": int(entry["stop_frame"]),
                "payload_sha256": payload_sha256,
                "frame_indices_sha256": _array_sha256(window.frame_indices),
            }
        )
    return manifest, windows, lineage


def _frame_zero_alignment(
    first_window: Any,
    raw_case_dir: Path,
    *,
    camera_index: int,
    stride_pixels: int,
    trim_fraction: float,
    iterations: int,
) -> tuple[Any, Mapping[str, Any], Path]:
    from prob4d.sim3 import Sim3

    camera_points, reference_source = load_phystwin_world_point_grid(raw_case_dir, 0)
    with (raw_case_dir / "mask" / "processed_masks.pkl").open("rb") as handle:
        masks = pickle.load(handle)
    target_shape = first_window.point_map.shape[1:3]
    object_mask = resample_cover_grid(
        np.asarray(masks[0][camera_index]["object"]), target_shape
    ).astype(bool)
    initial_world = resample_cover_grid(camera_points[camera_index], target_shape)
    rows, columns = np.indices(target_shape)
    selected = (
        object_mask
        & first_window.valid_mask[0]
        & np.all(np.isfinite(first_window.point_map[0]), axis=2)
        & np.all(np.isfinite(initial_world), axis=2)
        & (np.linalg.norm(initial_world, axis=2) > 1e-6)
        & (rows % stride_pixels == 0)
        & (columns % stride_pixels == 0)
    )
    transform = robust_similarity_transform(
        first_window.point_map[0, selected],
        initial_world[selected],
        trim_fraction=trim_fraction,
        iterations=iterations,
    )
    sim3 = Sim3(
        scale=float(transform["scale"]),
        rotation=np.asarray(transform["rotation"], dtype=np.float64),
        translation=np.asarray(transform["translation"], dtype=np.float64),
    )
    reference_path = raw_case_dir / str(reference_source)
    return sim3, transform, reference_path


def _build_prob4d_bundle(
    *,
    case_id: str,
    windows: Sequence[Any],
    causal_frame_stop: int,
    metric_transform: Any,
    metric_covariance: np.ndarray,
    graph_initial_m: np.ndarray,
    baseline_m: np.ndarray,
    reserved_graph_nodes: np.ndarray,
    protocol: Mapping[str, Any],
    source_revision: str,
) -> tuple[Any, dict[tuple[str, int], GraphAssociation], Mapping[str, Any]]:
    from prob4d.alignment import align_windows, alignment_covariance_context
    from prob4d.causal_tracklets import (
        build_causal_scene_flow_tracklets,
        tracklets_to_observation_factors,
    )
    from prob4d.gauge import GaugeEstimate
    from prob4d.observation_export import estimate_joint_gauge_tree
    from prob4d.observation_factors import ObservationFactorBundle
    from prob4d.uncertainty import (
        DepthDisagreementModel,
        StructuredCovariance,
        accumulate_disagreement,
    )
    from scipy.spatial import cKDTree

    identity = protocol["identity"]
    physical = protocol["physical_state"]
    alignments = []
    with alignment_covariance_context(calibration=None, fallback_policy="pointwise"):
        for moving_position, moving in enumerate(windows):
            for reference_position, reference in enumerate(windows[:moving_position]):
                if len(reference.common_frames(moving)):
                    alignments.append(
                        align_windows(
                            reference,
                            moving,
                            seed=10_000 * moving_position + reference_position,
                        )
                    )
    posterior = estimate_joint_gauge_tree(
        windows,
        alignments,
        initial_transform=metric_transform,
        initial_covariance=metric_covariance,
    )
    evidence = accumulate_disagreement(
        {window.window_id: window for window in windows}, alignments
    )
    uncertainty_model = DepthDisagreementModel()

    allowed_mask = np.ones(len(graph_initial_m), dtype=bool)
    allowed_mask[np.asarray(reserved_graph_nodes, dtype=np.int64)] = False
    allowed_nodes = np.flatnonzero(allowed_mask)
    _require(len(allowed_nodes) >= 4, "reserved identities leave too few graph nodes")

    factors: list[Any] = []
    associations: dict[tuple[str, int], GraphAssociation] = {}
    reports: list[dict[str, Any]] = []
    for window in windows:
        disagreement = evidence[window.window_id]
        covariance = uncertainty_model.predict(window, disagreement)
        reliability = source_only_prior_reliability(
            disagreement.parallel_mean,
            disagreement.lateral_mean,
            covariance.parallel_variance,
            covariance.lateral_variance,
            disagreement.count,
            minimum=0.05,
        )
        transform = posterior.estimates[window.window_id]
        segment_length = int(identity["segment_length_frames"])
        segment_step = int(identity["segment_step_frames"])
        for segment_start in range(0, window.shape[0] - 1, segment_step):
            segment_stop = min(segment_start + segment_length, window.shape[0])
            if segment_stop - segment_start < 2:
                continue
            segment_name = f"{window.window_id}:segment-{segment_start:02d}"
            segment = _slice_window(
                window,
                segment_start,
                segment_stop,
                window_id=segment_name,
            )
            segment_covariance = StructuredCovariance(
                ray_directions=covariance.ray_directions[
                    segment_start:segment_stop
                ],
                parallel_variance=covariance.parallel_variance[
                    segment_start:segment_stop
                ],
                lateral_variance=covariance.lateral_variance[
                    segment_start:segment_stop
                ],
            )
            tracklets, report = build_causal_scene_flow_tracklets(
                segment,
                causal_frame_stop=causal_frame_stop,
                seed_stride=int(identity["seed_stride_pixels"]),
                search_radius_pixels=int(identity["search_radius_pixels"]),
                maximum_step_error_local=float(
                    identity["maximum_step_error_local"]
                ),
                association_sigma_local=float(identity["association_sigma_local"]),
                minimum_link_probability=float(
                    identity["minimum_link_probability"]
                ),
                minimum_track_length=int(identity["minimum_track_length"]),
                target_deform_mask_policy=str(
                    identity["target_deform_mask_policy"]
                ),
            )
            prefix = f"{case_id}:{segment_name}"
            segment_factors = tracklets_to_observation_factors(
                tracklets,
                segment_covariance,
                view_id="camera0",
                prior_reliability=reliability[segment_start:segment_stop],
                prior_nominal_probability=0.9,
                effective_samples_per_group=float(
                    physical["effective_samples_per_frame"]
                ),
                correlation_group_prefix=f"{case_id}:camera-frame",
                factor_id_prefix=prefix,
            )
            if segment_start:
                segment_factors = tuple(
                    factor
                    for factor in segment_factors
                    if factor.frame_index != tracklets.seed_frame_index
                )
            segment_factors = tuple(
                replace(
                    factor,
                    window_id=window.window_id,
                    gauge_id=window.window_id,
                )
                for factor in segment_factors
            )
            factors.extend(segment_factors)

            seed_rows = np.asarray(
                [
                    np.flatnonzero(tracklets.track_ids == track_id)[0]
                    for track_id in range(tracklets.track_count)
                ],
                dtype=np.int64,
            )
            seed_points_world = transform.transform_points(
                tracklets.points_local[seed_rows]
            )
            seed_frame = int(tracklets.seed_frame_index)
            tree = cKDTree(baseline_m[seed_frame, allowed_nodes])
            neighbour_count = min(
                int(physical["association_neighbour_count"]), len(allowed_nodes)
            )
            distances, local_indices = tree.query(
                seed_points_world, k=neighbour_count
            )
            if neighbour_count == 1:
                distances = distances[:, None]
                local_indices = local_indices[:, None]
            for track_id in range(tracklets.track_count):
                association = soft_graph_association_from_neighbours(
                    allowed_nodes[np.asarray(local_indices[track_id], dtype=np.int64)],
                    np.asarray(distances[track_id], dtype=np.float64),
                    distance_scale_m=float(physical["association_distance_scale_m"]),
                    maximum_distance_m=float(
                        physical["maximum_association_distance_m"]
                    ),
                )
                if association.nearest_distance_m > float(
                    physical["maximum_association_distance_m"]
                ):
                    association = replace(association, probability=0.0)
                associations[(prefix, track_id)] = association
            reports.append(
                {
                    "window_id": window.window_id,
                    "segment_start": segment_start,
                    "segment_stop": segment_stop,
                    **asdict(report),
                }
            )

    _require(bool(factors), "no persistent real-camera factors were constructed")
    group_counts: dict[str, int] = {}
    for factor in factors:
        group_counts[factor.correlation_group_id] = (
            group_counts.get(factor.correlation_group_id, 0)
            + int(np.count_nonzero(factor.valid_mask))
        )
    effective = float(physical["effective_samples_per_frame"])
    factors = [
        replace(
            factor,
            composite_weight=min(
                1.0, effective / group_counts[factor.correlation_group_id]
            ),
        )
        for factor in factors
    ]
    gauges = tuple(
        GaugeEstimate(
            window_id=window_id,
            global_from_local=posterior.estimates[window_id],
            covariance=posterior.joint_covariance[
                7 * index : 7 * (index + 1),
                7 * index : 7 * (index + 1),
            ],
        )
        for index, window_id in enumerate(posterior.window_ids)
    )
    bundle = ObservationFactorBundle(
        sequence_id=f"{case_id}:real-camera-validation-v1",
        case_id=case_id,
        stream_id="prob4d:rolling-causal-scene-flow-tracklets",
        factors=tuple(factors),
        gauges=gauges,
        source_revision=source_revision,
        causal_frame_stop=causal_frame_stop,
        joint_gauge_covariance=posterior.joint_covariance,
        gauge_covariance_semantics="joint-cross-window",
        metadata={
            "claim_bearing": False,
            "experiment": "retrospective-real-camera-validation-v1",
            "point_covariance": "uncalibrated-exploratory",
            "alignment_covariance_fallback": "pointwise",
        },
    )
    diagnostics = {
        "window_ids": list(posterior.window_ids),
        "joint_gauge_dimension": int(posterior.joint_covariance.shape[0]),
        "cross_window_gauge_covariance_preserved": bool(
            posterior.cross_window_covariance_preserved
        ),
        "alignment_count": len(alignments),
        "alignment_residual_rms_m": [
            float(value.result.residual_rms) for value in alignments
        ],
        "factor_count": len(factors),
        "tracklet_segments": reports,
    }
    return bundle, associations, diagnostics


def _row_prefix(factor_id: str) -> str:
    marker = ":frame-"
    if marker not in factor_id:
        raise ValueError("tracklet factor ID lost its frame suffix")
    return factor_id.rsplit(marker, 1)[0]


def _candidate_for_method(
    *,
    method_id: str,
    protocol: Mapping[str, Any],
    stack: Any,
    row_physical_prediction_m: np.ndarray,
    row_state_jacobian: np.ndarray,
    row_assignment_covariance_m2: np.ndarray,
    total_association_probability: np.ndarray,
    query_state_jacobian: np.ndarray,
    query_frame: int,
    physical_response_scale_m: float,
) -> Candidate:
    if method_id == "B0_physical_fallback":
        correction = np.zeros((len(query_state_jacobian), 3), dtype=np.float64)
        covariance = np.zeros((len(query_state_jacobian), 3, 3), dtype=np.float64)
        return Candidate(
            method_id=method_id,
            inference_admissible=False,
            reason="physical-fallback-reference",
            raw_correction_m=correction,
            covariance_m2=covariance,
            risk_score=1e300,
            guard_accepted=False,
            deployed_correction_m=correction.copy(),
            diagnostics={"exact_fallback": True},
        )

    selected = np.ones(len(stack.world_mean_m), dtype=bool)
    if method_id == "P2_explicit_gauge_framewise":
        selected &= np.asarray(stack.frame_indices) == query_frame
    selected &= total_association_probability >= float(
        protocol["physical_state"]["minimum_total_association_probability"]
    )
    if not np.any(selected):
        correction = np.zeros((len(query_state_jacobian), 3), dtype=np.float64)
        covariance = np.zeros((len(query_state_jacobian), 3, 3), dtype=np.float64)
        return Candidate(
            method_id=method_id,
            inference_admissible=False,
            reason="no-associated-camera-rows",
            raw_correction_m=correction,
            covariance_m2=covariance,
            risk_score=1e300,
            guard_accepted=False,
            deployed_correction_m=correction.copy(),
            diagnostics={"selected_observation_count": 0, "exact_fallback": True},
        )

    if method_id == "P1_marginal_gauge_persistent":
        observation_covariance = (
            np.asarray(stack.marginal_world_covariance_m2)[selected]
            + row_assignment_covariance_m2[selected]
        )
        gauge_jacobian = np.zeros((np.count_nonzero(selected), 3, 0))
        gauge_prior = np.zeros((0, 0))
        gauge_semantics = "marginal-rowwise"
    else:
        observation_covariance = (
            np.asarray(stack.conditional_world_covariance_m2)[selected]
            + row_assignment_covariance_m2[selected]
        )
        gauge_jacobian = np.asarray(stack.gauge_jacobian)[selected]
        gauge_prior = np.asarray(stack.gauge_prior_covariance)
        gauge_semantics = "explicit-joint-cross-window"

    selected_groups = tuple(
        group
        for group, keep in zip(
            stack.correlation_group_ids, selected, strict=True
        )
        if keep
    )
    composite = group_capped_composite_weights(
        selected_groups,
        total_association_probability[selected],
        effective_samples_per_group=float(
            protocol["physical_state"]["effective_samples_per_frame"]
        ),
    )
    physical = protocol["physical_state"]
    state_count = query_state_jacobian.shape[2]
    batch = GaugeAwareObservationBatch(
        innovation_m=(
            np.asarray(stack.world_mean_m)[selected]
            - row_physical_prediction_m[selected]
        ),
        observation_covariance_m2=observation_covariance,
        state_jacobian=row_state_jacobian[selected],
        gauge_jacobian=gauge_jacobian,
        shared_bias_jacobian=global_translation_bias_jacobian(
            int(np.count_nonzero(selected))
        ),
        view_bias_jacobian=np.zeros((np.count_nonzero(selected), 3, 0)),
        query_state_jacobian=query_state_jacobian,
        gauge_prior_covariance=gauge_prior,
        correlation_group_ids=selected_groups,
        prior_reliability=np.asarray(stack.prior_reliability)[selected],
        prior_nominal_probability=np.asarray(
            stack.prior_nominal_probability
        )[selected],
        composite_weight=composite,
        state_prior_covariance_m2=(
            np.eye(state_count) * float(physical["state_prior_std_m"]) ** 2
        ),
        physical_response_scale_m=physical_response_scale_m,
        composite_weight_mode=COMPOSITE_WEIGHT_MODE_PROVIDER_FINAL,
        metadata={
            "experiment": "prob4d-real-camera-validation-v1",
            "method_id": method_id,
            "gauge_covariance_semantics": gauge_semantics,
            "association_probability_semantics": (
                "generalized-Bayes-group-power-not-prior-reliability"
            ),
            "assignment_mixture_spread_in_metric_covariance": True,
            "innovation_formed_once": True,
        },
    )
    result = update_prior_aware_gauge_belief(
        batch,
        config=PriorAwareGaugeConfigV1(
            state_prior_std_m=float(physical["state_prior_std_m"]),
            shared_bias_prior_std_m=float(physical["shared_bias_prior_std_m"]),
            view_bias_prior_std_m=float(physical["view_bias_prior_std_m"]),
            effective_samples_per_correlation_group=float(
                physical["effective_samples_per_frame"]
            ),
            degrees_of_freedom=5.0,
            outlier_covariance_multiplier=36.0,
            maximum_iterations=20,
            maximum_condition_number=1e13,
            minimum_conditional_information_fraction=1e-5,
            minimum_identifiable_fraction=0.02,
            minimum_query_sensitivity_fraction=1e-4,
            maximum_state_update_m=float(physical["maximum_state_update_m"]),
            maximum_update_to_physical_response_ratio=float(
                physical["maximum_update_to_physical_response_ratio"]
            ),
        ),
    )
    raw_correction = (
        np.einsum(
            "ncs,s->nc",
            query_state_jacobian,
            result.state_coefficients,
            optimize=True,
        )
        if result.inference_admissible
        else np.zeros((len(query_state_jacobian), 3), dtype=np.float64)
    )
    state_covariance = np.asarray(result.posterior_covariance)[
        :state_count, :state_count
    ]
    covariance = query_covariance(query_state_jacobian, state_covariance)
    risk, risk_diagnostics = _risk_score(
        result,
        covariance,
        physical_response_scale_m=physical_response_scale_m,
    )
    threshold = float(protocol["methods"][method_id]["risk_threshold"])
    guard_accepted, deployed = exact_fallback_selection(
        raw_correction,
        inference_admissible=bool(result.inference_admissible),
        risk_score=risk,
        risk_threshold=threshold,
    )
    diagnostics = {
        "selected_observation_count": int(np.count_nonzero(selected)),
        "risk_threshold": threshold,
        "risk": risk_diagnostics,
        "solver": _json_value(result.diagnostics),
        "mean_total_association_probability": float(
            np.mean(total_association_probability[selected])
        ),
        "mean_prior_reliability": float(
            np.mean(np.asarray(stack.prior_reliability)[selected])
        ),
        "exact_fallback": bool(
            guard_accepted or np.array_equal(deployed, np.zeros_like(deployed))
        ),
    }
    return Candidate(
        method_id=method_id,
        inference_admissible=bool(result.inference_admissible),
        reason=str(result.reason),
        raw_correction_m=raw_correction,
        covariance_m2=covariance,
        risk_score=risk,
        guard_accepted=guard_accepted,
        deployed_correction_m=deployed,
        diagnostics=diagnostics,
    )


def _prepare_case(
    *,
    case_id: str,
    manifest_path: Path,
    case_data_dir: Path,
    output_dir: Path,
    protocol: Mapping[str, Any],
    prob4d_revision: str,
) -> tuple[Path, Path]:
    manifest, windows, lineage = _load_selected_windows(
        manifest_path,
        count=int(protocol["prob4d"]["complete_overlap_window_count"]),
    )
    causal_frame_stop = int(lineage[-1]["source_frame_stop_exclusive"])
    query_frame = causal_frame_stop - 1
    video_path = Path(str(manifest["video_path"]))
    raw_case_dir = video_path.parent.parent
    _require(raw_case_dir.is_dir(), "real PhysTwin case directory is missing")

    final_path = case_data_dir / "final_data.pkl"
    baseline_path = case_data_dir / "inference.pkl"
    optimal_path = case_data_dir / "optimal_params.pkl"
    manual_path = case_data_dir / "gt_track_3d.pkl"
    for path in (final_path, baseline_path, optimal_path, manual_path):
        _require(path.is_file(), f"required case artifact is missing: {path}")

    data = _load_pickle(final_path)
    baseline = np.asarray(_load_pickle(baseline_path), dtype=np.float64)
    optimal = _load_pickle(optimal_path)
    # The pickle is monolithic. Only frame zero crosses into candidate construction;
    # the complete value is reloaded after the prediction has been written.
    manual_frame_zero = np.asarray(_load_pickle(manual_path), dtype=np.float64)[0]
    _require(query_frame < len(baseline), "query frame exceeds physical trajectory")

    observed_frame_zero = np.asarray(data["object_points"], dtype=np.float64)[0]
    surface_points = np.asarray(data["surface_points"], dtype=np.float64)
    interior_points = np.asarray(data["interior_points"], dtype=np.float64)
    structure = np.concatenate(
        (observed_frame_zero, surface_points, interior_points), axis=0
    )
    _require(baseline.shape[1] == len(structure), "graph and trajectory disagree")
    surface_count = len(observed_frame_zero) + len(surface_points)
    graph_initial = structure[:surface_count]
    graph = build_phystwin_spring_graph(
        structure,
        None,
        config=PhysTwinSpringGraphConfig(
            object_radius=float(optimal["object_radius"]),
            object_max_neighbours=int(optimal["object_max_neighbours"]),
            controller_radius=float(optimal["controller_radius"]),
            controller_max_neighbours=int(optimal["controller_max_neighbours"]),
        ),
    )
    surface_springs = graph.springs[np.all(graph.springs < surface_count, axis=1)]
    basis, frequencies, basis_diagnostics = _basis_with_metric_coefficients(
        graph_initial,
        surface_springs,
        rank=int(protocol["physical_state"]["graph_rank"]),
    )

    eligible_manual = np.flatnonzero(np.all(np.isfinite(manual_frame_zero), axis=1))
    reserve_count = max(
        1,
        min(
            len(eligible_manual) - 1,
            int(
                math.ceil(
                    len(eligible_manual)
                    * float(protocol["identity"]["reserved_manual_identity_fraction"])
                )
            ),
        ),
    )
    _require(reserve_count >= 1, "case cannot reserve a manual identity")
    reserved_indices = deterministic_farthest_point_indices(
        manual_frame_zero,
        eligible_manual,
        reserve_count,
    )
    manual_nodes = _nearest_manual_nodes(graph_initial, manual_frame_zero)
    reserved_nodes = np.unique(manual_nodes[reserved_indices])

    metric_transform, alignment, anchor_source_path = _frame_zero_alignment(
        windows[0],
        raw_case_dir,
        camera_index=int(protocol["prob4d"]["camera_index"]),
        stride_pixels=int(protocol["prob4d"]["alignment_stride_pixels"]),
        trim_fraction=float(protocol["prob4d"]["alignment_trim_fraction"]),
        iterations=int(protocol["prob4d"]["alignment_iterations"]),
    )
    anchor_covariance = _metric_anchor_covariance(
        float(alignment["inlier_rmse_m"]),
        float(optimal["object_radius"]),
    )
    bundle, graph_associations, provider_diagnostics = _build_prob4d_bundle(
        case_id=case_id,
        windows=windows,
        causal_frame_stop=causal_frame_stop,
        metric_transform=metric_transform,
        metric_covariance=anchor_covariance,
        graph_initial_m=graph_initial,
        baseline_m=baseline[:, :surface_count],
        reserved_graph_nodes=reserved_nodes,
        protocol=protocol,
        source_revision=prob4d_revision,
    )
    stack = bundle.stack()
    row_physical = np.empty_like(stack.world_mean_m)
    row_state = np.empty(
        (len(stack.world_mean_m), 3, basis.shape[2]), dtype=np.float64
    )
    row_assignment_covariance = np.empty(
        (len(stack.world_mean_m), 3, 3), dtype=np.float64
    )
    graph_probability = np.empty(len(stack.world_mean_m), dtype=np.float64)
    for row, (factor_id, point_id, frame) in enumerate(
        zip(
            stack.factor_ids,
            stack.point_ids,
            stack.frame_indices,
            strict=True,
        )
    ):
        association = graph_associations[(_row_prefix(factor_id), int(point_id))]
        candidates = baseline[int(frame), association.node_indices]
        row_physical[row] = np.sum(
            association.weights[:, None] * candidates, axis=0
        )
        row_state[row] = np.einsum(
            "n,ncs->cs",
            association.weights,
            basis[association.node_indices],
            optimize=True,
        )
        row_assignment_covariance[row] = assignment_mixture_covariance(
            candidates,
            association.weights,
            floor_m2=float(
                protocol["physical_state"]["assignment_covariance_floor_m2"]
            ),
        )
        graph_probability[row] = association.probability
    total_association = np.asarray(stack.association_probability) * graph_probability

    query_nodes = manual_nodes[reserved_indices]
    query_state = basis[query_nodes]
    baseline_query = baseline[query_frame, query_nodes]
    physical_response_scale = max(
        float(
            np.sqrt(
                np.mean(
                    np.square(
                        baseline[query_frame, query_nodes]
                        - baseline[0, query_nodes]
                    )
                )
            )
        ),
        0.001,
    )
    candidates = {
        method: _candidate_for_method(
            method_id=method,
            protocol=protocol,
            stack=stack,
            row_physical_prediction_m=row_physical,
            row_state_jacobian=row_state,
            row_assignment_covariance_m2=row_assignment_covariance,
            total_association_probability=total_association,
            query_state_jacobian=query_state,
            query_frame=query_frame,
            physical_response_scale_m=physical_response_scale,
        )
        for method in METHODS
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_npz = output_dir / "prediction.npz"
    np.savez_compressed(
        prediction_npz,
        query_frame=np.asarray(query_frame, dtype=np.int64),
        reserved_manual_indices=reserved_indices,
        reserved_graph_nodes=query_nodes,
        baseline_query_m=baseline_query,
        **{
            f"{method}__raw_correction_m": candidate.raw_correction_m
            for method, candidate in candidates.items()
        },
        **{
            f"{method}__deployed_correction_m": candidate.deployed_correction_m
            for method, candidate in candidates.items()
        },
        **{
            f"{method}__covariance_m2": candidate.covariance_m2
            for method, candidate in candidates.items()
        },
    )
    prediction_record = {
        "schema": "bayesian-phystwin-prob4d-real-camera-case-prediction",
        "schema_version": 1,
        "case_id": case_id,
        "query_frame": query_frame,
        "causal_frame_stop_exclusive": causal_frame_stop,
        "future_motioncrafter_payloads_opened": 0,
        "query_manual_coordinates_used": False,
        "manual_frame_zero_used_only_for_reserved_identity_mapping": True,
        "manual_track_pickle_is_monolithic": True,
        "reserved_manual_indices": reserved_indices.tolist(),
        "reserved_graph_nodes": query_nodes.tolist(),
        "physical_response_scale_m": physical_response_scale,
        "selected_window_lineage": lineage,
        "legacy_temporal_lineage_reconstructed": (
            "temporal_lineage" not in manifest
        ),
        "metric_anchor": {
            "source_path": str(anchor_source_path),
            "source_sha256": _sha256(anchor_source_path),
            "inlier_rmse_m": float(alignment["inlier_rmse_m"]),
            "scale": float(metric_transform.scale),
            "rotation": metric_transform.rotation.tolist(),
            "translation": metric_transform.translation.tolist(),
            "covariance": anchor_covariance.tolist(),
        },
        "graph": {
            "surface_node_count": surface_count,
            "surface_spring_count": int(len(surface_springs)),
            "basis_frequencies": frequencies.tolist(),
            "basis_diagnostics": _json_value(basis_diagnostics),
        },
        "provider": _json_value(provider_diagnostics),
        "rows": {
            "observation_count": int(len(stack.world_mean_m)),
            "mean_prior_reliability": float(np.mean(stack.prior_reliability)),
            "mean_tracklet_association_probability": float(
                np.mean(stack.association_probability)
            ),
            "mean_graph_association_probability": float(
                np.mean(graph_probability)
            ),
            "mean_total_association_probability": float(
                np.mean(total_association)
            ),
        },
        "methods": {
            method: {
                "inference_admissible": candidate.inference_admissible,
                "reason": candidate.reason,
                "risk_score": candidate.risk_score,
                "guard_accepted": candidate.guard_accepted,
                "diagnostics": _json_value(candidate.diagnostics),
            }
            for method, candidate in candidates.items()
        },
        "inputs": {
            "prediction_manifest": {
                "path": str(manifest_path),
                "sha256": _sha256(manifest_path),
            },
            "final_data": {"path": str(final_path), "sha256": _sha256(final_path)},
            "baseline": {
                "path": str(baseline_path),
                "sha256": _sha256(baseline_path),
            },
            "optimal_params": {
                "path": str(optimal_path),
                "sha256": _sha256(optimal_path),
            },
            "manual_tracks": {
                "path": str(manual_path),
                "sha256": _sha256(manual_path),
                "frame_zero_sha256": _array_sha256(manual_frame_zero),
            },
            "raw_case_dir": str(raw_case_dir),
            "video_path": str(video_path),
        },
        "prediction_npz": {
            "path": str(prediction_npz),
            "sha256": _sha256(prediction_npz),
        },
    }
    prediction_json = output_dir / "prediction.json"
    _write_json(prediction_json, prediction_record)
    return prediction_json, manual_path


def _rmse(prediction: np.ndarray, truth: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(prediction - truth))))


def _coverage(
    prediction: np.ndarray,
    truth: np.ndarray,
    covariance_m2: np.ndarray,
) -> tuple[float, float]:
    covered: list[bool] = []
    widths: list[float] = []
    for estimate, target, covariance in zip(
        prediction, truth, covariance_m2, strict=True
    ):
        matrix = 0.5 * (covariance + covariance.T) + np.eye(3) * 1e-12
        residual = target - estimate
        nees = float(residual @ np.linalg.solve(matrix, residual))
        covered.append(nees <= CHI_SQUARE_3_90)
        widths.append(float(np.sqrt(np.trace(matrix))))
    return float(np.mean(covered)), float(np.mean(widths))


def _score_case(
    prediction_json: Path,
    manual_path: Path,
    *,
    harmful_margin_m: float,
) -> Mapping[str, Any]:
    prediction_record = json.loads(prediction_json.read_text(encoding="utf-8"))
    prediction_npz = Path(prediction_record["prediction_npz"]["path"])
    _require(
        _sha256(prediction_npz) == prediction_record["prediction_npz"]["sha256"],
        "prediction artifact changed before scoring",
    )
    with np.load(prediction_npz, allow_pickle=False) as payload:
        query_frame = int(payload["query_frame"])
        reserved = np.asarray(payload["reserved_manual_indices"], dtype=np.int64)
        baseline_query = np.asarray(payload["baseline_query_m"], dtype=np.float64)
        manual = np.asarray(_load_pickle(manual_path), dtype=np.float64)
        truth = manual[query_frame, reserved]
        finite = np.all(np.isfinite(truth), axis=1)
        _require(np.any(finite), "reserved query identities are all unavailable")
        truth = truth[finite]
        baseline_query = baseline_query[finite]
        baseline_rmse = _rmse(baseline_query, truth)
        methods: dict[str, Any] = {}
        for method in METHODS:
            raw_correction = np.asarray(
                payload[f"{method}__raw_correction_m"], dtype=np.float64
            )[finite]
            deployed_correction = np.asarray(
                payload[f"{method}__deployed_correction_m"], dtype=np.float64
            )[finite]
            covariance = np.asarray(
                payload[f"{method}__covariance_m2"], dtype=np.float64
            )[finite]
            raw_prediction = baseline_query + raw_correction
            deployed_prediction = baseline_query + deployed_correction
            raw_rmse = _rmse(raw_prediction, truth)
            deployed_rmse = _rmse(deployed_prediction, truth)
            accepted = bool(prediction_record["methods"][method]["guard_accepted"])
            coverage, width = (
                _coverage(raw_prediction, truth, covariance)
                if method != "B0_physical_fallback"
                and prediction_record["methods"][method]["inference_admissible"]
                else (None, None)
            )
            methods[method] = {
                "baseline_rmse_m": baseline_rmse,
                "raw_rmse_m": raw_rmse,
                "deployed_rmse_m": deployed_rmse,
                "raw_improvement_fraction": 1.0 - raw_rmse / baseline_rmse,
                "deployed_improvement_fraction": 1.0
                - deployed_rmse / baseline_rmse,
                "raw_harmful": raw_rmse > baseline_rmse + harmful_margin_m,
                "deployed_harmful": (
                    accepted and deployed_rmse > baseline_rmse + harmful_margin_m
                ),
                "guard_accepted": accepted,
                "coverage_90": coverage,
                "predictive_width_rms_m": width,
                "exact_fallback": bool(
                    accepted
                    or np.array_equal(
                        deployed_correction, np.zeros_like(deployed_correction)
                    )
                ),
            }
    score = {
        "schema": "bayesian-phystwin-prob4d-real-camera-case-score",
        "schema_version": 1,
        "case_id": prediction_record["case_id"],
        "prediction_sha256": _sha256(prediction_json),
        "query_frame": query_frame,
        "scored_reserved_identity_count": int(np.count_nonzero(finite)),
        "methods": methods,
    }
    score_path = prediction_json.parent / "score.json"
    _write_json(score_path, score)
    return score


def _paired_interval(
    differences: np.ndarray,
    *,
    resamples: int,
    seed: int,
) -> tuple[float, float]:
    values = np.asarray(differences, dtype=np.float64)
    rng = np.random.default_rng(seed)
    sampled = np.empty(resamples, dtype=np.float64)
    for index in range(resamples):
        selected = rng.integers(0, len(values), size=len(values))
        sampled[index] = float(np.mean(values[selected]))
    lower, upper = np.quantile(sampled, [0.025, 0.975])
    return float(lower), float(upper)


def _aggregate(
    scores: Sequence[Mapping[str, Any]],
    *,
    protocol: Mapping[str, Any],
) -> Mapping[str, Any]:
    aggregate: dict[str, Any] = {}
    for method in METHODS:
        baseline = np.asarray(
            [score["methods"][method]["baseline_rmse_m"] for score in scores],
            dtype=np.float64,
        )
        raw = np.asarray(
            [score["methods"][method]["raw_rmse_m"] for score in scores],
            dtype=np.float64,
        )
        deployed = np.asarray(
            [score["methods"][method]["deployed_rmse_m"] for score in scores],
            dtype=np.float64,
        )
        accepted = np.asarray(
            [score["methods"][method]["guard_accepted"] for score in scores],
            dtype=bool,
        )
        coverage_values = [
            score["methods"][method]["coverage_90"]
            for score in scores
            if score["methods"][method]["guard_accepted"]
            and score["methods"][method]["coverage_90"] is not None
        ]
        aggregate[method] = {
            "case_count": len(scores),
            "baseline_mean_rmse_m": float(np.mean(baseline)),
            "raw_mean_rmse_m": float(np.mean(raw)),
            "deployed_mean_rmse_m": float(np.mean(deployed)),
            "raw_improvement_fraction": float(1.0 - np.mean(raw) / np.mean(baseline)),
            "deployed_improvement_fraction": float(
                1.0 - np.mean(deployed) / np.mean(baseline)
            ),
            "raw_win_count": int(np.sum(raw < baseline)),
            "deployed_win_count": int(np.sum(deployed < baseline)),
            "accepted_case_count": int(np.sum(accepted)),
            "harmful_accepted_count": int(
                np.sum(
                    [score["methods"][method]["deployed_harmful"] for score in scores]
                )
            ),
            "harmful_accepted_rate": float(
                np.mean(
                    [
                        score["methods"][method]["deployed_harmful"]
                        for score in scores
                        if score["methods"][method]["guard_accepted"]
                    ]
                )
                if np.any(accepted)
                else 0.0
            ),
            "accepted_coverage_90_mean": (
                None if not coverage_values else float(np.mean(coverage_values))
            ),
            "all_rejections_exact_fallback": bool(
                all(score["methods"][method]["exact_fallback"] for score in scores)
            ),
            "paired_deployed_minus_baseline_95_m": list(
                _paired_interval(
                    deployed - baseline,
                    resamples=int(protocol["bootstrap"]["resamples"]),
                    seed=int(protocol["bootstrap"]["seed"]),
                )
            ),
        }
    return aggregate


def _advancement_decision(
    aggregate: Mapping[str, Any],
    *,
    protocol: Mapping[str, Any],
    complete_case_count: int,
) -> Mapping[str, Any]:
    primary = aggregate[PRIMARY_METHOD]
    gates = protocol["advancement_gates"]
    checks = {
        "complete_expected_cohort": complete_case_count
        == int(protocol["cohort"]["expected_case_count"]),
        "deployed_mean_improvement": primary["deployed_improvement_fraction"]
        >= float(gates["deployed_mean_improvement_fraction_at_least"]),
        "paired_upper_below_zero": primary[
            "paired_deployed_minus_baseline_95_m"
        ][1]
        < 0.0,
        "harmful_accepted_rate": primary["harmful_accepted_rate"]
        <= float(gates["harmful_accepted_rate_at_most"]),
        "accepted_case_count": primary["accepted_case_count"]
        >= int(gates["accepted_case_count_at_least"]),
        "accepted_coverage": primary["accepted_coverage_90_mean"] is not None
        and primary["accepted_coverage_90_mean"]
        >= float(gates["accepted_coverage_90_at_least"]),
        "exact_fallback": bool(primary["all_rejections_exact_fallback"]),
    }
    passed = all(checks.values())
    return {
        "primary_method": PRIMARY_METHOD,
        "checks": checks,
        "passed": passed,
        "decision": (
            "justify-fresh-independent-real-camera-protocol"
            if passed
            else "do-not-advance-from-retrospective-real-camera-transfer"
        ),
    }


def _write_cases_csv(
    path: Path,
    scores: Sequence[Mapping[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "case_id",
                "method_id",
                "baseline_rmse_mm",
                "raw_rmse_mm",
                "deployed_rmse_mm",
                "raw_improvement_percent",
                "deployed_improvement_percent",
                "guard_accepted",
                "coverage_90",
            ],
        )
        writer.writeheader()
        for score in scores:
            for method in METHODS:
                values = score["methods"][method]
                writer.writerow(
                    {
                        "case_id": score["case_id"],
                        "method_id": method,
                        "baseline_rmse_mm": 1000.0
                        * values["baseline_rmse_m"],
                        "raw_rmse_mm": 1000.0 * values["raw_rmse_m"],
                        "deployed_rmse_mm": 1000.0
                        * values["deployed_rmse_m"],
                        "raw_improvement_percent": 100.0
                        * values["raw_improvement_fraction"],
                        "deployed_improvement_percent": 100.0
                        * values["deployed_improvement_fraction"],
                        "guard_accepted": values["guard_accepted"],
                        "coverage_90": values["coverage_90"],
                    }
                )


def _write_markdown(path: Path, report: Mapping[str, Any]) -> None:
    if not report["aggregate"]:
        path.write_text(
            "# Prob4D real-camera validation v1 result\n\n"
            f"Decision: **{report['decision']['decision']}**.\n\n"
            f"No cases were scorable; {report['technical_failure_count']} retained "
            "technical failure(s) are recorded in `report.json`.\n",
            encoding="utf-8",
        )
        return
    lines = [
        "# Prob4D real-camera validation v1 result",
        "",
        f"Decision: **{report['decision']['decision']}**.",
        "",
        "| Method | Deployed RMSE | Change vs physical | Wins | Accepted | Coverage |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for method in METHODS:
        values = report["aggregate"][method]
        coverage = values["accepted_coverage_90_mean"]
        lines.append(
            "| {method} | {rmse:.3f} mm | {change:+.2f}% | {wins}/{count} | "
            "{accepted}/{count} | {coverage} |".format(
                method=method,
                rmse=1000.0 * values["deployed_mean_rmse_m"],
                change=-100.0 * values["deployed_improvement_fraction"],
                wins=values["deployed_win_count"],
                count=values["case_count"],
                accepted=values["accepted_case_count"],
                coverage=("n/a" if coverage is None else f"{100.0 * coverage:.1f}%"),
            )
        )
    lines.extend(
        [
            "",
            "This is a retrospective real-camera mechanism-transfer result on a "
            "previously opened cohort. It is not prospective physical confirmation, "
            "future prediction, or a state-of-the-art claim.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def execute(args: argparse.Namespace) -> int:
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    _require(protocol.get("schema") == PROTOCOL_SCHEMA, "unexpected protocol schema")
    _require(protocol.get("schema_version") == 1, "unexpected protocol version")
    _require(
        args.prob4d_revision == protocol["prob4d"]["revision"],
        "Prob4D revision differs from the protocol",
    )
    if args.output_dir.exists():
        if not args.force:
            raise FileExistsError(f"output already exists: {args.output_dir}")
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True)
    shutil.copy2(args.protocol, args.output_dir / "protocol.json")

    runner_root = args.data_root / str(protocol["cohort"]["runner_directory"])
    cases = sorted(path.name for path in runner_root.iterdir() if path.is_dir())
    if args.case is not None:
        _require(args.case in cases, f"unknown case {args.case!r}")
        cases = [args.case]
    scores: list[Mapping[str, Any]] = []
    failures: list[dict[str, str]] = []
    for case_id in cases:
        case_output = args.output_dir / "cases" / case_id
        try:
            prediction_json, manual_path = _prepare_case(
                case_id=case_id,
                manifest_path=(
                    runner_root
                    / case_id
                    / "camera0_prob4d_uniform_o8"
                    / "bundle"
                    / "predictions.json"
                ),
                case_data_dir=args.case_data_root / case_id,
                output_dir=case_output,
                protocol=protocol,
                prob4d_revision=args.prob4d_revision,
            )
            scores.append(
                _score_case(
                    prediction_json,
                    manual_path,
                    harmful_margin_m=float(protocol["scoring"]["harmful_margin_m"]),
                )
            )
        except Exception as error:  # retained technical failure, never replaced
            failures.append(
                {
                    "case_id": case_id,
                    "error_type": type(error).__name__,
                    "message": str(error),
                }
            )
            _write_json(
                case_output / "technical_failure.json",
                {
                    "case_id": case_id,
                    "retained_without_replacement": True,
                    "error_type": type(error).__name__,
                    "message": str(error),
                },
            )

    aggregate = _aggregate(scores, protocol=protocol) if scores else {}
    decision = (
        _advancement_decision(
            aggregate,
            protocol=protocol,
            complete_case_count=len(scores),
        )
        if scores
        else {
            "primary_method": PRIMARY_METHOD,
            "checks": {},
            "passed": False,
            "decision": "no-scorable-real-camera-cases",
        }
    )
    report = {
        "schema": REPORT_SCHEMA,
        "schema_version": 1,
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": _canonical_sha256(protocol),
        "repository_revision": args.repository_revision,
        "prob4d_revision": args.prob4d_revision,
        "case_count_requested": len(cases),
        "case_count_scored": len(scores),
        "technical_failure_count": len(failures),
        "technical_failures": failures,
        "aggregate": aggregate,
        "decision": decision,
        "claim_boundary": protocol["claim_boundary"],
    }
    _write_json(args.output_dir / "report.json", report)
    _write_cases_csv(args.output_dir / "cases.csv", scores)
    _write_markdown(args.output_dir / "summary.md", report)
    checksums = []
    for path in sorted(args.output_dir.rglob("*")):
        if path.is_file() and path.name != "SHA256SUMS":
            checksums.append(f"{_sha256(path)}  {path.relative_to(args.output_dir)}")
    (args.output_dir / "SHA256SUMS").write_text(
        "\n".join(checksums) + "\n", encoding="utf-8"
    )
    return 0 if scores else 2


def main() -> None:
    raise SystemExit(execute(_parse_args()))


if __name__ == "__main__":
    main()
