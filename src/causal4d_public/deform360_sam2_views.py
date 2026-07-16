"""Cross-view reliability gate for public SAM2 Deform360 rope masks."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .deform360_sam2 import (
    PINNED_SAM2_CHECKPOINT_SHA256,
    PINNED_SAM2_CHECKPOINT_URL,
    PINNED_SAM2_COMMIT,
    PINNED_SAM2_MODEL_CONFIG,
    PINNED_SAM2_REPOSITORY,
)


DEFORM360_SAM2_VIEW_AUDIT_SCHEMA_VERSION = 1


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


@dataclass(frozen=True)
class CrossViewMaskReliabilityConfig:
    """Frozen source-only visual-hull agreement thresholds."""

    cube_half_extent_m: float = 0.5
    voxel_resolution: int = 80
    consensus_fraction_of_peak: float = 0.55
    minimum_consensus_votes: int = 8
    minimum_leave_one_out_recall: float = 0.60

    def __post_init__(self) -> None:
        _require(self.cube_half_extent_m > 0.0, "cube extent must be positive")
        _require(self.voxel_resolution >= 16, "voxel resolution is too small")
        _require(
            0.0 < self.consensus_fraction_of_peak <= 1.0,
            "invalid consensus fraction",
        )
        _require(
            self.minimum_consensus_votes >= 2,
            "minimum consensus votes must be at least two",
        )
        _require(
            0.0 <= self.minimum_leave_one_out_recall <= 1.0,
            "invalid leave-one-out recall threshold",
        )


@dataclass(frozen=True)
class JointMultiviewMaskSelectionConfig:
    """Appearance-first candidate selection with calibrated 3D feasibility."""

    maximum_candidates_per_camera: int = 4
    voxel_resolution: int = 32
    coordinate_descent_passes: int = 4
    appearance_weight: float = 0.05
    projected_volume_penalty: float = 0.10

    def __post_init__(self) -> None:
        _require(
            self.maximum_candidates_per_camera >= 1,
            "maximum candidate count must be positive",
        )
        _require(self.voxel_resolution >= 16, "selection grid is too small")
        _require(
            self.coordinate_descent_passes >= 1,
            "coordinate-descent pass count must be positive",
        )
        _require(self.appearance_weight >= 0.0, "appearance weight is negative")
        _require(
            self.projected_volume_penalty >= 0.0,
            "projected-volume penalty is negative",
        )


def _world_grid_for_cameras(
    cameras: Sequence[str],
    camera_to_world_by_camera: Mapping[str, np.ndarray],
    *,
    cube_half_extent_m: float,
    voxel_resolution: int,
) -> np.ndarray:
    centers = []
    for camera in cameras:
        transform = np.asarray(camera_to_world_by_camera[camera], dtype=np.float64)
        _require(transform.shape == (4, 4), f"invalid extrinsics for {camera}")
        _require(np.isfinite(transform).all(), f"non-finite extrinsics for {camera}")
        centers.append(transform[:3, 3])
    center = np.mean(centers, axis=0)
    axis = np.linspace(-cube_half_extent_m, cube_half_extent_m, voxel_resolution)
    return (
        np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1).reshape(-1, 3)
        + center
    )


def _mask_hits_on_world_grid(
    mask: np.ndarray,
    intrinsics: np.ndarray,
    camera_to_world: np.ndarray,
    grid_world_m: np.ndarray,
    *,
    camera: str,
) -> np.ndarray:
    candidate = np.asarray(mask, dtype=bool)
    _require(candidate.ndim == 2, f"mask for {camera} must be 2D")
    height, width = candidate.shape
    calibration = np.asarray(intrinsics, dtype=np.float64)
    _require(calibration.shape == (3, 3), f"invalid intrinsics for {camera}")
    world_to_camera = np.linalg.inv(np.asarray(camera_to_world, dtype=np.float64))
    points = grid_world_m @ world_to_camera[:3, :3].T + world_to_camera[:3, 3]
    depth = points[:, 2]
    in_front = depth > 1e-6
    safe_depth = np.where(in_front, depth, 1.0)
    u = points[:, 0] / safe_depth * calibration[0, 0] + calibration[0, 2]
    v = points[:, 1] / safe_depth * calibration[1, 1] + calibration[1, 2]
    in_bounds = in_front & (u >= 0.0) & (u < width) & (v >= 0.0) & (v < height)
    columns = np.clip(u, 0, width - 1).astype(np.int64)
    rows = np.clip(v, 0, height - 1).astype(np.int64)
    return in_bounds & candidate[rows, columns]


def _joint_candidate_objective(
    selected_hits: np.ndarray,
    selected_prior_scores: np.ndarray,
    *,
    minimum_consensus_votes: int,
    appearance_weight: float,
    projected_volume_penalty: float,
) -> tuple[float, dict[str, float]]:
    hits = np.asarray(selected_hits, dtype=bool)
    priors = np.asarray(selected_prior_scores, dtype=np.float64)
    _require(hits.ndim == 2 and len(hits) >= 3, "invalid selected hit matrix")
    _require(priors.shape == (len(hits),), "selected prior-score shape mismatch")
    votes = hits.sum(axis=0).astype(np.float64)
    total_hits = float(votes.sum())
    camera_count = len(hits)
    if total_hits:
        pairwise_agreement = float(
            np.sum(votes * np.maximum(votes - 1.0, 0.0))
            / ((camera_count - 1.0) * total_hits)
        )
    else:
        pairwise_agreement = 0.0
    peak_fraction = float(votes.max(initial=0.0) / camera_count)
    required_votes = min(camera_count, minimum_consensus_votes)
    supported = votes >= 2.0
    core = votes >= required_votes
    core_fraction = float(np.count_nonzero(core) / max(np.count_nonzero(supported), 1))
    projected_volume_fraction = float(np.mean(hits))
    appearance = float(np.mean(priors))
    log_appearance = float(np.mean(np.log(np.maximum(priors, 1e-12))))
    objective = float(
        2.0 * peak_fraction
        + 2.0 * pairwise_agreement
        + 0.25 * core_fraction
        + appearance_weight * appearance
        - projected_volume_penalty * projected_volume_fraction
    )
    return objective, {
        "objective": objective,
        "peak_vote_fraction": peak_fraction,
        "pairwise_agreement": pairwise_agreement,
        "minimum_vote_core_fraction": core_fraction,
        "projected_volume_fraction": projected_volume_fraction,
        "mean_normalized_appearance_score": appearance,
        "mean_log_relative_appearance_score": log_appearance,
    }


def select_joint_mask_candidate_hits(
    candidates_by_camera: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    minimum_consensus_votes: int,
    config: JointMultiviewMaskSelectionConfig | None = None,
) -> dict[str, Any]:
    """Choose one candidate per camera from calibration-derived hit vectors."""

    cfg = config or JointMultiviewMaskSelectionConfig()
    cameras = tuple(sorted(candidates_by_camera))
    _require(len(cameras) >= 3, "at least three candidate cameras are required")
    prepared: dict[str, list[dict[str, Any]]] = {}
    hit_count: int | None = None
    for camera in cameras:
        records = list(candidates_by_camera[camera])[
            : cfg.maximum_candidates_per_camera
        ]
        _require(records, f"camera {camera} has no mask candidates")
        scores = np.asarray(
            [float(record["prior_score"]) for record in records], dtype=np.float64
        )
        _require(np.isfinite(scores).all(), f"camera {camera} has invalid prior scores")
        high = float(scores.max())
        normalized = scores / high if high > 0.0 else np.ones_like(scores)
        prepared[camera] = []
        for rank, (record, prior) in enumerate(zip(records, normalized, strict=True)):
            hits = np.asarray(record["hits"], dtype=bool)
            _require(hits.ndim == 1, f"camera {camera} candidate hits must be 1D")
            if hit_count is None:
                hit_count = len(hits)
            _require(len(hits) == hit_count, "candidate hit-vector lengths differ")
            prepared[camera].append(
                {
                    **dict(record),
                    "hits": hits,
                    "normalized_prior_score": float(prior),
                    "rank": rank,
                }
            )

    maximum_rank = max(len(records) for records in prepared.values())
    runs = []

    def selection_key(
        objective: float, components: Mapping[str, float], rank_sum: int
    ) -> tuple[float, ...]:
        peak_votes = components["peak_vote_fraction"] * len(cameras)
        appearance = components["mean_normalized_appearance_score"]
        log_appearance = components["mean_log_relative_appearance_score"]
        if peak_votes + 1e-9 >= minimum_consensus_votes:
            return (1.0, log_appearance, appearance, objective, -float(rank_sum))
        return (
            0.0,
            components["peak_vote_fraction"],
            components["pairwise_agreement"],
            log_appearance,
            objective,
            -float(rank_sum),
        )

    for initial_rank in range(maximum_rank):
        selected = {
            camera: min(initial_rank, len(prepared[camera]) - 1) for camera in cameras
        }
        passes = 0
        for _ in range(cfg.coordinate_descent_passes):
            changed = False
            passes += 1
            for camera in cameras:
                best_index = selected[camera]
                best_key: tuple[float, ...] | None = None
                for candidate_index, _candidate in enumerate(prepared[camera]):
                    proposal = dict(selected)
                    proposal[camera] = candidate_index
                    selected_records = [
                        prepared[name][proposal[name]] for name in cameras
                    ]
                    objective, components = _joint_candidate_objective(
                        np.asarray([record["hits"] for record in selected_records]),
                        np.asarray(
                            [
                                record["normalized_prior_score"]
                                for record in selected_records
                            ]
                        ),
                        minimum_consensus_votes=minimum_consensus_votes,
                        appearance_weight=cfg.appearance_weight,
                        projected_volume_penalty=cfg.projected_volume_penalty,
                    )
                    key = selection_key(
                        objective,
                        components,
                        sum(proposal.values()),
                    )
                    if best_key is None or key > best_key:
                        best_key = key
                        best_index = candidate_index
                changed |= best_index != selected[camera]
                selected[camera] = best_index
            if not changed:
                break
        selected_records = [prepared[name][selected[name]] for name in cameras]
        objective, components = _joint_candidate_objective(
            np.asarray([record["hits"] for record in selected_records]),
            np.asarray(
                [record["normalized_prior_score"] for record in selected_records]
            ),
            minimum_consensus_votes=minimum_consensus_votes,
            appearance_weight=cfg.appearance_weight,
            projected_volume_penalty=cfg.projected_volume_penalty,
        )
        runs.append(
            {
                "initial_rank": initial_rank,
                "passes": passes,
                "selected": selected,
                "objective": objective,
                "components": components,
                "selection_key": selection_key(
                    objective, components, sum(selected.values())
                ),
            }
        )

    winner = max(
        runs,
        key=lambda run: (*run["selection_key"], -run["initial_rank"]),
    )
    return {
        "selected_candidate_by_camera": dict(winner["selected"]),
        "objective": winner["objective"],
        "objective_components": winner["components"],
        "coordinate_descent_passes": winner["passes"],
        "restart_count": len(runs),
    }


def select_joint_multiview_masks(
    candidates_by_camera: Mapping[str, Sequence[Mapping[str, Any]]],
    intrinsics_by_camera: Mapping[str, np.ndarray],
    camera_to_world_by_camera: Mapping[str, np.ndarray],
    reliability_config: CrossViewMaskReliabilityConfig | None = None,
    selection_config: JointMultiviewMaskSelectionConfig | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Select appearance candidates jointly in calibrated 3D and audit them."""

    reliability = reliability_config or CrossViewMaskReliabilityConfig()
    selection = selection_config or JointMultiviewMaskSelectionConfig()
    cameras = tuple(sorted(candidates_by_camera))
    _require(
        all(camera in intrinsics_by_camera for camera in cameras),
        "intrinsics are missing for one or more candidate cameras",
    )
    _require(
        all(camera in camera_to_world_by_camera for camera in cameras),
        "extrinsics are missing for one or more candidate cameras",
    )
    grid = _world_grid_for_cameras(
        cameras,
        camera_to_world_by_camera,
        cube_half_extent_m=reliability.cube_half_extent_m,
        voxel_resolution=selection.voxel_resolution,
    )
    hit_candidates: dict[str, list[dict[str, Any]]] = {}
    for camera in cameras:
        hit_candidates[camera] = []
        for rank, candidate in enumerate(candidates_by_camera[camera]):
            if rank >= selection.maximum_candidates_per_camera:
                break
            mask = np.asarray(candidate["mask"], dtype=bool)
            hit_candidates[camera].append(
                {
                    "hits": _mask_hits_on_world_grid(
                        mask,
                        intrinsics_by_camera[camera],
                        camera_to_world_by_camera[camera],
                        grid,
                        camera=camera,
                    ),
                    "prior_score": float(candidate["prior_score"]),
                    "candidate_index": int(candidate.get("candidate_index", rank)),
                }
            )
    top_masks = {
        camera: np.asarray(candidates_by_camera[camera][0]["mask"], dtype=bool)
        for camera in cameras
    }
    try:
        top_consistency = multiview_mask_consistency(
            top_masks,
            intrinsics_by_camera,
            camera_to_world_by_camera,
            reliability,
        )
    except ValueError as error:
        top_consistency = None
        top_failure = str(error)
    else:
        top_failure = None
    if top_consistency is not None:
        selected_records = [
            {
                "camera": camera,
                "candidate_rank": 0,
                "candidate_index": int(
                    candidates_by_camera[camera][0].get("candidate_index", 0)
                ),
                "prior_score": float(candidates_by_camera[camera][0]["prior_score"]),
            }
            for camera in cameras
        ]
        return top_masks, {
            "policy": "appearance-first-calibrated-3d-feasibility-v2",
            "selection": {
                "selected_candidate_by_camera": {camera: 0 for camera in cameras},
                "search_required": False,
                "top_appearance_failure": None,
            },
            "selected_candidates": selected_records,
            "cross_view_consistency": top_consistency,
        }

    selection_result = select_joint_mask_candidate_hits(
        hit_candidates,
        minimum_consensus_votes=reliability.minimum_consensus_votes,
        config=selection,
    )
    selected_masks = {
        camera: np.asarray(
            candidates_by_camera[camera][
                selection_result["selected_candidate_by_camera"][camera]
            ]["mask"],
            dtype=bool,
        )
        for camera in cameras
    }
    consistency = multiview_mask_consistency(
        selected_masks,
        intrinsics_by_camera,
        camera_to_world_by_camera,
        reliability,
    )
    selected_records = []
    for camera in cameras:
        rank = selection_result["selected_candidate_by_camera"][camera]
        candidate = candidates_by_camera[camera][rank]
        selected_records.append(
            {
                "camera": camera,
                "candidate_rank": rank,
                "candidate_index": int(candidate.get("candidate_index", rank)),
                "prior_score": float(candidate["prior_score"]),
            }
        )
    return selected_masks, {
        "policy": "appearance-first-calibrated-3d-feasibility-v2",
        "selection": {
            **selection_result,
            "search_required": True,
            "top_appearance_failure": top_failure,
        },
        "selected_candidates": selected_records,
        "cross_view_consistency": consistency,
    }


def summarize_multiview_mask_hits(
    hits: np.ndarray,
    camera_names: Sequence[str],
    grid_world_m: np.ndarray,
    config: CrossViewMaskReliabilityConfig,
) -> dict[str, Any]:
    """Select cameras whose masks contain a common leave-one-view 3D core."""

    camera_hits = np.asarray(hits, dtype=bool)
    grid = np.asarray(grid_world_m, dtype=np.float64)
    names = tuple(camera_names)
    _require(camera_hits.ndim == 2, "camera hits must be a 2D array")
    _require(len(names) == camera_hits.shape[0], "camera-name count mismatch")
    _require(len(set(names)) == len(names), "camera names must be unique")
    _require(len(names) >= 3, "at least three candidate views are required")
    _require(
        grid.shape == (camera_hits.shape[1], 3),
        "world grid shape does not match camera hits",
    )

    vote_counts = camera_hits.sum(axis=0)
    peak_votes = int(vote_counts.max(initial=0))
    consensus_votes = max(
        config.minimum_consensus_votes,
        int(math.ceil(config.consensus_fraction_of_peak * peak_votes)),
    )
    _require(
        peak_votes >= consensus_votes,
        "candidate masks do not form the required multiview consensus",
    )
    core = vote_counts >= consensus_votes
    _require(np.any(core), "multiview consensus core is empty")

    per_camera = []
    accepted = []
    rejected = []
    leave_one_out_votes = max(2, consensus_votes - 1)
    for index, camera in enumerate(names):
        leave_one_out_core = (vote_counts - camera_hits[index]) >= leave_one_out_votes
        core_voxels = int(np.count_nonzero(leave_one_out_core))
        recall = (
            float(np.mean(camera_hits[index, leave_one_out_core]))
            if core_voxels
            else 0.0
        )
        keep = bool(recall >= config.minimum_leave_one_out_recall)
        (accepted if keep else rejected).append(camera)
        per_camera.append(
            {
                "camera": camera,
                "accepted": keep,
                "leave_one_out_core_recall": recall,
                "leave_one_out_core_voxels": core_voxels,
            }
        )

    core_points = grid[core]
    quantiles = np.percentile(core_points, [1.0, 50.0, 99.0], axis=0)
    return {
        "candidate_camera_count": len(names),
        "accepted_camera_count": len(accepted),
        "rejected_camera_count": len(rejected),
        "accepted_cameras": accepted,
        "rejected_cameras": rejected,
        "peak_vote_count": peak_votes,
        "consensus_vote_count": consensus_votes,
        "consensus_core_voxel_count": int(np.count_nonzero(core)),
        "consensus_core_world_m": {
            "q01": quantiles[0].tolist(),
            "median": quantiles[1].tolist(),
            "q99": quantiles[2].tolist(),
            "q01_to_q99_span": (quantiles[2] - quantiles[0]).tolist(),
        },
        "per_camera": per_camera,
    }


def multiview_mask_consistency(
    masks_by_camera: Mapping[str, np.ndarray],
    intrinsics_by_camera: Mapping[str, np.ndarray],
    camera_to_world_by_camera: Mapping[str, np.ndarray],
    config: CrossViewMaskReliabilityConfig | None = None,
) -> dict[str, Any]:
    """Project a common world grid into candidate masks and audit agreement."""

    cfg = config or CrossViewMaskReliabilityConfig()
    cameras = tuple(sorted(masks_by_camera))
    _require(len(cameras) >= 3, "at least three SAM2 masks are required")
    _require(
        all(camera in intrinsics_by_camera for camera in cameras),
        "intrinsics are missing for one or more candidate cameras",
    )
    _require(
        all(camera in camera_to_world_by_camera for camera in cameras),
        "extrinsics are missing for one or more candidate cameras",
    )

    grid = _world_grid_for_cameras(
        cameras,
        camera_to_world_by_camera,
        cube_half_extent_m=cfg.cube_half_extent_m,
        voxel_resolution=cfg.voxel_resolution,
    )

    hits = []
    for camera in cameras:
        hits.append(
            _mask_hits_on_world_grid(
                masks_by_camera[camera],
                intrinsics_by_camera[camera],
                camera_to_world_by_camera[camera],
                grid,
                camera=camera,
            )
        )

    return summarize_multiview_mask_hits(
        np.asarray(hits),
        cameras,
        grid,
        cfg,
    )


def camera_reliability_from_multiview_consistency(
    consistency: Mapping[str, Any],
    *,
    minimum_reliability: float = 0.05,
    recall_power: float = 2.0,
) -> dict[str, float]:
    """Turn source-only leave-one-view recall into conservative soft weights.

    The score is independent of any fitted filament or simulator residual.  A
    nonzero floor keeps a view available as weak contradictory evidence rather
    than silently deleting it at a hard threshold.
    """

    _require(
        0.0 < minimum_reliability <= 1.0,
        "minimum camera reliability must lie in (0,1]",
    )
    _require(recall_power > 0.0, "camera reliability power must be positive")
    records = consistency.get("per_camera")
    _require(isinstance(records, list) and records, "consistency has no cameras")
    output: dict[str, float] = {}
    for record in records:
        _require(isinstance(record, Mapping), "invalid camera consistency record")
        camera = record.get("camera")
        recall = record.get("leave_one_out_core_recall")
        _require(isinstance(camera, str) and camera, "camera name is missing")
        _require(camera not in output, "camera consistency names must be unique")
        _require(
            isinstance(recall, (float, int)) and np.isfinite(recall),
            f"camera recall is invalid for {camera}",
        )
        _require(0.0 <= recall <= 1.0, f"camera recall is invalid for {camera}")
        output[camera] = float(
            minimum_reliability
            + (1.0 - minimum_reliability) * float(recall) ** recall_power
        )
    return output


def sam2_view_audit_sha256(payload: Mapping[str, Any]) -> str:
    canonical = dict(payload)
    canonical.pop("result_sha256", None)
    return hashlib.sha256(_canonical_bytes(canonical)).hexdigest()


def build_sam2_view_audit(
    *,
    protocol_id: str,
    episode_access: Mapping[str, Any],
    automatic_view_diagnostics: Sequence[Mapping[str, Any]],
    consistency: Mapping[str, Any],
    reliability_config: CrossViewMaskReliabilityConfig,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": DEFORM360_SAM2_VIEW_AUDIT_SCHEMA_VERSION,
        "artifact_kind": "Deform360RopeSam2ViewAudit",
        "protocol_id": protocol_id,
        "episode_access": dict(episode_access),
        "upstream": {
            "repository": PINNED_SAM2_REPOSITORY,
            "commit": PINNED_SAM2_COMMIT,
            "checkpoint_url": PINNED_SAM2_CHECKPOINT_URL,
            "checkpoint_sha256": PINNED_SAM2_CHECKPOINT_SHA256,
            "model_config": PINNED_SAM2_MODEL_CONFIG,
        },
        "parameters": asdict(reliability_config),
        "automatic_view_diagnostics": [
            dict(diagnostic) for diagnostic in automatic_view_diagnostics
        ],
        "cross_view_consistency": dict(consistency),
        "claim_boundary": (
            "Camera reliability was selected from first-frame SAM2 masks and "
            "calibration consistency only; no target outcome metric was used."
        ),
    }
    payload["result_sha256"] = sam2_view_audit_sha256(payload)
    return payload


def validate_sam2_view_audit(payload: Mapping[str, Any]) -> dict[str, Any]:
    _require(
        payload.get("schema_version") == DEFORM360_SAM2_VIEW_AUDIT_SCHEMA_VERSION,
        "unsupported SAM2 view-audit schema",
    )
    _require(
        payload.get("artifact_kind") == "Deform360RopeSam2ViewAudit",
        "unexpected SAM2 view-audit artifact kind",
    )
    _require(
        payload.get("result_sha256") == sam2_view_audit_sha256(payload),
        "SAM2 view-audit checksum mismatch",
    )
    accepted = payload.get("cross_view_consistency", {}).get("accepted_cameras", [])
    _require(isinstance(accepted, list) and accepted, "view audit accepted no cameras")
    return {
        "passed": True,
        "result_sha256": payload["result_sha256"],
        "accepted_camera_count": len(accepted),
    }


def load_sam2_view_audit(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    _require(isinstance(payload, dict), "SAM2 view audit must contain an object")
    validate_sam2_view_audit(payload)
    return payload


def write_sam2_view_audit(path: str | Path, payload: Mapping[str, Any]) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return output


__all__ = [
    "CrossViewMaskReliabilityConfig",
    "JointMultiviewMaskSelectionConfig",
    "build_sam2_view_audit",
    "camera_reliability_from_multiview_consistency",
    "load_sam2_view_audit",
    "multiview_mask_consistency",
    "sam2_view_audit_sha256",
    "summarize_multiview_mask_hits",
    "select_joint_mask_candidate_hits",
    "select_joint_multiview_masks",
    "validate_sam2_view_audit",
    "write_sam2_view_audit",
]
