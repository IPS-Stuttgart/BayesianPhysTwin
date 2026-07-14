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

    centers = []
    for camera in cameras:
        transform = np.asarray(camera_to_world_by_camera[camera], dtype=np.float64)
        _require(transform.shape == (4, 4), f"invalid extrinsics for {camera}")
        _require(np.isfinite(transform).all(), f"non-finite extrinsics for {camera}")
        centers.append(transform[:3, 3])
    center = np.mean(centers, axis=0)
    axis = np.linspace(
        -cfg.cube_half_extent_m,
        cfg.cube_half_extent_m,
        cfg.voxel_resolution,
    )
    grid = (
        np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1).reshape(-1, 3)
        + center
    )

    hits = []
    for camera in cameras:
        mask = np.asarray(masks_by_camera[camera], dtype=bool)
        _require(mask.ndim == 2, f"mask for {camera} must be 2D")
        height, width = mask.shape
        intrinsics = np.asarray(intrinsics_by_camera[camera], dtype=np.float64)
        _require(intrinsics.shape == (3, 3), f"invalid intrinsics for {camera}")
        world_to_camera = np.linalg.inv(
            np.asarray(camera_to_world_by_camera[camera], dtype=np.float64)
        )
        points = grid @ world_to_camera[:3, :3].T + world_to_camera[:3, 3]
        depth = points[:, 2]
        in_front = depth > 1e-6
        safe_depth = np.where(in_front, depth, 1.0)
        u = points[:, 0] / safe_depth * intrinsics[0, 0] + intrinsics[0, 2]
        v = points[:, 1] / safe_depth * intrinsics[1, 1] + intrinsics[1, 2]
        in_bounds = in_front & (u >= 0.0) & (u < width) & (v >= 0.0) & (v < height)
        columns = np.clip(u, 0, width - 1).astype(np.int64)
        rows = np.clip(v, 0, height - 1).astype(np.int64)
        hits.append(in_bounds & mask[rows, columns])

    return summarize_multiview_mask_hits(
        np.asarray(hits),
        cameras,
        grid,
        cfg,
    )


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
    "build_sam2_view_audit",
    "load_sam2_view_audit",
    "multiview_mask_consistency",
    "sam2_view_audit_sha256",
    "summarize_multiview_mask_hits",
    "validate_sam2_view_audit",
    "write_sam2_view_audit",
]
