"""Leakage-safe target-prefix rope geometry for Deform360 ``001-rope``."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .deform360 import Deform360ProtocolConfig
from .deform360_rope_graph import (
    RopeCenterlineConfig,
    extract_rope_centerline,
    initialize_rope_centerline_pca,
)
from .deform360_sam2_prefix import (
    decode_video_frame_window,
    validate_sam2_prefix_mask_artifact,
)
from .deform360_rope_sequence import validate_rope_sequence_artifact
from .deform360_visual_hull import AdaptiveRopeHullConfig, adaptive_rope_visual_hull


DEFORM360_ROPE_PREFIX_GEOMETRY_SCHEMA_VERSION = 2


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    descriptor = _canonical_bytes(
        {"dtype": array.dtype.str, "shape": list(array.shape)}
    )
    return hashlib.sha256(
        descriptor + b"\0" + array.view(np.uint8).tobytes()
    ).hexdigest()


@dataclass(frozen=True)
class RopePrefixGeometryConfig:
    initial_cube_half_extent_m: float = 0.5
    initial_voxel_resolution: int = 100
    initial_minimum_hull_points: int = 256
    maximum_temporal_node_displacement_m: float = 0.05

    def __post_init__(self) -> None:
        _require(self.initial_cube_half_extent_m > 0.0, "initial cube must be positive")
        _require(self.initial_voxel_resolution >= 16, "initial grid is too coarse")
        _require(self.initial_minimum_hull_points >= 16, "initial hull is too small")
        _require(
            self.maximum_temporal_node_displacement_m > 0.0,
            "temporal displacement gate must be positive",
        )


def rope_prefix_geometry_artifact_sha256(payload: Mapping[str, Any]) -> str:
    canonical = dict(payload)
    canonical.pop("result_sha256", None)
    return hashlib.sha256(_canonical_bytes(canonical)).hexdigest()


def _source_length_prior(
    protocol: Deform360ProtocolConfig,
    source_sequences: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    _require(
        len(source_sequences) >= 3, "at least three source geometries are required"
    )
    records = []
    for sequence in source_sequences:
        validate_rope_sequence_artifact(sequence, verify_archive=True)
        episode_index = int(sequence["episode_index"])
        _require(
            episode_index in protocol.source_episode_ids,
            "rope-length prior contains a non-source episode",
        )
        with np.load(sequence["archive"]["path"], allow_pickle=False) as stored:
            centerlines = np.asarray(stored["centerlines_m"], dtype=np.float64)
        lengths = np.linalg.norm(np.diff(centerlines, axis=1), axis=2).sum(axis=1)
        records.append(
            {
                "episode_index": episode_index,
                "result_sha256": sequence["result_sha256"],
                "median_centerline_length_m": float(np.median(lengths)),
            }
        )
    _require(
        len({record["episode_index"] for record in records}) == len(records),
        "rope-length source episodes repeat",
    )
    values = np.asarray(
        [record["median_centerline_length_m"] for record in records],
        dtype=np.float64,
    )
    return {
        "selection_scope": "quality-passing source episodes only",
        "source_records": sorted(records, key=lambda record: record["episode_index"]),
        "median_centerline_length_m": float(np.median(values)),
        "minimum_centerline_length_m": float(np.min(values)),
        "maximum_centerline_length_m": float(np.max(values)),
    }


def _centerline_with_length(centerline: np.ndarray, length_m: float) -> np.ndarray:
    values = np.asarray(centerline, dtype=np.float64)
    direction = values[-1] - values[0]
    norm = float(np.linalg.norm(direction))
    _require(norm > 1e-8, "initial target centerline has no direction")
    unit = direction / norm
    midpoint = np.mean(values, axis=0)
    coordinates = np.linspace(-0.5 * length_m, 0.5 * length_m, len(values))
    return midpoint + coordinates[:, None] * unit


def build_target_prefix_rope_geometry(
    processed_root: str | Path,
    protocol: Deform360ProtocolConfig,
    prefix_mask_audit: Mapping[str, Any],
    source_sequences: Sequence[Mapping[str, Any]],
    output_archive_path: str | Path,
    *,
    config: RopePrefixGeometryConfig = RopePrefixGeometryConfig(),
    centerline_config: RopeCenterlineConfig = RopeCenterlineConfig(),
    hull_config: AdaptiveRopeHullConfig = AdaptiveRopeHullConfig(),
) -> dict[str, Any]:
    """Reconstruct only the contact-sealed six-frame target prefix."""

    validate_sam2_prefix_mask_artifact(prefix_mask_audit)
    _require(
        prefix_mask_audit.get("protocol_id") == protocol.protocol_id,
        "prefix masks belong to another protocol",
    )
    target_index = protocol.target_episode_ids[0]
    expected_episode_id = f"{protocol.object_id}/episode_{target_index:04d}"
    _require(
        prefix_mask_audit.get("target_episode_id") == expected_episode_id,
        "prefix masks belong to another target episode",
    )
    boundary = prefix_mask_audit.get("information_boundary", {})
    _require(
        boundary.get("target_future_visual_frames_read") is False,
        "prefix mask artifact read target future frames",
    )
    interval = prefix_mask_audit["target_prefix"]
    start = int(interval["start_frame"])
    stop = int(interval["stop_frame_exclusive"])
    _require(
        stop - start == protocol.prefix_frame_count,
        "prefix geometry interval differs from the protocol",
    )
    source_length_prior = _source_length_prior(protocol, source_sequences)
    episode_dir = Path(processed_root).resolve() / f"episode_{target_index:04d}"
    _require(episode_dir.is_dir(), f"target episode is missing: {episode_dir}")
    try:
        from deform360.processing.episode import load_episode_calibration
        from deform360.processing.reconstruct_stage import visual_hull_points
    except ImportError as error:  # pragma: no cover - GPU-host integration
        raise RuntimeError(
            "the pinned Deform360 processing environment is required"
        ) from error
    intrinsics, extrinsics = load_episode_calibration(episode_dir)
    outputs = {str(row["camera"]): row for row in prefix_mask_audit.get("outputs", [])}
    cameras = list(prefix_mask_audit["camera_policy"]["selected_cameras"])
    _require(set(cameras) == set(outputs), "prefix camera outputs are incomplete")
    masks_by_camera = {}
    frames_by_camera = {}
    input_checks = []
    for camera in cameras:
        record = outputs[camera]
        mask_path = Path(record["mask_path"])
        _require(mask_path.is_file(), f"prefix mask is missing for {camera}")
        _require(
            _sha256_file(mask_path) == record["mask_sha256"],
            f"prefix mask checksum mismatch for {camera}",
        )
        masks = np.load(mask_path, allow_pickle=False)
        video_path = episode_dir / camera / "undistorted.mp4"
        _require(
            str(video_path) == record["source_video_path"],
            f"prefix video path mismatch for {camera}",
        )
        frames = decode_video_frame_window(video_path, start, stop)
        frame_hashes = [_sha256_array(frame) for frame in frames]
        _require(
            frame_hashes == record["decoded_frame_sha256"],
            f"decoded prefix frame checksum mismatch for {camera}",
        )
        _require(
            masks.shape[0] == len(frames) and masks.shape[1:] == frames.shape[1:3],
            f"prefix mask/video shape mismatch for {camera}",
        )
        masks_by_camera[camera] = np.asarray(masks, dtype=np.uint8)
        frames_by_camera[camera] = frames
        input_checks.append(
            {
                "camera": camera,
                "mask_sha256": record["mask_sha256"],
                "decoded_frame_sha256": frame_hashes,
            }
        )

    centerlines = []
    diagnostics = []
    previous = None
    for local_index, frame_index in enumerate(range(start, stop)):
        frame_masks = {
            camera: masks_by_camera[camera][local_index] for camera in cameras
        }
        frame_images = {
            camera: frames_by_camera[camera][local_index] for camera in cameras
        }
        initial_diagnostics = None
        if previous is None:
            coarse_points, _ = visual_hull_points(
                frame_masks,
                frame_images,
                {camera: intrinsics[camera] for camera in cameras},
                {camera: extrinsics[camera] for camera in cameras},
                cube_half_extent_m=config.initial_cube_half_extent_m,
                voxel_resolution=config.initial_voxel_resolution,
                min_points=config.initial_minimum_hull_points,
            )
            previous, initial_diagnostics = initialize_rope_centerline_pca(
                coarse_points, config=centerline_config
            )
            raw_length = float(np.linalg.norm(np.diff(previous, axis=0), axis=1).sum())
            previous = _centerline_with_length(
                previous, source_length_prior["median_centerline_length_m"]
            )
            initial_diagnostics = {
                **initial_diagnostics,
                "raw_target_prefix_centerline_length_m": raw_length,
                "source_locked_centerline_length_m": source_length_prior[
                    "median_centerline_length_m"
                ],
            }
        hull, hull_diagnostics = adaptive_rope_visual_hull(
            previous,
            frame_masks,
            {camera: intrinsics[camera] for camera in cameras},
            {camera: extrinsics[camera] for camera in cameras},
            config=hull_config,
        )
        current, centerline_diagnostics = extract_rope_centerline(
            hull,
            config=centerline_config,
            initial_centerline_m=previous,
            reference_centerline_m=previous,
        )
        centerlines.append(current)
        diagnostics.append(
            {
                "frame_index": frame_index,
                "initial_coarse_centerline": initial_diagnostics,
                "adaptive_hull": hull_diagnostics,
                "centerline": centerline_diagnostics,
            }
        )
        previous = current
    trajectories = np.asarray(centerlines, dtype=np.float64)
    temporal = np.linalg.norm(np.diff(trajectories, axis=0), axis=2)
    maximum_temporal = float(np.max(temporal))
    quality = {
        "passed": maximum_temporal <= config.maximum_temporal_node_displacement_m,
        "maximum_temporal_node_displacement_m": {
            "value": maximum_temporal,
            "maximum": config.maximum_temporal_node_displacement_m,
        },
        "centerline_length_m": {
            "first": float(
                np.linalg.norm(np.diff(trajectories[0], axis=0), axis=1).sum()
            ),
            "last": float(
                np.linalg.norm(np.diff(trajectories[-1], axis=0), axis=1).sum()
            ),
        },
    }
    output = Path(output_archive_path).resolve()
    _require(output.suffix == ".npz", "prefix geometry archive must end in .npz")
    output.parent.mkdir(parents=True, exist_ok=True)
    frame_indices = np.arange(start, stop, dtype=np.int32)
    np.savez_compressed(output, frame_indices=frame_indices, centerlines_m=trajectories)
    payload: dict[str, Any] = {
        "schema_version": DEFORM360_ROPE_PREFIX_GEOMETRY_SCHEMA_VERSION,
        "artifact_kind": "Deform360TargetPrefixRopeGeometry",
        "protocol_id": protocol.protocol_id,
        "target_episode_id": expected_episode_id,
        "prefix_mask_audit_result_sha256": prefix_mask_audit["result_sha256"],
        "source_length_prior": source_length_prior,
        "parameters": {
            "prefix_geometry": asdict(config),
            "centerline": asdict(centerline_config),
            "adaptive_hull": asdict(hull_config),
        },
        "camera_policy": prefix_mask_audit["camera_policy"],
        "frame_indices": frame_indices.astype(int).tolist(),
        "input_checks": input_checks,
        "frame_diagnostics": diagnostics,
        "quality": quality,
        "archive": {
            "path": str(output),
            "sha256": _sha256_file(output),
            "bytes": output.stat().st_size,
            "centerlines_sha256": _sha256_array(trajectories),
            "shape": list(trajectories.shape),
        },
        "information_boundary": {
            "target_visual_prefix_read": True,
            "target_future_visual_frames_read": False,
            "target_tactile_oracle_read": False,
            "target_prediction_metrics_computed": False,
            "full_target_video_hashes_computed": False,
        },
        "measurement_semantics": (
            "Ordered normalized-arc-length silhouette centerlines; nodes are "
            "pseudo-correspondences, not independently verified material tracks."
        ),
    }
    payload["result_sha256"] = rope_prefix_geometry_artifact_sha256(payload)
    return payload


def validate_target_prefix_rope_geometry(
    payload: Mapping[str, Any], *, verify_archive: bool = True
) -> dict[str, Any]:
    _require(
        payload.get("schema_version") == DEFORM360_ROPE_PREFIX_GEOMETRY_SCHEMA_VERSION,
        "unsupported target-prefix geometry schema",
    )
    _require(
        payload.get("artifact_kind") == "Deform360TargetPrefixRopeGeometry",
        "unexpected target-prefix geometry artifact kind",
    )
    _require(
        payload.get("result_sha256") == rope_prefix_geometry_artifact_sha256(payload),
        "target-prefix geometry checksum mismatch",
    )
    boundary = payload.get("information_boundary", {})
    _require(
        boundary.get("target_future_visual_frames_read") is False,
        "target-prefix geometry read future visual frames",
    )
    _require(
        boundary.get("target_tactile_oracle_read") is False,
        "target-prefix geometry read target tactile oracle",
    )
    _require(
        payload.get("quality", {}).get("passed") is True, "prefix geometry failed QA"
    )
    if verify_archive:
        archive = Path(payload["archive"]["path"])
        _require(archive.is_file(), "target-prefix geometry archive is missing")
        _require(
            _sha256_file(archive) == payload["archive"]["sha256"],
            "target-prefix geometry archive checksum mismatch",
        )
        with np.load(archive, allow_pickle=False) as stored:
            _require(
                set(stored.files) == {"frame_indices", "centerlines_m"},
                "target-prefix geometry fields differ",
            )
            _require(
                _sha256_array(stored["centerlines_m"])
                == payload["archive"]["centerlines_sha256"],
                "target-prefix centerline checksum mismatch",
            )
    return {
        "passed": True,
        "result_sha256": payload["result_sha256"],
        "prefix_end_frame": int(payload["frame_indices"][-1]),
    }


def write_target_prefix_rope_geometry(
    path: str | Path, payload: Mapping[str, Any]
) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return output


__all__ = [
    "DEFORM360_ROPE_PREFIX_GEOMETRY_SCHEMA_VERSION",
    "RopePrefixGeometryConfig",
    "build_target_prefix_rope_geometry",
    "rope_prefix_geometry_artifact_sha256",
    "validate_target_prefix_rope_geometry",
    "write_target_prefix_rope_geometry",
]
