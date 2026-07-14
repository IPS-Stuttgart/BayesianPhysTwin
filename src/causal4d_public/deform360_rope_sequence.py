"""Source-only ordered rope trajectories from adaptive multiview hulls."""

from __future__ import annotations

import hashlib
import json
from contextlib import ExitStack
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .deform360 import Deform360ProtocolConfig
from .deform360_rope_graph import (
    RopeCenterlineConfig,
    extract_rope_centerline,
    initialize_rope_centerline_pca,
)
from .deform360_sam2 import validate_sam2_mask_artifact
from .deform360_sam2_prefix import select_source_locked_prefix_cameras
from .deform360_visual_hull import AdaptiveRopeHullConfig, adaptive_rope_visual_hull


DEFORM360_ROPE_SEQUENCE_SCHEMA_VERSION = 5


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
class RopeCenterlineSequenceConfig:
    source_episode_index: int = 0
    frame_start: int = 0
    frame_stop_exclusive: int | None = None
    frame_stride: int = 2
    minimum_synchronization_reliability: float = 0.85
    minimum_camera_count: int = 8
    initial_cube_half_extent_m: float = 0.5
    initial_voxel_resolution: int = 100
    initial_minimum_hull_points: int = 256
    maximum_length_coefficient_of_variation: float = 0.10
    maximum_temporal_node_displacement_m: float = 0.05

    def __post_init__(self) -> None:
        _require(self.source_episode_index >= 0, "episode index must be nonnegative")
        _require(self.frame_start >= 0, "frame start must be nonnegative")
        _require(
            self.frame_stop_exclusive is None
            or self.frame_stop_exclusive > self.frame_start,
            "frame stop must exceed frame start",
        )
        _require(self.frame_stride >= 1, "frame stride must be positive")
        _require(
            0.0 <= self.minimum_synchronization_reliability <= 1.0,
            "invalid synchronization-reliability threshold",
        )
        _require(self.minimum_camera_count >= 2, "at least two cameras are required")
        _require(self.initial_cube_half_extent_m > 0.0, "initial cube must be positive")
        _require(self.initial_voxel_resolution >= 16, "initial grid is too coarse")
        _require(self.initial_minimum_hull_points >= 16, "initial hull is too small")
        _require(
            self.maximum_length_coefficient_of_variation > 0.0,
            "length-variation gate must be positive",
        )
        _require(
            self.maximum_temporal_node_displacement_m > 0.0,
            "temporal-displacement gate must be positive",
        )


def _sequence_quality(
    centerlines_m: np.ndarray,
    config: RopeCenterlineSequenceConfig,
) -> dict[str, Any]:
    trajectories = np.asarray(centerlines_m, dtype=np.float64)
    lengths = np.linalg.norm(np.diff(trajectories, axis=1), axis=2).sum(axis=1)
    temporal = np.linalg.norm(np.diff(trajectories, axis=0), axis=2)
    length_cv = float(np.std(lengths) / np.mean(lengths))
    maximum_temporal = float(np.max(temporal)) if temporal.size else 0.0
    gates = {
        "length_coefficient_of_variation": {
            "value": length_cv,
            "maximum": config.maximum_length_coefficient_of_variation,
            "passed": length_cv <= config.maximum_length_coefficient_of_variation,
        },
        "temporal_node_displacement_m": {
            "value": maximum_temporal,
            "maximum": config.maximum_temporal_node_displacement_m,
            "passed": maximum_temporal <= config.maximum_temporal_node_displacement_m,
        },
    }
    return {
        "passed": all(gate["passed"] for gate in gates.values()),
        "gates": gates,
        "centerline_length_m": {
            "minimum": float(np.min(lengths)),
            "median": float(np.median(lengths)),
            "maximum": float(np.max(lengths)),
        },
        "temporal_node_displacement_m": {
            "median": float(np.median(temporal)) if temporal.size else 0.0,
            "p95": float(np.quantile(temporal, 0.95)) if temporal.size else 0.0,
            "p99": float(np.quantile(temporal, 0.99)) if temporal.size else 0.0,
            "maximum": maximum_temporal,
        },
    }


def rope_sequence_artifact_sha256(payload: Mapping[str, Any]) -> str:
    canonical = dict(payload)
    canonical.pop("result_sha256", None)
    return hashlib.sha256(_canonical_bytes(canonical)).hexdigest()


def select_source_propagation_stable_cameras(
    source_view_audit: Mapping[str, Any],
    source_mask_audit: Mapping[str, Any],
    preflight: Mapping[str, Any],
    *,
    minimum_synchronization_reliability: float,
    minimum_camera_count: int,
) -> dict[str, Any]:
    """Keep only source cameras with reliable sync and no empty SAM2 frame."""

    validate_sam2_mask_artifact(source_mask_audit)
    initial = select_source_locked_prefix_cameras(
        source_view_audit,
        preflight,
        minimum_synchronization_reliability=minimum_synchronization_reliability,
        minimum_camera_count=minimum_camera_count,
    )
    _require(
        source_mask_audit.get("episode_access")
        == source_view_audit.get("episode_access"),
        "source mask and view audits describe different episode access",
    )
    _require(
        source_mask_audit.get("view_selection", {}).get("view_audit_result_sha256")
        == source_view_audit.get("result_sha256"),
        "source mask audit is not bound to the supplied view audit",
    )
    diagnostics = {
        record["camera"]: record
        for record in source_mask_audit.get("camera_diagnostics", [])
    }
    retained = []
    rejected = []
    for camera in initial["selected_cameras"]:
        _require(camera in diagnostics, f"source mask diagnostic is missing: {camera}")
        _require(
            camera in source_mask_audit.get("outputs", {}),
            f"source mask output is missing: {camera}",
        )
        empty = int(diagnostics[camera]["propagation"]["empty_frame_count"])
        record = {"camera": camera, "empty_frame_count": empty}
        if empty == 0:
            retained.append(camera)
        else:
            rejected.append(record)
    _require(
        len(retained) >= minimum_camera_count,
        "too few propagation-stable source cameras remain",
    )
    return {
        "selection_rule": (
            "cross-view-consistent first-frame SAM2 mask, synchronization "
            "reliability at or above the frozen threshold, and zero empty "
            "frames during source-only propagation"
        ),
        "minimum_synchronization_reliability": minimum_synchronization_reliability,
        "minimum_camera_count": minimum_camera_count,
        "view_audit_result_sha256": source_view_audit["result_sha256"],
        "mask_audit_result_sha256": source_mask_audit["result_sha256"],
        "preflight_result_sha256": preflight["result_sha256"],
        "initial_selected_camera_count": len(initial["selected_cameras"]),
        "selected_camera_count": len(retained),
        "selected_cameras": retained,
        "propagation_rejected_cameras": rejected,
    }


def run_source_rope_centerline_sequence(
    processed_root: str | Path,
    protocol: Deform360ProtocolConfig,
    source_view_audit: Mapping[str, Any],
    source_mask_audit: Mapping[str, Any],
    preflight: Mapping[str, Any],
    output_archive_path: str | Path,
    *,
    sequence_config: RopeCenterlineSequenceConfig | None = None,
    centerline_config: RopeCenterlineConfig | None = None,
    hull_config: AdaptiveRopeHullConfig | None = None,
) -> dict[str, Any]:
    """Extract source centerlines without reading any target episode files."""

    sequence = sequence_config or RopeCenterlineSequenceConfig()
    centerline_cfg = centerline_config or RopeCenterlineConfig()
    hull_cfg = hull_config or AdaptiveRopeHullConfig()
    _require(
        sequence.source_episode_index in protocol.source_episode_ids,
        "centerline sequence episode is not in the locked source split",
    )
    _require(
        source_view_audit.get("episode_access", {}).get("episode_index")
        == sequence.source_episode_index,
        "source view audit and centerline episode differ",
    )
    policy = select_source_propagation_stable_cameras(
        source_view_audit,
        source_mask_audit,
        preflight,
        minimum_synchronization_reliability=(
            sequence.minimum_synchronization_reliability
        ),
        minimum_camera_count=sequence.minimum_camera_count,
    )
    try:
        import cv2
        from deform360.annotations import H5Array
        from deform360.processing.episode import (
            camera_frame_count,
            load_episode_calibration,
        )
        from deform360.processing.reconstruct_stage import visual_hull_points
    except ImportError as error:  # pragma: no cover - GPU-host integration
        raise RuntimeError(
            "the pinned Deform360 processing environment is required"
        ) from error

    episode_dir = (
        Path(processed_root).resolve() / f"episode_{sequence.source_episode_index:04d}"
    )
    _require(episode_dir.is_dir(), f"source episode is missing: {episode_dir}")
    cameras = list(policy["selected_cameras"])
    intrinsics, extrinsics = load_episode_calibration(episode_dir)
    frame_count = camera_frame_count(episode_dir, cameras[0])
    stop = sequence.frame_stop_exclusive or frame_count
    _require(stop <= frame_count, "centerline frame stop exceeds the episode")
    frame_indices = list(range(sequence.frame_start, stop, sequence.frame_stride))
    _require(frame_indices, "centerline sequence selected no frames")

    centerlines = []
    frame_diagnostics = []
    frame_input_hashes = []
    with ExitStack() as stack:
        masks = {
            camera: stack.enter_context(
                H5Array(episode_dir / camera / "mask_refined.h5")
            )
            for camera in cameras
        }
        captures = {
            camera: cv2.VideoCapture(str(episode_dir / camera / "undistorted.mp4"))
            for camera in cameras
        }
        for camera, capture in captures.items():
            _require(capture.isOpened(), f"cannot open source video for {camera}")
            stack.callback(capture.release)

        previous = None
        for frame_index in frame_indices:
            masks_by_camera = {}
            images_by_camera = {}
            per_camera_hashes = []
            for camera in cameras:
                mask = np.asarray(masks[camera][frame_index], dtype=np.uint8)
                capture = captures[camera]
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                ok, bgr = capture.read()
                _require(ok, f"cannot read source frame {frame_index} for {camera}")
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                _require(
                    mask.shape == rgb.shape[:2], f"mask/video mismatch for {camera}"
                )
                masks_by_camera[camera] = mask
                images_by_camera[camera] = rgb
                per_camera_hashes.append(
                    {
                        "camera": camera,
                        "mask_sha256": _sha256_array(mask),
                        "rgb_sha256": _sha256_array(rgb),
                    }
                )

            initial_diagnostics = None
            if previous is None:
                coarse_points, _ = visual_hull_points(
                    masks_by_camera,
                    images_by_camera,
                    {camera: intrinsics[camera] for camera in cameras},
                    {camera: extrinsics[camera] for camera in cameras},
                    cube_half_extent_m=sequence.initial_cube_half_extent_m,
                    voxel_resolution=sequence.initial_voxel_resolution,
                    min_points=sequence.initial_minimum_hull_points,
                )
                previous, initial_diagnostics = initialize_rope_centerline_pca(
                    coarse_points,
                    config=centerline_cfg,
                )
            hull, hull_diagnostics = adaptive_rope_visual_hull(
                previous,
                masks_by_camera,
                {camera: intrinsics[camera] for camera in cameras},
                {camera: extrinsics[camera] for camera in cameras},
                config=hull_cfg,
            )
            current, graph_diagnostics = extract_rope_centerline(
                hull,
                config=centerline_cfg,
                initial_centerline_m=previous,
                reference_centerline_m=previous,
            )
            centerlines.append(current)
            frame_input_hashes.append(
                {"frame_index": frame_index, "cameras": per_camera_hashes}
            )
            frame_diagnostics.append(
                {
                    "frame_index": frame_index,
                    "initial_coarse_centerline": initial_diagnostics,
                    "adaptive_hull": hull_diagnostics,
                    "centerline": graph_diagnostics,
                }
            )
            previous = current

    trajectories = np.asarray(centerlines, dtype=np.float64)
    quality = _sequence_quality(trajectories, sequence)
    output = Path(output_archive_path).resolve()
    _require(output.suffix == ".npz", "centerline archive must end in .npz")
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        frame_indices=np.asarray(frame_indices, dtype=np.int32),
        centerlines_m=trajectories,
    )
    payload: dict[str, Any] = {
        "schema_version": DEFORM360_ROPE_SEQUENCE_SCHEMA_VERSION,
        "artifact_kind": "Deform360SourceRopeCenterlineSequence",
        "protocol_id": protocol.protocol_id,
        "episode_index": sequence.source_episode_index,
        "split": "source",
        "sequence_parameters": asdict(sequence),
        "centerline_parameters": asdict(centerline_cfg),
        "hull_parameters": asdict(hull_cfg),
        "camera_policy": policy,
        "frame_indices": frame_indices,
        "frame_input_hashes": frame_input_hashes,
        "frame_diagnostics": frame_diagnostics,
        "quality": quality,
        "archive": {
            "path": str(output),
            "sha256": _sha256_file(output),
            "bytes": output.stat().st_size,
            "centerlines_sha256": _sha256_array(trajectories),
            "shape": list(trajectories.shape),
        },
        "information_boundary": {
            "source_episode_only": True,
            "target_files_read": False,
            "target_metrics_computed": False,
        },
        "measurement_semantics": (
            "Ordered normalized-arc-length silhouette centerlines; nodes are "
            "pseudo-correspondences, not independently verified material tracks."
        ),
    }
    payload["result_sha256"] = rope_sequence_artifact_sha256(payload)
    return payload


def validate_rope_sequence_artifact(
    payload: Mapping[str, Any], *, verify_archive: bool = True
) -> dict[str, Any]:
    _require(
        payload.get("schema_version") == DEFORM360_ROPE_SEQUENCE_SCHEMA_VERSION,
        "unsupported rope-sequence artifact schema",
    )
    _require(
        payload.get("artifact_kind") == "Deform360SourceRopeCenterlineSequence",
        "unexpected rope-sequence artifact kind",
    )
    _require(
        payload.get("result_sha256") == rope_sequence_artifact_sha256(payload),
        "rope-sequence artifact checksum mismatch",
    )
    _require(payload.get("split") == "source", "rope sequence is not source-only")
    _require(
        payload.get("information_boundary", {}).get("target_files_read") is False,
        "rope sequence read target files",
    )
    _require(
        payload.get("quality", {}).get("passed") is True,
        "rope sequence failed temporal quality gates",
    )
    if verify_archive:
        archive = Path(payload["archive"]["path"])
        _require(archive.is_file(), "rope centerline archive is missing")
        _require(
            _sha256_file(archive) == payload["archive"]["sha256"],
            "rope centerline archive checksum mismatch",
        )
        with np.load(archive, allow_pickle=False) as stored:
            _require(
                set(stored.files) == {"frame_indices", "centerlines_m"},
                "rope centerline archive fields differ",
            )
            centerlines = np.asarray(stored["centerlines_m"], dtype=np.float64)
            _require(
                _sha256_array(centerlines) == payload["archive"]["centerlines_sha256"],
                "rope centerline array checksum mismatch",
            )
    return {
        "passed": True,
        "result_sha256": payload["result_sha256"],
        "frame_count": len(payload.get("frame_indices", [])),
    }


def write_rope_sequence_artifact(path: str | Path, payload: Mapping[str, Any]) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return output


__all__ = [
    "DEFORM360_ROPE_SEQUENCE_SCHEMA_VERSION",
    "RopeCenterlineSequenceConfig",
    "rope_sequence_artifact_sha256",
    "run_source_rope_centerline_sequence",
    "select_source_propagation_stable_cameras",
    "validate_rope_sequence_artifact",
    "write_rope_sequence_artifact",
]
