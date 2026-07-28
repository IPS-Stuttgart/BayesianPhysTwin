"""Prefix-only camera admission for Dynamic TAPNext++ V2.

V1 source preparation admitted episodes with at least eight complete cameras,
but its provider later required all twelve frozen cameras. V2 makes the
complete-camera set an explicit, checksummed input. Only frames 0--57 may
contribute to the certificate.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .deform360_dynamic_tapnextpp_source_window import FROZEN_CAMERA_PANEL
from .observation_belief import array_sha256

CAUSAL_FRAME_COUNT = 58
PROTOCOL_ID = "deform360-dynamic-tapnextpp-admission-v2"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _valid_digest(value: str) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _canonical_sha256(payload: Mapping[str, Any], *, digest_key: str) -> str:
    canonical = dict(payload)
    canonical.pop(digest_key, None)
    return hashlib.sha256(
        json.dumps(
            canonical,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _load_calibration_dict(path: Path) -> dict[str, np.ndarray]:
    stored = np.load(path, allow_pickle=True)
    _require(stored.shape == (), f"calibration archive is not a dictionary: {path}")
    mapping = stored.item()
    _require(isinstance(mapping, dict), f"calibration archive is invalid: {path}")
    return {str(name): np.asarray(value) for name, value in mapping.items()}


@dataclass(frozen=True)
class CameraPrefixEvidence:
    """Evidence that one camera has a complete causal RGB-D-mask prefix."""

    image_shape_hw: tuple[int, int]
    rgb_prefix_sha256: str
    depth_prefix_sha256: str
    mask_prefix_sha256: str

    def __post_init__(self) -> None:
        _require(
            len(self.image_shape_hw) == 2
            and all(int(value) > 0 for value in self.image_shape_hw),
            "camera image shape is invalid",
        )
        for name, digest in (
            ("rgb_prefix_sha256", self.rgb_prefix_sha256),
            ("depth_prefix_sha256", self.depth_prefix_sha256),
            ("mask_prefix_sha256", self.mask_prefix_sha256),
        ):
            _require(_valid_digest(digest), f"{name} is invalid")


PrefixProbe = Callable[[Path, int], CameraPrefixEvidence]


def probe_camera_prefix(
    camera_dir: Path,
    frame_count: int = CAUSAL_FRAME_COUNT,
) -> CameraPrefixEvidence:
    """Decode and hash exactly the permitted prefix of one camera."""

    import cv2
    import h5py

    _require(
        1 <= frame_count <= CAUSAL_FRAME_COUNT,
        "causal frame count is outside the certified prefix",
    )
    video_path = camera_dir / "undistorted.mp4"
    capture = cv2.VideoCapture(str(video_path))
    rgb_frames: list[np.ndarray] = []
    try:
        for frame_index in range(frame_count):
            ok, bgr = capture.read()
            observed = int(capture.get(cv2.CAP_PROP_POS_FRAMES)) - 1
            _require(
                bool(ok) and observed == frame_index,
                f"cannot decode exact RGB frame {frame_index}: {video_path}",
            )
            rgb_frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    finally:
        capture.release()
    rgb = np.stack(rgb_frames).astype(np.uint8, copy=False)

    prefixes: dict[str, np.ndarray] = {}
    for filename in ("rendered_depth.h5", "mask_refined.h5"):
        path = camera_dir / filename
        with h5py.File(path, "r") as stream:
            _require(
                "data" in stream
                and stream["data"].ndim == 3
                and len(stream["data"]) >= frame_count,
                f"invalid causal HDF5 archive: {path}",
            )
            prefixes[filename] = np.asarray(stream["data"][:frame_count])
    depth = prefixes["rendered_depth.h5"]
    mask = prefixes["mask_refined.h5"]
    _require(
        rgb.shape[:-1] == depth.shape == mask.shape,
        f"causal camera stream shapes differ: {camera_dir.name}",
    )
    return CameraPrefixEvidence(
        image_shape_hw=tuple(map(int, depth.shape[-2:])),
        rgb_prefix_sha256=array_sha256(rgb),
        depth_prefix_sha256=array_sha256(depth),
        mask_prefix_sha256=array_sha256(mask),
    )


@dataclass(frozen=True)
class CompleteCameraGeometry:
    """Calibrated camera geometry restricted to complete causal streams."""

    camera_names: tuple[str, ...]
    frozen_panel_indices: np.ndarray
    intrinsics: np.ndarray
    camera_to_world: np.ndarray
    image_shapes_hw: np.ndarray
    prefix_evidence: tuple[CameraPrefixEvidence, ...]
    causal_frame_count: int
    rejected_cameras: Mapping[str, str]
    calibration_prefix_sha256: str
    artifact_sha256: str

    def __post_init__(self) -> None:
        indices = np.ascontiguousarray(
            np.asarray(self.frozen_panel_indices, dtype=np.int64)
        )
        intrinsics = np.ascontiguousarray(
            np.asarray(self.intrinsics, dtype=np.float64)
        )
        poses = np.ascontiguousarray(
            np.asarray(self.camera_to_world, dtype=np.float64)
        )
        shapes = np.ascontiguousarray(
            np.asarray(self.image_shapes_hw, dtype=np.int64)
        )
        count = len(self.camera_names)
        _require(
            1 <= self.causal_frame_count <= CAUSAL_FRAME_COUNT,
            "complete-camera causal frame count is invalid",
        )
        _require(count >= 8, "fewer than eight complete cameras")
        _require(
            indices.shape == (count,)
            and intrinsics.shape == (count, 3, 3)
            and poses.shape == (count, 4, 4)
            and shapes.shape == (count, 2)
            and len(self.prefix_evidence) == count,
            "complete-camera arrays have inconsistent shapes",
        )
        _require(
            len(set(self.camera_names)) == count
            and len(set(map(int, indices))) == count,
            "complete-camera identities are not unique",
        )
        _require(
            np.all(np.isfinite(intrinsics))
            and np.all(np.isfinite(poses))
            and np.all(shapes > 0),
            "complete-camera geometry is invalid",
        )
        _require(
            _valid_digest(self.calibration_prefix_sha256)
            and _valid_digest(self.artifact_sha256),
            "complete-camera digest is invalid",
        )
        for values in (indices, intrinsics, poses, shapes):
            values.setflags(write=False)
        object.__setattr__(self, "frozen_panel_indices", indices)
        object.__setattr__(self, "intrinsics", intrinsics)
        object.__setattr__(self, "camera_to_world", poses)
        object.__setattr__(self, "image_shapes_hw", shapes)
        object.__setattr__(
            self,
            "rejected_cameras",
            dict(sorted((str(key), str(value)) for key, value in self.rejected_cameras.items())),
        )

    def descriptor(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "artifact_kind": "Deform360DynamicTAPNextPPCompleteCameraGeometry",
            "protocol_id": PROTOCOL_ID,
            "causal_frame_range_half_open": [0, self.causal_frame_count],
            "camera_names": list(self.camera_names),
            "frozen_panel_indices": self.frozen_panel_indices.tolist(),
            "intrinsics_sha256": array_sha256(self.intrinsics),
            "camera_to_world_sha256": array_sha256(self.camera_to_world),
            "image_shapes_hw": self.image_shapes_hw.tolist(),
            "prefix_evidence": [
                {
                    "camera_name": camera,
                    "rgb_prefix_sha256": evidence.rgb_prefix_sha256,
                    "depth_prefix_sha256": evidence.depth_prefix_sha256,
                    "mask_prefix_sha256": evidence.mask_prefix_sha256,
                }
                for camera, evidence in zip(
                    self.camera_names,
                    self.prefix_evidence,
                    strict=True,
                )
            ],
            "rejected_cameras": dict(self.rejected_cameras),
            "calibration_prefix_sha256": self.calibration_prefix_sha256,
            "information_boundary": {
                "maximum_rgb_depth_mask_frame_read": self.causal_frame_count - 1,
                "future_object_observation_read": False,
                "target_metric_read": False,
            },
        }


def load_complete_camera_geometry(
    processed_episode_dir: str | Path,
    *,
    candidate_camera_names: Sequence[str] = FROZEN_CAMERA_PANEL,
    minimum_complete_camera_count: int = 8,
    prefix_probe: PrefixProbe = probe_camera_prefix,
    frame_count: int | None = None,
) -> CompleteCameraGeometry:
    """Load only cameras whose calibration and causal streams are complete."""

    root = Path(processed_episode_dir).resolve()
    names = tuple(map(str, candidate_camera_names))
    _require(
        len(names) == len(set(names)),
        "candidate camera names are not unique",
    )
    _require(
        minimum_complete_camera_count >= 8,
        "claim-bearing camera minimum is below eight",
    )
    requested_frame_count = (
        CAUSAL_FRAME_COUNT if frame_count is None else int(frame_count)
    )
    _require(
        1 <= requested_frame_count <= CAUSAL_FRAME_COUNT,
        "requested camera certificate exceeds the causal prefix",
    )
    intrinsics_path = root / "undistorted_intrinsics.npy"
    extrinsics_path = root / "extrinsics.npy"
    intrinsics_by_name = _load_calibration_dict(intrinsics_path)
    poses_by_name = _load_calibration_dict(extrinsics_path)

    complete_names: list[str] = []
    complete_indices: list[int] = []
    intrinsics: list[np.ndarray] = []
    poses: list[np.ndarray] = []
    shapes: list[tuple[int, int]] = []
    evidence_rows: list[CameraPrefixEvidence] = []
    rejected: dict[str, str] = {}
    for index, camera in enumerate(names):
        if camera not in intrinsics_by_name or camera not in poses_by_name:
            rejected[camera] = "missing_calibration"
            continue
        intrinsic = np.asarray(intrinsics_by_name[camera], dtype=np.float64)
        pose = np.asarray(poses_by_name[camera], dtype=np.float64)
        if (
            intrinsic.shape != (3, 3)
            or pose.shape != (4, 4)
            or not np.all(np.isfinite(intrinsic))
            or not np.all(np.isfinite(pose))
        ):
            rejected[camera] = "invalid_calibration"
            continue
        camera_dir = root / camera
        required_paths = (
            camera_dir / "undistorted.mp4",
            camera_dir / "rendered_depth.h5",
            camera_dir / "mask_refined.h5",
        )
        if not all(path.is_file() for path in required_paths):
            rejected[camera] = "missing_causal_stream"
            continue
        try:
            evidence = prefix_probe(camera_dir, requested_frame_count)
        except (OSError, RuntimeError, ValueError) as error:
            rejected[camera] = f"invalid_causal_prefix:{type(error).__name__}"
            continue
        complete_names.append(camera)
        complete_indices.append(index)
        intrinsics.append(intrinsic)
        poses.append(pose)
        shapes.append(evidence.image_shape_hw)
        evidence_rows.append(evidence)

    _require(
        len(complete_names) >= minimum_complete_camera_count,
        "too few complete causal cameras",
    )
    intrinsics_array = np.stack(intrinsics)
    poses_array = np.stack(poses)
    shapes_array = np.asarray(shapes, dtype=np.int64)
    calibration_prefix_sha256 = hashlib.sha256(
        (
            array_sha256(intrinsics_array)
            + array_sha256(poses_array)
            + json.dumps(complete_names, separators=(",", ":"))
        ).encode("ascii")
    ).hexdigest()
    provisional = CompleteCameraGeometry(
        camera_names=tuple(complete_names),
        frozen_panel_indices=np.asarray(complete_indices, dtype=np.int64),
        intrinsics=intrinsics_array,
        camera_to_world=poses_array,
        image_shapes_hw=shapes_array,
        prefix_evidence=tuple(evidence_rows),
        causal_frame_count=requested_frame_count,
        rejected_cameras=rejected,
        calibration_prefix_sha256=calibration_prefix_sha256,
        artifact_sha256="0" * 64,
    )
    artifact_sha256 = _canonical_sha256(
        provisional.descriptor(),
        digest_key="artifact_sha256",
    )
    result = CompleteCameraGeometry(
        camera_names=provisional.camera_names,
        frozen_panel_indices=provisional.frozen_panel_indices,
        intrinsics=provisional.intrinsics,
        camera_to_world=provisional.camera_to_world,
        image_shapes_hw=provisional.image_shapes_hw,
        prefix_evidence=provisional.prefix_evidence,
        causal_frame_count=provisional.causal_frame_count,
        rejected_cameras=provisional.rejected_cameras,
        calibration_prefix_sha256=provisional.calibration_prefix_sha256,
        artifact_sha256=artifact_sha256,
    )
    _require(
        _canonical_sha256(
            result.descriptor(),
            digest_key="artifact_sha256",
        )
        == result.artifact_sha256,
        "complete-camera descriptor changed after construction",
    )
    return result


def load_selected_complete_causal_inputs(
    processed_episode_dir: str | Path,
    geometry: CompleteCameraGeometry,
    camera_indices: Sequence[int],
    *,
    depth_scale_to_m: float = 0.001,
    frame_count: int | None = None,
) -> Any:
    """Decode the causal prefix from cameras admitted by the V2 certificate."""

    from .deform360_dynamic_tapnextpp_provider import (
        CausalCameraInputs,
        _decode_rgb_prefix,
        _read_h5_prefix,
    )

    root = Path(processed_episode_dir).resolve()
    indices = np.asarray(camera_indices, dtype=np.int64)
    _require(
        indices.shape == (8,)
        and len(np.unique(indices)) == 8
        and np.all((indices >= 0) & (indices < len(geometry.camera_names))),
        "selected complete-camera indices are invalid",
    )
    _require(
        np.isfinite(depth_scale_to_m) and depth_scale_to_m > 0.0,
        "depth scale must be finite and positive",
    )
    requested_frame_count = (
        CAUSAL_FRAME_COUNT if frame_count is None else int(frame_count)
    )
    _require(
        1 <= requested_frame_count <= geometry.causal_frame_count,
        "requested causal frame count is outside the certified prefix",
    )
    selected_names = tuple(
        geometry.camera_names[int(index)] for index in indices
    )
    rgbs: list[np.ndarray] = []
    depths: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    prefix_hashes: dict[str, Any] = {}
    for camera in selected_names:
        directory = root / camera
        rgb = _decode_rgb_prefix(
            directory / "undistorted.mp4",
            requested_frame_count,
        )
        encoded_depth = _read_h5_prefix(
            directory / "rendered_depth.h5",
            requested_frame_count,
        )
        mask = _read_h5_prefix(
            directory / "mask_refined.h5",
            requested_frame_count,
        ).astype(bool, copy=False)
        depth = encoded_depth.astype(np.float32) * depth_scale_to_m
        _require(
            rgb.shape[:-1] == depth.shape == mask.shape,
            f"selected causal camera shapes differ: {camera}",
        )
        rgbs.append(rgb)
        depths.append(depth)
        masks.append(mask)
        prefix_hashes[camera] = {
            "decoded_rgb_sha256": array_sha256(rgb),
            "decoded_depth_m_sha256": array_sha256(depth),
            "decoded_mask_sha256": array_sha256(mask),
        }
    return CausalCameraInputs(
        camera_indices=indices,
        camera_names=selected_names,
        rgbs=np.stack(rgbs),
        depths_m=np.stack(depths),
        object_masks=np.stack(masks),
        intrinsics=geometry.intrinsics[indices],
        camera_to_world=geometry.camera_to_world[indices],
        provenance={
            "complete_camera_certificate_sha256": geometry.artifact_sha256,
            "selected_complete_camera_indices": indices.tolist(),
            "selected_camera_names": list(selected_names),
            "maximum_frame_read": requested_frame_count - 1,
            "future_frame_read": False,
            "decoded_prefix_sha256": prefix_hashes,
        },
    )


__all__ = [
    "CAUSAL_FRAME_COUNT",
    "PROTOCOL_ID",
    "CameraPrefixEvidence",
    "CompleteCameraGeometry",
    "load_complete_camera_geometry",
    "load_selected_complete_causal_inputs",
    "probe_camera_prefix",
]
