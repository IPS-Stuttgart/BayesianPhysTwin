"""Digest-complete released PhysTwin visual-mapping inputs for Causal4D.

The released visual preprocessing stack contains legacy pickle and NPZ files.
This module verifies every declared byte identity before opening any payload,
revalidates the same identities after loading, and returns immutable typed data.
"""

from __future__ import annotations

import hashlib
import hmac
import json
from collections.abc import Mapping
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any

import numpy as np

from ._canonical_contracts import immutable_array
from .legacy_artifacts import load_trusted_legacy_phystwin_pickle
from .phystwin.artifacts import sha256_file

CAUSAL4D_ARTIFACT_API_VERSION = 2
CAUSAL4D_ARTIFACT_CAPABILITIES = (
    "digest_preflight_before_all_visual_payloads",
    "immutable_released_visual_inputs",
    "postload_digest_revalidation",
    "released_raw_track_correspondence",
)


def _digest(value: str, *, name: str) -> str:
    normalized = str(value)
    if (
        normalized != normalized.lower()
        or len(normalized) != 64
        or any(character not in "0123456789abcdef" for character in normalized)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return normalized


def _readonly(values: np.ndarray, *, dtype: Any) -> np.ndarray:
    return immutable_array(values, dtype=dtype)


def _verify_identity(path: Path, expected_sha256: str, *, name: str) -> None:
    expected = _digest(expected_sha256, name=f"{name} SHA-256")
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = sha256_file(path)
    if not hmac.compare_digest(actual, expected):
        raise ValueError(f"{name} SHA-256 mismatch; refusing to open released input")


def _canonical_artifact_id(payload: Mapping[str, object]) -> str:
    encoded = json.dumps(
        dict(payload),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class ReleasedPhysTwinVisualInputsV2:
    """Immutable visual-mapping bundle with complete released-file provenance."""

    raw_case_dir: Path
    final_data_sha256: str
    metadata_sha256: str
    pcd_sha256: str
    calibration_sha256: str
    cotracker_sha256: tuple[tuple[str, str], ...]
    initial_match_tolerance_m: float
    object_points_m: np.ndarray
    object_visibility: np.ndarray
    object_motion_valid: np.ndarray
    track_paths: tuple[Path, ...]
    tracks_by_camera: tuple[np.ndarray, ...]
    visibility_by_camera: tuple[np.ndarray, ...]
    source_camera: np.ndarray
    source_track: np.ndarray
    source_world_points_m: np.ndarray
    initial_match_distance_m: np.ndarray
    intrinsics: np.ndarray
    camera_to_world: np.ndarray
    source_fps: float
    image_width: int
    image_height: int

    def __post_init__(self) -> None:
        raw_case_dir = Path(self.raw_case_dir)
        final_digest = _digest(self.final_data_sha256, name="final_data_sha256")
        metadata_digest = _digest(self.metadata_sha256, name="metadata_sha256")
        pcd_digest = _digest(self.pcd_sha256, name="pcd_sha256")
        calibration_digest = _digest(self.calibration_sha256, name="calibration_sha256")
        track_digests = tuple(
            (str(name), _digest(value, name=name))
            for name, value in self.cotracker_sha256
        )
        if not track_digests or len({name for name, _ in track_digests}) != len(
            track_digests
        ):
            raise ValueError("cotracker_sha256 must identify unique nonempty archives")
        if any(
            not name.startswith("cotracker/") or name.endswith("/")
            for name, _ in track_digests
        ):
            raise ValueError(
                "cotracker identities must use cotracker/<archive>.npz paths"
            )
        if (
            not np.isfinite(self.initial_match_tolerance_m)
            or self.initial_match_tolerance_m <= 0.0
        ):
            raise ValueError("initial_match_tolerance_m must be positive and finite")

        object_points = _readonly(self.object_points_m, dtype=float)
        visibility = _readonly(self.object_visibility, dtype=bool)
        motion_valid = _readonly(self.object_motion_valid, dtype=bool)
        if object_points.ndim != 3 or object_points.shape[2] != 3:
            raise ValueError("object_points_m must have shape (T, N, 3)")
        if visibility.shape != object_points.shape[:2]:
            raise ValueError("object_visibility must match object_points_m")
        if motion_valid.shape != object_points.shape[:2]:
            raise ValueError("object_motion_valid must match object_points_m")
        if not np.all(np.isfinite(object_points)):
            raise ValueError("object_points_m must be finite")

        paths = tuple(Path(value) for value in self.track_paths)
        tracks = tuple(_readonly(value, dtype=float) for value in self.tracks_by_camera)
        raw_visibility = tuple(
            _readonly(value, dtype=bool) for value in self.visibility_by_camera
        )
        if not paths or len(paths) != len(tracks) or len(paths) != len(raw_visibility):
            raise ValueError(
                "track paths, tracks, and visibility must identify each camera"
            )
        expected_track_names = tuple(
            path.relative_to(raw_case_dir).as_posix() for path in paths
        )
        if expected_track_names != tuple(name for name, _ in track_digests):
            raise ValueError(
                "cotracker_sha256 order must match the released track paths"
            )
        frame_count, object_count = object_points.shape[:2]
        for camera, (camera_tracks, camera_visibility) in enumerate(
            zip(tracks, raw_visibility, strict=True)
        ):
            if camera_tracks.ndim != 3 or camera_tracks.shape[2] != 2:
                raise ValueError(f"camera {camera} tracks must have shape (T, N, 2)")
            if len(camera_tracks) != frame_count:
                raise ValueError("raw tracks must use the final-data frame count")
            if camera_visibility.shape != camera_tracks.shape[:2]:
                raise ValueError(f"camera {camera} visibility must match tracks")
            if not np.all(np.isfinite(camera_tracks)):
                raise ValueError(f"camera {camera} tracks must be finite")

        source_camera = _readonly(self.source_camera, dtype=np.int64)
        source_track = _readonly(self.source_track, dtype=np.int64)
        source_world = _readonly(self.source_world_points_m, dtype=float)
        match_distance = _readonly(self.initial_match_distance_m, dtype=float)
        if source_camera.shape != (object_count,) or source_track.shape != (
            object_count,
        ):
            raise ValueError("source camera and track must identify every object point")
        if source_world.shape != (object_count, 3):
            raise ValueError("source_world_points_m must have shape (N, 3)")
        if match_distance.shape != (object_count,):
            raise ValueError("initial_match_distance_m must have shape (N,)")
        if not np.all(np.isfinite(source_world)) or not np.all(
            np.isfinite(match_distance)
        ):
            raise ValueError("source-world points and match distances must be finite")
        if np.any(match_distance < 0.0) or np.any(
            match_distance > self.initial_match_tolerance_m
        ):
            raise ValueError("initial match distance exceeds the declared tolerance")
        if np.any(source_camera < 0) or np.any(source_camera >= len(tracks)):
            raise ValueError("source_camera references an unavailable camera")
        for camera in range(len(tracks)):
            selected = source_camera == camera
            if np.any(source_track[selected] < 0) or np.any(
                source_track[selected] >= tracks[camera].shape[1]
            ):
                raise ValueError("source_track references an unavailable raw track")

        intrinsics = _readonly(self.intrinsics, dtype=float)
        camera_to_world = _readonly(self.camera_to_world, dtype=float)
        if intrinsics.shape != (len(tracks), 3, 3):
            raise ValueError("intrinsics must have shape (C, 3, 3)")
        if camera_to_world.shape != (len(tracks), 4, 4):
            raise ValueError("camera_to_world must have shape (C, 4, 4)")
        if not np.all(np.isfinite(intrinsics)) or not np.all(
            np.isfinite(camera_to_world)
        ):
            raise ValueError("camera calibration arrays must be finite")
        if not np.isfinite(self.source_fps) or self.source_fps <= 0.0:
            raise ValueError("source_fps must be positive and finite")
        if self.image_width < 1 or self.image_height < 1:
            raise ValueError("image dimensions must be positive")

        object.__setattr__(self, "raw_case_dir", raw_case_dir)
        object.__setattr__(self, "final_data_sha256", final_digest)
        object.__setattr__(self, "metadata_sha256", metadata_digest)
        object.__setattr__(self, "pcd_sha256", pcd_digest)
        object.__setattr__(self, "calibration_sha256", calibration_digest)
        object.__setattr__(self, "cotracker_sha256", track_digests)
        object.__setattr__(self, "object_points_m", object_points)
        object.__setattr__(self, "object_visibility", visibility)
        object.__setattr__(self, "object_motion_valid", motion_valid)
        object.__setattr__(self, "track_paths", paths)
        object.__setattr__(self, "tracks_by_camera", tracks)
        object.__setattr__(self, "visibility_by_camera", raw_visibility)
        object.__setattr__(self, "source_camera", source_camera)
        object.__setattr__(self, "source_track", source_track)
        object.__setattr__(self, "source_world_points_m", source_world)
        object.__setattr__(self, "initial_match_distance_m", match_distance)
        object.__setattr__(self, "intrinsics", intrinsics)
        object.__setattr__(self, "camera_to_world", camera_to_world)

    @property
    def artifact_id(self) -> str:
        """Return a content identity for the complete released-input contract."""

        return _canonical_artifact_id(
            {
                "schema_version": CAUSAL4D_ARTIFACT_API_VERSION,
                "final_data_sha256": self.final_data_sha256,
                "metadata_sha256": self.metadata_sha256,
                "pcd_sha256": self.pcd_sha256,
                "calibration_sha256": self.calibration_sha256,
                "cotracker_sha256": list(self.cotracker_sha256),
                "initial_match_tolerance_m": self.initial_match_tolerance_m,
                "frame_count": int(self.object_points_m.shape[0]),
                "object_count": int(self.object_points_m.shape[1]),
                "camera_count": len(self.track_paths),
            }
        )

    def input_digests(self) -> dict[str, str]:
        """Return a JSON-safe copy of every verified released input identity."""

        return {
            "final_data.pkl": self.final_data_sha256,
            "metadata.json": self.metadata_sha256,
            "pcd/0.npz": self.pcd_sha256,
            "calibrate.pkl": self.calibration_sha256,
            **dict(self.cotracker_sha256),
        }


def load_released_phystwin_visual_inputs(
    final_data_path: str | Path,
    raw_case_dir: str | Path,
    *,
    final_data_sha256: str,
    metadata_sha256: str,
    pcd_sha256: str,
    calibration_sha256: str,
    cotracker_sha256: Mapping[str, str],
    initial_match_tolerance_m: float = 1e-6,
) -> ReleasedPhysTwinVisualInputsV2:
    """Verify all mapping/calibration inputs used by visual query preparation."""

    final_path = Path(final_data_path)
    raw_path = Path(raw_case_dir)
    metadata_path = raw_path / "metadata.json"
    pcd_path = raw_path / "pcd" / "0.npz"
    calibration_path = raw_path / "calibrate.pkl"
    track_paths = tuple(sorted((raw_path / "cotracker").glob("*.npz")))
    if not track_paths:
        raise FileNotFoundError("raw case contains no cotracker NPZ files")
    actual_track_names = tuple(
        path.relative_to(raw_path).as_posix() for path in track_paths
    )
    expected_tracks = {
        str(name): str(value) for name, value in cotracker_sha256.items()
    }
    if set(expected_tracks) != set(actual_track_names):
        missing = sorted(set(actual_track_names) - set(expected_tracks))
        extra = sorted(set(expected_tracks) - set(actual_track_names))
        raise ValueError(
            f"cotracker digest inventory differs: missing={missing}, extra={extra}"
        )

    identities = {
        "final_data.pkl": (final_path, final_data_sha256),
        "metadata.json": (metadata_path, metadata_sha256),
        "pcd/0.npz": (pcd_path, pcd_sha256),
        "calibrate.pkl": (calibration_path, calibration_sha256),
        **{
            name: (raw_path / name, expected_tracks[name])
            for name in actual_track_names
        },
    }
    for name, (path, digest) in identities.items():
        _verify_identity(path, digest, name=name)

    final_data = load_trusted_legacy_phystwin_pickle(
        final_path,
        expected_sha256=final_data_sha256,
        artifact_kind="mapping",
        required_keys=(
            "object_points",
            "object_visibilities",
            "object_motions_valid",
        ),
    )
    camera_to_world = np.asarray(
        load_trusted_legacy_phystwin_pickle(
            calibration_path,
            expected_sha256=calibration_sha256,
            artifact_kind="ndarray",
        ),
        dtype=float,
    )
    metadata_payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(metadata_payload, Mapping):
        raise ValueError("metadata.json must contain an object")

    module = import_module("bayesian_phystwin.phystwin_raw_cues")
    config = module.PhysTwinRawCueConfig(
        initial_match_tolerance_m=initial_match_tolerance_m
    )
    mapping = module.load_phystwin_raw_track_map(
        final_path,
        raw_path,
        config=config,
        final_data_payload=final_data,
    )

    for name, (path, digest) in identities.items():
        _verify_identity(path, digest, name=name)

    object_points = np.asarray(final_data["object_points"], dtype=float)
    object_visibility = np.asarray(final_data["object_visibilities"], dtype=bool)
    object_motion_valid = np.asarray(final_data["object_motions_valid"], dtype=bool)
    if not np.array_equal(np.asarray(mapping.final_points), object_points):
        raise ValueError(
            "raw-track mapping final points differ from the trusted final data"
        )
    if not np.array_equal(np.asarray(mapping.final_visible), object_visibility):
        raise ValueError(
            "raw-track mapping visibility differs from the trusted final data"
        )

    intrinsics = np.asarray(metadata_payload["intrinsics"], dtype=float)
    image_width, image_height = map(int, metadata_payload["WH"])
    source_fps = float(metadata_payload["fps"])
    return ReleasedPhysTwinVisualInputsV2(
        raw_case_dir=raw_path,
        final_data_sha256=final_data_sha256,
        metadata_sha256=metadata_sha256,
        pcd_sha256=pcd_sha256,
        calibration_sha256=calibration_sha256,
        cotracker_sha256=tuple(
            (name, _digest(expected_tracks[name], name=name))
            for name in actual_track_names
        ),
        initial_match_tolerance_m=float(initial_match_tolerance_m),
        object_points_m=object_points,
        object_visibility=object_visibility,
        object_motion_valid=object_motion_valid,
        track_paths=tuple(mapping.track_paths),
        tracks_by_camera=tuple(mapping.tracks_by_camera),
        visibility_by_camera=tuple(mapping.visibility_by_camera),
        source_camera=mapping.source_camera,
        source_track=mapping.source_track,
        source_world_points_m=mapping.source_world_points,
        initial_match_distance_m=mapping.initial_match_distance_m,
        intrinsics=intrinsics,
        camera_to_world=camera_to_world,
        source_fps=source_fps,
        image_width=image_width,
        image_height=image_height,
    )


def causal4d_artifact_provider_manifest() -> dict[str, object]:
    """Return the complete released visual-mapping provider descriptor."""

    return {
        "provider_api": "bayesian_phystwin.causal4d_artifacts_v2",
        "provider_api_version": CAUSAL4D_ARTIFACT_API_VERSION,
        "capabilities": list(CAUSAL4D_ARTIFACT_CAPABILITIES),
        "artifact_schema_versions": {"ReleasedPhysTwinVisualInputs": 2},
        "new_artifact_policy": "json-npz-only",
    }


__all__ = [
    "CAUSAL4D_ARTIFACT_API_VERSION",
    "CAUSAL4D_ARTIFACT_CAPABILITIES",
    "ReleasedPhysTwinVisualInputsV2",
    "causal4d_artifact_provider_manifest",
    "load_released_phystwin_visual_inputs",
]
