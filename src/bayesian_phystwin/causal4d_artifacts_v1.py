"""Stable legacy-artifact boundary for Causal4D integrations.

New cross-repository artifacts must use the versioned JSON/NPZ contracts. This
module exists only for hash-locked released PhysTwin pickles that cannot be
migrated retroactively.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any

import numpy as np

from .legacy_artifacts import (
    LegacyPhysTwinArtifactKind,
    load_trusted_legacy_phystwin_pickle,
)

CAUSAL4D_ARTIFACT_API_VERSION = 1
CAUSAL4D_ARTIFACT_CAPABILITIES = (
    "digest_preflight_before_pickle",
    "released_raw_track_map",
    "top_level_artifact_contract",
)


def _readonly(values: np.ndarray, *, dtype: Any) -> np.ndarray:
    array = np.asarray(values, dtype=dtype).copy()
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class ReleasedPhysTwinRawTrackMapV1:
    """Immutable raw-query correspondence required by Causal4D visual studies."""

    final_data_sha256: str
    raw_case_dir: Path
    track_paths: tuple[Path, ...]
    tracks_by_camera: tuple[np.ndarray, ...]
    visibility_by_camera: tuple[np.ndarray, ...]
    source_camera: np.ndarray
    source_track: np.ndarray

    def __post_init__(self) -> None:
        if len(self.final_data_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in self.final_data_sha256
        ):
            raise ValueError("final_data_sha256 must be a lowercase SHA-256 digest")
        paths = tuple(Path(value) for value in self.track_paths)
        tracks = tuple(_readonly(value, dtype=float) for value in self.tracks_by_camera)
        visibility = tuple(
            _readonly(value, dtype=bool) for value in self.visibility_by_camera
        )
        if not paths or len(paths) != len(tracks) or len(paths) != len(visibility):
            raise ValueError(
                "track paths, tracks, and visibility must identify each camera"
            )
        frame_count: int | None = None
        for camera, (camera_tracks, camera_visibility) in enumerate(
            zip(tracks, visibility, strict=True)
        ):
            if camera_tracks.ndim != 3 or camera_tracks.shape[2] != 2:
                raise ValueError(f"camera {camera} tracks must have shape (T, N, 2)")
            if camera_visibility.shape != camera_tracks.shape[:2]:
                raise ValueError(f"camera {camera} visibility must match tracks")
            if not np.all(np.isfinite(camera_tracks)):
                raise ValueError(f"camera {camera} tracks must be finite")
            if frame_count is None:
                frame_count = len(camera_tracks)
            elif len(camera_tracks) != frame_count:
                raise ValueError("all raw cameras must have the same frame count")
        source_camera = _readonly(self.source_camera, dtype=np.int64)
        source_track = _readonly(self.source_track, dtype=np.int64)
        if source_camera.ndim != 1 or source_track.shape != source_camera.shape:
            raise ValueError("source camera and track arrays must be matching vectors")
        if np.any(source_camera < 0) or np.any(source_camera >= len(tracks)):
            raise ValueError("source_camera references an unavailable camera")
        for camera in range(len(tracks)):
            selected = source_camera == camera
            if np.any(source_track[selected] < 0) or np.any(
                source_track[selected] >= tracks[camera].shape[1]
            ):
                raise ValueError("source_track references an unavailable raw track")
        object.__setattr__(self, "raw_case_dir", Path(self.raw_case_dir))
        object.__setattr__(self, "track_paths", paths)
        object.__setattr__(self, "tracks_by_camera", tracks)
        object.__setattr__(self, "visibility_by_camera", visibility)
        object.__setattr__(self, "source_camera", source_camera)
        object.__setattr__(self, "source_track", source_track)


def load_released_phystwin_raw_track_map(
    final_data_path: str | Path,
    raw_case_dir: str | Path,
    *,
    final_data_sha256: str,
    initial_match_tolerance_m: float = 1e-6,
) -> ReleasedPhysTwinRawTrackMapV1:
    """Load released raw-track identities after hash-locking the legacy pickle."""

    final_data = load_trusted_legacy_phystwin_pickle(
        final_data_path,
        expected_sha256=final_data_sha256,
        artifact_kind="mapping",
        required_keys=("object_points", "object_visibilities"),
    )
    module = import_module("bayesian_phystwin.phystwin_raw_cues")
    config = module.PhysTwinRawCueConfig(
        initial_match_tolerance_m=initial_match_tolerance_m
    )
    mapping = module.load_phystwin_raw_track_map(
        final_data_path,
        raw_case_dir,
        config=config,
        final_data_payload=final_data,
    )
    return ReleasedPhysTwinRawTrackMapV1(
        final_data_sha256=final_data_sha256,
        raw_case_dir=Path(raw_case_dir),
        track_paths=tuple(mapping.track_paths),
        tracks_by_camera=tuple(mapping.tracks_by_camera),
        visibility_by_camera=tuple(mapping.visibility_by_camera),
        source_camera=mapping.source_camera,
        source_track=mapping.source_track,
    )


def causal4d_artifact_provider_manifest() -> dict[str, object]:
    """Return the stable legacy-artifact provider descriptor."""

    return {
        "provider_api": "bayesian_phystwin.causal4d_artifacts_v1",
        "provider_api_version": CAUSAL4D_ARTIFACT_API_VERSION,
        "capabilities": list(CAUSAL4D_ARTIFACT_CAPABILITIES),
        "new_artifact_policy": "json-npz-only",
    }


__all__ = [
    "CAUSAL4D_ARTIFACT_API_VERSION",
    "CAUSAL4D_ARTIFACT_CAPABILITIES",
    "LegacyPhysTwinArtifactKind",
    "ReleasedPhysTwinRawTrackMapV1",
    "causal4d_artifact_provider_manifest",
    "load_released_phystwin_raw_track_map",
    "load_trusted_legacy_phystwin_pickle",
]
