"""Per-camera causal observations for cross-view Deform360 validation.

The frozen raw-camera artifact stores only its robustly triangulated 3-D
measurement.  This opt-in supplement replays the same RGB prefixes and keeps
the camera-level tracked pixels needed for a disjoint fit/validation split.  It
does not replace or mutate the frozen artifact and accepts no outcome input.
"""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .deform360_bias_aware_prospective_artifacts import (
    canonical_sha256,
    file_sha256,
)
from .deform360_raw_camera_observation import (
    AllTrackerPrefixRuntime,
    MANIFEST_FILENAME as SOURCE_MANIFEST_FILENAME,
    MEASUREMENT_FILENAME as SOURCE_ARCHIVE_FILENAME,
    RawCameraObservationConfig,
    _canonical_sha256 as source_canonical_sha256,
    _load_calibration,
    frame_zero_camera_support,
)


PROTOCOL_ID = "deform360-crossview-track-supplement-v1-development"
ARCHIVE_FILENAME = "crossview_tracks.npz"
MANIFEST_FILENAME = "crossview_tracks_manifest.json"
ARTIFACT_KIND = "Deform360CausalCrossViewTrackSupplement"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"JSON object expected: {path}")
    return value


def _source_config(value: Mapping[str, Any]) -> RawCameraObservationConfig:
    config = dict(value)
    config["update_frames"] = tuple(int(frame) for frame in config["update_frames"])
    return RawCameraObservationConfig(**config)


def load_source_raw_camera_config(
    measurement_dir: str | Path,
) -> RawCameraObservationConfig:
    """Load the checksummed configuration of a frozen source measurement."""

    root = Path(measurement_dir).resolve()
    manifest = _load_json(root / SOURCE_MANIFEST_FILENAME)
    _validate_source_manifest(manifest)
    return _source_config(manifest["config"])


def _validate_source_manifest(manifest: Mapping[str, Any]) -> None:
    unsigned = dict(manifest)
    claimed = unsigned.pop("result_sha256", None)
    _require(
        isinstance(claimed, str) and claimed == source_canonical_sha256(unsigned),
        "source measurement manifest checksum changed",
    )
    boundary = manifest.get("information_boundary", {})
    _require(
        boundary.get("target_data_read") is False
        and boundary.get("outcome_manifest_read") is False
        and boundary.get("future_reconstruction_after_frame_zero_read") is False,
        "source measurement crossed its causal boundary",
    )


def align_camera_tracks(
    center_ids: np.ndarray,
    query_ids: np.ndarray,
    tracks_xy: np.ndarray,
    visible: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Align one camera's query-ordered tracks to a common centre ordering."""

    centers = np.asarray(center_ids, dtype=np.int64)
    queries = np.asarray(query_ids, dtype=np.int64)
    tracks = np.asarray(tracks_xy, dtype=np.float32)
    mask = np.asarray(visible, dtype=bool)
    _require(centers.ndim == 1 and len(np.unique(centers)) == len(centers), "centres changed")
    _require(queries.ndim == 1 and len(np.unique(queries)) == len(queries), "queries changed")
    _require(tracks.shape == (len(queries), 2), "track shape changed")
    _require(mask.shape == (len(queries),), "visibility shape changed")
    lookup = {int(point_id): index for index, point_id in enumerate(centers)}
    _require(all(int(point_id) in lookup for point_id in queries), "query is not a centre")
    aligned = np.full((len(centers), 2), np.nan, dtype=np.float32)
    aligned_visible = np.zeros(len(centers), dtype=bool)
    for source_index, point_id in enumerate(queries):
        destination = lookup[int(point_id)]
        aligned[destination] = tracks[source_index]
        aligned_visible[destination] = bool(mask[source_index])
    aligned_visible &= np.all(np.isfinite(aligned), axis=1)
    return aligned, aligned_visible


def _validate_arrays(arrays: Mapping[str, np.ndarray]) -> None:
    required = {
        "track_pixels_xy",
        "track_visibility",
        "frame_zero_pixels_xy",
        "frame_zero_support",
        "center_ids",
        "selected_cameras",
        "update_frames",
        "center_frame_zero_points_m",
        "intrinsics",
        "camera_to_world",
    }
    _require(required.issubset(arrays), "cross-view archive is incomplete")
    tracks = np.asarray(arrays["track_pixels_xy"])
    visibility = np.asarray(arrays["track_visibility"])
    initial_pixels = np.asarray(arrays["frame_zero_pixels_xy"])
    initial_support = np.asarray(arrays["frame_zero_support"])
    centers = np.asarray(arrays["center_ids"])
    cameras = np.asarray(arrays["selected_cameras"])
    updates = np.asarray(arrays["update_frames"])
    points = np.asarray(arrays["center_frame_zero_points_m"])
    intrinsics = np.asarray(arrays["intrinsics"])
    camera_to_world = np.asarray(arrays["camera_to_world"])
    expected = (len(updates), len(cameras), len(centers))
    _require(tracks.shape == (*expected, 2), "cross-view track shape changed")
    _require(visibility.shape == expected, "cross-view visibility shape changed")
    _require(initial_pixels.shape == (len(cameras), len(centers), 2), "initial pixels changed")
    _require(initial_support.shape == (len(cameras), len(centers)), "initial support changed")
    _require(points.shape == (len(centers), 3), "frame-zero centre shape changed")
    _require(intrinsics.shape == (len(cameras), 3, 3), "intrinsics shape changed")
    _require(camera_to_world.shape == (len(cameras), 4, 4), "extrinsics shape changed")
    _require(len(np.unique(centers)) == len(centers), "centre IDs are not unique")
    _require(len(np.unique(cameras)) == len(cameras), "camera IDs are not unique")
    _require(np.all(np.diff(updates.astype(np.int64)) > 0), "update frames changed")
    _require(np.all(np.isfinite(points)), "frame-zero centres are non-finite")
    _require(np.all(np.isfinite(intrinsics)), "intrinsics are non-finite")
    _require(np.all(np.isfinite(camera_to_world)), "extrinsics are non-finite")
    _require(
        np.all(np.isfinite(tracks[visibility])),
        "visible camera tracks are non-finite",
    )


def load_crossview_track_supplement(
    artifact_dir: str | Path,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Load and checksum one cross-view supplement."""

    root = Path(artifact_dir).resolve()
    manifest_path = root / MANIFEST_FILENAME
    archive_path = root / ARCHIVE_FILENAME
    manifest = _load_json(manifest_path)
    _require(
        manifest.get("artifact_kind") == ARTIFACT_KIND
        and manifest.get("protocol_id") == PROTOCOL_ID
        and manifest.get("result_sha256")
        == canonical_sha256(manifest, digest_key="result_sha256"),
        "cross-view manifest changed",
    )
    _require(
        manifest.get("output", {}).get("archive_file_sha256")
        == file_sha256(archive_path),
        "cross-view archive checksum changed",
    )
    with np.load(archive_path, allow_pickle=False) as stored:
        arrays = {name: np.asarray(stored[name]) for name in stored.files}
    _validate_arrays(arrays)
    return manifest, arrays


def build_crossview_track_supplement(
    measurement_dir: str | Path,
    output_dir: str | Path,
    runtime: AllTrackerPrefixRuntime,
) -> dict[str, Any]:
    """Replay the frozen causal camera plan and retain per-camera tracks."""

    source_root = Path(measurement_dir).resolve()
    output = Path(output_dir).resolve()
    _require(not output.exists(), f"cross-view output exists: {output}")
    source_manifest_path = source_root / SOURCE_MANIFEST_FILENAME
    source_archive_path = source_root / SOURCE_ARCHIVE_FILENAME
    manifest = _load_json(source_manifest_path)
    _validate_source_manifest(manifest)
    _require(
        manifest.get("output", {}).get("measurement_archive_sha256")
        == file_sha256(source_archive_path),
        "source measurement archive checksum changed",
    )
    with np.load(source_archive_path, allow_pickle=False) as stored:
        center_ids = np.asarray(stored["center_ids"], dtype=np.int64)
        selected_cameras = tuple(str(value) for value in stored["selected_cameras"])
        update_frames = tuple(int(value) for value in stored["update_frames"])
    _require(
        list(selected_cameras) == manifest.get("plan", {}).get("selected_cameras")
        and center_ids.tolist() == manifest.get("plan", {}).get("center_ids"),
        "source camera plan changed",
    )
    config = _source_config(manifest["config"])
    _require(runtime.config == config, "AllTracker runtime config differs from source")
    _require(update_frames == config.update_frames, "source update frames changed")

    prediction_record = manifest.get("inputs", {}).get("prediction_archive", {})
    prediction_path = Path(str(prediction_record.get("path"))).resolve()
    _require(
        prediction_record.get("sha256") == file_sha256(prediction_path),
        "source prediction archive checksum changed",
    )
    with np.load(prediction_path, allow_pickle=False) as stored:
        frame_zero = np.asarray(stored["frame_zero_points_m"], dtype=np.float64)
    _require(np.max(center_ids, initial=-1) < len(frame_zero), "centre exceeds prediction")

    intrinsic_record = manifest.get("inputs", {}).get("intrinsics", {})
    extrinsic_record = manifest.get("inputs", {}).get("extrinsics", {})
    intrinsic_path = Path(str(intrinsic_record.get("path"))).resolve()
    extrinsic_path = Path(str(extrinsic_record.get("path"))).resolve()
    _require(
        intrinsic_record.get("sha256") == file_sha256(intrinsic_path)
        and extrinsic_record.get("sha256") == file_sha256(extrinsic_path),
        "source calibration checksum changed",
    )
    processed = intrinsic_path.parent
    intrinsics, extrinsics = _load_calibration(processed)
    cameras, support, projected = frame_zero_camera_support(
        frame_zero,
        processed,
        intrinsics,
        extrinsics,
        depth_tolerance_m=config.frame_zero_depth_tolerance_m,
    )
    camera_index = {camera: index for index, camera in enumerate(cameras)}
    _require(all(camera in camera_index for camera in selected_cameras), "selected camera vanished")

    update_count = len(update_frames)
    view_count = len(selected_cameras)
    center_count = len(center_ids)
    tracks = np.full((update_count, view_count, center_count, 2), np.nan, dtype=np.float32)
    visible = np.zeros((update_count, view_count, center_count), dtype=bool)
    initial_pixels = np.empty((view_count, center_count, 2), dtype=np.float32)
    initial_support = np.zeros((view_count, center_count), dtype=bool)
    tracker_records: list[dict[str, Any]] = []
    for view_index, camera in enumerate(selected_cameras):
        source_index = camera_index[camera]
        initial_pixels[view_index] = np.asarray(projected[camera][center_ids], dtype=np.float32)
        initial_support[view_index] = support[center_ids, source_index]
    for update_index, update_frame in enumerate(update_frames):
        update_record: dict[str, Any] = {"frame": update_frame, "cameras": []}
        for view_index, camera in enumerate(selected_cameras):
            query_mask = initial_support[view_index]
            query_ids = center_ids[query_mask]
            query_pixels = initial_pixels[view_index, query_mask]
            camera_tracks, camera_visible, tracker_record = runtime.track_prefix(
                processed / camera / "undistorted.mp4",
                query_pixels,
                update_frame,
            )
            aligned, aligned_visible = align_camera_tracks(
                center_ids,
                query_ids,
                camera_tracks,
                camera_visible,
            )
            tracks[update_index, view_index] = aligned
            visible[update_index, view_index] = aligned_visible
            update_record["cameras"].append({"camera": camera, **tracker_record})
        tracker_records.append(update_record)

    arrays = {
        "track_pixels_xy": tracks,
        "track_visibility": visible,
        "frame_zero_pixels_xy": initial_pixels,
        "frame_zero_support": initial_support,
        "center_ids": center_ids,
        "selected_cameras": np.asarray(selected_cameras),
        "update_frames": np.asarray(update_frames, dtype=np.int64),
        "center_frame_zero_points_m": frame_zero[center_ids].astype(np.float32),
        "intrinsics": np.stack([np.asarray(intrinsics[camera]) for camera in selected_cameras]),
        "camera_to_world": np.stack([np.asarray(extrinsics[camera]) for camera in selected_cameras]),
    }
    _validate_arrays(arrays)
    output.mkdir(parents=True)
    archive_path = output / ARCHIVE_FILENAME
    np.savez_compressed(archive_path, **arrays)
    payload: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": ARTIFACT_KIND,
        "protocol_id": PROTOCOL_ID,
        "case": manifest["case"],
        "object_id": manifest["object_id"],
        "episode_id": manifest["episode_id"],
        "episode_key": manifest["episode_key"],
        "source_measurement": {
            "protocol_id": manifest["protocol_id"],
            "manifest_file_sha256": file_sha256(source_manifest_path),
            "manifest_result_sha256": manifest["result_sha256"],
            "archive_file_sha256": file_sha256(source_archive_path),
        },
        "config": asdict(config),
        "selected_cameras": list(selected_cameras),
        "center_ids": center_ids.tolist(),
        "update_frames": list(update_frames),
        "tracker": {
            "runtime_source_sha256": runtime.source_sha256,
            "checkpoint_sha256": runtime.checkpoint_sha256,
            "device": runtime.device_name,
            "updates": tracker_records,
        },
        "inputs_sha256": {
            "prediction_archive": file_sha256(prediction_path),
            "intrinsics": file_sha256(intrinsic_path),
            "extrinsics": file_sha256(extrinsic_path),
        },
        "output": {
            "archive": str(archive_path),
            "archive_file_sha256": file_sha256(archive_path),
            "visible_observation_count": int(np.sum(visible)),
        },
        "information_boundary": {
            "source_measurement_preexisting_and_unchanged": True,
            "video_prefix_rule": "update u reads exactly frames [0, u]",
            "maximum_video_frame_read_by_update": list(update_frames),
            "frame_zero_hdf5_indices_read": [0],
            "target_data_read": False,
            "outcome_manifest_read": False,
        },
        "claim_boundary": (
            "Post-open method-development observation supplement. It cannot "
            "modify or rescue a frozen prospective result."
        ),
    }
    payload["result_sha256"] = canonical_sha256(payload, digest_key="result_sha256")
    (output / MANIFEST_FILENAME).write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return payload


__all__ = [
    "ARCHIVE_FILENAME",
    "ARTIFACT_KIND",
    "MANIFEST_FILENAME",
    "PROTOCOL_ID",
    "align_camera_tracks",
    "build_crossview_track_supplement",
    "load_crossview_track_supplement",
    "load_source_raw_camera_config",
]
