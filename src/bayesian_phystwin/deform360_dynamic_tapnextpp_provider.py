"""Sealed input and output helpers for the dynamic TAPNext++ provider."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .deform360_dynamic_query import DynamicQuerySchedule
from .deform360_dynamic_tapnextpp_assimilation import (
    CANDIDATE_ARM,
    PERSISTENCE_ARM,
    SELECTED_BACKBONE_ARM,
)
from .deform360_dynamic_tapnextpp_source_window import FROZEN_CAMERA_PANEL
from .deform360_object_exclusion import file_sha256
from .observation_belief import array_sha256
from .tapnextpp_dynamic_multiview import (
    PROTOCOL_ID,
    DynamicMultiviewResult,
    dynamic_multiview_result_sha256,
)
from .tapnextpp_dynamic_runtime import (
    DynamicBirthAssociations,
    DynamicTAPNextPPRuntimeResult,
)

CAUSAL_FRAME_STOP_EXCLUSIVE = 58
QUERY_SCHEDULE_FILENAME = "dynamic_tapnextpp_query_schedule.json"
ASSIMILATION_REPORT_FILENAME = "dynamic_tapnextpp_assimilation.json"
ASSIMILATION_ARCHIVE_FILENAME = "dynamic_tapnextpp_assimilation.npz"
PREDICTION_INPUT_FILENAME = "dynamic_tapnextpp_prediction_input.npz"
RUNTIME_REPORT_FILENAME = "dynamic_tapnextpp_runtime.json"
PROVIDER_ARCHIVE_FILENAME = "dynamic_tapnextpp_provider.npz"
PROVIDER_REPORT_FILENAME = "dynamic_tapnextpp_provider.json"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _canonical_sha256(
    payload: Mapping[str, Any],
    *,
    digest_key: str,
) -> str:
    canonical = dict(payload)
    canonical.pop(digest_key, None)
    encoded = json.dumps(
        canonical,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(
            dict(payload),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _valid_digest(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


@dataclass(frozen=True)
class CausalCameraInputs:
    """Eight selected camera streams truncated at a declared causal update."""

    camera_indices: np.ndarray
    camera_names: tuple[str, ...]
    rgbs: np.ndarray
    depths_m: np.ndarray
    object_masks: np.ndarray
    intrinsics: np.ndarray
    camera_to_world: np.ndarray
    provenance: Mapping[str, Any]

    def __post_init__(self) -> None:
        indices = np.ascontiguousarray(
            np.asarray(self.camera_indices, dtype=np.int64)
        )
        rgbs = np.ascontiguousarray(np.asarray(self.rgbs, dtype=np.uint8))
        depths = np.ascontiguousarray(
            np.asarray(self.depths_m, dtype=np.float32)
        )
        masks = np.ascontiguousarray(np.asarray(self.object_masks, dtype=bool))
        intrinsics = np.ascontiguousarray(
            np.asarray(self.intrinsics, dtype=np.float64)
        )
        poses = np.ascontiguousarray(
            np.asarray(self.camera_to_world, dtype=np.float64)
        )
        camera_count = len(indices)
        _require(
            camera_count == len(self.camera_names) == 8,
            "causal camera panel must contain eight cameras",
        )
        _require(
            len(np.unique(indices)) == camera_count and np.all(indices >= 0),
            "causal camera identities are invalid",
        )
        _require(
            rgbs.ndim == 5
            and rgbs.shape[0] == camera_count
            and 1 <= rgbs.shape[1] <= CAUSAL_FRAME_STOP_EXCLUSIVE
            and rgbs.shape[-1] == 3,
            "causal RGB array must have shape (8, T<=58, H, W, 3)",
        )
        _require(
            depths.shape == masks.shape == rgbs.shape[:-1],
            "causal RGB, depth, and mask shapes differ",
        )
        _require(
            intrinsics.shape == (camera_count, 3, 3)
            and poses.shape == (camera_count, 4, 4)
            and np.all(np.isfinite(intrinsics))
            and np.all(np.isfinite(poses)),
            "causal camera calibration is invalid",
        )
        _require(
            np.all(np.isfinite(depths)) and np.all(depths >= 0.0),
            "causal depth is invalid",
        )
        _require(
            self.provenance.get("maximum_frame_read") == rgbs.shape[1] - 1,
            "causal input provenance differs from its truncated update",
        )
        for values in (indices, rgbs, depths, masks, intrinsics, poses):
            values.setflags(write=False)
        object.__setattr__(self, "camera_indices", indices)
        object.__setattr__(self, "rgbs", rgbs)
        object.__setattr__(self, "depths_m", depths)
        object.__setattr__(self, "object_masks", masks)
        object.__setattr__(self, "intrinsics", intrinsics)
        object.__setattr__(self, "camera_to_world", poses)
        object.__setattr__(
            self,
            "provenance",
            json.loads(json.dumps(dict(self.provenance), allow_nan=False)),
        )


def _load_calibration_dict(path: Path) -> dict[str, np.ndarray]:
    stored = np.load(path, allow_pickle=True)
    _require(stored.shape == (), f"calibration archive is not a dictionary: {path}")
    mapping = stored.item()
    _require(isinstance(mapping, dict), f"calibration archive is invalid: {path}")
    return {
        str(name): np.asarray(value)
        for name, value in mapping.items()
    }


def load_camera_geometry(
    processed_episode_dir: str | Path,
) -> tuple[
    tuple[str, ...],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    dict[str, str],
]:
    """Load target-free calibration and image shapes for all frozen cameras."""

    root = Path(processed_episode_dir).resolve()
    intrinsics_path = root / "undistorted_intrinsics.npy"
    extrinsics_path = root / "extrinsics.npy"
    intrinsics_by_name = _load_calibration_dict(intrinsics_path)
    poses_by_name = _load_calibration_dict(extrinsics_path)
    intrinsics: list[np.ndarray] = []
    poses: list[np.ndarray] = []
    image_shapes: list[tuple[int, int]] = []
    import h5py

    for camera in FROZEN_CAMERA_PANEL:
        _require(
            camera in intrinsics_by_name and camera in poses_by_name,
            f"frozen camera calibration is missing: {camera}",
        )
        intrinsic = np.asarray(intrinsics_by_name[camera], dtype=np.float64)
        pose = np.asarray(poses_by_name[camera], dtype=np.float64)
        _require(
            intrinsic.shape == (3, 3)
            and pose.shape == (4, 4)
            and np.all(np.isfinite(intrinsic))
            and np.all(np.isfinite(pose)),
            f"frozen camera calibration is invalid: {camera}",
        )
        with h5py.File(root / camera / "rendered_depth.h5", "r") as stream:
            _require(
                "data" in stream
                and stream["data"].ndim == 3
                and len(stream["data"]) >= CAUSAL_FRAME_STOP_EXCLUSIVE,
                f"rendered depth contract is invalid: {camera}",
            )
            image_shapes.append(tuple(map(int, stream["data"].shape[-2:])))
        intrinsics.append(intrinsic)
        poses.append(pose)
    return (
        FROZEN_CAMERA_PANEL,
        np.stack(intrinsics),
        np.stack(poses),
        np.asarray(image_shapes, dtype=np.int64),
        {
            "intrinsics": file_sha256(intrinsics_path),
            "extrinsics": file_sha256(extrinsics_path),
        },
    )


def _decode_rgb_prefix(path: Path, frame_count: int) -> np.ndarray:
    import cv2

    capture = cv2.VideoCapture(str(path))
    frames: list[np.ndarray] = []
    try:
        for frame_index in range(frame_count):
            ok, bgr = capture.read()
            observed = int(capture.get(cv2.CAP_PROP_POS_FRAMES)) - 1
            _require(
                bool(ok) and observed == frame_index,
                f"cannot decode exact RGB frame {frame_index}: {path}",
            )
            frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    finally:
        capture.release()
    return np.stack(frames).astype(np.uint8, copy=False)


def _read_h5_prefix(path: Path, frame_count: int) -> np.ndarray:
    import h5py

    with h5py.File(path, "r") as stream:
        _require(
            "data" in stream
            and stream["data"].ndim == 3
            and len(stream["data"]) >= frame_count,
            f"invalid causal HDF5 archive: {path}",
        )
        return np.asarray(stream["data"][:frame_count])


def load_selected_causal_inputs(
    processed_episode_dir: str | Path,
    camera_indices: Sequence[int],
    *,
    depth_scale_to_m: float = 0.001,
) -> CausalCameraInputs:
    """Decode only frames 0..57 from the frozen selected camera panel."""

    root = Path(processed_episode_dir).resolve()
    names, all_intrinsics, all_poses, _, calibration_hashes = (
        load_camera_geometry(root)
    )
    indices = np.asarray(camera_indices, dtype=np.int64)
    _require(
        indices.shape == (8,)
        and len(np.unique(indices)) == 8
        and np.all((indices >= 0) & (indices < len(names))),
        "selected camera indices are invalid",
    )
    _require(
        np.isfinite(depth_scale_to_m) and depth_scale_to_m > 0.0,
        "depth scale must be finite and positive",
    )
    rgbs: list[np.ndarray] = []
    depths: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    camera_provenance: dict[str, Any] = {}
    selected_names = tuple(names[int(index)] for index in indices)
    for camera in selected_names:
        directory = root / camera
        video_path = directory / "undistorted.mp4"
        depth_path = directory / "rendered_depth.h5"
        mask_path = directory / "mask_refined.h5"
        rgb = _decode_rgb_prefix(video_path, CAUSAL_FRAME_STOP_EXCLUSIVE)
        encoded_depth = _read_h5_prefix(
            depth_path,
            CAUSAL_FRAME_STOP_EXCLUSIVE,
        )
        mask = _read_h5_prefix(mask_path, CAUSAL_FRAME_STOP_EXCLUSIVE).astype(
            bool,
            copy=False,
        )
        depth = encoded_depth.astype(np.float32) * depth_scale_to_m
        _require(
            rgb.shape[:-1] == depth.shape == mask.shape,
            f"causal camera stream shapes differ: {camera}",
        )
        rgbs.append(rgb)
        depths.append(depth)
        masks.append(mask)
        camera_provenance[camera] = {
            "video_file_sha256": file_sha256(video_path),
            "depth_file_sha256": file_sha256(depth_path),
            "mask_file_sha256": file_sha256(mask_path),
            "decoded_rgb_sha256": array_sha256(rgb),
            "decoded_depth_m_sha256": array_sha256(depth),
            "decoded_mask_sha256": array_sha256(mask),
            "frames_read": list(range(CAUSAL_FRAME_STOP_EXCLUSIVE)),
        }
    return CausalCameraInputs(
        camera_indices=indices,
        camera_names=selected_names,
        rgbs=np.stack(rgbs),
        depths_m=np.stack(depths),
        object_masks=np.stack(masks),
        intrinsics=all_intrinsics[indices],
        camera_to_world=all_poses[indices],
        provenance={
            "calibration_file_sha256": calibration_hashes,
            "selected_camera_indices": indices.tolist(),
            "selected_camera_names": list(selected_names),
            "maximum_frame_read": CAUSAL_FRAME_STOP_EXCLUSIVE - 1,
            "future_frame_read": False,
            "cameras": camera_provenance,
        },
    )


def write_query_schedule_artifact(
    output_path: str | Path,
    schedule: DynamicQuerySchedule,
    *,
    case_hash: str,
    input_sha256: Mapping[str, str],
) -> dict[str, Any]:
    """Bind a target-free query schedule to one hash-only case identity."""

    _require(_valid_digest(case_hash), "query-schedule case hash is invalid")
    _require(
        bool(input_sha256)
        and all(_valid_digest(value) for value in input_sha256.values()),
        "query-schedule input checksum is invalid",
    )
    payload = schedule.descriptor()
    payload.update(
        {
            "case_hash": case_hash,
            "schedule_sha256": schedule.artifact_sha256,
            "inputs_sha256": dict(sorted(input_sha256.items())),
        }
    )
    payload["result_sha256"] = _canonical_sha256(
        payload,
        digest_key="result_sha256",
    )
    _write_json_atomic(Path(output_path), payload)
    return validate_query_schedule_artifact(output_path)


def validate_query_schedule_artifact(
    artifact: Mapping[str, Any] | str | Path,
) -> dict[str, Any]:
    payload = (
        json.loads(Path(artifact).read_text(encoding="utf-8"))
        if isinstance(artifact, (str, Path))
        else dict(artifact)
    )
    _require(
        payload.get("artifact_kind")
        == "Deform360DynamicTAPNextPPQuerySchedule"
        and payload.get("protocol_id") == PROTOCOL_ID,
        "query schedule belongs to another protocol",
    )
    _require(
        _valid_digest(payload.get("case_hash"))
        and _valid_digest(payload.get("schedule_sha256"))
        and payload.get("result_sha256")
        == _canonical_sha256(payload, digest_key="result_sha256"),
        "query schedule checksum or identity is invalid",
    )
    boundary = payload.get("information_boundary", {})
    _require(
        boundary.get("maximum_physical_frame_read") == 57
        and boundary.get("observed_object_trajectory_read") is False
        and boundary.get("target_metric_read") is False
        and boundary.get("future_frame_after_update_used_for_that_update")
        is False,
        "query schedule crossed its information boundary",
    )
    return payload


def write_assimilation_artifacts(
    output_dir: str | Path,
    report: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    *,
    case_hash: str,
    measurement_entity_ids: np.ndarray,
    update_frames: np.ndarray,
    input_sha256: Mapping[str, str],
) -> tuple[Path, Path, dict[str, Any]]:
    """Write the full candidate family and the narrow prediction-seal input."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    _require(_valid_digest(case_hash), "assimilation case hash is invalid")
    normalized_arrays = {
        name: np.ascontiguousarray(np.asarray(value))
        for name, value in arrays.items()
    }
    required = {
        SELECTED_BACKBONE_ARM,
        CANDIDATE_ARM,
        PERSISTENCE_ARM,
        "candidate_correction_variance_m2",
    }
    _require(required <= set(normalized_arrays), "assimilation arrays are incomplete")
    baseline = normalized_arrays[SELECTED_BACKBONE_ARM]
    candidate = normalized_arrays[CANDIDATE_ARM]
    persistence = normalized_arrays[PERSISTENCE_ARM]
    _require(
        baseline.shape == candidate.shape == persistence.shape
        and baseline.ndim == 3
        and baseline.shape[0] == 76
        and baseline.shape[2] == 3,
        "assimilation trajectories are invalid",
    )
    measurements = np.unique(
        np.asarray(measurement_entity_ids, dtype=np.int64)
    )
    _require(
        len(measurements) > 0
        and np.all((measurements >= 0) & (measurements < baseline.shape[1])),
        "measurement identities are invalid",
    )
    hidden = np.setdiff1d(
        np.arange(baseline.shape[1], dtype=np.int64),
        measurements,
        assume_unique=True,
    )
    _require(len(hidden) > 0, "no disjoint hidden identities remain")

    assimilation_archive = output / ASSIMILATION_ARCHIVE_FILENAME
    temporary_archive = output / (ASSIMILATION_ARCHIVE_FILENAME + ".tmp.npz")
    np.savez_compressed(temporary_archive, **normalized_arrays)
    temporary_archive.replace(assimilation_archive)

    prediction_input = output / PREDICTION_INPUT_FILENAME
    temporary_prediction = output / (PREDICTION_INPUT_FILENAME + ".tmp.npz")
    np.savez_compressed(
        temporary_prediction,
        baseline_prediction_m=baseline,
        candidate_prediction_m=candidate,
        persistence_prediction_m=persistence,
        measurement_entity_ids=measurements,
        hidden_entity_ids=hidden,
        update_frames=np.asarray(update_frames, dtype=np.int64),
    )
    temporary_prediction.replace(prediction_input)

    payload = json.loads(json.dumps(dict(report), allow_nan=False))
    payload.update(
        {
            "case_hash": case_hash,
            "inputs_sha256": dict(sorted(input_sha256.items())),
            "assimilation_archive": {
                "filename": ASSIMILATION_ARCHIVE_FILENAME,
                "file_sha256": file_sha256(assimilation_archive),
                "array_sha256": {
                    name: array_sha256(values)
                    for name, values in sorted(normalized_arrays.items())
                },
            },
            "prediction_input": {
                "filename": PREDICTION_INPUT_FILENAME,
                "file_sha256": file_sha256(prediction_input),
                "measurement_identity_count": int(len(measurements)),
                "hidden_identity_count": int(len(hidden)),
            },
        }
    )
    payload["result_sha256"] = _canonical_sha256(
        payload,
        digest_key="result_sha256",
    )
    report_path = output / ASSIMILATION_REPORT_FILENAME
    _write_json_atomic(report_path, payload)
    return prediction_input, assimilation_archive, payload


def write_provider_artifacts(
    output_dir: str | Path,
    result: DynamicMultiviewResult,
    runtime: DynamicTAPNextPPRuntimeResult,
    associations: DynamicBirthAssociations,
    schedule: DynamicQuerySchedule,
    *,
    case_hash: str,
    input_sha256: Mapping[str, str],
    runtime_provenance: Mapping[str, Any],
) -> tuple[Path, Path, dict[str, Any]]:
    """Persist every provider quantity needed for source competence scoring."""

    _require(_valid_digest(case_hash), "provider case hash is invalid")
    _require(
        bool(input_sha256)
        and all(_valid_digest(value) for value in input_sha256.values()),
        "provider input checksum is invalid",
    )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    arrays = {
        "tracks_xy": runtime.tracks_xy,
        "visibility_probability": runtime.visibility_probability,
        "runtime_active": runtime.active,
        "query_points_world_m": associations.query_points_world_m,
        "association_query_points_xy": associations.query_points_xy,
        "association_valid": associations.valid,
        "association_probability_per_camera": (
            associations.association_probability
        ),
        "association_entropy_per_camera": associations.association_entropy,
        "assignment_pixel_covariance_px2": (
            associations.candidate_pixel_covariance_px2
        ),
        "association_candidate_count": associations.candidate_count,
        "camera_indices": associations.camera_indices,
        "entity_ids": schedule.entity_ids,
        "birth_frames": schedule.birth_frames,
        "update_frames": schedule.update_frames,
        "trajectory_world_m": result.trajectory_world_m,
        "proposal_available": result.proposal_available,
        "accepted_support": result.accepted_support,
        "prior_reliability": result.prior_reliability,
        "association_probability": result.association_probability,
        "local_covariance_m2": result.local_covariance_m2,
        "naive_independent_covariance_m2": (
            result.naive_independent_covariance_m2
        ),
        "assignment_mixture_spread_m2": (
            result.assignment_mixture_spread_m2
        ),
        "independent_support_count": result.independent_support_count,
        "raw_support_count": result.raw_support_count,
        "reprojection_rmse_px": result.reprojection_rmse_px,
        "depth_residual_rmse_m": result.depth_residual_rmse_m,
        "inlier_camera_mask": result.inlier_camera_mask,
        "camera_cluster_ids": result.camera_cluster_ids,
    }
    normalized = {
        name: np.ascontiguousarray(np.asarray(values))
        for name, values in arrays.items()
    }
    archive = output / PROVIDER_ARCHIVE_FILENAME
    temporary = output / (PROVIDER_ARCHIVE_FILENAME + ".tmp.npz")
    np.savez_compressed(temporary, **normalized)
    temporary.replace(archive)

    births = np.asarray(schedule.birth_frames, dtype=np.int64)
    updates = np.asarray(schedule.update_frames, dtype=np.int64)
    active_rows = (
        np.arange(result.trajectory_world_m.shape[0])[:, None]
        >= births[None]
    ) & (
        np.arange(result.trajectory_world_m.shape[0])[:, None]
        <= updates[None]
    )
    endpoint_supported = (
        result.accepted_support[births, np.arange(len(births))]
        & result.accepted_support[updates, np.arange(len(updates))]
    )
    report: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": "Deform360DynamicTAPNextPPProvider",
        "protocol_id": PROTOCOL_ID,
        "case_hash": case_hash,
        "inputs_sha256": dict(sorted(input_sha256.items())),
        "provider_result_sha256": dynamic_multiview_result_sha256(result),
        "provider_archive": {
            "filename": PROVIDER_ARCHIVE_FILENAME,
            "file_sha256": file_sha256(archive),
            "array_sha256": {
                name: array_sha256(values)
                for name, values in sorted(normalized.items())
            },
        },
        "support": {
            "scheduled_identity_count": int(len(schedule.entity_ids)),
            "active_identity_frame_count": int(np.sum(active_rows)),
            "claim_bearing_identity_frame_count": int(
                np.sum(result.accepted_support & active_rows)
            ),
            "claim_bearing_fraction": float(
                np.sum(result.accepted_support & active_rows)
                / max(1, np.sum(active_rows))
            ),
            "birth_and_update_supported_count": int(
                np.sum(endpoint_supported)
            ),
            "birth_and_update_supported_fraction": float(
                np.mean(endpoint_supported)
            ),
        },
        "runtime": {
            "rollout_count": runtime.rollout_count,
            "model_frame_count": runtime.model_frame_count,
            "elapsed_seconds": runtime.elapsed_seconds,
            **dict(runtime_provenance),
        },
        "configuration": {
            "multiview": asdict(result.config),
            "camera_names": list(associations.camera_names),
        },
        "information_boundary": {
            "maximum_rgb_depth_mask_frame_read": 57,
            "future_frame_read": False,
            "future_object_geometry_read": False,
            "target_metric_read": False,
        },
    }
    report["result_sha256"] = _canonical_sha256(
        report,
        digest_key="result_sha256",
    )
    report_path = output / PROVIDER_REPORT_FILENAME
    _write_json_atomic(report_path, report)
    return archive, report_path, report


__all__ = [
    "ASSIMILATION_ARCHIVE_FILENAME",
    "ASSIMILATION_REPORT_FILENAME",
    "CAUSAL_FRAME_STOP_EXCLUSIVE",
    "CausalCameraInputs",
    "PREDICTION_INPUT_FILENAME",
    "PROVIDER_ARCHIVE_FILENAME",
    "PROVIDER_REPORT_FILENAME",
    "QUERY_SCHEDULE_FILENAME",
    "RUNTIME_REPORT_FILENAME",
    "load_camera_geometry",
    "load_selected_causal_inputs",
    "validate_query_schedule_artifact",
    "write_assimilation_artifacts",
    "write_provider_artifacts",
    "write_query_schedule_artifact",
]
