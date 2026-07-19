"""Outcome-free uncertainty sidecars for causal Deform360 camera measurements.

The raw AllTracker measurement archive intentionally stores only accepted 3-D
points.  This module replays the exact hashed RGB prefixes and estimates a 3-D
measurement covariance without opening ``target_data.pkl`` or ``outcome.json``.
It combines local projection-Jacobian covariance with leave-one-inlier-camera-
out jackknife covariance.  The original measurement archive is never modified.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .deform360_raw_camera_observation import (
    MANIFEST_FILENAME,
    MEASUREMENT_FILENAME,
    AllTrackerPrefixRuntime,
    _canonical_sha256,
    _linear_triangulation,
    _load_calibration,
    _load_measurement_artifact,
    _projection_matrix,
    _reproject,
    _resolve_prediction_archive,
    _sha256,
    _validate_prediction_seal,
    expected_open_case_names,
    project_world_points,
)

UNCERTAINTY_PROTOCOL_ID = (
    "deform360-raw-camera-alltracker-jackknife-uncertainty-v1-development"
)
UNCERTAINTY_ARCHIVE_FILENAME = "measurement_uncertainty.npz"
UNCERTAINTY_MANIFEST_FILENAME = "measurement_uncertainty_manifest.json"


@dataclass(frozen=True)
class RawCameraUncertaintyConfig:
    """Fixed causal covariance construction settings."""

    pixel_noise_floor_px: float = 0.5
    maximum_information_condition_number: float = 1.0e8
    replay_position_tolerance_m: float = 5.0e-4

    def validate(self) -> None:
        if self.pixel_noise_floor_px <= 0.0:
            raise ValueError("pixel noise floor must be positive")
        if self.maximum_information_condition_number <= 1.0:
            raise ValueError("condition-number ceiling must exceed one")
        if self.replay_position_tolerance_m <= 0.0:
            raise ValueError("replay tolerance must be positive")


def projection_jacobian(
    point_m: np.ndarray, projection_matrix: np.ndarray
) -> np.ndarray:
    """Return the analytic 2-by-3 perspective-projection Jacobian."""

    point = np.append(np.asarray(point_m, dtype=float), 1.0)
    matrix = np.asarray(projection_matrix, dtype=float)
    if matrix.shape != (3, 4):
        raise ValueError("projection matrix must have shape (3, 4)")
    projected = matrix @ point
    denominator = float(projected[2])
    if denominator <= 1e-12:
        raise ValueError("point is behind or on the camera plane")
    jacobian = np.stack(
        (
            (matrix[0, :3] * denominator - projected[0] * matrix[2, :3])
            / denominator**2,
            (matrix[1, :3] * denominator - projected[1] * matrix[2, :3])
            / denominator**2,
        )
    )
    if not np.all(np.isfinite(jacobian)):
        raise ValueError("projection Jacobian is non-finite")
    return jacobian


def jacobian_measurement_covariance(
    point_m: np.ndarray,
    projection_matrices: Sequence[np.ndarray],
    pixel_sigma: float,
    *,
    maximum_condition_number: float,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    """Linearized isotropic-pixel covariance for one multiview 3-D point."""

    if len(projection_matrices) < 2:
        return None, {"decision": "insufficient_views"}
    try:
        jacobian = np.concatenate(
            [projection_jacobian(point_m, matrix) for matrix in projection_matrices],
            axis=0,
        )
    except ValueError:
        return None, {"decision": "invalid_projection"}
    information = jacobian.T @ jacobian
    eigenvalues = np.linalg.eigvalsh(information)
    if eigenvalues[0] <= 0.0 or not np.all(np.isfinite(eigenvalues)):
        return None, {"decision": "singular_information"}
    condition = float(eigenvalues[-1] / eigenvalues[0])
    diagnostic = {
        "decision": "accepted",
        "information_condition_number": condition,
        "information_eigenvalues_px2_per_m2": eigenvalues.tolist(),
        "pixel_sigma": float(pixel_sigma),
    }
    if condition > maximum_condition_number:
        diagnostic["decision"] = "condition_number_failure"
        return None, diagnostic
    covariance = float(pixel_sigma) ** 2 * np.linalg.inv(information)
    covariance = 0.5 * (covariance + covariance.T)
    if not np.all(np.isfinite(covariance)):
        diagnostic["decision"] = "nonfinite_covariance"
        return None, diagnostic
    return covariance, diagnostic


def leave_one_camera_out_covariance(
    observations: Mapping[str, np.ndarray],
    projection_matrices: Mapping[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Return jackknife covariance and valid leave-one-camera-out estimates."""

    ordered = tuple(
        (camera, np.asarray(observations[camera], dtype=float))
        for camera in sorted(observations)
    )
    estimates: list[np.ndarray] = []
    if len(ordered) >= 3:
        for held_out_index in range(len(ordered)):
            subset = tuple(
                observation
                for index, observation in enumerate(ordered)
                if index != held_out_index
            )
            try:
                estimate = _linear_triangulation(subset, projection_matrices)
            except (ValueError, np.linalg.LinAlgError):
                continue
            positive_depth = all(
                _reproject(estimate, projection_matrices[camera])[1] > 0.0
                for camera, _ in subset
            )
            if positive_depth and np.all(np.isfinite(estimate)):
                estimates.append(estimate)
    samples = np.asarray(estimates, dtype=float).reshape(-1, 3)
    if len(samples) < 2:
        return np.zeros((3, 3), dtype=float), samples
    centered = samples - np.mean(samples, axis=0, keepdims=True)
    covariance = (len(samples) - 1.0) / len(samples) * centered.T @ centered
    return 0.5 * (covariance + covariance.T), samples


def _pixel_sigma_from_median_reprojection(
    median_reprojection_px: float,
    floor_px: float,
) -> float:
    # The 2-D norm of isotropic Gaussian residuals is Rayleigh-distributed.
    rayleigh_median_scale = np.sqrt(2.0 * np.log(2.0))
    return max(float(floor_px), float(median_reprojection_px) / rayleigh_median_scale)


def _load_frame_zero_and_calibration(
    case_dir: Path,
    processed_dir: Path,
    seal: Mapping[str, Any],
    measurement_manifest: Mapping[str, Any],
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray]]:
    archive_path = _resolve_prediction_archive(case_dir, seal)
    expected_archive_hash = measurement_manifest["inputs"]["prediction_archive"][
        "sha256"
    ]
    if _sha256(archive_path) != expected_archive_hash:
        raise ValueError("sealed prediction archive checksum changed")
    with np.load(archive_path, allow_pickle=False) as stored:
        frame_zero = np.asarray(stored["frame_zero_points_m"], dtype=float).copy()
    intrinsics_path = processed_dir / "undistorted_intrinsics.npy"
    extrinsics_path = processed_dir / "extrinsics.npy"
    if (
        _sha256(intrinsics_path)
        != measurement_manifest["inputs"]["intrinsics"]["sha256"]
    ):
        raise ValueError("intrinsics checksum changed")
    if (
        _sha256(extrinsics_path)
        != measurement_manifest["inputs"]["extrinsics"]["sha256"]
    ):
        raise ValueError("extrinsics checksum changed")
    intrinsics, extrinsics = _load_calibration(processed_dir)
    return frame_zero, intrinsics, extrinsics


def build_raw_camera_uncertainty_case(
    panel_case_dir: str | Path,
    processed_episode_dir: str | Path,
    measurement_dir: str | Path,
    output_dir: str | Path,
    runtime: AllTrackerPrefixRuntime,
    *,
    config: RawCameraUncertaintyConfig | None = None,
) -> dict[str, Any]:
    """Build one causal uncertainty sidecar without opening the outcome."""

    cfg = config or RawCameraUncertaintyConfig()
    cfg.validate()
    case_dir = Path(panel_case_dir).resolve()
    processed = Path(processed_episode_dir).resolve()
    measurement_path = Path(measurement_dir).resolve()
    output = Path(output_dir).resolve()
    if case_dir.name not in expected_open_case_names():
        raise ValueError("case is outside the explicit outcome-open panel")
    if output.exists():
        raise FileExistsError(output)
    seal_path = case_dir / "prediction_seal.json"
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    _validate_prediction_seal(seal)
    manifest, measurement_arrays = _load_measurement_artifact(
        case_dir,
        measurement_path,
        seal,
    )
    frame_zero, intrinsics, extrinsics = _load_frame_zero_and_calibration(
        case_dir,
        processed,
        seal,
        manifest,
    )
    measurement = np.asarray(measurement_arrays["measurement_m"], dtype=float)
    measurement_validity = np.asarray(
        measurement_arrays["measurement_validity"], dtype=bool
    )
    time_count, point_count = measurement.shape[:2]
    covariance = np.full((time_count, point_count, 3, 3), np.nan, dtype=np.float32)
    covariance_valid = np.zeros((time_count, point_count), dtype=bool)
    jacobian_covariance = np.full_like(covariance, np.nan)
    jackknife_covariance = np.full_like(covariance, np.nan)
    principal_std = np.full((time_count, point_count, 3), np.nan, dtype=np.float32)
    rms_std = np.full((time_count, point_count), np.nan, dtype=np.float32)
    maximum_std = np.full_like(rms_std, np.nan)
    condition_number = np.full_like(rms_std, np.nan)
    replay_error = np.full_like(rms_std, np.nan)
    loo_sample_count = np.zeros((time_count, point_count), dtype=np.int16)
    pixel_sigma = np.full_like(rms_std, np.nan)

    selected_cameras = tuple(manifest["plan"]["selected_cameras"])
    projection_matrices = {
        camera: _projection_matrix(intrinsics[camera], extrinsics[camera])
        for camera in selected_cameras
    }
    update_records: list[dict[str, Any]] = []
    for update in manifest["updates"]:
        frame = int(update["frame"])
        source_tracker = {str(record["camera"]): record for record in update["tracker"]}
        tracks_by_camera: dict[str, dict[int, np.ndarray]] = {}
        replay_tracker: list[dict[str, Any]] = []
        for camera in selected_cameras:
            source_record = source_tracker[camera]
            query_ids = np.asarray(source_record["query_ids"], dtype=np.int64)
            query_pixels = project_world_points(
                frame_zero[query_ids],
                intrinsics[camera],
                extrinsics[camera],
            )[0]
            tracks, visible, tracker_record = runtime.track_prefix(
                processed / camera / "undistorted.mp4",
                query_pixels,
                frame,
            )
            if (
                tracker_record["decoded_rgb_prefix_sha256"]
                != source_record["decoded_rgb_prefix_sha256"]
            ):
                raise ValueError("replayed RGB prefix checksum differs")
            tracks_by_camera[camera] = {
                int(point_id): tracks[index]
                for index, point_id in enumerate(query_ids)
                if visible[index]
            }
            tracker_record.update(
                {
                    "camera": camera,
                    "query_ids": query_ids.tolist(),
                    "source_prefix_sha256_matched": True,
                }
            )
            replay_tracker.append(tracker_record)

        center_records: list[dict[str, Any]] = []
        for source_center in update["centers"]:
            center_id = int(source_center["center_id"])
            record: dict[str, Any] = {
                "center_id": center_id,
                "source_measurement_accepted": bool(source_center["accepted"]),
                "covariance_valid": False,
                "decision": "source_measurement_rejected",
            }
            if (
                not source_center["accepted"]
                or not measurement_validity[frame, center_id]
            ):
                center_records.append(record)
                continue
            expected_inliers = tuple(source_center["inlier_cameras"])
            observations = {
                camera: tracks_by_camera[camera][center_id]
                for camera in expected_inliers
                if center_id in tracks_by_camera[camera]
            }
            record["expected_inlier_view_count"] = len(expected_inliers)
            record["replayed_inlier_view_count"] = len(observations)
            if tuple(sorted(observations)) != tuple(sorted(expected_inliers)):
                record["decision"] = "replay_visibility_mismatch"
                center_records.append(record)
                continue
            observation_sequence = tuple(sorted(observations.items()))
            try:
                replayed_point = _linear_triangulation(
                    observation_sequence,
                    projection_matrices,
                )
            except (ValueError, np.linalg.LinAlgError):
                record["decision"] = "replay_triangulation_failure"
                center_records.append(record)
                continue
            point = measurement[frame, center_id]
            position_difference = float(np.linalg.norm(replayed_point - point))
            replay_error[frame, center_id] = position_difference
            record["replay_position_difference_m"] = position_difference
            if position_difference > cfg.replay_position_tolerance_m:
                record["decision"] = "replay_position_mismatch"
                center_records.append(record)
                continue
            sigma = _pixel_sigma_from_median_reprojection(
                float(source_center["median_reprojection_error_px"]),
                cfg.pixel_noise_floor_px,
            )
            geometric_covariance, geometric_diagnostic = (
                jacobian_measurement_covariance(
                    point,
                    [projection_matrices[camera] for camera in sorted(observations)],
                    sigma,
                    maximum_condition_number=(cfg.maximum_information_condition_number),
                )
            )
            record["jacobian"] = geometric_diagnostic
            if geometric_covariance is None:
                record["decision"] = geometric_diagnostic["decision"]
                center_records.append(record)
                continue
            empirical_covariance, loo_estimates = leave_one_camera_out_covariance(
                observations,
                projection_matrices,
            )
            combined = geometric_covariance + empirical_covariance
            combined = 0.5 * (combined + combined.T)
            eigenvalues = np.linalg.eigvalsh(combined)
            if eigenvalues[0] <= 0.0 or not np.all(np.isfinite(eigenvalues)):
                record["decision"] = "combined_covariance_failure"
                center_records.append(record)
                continue
            standard_deviations = np.sqrt(eigenvalues)
            covariance[frame, center_id] = combined
            jacobian_covariance[frame, center_id] = geometric_covariance
            jackknife_covariance[frame, center_id] = empirical_covariance
            covariance_valid[frame, center_id] = True
            principal_std[frame, center_id] = standard_deviations
            rms_std[frame, center_id] = float(np.sqrt(np.trace(combined) / 3.0))
            maximum_std[frame, center_id] = float(standard_deviations[-1])
            condition_number[frame, center_id] = float(
                geometric_diagnostic["information_condition_number"]
            )
            loo_sample_count[frame, center_id] = len(loo_estimates)
            pixel_sigma[frame, center_id] = sigma
            record.update(
                {
                    "covariance_valid": True,
                    "decision": "accepted",
                    "pixel_sigma": sigma,
                    "leave_one_out_sample_count": len(loo_estimates),
                    "principal_standard_deviation_m": standard_deviations.tolist(),
                    "rms_standard_deviation_m": float(
                        np.sqrt(np.trace(combined) / 3.0)
                    ),
                    "maximum_standard_deviation_m": float(standard_deviations[-1]),
                }
            )
            center_records.append(record)
        update_records.append(
            {
                "frame": frame,
                "tracker": replay_tracker,
                "centers": center_records,
                "valid_covariance_count": int(np.sum(covariance_valid[frame])),
            }
        )

    output.mkdir(parents=True, exist_ok=False)
    archive_path = output / UNCERTAINTY_ARCHIVE_FILENAME
    np.savez_compressed(
        archive_path,
        measurement_covariance_m2=covariance,
        measurement_covariance_valid=covariance_valid,
        jacobian_covariance_m2=jacobian_covariance,
        jackknife_covariance_m2=jackknife_covariance,
        principal_standard_deviation_m=principal_std,
        rms_standard_deviation_m=rms_std,
        maximum_standard_deviation_m=maximum_std,
        information_condition_number=condition_number,
        replay_position_difference_m=replay_error,
        leave_one_out_sample_count=loo_sample_count,
        pixel_sigma=pixel_sigma,
    )
    sidecar_manifest: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": "Deform360CausalRawCameraMeasurementUncertainty",
        "protocol_id": UNCERTAINTY_PROTOCOL_ID,
        "case": case_dir.name,
        "object_id": str(seal["object_id"]),
        "episode_id": int(seal["episode_id"]),
        "episode_key": seal["episode_key"],
        "config": asdict(cfg),
        "inputs": {
            "measurement_manifest": {
                "path": str(measurement_path / MANIFEST_FILENAME),
                "sha256": _sha256(measurement_path / MANIFEST_FILENAME),
                "result_sha256": manifest["result_sha256"],
            },
            "measurement_archive": {
                "path": str(measurement_path / MEASUREMENT_FILENAME),
                "sha256": _sha256(measurement_path / MEASUREMENT_FILENAME),
            },
            "prediction_seal": {
                "path": str(seal_path),
                "sha256": _sha256(seal_path),
            },
            "intrinsics": manifest["inputs"]["intrinsics"],
            "extrinsics": manifest["inputs"]["extrinsics"],
        },
        "tracker": manifest["tracker"],
        "updates": update_records,
        "output": {
            "archive": str(archive_path),
            "archive_sha256": _sha256(archive_path),
            "valid_covariance_count_by_update": [
                int(record["valid_covariance_count"]) for record in update_records
            ],
        },
        "information_boundary": {
            "target_data_read": False,
            "outcome_manifest_read": False,
            "replayed_prefix_rule": "update u reads exactly frames [0, u]",
            "maximum_video_frame_read_by_update": [
                int(record["frame"]) for record in update_records
            ],
            "original_measurement_archive_modified": False,
        },
        "claim_boundary": (
            "outcome-free local covariance proxy; combines linearized isotropic "
            "pixel noise and leave-one-inlier-camera-out sensitivity; it is not "
            "a calibrated ground-truth error model"
        ),
    }
    sidecar_manifest["result_sha256"] = _canonical_sha256(sidecar_manifest)
    (output / UNCERTAINTY_MANIFEST_FILENAME).write_text(
        json.dumps(sidecar_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return sidecar_manifest


def build_raw_camera_uncertainty_cohort(
    panel_root: str | Path,
    processed_root: str | Path,
    measurement_root: str | Path,
    output_root: str | Path,
    runtime: AllTrackerPrefixRuntime,
    *,
    config: RawCameraUncertaintyConfig | None = None,
    shard_index: int = 0,
    shard_count: int = 1,
) -> dict[str, Any]:
    """Build a deterministic shard of the complete open-27 sidecar cohort."""

    if shard_count <= 0 or not 0 <= shard_index < shard_count:
        raise ValueError("invalid shard index/count")
    panel = Path(panel_root).resolve()
    processed = Path(processed_root).resolve()
    measurements = Path(measurement_root).resolve()
    output = Path(output_root).resolve()
    output.mkdir(parents=True, exist_ok=True)
    selected = [
        case
        for index, case in enumerate(expected_open_case_names())
        if index % shard_count == shard_index
    ]
    manifests: list[dict[str, Any]] = []
    for case in selected:
        case_output = output / case
        if case_output.exists():
            manifest_path = case_output / UNCERTAINTY_MANIFEST_FILENAME
            if not manifest_path.is_file():
                raise ValueError(
                    f"incomplete existing uncertainty output: {case_output}"
                )
            manifests.append(json.loads(manifest_path.read_text(encoding="utf-8")))
            continue
        manifests.append(
            build_raw_camera_uncertainty_case(
                panel / case,
                processed / case / "episode_0000",
                measurements / case,
                case_output,
                runtime,
                config=config,
            )
        )
    summary = {
        "protocol_id": UNCERTAINTY_PROTOCOL_ID,
        "shard_index": shard_index,
        "shard_count": shard_count,
        "case_count": len(manifests),
        "cases": [manifest["case"] for manifest in manifests],
        "manifest_sha256": {
            manifest["case"]: _sha256(
                output / manifest["case"] / UNCERTAINTY_MANIFEST_FILENAME
            )
            for manifest in manifests
        },
    }
    summary_path = output / f"build-uncertainty-shard-{shard_index:02d}.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


__all__ = [
    "RawCameraUncertaintyConfig",
    "UNCERTAINTY_ARCHIVE_FILENAME",
    "UNCERTAINTY_MANIFEST_FILENAME",
    "UNCERTAINTY_PROTOCOL_ID",
    "build_raw_camera_uncertainty_case",
    "build_raw_camera_uncertainty_cohort",
    "jacobian_measurement_covariance",
    "leave_one_camera_out_covariance",
    "projection_jacobian",
]
