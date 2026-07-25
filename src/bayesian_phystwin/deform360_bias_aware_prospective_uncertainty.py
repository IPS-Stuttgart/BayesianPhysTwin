"""Target-free covariance sidecars for the prospective Deform360 panel."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .deform360_bias_aware_prospective_artifacts import (
    BACKBONE_SEAL_FILENAME,
    MEASUREMENT_CYCLE_ARCHIVE_FILENAME,
    MEASUREMENT_CYCLE_MANIFEST_FILENAME,
    canonical_sha256,
    file_sha256,
    load_prospective_measurement,
    validate_prospective_backbone_seal,
)
from .deform360_bias_aware_prospective_protocol import PROTOCOL_ID
from .deform360_raw_camera_observation import (
    MANIFEST_FILENAME,
    MEASUREMENT_FILENAME,
    AllTrackerPrefixRuntime,
    _linear_triangulation,
    _load_calibration,
    _projection_matrix,
    _reproject,
    _resolve_prediction_archive,
    project_world_points,
)


UNCERTAINTY_ARCHIVE_FILENAME = "measurement_uncertainty.npz"
UNCERTAINTY_MANIFEST_FILENAME = "measurement_uncertainty_manifest.json"
UNCERTAINTY_PROTOCOL_ID = "deform360-bias-aware-prospective-jackknife-uncertainty-v1"
CYCLE_PROTOCOL_ID = "deform360-bias-aware-prospective-cycle-uncertainty-v1"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


@dataclass(frozen=True)
class ProspectiveRawCameraUncertaintyConfig:
    """Frozen covariance construction settings inherited from source v4."""

    pixel_noise_floor_px: float = 0.5
    maximum_information_condition_number: float = 1.0e8
    replay_position_tolerance_m: float = 5.0e-4

    def validate(self) -> None:
        _require(self.pixel_noise_floor_px > 0.0, "pixel floor must be positive")
        _require(
            self.maximum_information_condition_number > 1.0,
            "condition ceiling must exceed one",
        )
        _require(
            self.replay_position_tolerance_m > 0.0,
            "replay tolerance must be positive",
        )


@dataclass(frozen=True)
class ProspectiveRawCameraCycleConfig:
    """Frozen forward/backward cycle-inflation settings."""

    minimum_cycle_view_count: int = 2
    pixel_noise_floor_px: float = 0.5

    def validate(self) -> None:
        _require(
            self.minimum_cycle_view_count >= 2,
            "at least two cycle views are required",
        )
        _require(self.pixel_noise_floor_px > 0.0, "pixel floor must be positive")


def projection_jacobian(
    point_m: np.ndarray, projection_matrix: np.ndarray
) -> np.ndarray:
    """Return the analytic two-by-three perspective Jacobian."""

    point = np.append(np.asarray(point_m, dtype=np.float64), 1.0)
    matrix = np.asarray(projection_matrix, dtype=np.float64)
    _require(matrix.shape == (3, 4), "projection matrix shape changed")
    projected = matrix @ point
    denominator = float(projected[2])
    _require(denominator > 1.0e-12, "point is behind the camera")
    jacobian = np.stack(
        (
            (matrix[0, :3] * denominator - projected[0] * matrix[2, :3])
            / denominator**2,
            (matrix[1, :3] * denominator - projected[1] * matrix[2, :3])
            / denominator**2,
        )
    )
    _require(np.all(np.isfinite(jacobian)), "projection Jacobian is non-finite")
    return jacobian


def jacobian_measurement_covariance(
    point_m: np.ndarray,
    projection_matrices: Sequence[np.ndarray],
    pixel_sigma: float,
    *,
    maximum_condition_number: float,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    """Linearized isotropic-pixel covariance for one multiview point."""

    if len(projection_matrices) < 2:
        return None, {"decision": "insufficient_views"}
    try:
        jacobian = np.concatenate(
            [projection_jacobian(point_m, matrix) for matrix in projection_matrices]
        )
    except ValueError:
        return None, {"decision": "invalid_projection"}
    information = jacobian.T @ jacobian
    eigenvalues = np.linalg.eigvalsh(information)
    if eigenvalues[0] <= 0.0 or not np.all(np.isfinite(eigenvalues)):
        return None, {"decision": "singular_information"}
    condition = float(eigenvalues[-1] / eigenvalues[0])
    diagnostic: dict[str, Any] = {
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
    """Return jackknife covariance and valid leave-one-view estimates."""

    ordered = tuple(
        (camera, np.asarray(observations[camera], dtype=np.float64))
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
    samples = np.asarray(estimates, dtype=np.float64).reshape(-1, 3)
    if len(samples) < 2:
        return np.zeros((3, 3), dtype=np.float64), samples
    centered = samples - np.mean(samples, axis=0, keepdims=True)
    covariance = (len(samples) - 1.0) / len(samples) * centered.T @ centered
    return 0.5 * (covariance + covariance.T), samples


def _pixel_sigma_from_median_reprojection(
    median_reprojection_px: float, floor_px: float
) -> float:
    rayleigh_median_scale = np.sqrt(2.0 * np.log(2.0))
    return max(float(floor_px), float(median_reprojection_px) / rayleigh_median_scale)


def _load_frame_zero_and_calibration(
    protocol_path: str | Path,
    case_dir: Path,
    processed_dir: Path,
    seal: Mapping[str, Any],
    measurement_manifest: Mapping[str, Any],
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray]]:
    validate_prospective_backbone_seal(
        seal, protocol_path=protocol_path, case_dir=case_dir
    )
    archive = _resolve_prediction_archive(case_dir, seal)
    _require(
        file_sha256(archive)
        == measurement_manifest["inputs"]["prediction_archive"]["sha256"],
        "prediction archive checksum changed",
    )
    with np.load(archive, allow_pickle=False) as stored:
        frame_zero = np.asarray(stored["frame_zero_points_m"], dtype=np.float64).copy()
    intrinsics_path = processed_dir / "undistorted_intrinsics.npy"
    extrinsics_path = processed_dir / "extrinsics.npy"
    _require(
        file_sha256(intrinsics_path)
        == measurement_manifest["inputs"]["intrinsics"]["sha256"],
        "intrinsics changed",
    )
    _require(
        file_sha256(extrinsics_path)
        == measurement_manifest["inputs"]["extrinsics"]["sha256"],
        "extrinsics changed",
    )
    intrinsics, extrinsics = _load_calibration(processed_dir)
    return frame_zero, intrinsics, extrinsics


def build_prospective_raw_camera_uncertainty_case(
    protocol_path: str | Path,
    backbone_case_dir: str | Path,
    processed_episode_dir: str | Path,
    measurement_dir: str | Path,
    output_dir: str | Path,
    runtime: AllTrackerPrefixRuntime,
    *,
    config: ProspectiveRawCameraUncertaintyConfig | None = None,
) -> dict[str, Any]:
    """Build the source-v4 Jacobian/jackknife sidecar without an outcome."""

    cfg = config or ProspectiveRawCameraUncertaintyConfig()
    cfg.validate()
    case_dir = Path(backbone_case_dir).resolve()
    processed = Path(processed_episode_dir).resolve()
    measurement_root = Path(measurement_dir).resolve()
    output = Path(output_dir).resolve()
    _require(not output.exists(), "uncertainty output already exists")
    manifest, measurement_arrays, seal = load_prospective_measurement(
        protocol_path, case_dir, measurement_root
    )
    frame_zero, intrinsics, extrinsics = _load_frame_zero_and_calibration(
        protocol_path, case_dir, processed, seal, manifest
    )
    measurement = np.asarray(measurement_arrays["measurement_m"], dtype=np.float64)
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
    leave_one_out_count = np.zeros((time_count, point_count), dtype=np.int16)
    pixel_sigma = np.full_like(rms_std, np.nan)

    selected_cameras = tuple(manifest["plan"]["selected_cameras"])
    projections = {
        camera: _projection_matrix(intrinsics[camera], extrinsics[camera])
        for camera in selected_cameras
    }
    update_records: list[dict[str, Any]] = []
    for source_update in manifest["updates"]:
        frame = int(source_update["frame"])
        source_tracker = {
            str(record["camera"]): record for record in source_update["tracker"]
        }
        tracks_by_camera: dict[str, dict[int, np.ndarray]] = {}
        replay_tracker: list[dict[str, Any]] = []
        for camera in selected_cameras:
            source_record = source_tracker[camera]
            query_ids = np.asarray(source_record["query_ids"], dtype=np.int64)
            query_pixels = project_world_points(
                frame_zero[query_ids], intrinsics[camera], extrinsics[camera]
            )[0]
            tracks, visible, tracker_record = runtime.track_prefix(
                processed / camera / "undistorted.mp4", query_pixels, frame
            )
            _require(
                tracker_record["decoded_rgb_prefix_sha256"]
                == source_record["decoded_rgb_prefix_sha256"],
                "replayed RGB prefix differs",
            )
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
        for source_center in source_update["centers"]:
            center_id = int(source_center["center_id"])
            record: dict[str, Any] = {
                "center_id": center_id,
                "source_measurement_accepted": bool(source_center["accepted"]),
                "covariance_valid": False,
                "decision": "source_measurement_rejected",
            }
            if not source_center["accepted"] or not measurement_validity[frame, center_id]:
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
            sequence = tuple(sorted(observations.items()))
            try:
                replayed = _linear_triangulation(sequence, projections)
            except (ValueError, np.linalg.LinAlgError):
                record["decision"] = "replay_triangulation_failure"
                center_records.append(record)
                continue
            point = measurement[frame, center_id]
            difference = float(np.linalg.norm(replayed - point))
            replay_error[frame, center_id] = difference
            record["replay_position_difference_m"] = difference
            if difference > cfg.replay_position_tolerance_m:
                record["decision"] = "replay_position_mismatch"
                center_records.append(record)
                continue
            sigma = _pixel_sigma_from_median_reprojection(
                float(source_center["median_reprojection_error_px"]),
                cfg.pixel_noise_floor_px,
            )
            geometric, geometric_diagnostic = jacobian_measurement_covariance(
                point,
                [projections[camera] for camera in sorted(observations)],
                sigma,
                maximum_condition_number=cfg.maximum_information_condition_number,
            )
            record["jacobian"] = geometric_diagnostic
            if geometric is None:
                record["decision"] = geometric_diagnostic["decision"]
                center_records.append(record)
                continue
            empirical, estimates = leave_one_camera_out_covariance(
                observations, projections
            )
            combined = 0.5 * (geometric + empirical + (geometric + empirical).T)
            eigenvalues = np.linalg.eigvalsh(combined)
            if eigenvalues[0] <= 0.0 or not np.all(np.isfinite(eigenvalues)):
                record["decision"] = "combined_covariance_failure"
                center_records.append(record)
                continue
            standard_deviations = np.sqrt(eigenvalues)
            covariance[frame, center_id] = combined
            jacobian_covariance[frame, center_id] = geometric
            jackknife_covariance[frame, center_id] = empirical
            covariance_valid[frame, center_id] = True
            principal_std[frame, center_id] = standard_deviations
            rms_std[frame, center_id] = float(np.sqrt(np.trace(combined) / 3.0))
            maximum_std[frame, center_id] = float(standard_deviations[-1])
            condition_number[frame, center_id] = float(
                geometric_diagnostic["information_condition_number"]
            )
            leave_one_out_count[frame, center_id] = len(estimates)
            pixel_sigma[frame, center_id] = sigma
            record.update(
                {
                    "covariance_valid": True,
                    "decision": "accepted",
                    "pixel_sigma": sigma,
                    "leave_one_out_sample_count": len(estimates),
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
    archive = output / UNCERTAINTY_ARCHIVE_FILENAME
    np.savez_compressed(
        archive,
        measurement_covariance_m2=covariance,
        measurement_covariance_valid=covariance_valid,
        jacobian_covariance_m2=jacobian_covariance,
        jackknife_covariance_m2=jackknife_covariance,
        principal_standard_deviation_m=principal_std,
        rms_standard_deviation_m=rms_std,
        maximum_standard_deviation_m=maximum_std,
        information_condition_number=condition_number,
        replay_position_difference_m=replay_error,
        leave_one_out_sample_count=leave_one_out_count,
        pixel_sigma=pixel_sigma,
    )
    sidecar: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": "Deform360BiasAwareProspectiveMeasurementUncertainty",
        "protocol_id": UNCERTAINTY_PROTOCOL_ID,
        **{key: seal[key] for key in ("case", "object_id", "episode_id", "episode_key", "role")},
        "config": asdict(cfg),
        "inputs": {
            "measurement_manifest": {
                "path": str(measurement_root / MANIFEST_FILENAME),
                "sha256": file_sha256(measurement_root / MANIFEST_FILENAME),
                "result_sha256": manifest["result_sha256"],
            },
            "measurement_archive": {
                "path": str(measurement_root / MEASUREMENT_FILENAME),
                "sha256": file_sha256(measurement_root / MEASUREMENT_FILENAME),
            },
            "prediction_seal": {
                "path": str(case_dir / BACKBONE_SEAL_FILENAME),
                "sha256": file_sha256(case_dir / BACKBONE_SEAL_FILENAME),
            },
            "intrinsics": manifest["inputs"]["intrinsics"],
            "extrinsics": manifest["inputs"]["extrinsics"],
        },
        "tracker": manifest["tracker"],
        "updates": update_records,
        "output": {
            "archive": str(archive),
            "archive_sha256": file_sha256(archive),
            "valid_covariance_count_by_update": [
                int(record["valid_covariance_count"]) for record in update_records
            ],
        },
        "information_boundary": {
            "target_data_read": False,
            "outcome_manifest_read": False,
            "replayed_prefix_rule": "update u reads exactly frames [0,u]",
            "maximum_video_frame_read_by_update": [
                int(record["frame"]) for record in update_records
            ],
            "original_measurement_archive_modified": False,
        },
        "claim_boundary": (
            "outcome-free local covariance proxy; not a calibrated ground-truth "
            "error model"
        ),
    }
    sidecar["result_sha256"] = canonical_sha256(sidecar, digest_key="result_sha256")
    _write_json(output / UNCERTAINTY_MANIFEST_FILENAME, sidecar)
    return sidecar


def inflate_covariance_from_cycle(
    jacobian_covariance_m2: np.ndarray,
    jackknife_covariance_m2: np.ndarray,
    base_pixel_sigma: float,
    cycle_error_px: np.ndarray,
    *,
    pixel_noise_floor_px: float,
) -> tuple[np.ndarray, dict[str, float]]:
    """Inflate the Jacobian term using a Rayleigh-scaled cycle residual."""

    jacobian = np.asarray(jacobian_covariance_m2, dtype=np.float64)
    jackknife = np.asarray(jackknife_covariance_m2, dtype=np.float64)
    errors = np.asarray(cycle_error_px, dtype=np.float64)
    _require(jacobian.shape == jackknife.shape == (3, 3), "covariance shape changed")
    errors = errors[np.isfinite(errors) & (errors >= 0.0)]
    _require(len(errors) > 0, "cycle errors are empty")
    rayleigh_median_scale = np.sqrt(2.0 * np.log(2.0))
    median_error = float(np.median(errors))
    cycle_sigma = max(
        float(pixel_noise_floor_px), median_error / rayleigh_median_scale
    )
    effective_sigma = max(float(base_pixel_sigma), cycle_sigma)
    scale = effective_sigma / float(base_pixel_sigma)
    combined = scale**2 * jacobian + jackknife
    combined = 0.5 * (combined + combined.T)
    eigenvalues = np.linalg.eigvalsh(combined)
    _require(
        eigenvalues[0] > 0.0 and np.all(np.isfinite(eigenvalues)),
        "cycle covariance is not positive definite",
    )
    return combined, {
        "cycle_error_median_px": median_error,
        "cycle_error_maximum_px": float(np.max(errors)),
        "cycle_pixel_sigma": cycle_sigma,
        "base_pixel_sigma": float(base_pixel_sigma),
        "effective_pixel_sigma": effective_sigma,
        "jacobian_covariance_scale": scale**2,
    }


def _load_uncertainty_artifact(
    uncertainty_dir: Path,
    *,
    measurement_manifest: Mapping[str, Any],
    measurement_dir: Path,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    manifest_path = uncertainty_dir / UNCERTAINTY_MANIFEST_FILENAME
    archive_path = uncertainty_dir / UNCERTAINTY_ARCHIVE_FILENAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _require(
        manifest.get("result_sha256")
        == canonical_sha256(manifest, digest_key="result_sha256"),
        "uncertainty manifest checksum changed",
    )
    _require(
        manifest.get("inputs", {}).get("measurement_manifest", {}).get("sha256")
        == file_sha256(measurement_dir / MANIFEST_FILENAME)
        and manifest.get("inputs", {})
        .get("measurement_manifest", {})
        .get("result_sha256")
        == measurement_manifest["result_sha256"],
        "uncertainty used another measurement",
    )
    _require(
        manifest.get("output", {}).get("archive_sha256") == file_sha256(archive_path),
        "uncertainty archive changed",
    )
    with np.load(archive_path, allow_pickle=False) as stored:
        arrays = {name: np.asarray(stored[name]).copy() for name in stored.files}
    return manifest, arrays


def build_prospective_raw_camera_cycle_case(
    protocol_path: str | Path,
    backbone_case_dir: str | Path,
    processed_episode_dir: str | Path,
    measurement_dir: str | Path,
    uncertainty_dir: str | Path,
    output_dir: str | Path,
    runtime: AllTrackerPrefixRuntime,
    *,
    config: ProspectiveRawCameraCycleConfig | None = None,
) -> dict[str, Any]:
    """Build the source-v4 forward/backward cycle sidecar without outcomes."""

    cfg = config or ProspectiveRawCameraCycleConfig()
    cfg.validate()
    case_dir = Path(backbone_case_dir).resolve()
    processed = Path(processed_episode_dir).resolve()
    measurement_root = Path(measurement_dir).resolve()
    uncertainty_root = Path(uncertainty_dir).resolve()
    output = Path(output_dir).resolve()
    _require(not output.exists(), "cycle output already exists")
    measurement_manifest, measurement_arrays, seal = load_prospective_measurement(
        protocol_path, case_dir, measurement_root
    )
    uncertainty_manifest, uncertainty = _load_uncertainty_artifact(
        uncertainty_root,
        measurement_manifest=measurement_manifest,
        measurement_dir=measurement_root,
    )
    frame_zero, intrinsics, extrinsics = _load_frame_zero_and_calibration(
        protocol_path, case_dir, processed, seal, measurement_manifest
    )
    measurement = np.asarray(measurement_arrays["measurement_m"], dtype=np.float64)
    base_validity = np.asarray(
        uncertainty["measurement_covariance_valid"], dtype=bool
    )
    base_jacobian = np.asarray(uncertainty["jacobian_covariance_m2"], dtype=np.float64)
    base_jackknife = np.asarray(
        uncertainty["jackknife_covariance_m2"], dtype=np.float64
    )
    base_pixel_sigma = np.asarray(uncertainty["pixel_sigma"], dtype=np.float64)
    covariance = np.full(measurement.shape[:2] + (3, 3), np.nan, dtype=np.float32)
    covariance_valid = np.zeros(measurement.shape[:2], dtype=bool)
    cycle_error_median = np.full(measurement.shape[:2], np.nan, dtype=np.float32)
    cycle_error_maximum = np.full_like(cycle_error_median, np.nan)
    cycle_view_count = np.zeros(measurement.shape[:2], dtype=np.int16)
    cycle_pixel_sigma = np.full_like(cycle_error_median, np.nan)
    jacobian_scale = np.full_like(cycle_error_median, np.nan)

    selected_cameras = tuple(measurement_manifest["plan"]["selected_cameras"])
    update_records: list[dict[str, Any]] = []
    for source_update in measurement_manifest["updates"]:
        frame = int(source_update["frame"])
        source_tracker = {
            str(record["camera"]): record for record in source_update["tracker"]
        }
        cycles_by_camera: dict[str, dict[int, float]] = {}
        tracker_records: list[dict[str, Any]] = []
        for camera in selected_cameras:
            source_record = source_tracker[camera]
            query_ids = np.asarray(source_record["query_ids"], dtype=np.int64)
            initial_pixels = project_world_points(
                frame_zero[query_ids], intrinsics[camera], extrinsics[camera]
            )[0]
            endpoint, forward_visible, forward_record = runtime.track_prefix(
                processed / camera / "undistorted.mp4", initial_pixels, frame
            )
            recovered, reverse_visible, reverse_record = runtime.track_reversed_prefix(
                processed / camera / "undistorted.mp4", endpoint, frame
            )
            expected_prefix = source_record["decoded_rgb_prefix_sha256"]
            _require(
                forward_record["decoded_rgb_prefix_sha256"] == expected_prefix
                and reverse_record["decoded_rgb_prefix_sha256"] == expected_prefix,
                "cycle RGB prefix differs",
            )
            cycle_valid = forward_visible & reverse_visible
            errors = np.linalg.norm(recovered - initial_pixels, axis=1)
            cycles_by_camera[camera] = {
                int(point_id): float(errors[index])
                for index, point_id in enumerate(query_ids)
                if cycle_valid[index]
            }
            tracker_records.append(
                {
                    "camera": camera,
                    "query_ids": query_ids.tolist(),
                    "cycle_valid_count": int(np.sum(cycle_valid)),
                    "cycle_error_median_px": (
                        None
                        if not np.any(cycle_valid)
                        else float(np.median(errors[cycle_valid]))
                    ),
                    "cycle_error_maximum_px": (
                        None
                        if not np.any(cycle_valid)
                        else float(np.max(errors[cycle_valid]))
                    ),
                    "forward": forward_record,
                    "reverse": reverse_record,
                    "source_prefix_sha256_matched": True,
                }
            )

        center_records: list[dict[str, Any]] = []
        for source_center in source_update["centers"]:
            center_id = int(source_center["center_id"])
            record: dict[str, Any] = {
                "center_id": center_id,
                "source_measurement_accepted": bool(source_center["accepted"]),
                "covariance_valid": False,
                "decision": "source_measurement_rejected",
            }
            if not source_center["accepted"] or not base_validity[frame, center_id]:
                center_records.append(record)
                continue
            inlier_cameras = tuple(source_center["inlier_cameras"])
            errors = np.asarray(
                [
                    cycles_by_camera[camera][center_id]
                    for camera in inlier_cameras
                    if center_id in cycles_by_camera[camera]
                ],
                dtype=np.float64,
            )
            cycle_view_count[frame, center_id] = len(errors)
            record["source_inlier_view_count"] = len(inlier_cameras)
            record["cycle_valid_view_count"] = len(errors)
            if len(errors) < cfg.minimum_cycle_view_count:
                record["decision"] = "insufficient_cycle_views"
                center_records.append(record)
                continue
            try:
                combined, diagnostic = inflate_covariance_from_cycle(
                    base_jacobian[frame, center_id],
                    base_jackknife[frame, center_id],
                    float(base_pixel_sigma[frame, center_id]),
                    errors,
                    pixel_noise_floor_px=cfg.pixel_noise_floor_px,
                )
            except ValueError as error:
                record["decision"] = "cycle_covariance_failure"
                record["error"] = str(error)
                center_records.append(record)
                continue
            covariance[frame, center_id] = combined
            covariance_valid[frame, center_id] = True
            cycle_error_median[frame, center_id] = diagnostic["cycle_error_median_px"]
            cycle_error_maximum[frame, center_id] = diagnostic["cycle_error_maximum_px"]
            cycle_pixel_sigma[frame, center_id] = diagnostic["cycle_pixel_sigma"]
            jacobian_scale[frame, center_id] = diagnostic["jacobian_covariance_scale"]
            record.update(
                {
                    "covariance_valid": True,
                    "decision": "accepted",
                    **diagnostic,
                    "principal_standard_deviation_m": np.sqrt(
                        np.linalg.eigvalsh(combined)
                    ).tolist(),
                    "rms_standard_deviation_m": float(
                        np.sqrt(np.trace(combined) / 3.0)
                    ),
                }
            )
            center_records.append(record)
        update_records.append(
            {
                "frame": frame,
                "tracker": tracker_records,
                "centers": center_records,
                "valid_covariance_count": int(np.sum(covariance_valid[frame])),
            }
        )

    output.mkdir(parents=True, exist_ok=False)
    archive = output / MEASUREMENT_CYCLE_ARCHIVE_FILENAME
    np.savez_compressed(
        archive,
        measurement_covariance_m2=covariance,
        measurement_covariance_valid=covariance_valid,
        cycle_error_median_px=cycle_error_median,
        cycle_error_maximum_px=cycle_error_maximum,
        cycle_valid_view_count=cycle_view_count,
        cycle_pixel_sigma=cycle_pixel_sigma,
        jacobian_covariance_scale=jacobian_scale,
    )
    cycle_manifest: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": "Deform360BiasAwareProspectiveCycleUncertainty",
        "protocol_id": CYCLE_PROTOCOL_ID,
        "parent_protocol_id": PROTOCOL_ID,
        **{key: seal[key] for key in ("case", "object_id", "episode_id", "episode_key", "role")},
        "config": asdict(cfg),
        "inputs": {
            "measurement_manifest": {
                "path": str(measurement_root / MANIFEST_FILENAME),
                "sha256": file_sha256(measurement_root / MANIFEST_FILENAME),
                "result_sha256": measurement_manifest["result_sha256"],
            },
            "uncertainty_manifest": {
                "path": str(uncertainty_root / UNCERTAINTY_MANIFEST_FILENAME),
                "sha256": file_sha256(
                    uncertainty_root / UNCERTAINTY_MANIFEST_FILENAME
                ),
                "result_sha256": uncertainty_manifest["result_sha256"],
            },
            "uncertainty_archive": {
                "path": str(uncertainty_root / UNCERTAINTY_ARCHIVE_FILENAME),
                "sha256": file_sha256(uncertainty_root / UNCERTAINTY_ARCHIVE_FILENAME),
            },
            "prediction_seal": {
                "path": str(case_dir / BACKBONE_SEAL_FILENAME),
                "sha256": file_sha256(case_dir / BACKBONE_SEAL_FILENAME),
            },
        },
        "tracker": measurement_manifest["tracker"],
        "updates": update_records,
        "output": {
            "archive": str(archive),
            "archive_sha256": file_sha256(archive),
            "valid_covariance_count_by_update": [
                int(record["valid_covariance_count"]) for record in update_records
            ],
        },
        "information_boundary": {
            "target_data_read": False,
            "outcome_manifest_read": False,
            "video_prefix_rule": "update u reads exactly frames [0,u]",
            "forward_model_order": "0 through u",
            "reverse_model_order": "u through 0",
            "maximum_source_video_frame_read_by_update": [
                int(record["frame"]) for record in update_records
            ],
            "future_frame_read": False,
            "original_measurement_and_uncertainty_archives_modified": False,
        },
        "claim_boundary": (
            "outcome-free forward/backward cycle proxy; common-mode camera bias "
            "can remain invisible"
        ),
    }
    cycle_manifest["result_sha256"] = canonical_sha256(
        cycle_manifest, digest_key="result_sha256"
    )
    _write_json(output / MEASUREMENT_CYCLE_MANIFEST_FILENAME, cycle_manifest)
    return cycle_manifest


__all__ = [
    "CYCLE_PROTOCOL_ID",
    "ProspectiveRawCameraCycleConfig",
    "ProspectiveRawCameraUncertaintyConfig",
    "UNCERTAINTY_ARCHIVE_FILENAME",
    "UNCERTAINTY_MANIFEST_FILENAME",
    "UNCERTAINTY_PROTOCOL_ID",
    "build_prospective_raw_camera_cycle_case",
    "build_prospective_raw_camera_uncertainty_case",
    "inflate_covariance_from_cycle",
    "jacobian_measurement_covariance",
    "leave_one_camera_out_covariance",
    "projection_jacobian",
]
