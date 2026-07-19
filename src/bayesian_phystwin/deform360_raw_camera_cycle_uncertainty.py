"""Forward/backward cycle uncertainty for causal Deform360 camera tracks.

Each update reuses exactly the forward RGB prefix ``[0,u]`` and runs AllTracker
once more on that prefix in reverse frame order.  The endpoint track is queried
back toward frame zero.  Round-trip pixel error inflates the independent
Jacobian covariance; no target, outcome, depth after frame zero, or future frame
is read.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np

from .deform360_raw_camera_gated_evaluation import _load_uncertainty_artifact
from .deform360_raw_camera_observation import (
    MANIFEST_FILENAME,
    AllTrackerPrefixRuntime,
    _canonical_sha256,
    _load_measurement_artifact,
    _sha256,
    _validate_prediction_seal,
    expected_open_case_names,
    project_world_points,
)
from .deform360_raw_camera_uncertainty import (
    UNCERTAINTY_ARCHIVE_FILENAME,
    UNCERTAINTY_MANIFEST_FILENAME,
    _load_frame_zero_and_calibration,
)

CYCLE_PROTOCOL_ID = (
    "deform360-raw-camera-alltracker-forward-backward-cycle-v1-development"
)
CYCLE_ARCHIVE_FILENAME = "measurement_cycle_uncertainty.npz"
CYCLE_MANIFEST_FILENAME = "measurement_cycle_uncertainty_manifest.json"


@dataclass(frozen=True)
class RawCameraCycleUncertaintyConfig:
    """Fixed outcome-free cycle covariance settings."""

    minimum_cycle_view_count: int = 2
    pixel_noise_floor_px: float = 0.5

    def validate(self) -> None:
        if self.minimum_cycle_view_count < 2:
            raise ValueError("at least two cycle-consistent views are required")
        if self.pixel_noise_floor_px <= 0.0:
            raise ValueError("pixel noise floor must be positive")


def inflate_covariance_from_cycle(
    jacobian_covariance_m2: np.ndarray,
    jackknife_covariance_m2: np.ndarray,
    base_pixel_sigma: float,
    cycle_error_px: np.ndarray,
    *,
    pixel_noise_floor_px: float,
) -> tuple[np.ndarray, dict[str, float]]:
    """Inflate the Jacobian term using a Rayleigh-scaled cycle residual."""

    jacobian = np.asarray(jacobian_covariance_m2, dtype=float)
    jackknife = np.asarray(jackknife_covariance_m2, dtype=float)
    errors = np.asarray(cycle_error_px, dtype=float)
    if jacobian.shape != (3, 3) or jackknife.shape != (3, 3):
        raise ValueError("covariance terms must have shape (3, 3)")
    errors = errors[np.isfinite(errors) & (errors >= 0.0)]
    if not len(errors):
        raise ValueError("cycle errors must contain a finite nonnegative value")
    rayleigh_median_scale = np.sqrt(2.0 * np.log(2.0))
    median_error = float(np.median(errors))
    cycle_sigma = max(
        float(pixel_noise_floor_px),
        median_error / rayleigh_median_scale,
    )
    effective_sigma = max(float(base_pixel_sigma), cycle_sigma)
    scale = effective_sigma / float(base_pixel_sigma)
    combined = scale**2 * jacobian + jackknife
    combined = 0.5 * (combined + combined.T)
    eigenvalues = np.linalg.eigvalsh(combined)
    if eigenvalues[0] <= 0.0 or not np.all(np.isfinite(eigenvalues)):
        raise ValueError("cycle-inflated covariance is not positive definite")
    return combined, {
        "cycle_error_median_px": median_error,
        "cycle_error_maximum_px": float(np.max(errors)),
        "cycle_pixel_sigma": cycle_sigma,
        "base_pixel_sigma": float(base_pixel_sigma),
        "effective_pixel_sigma": effective_sigma,
        "jacobian_covariance_scale": scale**2,
    }


def build_raw_camera_cycle_uncertainty_case(
    panel_case_dir: str | Path,
    processed_episode_dir: str | Path,
    measurement_dir: str | Path,
    uncertainty_dir: str | Path,
    output_dir: str | Path,
    runtime: AllTrackerPrefixRuntime,
    *,
    config: RawCameraCycleUncertaintyConfig | None = None,
) -> dict[str, Any]:
    """Build one forward/backward cycle sidecar without opening the outcome."""

    cfg = config or RawCameraCycleUncertaintyConfig()
    cfg.validate()
    case_dir = Path(panel_case_dir).resolve()
    processed = Path(processed_episode_dir).resolve()
    measurement_path = Path(measurement_dir).resolve()
    uncertainty_path = Path(uncertainty_dir).resolve()
    output = Path(output_dir).resolve()
    if case_dir.name not in expected_open_case_names():
        raise ValueError("case is outside the explicit outcome-open panel")
    if output.exists():
        raise FileExistsError(output)
    seal_path = case_dir / "prediction_seal.json"
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    _validate_prediction_seal(seal)
    measurement_manifest, measurement_arrays = _load_measurement_artifact(
        case_dir,
        measurement_path,
        seal,
    )
    uncertainty_manifest, uncertainty_arrays = _load_uncertainty_artifact(
        measurement_path,
        uncertainty_path,
        seal,
    )
    frame_zero, intrinsics, extrinsics = _load_frame_zero_and_calibration(
        case_dir,
        processed,
        seal,
        measurement_manifest,
    )
    measurement = np.asarray(measurement_arrays["measurement_m"], dtype=float)
    base_validity = np.asarray(
        uncertainty_arrays["measurement_covariance_valid"], dtype=bool
    )
    base_jacobian = np.asarray(
        uncertainty_arrays["jacobian_covariance_m2"], dtype=float
    )
    base_jackknife = np.asarray(
        uncertainty_arrays["jackknife_covariance_m2"], dtype=float
    )
    base_pixel_sigma = np.asarray(uncertainty_arrays["pixel_sigma"], dtype=float)
    covariance = np.full(measurement.shape[:2] + (3, 3), np.nan, dtype=np.float32)
    covariance_valid = np.zeros(measurement.shape[:2], dtype=bool)
    cycle_error_median = np.full(measurement.shape[:2], np.nan, dtype=np.float32)
    cycle_error_maximum = np.full_like(cycle_error_median, np.nan)
    cycle_view_count = np.zeros(measurement.shape[:2], dtype=np.int16)
    cycle_pixel_sigma = np.full_like(cycle_error_median, np.nan)
    jacobian_scale = np.full_like(cycle_error_median, np.nan)

    selected_cameras = tuple(measurement_manifest["plan"]["selected_cameras"])
    update_records: list[dict[str, Any]] = []
    for update in measurement_manifest["updates"]:
        frame = int(update["frame"])
        source_tracker = {str(record["camera"]): record for record in update["tracker"]}
        cycles_by_camera: dict[str, dict[int, float]] = {}
        tracker_records: list[dict[str, Any]] = []
        for camera in selected_cameras:
            source_record = source_tracker[camera]
            query_ids = np.asarray(source_record["query_ids"], dtype=np.int64)
            initial_pixels = project_world_points(
                frame_zero[query_ids],
                intrinsics[camera],
                extrinsics[camera],
            )[0]
            endpoint_pixels, forward_visible, forward_record = runtime.track_prefix(
                processed / camera / "undistorted.mp4",
                initial_pixels,
                frame,
            )
            if (
                forward_record["decoded_rgb_prefix_sha256"]
                != source_record["decoded_rgb_prefix_sha256"]
            ):
                raise ValueError("cycle forward RGB prefix checksum differs")
            recovered_pixels, reverse_visible, reverse_record = (
                runtime.track_reversed_prefix(
                    processed / camera / "undistorted.mp4",
                    endpoint_pixels,
                    frame,
                )
            )
            if (
                reverse_record["decoded_rgb_prefix_sha256"]
                != source_record["decoded_rgb_prefix_sha256"]
            ):
                raise ValueError("cycle reverse RGB prefix checksum differs")
            cycle_valid = forward_visible & reverse_visible
            errors = np.linalg.norm(recovered_pixels - initial_pixels, axis=1)
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
        for source_center in update["centers"]:
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
                dtype=float,
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
    archive_path = output / CYCLE_ARCHIVE_FILENAME
    np.savez_compressed(
        archive_path,
        measurement_covariance_m2=covariance,
        measurement_covariance_valid=covariance_valid,
        cycle_error_median_px=cycle_error_median,
        cycle_error_maximum_px=cycle_error_maximum,
        cycle_valid_view_count=cycle_view_count,
        cycle_pixel_sigma=cycle_pixel_sigma,
        jacobian_covariance_scale=jacobian_scale,
    )
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": "Deform360CausalRawCameraCycleUncertainty",
        "protocol_id": CYCLE_PROTOCOL_ID,
        "case": case_dir.name,
        "object_id": str(seal["object_id"]),
        "episode_id": int(seal["episode_id"]),
        "episode_key": seal["episode_key"],
        "config": asdict(cfg),
        "inputs": {
            "measurement_manifest": {
                "path": str(measurement_path / MANIFEST_FILENAME),
                "sha256": _sha256(measurement_path / MANIFEST_FILENAME),
                "result_sha256": measurement_manifest["result_sha256"],
            },
            "uncertainty_manifest": {
                "path": str(uncertainty_path / UNCERTAINTY_MANIFEST_FILENAME),
                "sha256": _sha256(uncertainty_path / UNCERTAINTY_MANIFEST_FILENAME),
                "result_sha256": uncertainty_manifest["result_sha256"],
            },
            "uncertainty_archive": {
                "path": str(uncertainty_path / UNCERTAINTY_ARCHIVE_FILENAME),
                "sha256": _sha256(uncertainty_path / UNCERTAINTY_ARCHIVE_FILENAME),
            },
            "prediction_seal": {
                "path": str(seal_path),
                "sha256": _sha256(seal_path),
            },
        },
        "tracker": measurement_manifest["tracker"],
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
            "outcome-free forward/backward cycle proxy; cycle consistency can "
            "remain overconfident under temporally reversible common-mode drift"
        ),
    }
    manifest["result_sha256"] = _canonical_sha256(manifest)
    (output / CYCLE_MANIFEST_FILENAME).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def build_raw_camera_cycle_uncertainty_cohort(
    panel_root: str | Path,
    processed_root: str | Path,
    measurement_root: str | Path,
    uncertainty_root: str | Path,
    output_root: str | Path,
    runtime: AllTrackerPrefixRuntime,
    *,
    config: RawCameraCycleUncertaintyConfig | None = None,
    shard_index: int = 0,
    shard_count: int = 1,
) -> dict[str, Any]:
    """Build a deterministic shard of all 27 cycle uncertainty artifacts."""

    if shard_count <= 0 or not 0 <= shard_index < shard_count:
        raise ValueError("invalid shard index/count")
    panel = Path(panel_root).resolve()
    processed = Path(processed_root).resolve()
    measurements = Path(measurement_root).resolve()
    uncertainties = Path(uncertainty_root).resolve()
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
            manifest_path = case_output / CYCLE_MANIFEST_FILENAME
            if not manifest_path.is_file():
                raise ValueError(f"incomplete existing cycle output: {case_output}")
            manifests.append(json.loads(manifest_path.read_text(encoding="utf-8")))
            continue
        manifests.append(
            build_raw_camera_cycle_uncertainty_case(
                panel / case,
                processed / case / "episode_0000",
                measurements / case,
                uncertainties / case,
                case_output,
                runtime,
                config=config,
            )
        )
    summary = {
        "protocol_id": CYCLE_PROTOCOL_ID,
        "shard_index": shard_index,
        "shard_count": shard_count,
        "case_count": len(manifests),
        "cases": [manifest["case"] for manifest in manifests],
        "manifest_sha256": {
            manifest["case"]: _sha256(
                output / manifest["case"] / CYCLE_MANIFEST_FILENAME
            )
            for manifest in manifests
        },
    }
    (output / f"build-cycle-shard-{shard_index:02d}.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


__all__ = [
    "CYCLE_ARCHIVE_FILENAME",
    "CYCLE_MANIFEST_FILENAME",
    "CYCLE_PROTOCOL_ID",
    "RawCameraCycleUncertaintyConfig",
    "build_raw_camera_cycle_uncertainty_case",
    "build_raw_camera_cycle_uncertainty_cohort",
    "inflate_covariance_from_cycle",
]
