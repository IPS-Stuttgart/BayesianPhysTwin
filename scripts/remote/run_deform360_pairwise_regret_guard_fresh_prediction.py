#!/usr/bin/env python3
"""Build one frozen raw-prefix guarded prediction before outcome opening."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

import numpy as np

from bayesian_phystwin.deform360_pairwise_regret_guard_fresh_artifacts import (
    FAILURE_SEAL_FILENAME,
    PHYSICAL_SEAL_FILENAME,
    build_fresh_guarded_prediction,
    build_fresh_runtime_failure_seal,
    fresh_case_records,
    validate_fresh_physical_seal,
)
from bayesian_phystwin.deform360_pairwise_regret_guard_fresh_processing import (
    PROCESSING_KIND,
    canonical_sha256,
    validate_case_artifact,
    validate_fresh_processing_protocol,
)
from bayesian_phystwin.deform360_pairwise_regret_guard_fresh_protocol import (
    file_sha256,
    validate_fresh_technical_lock,
)
from bayesian_phystwin.deform360_raw_camera_observation import (
    AllTrackerPrefixRuntime,
    RawCameraObservationConfig,
    build_raw_camera_measurement_case_with_contract,
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _load_json(path: str | Path) -> dict[str, object]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"JSON object expected: {path}")
    return value


def _require_clean_repository(repository: Path) -> str:
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=normal"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    _require(not status.strip(), f"repository is dirty: {repository}")
    return revision


def _stage_measurement_panel(
    processed_episode_dir: Path,
    output_dir: Path,
    physical_seal: dict[str, object],
    protocol: dict[str, object],
) -> dict[str, object]:
    """Expose exactly the cameras admitted by outcome-blind preprocessing."""

    processed = processed_episode_dir.resolve()
    output = output_dir.resolve()
    _require(not output.exists(), "measurement camera panel already exists")
    processing_path = processed / "fresh_pairwise_processing.json"
    processing = _load_json(processing_path)
    case = {
        key: physical_seal[key]
        for key in (
            "object_id",
            "episode_id",
            "action",
            "bimanual",
            "nonprehensile",
            "case",
        )
    }
    validate_case_artifact(
        processing,
        artifact_kind=PROCESSING_KIND,
        protocol=protocol,
        case=case,
    )
    _require(processing.get("status") == "admitted", "source was not admitted")
    _require(
        processing.get("result_sha256")
        == physical_seal.get("source_processing_result_sha256"),
        "processing artifact differs from the physical seal",
    )
    cameras_value = processing.get("cameras")
    _require(
        isinstance(cameras_value, list)
        and all(isinstance(camera, str) and camera for camera in cameras_value)
        and len(cameras_value) == len(set(cameras_value)),
        "admitted camera panel is malformed",
    )
    cameras = tuple(cameras_value)
    minimum = int(protocol["processing"]["minimum_processing_cameras"])
    _require(len(cameras) >= minimum, "admitted camera panel is below the minimum")

    intrinsic_source = processed / "undistorted_intrinsics.npy"
    extrinsic_source = processed / "extrinsics.npy"
    intrinsics = np.load(intrinsic_source, allow_pickle=True).item()
    extrinsics = np.load(extrinsic_source, allow_pickle=True).item()
    _require(
        isinstance(intrinsics, dict)
        and isinstance(extrinsics, dict)
        and set(cameras) <= set(intrinsics)
        and set(cameras) <= set(extrinsics),
        "admitted camera calibration is incomplete",
    )

    output.mkdir(parents=True, exist_ok=False)
    intrinsic_output = output / intrinsic_source.name
    extrinsic_output = output / extrinsic_source.name
    np.save(intrinsic_output, {camera: intrinsics[camera] for camera in cameras})
    np.save(extrinsic_output, {camera: extrinsics[camera] for camera in cameras})
    camera_inputs: dict[str, object] = {}
    for camera in cameras:
        source_camera = processed / camera
        destination_camera = output / camera
        destination_camera.mkdir()
        assets: dict[str, object] = {}
        for filename in ("undistorted.mp4", "mask_refined.h5", "rendered_depth.h5"):
            source = source_camera / filename
            _require(source.is_file(), f"admitted camera asset is missing: {source}")
            destination = destination_camera / filename
            os.symlink(source.resolve(), destination)
            assets[filename] = {
                "source_path": str(source.resolve()),
                "source_size_bytes": source.stat().st_size,
                "whole_file_hashed_or_read_during_staging": False,
            }
        camera_inputs[camera] = assets

    manifest: dict[str, object] = {
        "schema_version": 1,
        "artifact_kind": "Deform360PairwiseRegretGuardFreshMeasurementPanel",
        "protocol_id": physical_seal["protocol_id"],
        "technical_lock_sha256": physical_seal["technical_lock_sha256"],
        **case,
        "camera_count": len(cameras),
        "cameras": list(cameras),
        "source_processing": {
            "path": str(processing_path),
            "file_sha256": file_sha256(processing_path),
            "result_sha256": processing["result_sha256"],
        },
        "calibration": {
            "source_intrinsics_file_sha256": file_sha256(intrinsic_source),
            "source_extrinsics_file_sha256": file_sha256(extrinsic_source),
            "staged_intrinsics_file_sha256": file_sha256(intrinsic_output),
            "staged_extrinsics_file_sha256": file_sha256(extrinsic_output),
        },
        "camera_inputs": camera_inputs,
        "information_boundary": {
            "camera_panel_derived_from_outcome_blind_processing": True,
            "future_video_or_hdf5_bytes_read_during_staging": False,
            "target_or_metric_read": False,
            "outcome_manifest_read": False,
            "held_v8_runtime_or_target_artifact_access": False,
        },
    }
    manifest["result_sha256"] = canonical_sha256(manifest, digest_key="result_sha256")
    (output / "measurement_panel.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return manifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--technical-lock", type=Path, required=True)
    parser.add_argument("--processing-protocol", type=Path, required=True)
    parser.add_argument("--physical-case-dir", type=Path, required=True)
    parser.add_argument("--processed-episode-dir", type=Path, required=True)
    parser.add_argument("--measurement-stage-root", type=Path, required=True)
    parser.add_argument("--measurement-root", type=Path, required=True)
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--source-qualification", type=Path, required=True)
    parser.add_argument("--alltracker-source", type=Path, required=True)
    parser.add_argument("--alltracker-checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    repo = args.repo.resolve()
    revision = _require_clean_repository(repo)
    lock_path = args.technical_lock.resolve()
    protocol_path = args.processing_protocol.resolve()
    lock = _load_json(lock_path)
    protocol = _load_json(protocol_path)
    validate_fresh_technical_lock(lock)
    validate_fresh_processing_protocol(protocol)
    physical_root = args.physical_case_dir.resolve()
    physical_seal_path = physical_root / PHYSICAL_SEAL_FILENAME
    physical_seal = _load_json(physical_seal_path)
    validate_fresh_physical_seal(
        physical_seal,
        case_dir=physical_root,
        lock=lock,
        protocol=protocol,
    )
    case = str(physical_seal["case"])
    expected_cases = tuple(str(row["case"]) for row in fresh_case_records(lock))
    _require(case in expected_cases, "physical case is outside the technical lock")
    processed = args.processed_episode_dir.resolve()
    _require(
        processed.parent.name == physical_seal["object_id"]
        and int(processed.name.removeprefix("episode_"))
        == int(physical_seal["episode_id"]),
        "processed source differs from the physical seal",
    )
    measurement_dir = args.measurement_root.resolve() / case
    measurement_stage_dir = args.measurement_stage_root.resolve() / case
    prediction_dir = args.prediction_root.resolve() / case
    _require(not prediction_dir.exists(), "prediction disposition already exists")
    config = RawCameraObservationConfig()

    def validate(seal: dict[str, object]) -> None:
        validate_fresh_physical_seal(
            seal, case_dir=physical_root, lock=lock, protocol=protocol
        )

    runtime: AllTrackerPrefixRuntime | None = None
    stage_manifest: dict[str, object] | None = None
    try:
        stage_manifest = _stage_measurement_panel(
            processed,
            measurement_stage_dir,
            physical_seal,
            protocol,
        )
        runtime = AllTrackerPrefixRuntime(
            args.alltracker_source.resolve(),
            args.alltracker_checkpoint.resolve(),
            device=args.device,
            config=config,
        )
        measurement = build_raw_camera_measurement_case_with_contract(
            physical_root,
            measurement_stage_dir,
            measurement_dir,
            runtime,
            protocol_id=str(lock["protocol_id"]),
            expected_case_names=expected_cases,
            prediction_seal_validator=validate,
            claim_boundary=(
                "single-object multi-action technical replication; raw RGB "
                "prefix only, with future object outcomes still sealed"
            ),
            config=config,
        )
    except Exception as exc:
        prediction_dir.mkdir(parents=True, exist_ok=False)
        failure_inputs: dict[str, Path] = {
            "physical_seal": physical_seal_path,
            "technical_lock": lock_path,
            "processing_protocol": protocol_path,
            "source_processing": processed / "fresh_pairwise_processing.json",
        }
        stage_manifest_path = measurement_stage_dir / "measurement_panel.json"
        if stage_manifest_path.is_file():
            failure_inputs["measurement_panel"] = stage_manifest_path
        failure = build_fresh_runtime_failure_seal(
            lock_path,
            prediction_dir / FAILURE_SEAL_FILENAME,
            object_id=str(physical_seal["object_id"]),
            episode_id=int(physical_seal["episode_id"]),
            stage="prefix-camera-measurement",
            error_type=type(exc).__name__,
            error_message=str(exc),
            input_files=failure_inputs,
        )
        print(json.dumps(failure, indent=2, sort_keys=True, allow_nan=False))
        return 2
    finally:
        if runtime is not None:
            runtime.close()
    try:
        prediction = build_fresh_guarded_prediction(
            lock_path,
            protocol_path,
            physical_root,
            measurement_dir,
            args.source_qualification.resolve(),
            prediction_dir,
        )
    except Exception as exc:
        prediction_dir.mkdir(parents=True, exist_ok=False)
        failure = build_fresh_runtime_failure_seal(
            lock_path,
            prediction_dir / FAILURE_SEAL_FILENAME,
            object_id=str(physical_seal["object_id"]),
            episode_id=int(physical_seal["episode_id"]),
            stage="guarded-prediction",
            error_type=type(exc).__name__,
            error_message=str(exc),
            input_files={
                "physical_seal": physical_seal_path,
                "measurement_manifest": measurement_dir / "measurement_manifest.json",
                "measurement_archive": measurement_dir / "measurement.npz",
                "source_qualification": args.source_qualification.resolve(),
            },
        )
        print(json.dumps(failure, indent=2, sort_keys=True, allow_nan=False))
        return 2
    print(
        json.dumps(
            {
                "case": case,
                "code_revision": revision,
                "measurement_panel_result_sha256": stage_manifest["result_sha256"],
                "measurement_result_sha256": measurement["result_sha256"],
                "prediction_result_sha256": prediction["result_sha256"],
                "accepted_interval_count": prediction["accepted_interval_count"],
                "information_boundary": {
                    "future_object_rgb_read": False,
                    "future_object_geometry_read": False,
                    "target_metric_read": False,
                    "held_v8_access": False,
                },
            },
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
