#!/usr/bin/env python3
"""Seal one prospective V14 source prediction from the permitted prefix."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from bayesian_phystwin.deform360_causal_response_admission import (
    CausalResponseAdmissionConfig,
)
from bayesian_phystwin.deform360_causal_response_direct_depth import (
    predict_adaptive_direct_depth_v14,
    scan_adaptive_direct_depth_v14,
    write_adaptive_direct_depth_v14_artifacts,
)
from bayesian_phystwin.deform360_causal_response_direct_depth_admission_v14 import (
    ADMISSION_REPORT_FILENAME,
    PREFLIGHT_FILENAME,
    aggregate_source_sha256,
)
from bayesian_phystwin.deform360_causal_response_direct_depth_physical import (
    PHYSICAL_ARCHIVE_FILENAME,
    PHYSICAL_MANIFEST_FILENAME,
    validate_v14_physical_artifacts,
)
from bayesian_phystwin.deform360_causal_response_direct_depth_prediction_v14 import (
    PREFIX_FRAME_COUNT,
    build_v14_prefix_inputs,
    load_v14_admitted_carrier,
    load_v14_prediction_runtime,
    tactile_source_sha256,
    v14_prediction_case_record,
)
from bayesian_phystwin.deform360_causal_response_direct_depth_preflight import (
    load_adaptive_direct_depth_source_preflight_v14,
)
from bayesian_phystwin.deform360_causal_response_direct_depth_source_lock import (
    validate_adaptive_direct_depth_source_lock_v14,
)
from bayesian_phystwin.deform360_causal_response_event import (
    CausalResponseEventConfig,
)
from bayesian_phystwin.deform360_causal_response_prefix import (
    ARCHIVE_FILENAME as PREFIX_ARCHIVE_FILENAME,
)
from bayesian_phystwin.deform360_causal_response_prefix import (
    REPORT_FILENAME as PREFIX_REPORT_FILENAME,
)
from bayesian_phystwin.deform360_causal_response_prefix import (
    write_causal_response_prefix_artifacts,
)
from bayesian_phystwin.deform360_causal_response_update import (
    CausalResponseMeasurementConfig,
)
from bayesian_phystwin.deform360_direct_depth_provider import (
    DirectDepthEndpointConfig,
)
from bayesian_phystwin.deform360_object_exclusion import file_sha256
from bayesian_phystwin.phystwin_online_belief import RecursiveRbfBeliefConfig

PREFIX_DIRECTORY = "prefix"
PREDICTION_DIRECTORY = "prediction"
DISPOSITION_FILENAME = "prediction_disposition_v14.json"


def _require(condition: bool | np.bool_, message: str) -> None:
    if not bool(condition):
        raise ValueError(message)


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    canonical = dict(payload)
    canonical.pop("artifact_sha256", None)
    return hashlib.sha256(
        b"deform360-causal-response-direct-depth-prediction-disposition-v14\0"
        + json.dumps(
            canonical,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _git_output(repository: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _require_clean_repository(repository: Path) -> str:
    revision = _git_output(repository, "rev-parse", "HEAD")
    _require(
        not _git_output(
            repository, "status", "--porcelain", "--untracked-files=normal"
        ),
        "V14 prediction repository is dirty",
    )
    return revision


def _load_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    _require(isinstance(payload, dict), f"JSON artifact is not an object: {path}")
    return payload


def _load_calibration(path: Path) -> dict[str, np.ndarray]:
    stored = np.load(path, allow_pickle=True)
    _require(stored.shape == (), f"V14 calibration is not a dictionary: {path}")
    mapping = stored.item()
    _require(isinstance(mapping, dict), f"V14 calibration is invalid: {path}")
    return {str(name): np.asarray(value) for name, value in mapping.items()}


def _read_h5_prefix(path: Path, frame_count: int) -> np.ndarray:
    with h5py.File(path, "r") as stream:
        _require(
            "data" in stream
            and stream["data"].ndim == 3
            and len(stream["data"]) >= frame_count,
            f"V14 causal HDF5 stream is incomplete: {path}",
        )
        return np.asarray(stream["data"][:frame_count])


def _load_camera_prefix(
    processed: Path,
    camera_ids: tuple[str, ...],
    *,
    preflight: Any,
    depth_scale_to_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    intrinsics_path = processed / "undistorted_intrinsics.npy"
    extrinsics_path = processed / "extrinsics.npy"
    intrinsics_by_camera = _load_calibration(intrinsics_path)
    poses_by_camera = _load_calibration(extrinsics_path)
    depths: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    intrinsics: list[np.ndarray] = []
    poses: list[np.ndarray] = []
    for camera in camera_ids:
        _require(
            camera in intrinsics_by_camera and camera in poses_by_camera,
            f"V14 causal camera lacks calibration: {camera}",
        )
        depth_path = processed / camera / "rendered_depth.h5"
        mask_path = processed / camera / "mask_refined.h5"
        _require(
            file_sha256(depth_path) == preflight.source_sha256[f"depth/{camera}"]
            and file_sha256(mask_path) == preflight.source_sha256[f"mask/{camera}"],
            f"V14 causal camera source changed: {camera}",
        )
        calibration_digest = aggregate_source_sha256(
            f"calibration/{camera}",
            {
                "extrinsics": file_sha256(extrinsics_path),
                "intrinsics": file_sha256(intrinsics_path),
            },
        )
        _require(
            calibration_digest == preflight.source_sha256[f"calibration/{camera}"],
            f"V14 causal camera calibration changed: {camera}",
        )
        encoded = _read_h5_prefix(depth_path, PREFIX_FRAME_COUNT)
        mask = _read_h5_prefix(mask_path, PREFIX_FRAME_COUNT).astype(
            bool,
            copy=False,
        )
        _require(
            encoded.shape == mask.shape,
            f"V14 causal depth and mask differ: {camera}",
        )
        depths.append(encoded.astype(np.float32) * depth_scale_to_m)
        masks.append(mask)
        intrinsics.append(np.asarray(intrinsics_by_camera[camera], dtype=np.float64))
        poses.append(np.asarray(poses_by_camera[camera], dtype=np.float64))
    _require(
        len({values.shape for values in depths}) == 1,
        "V14 causal camera prefix shapes differ",
    )
    return (
        np.stack(intrinsics),
        np.stack(poses),
        np.stack(depths),
        np.stack(masks),
    )


def _load_tactile_prefix(
    staged: Path,
    *,
    preflight: Any,
) -> tuple[list[np.ndarray], dict[str, Path]]:
    paths = {
        directory.name: directory / "synced_tactile.npy"
        for directory in sorted(staged.iterdir())
        if directory.is_dir() and (directory / "synced_tactile.npy").is_file()
    }
    _require(
        tactile_source_sha256(paths) == preflight.source_sha256["tactile"],
        "V14 tactile source set changed after preflight",
    )
    values: list[np.ndarray] = []
    for sensor, path in paths.items():
        metadata = _load_json(path.parent / "metadata.json")
        normalization = metadata.get("normalization", {})
        _require(
            metadata.get("schema") == "deform360.tactile/v1"
            and metadata.get("values") == "unitless episode-peak-relative response"
            and normalization.get("baseline_subtracted") is True
            and float(normalization.get("threshold")) == 0.3,
            f"V14 tactile normalization changed: {sensor}",
        )
        values.append(np.load(path, allow_pickle=False))
    return values, paths


def _load_robot_actions(path: Path, *, preflight: Any) -> np.ndarray:
    _require(
        file_sha256(path) == preflight.source_sha256["robot"],
        "V14 measured robot source changed after preflight",
    )
    with np.load(path, allow_pickle=False) as stored:
        _require("actions" in stored.files, "V14 robot archive lacks actions")
        return np.asarray(stored["actions"])


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--prediction-runtime", type=Path, required=True)
    parser.add_argument("--method-protocol", type=Path, required=True)
    parser.add_argument("--source-lock", type=Path, required=True)
    parser.add_argument("--admission-prelock", type=Path, required=True)
    parser.add_argument("--physical-prelock", type=Path, required=True)
    parser.add_argument("--queue-rank", type=int, required=True)
    parser.add_argument("--admission-dir", type=Path, required=True)
    parser.add_argument("--physical-dir", type=Path, required=True)
    parser.add_argument("--processed-episode-dir", type=Path, required=True)
    parser.add_argument("--staged-episode-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    repository = args.repo.resolve()
    revision = _require_clean_repository(repository)
    method_path = args.method_protocol.resolve()
    method = _load_json(method_path)
    source_lock_path = args.source_lock.resolve()
    source_lock = validate_adaptive_direct_depth_source_lock_v14(source_lock_path)
    runtime_path = args.prediction_runtime.resolve()
    runtime = load_v14_prediction_runtime(
        runtime_path,
        method_protocol_path=method_path,
        source_lock_path=source_lock_path,
        admission_prelock_path=args.admission_prelock.resolve(),
        physical_prelock_path=args.physical_prelock.resolve(),
    )
    implementation_paths = {
        "prediction_module": (
            repository / "src/bayesian_phystwin/"
            "deform360_causal_response_direct_depth_prediction_v14.py"
        ),
        "prediction_runner": Path(__file__).resolve(),
        "preflight_module": (
            repository / "src/bayesian_phystwin/"
            "deform360_causal_response_direct_depth_preflight.py"
        ),
        "runtime_builder": (
            repository / "scripts/remote/"
            "prepare_deform360_causal_response_direct_depth_v14_prediction_runtime.py"
        ),
    }
    _require(
        all(
            file_sha256(path) == runtime["implementation"]["file_sha256"][name]
            for name, path in implementation_paths.items()
        )
        and _git_output(
            repository,
            "merge-base",
            "--is-ancestor",
            runtime["implementation"]["parent_commit"],
            revision,
        )
        == ""
        and _git_output(
            repository,
            "merge-base",
            "--is-ancestor",
            source_lock.repository_revision,
            revision,
        )
        == "",
        "V14 prediction implementation or ancestry changed",
    )
    _require(
        method.get("protocol_id") == "deform360-causal-response-direct-depth-v14-source"
        and method.get("config_sha256") == source_lock.method_config_sha256
        and all(
            file_sha256(repository / relative) == digest
            for relative, digest in method["implementation_file_sha256"].items()
            if relative
            != (
                "src/bayesian_phystwin/"
                "deform360_causal_response_direct_depth_preflight.py"
            )
        ),
        "V14 frozen estimator implementation changed",
    )
    runtime_case, locked_case = v14_prediction_case_record(
        runtime,
        source_lock,
        queue_rank=args.queue_rank,
    )

    admission_dir = args.admission_dir.resolve()
    admission, carrier = load_v14_admitted_carrier(admission_dir)
    _require(
        admission["queue_rank"] == args.queue_rank
        and admission["case_hash"] == locked_case.case_hash
        and admission["object_hash"] == locked_case.object_hash
        and admission["artifact_sha256"] == runtime_case["admission_artifact_sha256"]
        and file_sha256(admission_dir / ADMISSION_REPORT_FILENAME)
        == runtime_case["admission_file_sha256"],
        "V14 prediction admission differs from the runtime ledger",
    )
    preflight = load_adaptive_direct_depth_source_preflight_v14(
        admission_dir / PREFLIGHT_FILENAME
    )
    _require(
        preflight.admitted
        and preflight.artifact_sha256 == locked_case.source_preflight_sha256
        and preflight.carrier_artifact_sha256 == locked_case.carrier_artifact_sha256,
        "V14 prediction preflight differs from the source lock",
    )

    physical_dir = args.physical_dir.resolve()
    physical_manifest, physical = validate_v14_physical_artifacts(
        physical_dir,
        prelock_protocol_path=args.physical_prelock.resolve(),
    )
    _require(
        physical_manifest["case_hash"] == locked_case.case_hash
        and physical_manifest["object_hash"] == locked_case.object_hash
        and physical_manifest["artifact_sha256"]
        == runtime_case["physical_artifact_sha256"]
        and file_sha256(physical_dir / PHYSICAL_MANIFEST_FILENAME)
        == runtime_case["physical_manifest_file_sha256"]
        and file_sha256(physical_dir / PHYSICAL_ARCHIVE_FILENAME)
        == runtime_case["physical_archive_file_sha256"],
        "V14 prediction physical carrier differs from the runtime ledger",
    )

    processed = args.processed_episode_dir.resolve()
    intrinsics, poses, depths, masks = _load_camera_prefix(
        processed,
        carrier.available_camera_ids,
        preflight=preflight,
        depth_scale_to_m=float(runtime["numerical_contract"]["depth_scale_to_m"]),
    )
    staged = args.staged_episode_dir.resolve()
    tactile, tactile_paths = _load_tactile_prefix(
        staged,
        preflight=preflight,
    )
    robot_path = staged / "robot" / "robot.npz"
    robot_actions = _load_robot_actions(robot_path, preflight=preflight)
    prefix = build_v14_prefix_inputs(
        camera_ids=carrier.available_camera_ids,
        intrinsics=intrinsics,
        camera_to_world=poses,
        depths_m=depths,
        object_masks=masks,
        tactile_sensor_arrays=tactile,
        robot_actions=robot_actions,
    )

    output = args.output_dir.resolve()
    _require(not output.exists(), "V14 prediction output already exists")
    scratch = output.with_name(f".{output.name}.incomplete-{os.getpid()}")
    _require(not scratch.exists(), "V14 prediction scratch already exists")
    scratch.mkdir(parents=True)
    prefix_report = write_causal_response_prefix_artifacts(
        scratch / PREFIX_DIRECTORY,
        prefix,
        case_id=locked_case.case_hash,
        protocol_path=method_path,
        source_sha256={
            **{
                f"depth/{camera}": preflight.source_sha256[f"depth/{camera}"]
                for camera in carrier.available_camera_ids
            },
            **{
                f"mask/{camera}": preflight.source_sha256[f"mask/{camera}"]
                for camera in carrier.available_camera_ids
            },
            "robot": preflight.source_sha256["robot"],
            "tactile": preflight.source_sha256["tactile"],
        },
    )
    scan = scan_adaptive_direct_depth_v14(
        locked_case.case_hash,
        physical["physical_prediction_m"],
        carrier,
        prefix.intrinsics,
        prefix.camera_to_world,
        prefix.depths_m,
        prefix.object_masks,
        physical["action_support"],
        prefix.tactile_contact_probability,
        prefix.measured_actuator_positions_m,
        persistence_prediction_m=physical["persistence_prediction_m"],
        event_config=CausalResponseEventConfig(**method["event"]),
        depth_config=DirectDepthEndpointConfig(
            **method["direct_depth_arms"]["strict_3plus3"]
        ),
        admission_config=CausalResponseAdmissionConfig(**method["admission"]),
    )
    candidate_report, candidate_arrays = predict_adaptive_direct_depth_v14(
        physical["physical_prediction_m"],
        scan,
        persistence_prediction_m=physical["persistence_prediction_m"],
        measurement_config=CausalResponseMeasurementConfig(**method["measurement"]),
        belief_config=RecursiveRbfBeliefConfig(**method["belief"]),
    )
    prediction_report = write_adaptive_direct_depth_v14_artifacts(
        scratch / PREDICTION_DIRECTORY,
        carrier,
        scan,
        candidate_report,
        candidate_arrays,
        case_id=locked_case.case_hash,
        repository_revision=revision,
        protocol_path=method_path,
        input_sha256={
            "admission_report": file_sha256(admission_dir / ADMISSION_REPORT_FILENAME),
            "physical_archive": file_sha256(physical_dir / PHYSICAL_ARCHIVE_FILENAME),
            "physical_manifest": file_sha256(physical_dir / PHYSICAL_MANIFEST_FILENAME),
            "prediction_runtime": file_sha256(runtime_path),
            "prefix_archive": file_sha256(
                scratch / PREFIX_DIRECTORY / PREFIX_ARCHIVE_FILENAME
            ),
            "prefix_report": file_sha256(
                scratch / PREFIX_DIRECTORY / PREFIX_REPORT_FILENAME
            ),
            "source_lock": file_sha256(source_lock_path),
        },
    )
    disposition: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": "Deform360CausalResponseDirectDepthPredictionDispositionV14",
        "contract": (
            "deform360-causal-response-direct-depth-prediction-disposition-v14"
        ),
        "queue_rank": int(args.queue_rank),
        "object_hash": locked_case.object_hash,
        "case_hash": locked_case.case_hash,
        "status": prediction_report["status"],
        "repository_revision": revision,
        "source_lock_artifact_sha256": source_lock.artifact_sha256,
        "prediction_runtime_config_sha256": runtime["config_sha256"],
        "admission_artifact_sha256": admission["artifact_sha256"],
        "physical_artifact_sha256": physical_manifest["artifact_sha256"],
        "prefix_result_sha256": prefix_report["result_sha256"],
        "prediction_result_sha256": prediction_report["result_sha256"],
        "tactile_sensor_count": len(tactile_paths),
        "maximum_object_observation_frame": scan.scan.maximum_observation_frame,
        "event_admitted": scan.scan.admitted,
        "selected_backbone": scan.scan.selected_backbone,
        "candidate_applied": bool(candidate_report["candidate_applied"]),
        "information_boundary": {
            "maximum_object_observation_frame": scan.scan.maximum_observation_frame,
            "future_object_observation_read": False,
            "future_identity_or_metric_read": False,
            "source_outcome_read": False,
            "target_object_or_outcome_read": False,
            "held_v8_artifact_or_process_access": False,
        },
    }
    disposition["artifact_sha256"] = _canonical_sha256(disposition)
    (scratch / DISPOSITION_FILENAME).write_text(
        json.dumps(disposition, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    scratch.rename(output)
    print(
        json.dumps(
            {
                "queue_rank": args.queue_rank,
                "status": disposition["status"],
                "event_admitted": disposition["event_admitted"],
                "selected_backbone": disposition["selected_backbone"],
                "candidate_applied": disposition["candidate_applied"],
                "prediction_result_sha256": disposition["prediction_result_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
