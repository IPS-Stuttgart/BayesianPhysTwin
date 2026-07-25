#!/usr/bin/env python3
"""Prepare locked Deform360 calibration and robot state without target access."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
from typing import Any, Mapping

from bayesian_phystwin.deform360_bias_aware_prospective_artifacts import (
    canonical_sha256,
    file_sha256,
    prospective_case_record,
)
from bayesian_phystwin.deform360_bias_aware_prospective_download import (
    bias_aware_prospective_download_plan,
    validate_bias_aware_download_root,
)
from bayesian_phystwin.deform360_bias_aware_prospective_evaluation import (
    validate_bias_aware_calibration_gate,
)
from bayesian_phystwin.deform360_bias_aware_prospective_protocol import (
    DATASET_REVISION,
    PROTOCOL_ID,
    load_bias_aware_prospective_protocol,
)
from deform360 import undistort
from deform360.processing import robot_stage


DEFORM360_REVISION = "0fe36f0b7a7a917ba62b5f8cee707299a9a4a317"
UNDISTORT_SOURCE_SHA256 = (
    "06a500ab2ced8cc960d649d9e200d6d479804ef542ba5aac8fedc5733e74aba9"
)
ROBOT_STAGE_SOURCE_SHA256 = (
    "5944301cc781f179bea96470af50273836a13fdbb367af9a89a59ce1911c11e0"
)
SOURCE_PREPARATION_FILENAME = "bias_aware_source_preparation_manifest.json"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _git_revision(repository: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _require_clean_repository(repository: Path) -> str:
    revision = _git_revision(repository)
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=normal"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    _require(not status.strip(), f"repository is dirty: {repository}")
    return revision


def _validate_download_manifest(
    path: Path,
    *,
    protocol_config_sha256: str,
    object_id: str,
    episode_id: int,
    metadata_path: Path,
) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    _require(
        payload.get("artifact_kind") == "Deform360BiasAwareProspectiveDownload"
        and payload.get("protocol_id") == PROTOCOL_ID
        and payload.get("protocol_config_sha256") == protocol_config_sha256
        and payload.get("revision") == DATASET_REVISION,
        "download manifest is incompatible",
    )
    _require(
        payload.get("manifest_sha256")
        == canonical_sha256(payload, digest_key="manifest_sha256"),
        "download manifest checksum changed",
    )
    rows = [
        row
        for row in payload.get("objects", [])
        if isinstance(row, Mapping) and row.get("object_id") == object_id
    ]
    _require(len(rows) == 1, "download manifest object changed")
    _require(
        episode_id in rows[0].get("selected_episode_ids", []),
        "download manifest omitted the selected episode",
    )
    _require(
        rows[0].get("metadata_sha256") == file_sha256(metadata_path),
        "object metadata changed after download",
    )
    return payload


def _parse_bimanual(metadata_path: Path, episode_id: int) -> bool:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    value = metadata.get("sequences", {}).get(str(episode_id), {}).get("bimanual")
    _require(value in {"yes", "no"}, "released bimanual metadata changed")
    return value == "yes"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--deform360-repo", type=Path, required=True)
    parser.add_argument("--download-root", type=Path, required=True)
    parser.add_argument("--download-manifest", type=Path, required=True)
    parser.add_argument("--aligned-root", type=Path, required=True)
    parser.add_argument("--object-id", required=True)
    parser.add_argument("--episode-id", type=int, required=True)
    parser.add_argument("--calibration-gate", type=Path)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    protocol_path = args.protocol.resolve()
    protocol = load_bias_aware_prospective_protocol(protocol_path)
    record = prospective_case_record(
        protocol_path, object_id=args.object_id, episode_id=args.episode_id
    )
    calibration_gate: dict[str, Any] | None = None
    calibration_gate_path: Path | None = None
    if record["role"] == "target":
        _require(
            args.calibration_gate is not None, "target access needs calibration gate"
        )
        calibration_gate_path = args.calibration_gate.resolve()
        calibration_gate = json.loads(calibration_gate_path.read_text(encoding="utf-8"))
        validate_bias_aware_calibration_gate(
            calibration_gate, protocol_path=protocol_path, require_passed=True
        )
    else:
        _require(
            args.calibration_gate is None, "calibration must not consume target gate"
        )
    code_revision = _require_clean_repository(args.repo.resolve())
    deform360_repo = args.deform360_repo.resolve()
    _require(
        _git_revision(deform360_repo) == DEFORM360_REVISION,
        "Deform360 revision changed",
    )
    sources = {
        "undistort": deform360_repo / "deform360" / "undistort.py",
        "robot_stage": deform360_repo / "deform360" / "processing" / "robot_stage.py",
    }
    _require(
        file_sha256(sources["undistort"]) == UNDISTORT_SOURCE_SHA256,
        "official undistort source changed",
    )
    _require(
        file_sha256(sources["robot_stage"]) == ROBOT_STAGE_SOURCE_SHA256,
        "official robot source changed",
    )
    download_root = args.download_root.resolve()
    plan = bias_aware_prospective_download_plan(protocol_path)
    validate_bias_aware_download_root(download_root, plan=plan, require_complete=True)
    raw_object = download_root / "raw" / args.object_id
    metadata_path = raw_object / "metadata.json"
    _validate_download_manifest(
        args.download_manifest.resolve(),
        protocol_config_sha256=str(protocol["config_sha256"]),
        object_id=args.object_id,
        episode_id=args.episode_id,
        metadata_path=metadata_path,
    )
    bimanual = _parse_bimanual(metadata_path, args.episode_id)

    aligned_object = args.aligned_root.resolve() / args.object_id
    episode_dir = undistort.undistort_episode(
        raw_object,
        aligned_object,
        args.episode_id,
        overwrite=False,
        rebuild_timeline=False,
    )
    robot_path = robot_stage.process_robot_episode(
        aligned_object,
        args.episode_id,
        bimanual=bimanual,
        seed=0,
        overwrite=False,
        plot=False,
    )
    alignment_path = episode_dir / "alignment.json"
    alignment = json.loads(alignment_path.read_text(encoding="utf-8"))
    cameras = alignment.get("cameras")
    _require(isinstance(cameras, list) and len(cameras) >= 8, "camera alignment failed")
    frame_count = alignment.get("frame_count")
    _require(isinstance(frame_count, int) and frame_count >= 81, "episode is too short")
    manifest_path = episode_dir / SOURCE_PREPARATION_FILENAME
    _require(not manifest_path.exists(), "source preparation is already sealed")
    robot_metadata = robot_path.with_name("robot.meta.json")
    payload: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": "Deform360BiasAwareSourcePreparation",
        "protocol_id": PROTOCOL_ID,
        "protocol_config_sha256": protocol["config_sha256"],
        **record,
        "dataset_revision": DATASET_REVISION,
        "code_revision": code_revision,
        "deform360_revision": DEFORM360_REVISION,
        "bimanual": bimanual,
        "camera_count": len(cameras),
        "cameras": cameras,
        "aligned_frame_count": frame_count,
        "inputs_sha256": {
            "protocol": file_sha256(protocol_path),
            "download_manifest": file_sha256(args.download_manifest.resolve()),
            "object_metadata": file_sha256(metadata_path),
            "official_undistort_source": file_sha256(sources["undistort"]),
            "official_robot_stage_source": file_sha256(sources["robot_stage"]),
            "calibration_gate": (
                None
                if calibration_gate_path is None
                else file_sha256(calibration_gate_path)
            ),
        },
        "target_access_authorization": (
            None
            if calibration_gate is None
            else {
                "calibration_gate_result_sha256": calibration_gate["result_sha256"],
                "target_access_authorized": True,
            }
        ),
        "outputs_sha256": {
            "alignment": file_sha256(alignment_path),
            "undistorted_intrinsics": file_sha256(
                episode_dir / "undistorted_intrinsics.npy"
            ),
            "extrinsics": file_sha256(episode_dir / "extrinsics.npy"),
            "robot": file_sha256(robot_path),
            "robot_metadata": file_sha256(robot_metadata),
            "camera_metadata": {
                camera: file_sha256(episode_dir / camera / "metadata.json")
                for camera in cameras
            },
        },
        "information_boundary": {
            "stage_role": "source-data custodian",
            "full_rgb_decoded_for_camera_alignment": True,
            "full_rgb_decoded_for_released_robot_pose_recovery": True,
            "object_mask_created": False,
            "object_geometry_created_or_read": False,
            "future_particle_tracks_created_or_read": False,
            "tactile_created_or_read": False,
            "target_metric_created_or_read": False,
            "prediction_process_receives_only_separately_staged_prefix": True,
            "target_access_gate_verified": record["role"] == "target",
        },
    }
    payload["result_sha256"] = canonical_sha256(payload, digest_key="result_sha256")
    manifest_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
