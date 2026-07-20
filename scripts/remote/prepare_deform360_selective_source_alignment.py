#!/usr/bin/env python3
"""Prepare locked Deform360 calibration and robot state without target access."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any, Mapping

from bayesian_phystwin.deform360_selective_virtual_sensing_artifacts import (
    selective_case_records,
)
from bayesian_phystwin.deform360_selective_virtual_sensing_download import (
    selective_virtual_sensing_download_plan,
    validate_selective_download_root,
)
from bayesian_phystwin.deform360_selective_virtual_sensing_protocol import (
    DATASET_REVISION,
    PROTOCOL_ID,
    load_selective_virtual_sensing_protocol,
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
SOURCE_PREPARATION_FILENAME = "selective_source_preparation_manifest.json"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(payload: Mapping[str, Any], digest_key: str) -> str:
    unsigned = dict(payload)
    unsigned.pop(digest_key, None)
    encoded = json.dumps(
        unsigned, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


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
    _require(not status.strip(), f"repository has uncommitted files: {repository}")
    return revision


def _case_record(
    protocol_path: Path, object_id: str, episode_id: int
) -> dict[str, Any]:
    matches = [
        record
        for record in selective_case_records(protocol_path)
        if record["object_id"] == object_id
        and record["episode_id"] == episode_id
    ]
    _require(len(matches) == 1, "case is outside the locked prospective panel")
    return matches[0]


def _validate_download_manifest(
    manifest_path: Path,
    *,
    protocol_config_sha256: str,
    object_id: str,
    episode_id: int,
    metadata_path: Path,
) -> dict[str, Any]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    _require(
        payload.get("artifact_kind")
        == "Deform360SelectiveVirtualSensingDownload"
        and payload.get("protocol_id") == PROTOCOL_ID
        and payload.get("protocol_config_sha256") == protocol_config_sha256
        and payload.get("revision") == DATASET_REVISION,
        "download manifest is incompatible",
    )
    _require(
        payload.get("manifest_sha256")
        == _canonical_sha256(payload, "manifest_sha256"),
        "download manifest content checksum changed",
    )
    rows = [
        row
        for row in payload.get("objects", [])
        if isinstance(row, Mapping) and row.get("object_id") == object_id
    ]
    _require(len(rows) == 1, "download manifest object identity changed")
    row = rows[0]
    _require(
        episode_id in row.get("selected_episode_ids", []),
        "download manifest omitted the locked episode",
    )
    _require(
        row.get("metadata_sha256") == _sha256(metadata_path),
        "downloaded object metadata changed",
    )
    return payload


def _parse_bimanual(metadata_path: Path, episode_id: int) -> bool:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    sequence = metadata.get("sequences", {}).get(str(episode_id), {})
    value = sequence.get("bimanual")
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
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    protocol_path = args.protocol.resolve()
    protocol = load_selective_virtual_sensing_protocol(protocol_path)
    record = _case_record(protocol_path, args.object_id, args.episode_id)
    code_revision = _require_clean_repository(args.repo.resolve())

    deform360_repo = args.deform360_repo.resolve()
    _require(
        _git_revision(deform360_repo) == DEFORM360_REVISION,
        "Deform360 revision changed",
    )
    sources = {
        "undistort": deform360_repo / "deform360" / "undistort.py",
        "robot_stage": (
            deform360_repo / "deform360" / "processing" / "robot_stage.py"
        ),
    }
    _require(
        _sha256(sources["undistort"]) == UNDISTORT_SOURCE_SHA256,
        "official undistort source changed",
    )
    _require(
        _sha256(sources["robot_stage"]) == ROBOT_STAGE_SOURCE_SHA256,
        "official robot-stage source changed",
    )

    download_root = args.download_root.resolve()
    plan = selective_virtual_sensing_download_plan(protocol_path)
    validate_selective_download_root(
        download_root, plan=plan, require_complete=True
    )
    raw_object = download_root / "raw" / args.object_id
    metadata_path = raw_object / "metadata.json"
    download_manifest_path = args.download_manifest.resolve()
    _validate_download_manifest(
        download_manifest_path,
        protocol_config_sha256=str(protocol["config_sha256"]),
        object_id=args.object_id,
        episode_id=args.episode_id,
        metadata_path=metadata_path,
    )
    bimanual = _parse_bimanual(metadata_path, args.episode_id)

    aligned_object = args.aligned_root.resolve() / args.object_id
    existing_episode = aligned_object / f"episode_{args.episode_id:04d}"
    existing_manifest_path = existing_episode / SOURCE_PREPARATION_FILENAME
    if existing_manifest_path.is_file():
        existing = json.loads(
            existing_manifest_path.read_text(encoding="utf-8")
        )
        _require(
            existing.get("artifact_kind")
            == "Deform360SelectiveSourcePreparation"
            and existing.get("protocol_id") == PROTOCOL_ID
            and existing.get("protocol_config_sha256")
            == protocol["config_sha256"]
            and existing.get("code_revision") == code_revision
            and existing.get("result_sha256")
            == _canonical_sha256(existing, "result_sha256")
            and all(existing.get(key) == value for key, value in record.items()),
            "existing source-preparation manifest changed",
        )
        outputs = existing["outputs_sha256"]
        fixed_paths = {
            "alignment": existing_episode / "alignment.json",
            "undistorted_intrinsics": (
                existing_episode / "undistorted_intrinsics.npy"
            ),
            "extrinsics": existing_episode / "extrinsics.npy",
            "robot": existing_episode / "robot" / "robot.npz",
            "robot_metadata": existing_episode / "robot" / "robot.meta.json",
        }
        _require(
            all(
                path.is_file() and _sha256(path) == outputs[name]
                for name, path in fixed_paths.items()
            )
            and all(
                _sha256(existing_episode / camera / "metadata.json") == digest
                for camera, digest in outputs["camera_metadata"].items()
            ),
            "existing source-preparation outputs changed",
        )
        print(json.dumps(existing, indent=2, sort_keys=True))
        return 0
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
    _require(isinstance(cameras, list) and len(cameras) >= 8, "aligned camera panel failed")
    frame_count = alignment.get("frame_count")
    _require(isinstance(frame_count, int) and frame_count >= 81, "episode is too short")

    manifest_path = episode_dir / SOURCE_PREPARATION_FILENAME
    _require(not manifest_path.exists(), "source-preparation manifest already exists")
    robot_meta = robot_path.with_name("robot.meta.json")
    output_hashes = {
        "alignment": _sha256(alignment_path),
        "undistorted_intrinsics": _sha256(
            episode_dir / "undistorted_intrinsics.npy"
        ),
        "extrinsics": _sha256(episode_dir / "extrinsics.npy"),
        "robot": _sha256(robot_path),
        "robot_metadata": _sha256(robot_meta),
        "camera_metadata": {
            camera: _sha256(episode_dir / camera / "metadata.json")
            for camera in cameras
        },
    }
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": "Deform360SelectiveSourcePreparation",
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
            "protocol": _sha256(protocol_path),
            "download_manifest": _sha256(download_manifest_path),
            "object_metadata": _sha256(metadata_path),
            "official_undistort_source": _sha256(sources["undistort"]),
            "official_robot_stage_source": _sha256(sources["robot_stage"]),
        },
        "outputs_sha256": output_hashes,
        "information_boundary": {
            "stage_role": "independent source-data custodian",
            "full_rgb_decoded_for_camera_alignment": True,
            "full_rgb_decoded_for_released_robot_pose_recovery": True,
            "object_mask_created": False,
            "object_geometry_created_or_read": False,
            "future_particle_tracks_created_or_read": False,
            "tactile_created_or_read": False,
            "target_metric_created_or_read": False,
            "prediction_process_receives_only_separately_staged_prefix": True,
        },
    }
    manifest["result_sha256"] = _canonical_sha256(manifest, "result_sha256")
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
