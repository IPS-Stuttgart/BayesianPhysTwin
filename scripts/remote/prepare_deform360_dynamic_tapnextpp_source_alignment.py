#!/usr/bin/env python3
"""Recover alignment and robot state for one dynamic TAPNext++ source."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
from typing import Any, Mapping

from bayesian_phystwin.deform360_fresh_source_download import (
    fresh_source_download_plan,
    validate_fresh_download_root,
)
from bayesian_phystwin.deform360_dynamic_tapnextpp_source_window import (
    canonical_sha256,
    dynamic_source_case,
    file_sha256,
    validate_dynamic_window_sources,
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
SOURCE_PREPARATION_FILENAME = "dynamic_tapnextpp_source_preparation.json"


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
    _require(not status.strip(), f"repository has uncommitted files: {repository}")
    return revision


def _parse_bimanual(metadata_path: Path, episode_id: int) -> bool:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    sequence = metadata.get("sequences", {}).get(str(episode_id), {})
    value = sequence.get("bimanual")
    _require(value in {"yes", "no"}, "released bimanual metadata changed")
    return value == "yes"


def _download_row(
    download: Mapping[str, Any], object_id: str, episode_id: int
) -> Mapping[str, Any]:
    matches = [
        row
        for row in download.get("objects", ())
        if isinstance(row, Mapping)
        and row.get("object_id") == object_id
        and row.get("episode_id") == episode_id
    ]
    _require(len(matches) == 1, "download manifest case identity changed")
    return matches[0]


def _validate_object_inventory(
    object_root: Path,
    row: Mapping[str, Any],
) -> None:
    files = [path for path in object_root.rglob("*") if path.is_file()]
    _require(
        len(files) == row.get("file_count")
        and sum(path.stat().st_size for path in files) == row.get("total_bytes"),
        "downloaded object inventory changed",
    )
    _require(
        file_sha256(object_root / "metadata.json") == row.get("metadata_sha256"),
        "downloaded object metadata changed",
    )


def _validate_existing(
    manifest: Mapping[str, Any],
    *,
    protocol_sha256: str,
    case: Mapping[str, Any],
    code_revision: str,
    episode_dir: Path,
) -> None:
    _require(
        manifest.get("artifact_kind") == "Deform360DynamicTapNextppSourcePreparation"
        and manifest.get("protocol_config_sha256") == protocol_sha256
        and manifest.get("code_revision") == code_revision
        and manifest.get("result_sha256")
        == canonical_sha256(manifest, digest_key="result_sha256")
        and all(manifest.get(key) == value for key, value in case.items()),
        "existing fresh source preparation changed",
    )
    outputs = manifest.get("outputs_sha256")
    _require(isinstance(outputs, Mapping), "preparation output hashes are missing")
    paths = {
        "alignment": episode_dir / "alignment.json",
        "undistorted_intrinsics": episode_dir / "undistorted_intrinsics.npy",
        "extrinsics": episode_dir / "extrinsics.npy",
        "robot": episode_dir / "robot" / "robot.npz",
        "robot_metadata": episode_dir / "robot" / "robot.meta.json",
    }
    _require(
        all(
            path.is_file() and file_sha256(path) == outputs.get(name)
            for name, path in paths.items()
        ),
        "existing fresh source preparation outputs changed",
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--deform360-repo", type=Path, required=True)
    parser.add_argument("--download-root", type=Path, required=True)
    parser.add_argument("--download-manifest", type=Path, required=True)
    parser.add_argument("--aligned-root", type=Path, required=True)
    parser.add_argument("--object-id", required=True)
    parser.add_argument("--episode-id", type=int, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    protocol, queue, download = validate_dynamic_window_sources(
        args.protocol.resolve(),
        args.queue.resolve(),
        args.download_manifest.resolve(),
    )
    case = dynamic_source_case(queue, args.object_id, args.episode_id)
    code_revision = _require_clean_repository(args.repo.resolve())

    deform360_repo = args.deform360_repo.resolve()
    _require(
        _git_revision(deform360_repo) == DEFORM360_REVISION,
        "Deform360 revision changed",
    )
    sources = {
        "undistort": deform360_repo / "deform360" / "undistort.py",
        "robot_stage": (deform360_repo / "deform360" / "processing" / "robot_stage.py"),
    }
    _require(
        file_sha256(sources["undistort"]) == UNDISTORT_SOURCE_SHA256,
        "official undistort source changed",
    )
    _require(
        file_sha256(sources["robot_stage"]) == ROBOT_STAGE_SOURCE_SHA256,
        "official robot-stage source changed",
    )

    download_root = args.download_root.resolve()
    plan = fresh_source_download_plan(args.queue.resolve())
    validate_fresh_download_root(download_root, plan=plan, require_complete=True)
    raw_object = download_root / "raw" / args.object_id
    row = _download_row(download, args.object_id, args.episode_id)
    _validate_object_inventory(raw_object, row)
    metadata_path = raw_object / "metadata.json"
    bimanual = _parse_bimanual(metadata_path, args.episode_id)

    aligned_object = args.aligned_root.resolve() / args.object_id
    episode_dir = aligned_object / f"episode_{args.episode_id:04d}"
    manifest_path = episode_dir / SOURCE_PREPARATION_FILENAME
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        _validate_existing(
            manifest,
            protocol_sha256=str(protocol["config_sha256"]),
            case=case,
            code_revision=code_revision,
            episode_dir=episode_dir,
        )
        print(json.dumps(manifest, indent=2, sort_keys=True))
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
    _require(
        isinstance(cameras, list) and len(cameras) >= 12,
        "aligned camera panel is incomplete",
    )
    frame_count = alignment.get("frame_count")
    _require(
        isinstance(frame_count, int) and frame_count >= 81,
        "aligned episode is too short",
    )

    camera_hashes = {
        camera: file_sha256(episode_dir / camera / "metadata.json")
        for camera in cameras
    }
    output_hashes = {
        "alignment": file_sha256(alignment_path),
        "undistorted_intrinsics": file_sha256(
            episode_dir / "undistorted_intrinsics.npy"
        ),
        "extrinsics": file_sha256(episode_dir / "extrinsics.npy"),
        "robot": file_sha256(robot_path),
        "robot_metadata": file_sha256(robot_path.with_name("robot.meta.json")),
        "camera_metadata": camera_hashes,
    }
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": "Deform360DynamicTapNextppSourcePreparation",
        "protocol_id": protocol["protocol_id"],
        "protocol_config_sha256": protocol["config_sha256"],
        **case,
        "dataset_revision": download["revision"],
        "code_revision": code_revision,
        "deform360_revision": DEFORM360_REVISION,
        "bimanual": bimanual,
        "camera_count": len(cameras),
        "cameras": cameras,
        "aligned_frame_count": frame_count,
        "inputs_sha256": {
            "protocol": file_sha256(args.protocol.resolve()),
            "queue": file_sha256(args.queue.resolve()),
            "download_manifest": file_sha256(args.download_manifest.resolve()),
            "object_metadata": file_sha256(metadata_path),
            "official_undistort_source": file_sha256(sources["undistort"]),
            "official_robot_stage_source": file_sha256(sources["robot_stage"]),
        },
        "outputs_sha256": output_hashes,
        "information_boundary": {
            "stage_role": "source-data custodian",
            "full_rgb_decoded_for_camera_alignment": True,
            "full_rgb_decoded_for_released_robot_pose_recovery": True,
            "object_mask_created": False,
            "object_geometry_created_or_read": False,
            "future_particle_tracks_created_or_read": False,
            "tactile_created_or_read": False,
            "target_metric_created_or_read": False,
        },
    }
    manifest["result_sha256"] = canonical_sha256(manifest, digest_key="result_sha256")
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
