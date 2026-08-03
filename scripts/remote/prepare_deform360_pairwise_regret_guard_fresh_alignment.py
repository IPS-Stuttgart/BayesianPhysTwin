#!/usr/bin/env python3
"""Recover official alignment and robot state for one locked fresh episode."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

from deform360 import undistort
from deform360.processing import robot_stage

from bayesian_phystwin.deform360_pairwise_regret_guard_fresh_processing import (
    DEFORM360_REVISION,
    DEFORM360_SOURCE_SHA256,
    PREPARATION_KIND,
    fresh_processing_case,
    seal_case_artifact,
    validate_fresh_processing_sources,
    write_json_artifact,
)
from bayesian_phystwin.deform360_pairwise_regret_guard_fresh_protocol import (
    file_sha256,
)

PREPARATION_FILENAME = "fresh_pairwise_preparation.json"


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


def _require_clean_repository(repository: Path, expected: str) -> str:
    revision = _git_revision(repository)
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=normal"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    _require(not status.strip(), f"repository has uncommitted files: {repository}")
    _require(revision == expected, "processing implementation revision changed")
    return revision


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--technical-lock", type=Path, required=True)
    parser.add_argument("--source-plan", type=Path, required=True)
    parser.add_argument("--download-manifest", type=Path, required=True)
    parser.add_argument("--deform360-repo", type=Path, required=True)
    parser.add_argument("--download-root", type=Path, required=True)
    parser.add_argument("--aligned-root", type=Path, required=True)
    parser.add_argument("--object-id", required=True)
    parser.add_argument("--episode-id", type=int, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    protocol, lock, plan, download = validate_fresh_processing_sources(
        args.protocol.resolve(),
        args.technical_lock.resolve(),
        args.source_plan.resolve(),
        args.download_manifest.resolve(),
    )
    case = fresh_processing_case(lock, args.object_id, args.episode_id)
    code_revision = _require_clean_repository(
        args.repo.resolve(), str(protocol["implementation_commit"])
    )
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
        all(
            path.is_file() and file_sha256(path) == DEFORM360_SOURCE_SHA256[name]
            for name, path in sources.items()
        ),
        "official alignment dependency changed",
    )
    raw_object = args.download_root.resolve() / "raw" / args.object_id
    metadata = raw_object / "metadata.json"
    _require(metadata.is_file(), "downloaded metadata is missing")
    metadata_row = next(
        row
        for row in download["files"]
        if row["path"] == f"raw/{args.object_id}/metadata.json"
    )
    _require(
        file_sha256(metadata) == metadata_row["sha256"],
        "downloaded metadata changed",
    )
    episode_source = next(
        row for row in plan["episode_sources"] if row["episode_id"] == args.episode_id
    )
    for camera in episode_source["cameras"]:
        for key in ("video_path", "timestamp_path"):
            relative = camera[key]
            expected = next(row for row in download["files"] if row["path"] == relative)
            source = args.download_root.resolve() / relative
            _require(
                source.is_file() and file_sha256(source) == expected["sha256"],
                f"downloaded episode source changed: {relative}",
            )

    destination = args.aligned_root.resolve() / args.object_id
    episode = destination / f"episode_{args.episode_id:04d}"
    manifest_path = episode / PREPARATION_FILENAME
    _require(not manifest_path.exists(), "source preparation already exists")
    episode = undistort.undistort_episode(
        raw_object,
        destination,
        args.episode_id,
        overwrite=False,
        rebuild_timeline=False,
    )
    robot_path = robot_stage.process_robot_episode(
        destination,
        args.episode_id,
        bimanual=case["bimanual"] == "yes",
        seed=0,
        overwrite=False,
        plot=False,
    )
    alignment_path = episode / "alignment.json"
    alignment = json.loads(alignment_path.read_text(encoding="utf-8"))
    cameras = alignment.get("cameras")
    _require(
        isinstance(cameras, list)
        and set(protocol["dataset"]["camera_panel"]) <= set(cameras),
        "aligned camera panel is incomplete",
    )
    _require(alignment.get("frame_count", 0) >= 81, "aligned episode is too short")
    outputs: dict[str, Any] = {
        "alignment": file_sha256(alignment_path),
        "undistorted_intrinsics": file_sha256(episode / "undistorted_intrinsics.npy"),
        "extrinsics": file_sha256(episode / "extrinsics.npy"),
        "robot": file_sha256(robot_path),
        "robot_metadata": file_sha256(robot_path.with_name("robot.meta.json")),
        "camera_metadata": {
            camera: file_sha256(episode / camera / "metadata.json")
            for camera in cameras
        },
    }
    artifact = seal_case_artifact(
        PREPARATION_KIND,
        protocol=protocol,
        case=case,
        payload={
            "status": "prepared",
            "code_revision": code_revision,
            "deform360_revision": DEFORM360_REVISION,
            "aligned_frame_count": alignment["frame_count"],
            "cameras": cameras,
            "inputs_sha256": {
                "download_manifest": file_sha256(args.download_manifest),
                "object_metadata": file_sha256(metadata),
                **{
                    f"official_{name}_source": file_sha256(path)
                    for name, path in sources.items()
                },
            },
            "outputs_sha256": outputs,
            "information_boundary": {
                "full_rgb_decoded_for_alignment_and_robot_pose": True,
                "object_mask_created": False,
                "object_geometry_created_or_read": False,
                "future_particle_tracks_created_or_read": False,
                "target_metric_created_or_read": False,
                "held_v8_runtime_or_target_artifact_access": False,
            },
        },
    )
    write_json_artifact(artifact, manifest_path)
    print(json.dumps(artifact, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
