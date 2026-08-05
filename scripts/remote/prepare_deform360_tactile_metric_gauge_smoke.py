#!/usr/bin/env python3
"""Freeze a geometry-selected tactile metric-gauge source smoke."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from bayesian_phystwin._portable_contracts import content_id, write_atomic_json
from bayesian_phystwin.deform360_causal_robot_prefix import (
    load_deform360_causal_robot_prefix_lock,
)
from bayesian_phystwin.deform360_official_hub_motioncrafter_jobs import (
    load_deform360_motioncrafter_job_manifest,
)
from bayesian_phystwin.deform360_tactile_contact_geometry import (
    verify_tactile_contact_geometry_artifact,
)
from bayesian_phystwin.deform360_tactile_metric_gauge import (
    CONTACT_CAMERA_POLICY,
    DEFORM360_TACTILE_METRIC_GAUGE_LOCK_SCHEMA,
    TACTILE_METRIC_GAUGE_INFORMATION_BOUNDARY,
    TACTILE_METRIC_GAUGE_QUALITY_GATE,
    contact_camera_candidates,
    select_contact_camera_panel,
    validate_tactile_metric_gauge_lock,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_head(root: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _require_clean(root: Path) -> str:
    head = _git_head(root)
    status = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if status.strip():
        raise ValueError("Bayesian-PhysTwin checkout is dirty")
    return head


def _load_camera_dictionary(path: Path) -> dict[str, np.ndarray]:
    values = np.load(path, allow_pickle=True)
    if values.shape != () or values.dtype != object:
        raise ValueError(f"invalid first-party camera dictionary {path}")
    mapping = values.item()
    if not isinstance(mapping, dict):
        raise ValueError(f"invalid first-party camera dictionary {path}")
    return {str(name): np.asarray(value) for name, value in mapping.items()}


def _artifact_record(path: Path, artifact_id: str) -> dict[str, str]:
    return {
        "path": path.as_posix(),
        "artifact_id": artifact_id,
        "sha256": _sha256(path),
    }


def _camera_record(candidate: Any) -> dict[str, Any]:
    return {
        "camera": candidate.camera,
        "minimum_assignment_coverage": candidate.minimum_assignment_coverage,
        "minimum_margin_px": candidate.minimum_margin_px,
        "view_direction": candidate.view_direction.tolist(),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-job-manifest", type=Path, required=True)
    parser.add_argument("--processed-root", type=Path, required=True)
    parser.add_argument("--robot-lock", type=Path, required=True)
    parser.add_argument("--robot-manifest", type=Path, required=True)
    parser.add_argument("--tactile-manifest", type=Path, required=True)
    parser.add_argument("--runner-source", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    repository_root = args.repository_root.resolve()
    implementation_revision = _require_clean(repository_root)
    runner_source = args.runner_source.resolve()
    try:
        runner_source.relative_to(repository_root)
    except ValueError as error:
        raise ValueError("runner source is outside the repository") from error

    parent_path = args.parent_job_manifest.resolve()
    parent = load_deform360_motioncrafter_job_manifest(parent_path)
    robot_lock_path = args.robot_lock.resolve()
    robot_lock = load_deform360_causal_robot_prefix_lock(robot_lock_path)
    robot_path = args.robot_manifest.resolve()
    robot = json.loads(robot_path.read_text(encoding="utf-8"))
    robot_id = str(robot.get("artifact_id", ""))
    robot_descriptor = dict(robot)
    robot_descriptor.pop("artifact_id", None)
    if content_id(robot_descriptor) != robot_id:
        raise ValueError("robot-prefix artifact identity changed")
    tactile_path = args.tactile_manifest.resolve()
    tactile = verify_tactile_contact_geometry_artifact(tactile_path)

    if robot.get("lock_id") != robot_lock["artifact_id"]:
        raise ValueError("robot-prefix artifact is not bound to its supplied lock")
    source_case = robot_lock["source_case"]
    causal_window = robot_lock["causal_window"]
    assert isinstance(source_case, Mapping)
    assert isinstance(causal_window, Mapping)
    object_id = str(source_case["object_id"])
    episode_index = int(source_case["processing_episode_index"])
    if object_id != "026-sock-cloth" or episode_index != 0:
        raise ValueError("unexpected source smoke case")
    if (
        tactile["source_artifacts"].get("robot/causal_robot_prefix.json")
        != _sha256(robot_path)
    ):
        raise ValueError("tactile geometry is not bound to the robot prefix")

    episode_dir = (
        args.processed_root.resolve()
        / object_id
        / f"episode_{episode_index:04d}"
    )
    intrinsics_path = episode_dir / "undistorted_intrinsics.npy"
    extrinsics_path = episode_dir / "extrinsics.npy"
    intrinsics = _load_camera_dictionary(intrinsics_path)
    extrinsics = _load_camera_dictionary(extrinsics_path)
    archive_path = tactile_path.parent / str(tactile["archive"]["path"])
    with np.load(archive_path, allow_pickle=False) as archive:
        contact_points = np.asarray(archive["world_points_hypotheses_m"])
    candidates = contact_camera_candidates(
        contact_points,
        intrinsics_by_camera=intrinsics,
        world_from_camera_by_camera=extrinsics,
        source_shape=tuple(CONTACT_CAMERA_POLICY["source_shape"]),
        target_shape=tuple(CONTACT_CAMERA_POLICY["target_shape"]),
    )
    selected = select_contact_camera_panel(
        candidates,
        panel_size=int(CONTACT_CAMERA_POLICY["panel_size"]),
        minimum_coverage=float(CONTACT_CAMERA_POLICY["minimum_assignment_coverage"]),
        minimum_margin_px=float(CONTACT_CAMERA_POLICY["minimum_margin_px"]),
        minimum_angular_separation_deg=float(
            CONTACT_CAMERA_POLICY["minimum_angular_separation_deg"]
        ),
    )
    if len(selected) != int(CONTACT_CAMERA_POLICY["panel_size"]):
        raise ValueError("contact camera policy did not produce a complete panel")

    parent_jobs = [
        item for item in parent["jobs"] if item.get("object_id") == object_id
    ]
    if len(parent_jobs) != 3:
        raise ValueError("parent source case does not have three jobs")
    parent_by_camera = {str(item["camera"]): item for item in parent_jobs}
    template = parent_jobs[0]
    selected_names = [item.camera for item in selected]
    reused = [name for name in selected_names if name in parent_by_camera]
    supplemental = [name for name in selected_names if name not in parent_by_camera]
    jobs: list[dict[str, Any]] = []
    for camera in supplemental:
        video = episode_dir / camera / "undistorted.mp4"
        if not video.is_file():
            raise FileNotFoundError(video)
        relative = video.relative_to(args.processed_root.resolve()).as_posix()
        descriptor: dict[str, Any] = {
            "object_id": object_id,
            "episode": f"episode_{episode_index:04d}",
            "camera": camera,
            "source_video": {
                "path": relative,
                "sha256": _sha256(video),
                "bytes": video.stat().st_size,
            },
            "source_frame_start": template["source_frame_start"],
            "source_frame_stop_exclusive": template["source_frame_stop_exclusive"],
            "windows": template["windows"],
            "seed_schedule": template["seed_schedule"],
            "output_relative_path": (
                f"{object_id}/episode_{episode_index:04d}/{camera}"
            ),
        }
        jobs.append({"job_id": content_id(descriptor), **descriptor})

    parent_id = str(parent["manifest_sha256"])
    value: dict[str, Any] = {
        "schema": DEFORM360_TACTILE_METRIC_GAUGE_LOCK_SCHEMA,
        "schema_version": 1,
        "status": "locked-source-only-pre-supplement-provider",
        "implementation": {
            "revision": implementation_revision,
            "runner_source_sha256": _sha256(runner_source),
        },
        "source_case": {
            "object_id": object_id,
            "processing_episode_index": episode_index,
            "causal_frame_stop": int(causal_window["causal_frame_stop"]),
        },
        "parents": {
            "motioncrafter_job_manifest": _artifact_record(parent_path, parent_id),
            "robot_prefix_lock": _artifact_record(
                robot_lock_path, str(robot_lock["artifact_id"])
            ),
            "robot_prefix": _artifact_record(robot_path, robot_id),
            "tactile_contact_geometry": _artifact_record(
                tactile_path, str(tactile["artifact_id"])
            ),
        },
        "calibration": {
            "undistorted_intrinsics_sha256": _sha256(intrinsics_path),
            "extrinsics_sha256": _sha256(extrinsics_path),
        },
        "camera_selection": {
            "policy": CONTACT_CAMERA_POLICY,
            "selected_cameras": selected_names,
            "reused_provider_cameras": reused,
            "supplemental_provider_cameras": supplemental,
            "all_candidates": [_camera_record(item) for item in candidates],
        },
        "provider": {
            "parent_manifest_id": parent_id,
            "run_configuration": parent["run_configuration"],
            "motioncrafter": parent["motioncrafter"],
            "provider_lock": parent["provider_lock"],
            "quality_gate": TACTILE_METRIC_GAUGE_QUALITY_GATE,
        },
        "supplemental_jobs": jobs,
        "information_boundary": TACTILE_METRIC_GAUGE_INFORMATION_BOUNDARY,
        "claim_boundary": (
            "Source-only tactile metric-gauge feasibility. No calibration score, "
            "object-state update, confirmation payload, target outcome, or SOTA claim."
        ),
    }
    value = {"artifact_id": content_id(value), **value}
    validate_tactile_metric_gauge_lock(value)
    write_atomic_json(value, args.output, overwrite=args.overwrite)
    print(
        f"artifact_id={value['artifact_id']} selected={','.join(selected_names)} "
        f"supplemental={','.join(supplemental)}"
    )


if __name__ == "__main__":
    main()
