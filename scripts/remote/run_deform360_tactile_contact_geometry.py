#!/usr/bin/env python3
"""Build the frozen source-only tactile assignment-mixture geometry."""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path

import numpy as np

from bayesian_phystwin.deform360_causal_robot_prefix import (
    verify_causal_robot_prefix_artifact,
)
from bayesian_phystwin.deform360_tactile_contact_geometry import (
    build_assignment_mixture_geometry,
    evaluate_tactile_contact_geometry_quality,
    extract_active_tactile_rows,
    load_deform360_tactile_contact_geometry_lock,
    write_tactile_contact_geometry_artifact,
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


def _require_clean(root: Path, *, name: str) -> str:
    head = _git_head(root)
    status = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if status.strip():
        raise ValueError(f"{name} checkout is dirty")
    return head


def _require_ancestor(root: Path, expected: str, *, name: str) -> str:
    head = _require_clean(root, name=name)
    result = subprocess.run(
        ["git", "merge-base", "--is-ancestor", expected, "HEAD"],
        cwd=root,
        check=False,
    )
    if result.returncode != 0:
        raise ValueError(f"{name} does not contain frozen implementation {expected}")
    return head


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--robot-prefix-manifest", type=Path, required=True)
    parser.add_argument("--processed-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    lock = load_deform360_tactile_contact_geometry_lock(args.lock)
    geometry_config = lock["geometry"]
    source = lock["source"]
    quality_gate = lock["quality_gate"]
    assert isinstance(geometry_config, Mapping)
    assert isinstance(source, Mapping)
    assert isinstance(quality_gate, Mapping)
    runtime_revision = _require_ancestor(
        args.repository_root.resolve(),
        str(geometry_config["implementation_revision"]),
        name="Bayesian-PhysTwin",
    )
    upstream_revision = _require_clean(
        args.upstream_root.resolve(), name="Deform360 processing"
    )
    if upstream_revision != geometry_config["processing_revision"]:
        raise ValueError("Deform360 processing revision changed")

    robot_manifest_path = args.robot_prefix_manifest.resolve()
    if _sha256(robot_manifest_path) != source["robot_prefix_manifest_sha256"]:
        raise ValueError("robot-prefix manifest digest changed")
    robot_manifest = verify_causal_robot_prefix_artifact(robot_manifest_path)
    if (
        robot_manifest["artifact_id"] != source["robot_prefix_artifact_id"]
        or robot_manifest["array_archive"]["sha256"]
        != source["robot_prefix_archive_sha256"]
        or robot_manifest["anchor_authorized"] is not True
    ):
        raise ValueError("robot-prefix admission or identity changed")
    robot_archive = robot_manifest_path.parent / robot_manifest["array_archive"]["path"]
    with np.load(robot_archive, allow_pickle=False) as payload:
        robot_frame_ids = np.asarray(payload["source_frame_ids"]).copy()
        robot_transforms = np.asarray(payload["T_worlds"]).copy()
        robot_openings = np.asarray(payload["openings"]).copy()

    episode_dir = (
        args.processed_root.resolve()
        / str(source["object_id"])
        / f"episode_{int(source['processing_episode_index']):04d}"
    )
    tactile_records = source["tactile_files"]
    if not isinstance(tactile_records, Mapping):
        raise ValueError("tactile file lock is missing")
    tactile_by_sensor: dict[str, np.ndarray] = {}
    source_artifacts = {
        "robot/causal_robot_prefix.json": _sha256(robot_manifest_path),
        "robot/causal_robot_prefix.npz": _sha256(robot_archive),
    }
    for sensor_name in sorted(tactile_records):
        record = tactile_records[sensor_name]
        if not isinstance(record, Mapping):
            raise ValueError(f"invalid tactile record {sensor_name}")
        path = episode_dir / str(record["relative_path"])
        if _sha256(path) != record["sha256"]:
            raise ValueError(f"tactile file changed: {sensor_name}")
        tactile_by_sensor[sensor_name] = np.load(
            path, allow_pickle=False, mmap_mode="r"
        )
        source_artifacts[f"tactile/{sensor_name}.npy"] = str(record["sha256"])

    frame_start = int(source["contact_frame_start"])
    frame_stop = int(source["causal_frame_stop"])
    active = extract_active_tactile_rows(
        tactile_by_sensor,
        frame_start=frame_start,
        frame_stop=frame_stop,
        active_threshold=float(geometry_config["active_threshold"]),
    )
    sys.path.insert(0, str(args.upstream_root.resolve()))
    from deform360.processing.control_points_stage import (  # noqa: PLC0415
        gripper_taxel_points,
    )

    arrays = build_assignment_mixture_geometry(
        active,
        robot_source_frame_ids=robot_frame_ids,
        robot_transforms=robot_transforms,
        robot_openings_m=robot_openings,
        taxel_points=gripper_taxel_points,
    )
    quality = evaluate_tactile_contact_geometry_quality(
        arrays, quality_gate=quality_gate
    )
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = write_tactile_contact_geometry_artifact(
        arrays=arrays,
        quality=quality,
        lock=lock,
        output_npz=output_dir / "tactile_contact_geometry.npz",
        output_manifest=output_dir / "tactile_contact_geometry.json",
        implementation_revision=runtime_revision,
        source_artifacts=source_artifacts,
        overwrite=args.overwrite,
    )
    print(
        f"artifact_id={manifest['artifact_id']} admitted={quality.admitted} "
        f"reasons={','.join(quality.reason_codes) or 'none'}"
    )


if __name__ == "__main__":
    main()
