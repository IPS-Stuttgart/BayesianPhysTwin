#!/usr/bin/env python3
"""Run and checksum official Deform360 stages for one development episode."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from causal4d_public.deform360_reusable_sota_protocol import (
    load_reusable_sota_config,
)
from causal4d_public.deform360_sota_processing import (
    PINNED_COTRACKER_CHECKPOINT_SHA256,
    PINNED_COTRACKER_REVISION,
    PINNED_DEFORM360_PROCESSING_REVISION,
    authorize_development_processing,
    build_development_observations_manifest,
)
from deform360.processing import (
    control_points_stage,
    depth_stage,
    pcd_stage,
    tracking_stage,
)


def _git_revision(repository: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--deform360-repo", type=Path, required=True)
    parser.add_argument("--cotracker-repo", type=Path, required=True)
    parser.add_argument("--cotracker-checkpoint", type=Path, required=True)
    parser.add_argument("--processing-root", type=Path, required=True)
    parser.add_argument("--object-id", required=True)
    parser.add_argument("--episode-id", type=int, required=True)
    parser.add_argument("--role", choices=("fit", "held-development"), required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    protocol_path = args.protocol or (
        args.repo / "configs/causal4d_public/deform360_reusable_sota_v1.json"
    )
    protocol = load_reusable_sota_config(protocol_path)
    authorization = authorize_development_processing(
        protocol,
        object_id=args.object_id,
        episode_id=args.episode_id,
        role=args.role,
    )
    deform360_revision = _git_revision(args.deform360_repo)
    cotracker_revision = _git_revision(args.cotracker_repo)
    if deform360_revision != PINNED_DEFORM360_PROCESSING_REVISION:
        raise ValueError("Deform360 processing revision changed")
    if cotracker_revision != PINNED_COTRACKER_REVISION:
        raise ValueError("CoTracker revision changed")
    if (
        not args.cotracker_checkpoint.is_file()
        or _file_sha256(args.cotracker_checkpoint)
        != PINNED_COTRACKER_CHECKPOINT_SHA256
    ):
        raise ValueError("CoTracker checkpoint changed")

    episode_dir = (
        args.processing_root.resolve()
        / args.object_id
        / f"episode_{args.episode_id:04d}"
    )
    staging = json.loads(
        (episode_dir / "development_staging.json").read_text(encoding="utf-8")
    )
    if staging.get("authorization") != authorization:
        raise ValueError("development staging uses another authorization")
    cameras = sorted(
        path.name
        for path in episode_dir.iterdir()
        if path.is_dir() and (path / "mask_refined.h5").is_file()
    )
    if len(cameras) != int(staging["camera_count"]):
        raise ValueError("development camera panel changed")

    depth_stage.process_depth_episode(
        args.processing_root / args.object_id,
        args.episode_id,
        cameras=cameras,
        overwrite=args.overwrite,
        preview=False,
    )
    tracking_stage.process_tracking_episode(
        args.processing_root / args.object_id,
        args.episode_id,
        cameras=cameras,
        checkpoint=args.cotracker_checkpoint,
        overwrite=args.overwrite,
    )
    pcd_stage.process_pcd_episode(
        args.processing_root / args.object_id,
        args.episode_id,
        cameras=cameras,
        overwrite=args.overwrite,
        rng_seed=0,
    )
    control_points_stage.process_control_points_episode(
        args.processing_root / args.object_id,
        args.episode_id,
        cameras=cameras,
        overwrite=args.overwrite,
    )
    result = build_development_observations_manifest(
        authorization=authorization,
        processing_root=args.processing_root,
        deform360_processing_revision=deform360_revision,
        cotracker_revision=cotracker_revision,
        cotracker_checkpoint=args.cotracker_checkpoint,
    )
    output = episode_dir / "development_observations.json"
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"observation manifest already exists: {output}")
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "passed": True,
                "object_id": result["object_id"],
                "episode_id": result["episode_id"],
                "point_frame_count": result["point_frame_count"],
                "material_point_count": result["material_point_count"],
                "material_identity_sha256": result["material_identity_sha256"],
                "result_sha256": result["result_sha256"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
