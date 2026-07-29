#!/usr/bin/env python3
"""Download ranked V14 Deform360 camera and tactile source inputs."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from bayesian_phystwin.deform360_causal_response_direct_depth_cohort import (
    validate_v14_staging_queue,
)
from bayesian_phystwin.deform360_causal_response_preflight import (
    REGISTERED_CAMERA_IDS,
)
from bayesian_phystwin.deform360_fresh_source_download import (
    download_fresh_causal_episode_sources_from_index,
    fresh_source_download_plan,
    write_fresh_download_manifest,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--git-index", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--candidate-rank", type=int, action="append", required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--object-delay-seconds", type=float, default=0.0)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    queue_path = args.queue.resolve()
    queue = validate_v14_staging_queue(queue_path)
    plan = fresh_source_download_plan(queue_path)
    ranks = tuple(sorted(set(args.candidate_rank)))
    if (
        not ranks
        or ranks[0] < 1
        or ranks[-1] > len(queue["candidates"])
    ):
        raise ValueError("candidate ranks are outside the frozen V14 queue")
    if args.dry_run:
        print(
            {
                "queue_sha256": queue["queue_sha256"],
                "candidate_ranks": ranks,
                "required_camera_ids": REGISTERED_CAMERA_IDS,
            }
        )
        return
    if args.manifest.exists():
        raise FileExistsError(f"refusing to replace source manifest: {args.manifest}")
    try:
        from huggingface_hub import hf_hub_download
    except ModuleNotFoundError as error:  # pragma: no cover - integration dependency
        raise RuntimeError(
            "V14 source download requires the optional huggingface_hub package"
        ) from error

    index = args.git_index.resolve()
    subprocess.run(
        [
            "git",
            "-C",
            str(index),
            "cat-file",
            "-e",
            f"{plan.revision}^{{commit}}",
        ],
        check=True,
    )

    def list_object_files(object_id: str) -> list[str]:
        result = subprocess.run(
            [
                "git",
                "-C",
                str(index),
                "ls-tree",
                "-r",
                "--name-only",
                "-z",
                plan.revision,
                "--",
                f"raw/{object_id}",
            ],
            check=True,
            capture_output=True,
        )
        return [
            value.decode("utf-8")
            for value in result.stdout.split(b"\0")
            if value
        ]

    manifest = download_fresh_causal_episode_sources_from_index(
        queue_path,
        args.output_root,
        candidate_ranks=ranks,
        required_camera_ids=REGISTERED_CAMERA_IDS,
        max_workers=args.workers,
        object_delay_seconds=args.object_delay_seconds,
        list_object_files=list_object_files,
        hub_download=hf_hub_download,
    )
    write_fresh_download_manifest(args.manifest, manifest)
    print(manifest["manifest_sha256"])


if __name__ == "__main__":
    main()
