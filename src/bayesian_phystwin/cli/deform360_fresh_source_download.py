"""Download the frozen fresh-object Deform360 source queue."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess

from bayesian_phystwin.deform360_fresh_source_download import (
    download_fresh_episode_sources_from_index,
    download_fresh_source_queue_by_object,
    fresh_source_download_plan,
    write_fresh_download_manifest,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--object-delay-seconds", type=float, default=2.0)
    parser.add_argument(
        "--git-index",
        type=Path,
        help="Pinned Deform360 partial clone used only as a file-name index.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    plan = fresh_source_download_plan(args.queue)
    if args.dry_run:
        print(plan)
        return
    try:
        from huggingface_hub import HfApi, hf_hub_download
    except ModuleNotFoundError as error:  # pragma: no cover - integration dependency
        raise RuntimeError(
            "Deform360 download requires the optional huggingface_hub package"
        ) from error
    if args.git_index is None:
        manifest = download_fresh_source_queue_by_object(
            args.queue,
            args.output_root,
            max_workers=args.workers,
            object_delay_seconds=args.object_delay_seconds,
            list_repo_tree=HfApi().list_repo_tree,
            hub_download=hf_hub_download,
        )
    else:
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

        manifest = download_fresh_episode_sources_from_index(
            args.queue,
            args.output_root,
            max_workers=args.workers,
            object_delay_seconds=args.object_delay_seconds,
            list_object_files=list_object_files,
            hub_download=hf_hub_download,
        )
    write_fresh_download_manifest(args.manifest, manifest)
    print(manifest["manifest_sha256"])


if __name__ == "__main__":
    main()
