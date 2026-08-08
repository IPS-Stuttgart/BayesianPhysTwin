"""CLI for the locked official-Hub Deform360 calibration-source stage."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .download import DOWNLOAD_MAX_WORKERS, download_plan
from .planning import build_plan
from .prepare import prepare_sources


def _common_paths(args: argparse.Namespace) -> dict[str, Path]:
    return {
        "protocol_path": args.protocol.resolve(),
        "selection_path": args.selection_lock.resolve(),
        "provider_path": args.visual_provider_lock.resolve(),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    def common(command: argparse.ArgumentParser) -> None:
        command.add_argument("--protocol", type=Path, required=True)
        command.add_argument("--selection-lock", type=Path, required=True)
        command.add_argument(
            "--visual-provider-lock",
            type=Path,
            required=True,
        )

    plan = subparsers.add_parser(
        "plan",
        help="seal exact calibration file names",
    )
    common(plan)
    plan.add_argument("--output", type=Path, required=True)

    download = subparsers.add_parser(
        "download",
        help="download only sealed calibration files",
    )
    common(download)
    download.add_argument("--plan", type=Path, required=True)
    download.add_argument("--data-root", type=Path, required=True)
    download.add_argument("--output", type=Path, required=True)
    download.add_argument("--workers", type=int, default=DOWNLOAD_MAX_WORKERS)

    prepare = subparsers.add_parser(
        "prepare",
        help="align calibration RGB, tactile, and robot source",
    )
    common(prepare)
    prepare.add_argument("--plan", type=Path, required=True)
    prepare.add_argument("--download-manifest", type=Path, required=True)
    prepare.add_argument("--data-root", type=Path, required=True)
    prepare.add_argument("--staged-raw-root", type=Path, required=True)
    prepare.add_argument("--processed-root", type=Path, required=True)
    prepare.add_argument("--processing-repository", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    paths = _common_paths(args)
    if args.command == "plan":
        from huggingface_hub import HfApi

        result = build_plan(
            **paths,
            output_path=args.output.resolve(),
            api=HfApi(),
        )
        print(
            json.dumps(
                {"plan_sha256": result["plan_sha256"], **result["gate"]},
                sort_keys=True,
            )
        )
        return 0 if result["gate"]["support_passed"] else 3
    if args.command == "download":
        os.environ["HF_HUB_DISABLE_XET"] = "1"
        from huggingface_hub import hf_hub_download

        result = download_plan(
            **paths,
            plan_path=args.plan.resolve(),
            data_root=args.data_root.resolve(),
            output_path=args.output.resolve(),
            max_workers=args.workers,
            hub_download=hf_hub_download,
        )
        print(
            json.dumps(
                {
                    "download_sha256": result["download_sha256"],
                    "file_count": len(result["files"]),
                },
                sort_keys=True,
            )
        )
        return 0
    result = prepare_sources(
        **paths,
        plan_path=args.plan.resolve(),
        download_path=args.download_manifest.resolve(),
        data_root=args.data_root.resolve(),
        staged_root=args.staged_raw_root.resolve(),
        processed_root=args.processed_root.resolve(),
        processing_repository=args.processing_repository.resolve(),
        output_path=args.output.resolve(),
    )
    print(
        json.dumps(
            {"result_sha256": result["result_sha256"], **result["gate"]},
            sort_keys=True,
        )
    )
    return 0 if result["gate"]["support_passed"] else 3
