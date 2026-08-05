#!/usr/bin/env python3
"""Materialize only the exact locked Deform360 calibration-episode inputs.

The command resolves the immutable official-Hub dataset revision from the
committed Stage-0 lock. It lists only the ten calibration object directories,
reproduces the official filename-sorted episode indexing, and optionally opens a
narrow prefix payload set: metadata, trusted camera calibration, selected camera
timestamp sidecars, and exact selected tactile recordings. Camera video bytes,
robot arrays, geometry annotations, confirmation payloads, and target outcomes
remain unopened.
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Sequence
from pathlib import Path

from bayesian_phystwin.deform360_calibration_payload_plan import (
    HubFile,
    UnitPlan,
    build_unit_plan,
    canonical_hub_path,
    hub_file_from_member,
)
from bayesian_phystwin.deform360_calibration_payload_runtime import (
    build_manifest,
    execute_materialization,
    file_sha256,
    materialize_paths,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-lock", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--visual-provider-lock", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--processing-checkout", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--implementation-revision", required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--open-calibration-payloads",
        action="store_true",
        help=(
            "download and hash the exact selected calibration-prefix files; "
            "confirmation paths remain forbidden"
        ),
    )
    parser.add_argument(
        "--token-environment-variable",
        default="HF_TOKEN",
        help="environment variable holding an optional Hugging Face token",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    if arguments.workers < 1:
        raise SystemExit("--workers must be positive")
    try:
        from huggingface_hub import HfApi, hf_hub_download
    except ImportError as error:
        raise SystemExit("huggingface_hub is required") from error

    manifest = execute_materialization(
        selection_lock=arguments.selection_lock.resolve(),
        protocol_path=arguments.protocol.resolve(),
        visual_provider_lock=arguments.visual_provider_lock.resolve(),
        repository_root=arguments.repository_root.resolve(),
        processing_checkout=arguments.processing_checkout.resolve(),
        dataset_root=arguments.dataset_root.resolve(),
        output=arguments.output.resolve(),
        implementation_revision=arguments.implementation_revision,
        workers=arguments.workers,
        open_calibration_payloads=arguments.open_calibration_payloads,
        token=os.environ.get(arguments.token_environment_variable),
        api=HfApi(),
        download_file=hf_hub_download,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


__all__ = [
    "HubFile",
    "UnitPlan",
    "build_manifest",
    "build_parser",
    "build_unit_plan",
    "canonical_hub_path",
    "file_sha256",
    "hub_file_from_member",
    "main",
    "materialize_paths",
]


if __name__ == "__main__":
    raise SystemExit(main())
