"""Validate and download the bias-aware Deform360 prospective panel."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from bayesian_phystwin.deform360_bias_aware_prospective_artifacts import (
    build_prospective_prediction_cohort_seal,
)
from bayesian_phystwin.deform360_bias_aware_prospective_download import (
    bias_aware_prospective_download_plan,
    download_bias_aware_prospective_panel,
    write_bias_aware_download_manifest,
)
from bayesian_phystwin.deform360_bias_aware_prospective_protocol import (
    load_bias_aware_prospective_protocol,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate")
    validate.add_argument("protocol", type=Path)

    plan = subparsers.add_parser("plan")
    plan.add_argument("protocol", type=Path)

    download = subparsers.add_parser("download")
    download.add_argument("protocol", type=Path)
    download.add_argument("output_root", type=Path)
    download.add_argument("--manifest", type=Path, required=True)
    download.add_argument("--max-workers", type=int, default=4)

    seal = subparsers.add_parser("seal-predictions")
    seal.add_argument("protocol", type=Path)
    seal.add_argument("role", choices=("calibration", "target"))
    seal.add_argument("artifact_root", type=Path)
    seal.add_argument("output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "validate":
        protocol = load_bias_aware_prospective_protocol(args.protocol)
        result = {
            "status": "valid",
            "config_sha256": protocol["config_sha256"],
            "calibration_object_count": sum(
                len(records) for records in protocol["calibration_cohort"].values()
            ),
            "target_object_count": sum(
                len(records) for records in protocol["target_cohort"].values()
            ),
        }
    elif args.command == "plan":
        plan = bias_aware_prospective_download_plan(args.protocol)
        result = {
            "repository": plan.repository,
            "revision": plan.revision,
            "calibration_objects": list(plan.calibration_objects),
            "target_objects": list(plan.target_objects),
            "allow_patterns": list(plan.allow_patterns),
            "ignore_patterns": list(plan.ignore_patterns),
            "protocol_config_sha256": plan.protocol_config_sha256,
        }
    elif args.command == "download":
        from huggingface_hub import snapshot_download

        result = download_bias_aware_prospective_panel(
            args.protocol,
            args.output_root,
            max_workers=args.max_workers,
            snapshot_download=snapshot_download,
        )
        write_bias_aware_download_manifest(args.manifest, result)
    else:
        result = build_prospective_prediction_cohort_seal(
            args.protocol,
            args.role,
            args.artifact_root,
            args.output,
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
