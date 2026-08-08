#!/usr/bin/env python3
"""Build the exact all-camera Deform360 calibration visual-job manifest."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from bayesian_phystwin.deform360_calibration_visual_jobs import (
    build_deform360_calibration_visual_job_manifest,
    save_deform360_calibration_visual_job_manifest,
)

INSUFFICIENT_SUPPORT_EXIT_CODE = 3
CONTRACT_FAILURE_EXIT_CODE = 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage0-protocol", type=Path, required=True)
    parser.add_argument("--selection-lock", type=Path, required=True)
    parser.add_argument("--visual-provider-lock", type=Path, required=True)
    parser.add_argument(
        "--calibration-source-run-record",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--calibration-source-result",
        type=Path,
        required=True,
    )
    parser.add_argument("--processed-root", type=Path, required=True)
    parser.add_argument("--implementation-revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        manifest = build_deform360_calibration_visual_job_manifest(
            stage0_protocol_path=args.stage0_protocol,
            selection_lock_path=args.selection_lock,
            visual_provider_lock_path=args.visual_provider_lock,
            calibration_source_run_record_path=(
                args.calibration_source_run_record
            ),
            calibration_source_result_path=args.calibration_source_result,
            processed_root=args.processed_root,
            implementation_revision=args.implementation_revision,
        )
        save_deform360_calibration_visual_job_manifest(
            args.output,
            manifest,
        )
    except (OSError, TypeError, ValueError) as error:
        print(
            json.dumps(
                {"passed": False, "error": str(error)},
                indent=2,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return CONTRACT_FAILURE_EXIT_CODE

    gate = manifest["support_gate"]
    assert isinstance(gate, dict)
    summary = {
        "passed": gate["support_passed"],
        "manifest_id": manifest["manifest_id"],
        "status": manifest["status"],
        "planned_object_count": gate["planned_object_count"],
        "technical_failure_count": gate["technical_failure_count"],
        "job_count": sum(
            len(item["jobs"])
            for item in manifest["objects"]
            if isinstance(item, dict)
        ),
        "output": str(args.output),
    }
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    return 0 if gate["support_passed"] is True else INSUFFICIENT_SUPPORT_EXIT_CODE


if __name__ == "__main__":
    raise SystemExit(main())
