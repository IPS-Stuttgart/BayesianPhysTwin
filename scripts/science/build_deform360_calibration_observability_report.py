#!/usr/bin/env python3
"""Build the locked ten-object Deform360 calibration observability report."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from bayesian_phystwin.deform360_calibration_observability_report import (
    build_report_from_paths,
    save_deform360_calibration_observability_report,
)

INSUFFICIENT_SUPPORT_EXIT_CODE = 3
CONTRACT_FAILURE_EXIT_CODE = 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-lock", type=Path, required=True)
    parser.add_argument("--stage0-protocol", type=Path, required=True)
    parser.add_argument("--visual-provider-lock", type=Path, required=True)
    parser.add_argument(
        "--calibration-source-run-record",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--case",
        dest="case_paths",
        action="append",
        type=Path,
        required=True,
        help="One strict per-object observability case JSON; repeat ten times.",
    )
    parser.add_argument("--implementation-revision", required=True)
    parser.add_argument("--physical-query-id", required=True)
    parser.add_argument(
        "--numerical-positive-tolerance",
        type=float,
        default=1e-12,
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = build_report_from_paths(
            selection_lock_path=args.selection_lock,
            stage0_protocol_path=args.stage0_protocol,
            visual_provider_lock_path=args.visual_provider_lock,
            calibration_source_run_record_path=(
                args.calibration_source_run_record
            ),
            case_paths=args.case_paths,
            implementation_revision=args.implementation_revision,
            physical_query_id=args.physical_query_id,
            numerical_positive_tolerance=args.numerical_positive_tolerance,
        )
        save_deform360_calibration_observability_report(report, args.output)
    except (OSError, TypeError, ValueError) as error:
        print(
            json.dumps(
                {
                    "passed": False,
                    "error": str(error),
                },
                indent=2,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return CONTRACT_FAILURE_EXIT_CODE

    summary = {
        "passed": report.support_gate["support_passed"],
        "report_id": report.report_id,
        "status": report.status,
        "support_gate": report.support_gate,
        "overall": report.overall,
        "output": str(args.output),
    }
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    if report.support_gate["support_passed"] is True:
        return 0
    return INSUFFICIENT_SUPPORT_EXIT_CODE


if __name__ == "__main__":
    raise SystemExit(main())
