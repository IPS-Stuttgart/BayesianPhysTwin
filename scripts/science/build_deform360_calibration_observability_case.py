#!/usr/bin/env python3
"""Build one strict Deform360 calibration observability case artifact."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from bayesian_phystwin.deform360_calibration_observability_case_builder import (
    build_evaluated_case_from_paths,
    build_technical_failure_case_from_paths,
)
from bayesian_phystwin.deform360_calibration_observability_report import (
    save_deform360_calibration_observability_case,
)


def _common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--source-protocol", type=Path, required=True)
    parser.add_argument("--stage0-protocol", type=Path, required=True)
    parser.add_argument("--selection-lock", type=Path, required=True)
    parser.add_argument("--visual-provider-lock", type=Path, required=True)
    parser.add_argument("--calibration-source-plan", type=Path, required=True)
    parser.add_argument("--calibration-source-download", type=Path, required=True)
    parser.add_argument(
        "--calibration-source-run-record",
        type=Path,
        required=True,
    )
    parser.add_argument("--calibration-source-result", type=Path, required=True)
    parser.add_argument("--object-id", required=True)
    parser.add_argument("--implementation-revision", required=True)
    parser.add_argument("--query-jacobian", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)

    evaluated = subparsers.add_parser(
        "evaluated",
        help="Build a numerical visual-versus-contact comparison case",
    )
    _common_arguments(evaluated)
    evaluated.add_argument(
        "--reference-marginal-precision",
        type=Path,
        required=True,
    )
    evaluated.add_argument(
        "--candidate-marginal-precision",
        type=Path,
        required=True,
    )
    evaluated.add_argument("--contact-anchor-artifact", type=Path, required=True)

    failure = subparsers.add_parser(
        "technical-failure",
        help="Retain one object without a numerical observability result",
    )
    _common_arguments(failure)
    failure.add_argument("--failure-evidence", type=Path, required=True)
    failure.add_argument("--failure-reason", required=True)
    return parser


def _common(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "source_protocol_path": args.source_protocol,
        "stage0_protocol_path": args.stage0_protocol,
        "selection_lock_path": args.selection_lock,
        "visual_provider_lock_path": args.visual_provider_lock,
        "calibration_source_plan_path": args.calibration_source_plan,
        "calibration_source_download_path": args.calibration_source_download,
        "calibration_source_run_record_path": args.calibration_source_run_record,
        "calibration_source_result_path": args.calibration_source_result,
        "object_id": args.object_id,
        "implementation_revision": args.implementation_revision,
        "query_jacobian_path": args.query_jacobian,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        common = _common(args)
        if args.mode == "evaluated":
            case = build_evaluated_case_from_paths(
                **common,
                reference_marginal_precision_path=(args.reference_marginal_precision),
                candidate_marginal_precision_path=(args.candidate_marginal_precision),
                contact_anchor_artifact_path=args.contact_anchor_artifact,
            )
        else:
            case = build_technical_failure_case_from_paths(
                **common,
                failure_evidence_path=args.failure_evidence,
                failure_reason=args.failure_reason,
            )
        save_deform360_calibration_observability_case(case, args.output)
    except (OSError, TypeError, ValueError) as error:
        print(f"cannot build calibration observability case: {error}", file=sys.stderr)
        return 2
    print(json.dumps(case.to_record(), indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
