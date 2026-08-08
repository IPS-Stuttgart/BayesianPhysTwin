#!/usr/bin/env python3
"""Build or validate the frozen ten-object visual-production plan."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from bayesian_phystwin.deform360_calibration_visual_production_plan import (
    build_deform360_calibration_visual_production_plan,
    load_deform360_calibration_visual_production_plan,
    save_deform360_calibration_visual_production_plan,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build", help="build the target-blind plan")
    build.add_argument("--source-protocol", type=Path, required=True)
    build.add_argument("--stage0-protocol", type=Path, required=True)
    build.add_argument("--selection-lock", type=Path, required=True)
    build.add_argument("--visual-provider-lock", type=Path, required=True)
    build.add_argument("--calibration-source-plan", type=Path, required=True)
    build.add_argument("--calibration-source-download", type=Path, required=True)
    build.add_argument(
        "--calibration-source-run-record",
        type=Path,
        required=True,
    )
    build.add_argument("--calibration-source-result", type=Path, required=True)
    build.add_argument("--implementation-revision", required=True)
    build.add_argument("--output", type=Path, required=True)

    validate = subparsers.add_parser("validate", help="revalidate one plan")
    validate.add_argument("plan", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "validate":
        plan = load_deform360_calibration_visual_production_plan(args.plan)
    else:
        plan = build_deform360_calibration_visual_production_plan(
            source_protocol_path=args.source_protocol,
            stage0_protocol_path=args.stage0_protocol,
            selection_lock_path=args.selection_lock,
            visual_provider_lock_path=args.visual_provider_lock,
            calibration_source_plan_path=args.calibration_source_plan,
            calibration_source_download_path=args.calibration_source_download,
            calibration_source_run_record_path=(args.calibration_source_run_record),
            calibration_source_result_path=args.calibration_source_result,
            implementation_revision=args.implementation_revision,
        )
        save_deform360_calibration_visual_production_plan(args.output, plan)
    print(
        json.dumps(
            {
                "plan_id": plan["plan_id"],
                "object_count": plan["object_count"],
                "camera_view_count": plan["camera_view_count"],
                "confirmation_payloads_opened": plan["information_boundary"][
                    "confirmation_payloads_opened"
                ],
                "target_outcomes_used": plan["information_boundary"][
                    "target_outcomes_used"
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
