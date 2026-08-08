#!/usr/bin/env python3
"""Bind a frozen visual plan to exact retained calibration source metadata."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from bayesian_phystwin.deform360_calibration_visual_execution_admission import (
    build_deform360_calibration_visual_execution_admission,
    save_deform360_calibration_visual_execution_admission,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--visual-production-plan", type=Path, required=True)
    parser.add_argument("--prepared-source-inventory", type=Path, required=True)
    parser.add_argument("--implementation-revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        admission = build_deform360_calibration_visual_execution_admission(
            visual_production_plan_path=args.visual_production_plan,
            prepared_source_inventory_path=args.prepared_source_inventory,
            implementation_revision=args.implementation_revision,
        )
        save_deform360_calibration_visual_execution_admission(
            args.output,
            admission,
        )
    except (OSError, TypeError, ValueError) as error:
        print(
            json.dumps(
                {"admitted": False, "error": str(error)},
                indent=2,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2
    print(
        json.dumps(
            {
                "admitted": True,
                "admission_id": admission["admission_id"],
                "object_count": admission["object_count"],
                "camera_view_count": admission["camera_view_count"],
                "output": str(args.output),
                "information_boundary": admission["information_boundary"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
