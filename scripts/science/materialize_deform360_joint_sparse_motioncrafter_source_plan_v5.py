#!/usr/bin/env python3
"""Build or validate the public Deform360 v5.1 MotionCrafter source plan."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from bayesian_phystwin.deform360_joint_sparse_motioncrafter_source_v5 import (
    build_deform360_joint_sparse_motioncrafter_source_plan_from_paths_v5,
    load_deform360_joint_sparse_motioncrafter_source_plan_v5,
    save_deform360_joint_sparse_motioncrafter_source_plan_v5,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build")
    build.add_argument("--execution-lock", type=Path, required=True)
    build.add_argument("--prepared-source-inventory", type=Path, required=True)
    build.add_argument("--camera-roster-manifest", type=Path, required=True)
    build.add_argument("--implementation-revision", required=True)
    build.add_argument("--runner-source", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)

    validate = subparsers.add_parser("validate")
    validate.add_argument("plan", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.command == "validate":
        plan = load_deform360_joint_sparse_motioncrafter_source_plan_v5(arguments.plan)
    else:
        plan = build_deform360_joint_sparse_motioncrafter_source_plan_from_paths_v5(
            execution_lock_path=arguments.execution_lock,
            prepared_source_inventory_path=arguments.prepared_source_inventory,
            legacy_job_manifest_path=arguments.camera_roster_manifest,
            implementation_revision=arguments.implementation_revision,
            runner_source_path=arguments.runner_source,
        )
        save_deform360_joint_sparse_motioncrafter_source_plan_v5(arguments.output, plan)
    print(
        json.dumps(
            {
                "manifest_sha256": plan["manifest_sha256"],
                "object_count": plan["object_count"],
                "job_count": plan["job_count"],
                "provider_outputs_opened": plan["information_boundary"][
                    "provider_outputs_opened"
                ],
                "development_suffix_opened": plan["information_boundary"][
                    "development_suffix_opened"
                ],
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
