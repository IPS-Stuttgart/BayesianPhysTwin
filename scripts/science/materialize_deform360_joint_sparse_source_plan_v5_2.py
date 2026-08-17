#!/usr/bin/env python3
"""Freeze the v5.2 camera-recovered public source prediction plan."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

from bayesian_phystwin._portable_contracts import (
    load_strict_json_object,
    write_atomic_json,
)
from bayesian_phystwin.deform360_joint_sparse_camera_recovery_v5_2 import (
    validate_deform360_joint_sparse_camera_audit_v5_2,
)
from bayesian_phystwin.deform360_joint_sparse_source_gate_v5 import (
    load_deform360_joint_sparse_source_execution_lock_v5,
)
from bayesian_phystwin.deform360_joint_sparse_source_runner_v5 import (
    validate_deform360_joint_sparse_source_prediction_plan_v5,
)
from bayesian_phystwin.deform360_joint_sparse_source_runner_v5_2 import (
    build_deform360_joint_sparse_source_prediction_plan_v5_2,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execution-lock", type=Path, required=True)
    parser.add_argument("--implementation-revision", required=True)
    parser.add_argument(
        "--combined-camera-audit-plan",
        type=Path,
        required=True,
        help="Validated all-attempted-camera v5 audit plan.",
    )
    parser.add_argument("--final-camera-audit", type=Path, required=True)
    parser.add_argument(
        "--camera-recovery-lineage",
        type=Path,
        required=True,
        help="Strict JSON object containing only a 'camera_recovery' record.",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    lock = load_deform360_joint_sparse_source_execution_lock_v5(
        arguments.execution_lock
    )
    combined_plan = validate_deform360_joint_sparse_source_prediction_plan_v5(
        load_strict_json_object(
            arguments.combined_camera_audit_plan,
            label="combined camera audit plan",
        ),
        lock=lock,
    )
    raw_objects = combined_plan["objects"]
    if isinstance(raw_objects, (str, bytes)) or not isinstance(raw_objects, Sequence):
        raise ValueError("combined camera audit objects must be a JSON array")
    if not all(isinstance(value, Mapping) for value in raw_objects):
        raise ValueError("every combined-plan object must be a JSON object")
    audit = validate_deform360_joint_sparse_camera_audit_v5_2(
        load_strict_json_object(
            arguments.final_camera_audit, label="final camera audit"
        ),
        lock=lock,
    )
    if audit["base_source_plan_id"] != combined_plan["plan_id"]:
        raise ValueError("final camera audit does not bind the combined audit plan")
    lineage_payload = load_strict_json_object(
        arguments.camera_recovery_lineage, label="camera recovery lineage"
    )
    if set(lineage_payload) != {"camera_recovery"}:
        raise ValueError("recovery lineage must contain only 'camera_recovery'")
    plan = build_deform360_joint_sparse_source_prediction_plan_v5_2(
        lock=lock,
        implementation_revision=arguments.implementation_revision,
        attempted_objects=cast(Sequence[Mapping[str, Any]], raw_objects),
        final_camera_audit=audit,
        camera_recovery=cast(Mapping[str, Any], lineage_payload["camera_recovery"]),
    )
    write_atomic_json(plan, arguments.output, overwrite=False)
    print(plan["plan_id"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
