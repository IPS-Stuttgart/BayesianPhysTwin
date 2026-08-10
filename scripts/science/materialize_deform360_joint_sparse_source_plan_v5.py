#!/usr/bin/env python3
"""Freeze the prefix-only input plan for the public Deform360 v5 source panel."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

from bayesian_phystwin._portable_contracts import (
    load_strict_json_object,
    write_atomic_json,
)
from bayesian_phystwin.deform360_joint_sparse_source_gate_v5 import (
    load_deform360_joint_sparse_source_execution_lock_v5,
)
from bayesian_phystwin.deform360_joint_sparse_source_runner_v5 import (
    build_deform360_joint_sparse_source_prediction_plan_v5,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execution-lock", type=Path, required=True)
    parser.add_argument("--implementation-revision", required=True)
    parser.add_argument(
        "--objects",
        type=Path,
        required=True,
        help="Strict JSON object containing only an 'objects' array.",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    lock = load_deform360_joint_sparse_source_execution_lock_v5(
        arguments.execution_lock
    )
    payload = load_strict_json_object(arguments.objects, label="source plan inputs")
    if set(payload) != {"objects"}:
        raise ValueError("source plan inputs must contain only 'objects'")
    raw_objects = payload["objects"]
    if isinstance(raw_objects, (str, bytes)) or not isinstance(raw_objects, Sequence):
        raise ValueError("source plan objects must be a JSON array")
    if not all(isinstance(value, Mapping) for value in raw_objects):
        raise ValueError("every source plan object must be a JSON object")
    plan = build_deform360_joint_sparse_source_prediction_plan_v5(
        lock=lock,
        implementation_revision=arguments.implementation_revision,
        objects=cast(Sequence[Mapping[str, Any]], raw_objects),
    )
    write_atomic_json(plan, arguments.output, overwrite=False)
    print(plan["plan_id"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
