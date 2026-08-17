#!/usr/bin/env python3
"""Freeze v5.2 development endpoint files after all forecasts are sealed."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

from bayesian_phystwin._portable_contracts import (
    load_strict_json_object,
    write_atomic_json,
)
from bayesian_phystwin.deform360_joint_sparse_source_evidence_v5 import (
    validate_deform360_joint_sparse_source_prediction_batch_v5,
)
from bayesian_phystwin.deform360_joint_sparse_source_gate_v5 import (
    load_deform360_joint_sparse_source_execution_lock_v5,
)
from bayesian_phystwin.deform360_joint_sparse_source_runner_v5 import _sha256_file
from bayesian_phystwin.deform360_joint_sparse_source_runner_v5_2 import (
    validate_deform360_joint_sparse_source_prediction_plan_v5_2,
    validate_deform360_joint_sparse_source_prediction_receipt_v5_2,
)
from bayesian_phystwin.deform360_joint_sparse_source_scoring_v5_2 import (
    build_deform360_joint_sparse_source_endpoint_plan_v5_2,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execution-lock", type=Path, required=True)
    parser.add_argument("--source-prediction-plan", type=Path, required=True)
    parser.add_argument("--source-prediction-root", type=Path, required=True)
    parser.add_argument("--objects", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    lock = load_deform360_joint_sparse_source_execution_lock_v5(
        arguments.execution_lock
    )
    source_plan = validate_deform360_joint_sparse_source_prediction_plan_v5_2(
        load_strict_json_object(
            arguments.source_prediction_plan, label="v5.2 source prediction plan"
        ),
        lock=lock,
    )
    batch_path = arguments.source_prediction_root / "source-prediction-batch.json"
    receipt_path = arguments.source_prediction_root / "source-prediction-receipt.json"
    batch = validate_deform360_joint_sparse_source_prediction_batch_v5(
        load_strict_json_object(batch_path, label="source prediction batch"), lock
    )
    receipt = validate_deform360_joint_sparse_source_prediction_receipt_v5_2(
        load_strict_json_object(receipt_path, label="v5.2 source prediction receipt"),
        lock=lock,
        plan=source_plan,
        prediction_batch=batch,
        prediction_batch_file_sha256=_sha256_file(batch_path),
    )

    # Endpoint names are intentionally read only after the forecast receipt passes.
    payload = load_strict_json_object(arguments.objects, label="source endpoint inputs")
    if set(payload) != {"objects"}:
        raise ValueError("source endpoint inputs must contain only 'objects'")
    raw_objects = payload["objects"]
    if isinstance(raw_objects, (str, bytes)) or not isinstance(raw_objects, Sequence):
        raise ValueError("source endpoint objects must be a JSON array")
    if not all(isinstance(value, Mapping) for value in raw_objects):
        raise ValueError("every source endpoint object must be a JSON object")
    plan = build_deform360_joint_sparse_source_endpoint_plan_v5_2(
        lock=lock,
        source_prediction_plan=source_plan,
        prediction_batch=batch,
        source_prediction_receipt=receipt,
        objects=cast(Sequence[Mapping[str, Any]], raw_objects),
    )
    write_atomic_json(plan, arguments.output, overwrite=False)
    print(plan["endpoint_plan_id"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
