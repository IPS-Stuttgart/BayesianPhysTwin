#!/usr/bin/env python3
"""Seal nested Deform360 v5 predictions and assemble source evidence."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from bayesian_phystwin._portable_contracts import load_strict_json_object
from bayesian_phystwin.deform360_joint_sparse_source_evidence_v5 import (
    assemble_deform360_joint_sparse_source_evidence_v5,
    build_deform360_joint_sparse_source_prediction_batch_v5,
    load_source_execution_lock_and_artifacts_v5,
    publish_deform360_joint_sparse_source_evidence_v5,
    publish_deform360_joint_sparse_source_prediction_batch_v5,
)
from bayesian_phystwin.deform360_joint_sparse_source_gate_v5 import (
    load_deform360_joint_sparse_source_execution_lock_v5,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    seal = subparsers.add_parser(
        "seal-batch",
        help="Publish the complete outcome-free 10-by-10 prediction batch.",
    )
    seal.add_argument("--execution-lock", type=Path, required=True)
    seal.add_argument(
        "--prediction-seal",
        type=Path,
        action="append",
        required=True,
        dest="prediction_seals",
    )
    seal.add_argument("--output", type=Path, required=True)

    assemble = subparsers.add_parser(
        "assemble",
        help="Attach scored suffix outcomes to a pre-existing prediction batch.",
    )
    assemble.add_argument("--execution-lock", type=Path, required=True)
    assemble.add_argument("--prediction-batch", type=Path, required=True)
    assemble.add_argument(
        "--outcome",
        type=Path,
        action="append",
        required=True,
        dest="outcomes",
    )
    assemble.add_argument("--output", type=Path, required=True)
    return parser


def _seal_batch(arguments: argparse.Namespace) -> dict[str, Any]:
    lock, seals = load_source_execution_lock_and_artifacts_v5(
        execution_lock_path=arguments.execution_lock,
        artifact_paths=arguments.prediction_seals,
        label="prediction seal",
    )
    batch = build_deform360_joint_sparse_source_prediction_batch_v5(seals, lock)
    return publish_deform360_joint_sparse_source_prediction_batch_v5(
        batch,
        lock=lock,
        output_path=arguments.output,
    )


def _assemble(arguments: argparse.Namespace) -> dict[str, Any]:
    lock = load_deform360_joint_sparse_source_execution_lock_v5(
        arguments.execution_lock
    )
    batch = load_strict_json_object(
        arguments.prediction_batch,
        label="source prediction batch",
    )
    outcomes = [
        load_strict_json_object(path, label="source outcome")
        for path in arguments.outcomes
    ]
    evidence = assemble_deform360_joint_sparse_source_evidence_v5(
        lock=lock,
        prediction_batch=batch,
        outcomes=outcomes,
    )
    return publish_deform360_joint_sparse_source_evidence_v5(
        evidence,
        lock=lock,
        output_path=arguments.output,
    )


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    result = (
        _seal_batch(arguments)
        if arguments.command == "seal-batch"
        else _assemble(arguments)
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
