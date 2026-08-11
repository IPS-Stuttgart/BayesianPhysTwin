#!/usr/bin/env python3
"""Materialize v5.2 public Deform360 endpoint archives after panel sealing."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from bayesian_phystwin.deform360_joint_sparse_public_endpoint_v5_2 import (
    materialize_public_endpoint_inputs_v5_2,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execution-lock", type=Path, required=True)
    parser.add_argument("--source-prediction-plan", type=Path, required=True)
    parser.add_argument("--source-prediction-root", type=Path, required=True)
    parser.add_argument("--processing-lock", type=Path, required=True)
    parser.add_argument("--processed-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--objects-output", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    manifest = materialize_public_endpoint_inputs_v5_2(
        execution_lock_path=args.execution_lock,
        source_prediction_plan_path=args.source_prediction_plan,
        source_prediction_root=args.source_prediction_root,
        processing_lock_path=args.processing_lock,
        processed_root=args.processed_root,
        output_root=args.output_root,
        objects_output_path=args.objects_output,
        manifest_output_path=args.manifest_output,
    )
    print(manifest["manifest_id"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
