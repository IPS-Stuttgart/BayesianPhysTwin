#!/usr/bin/env python3
"""Evaluate one frozen public Deform360 v5 source evidence artifact."""

from __future__ import annotations

import argparse
from pathlib import Path

from bayesian_phystwin.deform360_joint_sparse_source_gate_v5 import (
    publish_deform360_joint_sparse_source_gate_v5,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--execution-lock", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    result = publish_deform360_joint_sparse_source_gate_v5(
        args.evidence,
        args.execution_lock,
        args.output,
        overwrite=args.overwrite,
    )
    print(result["result_id"])
    print(result["status"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
