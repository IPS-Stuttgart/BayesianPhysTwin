#!/usr/bin/env python3
"""Publish the sealed 10-by-10 public Deform360 v5 source forecasts."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from bayesian_phystwin.deform360_joint_sparse_source_runner_v5 import (
    publish_deform360_joint_sparse_source_prediction_panel_v5,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execution-lock", type=Path, required=True)
    parser.add_argument("--source-plan", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    receipt = publish_deform360_joint_sparse_source_prediction_panel_v5(
        execution_lock_path=arguments.execution_lock,
        source_plan_path=arguments.source_plan,
        input_root=arguments.input_root,
        output_root=arguments.output_root,
    )
    print(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
