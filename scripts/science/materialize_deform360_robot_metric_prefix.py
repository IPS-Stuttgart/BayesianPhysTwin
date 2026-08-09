#!/usr/bin/env python3
"""Project released Deform360 robot geometry into one causal camera prefix."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from bayesian_phystwin.deform360_robot_metric_prefix import (
    materialize_deform360_robot_metric_prefix,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-source-inventory", type=Path, required=True)
    parser.add_argument("--processed-root", type=Path, required=True)
    parser.add_argument("--object-id", required=True)
    parser.add_argument("--camera-id", required=True)
    parser.add_argument("--processing-revision", required=True)
    parser.add_argument("--target-height", type=int, required=True)
    parser.add_argument("--target-width", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    result = materialize_deform360_robot_metric_prefix(
        prepared_source_inventory_path=arguments.prepared_source_inventory,
        processed_root=arguments.processed_root,
        object_id=arguments.object_id,
        camera_id=arguments.camera_id,
        expected_processing_revision=arguments.processing_revision,
        target_height=arguments.target_height,
        target_width=arguments.target_width,
        output_directory=arguments.output_dir,
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
