#!/usr/bin/env python3
"""Copy and attest one released Deform360 v5 physical-source episode."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from bayesian_phystwin.deform360_joint_sparse_physical_source_v5 import (
    materialize_joint_sparse_physical_source_v5,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execution-lock", type=Path, required=True)
    parser.add_argument("--prepared-source-inventory", type=Path, required=True)
    parser.add_argument("--processed-root", type=Path, required=True)
    parser.add_argument("--object-id", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    result = materialize_joint_sparse_physical_source_v5(
        execution_lock_path=args.execution_lock,
        prepared_source_inventory_path=args.prepared_source_inventory,
        processed_root=args.processed_root,
        object_id=args.object_id,
        output_root=args.output_root,
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
