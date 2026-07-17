#!/usr/bin/env python3
"""Evaluate one fresh held prediction after the complete cohort was sealed."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from causal4d_public.deform360_reusable_trust_evaluation import (
    evaluate_reusable_trust_held_prediction,
)
from causal4d_public.deform360_reusable_trust_protocol import (
    load_reusable_trust_protocol,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-lock", type=Path, required=True)
    parser.add_argument("--physics-addendum", type=Path, required=True)
    parser.add_argument("--execution-lock", type=Path, required=True)
    parser.add_argument("--prediction", type=Path, required=True)
    parser.add_argument("--target-data", type=Path, required=True)
    parser.add_argument("--outcome", type=Path, required=True)
    parser.add_argument("--cohort-seal", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    protocol = load_reusable_trust_protocol(
        args.parent_lock, args.physics_addendum, args.execution_lock
    )
    payload = evaluate_reusable_trust_held_prediction(
        json.loads(args.prediction.read_text(encoding="utf-8")),
        target_data_path=args.target_data,
        outcome=json.loads(args.outcome.read_text(encoding="utf-8")),
        cohort_seal=json.loads(args.cohort_seal.read_text(encoding="utf-8")),
        protocol=protocol,
    )
    if args.output.exists():
        raise FileExistsError(f"evaluation already exists: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
