#!/usr/bin/env python3
"""Seal every fresh Deform360 prediction before opening any held outcome."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from causal4d_public.deform360_reusable_trust_protocol import (
    build_reusable_trust_prediction_cohort_seal,
    load_reusable_trust_protocol,
    validate_reusable_trust_prediction_cohort_seal,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-lock", type=Path, required=True)
    parser.add_argument("--physics-addendum", type=Path, required=True)
    parser.add_argument("--execution-lock", type=Path, required=True)
    parser.add_argument("--prediction-json", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    protocol = load_reusable_trust_protocol(
        args.parent_lock, args.physics_addendum, args.execution_lock
    )
    seal = build_reusable_trust_prediction_cohort_seal(
        list(args.prediction_json), protocol=protocol
    )
    validate_reusable_trust_prediction_cohort_seal(
        seal, protocol=protocol, verify_predictions=True
    )
    if args.output.exists():
        raise FileExistsError(f"cohort seal already exists: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(seal, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(seal, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
