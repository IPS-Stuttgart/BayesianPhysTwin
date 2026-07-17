#!/usr/bin/env python3
"""Score one previously sealed independent-source prediction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from causal4d_public.deform360_independent_source import (
    evaluate_independent_source_prediction,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--prediction-seal", type=Path, required=True)
    parser.add_argument("--target-data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    seal = json.loads(args.prediction_seal.read_text(encoding="utf-8"))
    result = evaluate_independent_source_prediction(
        seal, args.target_data, lock_path=args.lock
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
