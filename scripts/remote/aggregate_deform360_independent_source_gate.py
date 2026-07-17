#!/usr/bin/env python3
"""Apply the frozen conjunctive gate to all 27 source evaluations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from causal4d_public.deform360_independent_source import (
    aggregate_independent_source_gate,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    evaluations = [
        json.loads(path.read_text(encoding="utf-8")) for path in args.evaluation
    ]
    result = aggregate_independent_source_gate(evaluations, lock_path=args.lock)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
