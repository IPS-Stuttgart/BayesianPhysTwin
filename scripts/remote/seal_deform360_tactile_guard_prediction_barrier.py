#!/usr/bin/env python3
"""Seal or explicitly block the all-case tactile-guard prediction barrier."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from bayesian_phystwin.deform360_tactile_guard_outcome_sealed import (  # noqa: E402
    build_prediction_barrier,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build_prediction_barrier(
        args.output,
        protocol_path=args.protocol,
        prediction_root=args.prediction_root,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["barrier_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
