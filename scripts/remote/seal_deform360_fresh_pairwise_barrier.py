#!/usr/bin/env python3
"""Seal the all-case prediction completeness barrier without reading outcomes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from bayesian_phystwin.deform360_fresh_pairwise_protocol import (
    build_completeness_barrier,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--cohort-lock", type=Path, required=True)
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    barrier = build_completeness_barrier(
        args.output,
        protocol_path=args.protocol,
        cohort_path=args.cohort_lock,
        prediction_root=args.prediction_root,
    )
    print(
        json.dumps(
            {
                "barrier_passed": barrier["barrier_passed"],
                "ordinary_prediction_count": barrier["ordinary_prediction_count"],
                "result_sha256": barrier["result_sha256"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
