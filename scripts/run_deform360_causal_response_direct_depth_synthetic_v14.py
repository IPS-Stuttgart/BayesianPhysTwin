#!/usr/bin/env python3
"""Run and seal the V14 production-path synthetic controls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from bayesian_phystwin.deform360_causal_response_direct_depth_synthetic import (
    run_adaptive_direct_depth_synthetic_v14,
    write_adaptive_direct_depth_synthetic_v14,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "results/sota/"
            "deform360_causal_response_direct_depth_synthetic_v14/"
            "summary.json"
        ),
    )
    args = parser.parse_args()
    result = run_adaptive_direct_depth_synthetic_v14()
    write_adaptive_direct_depth_synthetic_v14(args.output, result)
    print(json.dumps(result.descriptor(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
