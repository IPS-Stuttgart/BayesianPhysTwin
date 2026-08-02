#!/usr/bin/env python3
"""Close an impossible tactile-guard advancement without opening outcomes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from bayesian_phystwin.deform360_tactile_guard_preoutcome_result import (
    build_preoutcome_impossibility_result,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--barrier", type=Path, required=True)
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--runtime-revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build_preoutcome_impossibility_result(
        args.output,
        protocol_path=args.protocol,
        barrier_path=args.barrier,
        prediction_root=args.prediction_root,
        runtime_revision=args.runtime_revision,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
