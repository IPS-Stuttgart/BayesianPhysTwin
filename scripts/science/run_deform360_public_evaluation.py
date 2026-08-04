#!/usr/bin/env python3
"""Evaluate frozen BayesianPhysTwin endpoint dynamics on public Deform360 data."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from bayesian_phystwin.deform360_public_evaluation import (
    EvaluationLimits,
    evaluate_deform360_public_data,
    write_evaluation,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-archives", type=int, default=64)
    parser.add_argument("--max-frames-per-archive", type=int, default=96)
    parser.add_argument("--max-tracks", type=int, default=2048)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    result = evaluate_deform360_public_data(
        args.data_root,
        limits=EvaluationLimits(
            max_archives=args.max_archives,
            max_frames_per_archive=args.max_frames_per_archive,
            max_tracks=args.max_tracks,
        ),
        revision=os.environ.get("GITHUB_SHA"),
    )
    write_evaluation(args.output, result)
    print(json.dumps(result["summary"], indent=2, sort_keys=True))
    if result["inventory"]["archives_evaluated"] == 0:
        raise SystemExit(
            "no supported Deform360 trajectory archive was found; "
            "inspect the uploaded inventory"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
