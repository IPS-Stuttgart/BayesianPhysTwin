#!/usr/bin/env python3
"""Seal one driven/zero Deform360 prediction before future scoring."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from causal4d_public.deform360_independent_source import (
    seal_independent_source_prediction,
    validate_independent_source_prediction_seal,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--object-id", required=True)
    parser.add_argument("--episode-id", type=int, required=True)
    parser.add_argument("--prediction-data", type=Path, required=True)
    parser.add_argument("--simulator-data", type=Path, required=True)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--readout", type=Path, required=True)
    parser.add_argument("--twin-summary", type=Path, required=True)
    parser.add_argument("--driven-result", type=Path, required=True)
    parser.add_argument("--zero-result", type=Path, required=True)
    parser.add_argument("--prediction-archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    seal = seal_independent_source_prediction(
        args.prediction_archive,
        lock_path=args.lock,
        object_id=args.object_id,
        episode_id=args.episode_id,
        prediction_data_path=args.prediction_data,
        simulator_data_path=args.simulator_data,
        graph_path=args.graph,
        readout_path=args.readout,
        twin_summary_path=args.twin_summary,
        driven_result_path=args.driven_result,
        zero_result_path=args.zero_result,
    )
    validate_independent_source_prediction_seal(seal, verify_archive=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(seal, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(seal, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
