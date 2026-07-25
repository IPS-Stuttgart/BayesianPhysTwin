#!/usr/bin/env python3
"""Build one leakage-safe AllTracker cue artifact for a PhysTwin case."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from bayesian_phystwin.phystwin_alltracker_cues import (
    PhysTwinAllTrackerCueConfig,
    build_phystwin_alltracker_cues,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--final-data", type=Path, required=True)
    parser.add_argument("--raw-case-dir", type=Path, required=True)
    parser.add_argument("--alltracker-source", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-end-frame", type=int, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-side", type=int, default=512)
    parser.add_argument("--iterations", type=int, default=4)
    parser.add_argument("--window-length", type=int, default=16)
    parser.add_argument("--minimum-cycle-quality", type=float, default=0.5)
    parser.add_argument("--visibility-threshold", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_phystwin_alltracker_cues(
        args.final_data,
        args.raw_case_dir,
        args.alltracker_source,
        args.checkpoint,
        args.output,
        config=PhysTwinAllTrackerCueConfig(
            train_end_frame=args.train_end_frame,
            max_side=args.max_side,
            inference_iterations=args.iterations,
            window_length=args.window_length,
            minimum_cycle_quality=args.minimum_cycle_quality,
            visibility_threshold=args.visibility_threshold,
        ),
        device=args.device,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
