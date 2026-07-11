"""CLI for regenerating continuous CoTracker3 and multiview PhysTwin cues."""

from __future__ import annotations

import argparse
import json

from bayesian_phystwin.phystwin_cotracker3_cues import (
    CoTracker3CueConfig,
    build_phystwin_cotracker3_cues,
    write_cotracker3_cue_summary,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Rerun official CoTracker3 on archived PhysTwin queries and build "
            "continuous confidence, cycle, boundary, and multiview cues."
        )
    )
    parser.add_argument("final_data")
    parser.add_argument("raw_case_dir")
    parser.add_argument("checkpoint")
    parser.add_argument("cotracker_root")
    parser.add_argument("output_npz")
    parser.add_argument("--train-end-frame", type=int, required=True)
    parser.add_argument("--base-cues")
    parser.add_argument("--summary-json")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--iterations", type=int, default=6)
    parser.add_argument("--window-length", type=int, default=16)
    parser.add_argument("--minimum-cycle-quality", type=float, default=0.1)
    parser.add_argument("--minimum-multiview-quality", type=float, default=0.1)
    parser.add_argument(
        "--multiview-initial-depth-tolerance-m",
        type=float,
        default=0.02,
    )
    parser.add_argument("--initial-match-tolerance-m", type=float, default=1e-6)
    args = parser.parse_args()
    summary = build_phystwin_cotracker3_cues(
        args.final_data,
        args.raw_case_dir,
        args.checkpoint,
        args.cotracker_root,
        args.output_npz,
        config=CoTracker3CueConfig(
            train_end_frame=args.train_end_frame,
            iterations=args.iterations,
            window_length=args.window_length,
            minimum_cycle_quality=args.minimum_cycle_quality,
            minimum_multiview_quality=args.minimum_multiview_quality,
            multiview_initial_depth_tolerance_m=(
                args.multiview_initial_depth_tolerance_m
            ),
            initial_match_tolerance_m=args.initial_match_tolerance_m,
        ),
        base_cues_path=args.base_cues,
        device=args.device,
    )
    if args.summary_json:
        write_cotracker3_cue_summary(summary, args.summary_json)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
