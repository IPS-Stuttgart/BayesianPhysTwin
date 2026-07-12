"""CLI for the official-Warp structural calibration diagnostic."""

from __future__ import annotations

import argparse
import json

from bayesian_phystwin.phystwin_structural_diagnostic import (
    evaluate_phystwin_structural_case,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("official_repo")
    parser.add_argument("final_data")
    parser.add_argument("baseline_trajectory")
    parser.add_argument("optimal_params")
    parser.add_argument("checkpoint")
    parser.add_argument("gt_track")
    parser.add_argument("output_dir")
    parser.add_argument("--train-end-frame", type=int, required=True)
    parser.add_argument("--graph-persistence-archive")
    parser.add_argument("--maximum-fit-tracks", type=int, default=512)
    parser.add_argument("--settle-steps", type=int, default=30)
    parser.add_argument("--basis-step", type=float, default=0.05)
    parser.add_argument("--num-substeps", type=int, default=667)
    args = parser.parse_args()
    result = evaluate_phystwin_structural_case(
        args.official_repo,
        args.final_data,
        args.baseline_trajectory,
        args.optimal_params,
        args.checkpoint,
        args.output_dir,
        train_end_frame=args.train_end_frame,
        gt_track_path=args.gt_track,
        graph_persistence_archive_path=args.graph_persistence_archive,
        maximum_fit_tracks=args.maximum_fit_tracks,
        settle_steps=args.settle_steps,
        basis_step=args.basis_step,
        num_substeps=args.num_substeps,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
