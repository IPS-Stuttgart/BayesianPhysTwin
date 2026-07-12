"""CLI for the official-Warp discrepancy-localization diagnostic."""

from __future__ import annotations

import argparse
import json

from bayesian_phystwin.phystwin_discrepancy_localization import (
    evaluate_phystwin_discrepancy_localization_case,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("official_repo")
    parser.add_argument("final_data")
    parser.add_argument("baseline_trajectory")
    parser.add_argument("optimal_params")
    parser.add_argument("checkpoint")
    parser.add_argument("parameter_profile")
    parser.add_argument("twin_belief")
    parser.add_argument("gt_track")
    parser.add_argument("output_dir")
    parser.add_argument("--train-end-frame", type=int, required=True)
    parser.add_argument("--num-substeps", type=int, default=667)
    parser.add_argument("--force-step-n", type=float, default=0.01)
    parser.add_argument("--structural-step-m", type=float, default=0.002)
    args = parser.parse_args()
    result = evaluate_phystwin_discrepancy_localization_case(
        args.official_repo,
        args.final_data,
        args.baseline_trajectory,
        args.optimal_params,
        args.checkpoint,
        args.parameter_profile,
        args.twin_belief,
        args.gt_track,
        args.output_dir,
        train_end_frame=args.train_end_frame,
        num_substeps=args.num_substeps,
        force_step_n=args.force_step_n,
        structural_step_m=args.structural_step_m,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
