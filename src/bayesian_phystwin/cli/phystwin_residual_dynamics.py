"""CLI for causal action-conditioned PhysTwin residual dynamics."""

from __future__ import annotations

import argparse
import json

from bayesian_phystwin.phystwin_residual_dynamics import (
    PhysTwinResidualDynamicsConfig,
    fit_action_conditioned_residual_dynamics,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fit and validate a constrained low-rank PhysTwin residual model."
    )
    parser.add_argument("final_data")
    parser.add_argument("baseline_trajectory")
    parser.add_argument("gt_track_3d")
    parser.add_argument("output_dir")
    parser.add_argument("--fit-end-frame", type=int, required=True)
    parser.add_argument("--train-end-frame", type=int, required=True)
    parser.add_argument("--rank", type=int, action="append", dest="ranks")
    parser.add_argument(
        "--persistence",
        type=float,
        action="append",
        dest="persistence",
    )
    parser.add_argument("--ridge", type=float, action="append", dest="ridge")
    parser.add_argument("--maximum-residual-m", type=float, default=0.03)
    args = parser.parse_args()
    defaults = PhysTwinResidualDynamicsConfig(
        fit_end_frame=args.fit_end_frame,
        train_end_frame=args.train_end_frame,
    )
    summary = fit_action_conditioned_residual_dynamics(
        args.final_data,
        args.baseline_trajectory,
        args.gt_track_3d,
        args.output_dir,
        config=PhysTwinResidualDynamicsConfig(
            fit_end_frame=args.fit_end_frame,
            train_end_frame=args.train_end_frame,
            rank_candidates=defaults.rank_candidates if args.ranks is None else tuple(args.ranks),
            persistence_candidates=(
                defaults.persistence_candidates
                if args.persistence is None
                else tuple(args.persistence)
            ),
            ridge_candidates=(
                defaults.ridge_candidates if args.ridge is None else tuple(args.ridge)
            ),
            maximum_residual_m=args.maximum_residual_m,
        ),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
