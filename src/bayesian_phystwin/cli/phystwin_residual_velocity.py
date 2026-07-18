"""CLI for causal recurrent PhysTwin residual-velocity forecasting."""

from __future__ import annotations

import argparse
import json

from bayesian_phystwin.phystwin_residual_velocity import (
    PhysTwinResidualVelocityConfig,
    fit_recurrent_residual_velocity,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fit a recursively integrated low-rank residual-velocity model."
    )
    parser.add_argument("final_data")
    parser.add_argument("baseline_trajectory")
    parser.add_argument("gt_track_3d")
    parser.add_argument("output_dir")
    parser.add_argument("--fit-end-frame", type=int, required=True)
    parser.add_argument("--train-end-frame", type=int, required=True)
    parser.add_argument("--rank", type=int, action="append", dest="ranks")
    parser.add_argument(
        "--velocity-persistence",
        type=float,
        action="append",
        dest="velocity_persistence",
    )
    parser.add_argument("--ridge", type=float, action="append", dest="ridge")
    parser.add_argument("--maximum-residual-m", type=float, default=0.01)
    parser.add_argument("--minimum-dynamic-improvement", type=float, default=0.01)
    parser.add_argument("--maximum-metric-ratio", type=float, default=1.02)
    args = parser.parse_args()
    defaults = PhysTwinResidualVelocityConfig(
        fit_end_frame=args.fit_end_frame,
        train_end_frame=args.train_end_frame,
    )
    summary = fit_recurrent_residual_velocity(
        args.final_data,
        args.baseline_trajectory,
        args.gt_track_3d,
        args.output_dir,
        config=PhysTwinResidualVelocityConfig(
            fit_end_frame=args.fit_end_frame,
            train_end_frame=args.train_end_frame,
            rank_candidates=defaults.rank_candidates if args.ranks is None else tuple(args.ranks),
            velocity_persistence_candidates=(
                defaults.velocity_persistence_candidates
                if args.velocity_persistence is None
                else tuple(args.velocity_persistence)
            ),
            ridge_candidates=(
                defaults.ridge_candidates if args.ridge is None else tuple(args.ridge)
            ),
            maximum_residual_m=args.maximum_residual_m,
            minimum_dynamic_improvement=args.minimum_dynamic_improvement,
            maximum_metric_ratio=args.maximum_metric_ratio,
        ),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
