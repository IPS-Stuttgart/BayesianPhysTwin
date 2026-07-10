"""CLI for matched PhysTwin residual-dynamics comparators."""

from __future__ import annotations

import argparse
import json

from bayesian_phystwin.phystwin_residual_baselines import (
    fit_residual_dynamics_baselines,
)
from bayesian_phystwin.phystwin_residual_dynamics import (
    PhysTwinResidualDynamicsConfig,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fit matched last-residual, autonomous, and DMDc comparators."
    )
    parser.add_argument("final_data")
    parser.add_argument("baseline_trajectory")
    parser.add_argument("gt_track_3d")
    parser.add_argument("output_dir")
    parser.add_argument("--fit-end-frame", type=int, required=True)
    parser.add_argument("--train-end-frame", type=int, required=True)
    parser.add_argument("--maximum-residual-m", type=float, default=0.01)
    args = parser.parse_args()
    summary = fit_residual_dynamics_baselines(
        args.final_data,
        args.baseline_trajectory,
        args.gt_track_3d,
        args.output_dir,
        config=PhysTwinResidualDynamicsConfig(
            fit_end_frame=args.fit_end_frame,
            train_end_frame=args.train_end_frame,
            maximum_residual_m=args.maximum_residual_m,
        ),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
