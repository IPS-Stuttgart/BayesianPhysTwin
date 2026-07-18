"""CLI for leave-one-episode-out shared residual-velocity development."""

from __future__ import annotations

import argparse
import json

from bayesian_phystwin.phystwin_shared_residual_velocity import (
    SharedResidualVelocityConfig,
    fit_shared_residual_velocity_development,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cross-fit a shared pointwise residual-velocity prior."
    )
    parser.add_argument("manifest")
    parser.add_argument("output_dir")
    parser.add_argument("--smoothing", type=float, action="append", dest="smoothing")
    parser.add_argument(
        "--local-prior-strength", type=float, action="append", dest="prior_strength"
    )
    parser.add_argument("--maximum-training-points", type=int, default=512)
    parser.add_argument("--minimum-development-improvement", type=float, default=0.03)
    args = parser.parse_args()
    defaults = SharedResidualVelocityConfig()
    summary = fit_shared_residual_velocity_development(
        args.manifest,
        args.output_dir,
        config=SharedResidualVelocityConfig(
            smoothing_candidates=(
                defaults.smoothing_candidates
                if args.smoothing is None
                else tuple(args.smoothing)
            ),
            local_prior_strength_candidates=(
                defaults.local_prior_strength_candidates
                if args.prior_strength is None
                else tuple(args.prior_strength)
            ),
            maximum_training_points=args.maximum_training_points,
            minimum_development_improvement=args.minimum_development_improvement,
        ),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
