"""CLI for training-only latent PhysTwin controller-bias inference."""

from __future__ import annotations

import argparse
import json

from bayesian_phystwin.phystwin_controller_inference import (
    DEFAULT_BIAS_SCALES_M,
    run_latent_controller_bias,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Infer a regularized latent PhysTwin controller bias."
    )
    parser.add_argument("official_repo")
    parser.add_argument("data_root")
    parser.add_argument("output_dir")
    parser.add_argument(
        "--cohort",
        choices=("all", "development", "confirmation"),
        default="development",
    )
    parser.add_argument("--case", action="append", dest="cases")
    parser.add_argument("--bias-scale-m", action="append", type=float, dest="scales")
    parser.add_argument("--validation-fraction", type=float, default=0.75)
    parser.add_argument("--finite-difference-m", type=float, default=0.001)
    parser.add_argument("--ramp-frames", type=int, default=5)
    parser.add_argument("--maximum-group-norm-m", type=float, default=0.003)
    parser.add_argument("--observation-sigma-m", type=float, default=0.005)
    parser.add_argument("--controller-sigma-m", type=float, default=0.002)
    parser.add_argument("--smoothness-sigma-m", type=float, default=0.0005)
    parser.add_argument("--atomic-spring-forces", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    result = run_latent_controller_bias(
        args.official_repo,
        args.data_root,
        args.output_dir,
        cohort=args.cohort,
        cases=args.cases,
        validation_fraction=args.validation_fraction,
        finite_difference_m=args.finite_difference_m,
        bias_scales_m=(
            DEFAULT_BIAS_SCALES_M if args.scales is None else tuple(args.scales)
        ),
        ramp_frames=args.ramp_frames,
        maximum_group_norm_m=args.maximum_group_norm_m,
        observation_sigma_m=args.observation_sigma_m,
        controller_sigma_m=args.controller_sigma_m,
        smoothness_sigma_m=args.smoothness_sigma_m,
        deterministic_spring_forces=not args.atomic_spring_forces,
        force=args.force,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
