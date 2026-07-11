"""CLI for matched PhysTwin controller-jitter sensitivity."""

from __future__ import annotations

import argparse
import json

from bayesian_phystwin.phystwin_controller_sensitivity import (
    DEFAULT_JITTER_SCALES_M,
    run_controller_jitter_sensitivity,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure PhysTwin sensitivity to smooth controller error."
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
    parser.add_argument("--jitter-scale-m", action="append", type=float, dest="scales")
    parser.add_argument("--antithetic-pair-count", type=int, default=4)
    parser.add_argument("--correlation-frames", type=float, default=5.0)
    parser.add_argument("--atomic-spring-forces", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    result = run_controller_jitter_sensitivity(
        args.official_repo,
        args.data_root,
        args.output_dir,
        cohort=args.cohort,
        cases=args.cases,
        jitter_scales_m=(
            DEFAULT_JITTER_SCALES_M if args.scales is None else tuple(args.scales)
        ),
        antithetic_pair_count=args.antithetic_pair_count,
        correlation_frames=args.correlation_frames,
        deterministic_spring_forces=not args.atomic_spring_forces,
        force=args.force,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
