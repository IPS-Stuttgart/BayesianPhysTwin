"""CLI for main-cohort PhysTwin endpoint spatial-mode diagnosis."""

from __future__ import annotations

import argparse
import json

from bayesian_phystwin.phystwin_spatial_mode_analysis import (
    run_spatial_mode_analysis,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Diagnose endpoint anchors with rigid and affine controls."
    )
    parser.add_argument("data_root")
    parser.add_argument("output_dir")
    parser.add_argument(
        "--cohort",
        choices=("all", "development", "confirmation"),
        default="confirmation",
    )
    parser.add_argument("--case", action="append", dest="cases")
    parser.add_argument("--ground-band-m", type=float, default=0.01)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    result = run_spatial_mode_analysis(
        args.data_root,
        args.output_dir,
        cohort=args.cohort,
        cases=args.cases,
        ground_band_m=args.ground_band_m,
        force=args.force,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
