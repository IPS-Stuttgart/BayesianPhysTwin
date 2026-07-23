"""Evaluate the target-free adaptive covariance-ranked Deform360 predictor."""

from __future__ import annotations

import argparse
import json

from bayesian_phystwin.deform360_adaptive_covariance_evaluation import (
    evaluate_adaptive_covariance_cohort,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("panel_root")
    parser.add_argument("four_view_measurement_root")
    parser.add_argument("four_view_uncertainty_root")
    parser.add_argument("eight_view_measurement_root")
    parser.add_argument("eight_view_uncertainty_root")
    parser.add_argument("output_dir")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = evaluate_adaptive_covariance_cohort(
        args.panel_root,
        {
            4: args.four_view_measurement_root,
            8: args.eight_view_measurement_root,
        },
        {
            4: args.four_view_uncertainty_root,
            8: args.eight_view_uncertainty_root,
        },
        args.output_dir,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
