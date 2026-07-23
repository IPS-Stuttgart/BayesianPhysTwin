"""CLI for the frozen selected-RBF Deform360 camera-budget transfer."""

from __future__ import annotations

import argparse
import json

from bayesian_phystwin.deform360_primary_camera_budget_transfer import (
    analyze_primary_camera_budget_transfer,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config")
    parser.add_argument(
        "--output",
        help="Must exactly equal the prospectively frozen analysis_output path.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = analyze_primary_camera_budget_transfer(
        args.config,
        output=args.output,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
