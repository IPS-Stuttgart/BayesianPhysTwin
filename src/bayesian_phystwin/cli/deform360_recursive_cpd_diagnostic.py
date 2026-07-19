"""CLI for the bounded open-panel recursive CPD diagnostic."""

from __future__ import annotations

import argparse
import json

from bayesian_phystwin.deform360_recursive_cpd_diagnostic import (
    evaluate_deform360_recursive_cpd_cohort,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare independent and recursively tempered CPD updates under fixed "
            "clean/noise/outlier/occlusion conditions on the already-open panel."
        )
    )
    parser.add_argument("cohort_root")
    parser.add_argument("output_dir")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = evaluate_deform360_recursive_cpd_cohort(
        args.cohort_root,
        args.output_dir,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
