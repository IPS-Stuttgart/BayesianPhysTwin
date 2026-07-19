"""CLI for the open Deform360 online-belief transfer protocol."""

from __future__ import annotations

import argparse
import json

from bayesian_phystwin.deform360_online_belief_evaluation import (
    evaluate_deform360_online_belief_cohort,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate risk-limited sparse online beliefs on the fixed, "
            "outcome-open Deform360 source panel."
        )
    )
    parser.add_argument("cohort_root")
    parser.add_argument("output_dir")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = evaluate_deform360_online_belief_cohort(
        args.cohort_root,
        args.output_dir,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
