"""CLI for the frozen sparse online-belief evaluation."""

from __future__ import annotations

import argparse
import json

from bayesian_phystwin.phystwin_online_belief_evaluation import (
    evaluate_online_belief_cohort,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate recursive sparse-observation PhysTwin beliefs."
    )
    parser.add_argument("protocol_json")
    parser.add_argument("output_dir")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = evaluate_online_belief_cohort(args.protocol_json, args.output_dir)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
