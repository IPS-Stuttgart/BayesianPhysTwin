"""CLI for disjoint causal MatPhys graph-part family selection."""

from __future__ import annotations

import argparse
import json

from bayesian_phystwin.matphys_part_family_gate import (
    run_matphys_part_family_gate,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Select a learned MatPhys graph-part residual against its paired "
            "exact teacher using only the disjoint released-prefix suffix."
        )
    )
    parser.add_argument("data_root")
    parser.add_argument("output_dir")
    parser.add_argument("candidate_manifest")
    parser.add_argument("--cases", help="Comma-separated ordered source panel.")
    parser.add_argument(
        "--minimum-relative-score-improvement",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--maximum-metric-regression",
        type=float,
        default=0.0,
    )
    parser.add_argument("--required-learned-case-count", type=int, default=1)
    parser.add_argument("--source-protocol")
    args = parser.parse_args()
    cases = None
    if args.cases:
        cases = tuple(value.strip() for value in args.cases.split(",") if value.strip())
    result = run_matphys_part_family_gate(
        args.data_root,
        args.output_dir,
        args.candidate_manifest,
        case_names=cases,
        minimum_relative_score_improvement=(
            args.minimum_relative_score_improvement
        ),
        maximum_metric_regression=args.maximum_metric_regression,
        required_learned_case_count=args.required_learned_case_count,
        source_protocol=args.source_protocol,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
