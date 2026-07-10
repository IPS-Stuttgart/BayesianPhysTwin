"""CLI for matched PhysTwin graph-discrepancy anchor comparisons."""

from __future__ import annotations

import argparse
import json

from bayesian_phystwin.phystwin_graph_anchor_comparison import (
    run_graph_anchor_comparison,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare raw, kNN-lifted, and graph-smoothed anchors."
    )
    parser.add_argument("data_root")
    parser.add_argument("output_dir")
    parser.add_argument(
        "--cohort",
        choices=("all", "development", "confirmation"),
        default="all",
    )
    parser.add_argument("--case", action="append", dest="cases")
    parser.add_argument(
        "--prior-strength",
        action="append",
        type=float,
        dest="prior_strengths",
    )
    parser.add_argument("--select-prior-strength", action="store_true")
    parser.add_argument("--covariance-probes", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    result = run_graph_anchor_comparison(
        args.data_root,
        args.output_dir,
        cohort=args.cohort,
        cases=args.cases,
        prior_strengths=(
            (0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0)
            if args.prior_strengths is None
            else tuple(args.prior_strengths)
        ),
        select_prior_strength=args.select_prior_strength,
        covariance_probes=args.covariance_probes,
        force=args.force,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
