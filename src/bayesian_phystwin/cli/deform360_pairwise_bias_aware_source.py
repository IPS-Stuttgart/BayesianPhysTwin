"""CLI for the frozen pairwise bias-aware open-27 source comparison."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence

from bayesian_phystwin.deform360_pairwise_bias_aware_source import (
    evaluate_pairwise_bias_aware_source,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the pairwise bias-aware candidate on the already-open "
            "Deform360 27-case source panel."
        )
    )
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--measurement-root", required=True)
    parser.add_argument("--uncertainty-root", required=True)
    parser.add_argument("--selected-baseline-root", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = evaluate_pairwise_bias_aware_source(
        args.source_root,
        args.measurement_root,
        args.uncertainty_root,
        args.selected_baseline_root,
        args.output,
    )
    print(
        json.dumps(
            {
                "aggregate": result["aggregate"],
                "comparisons": result["comparisons"],
                "advancement_gates": result["advancement_gates"],
                "larger_preregistered_run_justified": result[
                    "larger_preregistered_run_justified"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
