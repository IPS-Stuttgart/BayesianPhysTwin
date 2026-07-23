"""CLI for primary-only selected-backbone Deform360 evaluation and parity."""

from __future__ import annotations

import argparse
import json

from bayesian_phystwin.deform360_raw_camera_primary_evaluation import (
    compare_primary_to_gated_cohort,
    evaluate_primary_cohort,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    evaluate = subparsers.add_parser(
        "evaluate-cohort",
        help="Evaluate hash-verified measurements without uncertainty sidecars.",
    )
    evaluate.add_argument("panel_root")
    evaluate.add_argument("measurement_root")
    evaluate.add_argument("output_dir")

    parity = subparsers.add_parser(
        "compare-gated-parity",
        help="Read-only comparison with all existing 8-view gated outputs.",
    )
    parity.add_argument("panel_root")
    parity.add_argument("measurement_root")
    parity.add_argument("gated_reference_root")
    parity.add_argument("--expected-gated-summary-file-sha256", required=True)
    parity.add_argument("--expected-gated-summary-result-sha256", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "evaluate-cohort":
        result = evaluate_primary_cohort(
            args.panel_root,
            args.measurement_root,
            args.output_dir,
        )
    else:
        result = compare_primary_to_gated_cohort(
            args.panel_root,
            args.measurement_root,
            args.gated_reference_root,
            expected_gated_summary_file_sha256=(
                args.expected_gated_summary_file_sha256
            ),
            expected_gated_summary_result_sha256=(
                args.expected_gated_summary_result_sha256
            ),
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
