"""CLI for the frozen pairwise bias-aware open-27 source comparison."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence

from bayesian_phystwin.deform360_pairwise_bias_aware_source import (
    evaluate_pairwise_bias_aware_source,
)
from bayesian_phystwin.deform360_pairwise_bias_aware_transfer import (
    validate_open27_transfer_bundle,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the pairwise bias-aware candidate on the already-open "
            "Deform360 27-case source panel."
        )
    )
    parser.add_argument(
        "--bundle-root",
        required=True,
        help="Complete open-27 bundle validated before source outcomes are scored.",
    )
    parser.add_argument("--output", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    transfer = validate_open27_transfer_bundle(args.bundle_root)
    roots = transfer["roots"]
    result = evaluate_pairwise_bias_aware_source(
        roots["source"],
        roots["measurement"],
        roots["uncertainty"],
        roots["selected_baseline"],
        args.output,
        transfer_manifest_sha256=transfer["manifest_sha256"],
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
