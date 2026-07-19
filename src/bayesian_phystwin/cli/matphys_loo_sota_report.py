"""Build the post-opening report for the sealed MatPhys LOO22 study."""

from __future__ import annotations

import argparse
import json

from bayesian_phystwin.matphys_loo_sota_report import (
    build_matphys_loo_sota_report,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_root")
    parser.add_argument("selection_summary")
    parser.add_argument("future_summary")
    parser.add_argument("output_path")
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--bootstrap-block-length", type=int, default=5)
    parser.add_argument("--bootstrap-seed", type=int, default=20260719)
    args = parser.parse_args()
    result = build_matphys_loo_sota_report(
        args.data_root,
        args.selection_summary,
        args.future_summary,
        args.output_path,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_block_length=args.bootstrap_block_length,
        bootstrap_seed=args.bootstrap_seed,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
