"""CLI for paired PhysTwin trajectory comparison."""

from __future__ import annotations

import argparse
import json

from bayesian_phystwin.phystwin_comparison import compare_phystwin_manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare paired PhysTwin trajectories with moving-block bootstrap."
    )
    parser.add_argument("manifest")
    parser.add_argument("output_json")
    parser.add_argument("--samples", type=int, default=10000)
    parser.add_argument("--block-length", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cluster-by-phystwin-object", action="store_true")
    args = parser.parse_args()
    result = compare_phystwin_manifest(
        args.manifest,
        args.output_json,
        samples=args.samples,
        block_length=args.block_length,
        seed=args.seed,
        cluster_by_phystwin_object=args.cluster_by_phystwin_object,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
