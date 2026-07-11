"""CLI for leave-one-interaction-out hierarchical residual magnitude scaling."""

from __future__ import annotations

import argparse
import json

from bayesian_phystwin.phystwin_residual_shrinkage import (
    HierarchicalResidualShrinkageProtocol,
    run_hierarchical_residual_shrinkage,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replace the residual cap with a hierarchical RMS scale."
    )
    parser.add_argument("data_root")
    parser.add_argument("output_dir")
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--bootstrap-block-length", type=int, default=5)
    parser.add_argument("--bootstrap-seed", type=int, default=20260711)
    args = parser.parse_args()
    result = run_hierarchical_residual_shrinkage(
        args.data_root,
        args.output_dir,
        protocol=HierarchicalResidualShrinkageProtocol(
            bootstrap_samples=args.bootstrap_samples,
            bootstrap_block_length=args.bootstrap_block_length,
            bootstrap_seed=args.bootstrap_seed,
        ),
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
