"""CLI for matched hard-cap and hierarchical residual-scale comparison."""

from __future__ import annotations

import argparse
import json

from bayesian_phystwin.phystwin_residual_scale_comparison import (
    compare_residual_magnitude_methods,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare 10/30 mm caps and hierarchical residual shrinkage."
    )
    parser.add_argument("data_root")
    parser.add_argument("shrinkage_run_dir")
    parser.add_argument("cap_control_run_dir")
    parser.add_argument("output_json")
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--bootstrap-block-length", type=int, default=5)
    parser.add_argument("--bootstrap-seed", type=int, default=20260711)
    args = parser.parse_args()
    result = compare_residual_magnitude_methods(
        args.data_root,
        args.shrinkage_run_dir,
        args.cap_control_run_dir,
        args.output_json,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_block_length=args.bootstrap_block_length,
        bootstrap_seed=args.bootstrap_seed,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
