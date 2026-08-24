#!/usr/bin/env python3
"""Run the target-free controlled BayesianPhysTwin SBC experiment."""

from __future__ import annotations

import argparse

from bayesian_phystwin.synthetic_benchmark import SyntheticBenchmarkConfig
from bayesian_phystwin_experiments.synthetic_benchmark_sbc_v1 import (
    SyntheticBenchmarkSBCConfigV1,
    run_synthetic_benchmark_sbc_v1,
    write_synthetic_benchmark_sbc_v1,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replicates", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument(
        "--likelihood-scale-multipliers",
        default="1,0.5,2",
        help="comma-separated positive values; must include 1",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> int:
    arguments = _parser().parse_args()
    multipliers = tuple(
        float(value)
        for value in arguments.likelihood_scale_multipliers.split(",")
        if value.strip()
    )
    result = run_synthetic_benchmark_sbc_v1(
        config=SyntheticBenchmarkSBCConfigV1(
            replicate_count=arguments.replicates,
            seed=arguments.seed,
            bin_count=arguments.bins,
            likelihood_scale_multipliers=multipliers,
        ),
        benchmark_config=SyntheticBenchmarkConfig(),
    )
    write_synthetic_benchmark_sbc_v1(
        result,
        arguments.output,
        overwrite=arguments.overwrite,
    )
    print(result["result_id"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
