"""CLI for the hierarchical structural-recovery benchmark."""

from __future__ import annotations

import argparse
import json

from bayesian_phystwin.structural_benchmark import (
    StructuralBenchmarkConfig,
    write_structural_recovery_benchmark,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_json")
    parser.add_argument("--seed", type=int, default=20260712)
    args = parser.parse_args()
    result = write_structural_recovery_benchmark(
        args.output_json,
        StructuralBenchmarkConfig(seed=args.seed),
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
