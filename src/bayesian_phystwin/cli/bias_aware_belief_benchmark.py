"""CLI for the guarded bias-aware belief synthetic controls."""

from __future__ import annotations

import argparse
import json
from typing import Sequence

from bayesian_phystwin.bias_aware_belief_benchmark import (
    BiasAwareBeliefBenchmarkConfig,
    run_bias_aware_belief_benchmark,
    write_bias_aware_belief_benchmark,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run bias-aware state/bias identifiability controls."
    )
    parser.add_argument("--output-json")
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument("--trials", type=int, default=128)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_bias_aware_belief_benchmark(
        BiasAwareBeliefBenchmarkConfig(
            seed=args.seed,
            trial_count=args.trials,
        )
    )
    if args.output_json:
        write_bias_aware_belief_benchmark(result, args.output_json)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["all_gates_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
