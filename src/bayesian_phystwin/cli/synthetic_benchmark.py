"""CLI for the fixed-graph correlated-corruption benchmark."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence

from bayesian_phystwin.synthetic_benchmark import (
    SyntheticBenchmarkConfig,
    run_synthetic_benchmark,
    write_benchmark_csv,
    write_benchmark_json,
    write_reliability_csv,
)


def _parse_seeds(value: str) -> list[int]:
    if ":" in value:
        parts = [int(part) for part in value.split(":")]
        if len(parts) not in {2, 3}:
            raise argparse.ArgumentTypeError("seed range must be start:stop[:step]")
        return list(range(*parts))
    try:
        return [int(part) for part in value.split(",") if part]
    except ValueError as error:
        raise argparse.ArgumentTypeError("seeds must be comma-separated or start:stop") from error


def _parse_choices(value: str, allowed: set[str]) -> list[str]:
    choices = [choice.strip() for choice in value.split(",") if choice.strip()]
    unknown = set(choices) - allowed
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown choices: {', '.join(sorted(unknown))}")
    return choices


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate reliability-aware inference on a fixed spring graph."
    )
    parser.add_argument("--seeds", default="0:10")
    parser.add_argument("--conditions", default="clean,iid,correlated")
    parser.add_argument("--action-modes", default="dynamic,quasi_static")
    parser.add_argument("--output-json")
    parser.add_argument("--output-csv")
    parser.add_argument("--output-reliability-csv")
    parser.add_argument("--steps", type=int, default=90)
    parser.add_argument("--train-steps", type=int, default=60)
    parser.add_argument("--stiffness-count", type=int, default=17)
    parser.add_argument("--damping-count", type=int, default=11)
    parser.add_argument("--control-scale-count", type=int, default=9)
    parser.add_argument("--bias-process-variance", type=float, default=1e-5)
    parser.add_argument("--bias-initial-variance", type=float, default=1e-7)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    seeds = _parse_seeds(args.seeds)
    conditions = _parse_choices(args.conditions, {"clean", "iid", "correlated"})
    action_modes = _parse_choices(args.action_modes, {"dynamic", "quasi_static"})
    config = SyntheticBenchmarkConfig(
        step_count=args.steps,
        train_step_count=args.train_steps,
        stiffness_count=args.stiffness_count,
        damping_count=args.damping_count,
        control_scale_count=args.control_scale_count,
        bias_process_variance=args.bias_process_variance,
        bias_initial_variance=args.bias_initial_variance,
    )
    result = run_synthetic_benchmark(
        seeds=seeds,
        conditions=conditions,
        action_modes=action_modes,
        config=config,
    )
    if args.output_json:
        write_benchmark_json(result, args.output_json)
    if args.output_csv:
        write_benchmark_csv(result, args.output_csv)
    if args.output_reliability_csv:
        write_reliability_csv(result, args.output_reliability_csv)
    print(json.dumps(result["aggregate"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
