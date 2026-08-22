"""CLI for the controlled recursive corruption benchmark."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence

from bayesian_phystwin.recursive_corruption_benchmark import (
    CONDITIONS,
    RecursiveCorruptionBenchmarkConfig,
    run_recursive_corruption_benchmark,
    write_recursive_corruption_csv,
    write_recursive_corruption_json,
)


def _parse_seeds(value: str) -> list[int]:
    try:
        if ":" in value:
            parts = [int(part) for part in value.split(":")]
            if len(parts) not in {2, 3}:
                raise argparse.ArgumentTypeError("seed range must be start:stop[:step]")
            return list(range(*parts))
        return [int(part) for part in value.split(",") if part]
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "seeds must be comma-separated or start:stop[:step]"
        ) from error


def _parse_conditions(value: str) -> list[str]:
    selected = [condition.strip() for condition in value.split(",") if condition]
    unknown = sorted(set(selected) - set(CONDITIONS))
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown conditions: {unknown}")
    return selected


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare physical, residual-persistence, recursive, and guarded "
            "one-step forecasts under controlled observation corruption."
        )
    )
    parser.add_argument("--seeds", default="0:10")
    parser.add_argument("--conditions", default=",".join(CONDITIONS))
    parser.add_argument("--output-json")
    parser.add_argument("--output-csv")
    parser.add_argument("--steps", type=int, default=180)
    parser.add_argument("--corruption-start", type=int, default=60)
    parser.add_argument("--corruption-length", type=int, default=30)
    parser.add_argument("--recovery-window", type=int, default=45)
    parser.add_argument("--minimum-reliability", type=float, default=0.50)
    parser.add_argument("--maximum-nis", type=float, default=9.0)
    parser.add_argument("--maximum-update-m", type=float, default=0.025)
    parser.add_argument("--maximum-residual-m", type=float, default=0.080)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = RecursiveCorruptionBenchmarkConfig(
        step_count=args.steps,
        corruption_start=args.corruption_start,
        corruption_length=args.corruption_length,
        recovery_window=args.recovery_window,
        minimum_reliability=args.minimum_reliability,
        maximum_nis=args.maximum_nis,
        maximum_update_m=args.maximum_update_m,
        maximum_residual_m=args.maximum_residual_m,
    )
    result = run_recursive_corruption_benchmark(
        seeds=_parse_seeds(args.seeds),
        conditions=_parse_conditions(args.conditions),
        config=config,
    )
    if args.output_json:
        write_recursive_corruption_json(result, args.output_json)
    if args.output_csv:
        write_recursive_corruption_csv(result, args.output_csv)
    print(json.dumps(result["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - module execution
    raise SystemExit(main())
