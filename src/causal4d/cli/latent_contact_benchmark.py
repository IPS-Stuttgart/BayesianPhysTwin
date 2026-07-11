"""CLI for leave-one-topology-out latent contact inference."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence

from causal4d.benchmark import CounterfactualBenchmarkConfig
from causal4d.contact_evaluation import (
    run_latent_contact_benchmark,
    write_latent_contact_artifacts,
)
from causal4d.contact_inference import LatentContactConfig


def _parse_seeds(value: str) -> list[int]:
    try:
        if ":" in value:
            parts = [int(part) for part in value.split(":")]
            if len(parts) not in {2, 3}:
                raise ValueError
            seeds = list(range(*parts))
        else:
            seeds = [int(part) for part in value.split(",") if part]
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "seeds must be comma-separated or start:stop[:step]"
        ) from error
    if not seeds:
        raise argparse.ArgumentTypeError("at least one seed is required")
    return seeds


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Infer realized graph contacts before and after a short observation "
            "prefix, with leave-one-topology-out controls."
        )
    )
    parser.add_argument("--output-dir", default="runs/causal4d-latent-contact-v1")
    parser.add_argument("--seeds", default="0:5")
    parser.add_argument("--frames", type=int, default=56)
    parser.add_argument("--training-repeats", type=int, default=2)
    parser.add_argument("--parameter-grid-count", type=int, default=5)
    parser.add_argument("--contact-parameter-particles", type=int, default=12)
    parser.add_argument("--observation-fraction", type=float, default=0.20)
    parser.add_argument("--observation-noise-mm", type=float, default=1.5)
    parser.add_argument(
        "--require-gates",
        action="store_true",
        help="return status 2 when any pre-registered milestone gate fails",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    benchmark_config = CounterfactualBenchmarkConfig(
        frame_count=args.frames,
        training_repeats=args.training_repeats,
        parameter_grid_count=args.parameter_grid_count,
        observation_noise_std_m=args.observation_noise_mm / 1000.0,
    )
    contact_config = LatentContactConfig(
        parameter_particle_count=args.contact_parameter_particles,
        observation_fraction=args.observation_fraction,
        observation_noise_std_m=args.observation_noise_mm / 1000.0,
        confidence_level=benchmark_config.confidence_level,
    )
    result = run_latent_contact_benchmark(
        seeds=_parse_seeds(args.seeds),
        benchmark_config=benchmark_config,
        contact_config=contact_config,
    )
    paths = write_latent_contact_artifacts(result, args.output_dir)
    print(
        json.dumps(
            {
                "success_gates": result["success_gates"],
                "aggregate": result["aggregate"],
                "artifacts": paths,
            },
            indent=2,
            sort_keys=True,
        )
    )
    if args.require_gates and not result["success_gates"]["overall_passed"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
