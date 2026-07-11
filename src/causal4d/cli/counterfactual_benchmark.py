"""CLI for the controlled Causal4D counterfactual benchmark."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence

from causal4d.benchmark import CounterfactualBenchmarkConfig
from causal4d.evaluation import (
    run_counterfactual_benchmark,
    write_benchmark_artifacts,
)


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
            "Evaluate generative, physical, and hybrid models on held-out "
            "deformable-object interventions."
        )
    )
    parser.add_argument("--output-dir", default="results/causal4d_counterfactual")
    parser.add_argument("--seeds", default="0:5")
    parser.add_argument("--frames", type=int, default=56)
    parser.add_argument("--training-repeats", type=int, default=2)
    parser.add_argument("--parameter-grid-count", type=int, default=5)
    parser.add_argument("--observation-noise-mm", type=float, default=1.5)
    parser.add_argument("--inference-noise-mm", type=float, default=6.0)
    parser.add_argument("--likelihood-power", type=float, default=0.12)
    parser.add_argument("--world-control-rotation-deg", type=float, default=8.0)
    parser.add_argument("--world-nonlinearity", type=float, default=0.18)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = CounterfactualBenchmarkConfig(
        frame_count=args.frames,
        training_repeats=args.training_repeats,
        parameter_grid_count=args.parameter_grid_count,
        observation_noise_std_m=args.observation_noise_mm / 1000.0,
        inference_noise_std_m=args.inference_noise_mm / 1000.0,
        likelihood_power=args.likelihood_power,
        world_control_rotation_deg=args.world_control_rotation_deg,
        world_nonlinear_stiffening=args.world_nonlinearity,
    )
    result = run_counterfactual_benchmark(
        seeds=_parse_seeds(args.seeds),
        config=config,
    )
    paths = write_benchmark_artifacts(result, args.output_dir)
    print(
        json.dumps(
            {"aggregate": result["aggregate"], "artifacts": paths},
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
