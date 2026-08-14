"""Run, materialize, or validate the optional Newton MPM backend smoke."""

from __future__ import annotations

import argparse
import json
import tempfile
from collections.abc import Sequence
from pathlib import Path

from bayesian_phystwin.newton_mpm_backend_v1 import (
    materialize_newton_mpm_backend,
    validate_newton_mpm_backend,
)
from bayesian_phystwin.newton_mpm_source_gate_v1 import (
    prepare_source_case,
    score_future_if_authorized,
    score_prefix_gate,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    smoke = commands.add_parser(
        "smoke",
        help="run a synthetic Newton MPM beam and publish a physical artifact",
    )
    smoke.add_argument("output_dir", type=Path)
    smoke.add_argument("--device", default="cuda:0")
    smoke.add_argument("--frames", type=int, default=76)
    smoke.add_argument("--queries", type=int, default=64)
    smoke.add_argument("--fps", type=float, default=120.0)
    smoke.add_argument("--voxel-size-m", type=float, default=0.025)
    smoke.add_argument("--max-iterations", type=int, default=50)

    materialize = commands.add_parser(
        "materialize",
        help="adapt an existing fixed-identity Newton particle rollout",
    )
    materialize.add_argument("raw_rollout", type=Path)
    materialize.add_argument("runtime_manifest", type=Path)
    materialize.add_argument("output_dir", type=Path)

    validate = commands.add_parser("validate", help="validate a published bundle")
    validate.add_argument("output_dir", type=Path)

    prepare_source = commands.add_parser(
        "source-prepare",
        help="prepare the frozen, already-open Newton MPM source case",
    )
    prepare_source.add_argument("protocol", type=Path)
    prepare_source.add_argument("final_data", type=Path)
    prepare_source.add_argument("optimal_params", type=Path)
    prepare_source.add_argument("incumbent_physical", type=Path)
    prepare_source.add_argument("matphys_physical", type=Path)
    prepare_source.add_argument("matphys_replay_result", type=Path)
    prepare_source.add_argument("output_dir", type=Path)

    run_source = commands.add_parser(
        "source-run-grid",
        help="run the outcome-blind frozen Newton MPM source grid",
    )
    run_source.add_argument("protocol", type=Path)
    run_source.add_argument("source_inputs", type=Path)
    run_source.add_argument("output_dir", type=Path)
    run_source.add_argument("--device", default="cuda:0")

    score_source = commands.add_parser(
        "source-score-prefix",
        help="score fit/validation frames and apply the frozen source gate",
    )
    score_source.add_argument("protocol", type=Path)
    score_source.add_argument("source_bundle_dir", type=Path)
    score_source.add_argument("grid_manifest", type=Path)
    score_source.add_argument("incumbent_physical", type=Path)
    score_source.add_argument("matphys_physical", type=Path)
    score_source.add_argument("output_dir", type=Path)

    future_source = commands.add_parser(
        "source-score-future",
        help="score the source future only when the frozen prefix gate authorizes it",
    )
    future_source.add_argument("protocol", type=Path)
    future_source.add_argument("source_bundle_dir", type=Path)
    future_source.add_argument("prefix_result_dir", type=Path)
    future_source.add_argument("grid_manifest", type=Path)
    future_source.add_argument("incumbent_physical", type=Path)
    future_source.add_argument("matphys_physical", type=Path)
    future_source.add_argument("output_path", type=Path)
    return parser


def _smoke(args: argparse.Namespace) -> dict[str, object]:
    try:
        from bayesian_phystwin._newton_mpm_runtime import (
            NewtonMpmSmokeConfig,
            run_newton_mpm_smoke,
        )
    except ImportError as error:
        raise RuntimeError(
            "Newton MPM is optional; install bayesian-phystwin[mpm]"
        ) from error

    config = NewtonMpmSmokeConfig(
        frame_count=args.frames,
        query_count=args.queries,
        fps=args.fps,
        voxel_size_m=args.voxel_size_m,
        max_iterations=args.max_iterations,
    )
    output = args.output_dir.absolute()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{output.name}.runtime.",
        dir=output.parent,
    ) as temporary:
        temporary_root = Path(temporary)
        raw = temporary_root / "newton-particle-rollout.npz"
        runtime = temporary_root / "newton-runtime.json"
        run_newton_mpm_smoke(
            raw_rollout_path=raw,
            runtime_manifest_path=runtime,
            device=args.device,
            config=config,
        )
        return materialize_newton_mpm_backend(
            raw_rollout_path=raw,
            runtime_manifest_path=runtime,
            output_dir=output,
        )


def _run_source_grid(args: argparse.Namespace) -> dict[str, object]:
    try:
        from bayesian_phystwin._newton_mpm_source_runtime import run_source_grid
    except ImportError as error:
        raise RuntimeError(
            "Newton MPM is optional; install bayesian-phystwin[mpm]"
        ) from error
    return run_source_grid(
        protocol_path=args.protocol,
        source_inputs_path=args.source_inputs,
        output_dir=args.output_dir,
        device=args.device,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "smoke":
        result = _smoke(args)
    elif args.command == "materialize":
        result = materialize_newton_mpm_backend(
            raw_rollout_path=args.raw_rollout,
            runtime_manifest_path=args.runtime_manifest,
            output_dir=args.output_dir,
        )
    elif args.command == "validate":
        result = validate_newton_mpm_backend(args.output_dir)
    elif args.command == "source-prepare":
        result = prepare_source_case(
            protocol_path=args.protocol,
            final_data_path=args.final_data,
            optimal_params_path=args.optimal_params,
            incumbent_physical_path=args.incumbent_physical,
            matphys_physical_path=args.matphys_physical,
            matphys_replay_result_path=args.matphys_replay_result,
            output_dir=args.output_dir,
        )
    elif args.command == "source-run-grid":
        result = _run_source_grid(args)
    elif args.command == "source-score-prefix":
        result = score_prefix_gate(
            protocol_path=args.protocol,
            source_bundle_dir=args.source_bundle_dir,
            grid_manifest_path=args.grid_manifest,
            incumbent_physical_path=args.incumbent_physical,
            matphys_physical_path=args.matphys_physical,
            output_dir=args.output_dir,
        )
    elif args.command == "source-score-future":
        result = score_future_if_authorized(
            protocol_path=args.protocol,
            source_bundle_dir=args.source_bundle_dir,
            prefix_result_dir=args.prefix_result_dir,
            grid_manifest_path=args.grid_manifest,
            incumbent_physical_path=args.incumbent_physical,
            matphys_physical_path=args.matphys_physical,
            output_path=args.output_path,
        )
    else:  # pragma: no cover - argparse enforces the command set
        raise AssertionError(f"unhandled command: {args.command}")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
