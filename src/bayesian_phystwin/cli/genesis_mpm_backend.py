"""Run, materialize, or validate the optional Genesis MPM backend smoke."""

from __future__ import annotations

import argparse
import json
import tempfile
from collections.abc import Sequence
from pathlib import Path

from bayesian_phystwin.genesis_mpm_backend_v1 import (
    materialize_genesis_mpm_backend,
    validate_genesis_mpm_backend,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    smoke = commands.add_parser(
        "smoke",
        help="run a synthetic Genesis MPM beam and publish a physical artifact",
    )
    smoke.add_argument("output_dir", type=Path)
    smoke.add_argument("--backend", choices=("gpu", "cpu"), default="gpu")
    smoke.add_argument("--frames", type=int, default=40)
    smoke.add_argument("--queries", type=int, default=64)
    smoke.add_argument("--fps", type=float, default=120.0)
    smoke.add_argument("--substeps", type=int, default=32)
    smoke.add_argument("--grid-density", type=int, default=64)
    smoke.add_argument("--attachment-stiffness", type=float, default=500.0)
    smoke.add_argument("--action-displacement-m", type=float, default=0.010)

    materialize = commands.add_parser(
        "materialize",
        help="adapt an existing fixed-identity Genesis particle rollout",
    )
    materialize.add_argument("raw_rollout", type=Path)
    materialize.add_argument("runtime_manifest", type=Path)
    materialize.add_argument("output_dir", type=Path)

    validate = commands.add_parser("validate", help="validate a published bundle")
    validate.add_argument("output_dir", type=Path)
    return parser


def _smoke(args: argparse.Namespace) -> dict[str, object]:
    try:
        from bayesian_phystwin._genesis_mpm_runtime import (
            GenesisMpmSmokeConfig,
            run_genesis_mpm_smoke,
        )
    except ImportError as error:
        raise RuntimeError(
            "Genesis MPM is optional; install bayesian-phystwin[genesis-mpm]"
        ) from error

    config = GenesisMpmSmokeConfig(
        frame_count=args.frames,
        query_count=args.queries,
        fps=args.fps,
        substeps=args.substeps,
        grid_density=args.grid_density,
        attachment_stiffness=args.attachment_stiffness,
        action_displacement_m=args.action_displacement_m,
    )
    output = args.output_dir.absolute()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{output.name}.runtime.", dir=output.parent
    ) as temporary:
        temporary_root = Path(temporary)
        raw = temporary_root / "genesis-particle-rollout.npz"
        runtime = temporary_root / "genesis-runtime.json"
        run_genesis_mpm_smoke(
            raw_rollout_path=raw,
            runtime_manifest_path=runtime,
            backend=args.backend,
            config=config,
        )
        return materialize_genesis_mpm_backend(
            raw_rollout_path=raw,
            runtime_manifest_path=runtime,
            output_dir=output,
        )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "smoke":
        result = _smoke(args)
    elif args.command == "materialize":
        result = materialize_genesis_mpm_backend(
            raw_rollout_path=args.raw_rollout,
            runtime_manifest_path=args.runtime_manifest,
            output_dir=args.output_dir,
        )
    elif args.command == "validate":
        result = validate_genesis_mpm_backend(args.output_dir)
    else:  # pragma: no cover - argparse enforces the command set
        raise AssertionError(f"unhandled command: {args.command}")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
