"""Describe, materialize, or validate external Lagrangian backend exports."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from bayesian_phystwin.lagrangian_backend_v1 import (
    describe_lagrangian_backend_profiles,
    materialize_lagrangian_backend,
    validate_lagrangian_backend,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser(
        "profiles",
        help="list admitted JAX-FEM and Genesis World producer profiles",
    )
    materialize = commands.add_parser(
        "materialize",
        help="adapt one fixed-identity external solver export",
    )
    materialize.add_argument("raw_rollout", type=Path)
    materialize.add_argument("runtime_manifest", type=Path)
    materialize.add_argument("output_dir", type=Path)
    validate = commands.add_parser("validate", help="validate a published bundle")
    validate.add_argument("output_dir", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "profiles":
        result: object = describe_lagrangian_backend_profiles()
    elif args.command == "materialize":
        result = materialize_lagrangian_backend(
            raw_rollout_path=args.raw_rollout,
            runtime_manifest_path=args.runtime_manifest,
            output_dir=args.output_dir,
        )
    elif args.command == "validate":
        result = validate_lagrangian_backend(args.output_dir)
    else:  # pragma: no cover - argparse enforces the command set
        raise AssertionError(f"unhandled command: {args.command}")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
