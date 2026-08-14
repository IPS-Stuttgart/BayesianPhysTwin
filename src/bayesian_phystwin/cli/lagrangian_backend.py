"""Discover, materialize, or validate registered external material backends."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from bayesian_phystwin.material_backend_v1 import (
    describe_material_backend_profiles,
    materialize_material_backend,
    validate_material_backend,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser(
        "profiles",
        help="list canonical simulator families and transport variants",
    )
    materialize = commands.add_parser(
        "materialize",
        help="adapt one registered external solver export",
    )
    materialize.add_argument("raw_rollout", type=Path)
    materialize.add_argument("runtime_manifest", type=Path)
    materialize.add_argument("output_dir", type=Path)
    materialize.add_argument(
        "--profile",
        help=(
            "optional canonical family or producer-profile assertion; the "
            "runtime manifest remains authoritative"
        ),
    )
    validate = commands.add_parser(
        "validate",
        help="auto-detect and validate one published backend bundle",
    )
    validate.add_argument("output_dir", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "profiles":
        result: object = describe_material_backend_profiles()
    elif args.command == "materialize":
        result = materialize_material_backend(
            raw_rollout_path=args.raw_rollout,
            runtime_manifest_path=args.runtime_manifest,
            output_dir=args.output_dir,
            profile_id=args.profile,
        )
    elif args.command == "validate":
        result = validate_material_backend(args.output_dir)
    else:  # pragma: no cover - argparse enforces the command set
        raise AssertionError(f"unhandled command: {args.command}")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
