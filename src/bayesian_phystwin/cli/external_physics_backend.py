"""List, describe, materialize, or validate external physics backends."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

from bayesian_phystwin._portable_contracts import load_strict_json_object
from bayesian_phystwin.external_physics_backend_v1 import (
    materialize_external_physics_backend,
    validate_external_physics_backend,
    write_external_physics_runtime_manifest,
)
from bayesian_phystwin.physics_backend_registry_v1 import (
    discover_backend_profiles,
    get_backend_profile,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    profiles = commands.add_parser(
        "profiles",
        help="list built-in external backend profiles",
    )
    profiles.add_argument(
        "--include-plugins",
        action="store_true",
        help="load opt-in third-party backend profile entry points",
    )

    runtime = commands.add_parser(
        "runtime",
        help="build a content-addressed runtime manifest for a raw rollout",
    )
    runtime.add_argument("profile_id")
    runtime.add_argument("raw_rollout", type=Path)
    runtime.add_argument("output_manifest", type=Path)
    runtime.add_argument("--engine-revision", required=True)
    runtime.add_argument("--engine-version", required=True)
    runtime.add_argument("--producer-repository", required=True)
    runtime.add_argument("--producer-revision", required=True)
    runtime.add_argument("--coordinate-frame", required=True)
    runtime.add_argument("--time-step-s", type=float, required=True)
    runtime.add_argument("--topology-sha256", required=True)
    runtime.add_argument("--material-model", required=True)
    runtime.add_argument(
        "--observation-end-frame-exclusive",
        type=int,
        required=True,
    )
    runtime.add_argument("--parameterization-json", type=Path)
    runtime.add_argument(
        "--producer-artifact",
        action="append",
        default=[],
        metavar="PATH=SHA256",
        help="bind one producer source/config artifact; may be repeated",
    )
    runtime.add_argument(
        "--include-plugins",
        action="store_true",
        help="load opt-in third-party backend profile entry points",
    )
    runtime.add_argument("--overwrite", action="store_true")

    materialize = commands.add_parser(
        "materialize",
        help="adapt a persistent-entity rollout to the portable physical contract",
    )
    materialize.add_argument("raw_rollout", type=Path)
    materialize.add_argument("runtime_manifest", type=Path)
    materialize.add_argument("output_dir", type=Path)

    validate = commands.add_parser(
        "validate",
        help="validate a published external backend bundle",
    )
    validate.add_argument("output_dir", type=Path)
    return parser


def _producer_artifacts(values: Sequence[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for value in values:
        path, separator, digest = value.partition("=")
        if not separator or not path or not digest:
            raise ValueError("--producer-artifact must use PATH=SHA256")
        if path in result:
            raise ValueError(f"duplicate producer artifact path: {path}")
        result[path] = digest
    return result


def _parameterization(path: Path | None) -> Mapping[str, Any]:
    if path is None:
        return {}
    return cast(
        Mapping[str, Any],
        load_strict_json_object(path, label="parameterization JSON"),
    )


def _runtime(args: argparse.Namespace) -> dict[str, Any]:
    profile = get_backend_profile(
        args.profile_id,
        include_plugins=args.include_plugins,
    )
    return write_external_physics_runtime_manifest(
        output_path=args.output_manifest,
        overwrite=args.overwrite,
        raw_rollout_path=args.raw_rollout,
        profile=profile,
        engine_revision=args.engine_revision,
        engine_version=args.engine_version,
        producer_repository=args.producer_repository,
        producer_revision=args.producer_revision,
        coordinate_frame=args.coordinate_frame,
        time_step_s=args.time_step_s,
        topology_sha256=args.topology_sha256,
        material_model=args.material_model,
        observation_end_frame_exclusive=(
            args.observation_end_frame_exclusive
        ),
        parameterization=_parameterization(args.parameterization_json),
        producer_artifacts=_producer_artifacts(args.producer_artifact),
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "profiles":
        result: object = [
            profile.to_dict()
            for profile in discover_backend_profiles(
                include_plugins=args.include_plugins,
            )
        ]
    elif args.command == "runtime":
        result = _runtime(args)
    elif args.command == "materialize":
        result = materialize_external_physics_backend(
            raw_rollout_path=args.raw_rollout,
            runtime_manifest_path=args.runtime_manifest,
            output_dir=args.output_dir,
        )
    elif args.command == "validate":
        result = validate_external_physics_backend(args.output_dir)
    else:  # pragma: no cover - argparse enforces the command set
        raise AssertionError(f"unhandled command: {args.command}")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
