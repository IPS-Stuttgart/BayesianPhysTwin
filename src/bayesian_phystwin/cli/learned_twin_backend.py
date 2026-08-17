"""Inspect, materialize, or validate portable learned-twin backend artifacts."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from bayesian_phystwin.learned_twin_backend_v1 import (
    LEARNED_TWIN_MODES,
    describe_learned_twin_profiles,
    materialize_learned_twin_backend,
    validate_learned_twin_backend,
)


def _key_value(values: Sequence[str], *, name: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for value in values:
        key, separator, item = value.partition("=")
        if not separator or not key or not item:
            raise ValueError(f"{name} must use KEY=VALUE syntax")
        if key in result:
            raise ValueError(f"duplicate {name} key: {key}")
        result[key] = item
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("profiles", help="print the frozen support matrix")

    build = commands.add_parser(
        "build",
        help="publish one producer-generated physical_rollout_v1 bundle",
    )
    build.add_argument("source_rollout", type=Path)
    build.add_argument("output_dir", type=Path)
    build.add_argument("--profile", required=True)
    build.add_argument("--mode", choices=sorted(LEARNED_TWIN_MODES), required=True)
    build.add_argument(
        "--model-artifact", action="append", default=[], metavar="PATH=FILE"
    )
    build.add_argument(
        "--source-artifact", action="append", default=[], metavar="PATH=SHA256"
    )
    build.add_argument("--producer-repository", required=True)
    build.add_argument("--producer-revision", required=True)
    build.add_argument("--case-id", required=True)
    build.add_argument("--target-object-id", required=True)
    build.add_argument("--training-object-id", action="append", default=[])
    build.add_argument("--evidence-start", required=True, type=int)
    build.add_argument("--evidence-stop", required=True, type=int)
    build.add_argument("--rollout-start", required=True, type=int)
    build.add_argument("--rollout-stop", required=True, type=int)
    build.add_argument("--target-future-observations-used", action="store_true")
    build.add_argument(
        "--known-future-controller-action-used",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    validate = commands.add_parser("validate", help="validate one published bundle")
    validate.add_argument("output_dir", type=Path)
    validate.add_argument("--verify-sources", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "profiles":
            result = describe_learned_twin_profiles()
        elif args.command == "build":
            result = materialize_learned_twin_backend(
                source_rollout_path=args.source_rollout,
                model_artifacts=_key_value(args.model_artifact, name="model artifact"),
                output_dir=args.output_dir,
                profile_id=args.profile,
                mode=args.mode,
                producer_repository=args.producer_repository,
                producer_revision=args.producer_revision,
                producer_source_artifacts=_key_value(
                    args.source_artifact, name="source artifact"
                ),
                case_id=args.case_id,
                target_object_id=args.target_object_id,
                training_object_ids=args.training_object_id,
                evidence_frame_range_half_open=(
                    args.evidence_start,
                    args.evidence_stop,
                ),
                rollout_frame_range_half_open=(
                    args.rollout_start,
                    args.rollout_stop,
                ),
                target_future_observations_used=(args.target_future_observations_used),
                known_future_controller_action_used=(
                    args.known_future_controller_action_used
                ),
            )
        elif args.command == "validate":
            result = validate_learned_twin_backend(
                args.output_dir,
                verify_sources=args.verify_sources,
            )
        else:  # pragma: no cover - argparse enforces the command set
            raise AssertionError(f"unhandled command: {args.command}")
    except ValueError as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
