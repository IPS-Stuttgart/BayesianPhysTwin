"""Seal, materialize, or validate a causal DeformMaster backend artifact."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from bayesian_phystwin.deformmaster_backend_v1 import (
    materialize_deformmaster_backend,
    seal_deformmaster_runtime_manifest,
    validate_deformmaster_backend,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    seal = commands.add_parser(
        "seal-runtime",
        help="seal a producer-attested causal runtime before future scoring",
    )
    seal.add_argument("raw_rollout", type=Path)
    seal.add_argument("checkpoint", type=Path)
    seal.add_argument("configuration", type=Path)
    seal.add_argument("training_manifest", type=Path)
    seal.add_argument("output", type=Path)
    seal.add_argument("--source-revision", required=True)
    seal.add_argument("--producer-repository", required=True)
    seal.add_argument("--producer-revision", required=True)
    seal.add_argument("--case-id", required=True)
    seal.add_argument("--target-object-id", required=True)
    seal.add_argument("--prefix-end-frame-exclusive", required=True, type=int)
    seal.add_argument("--time-step-s", required=True, type=float)

    materialize = commands.add_parser(
        "materialize",
        help="adapt a sealed causal surface rollout to physical_rollout_v1",
    )
    materialize.add_argument("raw_rollout", type=Path)
    materialize.add_argument("runtime_manifest", type=Path)
    materialize.add_argument("checkpoint", type=Path)
    materialize.add_argument("configuration", type=Path)
    materialize.add_argument("training_manifest", type=Path)
    materialize.add_argument("output_dir", type=Path)

    validate = commands.add_parser("validate", help="validate a published bundle")
    validate.add_argument("output_dir", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "seal-runtime":
        result = seal_deformmaster_runtime_manifest(
            raw_rollout_path=args.raw_rollout,
            checkpoint_path=args.checkpoint,
            configuration_path=args.configuration,
            training_manifest_path=args.training_manifest,
            output_path=args.output,
            source_revision=args.source_revision,
            producer_repository=args.producer_repository,
            producer_revision=args.producer_revision,
            case_id=args.case_id,
            target_object_id=args.target_object_id,
            prefix_end_frame_exclusive=args.prefix_end_frame_exclusive,
            time_step_s=args.time_step_s,
        )
    elif args.command == "materialize":
        result = materialize_deformmaster_backend(
            raw_rollout_path=args.raw_rollout,
            runtime_manifest_path=args.runtime_manifest,
            checkpoint_path=args.checkpoint,
            configuration_path=args.configuration,
            training_manifest_path=args.training_manifest,
            output_dir=args.output_dir,
        )
    elif args.command == "validate":
        result = validate_deformmaster_backend(args.output_dir)
    else:  # pragma: no cover - argparse enforces the command set
        raise AssertionError(f"unhandled command: {args.command}")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
