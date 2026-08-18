"""Build and validate first-class official MatPhys producer bundles."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import cast

from bayesian_phystwin._portable_contracts import (
    load_strict_json_object,
    sha256_digest,
    source_artifact_mapping,
)
from bayesian_phystwin.matphys_official_producer_v1 import (
    MATPHYS_CAUSAL_PREFIX_MODE,
    MATPHYS_OFFICIAL_PIPELINE_COMPONENTS,
    MATPHYS_PUBLISHED_PARITY_MODE,
    MatPhysOfficialMode,
    materialize_matphys_official_producer,
    validate_matphys_official_producer_artifact,
)


def _source_artifacts(value: str) -> Mapping[str, str]:
    try:
        raw = load_strict_json_object(Path(value), label="source-artifact mapping")
        return cast(
            Mapping[str, str],
            source_artifact_mapping(raw, name="source_artifacts"),
        )
    except (OSError, ValueError) as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def _component_artifacts(value: str) -> Mapping[str, str]:
    try:
        raw = load_strict_json_object(
            Path(value), label="MatPhys pipeline-component mapping"
        )
        if set(raw) != set(MATPHYS_OFFICIAL_PIPELINE_COMPONENTS):
            raise ValueError("official MatPhys pipeline component roster changed")
        return {
            name: sha256_digest(raw.get(name), name=f"component {name}")
            for name in MATPHYS_OFFICIAL_PIPELINE_COMPONENTS
        }
    except (OSError, ValueError) as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser(
        "build", help="convert a sealed official MatPhys replay export"
    )
    build.add_argument("replay_input", type=Path)
    build.add_argument("checkpoint", type=Path)
    build.add_argument("spring_field", type=Path)
    build.add_argument("candidate_parameters", type=Path)
    build.add_argument("identity_parameters", type=Path)
    build.add_argument("output_dir", type=Path)
    build.add_argument(
        "--mode",
        choices=(MATPHYS_PUBLISHED_PARITY_MODE, MATPHYS_CAUSAL_PREFIX_MODE),
        required=True,
    )
    build.add_argument("--source-revision", required=True)
    build.add_argument("--simulator-revision", required=True)
    build.add_argument("--case-id", required=True)
    build.add_argument("--target-object-id", required=True)
    build.add_argument(
        "--checkpoint-training-object-id", action="append", required=True
    )
    build.add_argument("--target-fit-start", type=int, required=True)
    build.add_argument("--target-fit-stop", type=int, required=True)
    build.add_argument("--future-frame-start", type=int, required=True)
    build.add_argument("--proposal-strength", type=float, required=True)
    build.add_argument(
        "--pipeline-component-artifacts",
        type=_component_artifacts,
        required=True,
    )
    build.add_argument("--source-artifacts", type=_source_artifacts, required=True)

    validate = commands.add_parser("validate", help="validate a producer bundle")
    validate.add_argument("output_dir", type=Path)
    validate.add_argument("--verify-sources", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "build":
        result = materialize_matphys_official_producer(
            replay_input_path=args.replay_input,
            checkpoint_path=args.checkpoint,
            spring_field_path=args.spring_field,
            candidate_parameter_path=args.candidate_parameters,
            identity_parameter_path=args.identity_parameters,
            output_dir=args.output_dir,
            mode=cast(MatPhysOfficialMode, args.mode),
            source_revision=args.source_revision,
            simulator_revision=args.simulator_revision,
            case_id=args.case_id,
            target_object_id=args.target_object_id,
            checkpoint_training_object_ids=tuple(
                sorted(args.checkpoint_training_object_id)
            ),
            target_fit_frame_range_half_open=(
                args.target_fit_start,
                args.target_fit_stop,
            ),
            future_frame_start=args.future_frame_start,
            proposal_strength=args.proposal_strength,
            pipeline_component_artifacts=args.pipeline_component_artifacts,
            source_artifacts=args.source_artifacts,
        )
    elif args.command == "validate":
        result = validate_matphys_official_producer_artifact(
            args.output_dir, verify_sources=args.verify_sources
        )
    else:  # pragma: no cover - argparse enforces the command set
        raise AssertionError(f"unhandled command: {args.command}")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
