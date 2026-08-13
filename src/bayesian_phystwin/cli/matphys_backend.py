"""Build and validate guarded MatPhys/Warp backend artifacts."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import cast

from bayesian_phystwin._portable_contracts import (
    load_strict_json_object,
    source_artifact_mapping,
    write_atomic_json,
)
from bayesian_phystwin.matphys_backend_v1 import (
    build_matphys_backend_gate,
    build_matphys_backend_proposal,
    materialize_matphys_backend,
    validate_matphys_backend_artifact,
)


def _source_artifacts(value: str) -> Mapping[str, str]:
    path = Path(value)
    try:
        raw = load_strict_json_object(path, label="source-artifact mapping")
        return cast(
            Mapping[str, str],
            source_artifact_mapping(raw, name="source_artifacts"),
        )
    except (OSError, ValueError) as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def _metrics(value: str) -> dict[str, float]:
    path = Path(value)
    try:
        raw = load_strict_json_object(path, label="metric record")
        if set(raw) != {"chamfer_distance_m", "track_error_m"}:
            raise ValueError("metric record fields changed")
        if any(
            isinstance(raw[name], bool)
            or not isinstance(raw[name], (int, float))
            for name in ("chamfer_distance_m", "track_error_m")
        ):
            raise ValueError("metric record values must be JSON numbers")
        return {
            "chamfer_distance_m": float(raw["chamfer_distance_m"]),
            "track_error_m": float(raw["track_error_m"]),
        }
    except (KeyError, OSError, TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    proposal = commands.add_parser(
        "proposal", help="write a future-blind MatPhys spring proposal manifest"
    )
    proposal.add_argument("output", type=Path)
    proposal.add_argument("--source-revision", required=True)
    proposal.add_argument("--simulator-revision", required=True)
    proposal.add_argument("--target-object-id", required=True)
    proposal.add_argument("--training-object-id", action="append", required=True)
    proposal.add_argument(
        "--target-evidence-end-frame-exclusive", type=int, required=True
    )
    proposal.add_argument("--proposal-strength", type=float, required=True)
    proposal.add_argument("--checkpoint", type=Path, required=True)
    proposal.add_argument("--spring-field", type=Path, required=True)
    proposal.add_argument(
        "--source-artifacts", type=_source_artifacts, required=True
    )

    gate = commands.add_parser(
        "gate", help="write a disjoint causal-prefix MatPhys selection gate"
    )
    gate.add_argument("output", type=Path)
    gate.add_argument("--proposal-id", required=True)
    gate.add_argument("--target-object-id", required=True)
    gate.add_argument("--case-id", required=True)
    gate.add_argument("--validation-start", type=int, required=True)
    gate.add_argument("--validation-stop", type=int, required=True)
    gate.add_argument("--future-frame-start", type=int, required=True)
    gate.add_argument("--incumbent-archive", type=Path, required=True)
    gate.add_argument("--candidate-archive", type=Path, required=True)
    gate.add_argument("--identity-replay-archive", type=Path, required=True)
    gate.add_argument("--incumbent-metrics", type=_metrics, required=True)
    gate.add_argument("--candidate-metrics", type=_metrics, required=True)
    gate.add_argument("--minimum-relative-improvement", type=float, required=True)
    gate.add_argument("--maximum-metric-regression", type=float, required=True)
    gate.add_argument("--maximum-identity-replay-rmse-m", type=float, required=True)
    gate.add_argument("--source-artifacts", type=_source_artifacts, required=True)

    materialize = commands.add_parser(
        "materialize", help="select and publish one guarded MatPhys backend"
    )
    materialize.add_argument("proposal_manifest", type=Path)
    materialize.add_argument("gate_manifest", type=Path)
    materialize.add_argument("incumbent_archive", type=Path)
    materialize.add_argument("candidate_archive", type=Path)
    materialize.add_argument("identity_replay_archive", type=Path)
    materialize.add_argument("output_dir", type=Path)

    validate = commands.add_parser(
        "validate", help="validate a published MatPhys backend bundle"
    )
    validate.add_argument("output_dir", type=Path)
    return parser


def _proposal(args: argparse.Namespace) -> dict[str, object]:
    proposal = build_matphys_backend_proposal(
        source_revision=args.source_revision,
        simulator_revision=args.simulator_revision,
        target_object_id=args.target_object_id,
        training_object_ids=tuple(sorted(args.training_object_id)),
        target_evidence_end_frame_exclusive=(
            args.target_evidence_end_frame_exclusive
        ),
        proposal_strength=args.proposal_strength,
        checkpoint_path=args.checkpoint,
        spring_field_path=args.spring_field,
        source_artifacts=args.source_artifacts,
    )
    write_atomic_json(proposal, args.output, overwrite=False)
    return proposal


def _gate(args: argparse.Namespace) -> dict[str, object]:
    gate = build_matphys_backend_gate(
        proposal_id=args.proposal_id,
        target_object_id=args.target_object_id,
        case_id=args.case_id,
        validation_frame_range_half_open=(
            args.validation_start,
            args.validation_stop,
        ),
        future_frame_start=args.future_frame_start,
        incumbent_archive_path=args.incumbent_archive,
        candidate_archive_path=args.candidate_archive,
        identity_replay_archive_path=args.identity_replay_archive,
        incumbent_metrics=args.incumbent_metrics,
        candidate_metrics=args.candidate_metrics,
        minimum_relative_improvement=args.minimum_relative_improvement,
        maximum_metric_regression=args.maximum_metric_regression,
        maximum_identity_replay_rmse_m=args.maximum_identity_replay_rmse_m,
        source_artifacts=args.source_artifacts,
    )
    write_atomic_json(gate, args.output, overwrite=False)
    return gate


def _materialize(args: argparse.Namespace) -> dict[str, object]:
    return materialize_matphys_backend(
        proposal_manifest_path=args.proposal_manifest,
        gate_manifest_path=args.gate_manifest,
        incumbent_archive_path=args.incumbent_archive,
        candidate_archive_path=args.candidate_archive,
        identity_replay_archive_path=args.identity_replay_archive,
        output_dir=args.output_dir,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "proposal":
        result = _proposal(args)
    elif args.command == "gate":
        result = _gate(args)
    elif args.command == "materialize":
        result = _materialize(args)
    elif args.command == "validate":
        result = validate_matphys_backend_artifact(args.output_dir)
    else:  # pragma: no cover - argparse enforces the command set
        raise AssertionError(f"unhandled command: {args.command}")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
