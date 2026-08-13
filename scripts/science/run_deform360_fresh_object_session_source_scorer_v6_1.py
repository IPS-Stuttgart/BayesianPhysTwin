#!/usr/bin/env python3
"""Authorize and score the frozen Deform360 v6.1 public-source suffix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from bayesian_phystwin._portable_contracts import (
    load_strict_json_object,
    write_atomic_json,
)
from bayesian_phystwin.deform360_fresh_object_session_source_scorer_v6_1 import (
    build_deform360_v61_source_endpoint_manifest,
    build_deform360_v61_source_suffix_authorization,
    load_deform360_v61_source_scoring_amendment,
    publish_deform360_v61_source_scores,
    publish_deform360_v61_source_suffix_authorization,
    retain_deform360_v61_source_scoring_failure,
    validate_deform360_v61_source_plan,
    validate_deform360_v61_source_scoring_receipt,
    validate_deform360_v61_source_scoring_technical_failure_receipt,
    validate_deform360_v61_source_suffix_authorization,
)


def _load_many(paths: list[Path], *, label: str) -> list[dict[str, Any]]:
    return [
        dict(load_strict_json_object(path, label=f"{label} {index}"))
        for index, path in enumerate(paths)
    ]


def _validate_amendment(args: argparse.Namespace) -> int:
    amendment = load_deform360_v61_source_scoring_amendment(args.scoring_amendment)
    print(json.dumps(amendment, indent=2, sort_keys=True, allow_nan=False))
    return 0


def _authorize(args: argparse.Namespace) -> int:
    authorization = build_deform360_v61_source_suffix_authorization(
        source_scoring_amendment_path=args.scoring_amendment,
        execution_lock_path=args.execution_lock,
        candidate_root=args.candidate_root,
        candidate_execution_receipt_path=args.candidate_execution_receipt,
        upstream_source_plan_path=args.source_plan,
        scorer_revision=args.scorer_revision,
        runner_name=args.runner_name,
        workflow_run_id=args.workflow_run_id,
        workflow_run_attempt=args.workflow_run_attempt,
    )
    publish_deform360_v61_source_suffix_authorization(
        authorization,
        args.output,
    )
    print(json.dumps(authorization, indent=2, sort_keys=True, allow_nan=False))
    return 0


def _endpoint_manifest(args: argparse.Namespace) -> int:
    authorization = validate_deform360_v61_source_suffix_authorization(
        load_strict_json_object(args.authorization, label="source authorization")
    )
    source_plan = validate_deform360_v61_source_plan(
        load_strict_json_object(args.source_plan, label="source plan")
    )
    manifest = build_deform360_v61_source_endpoint_manifest(
        authorization=authorization,
        source_plan=source_plan,
        processor_revision=args.processor_revision,
        objects=_load_many(args.object, label="endpoint object"),
    )
    write_atomic_json(manifest, args.output, overwrite=False)
    print(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False))
    return 0


def _score(args: argparse.Namespace) -> int:
    receipt = publish_deform360_v61_source_scores(
        source_scoring_amendment_path=args.scoring_amendment,
        execution_lock_path=args.execution_lock,
        candidate_root=args.candidate_root,
        candidate_execution_receipt_path=args.candidate_execution_receipt,
        upstream_source_plan_path=args.source_plan,
        authorization_path=args.authorization,
        endpoint_manifest_path=args.endpoint_manifest,
        endpoint_root=args.endpoint_root,
        output_root=args.output_root,
        scorer_revision=args.scorer_revision,
        runner_name=args.runner_name,
        workflow_run_id=args.workflow_run_id,
        workflow_run_attempt=args.workflow_run_attempt,
    )
    print(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False))
    return 0 if receipt["source_gate_passed"] else 3


def _retain_failure(args: argparse.Namespace) -> int:
    receipt = retain_deform360_v61_source_scoring_failure(
        scorer_revision=args.scorer_revision,
        runner_name=args.runner_name,
        workflow_run_id=args.workflow_run_id,
        workflow_run_attempt=args.workflow_run_attempt,
        authorization_path=args.authorization,
        terminal_stage=args.terminal_stage,
        exit_code=args.exit_code,
        source_suffix_opened=args.source_suffix_opened,
        artifact_root=args.artifact_root,
        output_path=args.output,
    )
    print(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False))
    return 0


def _validate_receipt(args: argparse.Namespace) -> int:
    value = load_strict_json_object(args.receipt, label="source-scoring receipt")
    if value.get("schema", "").endswith("technical-failure-receipt"):
        receipt = validate_deform360_v61_source_scoring_technical_failure_receipt(value)
    else:
        receipt = validate_deform360_v61_source_scoring_receipt(value)
    print(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False))
    return 0


def _common_barrier(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--scoring-amendment", type=Path, required=True)
    parser.add_argument("--execution-lock", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--candidate-execution-receipt", type=Path, required=True)
    parser.add_argument("--source-plan", type=Path, required=True)
    parser.add_argument("--scorer-revision", required=True)
    parser.add_argument("--runner-name", required=True)
    parser.add_argument("--workflow-run-id", type=int, required=True)
    parser.add_argument("--workflow-run-attempt", type=int, required=True)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    amendment = commands.add_parser(
        "validate-amendment",
        help="validate the frozen target-closed source-scoring amendment",
    )
    amendment.add_argument("--scoring-amendment", type=Path, required=True)
    amendment.set_defaults(run=_validate_amendment)

    authorize = commands.add_parser(
        "authorize",
        help="rehash the exact 100-record barrier before any suffix file is named",
    )
    _common_barrier(authorize)
    authorize.add_argument("--output", type=Path, required=True)
    authorize.set_defaults(run=_authorize)

    manifest = commands.add_parser(
        "build-endpoint-manifest",
        help="bind ten post-authorization public endpoint object records",
    )
    manifest.add_argument("--authorization", type=Path, required=True)
    manifest.add_argument("--source-plan", type=Path, required=True)
    manifest.add_argument("--processor-revision", required=True)
    manifest.add_argument("--object", type=Path, action="append", required=True)
    manifest.add_argument("--output", type=Path, required=True)
    manifest.set_defaults(run=_endpoint_manifest)

    score = commands.add_parser(
        "score",
        help="score 100 candidates and run the unchanged nested source gate once",
    )
    _common_barrier(score)
    score.add_argument("--authorization", type=Path, required=True)
    score.add_argument("--endpoint-manifest", type=Path, required=True)
    score.add_argument("--endpoint-root", type=Path, required=True)
    score.add_argument("--output-root", type=Path, required=True)
    score.set_defaults(run=_score)

    retain = commands.add_parser(
        "retain-failure",
        help="retain one post-authorization technical failure without a source gate",
    )
    retain.add_argument("--scorer-revision", required=True)
    retain.add_argument("--runner-name", required=True)
    retain.add_argument("--workflow-run-id", type=int, required=True)
    retain.add_argument("--workflow-run-attempt", type=int, required=True)
    retain.add_argument("--authorization", type=Path, required=True)
    retain.add_argument("--terminal-stage", required=True)
    retain.add_argument("--exit-code", type=int, required=True)
    retain.add_argument("--source-suffix-opened", action="store_true")
    retain.add_argument("--artifact-root", type=Path, required=True)
    retain.add_argument("--output", type=Path, required=True)
    retain.set_defaults(run=_retain_failure)

    validate = commands.add_parser("validate-receipt")
    validate.add_argument("--receipt", type=Path, required=True)
    validate.set_defaults(run=_validate_receipt)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    return int(args.run(args))


if __name__ == "__main__":
    raise SystemExit(main())
