#!/usr/bin/env python3
"""Seal the Deform360 v6.1 source-prefix candidate panel before suffix access."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from bayesian_phystwin._portable_contracts import load_strict_json_object
from bayesian_phystwin.deform360_fresh_object_session_candidate_runner_v6_1 import (
    publish_deform360_v61_candidate_panel,
    retain_deform360_v61_candidate_execution_failure,
    seal_deform360_v61_candidate_execution,
    validate_deform360_v61_candidate_execution_receipt,
    validate_deform360_v61_candidate_panel,
    validate_deform360_v61_candidate_panel_receipt,
    validate_deform360_v61_candidate_technical_failure_receipt,
)
from bayesian_phystwin.deform360_fresh_object_session_candidate_v6_1 import (
    load_deform360_v61_candidate_amendment,
)


def _validate_amendment(args: argparse.Namespace) -> int:
    value = load_deform360_v61_candidate_amendment(args.amendment)
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0


def _publish(args: argparse.Namespace) -> int:
    receipt = publish_deform360_v61_candidate_panel(
        candidate_amendment_path=args.amendment,
        execution_lock_path=args.execution_lock,
        source_plan_path=args.source_plan,
        upstream_prediction_batch_path=args.upstream_prediction_batch,
        upstream_prediction_receipt_path=args.upstream_prediction_receipt,
        upstream_execution_receipt_path=args.upstream_execution_receipt,
        upstream_source_seal_root=args.upstream_source_seal_root,
        upstream_prediction_root=args.upstream_prediction_root,
        input_root=args.input_root,
        output_root=args.output_root,
        candidate_revision=args.candidate_revision,
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


def _validate_receipt(args: argparse.Namespace) -> int:
    receipt = validate_deform360_v61_candidate_panel_receipt(
        load_strict_json_object(args.receipt, label="candidate panel receipt")
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


def _validate_panel(args: argparse.Namespace) -> int:
    receipt = validate_deform360_v61_candidate_panel(
        execution_lock_path=args.execution_lock,
        output_root=args.output_root,
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


def _seal_execution(args: argparse.Namespace) -> int:
    receipt = seal_deform360_v61_candidate_execution(
        candidate_amendment_path=args.amendment,
        execution_lock_path=args.execution_lock,
        upstream_source_plan_path=args.source_plan,
        upstream_prediction_batch_path=args.upstream_prediction_batch,
        upstream_prediction_receipt_path=args.upstream_prediction_receipt,
        upstream_execution_receipt_path=args.upstream_execution_receipt,
        candidate_output_root=args.output_root,
        candidate_revision=args.candidate_revision,
        runner_name=args.runner_name,
        workflow_run_id=args.workflow_run_id,
        workflow_run_attempt=args.workflow_run_attempt,
        output_path=args.output,
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


def _retain_failure(args: argparse.Namespace) -> int:
    receipt = retain_deform360_v61_candidate_execution_failure(
        candidate_revision=args.candidate_revision,
        runner_name=args.runner_name,
        workflow_run_id=args.workflow_run_id,
        workflow_run_attempt=args.workflow_run_attempt,
        terminal_stage=args.terminal_stage,
        exit_code=args.exit_code,
        artifact_root=args.artifact_root,
        output_path=args.output,
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


def _validate_execution(args: argparse.Namespace) -> int:
    value = load_strict_json_object(args.receipt, label="candidate execution receipt")
    if value.get("status") == "candidate-prefix-panel-sealed":
        receipt = validate_deform360_v61_candidate_execution_receipt(value)
    else:
        receipt = validate_deform360_v61_candidate_technical_failure_receipt(value)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    validate_amendment = commands.add_parser("validate-amendment")
    validate_amendment.add_argument("--amendment", type=Path, required=True)
    validate_amendment.set_defaults(run=_validate_amendment)

    publish = commands.add_parser("publish-panel")
    publish.add_argument("--amendment", type=Path, required=True)
    publish.add_argument("--execution-lock", type=Path, required=True)
    publish.add_argument("--source-plan", type=Path, required=True)
    publish.add_argument("--upstream-prediction-batch", type=Path, required=True)
    publish.add_argument("--upstream-prediction-receipt", type=Path, required=True)
    publish.add_argument("--upstream-execution-receipt", type=Path, required=True)
    publish.add_argument("--upstream-source-seal-root", type=Path, required=True)
    publish.add_argument("--upstream-prediction-root", type=Path, required=True)
    publish.add_argument("--input-root", type=Path, required=True)
    publish.add_argument("--output-root", type=Path, required=True)
    publish.add_argument("--candidate-revision", required=True)
    publish.set_defaults(run=_publish)

    validate_receipt = commands.add_parser("validate-receipt")
    validate_receipt.add_argument("--receipt", type=Path, required=True)
    validate_receipt.set_defaults(run=_validate_receipt)

    validate_panel = commands.add_parser("validate-panel")
    validate_panel.add_argument("--execution-lock", type=Path, required=True)
    validate_panel.add_argument("--output-root", type=Path, required=True)
    validate_panel.set_defaults(run=_validate_panel)

    seal = commands.add_parser("seal-execution")
    seal.add_argument("--amendment", type=Path, required=True)
    seal.add_argument("--execution-lock", type=Path, required=True)
    seal.add_argument("--source-plan", type=Path, required=True)
    seal.add_argument("--upstream-prediction-batch", type=Path, required=True)
    seal.add_argument("--upstream-prediction-receipt", type=Path, required=True)
    seal.add_argument("--upstream-execution-receipt", type=Path, required=True)
    seal.add_argument("--output-root", type=Path, required=True)
    seal.add_argument("--candidate-revision", required=True)
    seal.add_argument("--runner-name", required=True)
    seal.add_argument("--workflow-run-id", type=int, required=True)
    seal.add_argument("--workflow-run-attempt", type=int, required=True)
    seal.add_argument("--output", type=Path, required=True)
    seal.set_defaults(run=_seal_execution)

    retain = commands.add_parser("retain-failure")
    retain.add_argument("--candidate-revision", required=True)
    retain.add_argument("--runner-name", required=True)
    retain.add_argument("--workflow-run-id", type=int, required=True)
    retain.add_argument("--workflow-run-attempt", type=int, required=True)
    retain.add_argument("--terminal-stage", required=True)
    retain.add_argument("--exit-code", type=int, required=True)
    retain.add_argument("--artifact-root", type=Path, required=True)
    retain.add_argument("--output", type=Path, required=True)
    retain.set_defaults(run=_retain_failure)

    validate_execution = commands.add_parser("validate-execution")
    validate_execution.add_argument("--receipt", type=Path, required=True)
    validate_execution.set_defaults(run=_validate_execution)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    return int(args.run(args))


if __name__ == "__main__":
    raise SystemExit(main())
