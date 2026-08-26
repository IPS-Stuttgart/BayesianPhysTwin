#!/usr/bin/env python3
"""Operate the frozen target-closed Deform360 covariance source barrier."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from bayesian_phystwin.deform360_covariance_source_inventory_v1 import (
    build_covariance_source_inventory_v1,
    publish_covariance_source_inventory_v1,
    validate_covariance_source_inventory_v1,
)
from bayesian_phystwin.deform360_covariance_source_producer_v1 import (
    build_covariance_source_technical_receipt_v1,
    publish_covariance_source_panel_v1,
    validate_covariance_source_panel_v1,
)


def _write_json_once(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _common_contract_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--crossrepo-binding", type=Path, required=True)
    parser.add_argument("--implementation-revision", required=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    inventory = subparsers.add_parser("inventory", help="build a header-only inventory")
    _common_contract_arguments(inventory)
    inventory.add_argument("--source-root", type=Path, required=True)
    inventory.add_argument("--processed-root", type=Path, required=True)
    inventory.add_argument("--forbidden-confirmation-root", type=Path, required=True)
    inventory.add_argument("--output", type=Path, required=True)

    produce = subparsers.add_parser(
        "produce", help="publish the exact 100-record panel"
    )
    _common_contract_arguments(produce)
    produce.add_argument("--source-execution-lock", type=Path, required=True)
    produce.add_argument("--source-inventory", type=Path, required=True)
    produce.add_argument("--upstream-source-plan", type=Path, required=True)
    produce.add_argument("--upstream-prediction-batch", type=Path, required=True)
    produce.add_argument("--upstream-prediction-receipt", type=Path, required=True)
    produce.add_argument("--upstream-execution-receipt", type=Path, required=True)
    produce.add_argument("--upstream-run-root", type=Path, required=True)
    produce.add_argument("--input-root", type=Path, required=True)
    produce.add_argument("--forbidden-confirmation-root", type=Path, required=True)
    produce.add_argument("--repository-root", type=Path, required=True)
    produce.add_argument("--output-root", type=Path, required=True)
    produce.add_argument("--technical-receipt", type=Path, required=True)

    execute = subparsers.add_parser(
        "execute",
        help="inventory and publish one fail-closed source panel attempt",
    )
    _common_contract_arguments(execute)
    execute.add_argument("--source-root", type=Path, required=True)
    execute.add_argument("--processed-root", type=Path, required=True)
    execute.add_argument("--forbidden-confirmation-root", type=Path, required=True)
    execute.add_argument("--inventory-output", type=Path, required=True)
    execute.add_argument("--source-execution-lock", type=Path, required=True)
    execute.add_argument("--upstream-source-plan", type=Path, required=True)
    execute.add_argument("--upstream-prediction-batch", type=Path, required=True)
    execute.add_argument("--upstream-prediction-receipt", type=Path, required=True)
    execute.add_argument("--upstream-execution-receipt", type=Path, required=True)
    execute.add_argument("--upstream-run-root", type=Path, required=True)
    execute.add_argument("--input-root", type=Path, required=True)
    execute.add_argument("--repository-root", type=Path, required=True)
    execute.add_argument("--output-root", type=Path, required=True)
    execute.add_argument("--technical-receipt", type=Path, required=True)

    validate_inventory = subparsers.add_parser(
        "validate-inventory",
        help="validate an inventory without traversing data",
    )
    validate_inventory.add_argument("path", type=Path)
    validate_panel = subparsers.add_parser(
        "validate-panel",
        help="rehash a complete target-closed source panel",
    )
    validate_panel.add_argument("path", type=Path)
    return parser


def _inventory(args: argparse.Namespace) -> int:
    value = build_covariance_source_inventory_v1(
        protocol_path=args.protocol,
        selection_path=args.selection,
        crossrepo_binding_path=args.crossrepo_binding,
        calibration_source_root=args.source_root,
        calibration_processed_root=args.processed_root,
        forbidden_confirmation_root=args.forbidden_confirmation_root,
        implementation_revision=args.implementation_revision,
    )
    publish_covariance_source_inventory_v1(value, args.output)
    print(json.dumps({"inventory_id": value["inventory_id"]}, sort_keys=True))
    return 0


def _produce(args: argparse.Namespace) -> int:
    try:
        receipt = publish_covariance_source_panel_v1(
            protocol_path=args.protocol,
            selection_path=args.selection,
            crossrepo_binding_path=args.crossrepo_binding,
            source_execution_lock_path=args.source_execution_lock,
            source_inventory_path=args.source_inventory,
            upstream_source_plan_path=args.upstream_source_plan,
            upstream_prediction_batch_path=args.upstream_prediction_batch,
            upstream_prediction_receipt_path=args.upstream_prediction_receipt,
            upstream_execution_receipt_path=args.upstream_execution_receipt,
            upstream_run_root=args.upstream_run_root,
            input_root=args.input_root,
            forbidden_confirmation_root=args.forbidden_confirmation_root,
            repository_root=args.repository_root,
            output_root=args.output_root,
            implementation_revision=args.implementation_revision,
        )
    except (OSError, TypeError, ValueError):
        technical = build_covariance_source_technical_receipt_v1(
            implementation_revision=args.implementation_revision,
            terminal_stage="source-panel-production",
            diagnostic_code="unexpected-runtime-failure",
        )
        _write_json_once(args.technical_receipt, technical)
        raise
    validate_covariance_source_panel_v1(args.output_root)
    print(json.dumps(receipt, sort_keys=True))
    return 0


def _execute(args: argparse.Namespace) -> int:
    retained: dict[str, str] = {}
    try:
        inventory = build_covariance_source_inventory_v1(
            protocol_path=args.protocol,
            selection_path=args.selection,
            crossrepo_binding_path=args.crossrepo_binding,
            calibration_source_root=args.source_root,
            calibration_processed_root=args.processed_root,
            forbidden_confirmation_root=args.forbidden_confirmation_root,
            implementation_revision=args.implementation_revision,
        )
        publish_covariance_source_inventory_v1(inventory, args.inventory_output)
        retained["source-input-inventory.json"] = _sha256_file(args.inventory_output)
    except Exception:
        technical = build_covariance_source_technical_receipt_v1(
            implementation_revision=args.implementation_revision,
            terminal_stage="source-input-inventory",
            diagnostic_code="inventory-contract-failure",
            retained_artifacts=retained,
        )
        _write_json_once(args.technical_receipt, technical)
        raise

    try:
        receipt = publish_covariance_source_panel_v1(
            protocol_path=args.protocol,
            selection_path=args.selection,
            crossrepo_binding_path=args.crossrepo_binding,
            source_execution_lock_path=args.source_execution_lock,
            source_inventory_path=args.inventory_output,
            upstream_source_plan_path=args.upstream_source_plan,
            upstream_prediction_batch_path=args.upstream_prediction_batch,
            upstream_prediction_receipt_path=args.upstream_prediction_receipt,
            upstream_execution_receipt_path=args.upstream_execution_receipt,
            upstream_run_root=args.upstream_run_root,
            input_root=args.input_root,
            forbidden_confirmation_root=args.forbidden_confirmation_root,
            repository_root=args.repository_root,
            output_root=args.output_root,
            implementation_revision=args.implementation_revision,
        )
    except Exception:
        technical = build_covariance_source_technical_receipt_v1(
            implementation_revision=args.implementation_revision,
            terminal_stage="source-panel-production",
            diagnostic_code="provider-materialization-failure",
            retained_artifacts=retained,
        )
        _write_json_once(args.technical_receipt, technical)
        raise
    validate_covariance_source_panel_v1(args.output_root)
    print(json.dumps(receipt, sort_keys=True))
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "inventory":
        return _inventory(args)
    if args.command == "produce":
        return _produce(args)
    if args.command == "execute":
        return _execute(args)
    if args.command == "validate-inventory":
        value = validate_covariance_source_inventory_v1(
            json.loads(args.path.read_text(encoding="utf-8"))
        )
        print(json.dumps({"inventory_id": value["inventory_id"]}, sort_keys=True))
        return 0
    if args.command == "validate-panel":
        receipt = validate_covariance_source_panel_v1(args.path)
        print(json.dumps(receipt, sort_keys=True))
        return 0
    raise AssertionError("unreachable command")


if __name__ == "__main__":
    raise SystemExit(main())
