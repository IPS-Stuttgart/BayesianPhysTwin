#!/usr/bin/env python3
"""Validate candidate panels and seal the target-blind Deform360 v6 source bridge."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from bayesian_phystwin._portable_contracts import load_strict_json_object
from bayesian_phystwin.deform360_fresh_object_session_source_v6 import (
    load_deform360_fresh_object_session_v6_covariance_amendment,
    load_deform360_fresh_object_session_v6_policy,
    load_deform360_v6_source_selection,
)
from bayesian_phystwin.deform360_fresh_object_session_v6_prediction_bridge import (
    bridge_deform360_v6_source_prediction_batch,
    load_deform360_v6_source_execution_amendment,
    publish_deform360_v6_prediction_bridge,
    validate_deform360_v6_source_candidate_panel,
)
from bayesian_phystwin.deform360_joint_sparse_source_gate_v5 import (
    load_deform360_joint_sparse_source_execution_lock_v5,
)


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--covariance-amendment", type=Path, required=True)
    parser.add_argument("--source-execution-amendment", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--v5-execution-lock", type=Path, required=True)
    parser.add_argument("--v5-prediction-batch", type=Path, required=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser(
        "validate-panel",
        help="validate one target-blind candidate panel",
    )
    _common(validate)
    validate.add_argument("--candidate-panel", type=Path, required=True)

    bridge = subparsers.add_parser(
        "bridge",
        help="publish exactly ten v6 source prediction seals and their batch",
    )
    _common(bridge)
    bridge.add_argument(
        "--candidate-panel",
        type=Path,
        action="append",
        required=True,
        help="one candidate panel; supply exactly ten",
    )
    bridge.add_argument("--bridge-revision", required=True)
    bridge.add_argument("--output-directory", type=Path, required=True)
    return parser


def _load_common(arguments: argparse.Namespace):
    policy = load_deform360_fresh_object_session_v6_policy(arguments.policy)
    covariance = load_deform360_fresh_object_session_v6_covariance_amendment(
        arguments.covariance_amendment,
        policy,
    )
    selection, _ = load_deform360_v6_source_selection(arguments.selection, policy)
    v5_lock = load_deform360_joint_sparse_source_execution_lock_v5(
        arguments.v5_execution_lock
    )
    execution = load_deform360_v6_source_execution_amendment(
        arguments.source_execution_amendment,
        v5_execution_lock_id=v5_lock["execution_lock_id"],
    )
    v5_batch = load_strict_json_object(
        arguments.v5_prediction_batch,
        label="v5 source prediction batch",
    )
    return policy, covariance, execution, selection, v5_lock, v5_batch


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    policy, covariance, execution, selection, v5_lock, v5_batch = _load_common(
        arguments
    )
    if arguments.command == "validate-panel":
        panel = load_strict_json_object(
            arguments.candidate_panel,
            label="v6 source candidate panel",
        )
        validated = validate_deform360_v6_source_candidate_panel(
            panel,
            policy=policy,
            covariance_amendment=covariance,
            source_execution_amendment=execution,
            selection=selection,
            v5_execution_lock=v5_lock,
            v5_prediction_batch=v5_batch,
        )
        print(json.dumps(validated, indent=2, sort_keys=True, allow_nan=False))
        return 0

    panel_paths = list(arguments.candidate_panel)
    if len(panel_paths) != 10:
        raise ValueError("bridge requires exactly ten --candidate-panel arguments")
    panels = [
        load_strict_json_object(path, label=f"v6 candidate panel {index}")
        for index, path in enumerate(panel_paths)
    ]
    seals, batch, receipt = bridge_deform360_v6_source_prediction_batch(
        policy=policy,
        covariance_amendment=covariance,
        source_execution_amendment=execution,
        selection=selection,
        v5_execution_lock=v5_lock,
        v5_prediction_batch=v5_batch,
        candidate_panels=panels,
        bridge_revision=arguments.bridge_revision,
    )
    output = publish_deform360_v6_prediction_bridge(
        seals=seals,
        batch=batch,
        receipt=receipt,
        output_directory=arguments.output_directory,
    )
    print(
        json.dumps(
            {
                "output_directory": str(output),
                "bridge_receipt_id": receipt["bridge_receipt_id"],
                "v6_prediction_batch_id": batch["prediction_batch_id"],
                "record_count": batch["record_count"],
                "information_boundary": receipt["information_boundary"],
            },
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
