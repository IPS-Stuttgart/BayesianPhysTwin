"""Freeze and adjudicate the prospective Prob4D-to-BayesianPhysTwin protocol."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from bayesian_phystwin.prob4d_prospective_protocol import (
    build_prob4d_prospective_protocol,
    check_prob4d_prospective_readiness,
    decide_prob4d_prospective_gates,
    load_json_mapping,
    load_prob4d_prospective_protocol,
    save_prob4d_prospective_protocol,
    write_json_mapping,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    freeze = subparsers.add_parser(
        "freeze",
        help="validate an unhashed configuration and publish its content-addressed freeze",
    )
    freeze.add_argument("configuration", type=Path)
    freeze.add_argument("output", type=Path)

    validate = subparsers.add_parser(
        "validate",
        help="validate a frozen protocol and optionally verify all source-side artifacts",
    )
    validate.add_argument("protocol", type=Path)
    validate.add_argument("--artifact-root", type=Path)
    validate.add_argument("--output", type=Path)
    validate.add_argument("--require-ready", action="store_true")

    decide = subparsers.add_parser(
        "decide",
        help="apply the frozen provider gate before the physical-prediction gate",
    )
    decide.add_argument("protocol", type=Path)
    decide.add_argument("result", type=Path)
    decide.add_argument("output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)
    try:
        if arguments.command == "freeze":
            configuration = load_json_mapping(
                arguments.configuration,
                name="prospective protocol configuration",
            )
            protocol = build_prob4d_prospective_protocol(configuration)
            save_prob4d_prospective_protocol(arguments.output, protocol)
            result: dict[str, object] = {
                "status": "frozen",
                "protocol_id": protocol.protocol_id,
                "protocol_sha256": protocol.protocol_sha256,
                "output": str(arguments.output),
                "target_unit_count": len(protocol.split["target"]),
                "primary_candidate_method_ids": list(
                    protocol.primary_candidate_method_ids
                ),
            }
        elif arguments.command == "validate":
            protocol = load_prob4d_prospective_protocol(arguments.protocol)
            if arguments.artifact_root is None:
                if arguments.require_ready:
                    parser.error("--require-ready requires --artifact-root")
                result = {
                    "status": "valid",
                    "protocol_id": protocol.protocol_id,
                    "protocol_sha256": protocol.protocol_sha256,
                    "ready_for_target_opening": None,
                }
            else:
                result = check_prob4d_prospective_readiness(
                    protocol,
                    arguments.artifact_root,
                )
                if arguments.require_ready and not result["ready_for_target_opening"]:
                    if arguments.output is not None:
                        write_json_mapping(arguments.output, result)
                    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
                    return 3
            if arguments.output is not None:
                write_json_mapping(arguments.output, result)
        else:
            protocol = load_prob4d_prospective_protocol(arguments.protocol)
            observed = load_json_mapping(
                arguments.result,
                name="prospective target result",
            )
            result = decide_prob4d_prospective_gates(protocol, observed)
            write_json_mapping(arguments.output, result)
    except ValueError as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
