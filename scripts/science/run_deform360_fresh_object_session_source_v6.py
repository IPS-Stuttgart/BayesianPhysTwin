#!/usr/bin/env python3
"""Seal, assemble, and evaluate Deform360 v6 source-only evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from bayesian_phystwin._portable_contracts import load_strict_json_object
from bayesian_phystwin.deform360_fresh_object_session_source_v6 import (
    assemble_deform360_v6_source_evidence,
    build_deform360_v6_source_prediction_batch,
    evaluate_deform360_v6_source_gate,
    load_deform360_fresh_object_session_v6_covariance_amendment,
    load_deform360_fresh_object_session_v6_policy,
    load_deform360_v6_source_selection,
    publish_deform360_v6_source_evidence,
    publish_deform360_v6_source_prediction_batch,
    publish_deform360_v6_source_result,
    validate_deform360_v6_source_prediction_batch,
)


def _load_many(paths: list[Path], *, label: str) -> list[dict[str, Any]]:
    return [
        dict(load_strict_json_object(path, label=f"{label} {index}"))
        for index, path in enumerate(paths)
    ]


def _locks(args: argparse.Namespace):
    policy = load_deform360_fresh_object_session_v6_policy(args.policy)
    amendment = load_deform360_fresh_object_session_v6_covariance_amendment(
        args.amendment,
        policy,
    )
    selection, _ = load_deform360_v6_source_selection(args.selection, policy)
    return policy, amendment, selection


def _seal_batch(args: argparse.Namespace) -> int:
    policy, amendment, selection = _locks(args)
    seals = _load_many(args.prediction_seal, label="prediction seal")
    batch = build_deform360_v6_source_prediction_batch(
        seals,
        policy=policy,
        amendment=amendment,
        selection=selection,
    )
    publish_deform360_v6_source_prediction_batch(batch, args.output)
    print(json.dumps(batch, indent=2, sort_keys=True))
    return 0


def _assemble(args: argparse.Namespace) -> int:
    policy, amendment, selection = _locks(args)
    batch = dict(
        load_strict_json_object(args.prediction_batch, label="prediction batch")
    )
    batch = validate_deform360_v6_source_prediction_batch(
        batch,
        policy=policy,
        amendment=amendment,
        selection=selection,
    )
    outcomes = _load_many(args.outcome, label="source outcome")
    evidence = assemble_deform360_v6_source_evidence(
        prediction_batch=batch,
        outcomes=outcomes,
    )
    publish_deform360_v6_source_evidence(evidence, args.output)
    print(json.dumps(evidence, indent=2, sort_keys=True))
    return 0


def _evaluate(args: argparse.Namespace) -> int:
    policy = load_deform360_fresh_object_session_v6_policy(args.policy)
    evidence = dict(load_strict_json_object(args.evidence, label="source evidence"))
    result = evaluate_deform360_v6_source_gate(evidence, policy)
    publish_deform360_v6_source_result(result, args.output)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["source_gate_passed"] else 3


def _common_locks(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--amendment", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    seal = subparsers.add_parser(
        "seal-batch",
        help="seal the complete outcome-free ten-unit prediction batch",
    )
    _common_locks(seal)
    seal.add_argument(
        "--prediction-seal",
        type=Path,
        action="append",
        required=True,
    )
    seal.add_argument("--output", type=Path, required=True)
    seal.set_defaults(run=_seal_batch)

    assemble = subparsers.add_parser(
        "assemble",
        help="attach source suffix outcomes after the prediction barrier",
    )
    _common_locks(assemble)
    assemble.add_argument("--prediction-batch", type=Path, required=True)
    assemble.add_argument("--outcome", type=Path, action="append", required=True)
    assemble.add_argument("--output", type=Path, required=True)
    assemble.set_defaults(run=_assemble)

    evaluate = subparsers.add_parser(
        "evaluate",
        help="run the source-only challenger/covariance/guard gate",
    )
    evaluate.add_argument("--policy", type=Path, required=True)
    evaluate.add_argument("--evidence", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.set_defaults(run=_evaluate)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    return int(args.run(args))


if __name__ == "__main__":
    raise SystemExit(main())
