#!/usr/bin/env python3
"""Seal, assemble, and evaluate the corrected nested Deform360 v6 source gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from bayesian_phystwin._portable_contracts import load_strict_json_object
from bayesian_phystwin.deform360_fresh_object_session_source_v6 import (
    load_deform360_fresh_object_session_v6_covariance_amendment,
    load_deform360_fresh_object_session_v6_policy,
    load_deform360_v6_source_selection,
)
from bayesian_phystwin.deform360_fresh_object_session_source_v6_1 import (
    assemble_deform360_v6_nested_evidence,
    build_deform360_v6_raw_nested_batch,
    evaluate_deform360_v6_nested_source_gate,
    load_deform360_v6_nested_source_repair,
    publish_deform360_v6_nested_evidence,
    publish_deform360_v6_nested_result,
    publish_deform360_v6_raw_nested_batch,
    validate_deform360_v6_raw_nested_batch,
)


def _load_many(paths: list[Path], *, label: str) -> list[dict[str, Any]]:
    return [
        dict(load_strict_json_object(path, label=f"{label} {index}"))
        for index, path in enumerate(paths)
    ]


def _locks(
    args: argparse.Namespace,
) -> dict[str, tuple[int, str]]:
    policy = load_deform360_fresh_object_session_v6_policy(args.policy)
    load_deform360_fresh_object_session_v6_covariance_amendment(
        args.amendment,
        policy,
    )
    load_deform360_v6_nested_source_repair(args.repair)
    _, cohort = load_deform360_v6_source_selection(args.selection, policy)
    return cohort


def _seal_batch(args: argparse.Namespace) -> int:
    cohort = _locks(args)
    records = _load_many(args.prediction, label="raw nested prediction")
    batch = build_deform360_v6_raw_nested_batch(records, cohort=cohort)
    publish_deform360_v6_raw_nested_batch(batch, args.output, cohort=cohort)
    print(json.dumps(batch, indent=2, sort_keys=True))
    return 0


def _assemble(args: argparse.Namespace) -> int:
    cohort = _locks(args)
    batch = validate_deform360_v6_raw_nested_batch(
        load_strict_json_object(args.prediction_batch, label="raw nested batch"),
        cohort=cohort,
    )
    outcomes = _load_many(args.outcome, label="raw nested outcome")
    evidence = assemble_deform360_v6_nested_evidence(
        prediction_batch=batch,
        outcomes=outcomes,
        cohort=cohort,
    )
    publish_deform360_v6_nested_evidence(
        evidence,
        args.output,
        cohort=cohort,
    )
    print(json.dumps(evidence, indent=2, sort_keys=True))
    return 0


def _evaluate(args: argparse.Namespace) -> int:
    cohort = _locks(args)
    evidence = load_strict_json_object(args.evidence, label="nested source evidence")
    result = evaluate_deform360_v6_nested_source_gate(evidence, cohort=cohort)
    publish_deform360_v6_nested_result(
        result,
        args.output,
        evidence=evidence,
        cohort=cohort,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["source_gate_passed"] else 3


def _common_locks(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--amendment", type=Path, required=True)
    parser.add_argument("--repair", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    seal = subparsers.add_parser(
        "seal-batch",
        help="seal exactly 100 outcome-free nested raw predictions",
    )
    _common_locks(seal)
    seal.add_argument("--prediction", type=Path, action="append", required=True)
    seal.add_argument("--output", type=Path, required=True)
    seal.set_defaults(run=_seal_batch)

    assemble = subparsers.add_parser(
        "assemble",
        help="attach exactly 100 source outcomes after the prediction barrier",
    )
    _common_locks(assemble)
    assemble.add_argument("--prediction-batch", type=Path, required=True)
    assemble.add_argument("--outcome", type=Path, action="append", required=True)
    assemble.add_argument("--output", type=Path, required=True)
    assemble.set_defaults(run=_assemble)

    evaluate = subparsers.add_parser(
        "evaluate",
        help="run the nested source-only candidate/covariance/guard gate",
    )
    _common_locks(evaluate)
    evaluate.add_argument("--evidence", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.set_defaults(run=_evaluate)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    return int(args.run(args))


if __name__ == "__main__":
    raise SystemExit(main())
