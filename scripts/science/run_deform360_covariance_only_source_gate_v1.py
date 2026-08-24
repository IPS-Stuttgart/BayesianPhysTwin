#!/usr/bin/env python3
"""Seal or evaluate the frozen Deform360 covariance-only source gate."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from bayesian_phystwin.deform360_covariance_only_source_gate_v1 import (
    evaluate_source_gate,
    seal_prediction_batch,
    seal_source_scores,
    validate_prediction_batch,
)


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _load(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream, object_pairs_hook=_reject_duplicate_keys)
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return value


def _atomic_create(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to overwrite {path}")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    if temporary.exists() or temporary.is_symlink():
        raise FileExistsError(f"temporary output already exists: {temporary}")
    encoded = (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(temporary, flags, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(descriptor)
    os.replace(temporary, path)


def _seal_batch(args: argparse.Namespace) -> int:
    payload = seal_prediction_batch(_load(args.input))
    _atomic_create(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _validate_batch(args: argparse.Namespace) -> int:
    payload = validate_prediction_batch(_load(args.input))
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _seal_scores(args: argparse.Namespace) -> int:
    payload = seal_source_scores(_load(args.input))
    # Evaluation performs the batch-bound semantic validation.
    evaluate_source_gate(_load(args.batch), payload)
    _atomic_create(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _evaluate(args: argparse.Namespace) -> int:
    decision = evaluate_source_gate(_load(args.batch), _load(args.scores))
    _atomic_create(args.output, decision)
    print(json.dumps(decision, indent=2, sort_keys=True))
    return 0 if decision["status"] == "source-positive" else 3


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    seal_batch = commands.add_parser("seal-batch")
    seal_batch.add_argument("--input", type=Path, required=True)
    seal_batch.add_argument("--output", type=Path, required=True)
    seal_batch.set_defaults(run=_seal_batch)

    validate_batch = commands.add_parser("validate-batch")
    validate_batch.add_argument("--input", type=Path, required=True)
    validate_batch.set_defaults(run=_validate_batch)

    seal_scores = commands.add_parser("seal-scores")
    seal_scores.add_argument("--batch", type=Path, required=True)
    seal_scores.add_argument("--input", type=Path, required=True)
    seal_scores.add_argument("--output", type=Path, required=True)
    seal_scores.set_defaults(run=_seal_scores)

    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--batch", type=Path, required=True)
    evaluate.add_argument("--scores", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.set_defaults(run=_evaluate)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    return int(args.run(args))


if __name__ == "__main__":
    raise SystemExit(main())
