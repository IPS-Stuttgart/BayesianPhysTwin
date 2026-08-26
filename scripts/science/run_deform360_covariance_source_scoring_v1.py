#!/usr/bin/env python3
"""Seal and score the frozen Deform360 covariance source panel."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from bayesian_phystwin.strict_json_report_io import load_strict_json_mapping
from bayesian_phystwin_experiments.deform360_covariance_source_scoring_v1 import (
    publish_covariance_source_scores_v1,
    seal_source_observations_v1,
    validate_covariance_source_scores_v1,
    validate_source_observations_v1,
)


def _write_json_once(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    seal = commands.add_parser(
        "seal-observations",
        help="content-address a complete source-observation manifest",
    )
    seal.add_argument("--input", type=Path, required=True)
    seal.add_argument("--output", type=Path, required=True)

    validate_observations = commands.add_parser(
        "validate-observations",
        help="validate source-observation metadata without reading arrays",
    )
    validate_observations.add_argument("path", type=Path)

    score = commands.add_parser(
        "score",
        help="attach the source suffix once and publish the frozen decision",
    )
    score.add_argument("--panel-root", type=Path, required=True)
    score.add_argument("--source-observations", type=Path, required=True)
    score.add_argument("--source-observation-root", type=Path, required=True)
    score.add_argument("--forbidden-confirmation-root", type=Path, required=True)
    score.add_argument("--output-root", type=Path, required=True)

    validate_result = commands.add_parser(
        "validate-result",
        help="rehash a published source-score result without opening arrays",
    )
    validate_result.add_argument("path", type=Path)
    return parser


def _load_mapping(path: Path, *, label: str) -> dict[str, Any]:
    value, _ = load_strict_json_mapping(path, artifact_label=label)
    return dict(value)


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "seal-observations":
        value = seal_source_observations_v1(
            _load_mapping(args.input, label="source observations")
        )
        _write_json_once(args.output, value)
        print(json.dumps({"observation_set_id": value["observation_set_id"]}))
        return 0
    if args.command == "validate-observations":
        value = validate_source_observations_v1(
            _load_mapping(args.path, label="source observations")
        )
        print(json.dumps({"observation_set_id": value["observation_set_id"]}))
        return 0
    if args.command == "score":
        receipt = publish_covariance_source_scores_v1(
            panel_root=args.panel_root,
            source_observations_path=args.source_observations,
            source_observation_root=args.source_observation_root,
            forbidden_confirmation_root=args.forbidden_confirmation_root,
            output_root=args.output_root,
        )
        print(json.dumps(receipt, sort_keys=True))
        return 0
    if args.command == "validate-result":
        receipt = validate_covariance_source_scores_v1(args.path)
        print(json.dumps(receipt, sort_keys=True))
        return 0
    raise AssertionError("unreachable command")


if __name__ == "__main__":
    raise SystemExit(main())
