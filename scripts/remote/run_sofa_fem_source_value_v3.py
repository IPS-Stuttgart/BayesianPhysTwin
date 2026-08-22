#!/usr/bin/env python3
"""Run the frozen SOFA FEM source-value prediction and outcome gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from bayesian_phystwin.sofa_fem_source_value_v3 import (
    finalize_sofa_fem_source_value_pre_prefix_v3,
    generate_sofa_fem_source_value_predictions_v3,
    score_sofa_fem_source_value_future_v3,
    score_sofa_fem_source_value_prefix_v3,
)


def _binding(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("bindings use GROUP_ID=/absolute/path")
    group_id, raw = value.split("=", 1)
    path = Path(raw)
    if not group_id or group_id.strip() != group_id or not path.is_absolute():
        raise argparse.ArgumentTypeError("binding is not canonical")
    return group_id, path


def _bindings(values: list[tuple[str, Path]]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for group_id, path in values:
        if group_id in result:
            raise ValueError(f"duplicate binding: {group_id}")
        result[group_id] = path
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in (
        "predict",
        "finalize-pre-prefix",
        "score-prefix",
        "score-future",
    ):
        command = subparsers.add_parser(name)
        command.add_argument("--protocol", type=Path, required=True)
        command.add_argument(
            "--group-root", action="append", type=_binding, required=True
        )
    predict = subparsers.choices["predict"]
    predict.add_argument("--output-dir", type=Path, required=True)
    predict.add_argument("--physics-protocol", type=Path, required=True)
    predict.add_argument("--physics-result", type=Path, required=True)
    predict.add_argument("--qualification", type=Path, required=True)
    predict.add_argument("--repo-root", type=Path, required=True)
    predict.add_argument("--distribution-archive", type=Path, required=True)
    predict.add_argument("--sofa-root", type=Path, required=True)
    prefix = subparsers.choices["score-prefix"]
    prefix.add_argument("--outcome-root", action="append", type=_binding, required=True)
    prefix.add_argument("--grid-dir", type=Path, required=True)
    prefix.add_argument("--pre-prefix-dir", type=Path, required=True)
    prefix.add_argument("--output-dir", type=Path, required=True)
    pre_prefix = subparsers.choices["finalize-pre-prefix"]
    pre_prefix.add_argument("--grid-dir", type=Path, required=True)
    pre_prefix.add_argument("--output-dir", type=Path, required=True)
    future = subparsers.choices["score-future"]
    future.add_argument("--outcome-root", action="append", type=_binding, required=True)
    future.add_argument("--prefix-dir", type=Path, required=True)
    future.add_argument("--grid-dir", type=Path, required=True)
    future.add_argument("--pre-prefix-dir", type=Path, required=True)
    future.add_argument("--output-path", type=Path, required=True)
    args = parser.parse_args()

    result: dict[str, Any]
    roots = _bindings(args.group_root)
    if args.command == "predict":
        result = generate_sofa_fem_source_value_predictions_v3(
            protocol_path=args.protocol,
            physics_protocol_path=args.physics_protocol,
            physics_result_path=args.physics_result,
            qualification_path=args.qualification,
            group_roots=roots,
            output_dir=args.output_dir,
            repo_root=args.repo_root,
            distribution_archive=args.distribution_archive,
            sofa_root=args.sofa_root,
        )
    elif args.command == "score-prefix":
        result = score_sofa_fem_source_value_prefix_v3(
            protocol_path=args.protocol,
            group_roots=roots,
            outcome_roots=_bindings(args.outcome_root),
            grid_dir=args.grid_dir,
            pre_prefix_dir=args.pre_prefix_dir,
            output_dir=args.output_dir,
        )
    elif args.command == "finalize-pre-prefix":
        result = finalize_sofa_fem_source_value_pre_prefix_v3(
            protocol_path=args.protocol,
            group_roots=roots,
            grid_dir=args.grid_dir,
            output_dir=args.output_dir,
        )
    else:
        result = score_sofa_fem_source_value_future_v3(
            protocol_path=args.protocol,
            group_roots=roots,
            outcome_roots=_bindings(args.outcome_root),
            prefix_dir=args.prefix_dir,
            grid_dir=args.grid_dir,
            pre_prefix_dir=args.pre_prefix_dir,
            output_path=args.output_path,
        )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
