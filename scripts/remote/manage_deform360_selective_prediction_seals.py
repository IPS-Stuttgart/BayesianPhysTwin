#!/usr/bin/env python3
"""Record target-free failures or seal the prospective prediction cohort."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from bayesian_phystwin.deform360_selective_virtual_sensing_artifacts import (
    build_selective_prediction_cohort_seal,
    record_selective_quality_failure,
)


def _parse_evidence(values: list[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        name, separator, path = value.partition("=")
        if not separator or not name or not path or name in result:
            raise ValueError(f"invalid or repeated evidence NAME=PATH: {value}")
        result[name] = Path(path)
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    failure = subparsers.add_parser("record-failure")
    failure.add_argument("--protocol", type=Path, required=True)
    failure.add_argument("--failure-root", type=Path, required=True)
    failure.add_argument("--object-id", required=True)
    failure.add_argument("--episode-id", type=int, required=True)
    failure.add_argument("--stage", required=True)
    failure.add_argument("--error-type", required=True)
    failure.add_argument("--error-message", required=True)
    failure.add_argument("--evidence", action="append", default=[])

    cohort = subparsers.add_parser("seal-cohort")
    cohort.add_argument("--protocol", type=Path, required=True)
    cohort.add_argument("--prediction-root", type=Path, required=True)
    cohort.add_argument("--failure-root", type=Path, required=True)
    cohort.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.command == "record-failure":
        case_name = f"{args.object_id}-ep{args.episode_id:04d}"
        result = record_selective_quality_failure(
            args.protocol,
            args.failure_root.resolve() / case_name,
            object_id=args.object_id,
            episode_id=args.episode_id,
            stage=args.stage,
            error_type=args.error_type,
            error_message=args.error_message,
            evidence_paths=_parse_evidence(args.evidence),
        )
    else:
        result = build_selective_prediction_cohort_seal(
            args.protocol,
            args.prediction_root,
            args.failure_root,
            args.output,
        )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
