#!/usr/bin/env python3
"""Score authorized cases or aggregate the prospective Deform360 study."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from bayesian_phystwin.deform360_selective_virtual_sensing_evaluation import (
    CASE_EVALUATION_FILENAME,
    aggregate_selective_virtual_sensing_evaluations,
    evaluate_selective_virtual_sensing_case,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    case = subparsers.add_parser("case")
    case.add_argument("--protocol", type=Path, required=True)
    case.add_argument("--cohort-seal", type=Path, required=True)
    case.add_argument("--prediction-root", type=Path, required=True)
    case.add_argument("--failure-root", type=Path, required=True)
    case.add_argument("--measurement-root", type=Path, required=True)
    case.add_argument("--outcome-root", type=Path, required=True)
    case.add_argument("--object-id", required=True)
    case.add_argument("--episode-id", type=int, required=True)
    case.add_argument("--output-root", type=Path, required=True)

    aggregate = subparsers.add_parser("aggregate")
    aggregate.add_argument("--protocol", type=Path, required=True)
    aggregate.add_argument("--cohort-seal", type=Path, required=True)
    aggregate.add_argument("--evaluation-root", type=Path, required=True)
    aggregate.add_argument("--prediction-root", type=Path, required=True)
    aggregate.add_argument("--failure-root", type=Path, required=True)
    aggregate.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    cohort = json.loads(args.cohort_seal.read_text(encoding="utf-8"))
    if args.command == "case":
        payload = evaluate_selective_virtual_sensing_case(
            args.protocol,
            cohort,
            args.prediction_root,
            args.failure_root,
            args.measurement_root,
            args.outcome_root,
            object_id=args.object_id,
            episode_id=args.episode_id,
        )
        destination = (
            args.output_root.resolve()
            / str(payload["case"])
            / CASE_EVALUATION_FILENAME
        )
    else:
        payload = aggregate_selective_virtual_sensing_evaluations(
            args.protocol,
            cohort,
            args.evaluation_root,
            args.prediction_root,
            args.failure_root,
        )
        destination = args.output.resolve()
    if destination.exists():
        raise FileExistsError(f"evaluation already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
