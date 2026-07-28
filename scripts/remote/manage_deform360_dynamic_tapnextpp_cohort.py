#!/usr/bin/env python3
"""Seal terminal dispositions and lock the dynamic TAPNext++ cohort."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from bayesian_phystwin.deform360_dynamic_tapnextpp_cohort import (
    build_dynamic_provider_cohort_lock,
    build_terminal_disposition,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    terminal = commands.add_parser("terminal")
    terminal.add_argument("--output", type=Path, required=True)
    terminal.add_argument("--queue", type=Path, required=True)
    terminal.add_argument("--queue-rank", type=int, required=True)
    terminal.add_argument(
        "--stage",
        choices=("mask", "source_processing", "window_stage"),
        required=True,
    )
    terminal.add_argument("--reason-code", required=True)
    terminal.add_argument("--evidence", type=Path, required=True)
    terminal.add_argument("--producer-commit", required=True)

    lock = commands.add_parser("lock")
    lock.add_argument("--output", type=Path, required=True)
    lock.add_argument("--protocol", type=Path, required=True)
    lock.add_argument("--source-evaluation-protocol", type=Path, required=True)
    lock.add_argument("--queue", type=Path, required=True)
    lock.add_argument("--processing-protocol", type=Path, required=True)
    lock.add_argument("--runtime-amendment", type=Path, required=True)
    lock.add_argument("--admission-root", type=Path, required=True)
    lock.add_argument("--terminal-root", type=Path, required=True)
    lock.add_argument("--provider-commit", required=True)
    lock.add_argument("--source-processing-commit", required=True)
    lock.add_argument("--cohort-lock-builder-commit", required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.command == "terminal":
        result = build_terminal_disposition(
            args.output,
            queue_path=args.queue,
            queue_rank=args.queue_rank,
            stage=args.stage,
            reason_code=args.reason_code,
            evidence_path=args.evidence,
            producer_commit=args.producer_commit,
        )
    else:
        result = build_dynamic_provider_cohort_lock(
            args.output,
            protocol_path=args.protocol,
            source_evaluation_protocol_path=args.source_evaluation_protocol,
            queue_path=args.queue,
            processing_protocol_path=args.processing_protocol,
            runtime_amendment_path=args.runtime_amendment,
            admission_paths=sorted(args.admission_root.glob("*.admission.json")),
            terminal_disposition_paths=sorted(
                args.terminal_root.glob("*.terminal.json")
            ),
            provider_commit=args.provider_commit,
            source_processing_commit=args.source_processing_commit,
            cohort_lock_builder_commit=args.cohort_lock_builder_commit,
        )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
