#!/usr/bin/env python3
"""Build the source-only fresh technical-replication artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

from bayesian_phystwin.deform360_pairwise_regret_guard_fresh_protocol import (
    build_exclusion_union,
    build_fresh_technical_lock,
    validate_exclusion_union,
    validate_fresh_technical_lock,
    write_json_artifact,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    union = subparsers.add_parser("union")
    union.add_argument("output", type=Path)
    union.add_argument("manifest", nargs="+", type=Path)

    lock = subparsers.add_parser("lock")
    lock.add_argument("output", type=Path)
    lock.add_argument("--exclusion-union", required=True, type=Path)
    lock.add_argument("--public-catalog", required=True, type=Path)
    lock.add_argument("--metadata", required=True, type=Path)
    lock.add_argument("--source-protocol", required=True, type=Path)
    lock.add_argument("--source-qualification", required=True, type=Path)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "union":
        artifact = build_exclusion_union(args.manifest)
        validate_exclusion_union(artifact)
    else:
        artifact = build_fresh_technical_lock(
            args.exclusion_union,
            args.public_catalog,
            args.metadata,
            args.source_protocol,
            args.source_qualification,
        )
        validate_fresh_technical_lock(artifact)
    write_json_artifact(artifact, args.output)


if __name__ == "__main__":
    main()
