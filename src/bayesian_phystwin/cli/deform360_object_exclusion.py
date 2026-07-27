"""Merge hash-only Deform360 object-exclusion manifests."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from bayesian_phystwin.deform360_object_exclusion import (
    file_sha256,
    merge_object_exclusion_manifests,
    write_object_exclusion_manifest,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--owner", required=True)
    parser.add_argument("--input", type=Path, action="append", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    artifact = merge_object_exclusion_manifests(args.input, owner=args.owner)
    write_object_exclusion_manifest(args.output, artifact)
    summary = {
        "output": str(args.output.resolve()),
        "file_sha256": file_sha256(args.output),
        "exclusion_sha256": artifact["exclusion_sha256"],
        "member_count": artifact["composition"]["member_count"],
        "input_hash_count": artifact["composition"]["input_hash_count"],
        "unique_object_hash_count": artifact["composition"][
            "unique_object_hash_count"
        ],
    }
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
