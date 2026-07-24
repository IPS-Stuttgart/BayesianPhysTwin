#!/usr/bin/env python3
"""Analyze the locked AllTracker source-depth development smoke."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from bayesian_phystwin.phystwin_tracker_source_comparison import (
    analyze_tracker_source_comparison,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparator", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    comparator = json.loads(args.comparator.read_text(encoding="utf-8"))
    candidate = json.loads(args.candidate.read_text(encoding="utf-8"))
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    result = analyze_tracker_source_comparison(
        comparator,
        candidate,
        protocol,
    )
    result["inputs"] = {
        "comparator": {
            "path": str(args.comparator.resolve()),
            "sha256": _sha256(args.comparator),
        },
        "candidate": {
            "path": str(args.candidate.resolve()),
            "sha256": _sha256(args.candidate),
        },
        "protocol": {
            "path": str(args.protocol.resolve()),
            "sha256": _sha256(args.protocol),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "gates": result["gates"],
                "smoke_gate_passed": result["smoke_gate_passed"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
