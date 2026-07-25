#!/usr/bin/env python3
"""Analyze the locked exploratory CoTracker3 directional-priority comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from bayesian_phystwin.phystwin_directional_priority_analysis import (
    analyze_directional_priority_results,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--hard", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=100_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260724)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = json.loads(args.source.read_text(encoding="utf-8"))
    hard = json.loads(args.hard.read_text(encoding="utf-8"))
    candidate = json.loads(args.candidate.read_text(encoding="utf-8"))
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    result = analyze_directional_priority_results(
        source,
        hard,
        candidate,
        protocol,
        bootstrap_draws=args.bootstrap_draws,
        bootstrap_seed=args.bootstrap_seed,
    )
    result["inputs"] = {
        "source": {
            "path": str(args.source.resolve()),
            "sha256": _sha256(args.source),
        },
        "hard": {
            "path": str(args.hard.resolve()),
            "sha256": _sha256(args.hard),
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
                "fresh_evaluation_justified": result[
                    "fresh_evaluation_justified"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
