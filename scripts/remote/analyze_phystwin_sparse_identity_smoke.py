#!/usr/bin/env python3
"""Emit only locked fields from the sparse-identity development smoke."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from bayesian_phystwin.phystwin_sparse_identity_smoke_analysis import (
    analyze_sparse_identity_smoke_files,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--comparator", type=Path, required=True)
    parser.add_argument("--cue", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = analyze_sparse_identity_smoke_files(
        candidate_path=args.candidate,
        comparator_path=args.comparator,
        cue_path=args.cue,
        protocol_path=args.protocol,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
