#!/usr/bin/env python3
"""Analyze frozen disjoint sparse-identity source reports."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from bayesian_phystwin.phystwin_disjoint_sparse_identity_analysis import (
    analyze_disjoint_sparse_identity_reports,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report",
        action="append",
        required=True,
        metavar="COUNT=PATH",
        help="raw headroom report for one observed-identity count",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--primary-observed-count", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reports = {}
    inputs = {}
    for specification in args.report:
        count_text, separator, path_text = specification.partition("=")
        if not separator:
            raise ValueError("report must use COUNT=PATH syntax")
        count = int(count_text)
        path = Path(path_text).resolve()
        if count in reports:
            raise ValueError(f"duplicate observed-count report: {count}")
        reports[count] = json.loads(path.read_text(encoding="utf-8"))
        inputs[str(count)] = {"path": str(path), "sha256": _sha256(path)}
    result = analyze_disjoint_sparse_identity_reports(
        reports,
        primary_observed_count=args.primary_observed_count,
    )
    result["inputs"] = inputs
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
