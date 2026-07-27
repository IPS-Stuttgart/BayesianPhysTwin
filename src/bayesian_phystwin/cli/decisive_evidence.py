"""Summarize matched guarded evidence for prospective PhysTwin experiments."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

from bayesian_phystwin.decisive_evidence import (
    DEFAULT_REGRESSION_QUANTILES,
    DEFAULT_RELIABILITY_EDGES,
    DEFAULT_TARGET_COVERAGES,
    analyze_decisive_evidence,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_json", type=Path)
    parser.add_argument("output_json", type=Path)
    parser.add_argument(
        "--reference-method",
        help="method used for paired matched-coverage comparisons",
    )
    parser.add_argument(
        "--coverage",
        action="append",
        type=float,
        dest="coverages",
        help="target coverage; repeat in strictly increasing order",
    )
    parser.add_argument(
        "--regression-quantile",
        action="append",
        type=float,
        dest="regression_quantiles",
        help="high relative-regression quantile; repeat in increasing order",
    )
    parser.add_argument(
        "--reliability-edge",
        action="append",
        type=float,
        dest="reliability_edges",
        help="reliability-bin edge; repeat from 0 through 1",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace an existing output instead of failing closed",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    input_path = args.input_json.resolve()
    output_path = args.output_json.resolve()
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(output_path)
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("input JSON must contain an object")
    summary = analyze_decisive_evidence(
        payload,
        target_coverages=(
            DEFAULT_TARGET_COVERAGES
            if args.coverages is None
            else tuple(args.coverages)
        ),
        regression_quantiles=(
            DEFAULT_REGRESSION_QUANTILES
            if args.regression_quantiles is None
            else tuple(args.regression_quantiles)
        ),
        reliability_edges=(
            DEFAULT_RELIABILITY_EDGES
            if args.reliability_edges is None
            else tuple(args.reliability_edges)
        ),
        reference_method=args.reference_method,
    )
    summary["input_artifact"] = {
        "path": str(input_path),
        "sha256": _sha256(input_path),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": "written",
                "output": str(output_path),
                "protocol_id": summary["protocol_id"],
                "metric_count": len(summary["metrics"]),
                "reference_method": summary["reference_method"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
