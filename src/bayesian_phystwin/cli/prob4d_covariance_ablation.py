"""Audit a controlled five-way Prob4D covariance ablation."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from bayesian_phystwin.prob4d_covariance_ablation import (
    analyze_prob4d_covariance_ablation,
)
from bayesian_phystwin.strict_json_report_io import (
    load_strict_json_mapping,
    publish_json_report,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_json", type=Path)
    parser.add_argument("output_json", type=Path)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace an existing output instead of failing closed",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload, input_artifact = load_strict_json_mapping(
        args.input_json,
        artifact_label="Prob4D covariance-ablation",
    )
    report = analyze_prob4d_covariance_ablation(payload)
    emitted = publish_json_report(
        args.output_json,
        report,
        input_artifact=input_artifact,
        overwrite=args.overwrite,
    )
    summary = emitted["decisive_evidence_summary"]
    metrics = summary["metrics"]
    print(
        json.dumps(
            {
                "status": "written",
                "output": str(args.output_json.resolve()),
                "report_id": emitted["report_id"],
                "ablation_id": emitted["ablation_id"],
                "variant_count": len(emitted["variants"]),
                "metric_count": len(metrics),
                "claim_authorized": emitted["claim_authorized"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
