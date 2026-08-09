"""Classify source-only provider and guarded-update failure evidence."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from bayesian_phystwin.provider_failure_decomposition import (
    analyze_provider_failure_evidence,
)
from bayesian_phystwin.provider_failure_report_io import (
    load_provider_failure_input,
    publish_provider_failure_report,
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
    payload, input_artifact = load_provider_failure_input(args.input_json)
    report = analyze_provider_failure_evidence(payload)
    emitted = publish_provider_failure_report(
        args.output_json,
        report,
        input_artifact=input_artifact,
        overwrite=args.overwrite,
    )
    print(
        json.dumps(
            {
                "status": "written",
                "output": str(args.output_json.resolve()),
                "report_id": emitted["report_id"],
                "status_sha256": emitted["status_sha256"],
                "record_count": emitted["record_count"],
                "accepted_count": emitted["accepted_count"],
                "classified_rejection_count": emitted[
                    "classified_rejection_count"
                ],
                "unresolved_rejection_count": emitted[
                    "unresolved_rejection_count"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
