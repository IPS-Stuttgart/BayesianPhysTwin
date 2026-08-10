"""Select one source-only discrepancy candidate under a frozen tournament."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from bayesian_phystwin.discrepancy_candidate_tournament import (
    analyze_discrepancy_candidate_tournament,
    load_discrepancy_candidate_tournament_input,
    publish_discrepancy_candidate_tournament_report,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_json", type=Path)
    parser.add_argument("output_json", type=Path)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace an existing report instead of failing closed",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload, input_artifact = load_discrepancy_candidate_tournament_input(
        args.input_json
    )
    report = analyze_discrepancy_candidate_tournament(payload)
    emitted = publish_discrepancy_candidate_tournament_report(
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
                "selected_candidate": emitted["selected_candidate"],
                "source_gate_passed": emitted["source_gate_passed"],
                "decision": emitted["decision"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if bool(emitted["source_gate_passed"]) else 3


if __name__ == "__main__":
    raise SystemExit(main())
