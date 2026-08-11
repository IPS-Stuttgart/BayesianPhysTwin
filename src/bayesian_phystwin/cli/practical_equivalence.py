"""Assess prospective practical equivalence for matched physical losses."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

from bayesian_phystwin.practical_equivalence import assess_practical_equivalence
from bayesian_phystwin.strict_json_report_io import (
    DEFAULT_MAXIMUM_INPUT_BYTES,
    load_strict_json_mapping,
    publish_json_report,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence_json", type=Path)
    parser.add_argument("policy_json", type=Path)
    parser.add_argument("report_json", type=Path)
    parser.add_argument(
        "--maximum-input-bytes",
        type=int,
        default=DEFAULT_MAXIMUM_INPUT_BYTES,
        help="fail when either strict input exceeds this byte budget",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="atomically replace an existing report instead of failing closed",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    evidence, evidence_artifact = load_strict_json_mapping(
        args.evidence_json,
        artifact_label="practical-equivalence evidence",
        maximum_input_bytes=args.maximum_input_bytes,
    )
    policy, policy_artifact = load_strict_json_mapping(
        args.policy_json,
        artifact_label="practical-equivalence policy",
        maximum_input_bytes=args.maximum_input_bytes,
    )
    report = assess_practical_equivalence(evidence, policy)
    published = publish_json_report(
        args.report_json,
        report,
        input_artifact={
            "evidence": evidence_artifact,
            "policy": policy_artifact,
        },
        overwrite=args.overwrite,
    )
    summary = published.get("summary")
    if not isinstance(summary, Mapping):
        raise AssertionError("practical-equivalence report summary changed type")
    print(
        json.dumps(
            {
                "status": "written",
                "report": str(args.report_json.resolve(strict=False)),
                "report_id": published["report_id"],
                "policy_id": published["policy_id"],
                "overall_decision": summary["overall_decision"],
                "claim_authorized": published["claim_authorized"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
