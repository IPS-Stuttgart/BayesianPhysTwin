#!/usr/bin/env python3
"""Render non-sensitive receipts from a strictly validated execution record."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from bayesian_phystwin.deform360_calibration_source_run_record import (
    load_deform360_calibration_source_run_record,
)


def _load_bound_record(
    path: Path,
    *,
    source_revision: str,
    workflow_run_id: int,
    workflow_run_attempt: int,
) -> dict[str, Any]:
    record = load_deform360_calibration_source_run_record(path)
    if record["source_revision"] != source_revision:
        raise ValueError("execution-record source changed")
    if record["workflow_run_id"] != workflow_run_id:
        raise ValueError("execution-record run changed")
    if record["workflow_run_attempt"] != workflow_run_attempt:
        raise ValueError("execution-record attempt changed")
    return record


def _artifact_lines(record: Mapping[str, Any]) -> list[str]:
    lines: list[str] = []
    for label, valid_key, error_key, digest_key in (
        ("plan", "plan_valid", "plan_error", "plan_sha256"),
        ("download", "download_valid", "download_error", "download_sha256"),
        ("result", "result_valid", "result_error", "result_sha256"),
    ):
        if record[valid_key] is True:
            lines.append(f"- {label} digest: `{record[digest_key]}`")
        elif record[error_key] is not None:
            lines.append(f"- {label} record: `{record[error_key]}`")
    plan_gate = record["plan_support_gate"]
    if isinstance(plan_gate, Mapping):
        admitted_strata = json.dumps(
            plan_gate["supported_by_stratum"],
            sort_keys=True,
            separators=(",", ":"),
        )
        lines.extend(
            [
                f"- admitted objects: `{plan_gate['supported_object_count']}`",
                f"- admitted by stratum: `{admitted_strata}`",
                f"- admission gate passed: `{plan_gate['support_passed']}`",
            ]
        )
    support_gate = record["support_gate"]
    if isinstance(support_gate, Mapping):
        prepared_strata = json.dumps(
            support_gate["supported_by_stratum"],
            sort_keys=True,
            separators=(",", ":"),
        )
        lines.extend(
            [
                f"- prepared objects: `{support_gate['supported_object_count']}`",
                f"- prepared by stratum: `{prepared_strata}`",
                f"- support gate passed: `{support_gate['support_passed']}`",
            ]
        )
    return lines


def issue_body(
    record: Mapping[str, Any] | None,
    *,
    source_revision: str,
    workflow_run_id: int,
    workflow_run_attempt: int,
) -> str:
    """Render one issue-comment body without paths or object identities."""

    lines = [
        "## Direct ten-object calibration-source execution completed",
        "",
        f"- workflow run: `{workflow_run_id}`",
        f"- workflow attempt: `{workflow_run_attempt}`",
        f"- exact reviewed source revision: `{source_revision}`",
    ]
    if record is None:
        lines.extend(
            [
                "- status: `execution-record-unavailable`",
                "- confirmation boundary verified: `unknown`",
            ]
        )
    else:
        lines.extend(
            [
                f"- status: `{record['status']}`",
                f"- exit code: `{record['exit_code']}`",
                f"- failure stage: `{record['failure_stage']}`",
                (
                    "- confirmation boundary verified: "
                    f"`{record['confirmation_boundary_verified']}`"
                ),
                f"- execution-record digest: `{record['record_sha256']}`",
                *_artifact_lines(record),
            ]
        )
    lines.extend(
        [
            "",
            (
                "This receipt contains no local paths, object identities, "
                "or target outcomes."
            ),
        ]
    )
    return "\n".join(lines)


def summary_lines(record: Mapping[str, Any] | None) -> list[str]:
    """Render compact job-summary lines from the same strict record."""

    if record is None:
        return ["- Execution record: `unavailable or invalid`"]
    lines = [
        f"- Status: `{record['status']}`",
        f"- Exit code: `{record['exit_code']}`",
        (
            "- Confirmation boundary verified: "
            f"`{record['confirmation_boundary_verified']}`"
        ),
        *_artifact_lines(record),
        f"- Execution-record digest: `{record['record_sha256']}`",
    ]
    return lines


def _record_or_none(args: argparse.Namespace) -> dict[str, Any] | None:
    try:
        return _load_bound_record(
            args.manifest,
            source_revision=args.source_revision,
            workflow_run_id=args.workflow_run_id,
            workflow_run_attempt=args.workflow_run_attempt,
        )
    except (ImportError, KeyError, OSError, TypeError, ValueError):
        return None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("issue", "summary"))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--workflow-run-id", type=int, required=True)
    parser.add_argument("--workflow-run-attempt", type=int, required=True)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    record = _record_or_none(args)
    if args.mode == "summary":
        print("\n".join(summary_lines(record)))
        return 0
    if args.output is None:
        raise SystemExit("--output is required for issue mode")
    body = issue_body(
        record,
        source_revision=args.source_revision,
        workflow_run_id=args.workflow_run_id,
        workflow_run_attempt=args.workflow_run_attempt,
    )
    args.output.write_text(
        json.dumps({"body": body}, sort_keys=True),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
