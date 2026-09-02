#!/usr/bin/env python3
"""Validate, execute, and retain the source-only DLO2/DLO3 panel plan."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from plan_deform_dlo23_hierarchical_panel import plan


def _load(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _canonical_id(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--census", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    request = _load(args.request)
    census = _load(args.census)
    receipt = _load(args.receipt)

    if request.get("schema") != (
        "bayesian-phystwin.deform-dlo23-source-panel-plan-minimal-request"
    ):
        raise ValueError("unexpected source-panel request schema")
    if request.get("schema_version") != 3:
        raise ValueError("unexpected source-panel request version")
    if request.get("authorized_source_objects") != ["DLO2", "DLO3"]:
        raise ValueError("source object roster changed")
    if request.get("forbidden_target_objects") != ["DLO4", "DLO5"]:
        raise ValueError("protected target roster changed")
    if request.get("target_payload_read_authorized") is not False:
        raise ValueError("target payload access was authorized")
    if request.get("target_outcome_semantic_mapping_authorized") is not False:
        raise ValueError("target-informed semantic mapping was authorized")

    census_bytes = args.census.read_bytes()
    if hashlib.sha256(census_bytes).hexdigest() != receipt.get("result_sha256"):
        raise ValueError("source census differs from its receipt")
    if census.get("census_id") != receipt.get("census_id"):
        raise ValueError("source census ID mismatch")
    if request.get("source_census_id") != census.get("census_id"):
        raise ValueError("request is bound to another source census")
    if request.get("source_census_receipt_id") != receipt.get("receipt_id"):
        raise ValueError("request is bound to another census receipt")
    if receipt.get("dlo4_dlo5_payload_read") is not False:
        raise ValueError("source census crossed the DLO4/DLO5 boundary")
    if receipt.get("target_scores_read") is not False:
        raise ValueError("source census read target scores")

    result = plan(census)
    boundary = result["information_boundary"]
    forbidden_true = (
        "dlo4_payload_read",
        "dlo5_payload_read",
        "protected_parent_target_result_read",
        "target_outcome_used_for_semantic_mapping",
        "ambiguous_carrier_auto_selected",
    )
    if any(boundary[key] is not False for key in forbidden_true):
        raise ValueError("planner crossed a protected information boundary")

    args.output_dir.mkdir(parents=True, exist_ok=False)
    result_path = args.output_dir / "result.json"
    result_path.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    retained_receipt: dict[str, object] = {
        "schema": "bayesian-phystwin.deform-dlo23-source-panel-plan-receipt",
        "schema_version": 3,
        "run_id": int(os.environ.get("GITHUB_RUN_ID", "0")),
        "run_attempt": int(os.environ.get("GITHUB_RUN_ATTEMPT", "0")),
        "evaluated_commit_sha": os.environ.get("GITHUB_SHA"),
        "runner_name": os.environ.get("RUNNER_NAME"),
        "source_census_id": result["source_census_id"],
        "source_census_receipt_id": receipt["receipt_id"],
        "plan_id": result["plan_id"],
        "result_sha256": hashlib.sha256(result_path.read_bytes()).hexdigest(),
        "ready_for_panel_build": result["ready_for_panel_build"],
        "decisions": result["decisions"],
        "dlo4_dlo5_payload_read": False,
        "target_scores_read": False,
        "claim_scope": "source-only adapter planning",
    }
    retained_receipt["receipt_id"] = _canonical_id(retained_receipt)
    (args.output_dir / "receipt.json").write_text(
        json.dumps(retained_receipt, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    report = [
        "# DLO2/DLO3 hierarchical source-panel plan",
        "",
        f"- Plan ID: `{result['plan_id']}`",
        f"- Ready for panel build: `{str(result['ready_for_panel_build']).lower()}`",
        f"- Source census ID: `{result['source_census_id']}`",
        "- DLO4/DLO5 payload read: `false`",
        "- Target scores read: `false`",
        "",
        "## Per-source-object decisions",
        "",
    ]
    for dlo, decision in sorted(result["decisions"].items()):
        report.append(
            f"- **{dlo}**: `{decision['reason']}`; candidates "
            f"`{decision['candidate_count']}`; top score "
            f"`{decision['top_score']}`; margin `{decision['score_margin']}`."
        )
    (args.output_dir / "report.md").write_text(
        "\n".join(report) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "plan_id": result["plan_id"],
                "ready_for_panel_build": result["ready_for_panel_build"],
                "decisions": result["decisions"],
                "receipt_id": retained_receipt["receipt_id"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
