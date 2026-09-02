"""Nested cross-material expert routing for the public Tracking Cloth study.

This is a retrospective follow-up to ``tracking_cloth_selective_twin_v1``.
All target outcomes were already open before this implementation was designed.
The outer material is excluded from model fitting and from inner
hyperparameter/threshold selection, but the result is not fresh confirmation.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from experiments.tracking_cloth_deformation_v1.data import (
    audit_dataset,
    infer_source_scale,
    object_digest,
    read_prefix,
    write_json,
)
from experiments.tracking_cloth_selective_twin_v1.run import (
    _source_predictions,
    _twist_predictions,
    score_records,
)

from .analysis import analyze_rows, report
from .ridge import prepare_rows

HERE = Path(__file__).resolve().parent
BASE_HERE = HERE.parent / "tracking_cloth_deformation_v1"
V1_HERE = HERE.parent / "tracking_cloth_selective_twin_v1"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing empty table: {path.name}")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def execute(
    dataset_root: Path,
    output: Path,
    protocol_path: Path,
) -> None:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    base_protocol = json.loads(
        (BASE_HERE / "protocol.json").read_text(encoding="utf-8")
    )
    v1_protocol = json.loads((V1_HERE / "protocol.json").read_text(encoding="utf-8"))
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "protocol.json", protocol)
    write_json(output / "base_protocol.json", base_protocol)
    write_json(output / "v1_protocol.json", v1_protocol)

    cases, inventory = audit_dataset(dataset_root, base_protocol)
    write_json(output / "dataset_manifest.json", inventory)
    (output / "DATA_LICENSE.txt").write_text(
        inventory["included_license_text"],
        encoding="utf-8",
    )
    source_cases = [case for case in cases if case.motion == "shake"]
    scales = [
        infer_source_scale(
            case,
            read_prefix(case, base_protocol["prefix_seconds"])[1],
        )
        for case in source_cases
    ]
    if len(set(scales)) != 1:
        raise ValueError("Source recordings disagree about coordinate units")
    scale = scales[0]
    source_records, weights = _source_predictions(
        cases,
        base_protocol,
        scale,
    )
    twist_records = _twist_predictions(
        cases,
        base_protocol,
        scale,
        weights,
    )
    query_rows = score_records(
        source_records + twist_records,
        v1_protocol,
    )
    policy_rows, result = analyze_rows(
        query_rows,
        protocol,
        v1_protocol,
    )
    result["dataset"] = {
        "record": protocol["dataset_record"],
        "inventory_id": inventory["inventory_id"],
        "csv_count": inventory["csv_count"],
        "source_count": inventory["source_count"],
        "target_count": inventory["target_count"],
        "unused_count": inventory["unused_count"],
    }
    result["result_id"] = object_digest(result)

    _write_csv(output / "query_cases.csv", prepare_rows(query_rows))
    _write_csv(output / "policy_cases.csv", policy_rows)
    write_json(output / "result.json", result)
    (output / "report.md").write_text(report(result), encoding="utf-8")
    write_json(
        output / "run_manifest.json",
        {
            "schema": "bayesian-phystwin.tracking-cloth-selective-router-run.v2",
            "schema_version": 2,
            "created_at": now(),
            "github_sha": os.environ.get("GITHUB_SHA"),
            "github_run_id": os.environ.get("GITHUB_RUN_ID"),
            "github_run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT"),
            "runner_name": os.environ.get("RUNNER_NAME"),
            "python": sys.version,
            "numpy": np.__version__,
            "platform": platform.platform(),
            "protocol_id": object_digest(protocol),
            "inventory_id": inventory["inventory_id"],
            "result_id": result["result_id"],
            "raw_trajectory_upload": False,
            "paper_claim_authorized": False,
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=HERE / "protocol.json",
    )
    args = parser.parse_args()
    try:
        execute(args.dataset_root, args.output, args.protocol)
    except Exception as exc:
        args.output.mkdir(parents=True, exist_ok=True)
        write_json(
            args.output / "failure.json",
            {
                "schema": (
                    "bayesian-phystwin.tracking-cloth-selective-router-failure.v2"
                ),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "scientific_conclusion": None,
            },
        )
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
