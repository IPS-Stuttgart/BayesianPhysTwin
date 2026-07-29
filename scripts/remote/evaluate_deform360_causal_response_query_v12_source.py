#!/usr/bin/env python3
"""Summarize the frozen target-free V12 source query-feasibility audit."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from bayesian_phystwin.deform360_causal_response_query import (
    QUERY_REPORT_FILENAME,
    validate_causal_response_query_artifacts,
)
from bayesian_phystwin.observation_belief import file_sha256

CONFIG_RELATIVE_PATH = Path(
    "configs/sota/deform360_causal_response_query_feasibility_v12.json"
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _canonical_sha256(payload: dict[str, Any], *, key: str) -> str:
    canonical = dict(payload)
    canonical.pop(key, None)
    return hashlib.sha256(
        json.dumps(
            canonical,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    repo = args.repo.resolve()
    artifact_root = args.artifact_root.resolve()
    config_path = repo / CONFIG_RELATIVE_PATH
    protocol = json.loads(config_path.read_text(encoding="utf-8"))
    _require(
        protocol.get("config_sha256")
        == _canonical_sha256(protocol, key="config_sha256"),
        "protocol checksum changed",
    )
    rows: list[dict[str, Any]] = []
    repository_revisions: set[str] = set()
    for case_record in protocol["cases"]:
        case_id = str(case_record["case"])
        case_dir = artifact_root / case_id
        report_path = case_dir / QUERY_REPORT_FILENAME
        if not report_path.is_file():
            rows.append(
                {
                    "case": case_id,
                    "status": "technical_failure",
                    "admitted": False,
                    "eligible_entity_count": None,
                    "selected_entity_count": None,
                    "result_sha256": None,
                }
            )
            continue
        report, arrays = validate_causal_response_query_artifacts(case_dir)
        _require(report["case"] == case_id, "query report case changed")
        _require(
            report["inputs_sha256"]["protocol"] == file_sha256(config_path),
            "query report used a different protocol file",
        )
        repository_revisions.add(str(report["repository_revision"]))
        schedule = report["schedule"]
        rows.append(
            {
                "case": case_id,
                "status": report["status"],
                "admitted": bool(schedule["admitted"]),
                "eligible_entity_count": int(schedule["eligible_entity_count"]),
                "selected_entity_count": int(len(arrays["entity_ids"])),
                "result_sha256": report["result_sha256"],
            }
        )
    _require(
        len(repository_revisions) <= 1,
        "source reports used multiple repository revisions",
    )
    admitted_count = sum(row["admitted"] for row in rows)
    technical_failure_count = sum(row["status"] == "technical_failure" for row in rows)
    gate = protocol["source_feasibility_gate"]
    gate_passed = admitted_count >= int(
        gate["minimum_admitted_case_count"]
    ) and technical_failure_count <= int(gate["maximum_technical_failure_count"])
    result: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": "Deform360CausalResponseQueryFeasibilityV12SourceResult",
        "protocol_id": protocol["protocol_id"],
        "status": "passed" if gate_passed else "failed",
        "repository_revision": (
            next(iter(repository_revisions)) if repository_revisions else None
        ),
        "protocol_file_sha256": file_sha256(config_path),
        "case_count": len(rows),
        "admitted_case_count": admitted_count,
        "technical_failure_count": technical_failure_count,
        "gate": {
            **gate,
            "passed": gate_passed,
        },
        "cases": rows,
        "information_boundary": {
            "maximum_object_observation_frame": 0,
            "future_identity_read": False,
            "future_object_observation_read": False,
            "future_metric_read": False,
            "state_update_constructed": False,
            "tactile_read": False,
            "v1_sealed_target_read": False,
            "held_v8_artifact_or_process_access": False,
        },
    }
    result["result_sha256"] = _canonical_sha256(result, key="result_sha256")
    output = args.output.resolve()
    _require(not output.exists(), "source summary already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
