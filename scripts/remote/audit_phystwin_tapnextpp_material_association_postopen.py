#!/usr/bin/env python3
"""Audit source-frame TAPNext++ associations against fixed material identities."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from bayesian_phystwin.phystwin_official_evaluation import _nearest_distances
from bayesian_phystwin.phystwin_tapnextpp_competence import (
    canonical_sha256,
    file_sha256,
)

REPORT_FILENAME = "tapnextpp_sparse_assimilation_prediction_report.json"
PREDICTION_FILENAME = "tapnextpp_sparse_assimilation_prediction.npz"
OUTCOME_FILENAME = "withheld_future_outcome.npz"
SUMMARY_FILENAME = "tapnextpp_sparse_assimilation_source_summary.json"
OUTPUT_FILENAME = "tapnextpp_material_association_postopen_audit.json"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"JSON artifact is not an object: {path}")
    return value


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _case_audit(case_root: Path, case_name: str) -> dict[str, Any]:
    report_path = case_root / "prediction" / REPORT_FILENAME
    prediction_path = case_root / "prediction" / PREDICTION_FILENAME
    outcome_path = case_root / "withheld_outcome" / OUTCOME_FILENAME
    report = _load_json(report_path)
    _require(
        report.get("result_sha256") == canonical_sha256(report),
        f"prediction report checksum changed for {case_name}",
    )
    association = report.get("sparse_update", {}).get("association")
    if not isinstance(association, dict):
        return {
            "case": case_name,
            "disposition": "exact_fallback_without_sparse_association",
            "exact_match_count": 0,
            "scored_identity_count": 0,
            "inputs": {
                "prediction_archive_sha256": file_sha256(prediction_path),
                "prediction_report_sha256": file_sha256(report_path),
                "withheld_outcome_sha256": file_sha256(outcome_path),
            },
            "identities": [],
        }

    map_indices = np.asarray(association["map_indices"], dtype=np.int64)
    with np.load(prediction_path, allow_pickle=False) as stored:
        frame_zero = np.asarray(stored["physical_frame_zero_m"], dtype=np.float64)
        provider_ids = np.asarray(stored["provider_identity_ids"], dtype=np.int64)
    with np.load(outcome_path, allow_pickle=False) as stored:
        manual_frame_zero = np.asarray(
            stored["manual_track_frame_zero_m"],
            dtype=np.float64,
        )
        withheld_ids = np.asarray(stored["provider_identity_ids"], dtype=np.int64)

    _require(np.array_equal(provider_ids, withheld_ids), "provider identities changed")
    _require(len(map_indices) == len(provider_ids), "association count changed")
    _require(frame_zero.ndim == 2 and frame_zero.shape[1] == 3, "frame zero changed")

    identities: list[dict[str, Any]] = []
    exact_count = 0
    scored_count = 0
    for provider_id, source_map_node in zip(
        provider_ids.tolist(),
        map_indices.tolist(),
        strict=True,
    ):
        _require(
            0 <= provider_id < len(manual_frame_zero),
            "provider identity is outside manual identity range",
        )
        manual_point = manual_frame_zero[provider_id]
        if not np.all(np.isfinite(manual_point)):
            identities.append(
                {
                    "provider_identity_id": int(provider_id),
                    "status": "manual_frame_zero_unavailable",
                    "source_frame_map_node": int(source_map_node),
                }
            )
            continue
        distance, benchmark_node = _nearest_distances(
            frame_zero,
            manual_point[None],
            p=2,
        )
        benchmark_node_id = int(benchmark_node[0])
        exact = benchmark_node_id == int(source_map_node)
        scored_count += 1
        exact_count += int(exact)
        identities.append(
            {
                "provider_identity_id": int(provider_id),
                "status": "scored",
                "source_frame_map_node": int(source_map_node),
                "benchmark_frame_zero_node": benchmark_node_id,
                "exact_node_match": exact,
                "benchmark_attachment_distance_m": float(distance[0]),
                "source_map_to_benchmark_point_distance_m": float(
                    np.linalg.norm(frame_zero[source_map_node] - manual_point)
                ),
            }
        )

    return {
        "case": case_name,
        "disposition": "post_open_material_association_audit",
        "exact_match_count": exact_count,
        "scored_identity_count": scored_count,
        "exact_match_fraction": exact_count / scored_count if scored_count else None,
        "inputs": {
            "prediction_archive_sha256": file_sha256(prediction_path),
            "prediction_report_sha256": file_sha256(report_path),
            "withheld_outcome_sha256": file_sha256(outcome_path),
        },
        "identities": identities,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    source_root = args.source_root.resolve()
    summary_path = source_root / SUMMARY_FILENAME
    summary = _load_json(summary_path)
    _require(
        summary.get("result_sha256") == canonical_sha256(summary),
        "source summary checksum changed",
    )
    _require(
        summary.get("decision") == "stop-before-independent-evaluation",
        "post-open audit requires the stopped source study",
    )

    cases = [
        str(record["case"])
        for record in summary.get("case_dispositions", [])
    ]
    _require(len(cases) == int(summary["case_count"]), "case count changed")
    case_audits = [
        _case_audit(source_root / "cases" / case_name, case_name)
        for case_name in cases
    ]
    exact_count = sum(record["exact_match_count"] for record in case_audits)
    scored_count = sum(record["scored_identity_count"] for record in case_audits)
    output: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": "PhysTwinTAPNextPPMaterialAssociationPostOpenAudit",
        "protocol_id": summary["protocol_id"],
        "information_boundary": {
            "source_outcomes_opened_after_prediction_seals": True,
            "held_v8_accessed": False,
            "diagnostic_only": True,
        },
        "claim_boundary": (
            "Post-open diagnosis on the eight already-open source cases. The audit "
            "cannot tune the sealed arm or authorize independent evaluation."
        ),
        "inputs": {
            "source_summary_result_sha256": summary["result_sha256"],
            "source_summary_sha256": file_sha256(summary_path),
        },
        "case_count": len(case_audits),
        "scored_identity_count": scored_count,
        "exact_match_count": exact_count,
        "exact_match_fraction": exact_count / scored_count if scored_count else None,
        "cases": case_audits,
        "interpretation": (
            "Source-frame nearest-geometry association rarely preserves the fixed "
            "frame-zero material identity used by the benchmark."
        ),
    }
    output["result_sha256"] = canonical_sha256(output)
    output_path = args.output or (source_root / OUTPUT_FILENAME)
    _write_json(output_path, output)
    print(json.dumps({"output": str(output_path), **output}, sort_keys=True))


if __name__ == "__main__":
    main()
