#!/usr/bin/env python3
"""Stage the frozen TAPNext++ depth-completion source-transfer panel."""

from __future__ import annotations

import argparse
import json
import pickle
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from bayesian_phystwin.phystwin_tapnextpp_competence import (
    PROTOCOL_ID as TRACKER_PROTOCOL_ID,
)
from bayesian_phystwin.phystwin_tapnextpp_competence import (
    SOURCE_REPORT_FILENAME,
    canonical_sha256,
    file_sha256,
    prepare_source_artifacts,
)
from bayesian_phystwin.tapnextpp_transfer_staging import (
    TRANSFER_PROTOCOL_ID,
    plan_transfer_case,
    validate_transfer_protocol,
)

MANIFEST_FILENAME = "tapnextpp_depth_completion_transfer_source_manifest.json"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _load_pickle_array(path: Path) -> np.ndarray:
    with path.open("rb") as stream:
        return np.asarray(pickle.load(stream), dtype=np.float64)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort-protocol", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--physical-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def stage_transfer_panel(
    cohort_protocol_path: str | Path,
    raw_root: str | Path,
    physical_root: str | Path,
    output_root: str | Path,
) -> dict[str, Any]:
    """Stage every fixed case and retain technical failures without replacement."""

    protocol_path = Path(cohort_protocol_path).resolve()
    raw = Path(raw_root).resolve()
    physical = Path(physical_root).resolve()
    output = Path(output_root).resolve()
    _require(not output.exists(), "transfer staging output already exists")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    validate_transfer_protocol(protocol)
    output.mkdir(parents=True)
    records: list[dict[str, Any]] = []

    for case_name in protocol["fixed_source_cases"]:
        case_root = output / "cases" / case_name
        case_root.mkdir(parents=True)
        raw_case = raw / case_name
        manual_path = raw_case / "gt_track_3d.pkl"
        split_path = raw_case / "split.json"
        masks_path = raw_case / "mask" / "processed_masks.pkl"
        physical_path = physical / case_name / "inference.pkl"
        record: dict[str, Any] = {
            "case": case_name,
            "status": "technical-staging-failure",
            "replacement_permitted": False,
        }
        try:
            for path in (manual_path, split_path, masks_path, physical_path):
                _require(path.is_file(), f"required source file is missing: {path}")
            split = json.loads(split_path.read_text(encoding="utf-8"))
            train_end = int(split["train"][1])
            manual_tracks = _load_pickle_array(manual_path)
            physical_trajectory = _load_pickle_array(physical_path)
            plan = plan_transfer_case(
                case_name,
                physical_trajectory,
                manual_tracks,
                train_end_frame_exclusive=train_end,
                protocol=protocol,
            )
            source_artifact_root = case_root / "source_artifacts"
            source_report = prepare_source_artifacts(
                manual_path,
                split_path,
                masks_path,
                source_artifact_root,
                config=plan.tracker_config,
            )
            source_report_path = source_artifact_root / SOURCE_REPORT_FILENAME
            per_case_protocol: dict[str, Any] = {
                "schema_version": 1,
                "protocol_id": TRACKER_PROTOCOL_ID,
                "status": "locked-before-tapnextpp-prediction",
                "source_panel_protocol_id": TRANSFER_PROTOCOL_ID,
                "source_panel_protocol_sha256": file_sha256(protocol_path),
                "case": case_name,
                "source_frame_start": plan.physical_window.start_frame,
                "prefix_frame_count": plan.tracker_config.prefix_frame_count,
                "method_config": asdict(plan.tracker_config),
                "depth_completion_config": protocol[
                    "depth_completion_config"
                ],
                "depth_completion_gates": protocol["per_case_gates"],
                "source_artifacts": {
                    "source_report_sha256": file_sha256(source_report_path),
                    "prediction_input_sha256": source_report[
                        "prediction_input"
                    ]["sha256"],
                    "withheld_prefix_sha256": source_report[
                        "withheld_evaluation"
                    ]["sha256"],
                },
                "selection": {
                    "physical_trajectory_sha256": file_sha256(physical_path),
                    "physical_rms_endpoint_displacement_m": (
                        plan.physical_window.rms_endpoint_displacement_m
                    ),
                    "sampled_node_indices_sha256": canonical_sha256(
                        {
                            "indices": plan.physical_window.sampled_node_indices.tolist()
                        }
                    ),
                    "selected_identity_ids": list(plan.selected_identity_ids),
                    "selection_used_withheld_prefix_or_future_target": False,
                },
                "claim_boundary": protocol["claim_boundary"],
            }
            per_case_protocol["result_sha256"] = canonical_sha256(
                per_case_protocol
            )
            case_protocol_path = case_root / "tracker_protocol.json"
            _write_json(case_protocol_path, per_case_protocol)
            record.update(
                {
                    "status": "prediction-ready",
                    "source_frame_start": plan.physical_window.start_frame,
                    "source_frame_end_exclusive": (
                        plan.physical_window.end_frame_exclusive
                    ),
                    "selected_identity_ids": list(plan.selected_identity_ids),
                    "tracker_protocol_path": str(case_protocol_path),
                    "tracker_protocol_sha256": file_sha256(case_protocol_path),
                    "prediction_input_path": source_report["prediction_input"][
                        "path"
                    ],
                    "prediction_input_sha256": source_report[
                        "prediction_input"
                    ]["sha256"],
                    "withheld_prefix_path": source_report[
                        "withheld_evaluation"
                    ]["path"],
                    "withheld_prefix_sha256": source_report[
                        "withheld_evaluation"
                    ]["sha256"],
                }
            )
        except Exception as error:  # retain fixed-case technical dispositions
            record["error_type"] = type(error).__name__
            record["error_message"] = str(error)
        records.append(record)

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": "PhysTwinTAPNextPPDepthCompletionTransferSourceManifest",
        "protocol_id": TRANSFER_PROTOCOL_ID,
        "protocol_sha256": file_sha256(protocol_path),
        "fixed_case_count": len(protocol["fixed_source_cases"]),
        "prediction_ready_count": sum(
            record["status"] == "prediction-ready" for record in records
        ),
        "technical_staging_failure_count": sum(
            record["status"] == "technical-staging-failure"
            for record in records
        ),
        "case_records": records,
        "information_boundary": {
            "withheld_prefix_staged_separately": True,
            "withheld_prefix_scored": False,
            "tracker_prediction_run": False,
            "future_simulator_outcome_read": False,
            "held_v8_accessed": False,
            "failed_case_replacement_permitted": False,
        },
        "claim_boundary": protocol["claim_boundary"],
    }
    manifest["result_sha256"] = canonical_sha256(manifest)
    _write_json(output / MANIFEST_FILENAME, manifest)
    return manifest


def main() -> int:
    args = _parse_args()
    result = stage_transfer_panel(
        args.cohort_protocol,
        args.raw_root,
        args.physical_root,
        args.output_root,
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
