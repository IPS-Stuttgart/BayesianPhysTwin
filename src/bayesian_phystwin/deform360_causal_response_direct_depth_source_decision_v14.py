"""Outcome-blind V14 prediction barrier and early source-gate decision."""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .deform360_causal_response_direct_depth import (
    ARCHIVE_FILENAME,
    REPORT_FILENAME,
    validate_adaptive_direct_depth_v14_artifacts,
)
from .deform360_causal_response_direct_depth_reserve_prediction_v14 import (
    load_v14_reserve_prediction_runtime,
)
from .deform360_causal_response_direct_depth_source_lock import (
    validate_adaptive_direct_depth_source_lock_v14,
)
from .deform360_causal_response_direct_depth_spatial_support_runtime_v2 import (
    load_v14_sparse_spatial_support_runtime_v2,
)
from .deform360_object_exclusion import file_sha256

ARTIFACT_KIND = "Deform360CausalResponseDirectDepthSourceDecisionV14"
CONTRACT = "deform360-causal-response-direct-depth-source-decision-v14"
NAMESPACE = b"deform360-causal-response-direct-depth-source-decision-v14\0"
DISPOSITION_FILENAME = "prediction_disposition_v14.json"
DISPOSITION_NAMESPACE = (
    b"deform360-causal-response-direct-depth-prediction-disposition-v14\0"
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _canonical_sha256(payload: Mapping[str, Any], *, key: str) -> str:
    canonical = dict(payload)
    canonical.pop(key, None)
    return hashlib.sha256(
        NAMESPACE
        + json.dumps(
            canonical,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _canonical_disposition_sha256(payload: Mapping[str, Any]) -> str:
    canonical = dict(payload)
    canonical.pop("artifact_sha256", None)
    return hashlib.sha256(
        DISPOSITION_NAMESPACE
        + json.dumps(
            canonical,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _read_json(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read V14 source-decision input: {source}") from error
    _require(isinstance(payload, dict), f"JSON artifact is not an object: {source}")
    return payload


def _git_output(repository: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def evaluate_v14_outcome_blind_gates(
    dispositions: Sequence[Mapping[str, Any]],
    source_gate: Mapping[str, Any],
) -> dict[str, Any]:
    """Evaluate only gates that do not require a future source outcome."""

    rows = [dict(row) for row in dispositions]
    required = int(source_gate["required_prediction_or_exact_fallback_count"])
    maximum_failures = int(source_gate["maximum_technical_failure_count"])
    minimum_events = int(source_gate["minimum_event_admitted_object_count"])
    sealed_count = len(rows)
    technical_failures = max(0, required - sealed_count)
    event_count = sum(bool(row["event_admitted"]) for row in rows)
    candidate_count = sum(bool(row["candidate_applied"]) for row in rows)
    completeness_passed = (
        sealed_count == required and technical_failures <= maximum_failures
    )
    event_gate_passed = event_count >= minimum_events
    outcome_authorized = completeness_passed and event_gate_passed
    return {
        "required_prediction_or_exact_fallback_count": required,
        "sealed_prediction_or_exact_fallback_count": sealed_count,
        "maximum_technical_failure_count": maximum_failures,
        "technical_failure_count": technical_failures,
        "minimum_event_admitted_object_count": minimum_events,
        "event_admitted_object_count": event_count,
        "candidate_applied_object_count": candidate_count,
        "gates": {
            "prediction_completeness": completeness_passed,
            "event_admitted_object_count": event_gate_passed,
            "outcome_dependent_accuracy_safety_and_calibration": ("not_evaluated"),
        },
        "source_outcome_authorized": outcome_authorized,
        "source_gate_status": (
            "requires_source_outcome_evaluation"
            if outcome_authorized
            else "failed_before_source_outcome_authorization"
        ),
        "decision": (
            "proceed_to_registered_source_outcome_evaluation"
            if outcome_authorized
            else "close_v14_without_source_outcome_reveal"
        ),
    }


def finalize_v14_source_prediction_decision(
    *,
    repository: str | Path,
    method_protocol_path: str | Path,
    source_lock_path: str | Path,
    prediction_runtime_path: str | Path,
    admission_custody_path: str | Path,
    physical_custody_path: str | Path,
    spatial_support_runtime_path: str | Path,
    prediction_root: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Validate every V14 seal and stop early when a target-free gate fails."""

    repo = Path(repository).resolve()
    revision = _git_output(repo, "rev-parse", "HEAD")
    _require(
        not _git_output(
            repo,
            "status",
            "--porcelain",
            "--untracked-files=normal",
        ),
        "V14 source-decision repository is dirty",
    )
    method_path = Path(method_protocol_path).resolve()
    source_path = Path(source_lock_path).resolve()
    runtime_path = Path(prediction_runtime_path).resolve()
    admission_path = Path(admission_custody_path).resolve()
    physical_path = Path(physical_custody_path).resolve()
    amendment_path = Path(spatial_support_runtime_path).resolve()
    root = Path(prediction_root).resolve()
    output = Path(output_path).resolve()
    _require(not output.exists(), "V14 source-decision output already exists")

    method = _read_json(method_path)
    source_lock = validate_adaptive_direct_depth_source_lock_v14(source_path)
    runtime = load_v14_reserve_prediction_runtime(
        runtime_path,
        repository=repo,
        method_protocol_path=method_path,
        source_lock_path=source_path,
        admission_custody_path=admission_path,
        physical_custody_path=physical_path,
    )
    amendment = load_v14_sparse_spatial_support_runtime_v2(
        amendment_path,
        repository=repo,
        method_protocol_path=method_path,
        source_lock_path=source_path,
    )
    _require(
        _git_output(
            repo,
            "merge-base",
            "--is-ancestor",
            runtime["implementation"]["parent_commit"],
            revision,
        )
        == "",
        "V14 prediction revision is not an ancestor of the source decision",
    )

    locked_by_case = {case.case_hash: case for case in source_lock.cases}
    runtime_by_rank = {
        int(record["queue_rank"]): dict(record) for record in runtime["cases"]
    }
    _require(
        len(locked_by_case) == len(runtime_by_rank) == 12,
        "V14 source-decision parent cardinality changed",
    )
    observed_directories = {
        directory.name for directory in root.glob("rank-*") if directory.is_dir()
    }
    expected_directories = {f"rank-{rank:03d}" for rank in runtime_by_rank}
    _require(
        observed_directories == expected_directories,
        "V14 prediction root differs from the locked runtime ranks",
    )

    rows: list[dict[str, Any]] = []
    for rank, runtime_case in sorted(runtime_by_rank.items()):
        case = locked_by_case[runtime_case["case_hash"]]
        case_root = root / f"rank-{rank:03d}"
        disposition_path = case_root / DISPOSITION_FILENAME
        prediction_dir = case_root / "prediction"
        disposition = _read_json(disposition_path)
        report, arrays = validate_adaptive_direct_depth_v14_artifacts(prediction_dir)
        boundary = disposition.get("information_boundary")
        event_scan = report["adaptive_scan"]["event_scan"]
        _require(
            disposition.get("schema_version") == 1
            and disposition.get("artifact_kind")
            == "Deform360CausalResponseDirectDepthPredictionDispositionV14"
            and disposition.get("contract")
            == "deform360-causal-response-direct-depth-prediction-disposition-v14"
            and disposition.get("artifact_sha256")
            == _canonical_disposition_sha256(disposition),
            f"V14 prediction disposition is invalid at rank {rank}",
        )
        _require(
            disposition.get("queue_rank") == rank
            and disposition.get("case_hash") == case.case_hash
            and disposition.get("object_hash") == case.object_hash
            and disposition.get("repository_revision")
            == runtime["implementation"]["parent_commit"]
            and disposition.get("source_lock_artifact_sha256")
            == source_lock.artifact_sha256
            and disposition.get("prediction_runtime_config_sha256")
            == runtime["config_sha256"]
            and disposition.get("admission_artifact_sha256")
            == runtime_case["admission_artifact_sha256"]
            and disposition.get("physical_artifact_sha256")
            == runtime_case["physical_artifact_sha256"],
            f"V14 prediction custody changed at rank {rank}",
        )
        _require(
            disposition.get("status") == report["status"]
            and disposition.get("prediction_result_sha256") == report["result_sha256"]
            and disposition.get("event_admitted") is bool(event_scan["admitted"])
            and disposition.get("selected_backbone") == event_scan["selected_backbone"]
            and disposition.get("candidate_applied")
            is bool(report["candidate"]["candidate_applied"])
            and report["inputs_sha256"]["prediction_runtime"]
            == file_sha256(runtime_path)
            and report["inputs_sha256"]["source_lock"] == file_sha256(source_path)
            and report["protocol"]["file_sha256"] == file_sha256(method_path),
            f"V14 prediction result differs from its disposition at rank {rank}",
        )
        _require(
            isinstance(boundary, Mapping)
            and int(boundary.get("maximum_object_observation_frame", 10**9))
            <= int(method["information_boundary"]["maximum_object_observation_frame"])
            and boundary.get("future_object_observation_read") is False
            and boundary.get("future_identity_or_metric_read") is False
            and boundary.get("source_outcome_read") is False
            and boundary.get("target_object_or_outcome_read") is False
            and boundary.get("held_v8_artifact_or_process_access") is False,
            f"V14 prediction crossed its boundary at rank {rank}",
        )
        rows.append(
            {
                "queue_rank": rank,
                "case_hash": case.case_hash,
                "object_hash": case.object_hash,
                "status": report["status"],
                "event_admitted": bool(disposition["event_admitted"]),
                "selected_backbone": str(disposition["selected_backbone"]),
                "candidate_applied": bool(disposition["candidate_applied"]),
                "disposition_artifact_sha256": disposition["artifact_sha256"],
                "disposition_file_sha256": file_sha256(disposition_path),
                "prediction_result_sha256": report["result_sha256"],
                "prediction_report_file_sha256": file_sha256(
                    prediction_dir / REPORT_FILENAME
                ),
                "prediction_archive_file_sha256": file_sha256(
                    prediction_dir / ARCHIVE_FILENAME
                ),
                "prediction_array_names": sorted(arrays),
            }
        )

    early = evaluate_v14_outcome_blind_gates(rows, method["source_gate"])
    payload: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": ARTIFACT_KIND,
        "contract": CONTRACT,
        "protocol_id": method["protocol_id"],
        "status": early["source_gate_status"],
        "decision": early["decision"],
        "repository_revision": revision,
        "prediction_repository_revision": runtime["implementation"]["parent_commit"],
        "parent_artifacts": {
            "method_protocol": {
                "semantic_sha256": method["config_sha256"],
                "file_sha256": file_sha256(method_path),
            },
            "source_lock": {
                "semantic_sha256": source_lock.artifact_sha256,
                "file_sha256": file_sha256(source_path),
            },
            "prediction_runtime": {
                "semantic_sha256": runtime["config_sha256"],
                "file_sha256": file_sha256(runtime_path),
            },
            "spatial_support_runtime_v2": {
                "semantic_sha256": amendment["config_sha256"],
                "file_sha256": file_sha256(amendment_path),
            },
        },
        "outcome_blind_gate": early,
        "predictions": rows,
        "information_boundary": {
            "all_predictions_validated_before_decision": True,
            "maximum_object_observation_frame": int(
                method["information_boundary"]["maximum_object_observation_frame"]
            ),
            "future_object_observation_read": False,
            "future_identity_or_metric_read": False,
            "source_outcome_read": False,
            "target_object_or_outcome_read": False,
            "held_v8_artifact_or_process_access": False,
        },
        "claim_boundary": (
            "Prospective fresh-source prediction completeness and target-free "
            "event-admission decision only; no future source outcome, target "
            "evaluation, confirmation, or state-of-the-art claim."
        ),
    }
    payload["artifact_sha256"] = _canonical_sha256(
        payload,
        key="artifact_sha256",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return payload


__all__ = [
    "ARTIFACT_KIND",
    "CONTRACT",
    "DISPOSITION_FILENAME",
    "NAMESPACE",
    "evaluate_v14_outcome_blind_gates",
    "finalize_v14_source_prediction_decision",
]
