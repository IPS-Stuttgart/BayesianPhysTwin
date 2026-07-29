"""Checksummed prediction-only artifacts for the V12 source experiment."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from .deform360_causal_response_admission import (
    direct_depth_observation_sha256,
)
from .deform360_causal_response_event import CausalResponseEventScan
from .deform360_causal_response_query import CausalResponseQuerySchedule
from .deform360_causal_response_update import BASELINE_ARM, CANDIDATE_ARM
from .deform360_direct_depth_provider import (
    DirectDepthEndpointConfig,
    DirectDepthEndpointObservations,
)
from .observation_belief import array_sha256, file_sha256

CONTRACT = "deform360-causal-response-prediction-v12"
REPORT_FILENAME = "causal_response_prediction.json"
ARCHIVE_FILENAME = "causal_response_prediction.npz"


def _require(condition: bool | np.bool_, message: str) -> None:
    if not bool(condition):
        raise ValueError(message)


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    canonical = dict(payload)
    canonical.pop("result_sha256", None)
    return hashlib.sha256(
        b"deform360-causal-response-prediction-v12\0"
        + json.dumps(
            canonical,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _same_array_bytes(first: np.ndarray, second: np.ndarray) -> bool:
    left = np.asarray(first)
    right = np.asarray(second)
    return (
        left.dtype == right.dtype
        and left.shape == right.shape
        and left.tobytes(order="C") == right.tobytes(order="C")
    )


def _observation_arrays(
    prefix: str,
    observations: DirectDepthEndpointObservations,
) -> dict[str, np.ndarray]:
    return {
        f"{prefix}__endpoint_frames": observations.endpoint_frames,
        f"{prefix}__entity_ids": observations.entity_ids,
        f"{prefix}__point_world_m": observations.point_world_m,
        f"{prefix}__covariance_m2": observations.covariance_m2,
        f"{prefix}__accepted_support": observations.accepted_support,
        f"{prefix}__association_probability": (observations.association_probability),
        f"{prefix}__support_count": observations.support_count,
        f"{prefix}__maximum_view_scatter_m": (observations.maximum_view_scatter_m),
    }


def _observation_from_arrays(
    arrays: Mapping[str, np.ndarray],
    prefix: str,
    config: Mapping[str, Any],
) -> DirectDepthEndpointObservations:
    return DirectDepthEndpointObservations(
        endpoint_frames=arrays[f"{prefix}__endpoint_frames"],
        entity_ids=arrays[f"{prefix}__entity_ids"],
        point_world_m=arrays[f"{prefix}__point_world_m"],
        covariance_m2=arrays[f"{prefix}__covariance_m2"],
        accepted_support=arrays[f"{prefix}__accepted_support"],
        association_probability=arrays[f"{prefix}__association_probability"],
        support_count=arrays[f"{prefix}__support_count"],
        maximum_view_scatter_m=arrays[f"{prefix}__maximum_view_scatter_m"],
        config=DirectDepthEndpointConfig(**dict(config)),
    )


def _prediction_arrays(
    schedule: CausalResponseQuerySchedule,
    scan: CausalResponseEventScan,
    candidate_arrays: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    arrays = {
        BASELINE_ARM: np.asarray(candidate_arrays[BASELINE_ARM]),
        CANDIDATE_ARM: np.asarray(candidate_arrays[CANDIDATE_ARM]),
        "candidate_correction_variance_m2": np.asarray(
            candidate_arrays["candidate_correction_variance_m2"]
        ),
    }
    arrays.update(
        {f"query__{name}": values for name, values in schedule.arrays().items()}
    )
    if scan.selected_proposal is not None:
        arrays.update(_observation_arrays("proposal", scan.selected_proposal))
    if scan.selected_validation is not None:
        arrays.update(_observation_arrays("validation", scan.selected_validation))
    return {
        name: np.ascontiguousarray(np.asarray(values))
        for name, values in arrays.items()
    }


def write_causal_response_prediction_artifacts(
    output_dir: str | Path,
    schedule: CausalResponseQuerySchedule,
    scan: CausalResponseEventScan,
    candidate_report: Mapping[str, Any],
    candidate_arrays: Mapping[str, np.ndarray],
    *,
    case_id: str,
    repository_revision: str,
    protocol_path: str | Path,
    input_sha256: Mapping[str, str],
) -> dict[str, Any]:
    """Write one prediction seal without any hidden identity or future metric."""

    _require(bool(case_id.strip()), "case ID is empty")
    _require(
        len(repository_revision) == 40
        and all(character in "0123456789abcdef" for character in repository_revision),
        "repository revision is invalid",
    )
    _require(
        scan.case_id == case_id
        and scan.query_artifact_sha256 == schedule.artifact_sha256
        and candidate_report.get("event_scan_sha256") == scan.artifact_sha256
        and candidate_report.get("selected_backbone") == scan.selected_backbone,
        "query, event, and case provenance differ",
    )
    supplied_inputs = dict(sorted(input_sha256.items()))
    _require(
        supplied_inputs
        and all(
            len(digest) == 64
            and all(character in "0123456789abcdef" for character in digest)
            for digest in supplied_inputs.values()
        ),
        "prediction input digests are invalid",
    )
    arrays = _prediction_arrays(schedule, scan, candidate_arrays)
    baseline = arrays[BASELINE_ARM]
    candidate = arrays[CANDIDATE_ARM]
    variance = arrays["candidate_correction_variance_m2"]
    _require(
        baseline.shape == candidate.shape == variance.shape
        and baseline.ndim == 3
        and baseline.shape[2] == 3
        and np.all(np.isfinite(baseline))
        and np.all(np.isfinite(candidate))
        and np.all(np.isfinite(variance))
        and np.all(variance >= 0.0),
        "prediction arrays are invalid",
    )
    applied = bool(candidate_report.get("candidate_applied"))
    if not applied:
        _require(
            _same_array_bytes(candidate, baseline),
            "rejected prediction is not a bit-exact baseline fallback",
        )
    else:
        admission = scan.selected_admission
        _require(admission is not None, "applied candidate lacks an event")
        _require(
            _same_array_bytes(
                candidate[: admission.update_frame + 1],
                baseline[: admission.update_frame + 1],
            ),
            "candidate changed an observed prefix",
        )
    output = Path(output_dir).resolve()
    _require(not output.exists(), "prediction output already exists")
    output.mkdir(parents=True)
    archive_path = output / ARCHIVE_FILENAME
    temporary = archive_path.with_name(archive_path.name + ".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(archive_path)
    selected_admission = scan.selected_admission
    report: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": "Deform360CausalResponsePrediction",
        "contract": CONTRACT,
        "case": case_id,
        "status": (
            "candidate_prediction_sealed"
            if applied
            else "exact_baseline_fallback_sealed"
        ),
        "repository_revision": repository_revision,
        "protocol": {
            "path": str(Path(protocol_path)),
            "file_sha256": file_sha256(protocol_path),
        },
        "query": schedule.descriptor(),
        "event_scan": scan.descriptor(),
        "admission_attempts": [attempt.descriptor() for attempt in scan.attempts],
        "selected_depth_config": (
            None
            if scan.selected_proposal is None
            else asdict(scan.selected_proposal.config)
        ),
        "candidate": dict(candidate_report),
        "inputs_sha256": supplied_inputs,
        "archive": {
            "filename": ARCHIVE_FILENAME,
            "file_sha256": file_sha256(archive_path),
            "array_sha256": {
                name: array_sha256(values) for name, values in sorted(arrays.items())
            },
        },
        "information_boundary": {
            "maximum_object_observation_frame": (scan.maximum_observation_frame),
            "future_object_observation_read": False,
            "future_identity_or_metric_read": False,
            "validation_panel_formed_update": False,
            "query_abstention_or_event_rejection_is_exact_fallback": True,
            "prediction_sealed_before_outcome_authorization": True,
            "held_v8_artifact_or_process_access": False,
        },
        "selected_admission_sha256": (
            None if selected_admission is None else selected_admission.artifact_sha256
        ),
    }
    report["result_sha256"] = _canonical_sha256(report)
    (output / REPORT_FILENAME).write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    validate_causal_response_prediction_artifacts(output)
    return report


def validate_causal_response_prediction_artifacts(
    output_dir: str | Path,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Validate checksums, custody boundaries, and exact fallback semantics."""

    output = Path(output_dir).resolve()
    report = json.loads((output / REPORT_FILENAME).read_text(encoding="utf-8"))
    _require(
        report.get("artifact_kind") == "Deform360CausalResponsePrediction"
        and report.get("contract") == CONTRACT
        and report.get("status")
        in {
            "candidate_prediction_sealed",
            "exact_baseline_fallback_sealed",
        }
        and report.get("result_sha256") == _canonical_sha256(report),
        "prediction report is invalid",
    )
    boundary = report["information_boundary"]
    _require(
        boundary.get("future_object_observation_read") is False
        and boundary.get("future_identity_or_metric_read") is False
        and boundary.get("validation_panel_formed_update") is False
        and boundary.get("prediction_sealed_before_outcome_authorization") is True
        and boundary.get("held_v8_artifact_or_process_access") is False,
        "prediction crossed its information boundary",
    )
    archive_path = output / ARCHIVE_FILENAME
    _require(
        report["archive"]["file_sha256"] == file_sha256(archive_path),
        "prediction archive checksum changed",
    )
    with np.load(archive_path, allow_pickle=False) as stored:
        arrays = {name: np.asarray(stored[name]) for name in stored.files}
    observed_hashes = {
        name: array_sha256(values) for name, values in sorted(arrays.items())
    }
    _require(
        observed_hashes == report["archive"]["array_sha256"],
        "prediction array checksum changed",
    )
    baseline = arrays[BASELINE_ARM]
    candidate = arrays[CANDIDATE_ARM]
    applied = bool(report["candidate"].get("candidate_applied"))
    _require(
        applied == (report["status"] == "candidate_prediction_sealed")
        and report["candidate"].get("event_scan_sha256")
        == report["event_scan"]["artifact_sha256"]
        and report["candidate"].get("selected_backbone")
        == report["event_scan"]["selected_backbone"],
        "candidate status and report differ",
    )
    if not applied:
        _require(
            _same_array_bytes(candidate, baseline),
            "fallback candidate differs from the selected baseline",
        )
    selected_value = report["event_scan"]["selected_attempt_index"]
    if selected_value is not None:
        selected = int(selected_value)
        admission = report["admission_attempts"][selected]
        update = int(admission["update_frame"])
        depth_config = report["selected_depth_config"]
        proposal = _observation_from_arrays(
            arrays,
            "proposal",
            depth_config,
        )
        validation = _observation_from_arrays(
            arrays,
            "validation",
            depth_config,
        )
        _require(
            direct_depth_observation_sha256(proposal)
            == admission["proposal_observation_sha256"]
            and direct_depth_observation_sha256(validation)
            == admission["validation_observation_sha256"],
            "selected direct-depth evidence changed",
        )
        _require(
            array_sha256(np.asarray(baseline[: update + 1], dtype=np.float64))
            == admission["physical_prefix_sha256"],
            "selected baseline prefix differs from the admission",
        )
        selected_backbone = report["event_scan"]["selected_backbone"]
        expected_action_hash = report["event_scan"]["physical_candidate_prefix_sha256"]
        _require(
            admission["action_conditioning_prefix_sha256"] == expected_action_hash,
            "physical action-conditioning prefix differs from the admission",
        )
        expected_selected_hash = (
            expected_action_hash
            if selected_backbone == "physical"
            else report["event_scan"]["persistence_candidate_prefix_sha256"]
        )
        _require(
            admission["physical_prefix_sha256"] == expected_selected_hash,
            "selected backbone hash differs from the admission",
        )
    if applied:
        _require(selected_value is not None, "applied candidate lacks an event")
        selected = int(report["event_scan"]["selected_attempt_index"])
        admission = report["admission_attempts"][selected]
        update = int(admission["update_frame"])
        _require(
            _same_array_bytes(
                candidate[: update + 1],
                baseline[: update + 1],
            ),
            "candidate changed its observed prefix",
        )
    query = report["query"]
    query_hashes = {
        name.removeprefix("query__"): array_sha256(values)
        for name, values in arrays.items()
        if name.startswith("query__")
    }
    _require(
        query_hashes == query["output_array_sha256"],
        "query arrays differ from the query descriptor",
    )
    return report, arrays


__all__ = [
    "ARCHIVE_FILENAME",
    "CONTRACT",
    "REPORT_FILENAME",
    "validate_causal_response_prediction_artifacts",
    "write_causal_response_prediction_artifacts",
]
