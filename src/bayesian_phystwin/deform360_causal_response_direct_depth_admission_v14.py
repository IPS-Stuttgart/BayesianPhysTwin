"""Hash-only admission custody for the prospective V14 source panel."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .deform360_causal_response_adaptive_query import (
    ARCHIVE_FILENAME,
    REPORT_FILENAME,
    validate_adaptive_causal_response_query_artifacts,
)
from .deform360_causal_response_direct_depth_preflight import (
    load_adaptive_direct_depth_source_preflight_v14,
)
from .deform360_object_exclusion import file_sha256

ADMISSION_PROTOCOL_KIND = (
    "Deform360CausalResponseDirectDepthAdmissionPrelockProtocolV14"
)
ADMISSION_PROTOCOL_CONTRACT = (
    "deform360-causal-response-direct-depth-admission-prelock-v14"
)
ADMISSION_PROTOCOL_ID = (
    "deform360-causal-response-direct-depth-v14-admission-prelock"
)
ADMISSION_ARTIFACT_KIND = "Deform360CausalResponseDirectDepthAdmissionV14"
ADMISSION_CONTRACT = "deform360-causal-response-direct-depth-admission-v14"
ADMISSION_REPORT_FILENAME = "causal_response_direct_depth_admission_v14.json"
PREFLIGHT_FILENAME = "causal_response_direct_depth_preflight_v14.json"
CARRIER_DIRECTORY = "adaptive_carrier"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _valid_digest(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _canonical_sha256(
    payload: Mapping[str, Any],
    *,
    namespace: bytes,
    digest_key: str,
) -> str:
    canonical = dict(payload)
    canonical.pop(digest_key, None)
    return hashlib.sha256(
        namespace
        + json.dumps(
            canonical,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def aggregate_source_sha256(
    role: str,
    values: Mapping[str, str],
) -> str:
    """Hash an ordered set of source-file digests into one role digest."""

    _require(bool(role.strip()), "source aggregate role is empty")
    normalized = dict(sorted((str(key), str(value)) for key, value in values.items()))
    _require(
        normalized and all(_valid_digest(value) for value in normalized.values()),
        "source aggregate contains an invalid digest",
    )
    return hashlib.sha256(
        b"deform360-causal-response-direct-depth-source-role-v14\0"
        + role.encode("utf-8")
        + b"\0"
        + json.dumps(
            normalized,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def load_v14_admission_prelock_protocol(path: str | Path) -> dict[str, Any]:
    """Validate the child lock for carrier and source-preflight execution."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    _require(
        isinstance(payload, dict)
        and payload.get("schema_version") == 1
        and payload.get("artifact_kind") == ADMISSION_PROTOCOL_KIND
        and payload.get("contract") == ADMISSION_PROTOCOL_CONTRACT
        and payload.get("protocol_id") == ADMISSION_PROTOCOL_ID
        and payload.get("status") == "locked_before_first_source_admission",
        "V14 admission pre-lock identity changed",
    )
    _require(
        payload.get("config_sha256")
        == _canonical_sha256(
            payload,
            namespace=(
                b"deform360-causal-response-direct-depth-admission-prelock-v14\0"
            ),
            digest_key="config_sha256",
        ),
        "V14 admission pre-lock checksum changed",
    )
    parents = payload.get("parent_artifacts")
    _require(
        isinstance(parents, Mapping)
        and set(parents)
        == {
            "method_protocol",
            "physical_prelock",
            "physical_runtime_v2",
            "prefix_assets",
            "staging_queue",
        }
        and all(
            isinstance(record, Mapping)
            and _valid_digest(record.get("config_or_queue_sha256"))
            and _valid_digest(record.get("file_sha256"))
            for record in parents.values()
        ),
        "V14 admission pre-lock parent bindings changed",
    )
    numerical = payload.get("numerical_contract")
    _require(
        isinstance(numerical, Mapping)
        and numerical.get("camera_prefix_frame_count") == 58
        and numerical.get("physical_robot_tactile_frame_count") == 76
        and numerical.get("depth_scale_to_m") == 0.001
        and numerical.get("minimum_complete_camera_count") == 8
        and numerical.get("camera_completeness_excludes_projected_support")
        is True,
        "V14 admission numerical contract changed",
    )
    implementation = payload.get("implementation")
    _require(
        isinstance(implementation, Mapping)
        and isinstance(implementation.get("parent_commit"), str)
        and len(implementation["parent_commit"]) == 40
        and set(implementation.get("file_sha256", {}))
        == {"admission_module", "admission_runner", "preflight_module"}
        and all(
            _valid_digest(value)
            for value in implementation["file_sha256"].values()
        ),
        "V14 admission implementation binding changed",
    )
    boundary = payload.get("information_boundary")
    _require(
        isinstance(boundary, Mapping)
        and boundary.get("frame_zero_object_observation_read") is True
        and boundary.get("prefix_object_response_read") is False
        and boundary.get("future_object_observation_read") is False
        and boundary.get("future_identity_or_metric_read") is False
        and boundary.get("source_outcome_read") is False
        and boundary.get("target_object_or_outcome_read") is False
        and boundary.get("held_v8_artifact_or_process_access") is False,
        "V14 admission pre-lock crossed its information boundary",
    )
    return payload


def write_v14_admission_report(
    output_dir: str | Path,
    *,
    queue_rank: int,
    object_hash: str,
    case_hash: str,
    repository_revision: str,
    admission_protocol: Mapping[str, Any],
    physical_artifact_sha256: str,
    geometry_artifact_sha256: str,
    carrier_result_sha256: str,
    carrier_artifact_sha256: str,
    preflight_artifact_sha256: str,
    admitted: bool,
    input_files: Mapping[str, str | Path],
) -> dict[str, Any]:
    """Seal one pre-lock admission disposition without plaintext identity."""

    _require(
        queue_rank >= 1
        and len(repository_revision) == 40
        and all(
            _valid_digest(value)
            for value in (
                object_hash,
                case_hash,
                physical_artifact_sha256,
                geometry_artifact_sha256,
                carrier_result_sha256,
                carrier_artifact_sha256,
                preflight_artifact_sha256,
            )
        ),
        "V14 admission identity or provenance is invalid",
    )
    root = Path(output_dir).resolve()
    report_path = root / ADMISSION_REPORT_FILENAME
    _require(
        root.is_dir() and not report_path.exists(),
        "V14 admission output is missing or already sealed",
    )
    carrier_report, _ = validate_adaptive_causal_response_query_artifacts(
        root / CARRIER_DIRECTORY
    )
    preflight = load_adaptive_direct_depth_source_preflight_v14(
        root / PREFLIGHT_FILENAME
    )
    _require(
        carrier_report["result_sha256"] == carrier_result_sha256
        and carrier_report["schedule"]["artifact_sha256"]
        == carrier_artifact_sha256
        and preflight.artifact_sha256 == preflight_artifact_sha256
        and preflight.object_hash == object_hash
        and preflight.case_hash == case_hash
        and preflight.carrier_artifact_sha256 == carrier_artifact_sha256
        and preflight.admitted is admitted,
        "V14 admission components do not agree",
    )
    report: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": ADMISSION_ARTIFACT_KIND,
        "contract": ADMISSION_CONTRACT,
        "protocol_id": ADMISSION_PROTOCOL_ID,
        "status": "admitted" if admitted else "preflight_rejected",
        "queue_rank": int(queue_rank),
        "object_hash": object_hash,
        "case_hash": case_hash,
        "repository_revision": repository_revision,
        "admission_prelock_config_sha256": admission_protocol["config_sha256"],
        "physical_artifact_sha256": physical_artifact_sha256,
        "geometry_artifact_sha256": geometry_artifact_sha256,
        "carrier_result_sha256": carrier_result_sha256,
        "carrier_artifact_sha256": carrier_artifact_sha256,
        "preflight_artifact_sha256": preflight_artifact_sha256,
        "inputs_sha256": {
            name: file_sha256(path) for name, path in sorted(input_files.items())
        },
        "component_files": {
            "carrier_report": file_sha256(
                root / CARRIER_DIRECTORY / REPORT_FILENAME
            ),
            "carrier_archive": file_sha256(
                root / CARRIER_DIRECTORY / ARCHIVE_FILENAME
            ),
            "source_preflight": file_sha256(root / PREFLIGHT_FILENAME),
        },
        "information_boundary": {
            "object_observation_frames_used_for_admission": [0],
            "known_future_robot_action_read": True,
            "prefix_object_response_read": False,
            "future_object_observation_read": False,
            "future_identity_or_metric_read": False,
            "source_outcome_read": False,
            "source_lock_read": False,
            "plaintext_object_or_episode_identity_retained": False,
            "target_object_or_outcome_read": False,
            "held_v8_artifact_or_process_access": False,
        },
    }
    report["artifact_sha256"] = _canonical_sha256(
        report,
        namespace=b"deform360-causal-response-direct-depth-admission-v14\0",
        digest_key="artifact_sha256",
    )
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    validate_v14_admission_report(root)
    return report


def validate_v14_admission_report(output_dir: str | Path) -> dict[str, Any]:
    """Validate one immutable V14 admission disposition."""

    root = Path(output_dir).resolve()
    report = json.loads(
        (root / ADMISSION_REPORT_FILENAME).read_text(encoding="utf-8")
    )
    _require(
        report.get("artifact_kind") == ADMISSION_ARTIFACT_KIND
        and report.get("contract") == ADMISSION_CONTRACT
        and report.get("protocol_id") == ADMISSION_PROTOCOL_ID
        and report.get("status") in {"admitted", "preflight_rejected"}
        and report.get("artifact_sha256")
        == _canonical_sha256(
            report,
            namespace=b"deform360-causal-response-direct-depth-admission-v14\0",
            digest_key="artifact_sha256",
        ),
        "V14 admission report identity or checksum changed",
    )
    carrier_report, _ = validate_adaptive_causal_response_query_artifacts(
        root / CARRIER_DIRECTORY
    )
    preflight = load_adaptive_direct_depth_source_preflight_v14(
        root / PREFLIGHT_FILENAME
    )
    _require(
        report["carrier_result_sha256"] == carrier_report["result_sha256"]
        and report["carrier_artifact_sha256"]
        == carrier_report["schedule"]["artifact_sha256"]
        and report["preflight_artifact_sha256"] == preflight.artifact_sha256
        and report["object_hash"] == preflight.object_hash
        and report["case_hash"] == preflight.case_hash
        and (report["status"] == "admitted") is preflight.admitted
        and report["component_files"]["source_preflight"]
        == file_sha256(root / PREFLIGHT_FILENAME),
        "V14 admission report differs from its components",
    )
    _require(
        report["component_files"]["carrier_report"]
        == file_sha256(root / CARRIER_DIRECTORY / REPORT_FILENAME)
        and report["component_files"]["carrier_archive"]
        == file_sha256(root / CARRIER_DIRECTORY / ARCHIVE_FILENAME),
        "V14 admission carrier file checksums changed",
    )
    boundary = report["information_boundary"]
    _require(
        boundary["object_observation_frames_used_for_admission"] == [0]
        and boundary["prefix_object_response_read"] is False
        and boundary["future_object_observation_read"] is False
        and boundary["future_identity_or_metric_read"] is False
        and boundary["source_outcome_read"] is False
        and boundary["source_lock_read"] is False
        and boundary["plaintext_object_or_episode_identity_retained"] is False
        and boundary["target_object_or_outcome_read"] is False
        and boundary["held_v8_artifact_or_process_access"] is False,
        "V14 admission report crossed its information boundary",
    )
    return report


__all__ = [
    "ADMISSION_ARTIFACT_KIND",
    "ADMISSION_CONTRACT",
    "ADMISSION_PROTOCOL_CONTRACT",
    "ADMISSION_PROTOCOL_ID",
    "ADMISSION_PROTOCOL_KIND",
    "ADMISSION_REPORT_FILENAME",
    "CARRIER_DIRECTORY",
    "PREFLIGHT_FILENAME",
    "aggregate_source_sha256",
    "load_v14_admission_prelock_protocol",
    "validate_v14_admission_report",
    "write_v14_admission_report",
]
