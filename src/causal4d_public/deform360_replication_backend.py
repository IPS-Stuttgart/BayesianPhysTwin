"""Pre-target source-backend decision for the Deform360 replication."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .deform360_replication import validate_deform360_replication_protocol
from .deform360_replication_fit import validate_pooled_source_warp_fit


SOURCE_BACKEND_DECISION_SCHEMA_VERSION = 2
SOURCE_STAGE_FAILURE_SCHEMA_VERSION = 1
BACKEND_POLICY_SCHEMA_VERSION = 1
CANONICAL_BACKEND_POLICY_SHA256 = (
    "96e9a99d4b1052e97e53c28281af0af45b4c5cd3fee9b07dac06b722b460d478"
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _artifact_sha256(payload: Mapping[str, Any]) -> str:
    canonical = dict(payload)
    canonical.pop("result_sha256", None)
    return hashlib.sha256(_canonical_bytes(canonical)).hexdigest()


def backend_policy_sha256(payload: Mapping[str, Any]) -> str:
    canonical = dict(payload)
    canonical.pop("config_sha256", None)
    return hashlib.sha256(_canonical_bytes(canonical)).hexdigest()


def validate_backend_policy(payload: Mapping[str, Any]) -> dict[str, Any]:
    _require(
        payload.get("schema_version") == BACKEND_POLICY_SCHEMA_VERSION,
        "backend-policy schema changed",
    )
    observed = backend_policy_sha256(payload)
    _require(payload.get("config_sha256") == observed, "backend-policy checksum mismatch")
    _require(observed == CANONICAL_BACKEND_POLICY_SHA256, "backend policy differs from lock")
    config = payload["config"]
    _require(
        config["status"] == "locked-before-target-prefix-access",
        "backend policy is not pre-target locked",
    )
    geometry = config["geometry"]
    _require(
        geometry["minimum_consensus_votes"] == 8
        and geometry["consensus_relaxation_allowed"] is False
        and geometry["target_preseal_geometry"] == "prefix endpoint only",
        "backend geometry policy changed",
    )
    warp = config["warp"]
    _require(
        warp["candidate_count"] == 200
        and warp["substeps_per_video_frame"] == 128
        and warp["candidate_quality"]["maximum_p99_relative_edge_strain"]
        == 0.5,
        "backend Warp policy changed",
    )
    admission = config["backend_admission"]
    _require(
        admission["unit"] == "object"
        and admission["full_six_object_target_phase_requires_every_object_to_pass"]
        is True,
        "backend admission policy changed",
    )
    return {
        "passed": True,
        "policy_id": config["policy_id"],
        "config_sha256": observed,
    }


def load_backend_policy(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_backend_policy(payload)
    return payload


def build_source_stage_failure_artifact(
    protocol: Mapping[str, Any],
    backend_policy: Mapping[str, Any],
    *,
    object_id: str,
    stage: str,
    failed_episode_id: str | None,
    error_type: str,
    error_message: str,
    episode_status: Sequence[Mapping[str, Any]],
    evidence: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Record a locked source-stage failure without creating a fallback result."""

    validate_deform360_replication_protocol(protocol)
    validate_backend_policy(backend_policy)
    cohort = protocol["config"]["cohort"]
    matches = [row for row in cohort if row["object_id"] == object_id]
    _require(len(matches) == 1, "source-stage failure object is outside the cohort")
    expected_episode_ids = [
        f"{object_id}/episode_{int(index):04d}"
        for index in matches[0]["source_episode_ids"]
    ]
    statuses = [dict(row) for row in episode_status]
    _require(
        [row.get("episode_id") for row in statuses] == expected_episode_ids,
        "source-stage failure episode status changed",
    )
    allowed_status = {"completed", "failed", "not-attempted"}
    _require(
        all(row.get("status") in allowed_status for row in statuses),
        "source-stage failure has an unknown episode status",
    )
    failed_rows = [row for row in statuses if row["status"] == "failed"]
    _require(
        stage in {"source-geometry", "source-grid", "source-pooling"},
        "unknown source stage",
    )
    if stage == "source-pooling":
        _require(
            failed_episode_id is None and not failed_rows,
            "source-pooling failure cannot name one failed episode",
        )
        _require(
            all(row["status"] == "completed" for row in statuses),
            "source pooling requires every source grid",
        )
    else:
        _require(len(failed_rows) == 1, "exactly one source episode must be failed")
        _require(
            failed_rows[0]["episode_id"] == failed_episode_id,
            "failed source episode changed",
        )
    _require(bool(error_type) and bool(error_message), "source failure is underspecified")
    evidence_rows = [dict(row) for row in evidence]
    _require(bool(evidence_rows), "source-stage failure has no evidence")
    for row in evidence_rows:
        _require(bool(row.get("path")), "failure evidence path is missing")
        _require(
            isinstance(row.get("size_bytes"), int) and row["size_bytes"] >= 0,
            "failure evidence size is invalid",
        )
        digest = row.get("sha256", "")
        _require(
            isinstance(digest, str)
            and len(digest) == 64
            and all(char in "0123456789abcdef" for char in digest),
            "failure evidence checksum is invalid",
        )
    payload: dict[str, Any] = {
        "schema_version": SOURCE_STAGE_FAILURE_SCHEMA_VERSION,
        "artifact_kind": "Deform360ReplicationSourceStageFailure",
        "protocol_config_sha256": protocol["config_sha256"],
        "backend_policy_config_sha256": backend_policy["config_sha256"],
        "object_id": object_id,
        "stratum": matches[0]["stratum"],
        "failed_stage": stage,
        "failed_episode_id": failed_episode_id,
        "error_type": error_type,
        "error_message": error_message,
        "episode_status": statuses,
        "evidence": evidence_rows,
        "decision": {
            "competence_passed": False,
            "fallback_used": False,
            "target_prefix_access_permitted": False,
        },
        "information_boundary": {
            "source_geometry_and_tactile_read": True,
            "calibration_outcomes_read": False,
            "target_prefix_read": False,
            "target_future_geometry_read": False,
            "target_future_tactile_read": False,
        },
    }
    payload["result_sha256"] = _artifact_sha256(payload)
    return payload


def validate_source_stage_failure_artifact(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    _require(
        payload.get("schema_version") == SOURCE_STAGE_FAILURE_SCHEMA_VERSION,
        "source-stage failure schema changed",
    )
    _require(
        payload.get("artifact_kind") == "Deform360ReplicationSourceStageFailure",
        "source-stage failure kind changed",
    )
    _require(
        payload.get("result_sha256") == _artifact_sha256(payload),
        "source-stage failure checksum mismatch",
    )
    statuses = payload.get("episode_status", [])
    failed_rows = [row for row in statuses if row.get("status") == "failed"]
    _require(
        payload.get("failed_stage")
        in {"source-geometry", "source-grid", "source-pooling"},
        "source-stage failure has an unknown stage",
    )
    if payload["failed_stage"] == "source-pooling":
        _require(
            payload.get("failed_episode_id") is None
            and not failed_rows
            and all(row.get("status") == "completed" for row in statuses),
            "source-pooling failure status is inconsistent",
        )
    else:
        _require(
            len(failed_rows) == 1
            and failed_rows[0]["episode_id"] == payload.get("failed_episode_id"),
            "source-stage failed episode is inconsistent",
        )
    _require(
        bool(payload.get("error_type")) and bool(payload.get("error_message")),
        "source-stage failure is underspecified",
    )
    evidence = payload.get("evidence", [])
    _require(bool(evidence), "source-stage failure has no evidence")
    for row in evidence:
        _require(bool(row.get("path")), "failure evidence path is missing")
        _require(
            isinstance(row.get("size_bytes"), int) and row["size_bytes"] >= 0,
            "failure evidence size is invalid",
        )
        digest = row.get("sha256", "")
        _require(
            isinstance(digest, str)
            and len(digest) == 64
            and all(char in "0123456789abcdef" for char in digest),
            "failure evidence checksum is invalid",
        )
    decision = payload.get("decision", {})
    _require(
        decision.get("competence_passed") is False
        and decision.get("fallback_used") is False
        and decision.get("target_prefix_access_permitted") is False,
        "source-stage failure is not fail closed",
    )
    boundary = payload.get("information_boundary", {})
    _require(
        boundary.get("calibration_outcomes_read") is False
        and boundary.get("target_prefix_read") is False
        and boundary.get("target_future_geometry_read") is False
        and boundary.get("target_future_tactile_read") is False,
        "source-stage failure crossed its information boundary",
    )
    return {
        "passed": True,
        "object_id": payload["object_id"],
        "failed_stage": payload["failed_stage"],
        "result_sha256": payload["result_sha256"],
    }


def build_source_backend_decision_artifact(
    protocol: Mapping[str, Any],
    pooled_fits: Sequence[Mapping[str, Any]],
    backend_policy: Mapping[str, Any],
    source_stage_failures: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Apply the locked source gate without reading any target observation."""

    validate_deform360_replication_protocol(protocol)
    validate_backend_policy(backend_policy)
    _require(
        backend_policy["config"]["replication_protocol_config_sha256"]
        == protocol["config_sha256"],
        "backend policy belongs to another replication protocol",
    )
    by_object: dict[str, Mapping[str, Any]] = {}
    for fit in pooled_fits:
        validate_pooled_source_warp_fit(fit)
        object_id = str(fit["object_id"])
        _require(object_id not in by_object, "pooled object fit is repeated")
        by_object[object_id] = fit
    failures_by_object: dict[str, Mapping[str, Any]] = {}
    for failure in source_stage_failures:
        validate_source_stage_failure_artifact(failure)
        _require(
            failure["protocol_config_sha256"] == protocol["config_sha256"],
            "source failure belongs to another protocol",
        )
        _require(
            failure["backend_policy_config_sha256"]
            == backend_policy["config_sha256"],
            "source failure belongs to another backend policy",
        )
        object_id = str(failure["object_id"])
        _require(object_id not in failures_by_object, "source failure is repeated")
        _require(object_id not in by_object, "object has both a fit and a failure")
        failures_by_object[object_id] = failure
    cohort = protocol["config"]["cohort"]
    expected_objects = [record["object_id"] for record in cohort]
    _require(
        set(by_object) | set(failures_by_object) == set(expected_objects),
        "source outcome cohort changed",
    )
    records = []
    pooled_means = []
    persistence_means = []
    loo_wins = []
    for cohort_record in cohort:
        object_id = cohort_record["object_id"]
        if object_id in failures_by_object:
            failure = failures_by_object[object_id]
            records.append(
                {
                    "object_id": object_id,
                    "stratum": cohort_record["stratum"],
                    "outcome_kind": "source-stage-failure",
                    "source_stage_failure_result_sha256": failure[
                        "result_sha256"
                    ],
                    "failed_stage": failure["failed_stage"],
                    "failed_episode_id": failure["failed_episode_id"],
                    "error_type": failure["error_type"],
                    "error_message": failure["error_message"],
                    "competence_passed": False,
                }
            )
            continue
        fit = by_object[object_id]
        expected_episode_ids = [
            f"{object_id}/episode_{int(index):04d}"
            for index in cohort_record["source_episode_ids"]
        ]
        _require(
            fit["source_episode_ids"] == expected_episode_ids,
            f"{object_id} source episodes changed",
        )
        competence = fit["source_backend_competence"]
        pooled_means.append(float(fit["pooled_source_mean_chamfer_m"]))
        persistence_means.append(float(fit["persistence_source_mean_chamfer_m"]))
        loo_wins.extend(bool(row["win"]) for row in fit["leave_one_source"])
        records.append(
            {
                "object_id": object_id,
                "stratum": cohort_record["stratum"],
                "outcome_kind": "pooled-source-fit",
                "pooled_fit_result_sha256": fit["result_sha256"],
                "pooled_candidate_index": fit["selection"][
                    "pooled_candidate_index"
                ],
                "pooled_source_mean_chamfer_m": fit[
                    "pooled_source_mean_chamfer_m"
                ],
                "persistence_source_mean_chamfer_m": fit[
                    "persistence_source_mean_chamfer_m"
                ],
                "relative_improvement_vs_persistence": fit[
                    "pooled_source_relative_improvement_vs_persistence"
                ],
                "leave_one_source_win_fraction": fit[
                    "leave_one_source_win_fraction"
                ],
                "competence_passed": bool(competence["passed"]),
            }
        )
    pooled_mean = float(np.mean(pooled_means)) if pooled_means else None
    persistence_mean = (
        float(np.mean(persistence_means)) if persistence_means else None
    )
    aggregate_relative = (
        (persistence_mean - pooled_mean) / persistence_mean
        if pooled_mean is not None and persistence_mean
        else None
    )
    aggregate_loo = float(np.mean(loo_wins)) if loo_wins else None
    gate = protocol["config"]["gates"]["source_backend_competence"]
    complete_aggregate = len(by_object) == len(expected_objects)
    aggregate_passed = bool(
        complete_aggregate
        and aggregate_relative is not None
        and aggregate_loo is not None
        and aggregate_relative
        >= float(gate["minimum_pooled_chamfer_improvement_vs_persistence"])
        and aggregate_loo
        >= float(gate["minimum_leave_one_source_win_fraction"])
    )
    admitted = [row["object_id"] for row in records if row["competence_passed"]]
    rejected = [row["object_id"] for row in records if not row["competence_passed"]]
    full_admission = len(admitted) == len(records)
    payload: dict[str, Any] = {
        "schema_version": SOURCE_BACKEND_DECISION_SCHEMA_VERSION,
        "artifact_kind": "Deform360ReplicationSourceBackendDecision",
        "protocol_config_sha256": protocol["config_sha256"],
        "backend_policy_config_sha256": backend_policy["config_sha256"],
        "decision_scope": "per-object competence before the six-object target phase",
        "object_results": records,
        "admitted_object_ids": admitted,
        "rejected_object_ids": rejected,
        "cohort_aggregate_diagnostic": {
            "complete_cohort": complete_aggregate,
            "pooled_fit_object_count": len(by_object),
            "source_stage_failure_object_count": len(failures_by_object),
            "pooled_source_mean_chamfer_m": pooled_mean,
            "persistence_source_mean_chamfer_m": persistence_mean,
            "relative_improvement_vs_persistence": aggregate_relative,
            "leave_one_source_win_fraction": aggregate_loo,
            "passed": aggregate_passed,
        },
        "target_decision": {
            "full_six_object_official_warp_replication_admitted": full_admission,
            "target_prefix_access_permitted": full_admission,
            "target_future_access_permitted": False,
            "rule": "every object-level source backend must pass before target-prefix access",
        },
        "information_boundary": {
            "source_geometry_and_tactile_read": True,
            "calibration_outcomes_read": False,
            "target_prefix_read": False,
            "target_future_geometry_read": False,
            "target_future_tactile_read": False,
        },
    }
    payload["result_sha256"] = _artifact_sha256(payload)
    return payload


def validate_source_backend_decision_artifact(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    _require(
        payload.get("schema_version") == SOURCE_BACKEND_DECISION_SCHEMA_VERSION,
        "source-backend decision schema changed",
    )
    _require(
        payload.get("artifact_kind")
        == "Deform360ReplicationSourceBackendDecision",
        "source-backend decision kind changed",
    )
    _require(
        payload.get("result_sha256") == _artifact_sha256(payload),
        "source-backend decision checksum mismatch",
    )
    boundary = payload.get("information_boundary", {})
    _require(
        boundary.get("calibration_outcomes_read") is False
        and boundary.get("target_prefix_read") is False
        and boundary.get("target_future_geometry_read") is False
        and boundary.get("target_future_tactile_read") is False,
        "source-backend decision crossed its information boundary",
    )
    admitted = payload["admitted_object_ids"]
    rejected = payload["rejected_object_ids"]
    _require(
        len(admitted) + len(rejected) == len(payload["object_results"]),
        "source-backend object partition changed",
    )
    full_admission = len(rejected) == 0
    for row in payload["object_results"]:
        object_id = row["object_id"]
        if row.get("outcome_kind") == "source-stage-failure":
            _require(
                row["competence_passed"] is False and object_id in rejected,
                "source-stage failure was not rejected",
            )
        _require(
            (object_id in admitted) is bool(row["competence_passed"]),
            "source-backend object membership is inconsistent",
        )
    decision = payload["target_decision"]
    _require(
        decision["full_six_object_official_warp_replication_admitted"]
        is full_admission,
        "source-backend target decision is inconsistent",
    )
    _require(
        decision["target_prefix_access_permitted"] is full_admission
        and decision["target_future_access_permitted"] is False,
        "source-backend access decision changed",
    )
    return {
        "passed": True,
        "full_replication_admitted": full_admission,
        "admitted_object_count": len(admitted),
        "rejected_object_count": len(rejected),
        "result_sha256": payload["result_sha256"],
    }


def write_source_backend_decision_artifact(
    path: str | Path, payload: Mapping[str, Any]
) -> Path:
    validate_source_backend_decision_artifact(payload)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return output


def write_source_stage_failure_artifact(
    path: str | Path, payload: Mapping[str, Any]
) -> Path:
    validate_source_stage_failure_artifact(payload)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return output


__all__ = [
    "backend_policy_sha256",
    "build_source_backend_decision_artifact",
    "build_source_stage_failure_artifact",
    "load_backend_policy",
    "validate_backend_policy",
    "validate_source_backend_decision_artifact",
    "validate_source_stage_failure_artifact",
    "write_source_backend_decision_artifact",
    "write_source_stage_failure_artifact",
]
