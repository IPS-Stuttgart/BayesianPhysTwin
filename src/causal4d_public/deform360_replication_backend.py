"""Pre-target source-backend decision for the Deform360 replication."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .deform360_replication import validate_deform360_replication_protocol
from .deform360_replication_fit import validate_pooled_source_warp_fit


SOURCE_BACKEND_DECISION_SCHEMA_VERSION = 1
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


def build_source_backend_decision_artifact(
    protocol: Mapping[str, Any],
    pooled_fits: Sequence[Mapping[str, Any]],
    backend_policy: Mapping[str, Any],
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
    cohort = protocol["config"]["cohort"]
    expected_objects = [record["object_id"] for record in cohort]
    _require(set(by_object) == set(expected_objects), "pooled object cohort changed")
    records = []
    pooled_means = []
    persistence_means = []
    loo_wins = []
    for cohort_record in cohort:
        object_id = cohort_record["object_id"]
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
    pooled_mean = float(np.mean(pooled_means))
    persistence_mean = float(np.mean(persistence_means))
    aggregate_relative = (persistence_mean - pooled_mean) / persistence_mean
    aggregate_loo = float(np.mean(loo_wins))
    gate = protocol["config"]["gates"]["source_backend_competence"]
    aggregate_passed = bool(
        aggregate_relative
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


__all__ = [
    "backend_policy_sha256",
    "build_source_backend_decision_artifact",
    "load_backend_policy",
    "validate_backend_policy",
    "validate_source_backend_decision_artifact",
    "write_source_backend_decision_artifact",
]
