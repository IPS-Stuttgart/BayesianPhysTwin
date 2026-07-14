from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from causal4d_public.deform360_replication_backend import (
    backend_policy_sha256,
    build_source_backend_decision_artifact,
    build_source_stage_failure_artifact,
    load_backend_policy,
    validate_backend_policy,
    validate_source_backend_decision_artifact,
    validate_source_stage_failure_artifact,
)
from causal4d_public.deform360_replication_fit import (
    _artifact_sha256,
    pool_source_warp_candidate_grids,
)


def _source_grid(object_id: str, episode_id: int, physical_score: float) -> dict:
    rows = [
        {
            "candidate_index": index,
            "parameters": {
                "stretch_spring_y": float(index + 1),
                "bend_spring_y": 1.0,
                "controller_spring_y": 1.0,
                "ground_friction": 0.0,
            },
            "mean_chamfer_m": physical_score if index == 0 else 1.2 + index / 1000,
            "p99_relative_edge_strain": 0.2,
        }
        for index in range(200)
    ]
    payload = {
        "schema_version": 3,
        "artifact_kind": "Deform360ReplicationSourceWarpCandidateGrid",
        "episode_id": f"{object_id}/episode_{episode_id:04d}",
        "reference_geometry_result_sha256": "0" * 64,
        "reference_geometry_total_frame_count": 10,
        "reference_geometry_available_frame_count": 10,
        "raw_hull_frame_indices": list(range(10)),
        "config": {"maximum_p99_relative_edge_strain": 0.5},
        "persistence": {"mean_m": 1.0},
        "candidate_scores": rows,
        "information_boundary": {"target_future_read": False},
    }
    payload["result_sha256"] = _artifact_sha256(payload)
    return payload


def test_source_backend_decision_stops_before_targets_on_one_failure() -> None:
    root = Path(__file__).resolve().parents[1]
    protocol = json.loads(
        (
            root / "configs/causal4d_public/deform360_replication_v1.json"
        ).read_text(encoding="utf-8")
    )
    backend_policy = load_backend_policy(
        root / "configs/causal4d_public/deform360_replication_backend_v1.json"
    )
    fits = []
    for object_index, cohort in enumerate(protocol["config"]["cohort"]):
        score = 1.1 if object_index == 0 else 0.8
        fits.append(
            pool_source_warp_candidate_grids(
                [
                    _source_grid(cohort["object_id"], int(episode_id), score)
                    for episode_id in cohort["source_episode_ids"]
                ]
            )
        )
    artifact = build_source_backend_decision_artifact(
        protocol, fits, backend_policy
    )
    validation = validate_source_backend_decision_artifact(artifact)
    assert validation["full_replication_admitted"] is False
    assert validation["rejected_object_count"] == 1
    assert artifact["target_decision"]["target_prefix_access_permitted"] is False

    changed = copy.deepcopy(artifact)
    changed["rejected_object_ids"] = []
    with pytest.raises(ValueError, match="checksum"):
        validate_source_backend_decision_artifact(changed)


def test_backend_policy_is_canonical_and_fail_closed() -> None:
    root = Path(__file__).resolve().parents[1]
    policy = load_backend_policy(
        root / "configs/causal4d_public/deform360_replication_backend_v1.json"
    )
    assert validate_backend_policy(policy)["passed"] is True
    assert backend_policy_sha256(policy) == policy["config_sha256"]
    changed = copy.deepcopy(policy)
    changed["config"]["geometry"]["minimum_consensus_votes"] = 7
    with pytest.raises(ValueError, match="checksum"):
        validate_backend_policy(changed)


def test_source_stage_failure_blocks_targets_without_a_fabricated_fit() -> None:
    root = Path(__file__).resolve().parents[1]
    protocol = json.loads(
        (
            root / "configs/causal4d_public/deform360_replication_v1.json"
        ).read_text(encoding="utf-8")
    )
    backend_policy = load_backend_policy(
        root / "configs/causal4d_public/deform360_replication_backend_v1.json"
    )
    failed_cohort = protocol["config"]["cohort"][-1]
    source_ids = list(map(int, failed_cohort["source_episode_ids"]))
    failure = build_source_stage_failure_artifact(
        protocol,
        backend_policy,
        object_id=failed_cohort["object_id"],
        stage="source-geometry",
        failed_episode_id=(
            f"{failed_cohort['object_id']}/episode_{source_ids[1]:04d}"
        ),
        error_type="ValueError",
        error_message="too many future hull observations are unavailable",
        episode_status=[
            {
                "episode_id": (
                    f"{failed_cohort['object_id']}/episode_{episode_id:04d}"
                ),
                "status": (
                    "completed"
                    if offset == 0
                    else "failed"
                    if offset == 1
                    else "not-attempted"
                ),
            }
            for offset, episode_id in enumerate(source_ids)
        ],
        evidence=[
            {"path": "failures/source-geometry.log", "size_bytes": 12, "sha256": "0" * 64}
        ],
    )
    assert validate_source_stage_failure_artifact(failure)["passed"] is True

    fits = []
    for cohort in protocol["config"]["cohort"][:-1]:
        fits.append(
            pool_source_warp_candidate_grids(
                [
                    _source_grid(cohort["object_id"], int(episode_id), 0.8)
                    for episode_id in cohort["source_episode_ids"]
                ]
            )
        )
    artifact = build_source_backend_decision_artifact(
        protocol, fits, backend_policy, [failure]
    )
    validation = validate_source_backend_decision_artifact(artifact)
    assert validation["full_replication_admitted"] is False
    assert failed_cohort["object_id"] in artifact["rejected_object_ids"]
    assert artifact["cohort_aggregate_diagnostic"]["complete_cohort"] is False
    assert artifact["target_decision"]["target_prefix_access_permitted"] is False


def test_source_pooling_failure_requires_all_episode_grids() -> None:
    root = Path(__file__).resolve().parents[1]
    protocol = json.loads(
        (
            root / "configs/causal4d_public/deform360_replication_v1.json"
        ).read_text(encoding="utf-8")
    )
    backend_policy = load_backend_policy(
        root / "configs/causal4d_public/deform360_replication_backend_v1.json"
    )
    cohort = protocol["config"]["cohort"][0]
    statuses = [
        {
            "episode_id": f"{cohort['object_id']}/episode_{int(index):04d}",
            "status": "completed",
        }
        for index in cohort["source_episode_ids"]
    ]
    failure = build_source_stage_failure_artifact(
        protocol,
        backend_policy,
        object_id=cohort["object_id"],
        stage="source-pooling",
        failed_episode_id=None,
        error_type="ValueError",
        error_message="no candidate is valid on every source",
        episode_status=statuses,
        evidence=[
            {"path": "failures/source-pooling.log", "size_bytes": 12, "sha256": "0" * 64}
        ],
    )
    assert validate_source_stage_failure_artifact(failure)["passed"] is True

    changed = copy.deepcopy(failure)
    changed["episode_status"][0]["status"] = "not-attempted"
    changed["result_sha256"] = _artifact_sha256(changed)
    with pytest.raises(ValueError, match="pooling failure status"):
        validate_source_stage_failure_artifact(changed)
