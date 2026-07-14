from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from causal4d_public.deform360_replication_backend import (
    backend_policy_sha256,
    build_source_backend_decision_artifact,
    load_backend_policy,
    validate_backend_policy,
    validate_source_backend_decision_artifact,
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
