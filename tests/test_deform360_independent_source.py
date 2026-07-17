from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import causal4d_public.deform360_independent_source as independent


ROOT = Path(__file__).resolve().parents[1]
LOCK = (
    ROOT
    / "configs"
    / "causal4d_public"
    / "deform360_graph_action_support_independent_source_v1.json"
)


def _prediction_bundle() -> dict[str, object]:
    points = np.asarray(
        [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.2, 0.0, 0.0]],
        dtype=np.float32,
    )
    return {
        "object_points": np.repeat(points[None], 4, axis=0),
        "object_colors": np.zeros((4, 3, 3), dtype=np.float32),
        "object_visibilities": np.ones((4, 3), dtype=bool),
        "object_motions_valid": np.ones((4, 3), dtype=bool),
        "controller_points": np.zeros((4, 2, 3), dtype=np.float32),
        "prediction_only_input": {
            "schema_version": 1,
            "object_id": "002-rope-silk",
            "episode_id": 2,
            "object_observation_frames_used": [0],
            "known_future_robot_trajectory_used": True,
            "future_object_observations_present": False,
            "future_tactile_used": False,
        },
    }


def _evaluation(object_id: str, episode_id: int, improvement: float) -> dict:
    baseline_track = 0.04
    baseline_chamfer = 0.03
    interval = {
        "frame_count": 25,
        "track_rmse_m": baseline_track * (1.0 - improvement),
        "chamfer_m": baseline_chamfer * (1.0 - improvement),
        "persistence_track_rmse_m": baseline_track,
        "persistence_chamfer_m": baseline_chamfer,
        "relative_score_vs_persistence": 1.0 - improvement,
        "track_improvement_fraction": improvement,
        "chamfer_improvement_fraction": improvement,
    }
    payload = {
        "schema_version": 1,
        "artifact_kind": "Deform360IndependentSourceEpisodeEvaluation",
        "protocol_id": independent.INDEPENDENT_SOURCE_PROTOCOL_ID,
        "object_id": object_id,
        "episode_id": episode_id,
        "episode_key": f"{object_id}/{episode_id}",
        "prediction_seal_sha256": "a" * 64,
        "target_data_sha256": "b" * 64,
        "metrics": {
            "future": {**interval, "frame_count": 75},
            "early": interval,
            "middle": interval,
            "late": interval,
        },
        "joint_future_win": improvement > 0.0,
        "information_boundary": {
            "deployable_prediction_previously_sealed": True,
            "source_future_opened_for_scoring": True,
            "calibration_outcome_read": False,
            "target_outcome_read": False,
        },
        "claim_boundary": "test",
    }
    payload["result_sha256"] = independent._result_sha256(payload)
    return payload


def test_lock_authorizes_only_the_27_independent_source_episodes() -> None:
    lock = independent.load_independent_source_lock(LOCK)
    assert (
        lock["frozen_predictor"]["official_phystwin_revision"]
        == "2b6630528141b9cba5a7677c8b88b2129b4a8390"
    )
    assert lock["frozen_predictor"]["warp_dynamics"]["init_spring_y"] == 10000.0
    authorized = independent.authorize_independent_source_episode(
        lock, "002-rope-silk", 2
    )
    assert authorized["episode_key"] == "002-rope-silk/2"
    with pytest.raises(ValueError, match="not authorized"):
        independent.authorize_independent_source_episode(lock, "002-rope-silk", 0)


def test_prediction_bundle_rejects_changing_future_object_geometry() -> None:
    bundle = _prediction_bundle()
    validated = independent.validate_prediction_only_bundle(bundle)
    assert validated["frame_count"] == 4
    bundle["object_points"][2, 0, 1] = 0.01
    with pytest.raises(ValueError, match="changing future object observations"):
        independent.validate_prediction_only_bundle(bundle)


def test_prediction_seal_detects_archive_mutation(tmp_path: Path) -> None:
    archive = tmp_path / "prediction.npz"
    arrays = {
        "prediction_m": np.zeros((2, 3, 3), dtype=np.float32),
        "persistence_m": np.zeros((2, 3, 3), dtype=np.float32),
        "driven_readout_m": np.zeros((2, 3, 3), dtype=np.float32),
        "zero_action_readout_m": np.zeros((2, 3, 3), dtype=np.float32),
        "action_support": np.ones(3, dtype=np.float32),
        "frame_zero_points_m": np.zeros((3, 3), dtype=np.float32),
    }
    np.savez_compressed(archive, **arrays)
    payload = {
        "schema_version": 1,
        "artifact_kind": "Deform360IndependentSourcePredictionSeal",
        "protocol_id": independent.INDEPENDENT_SOURCE_PROTOCOL_ID,
        "lock_sha256": "c" * 64,
        "prediction_archive": {
            "path": str(archive),
            "file_sha256": independent.sha256_file(archive),
            "array_sha256": {
                name: independent.sha256_array(value) for name, value in arrays.items()
            },
        },
        "information_boundary": {
            "object_observation_frames_used": [0],
            "future_object_track_read": False,
            "future_object_visibility_read": False,
            "future_tactile_read": False,
            "external_target_scoring_in_warp": False,
            "prediction_hashed_before_future_outcome_scoring": True,
        },
    }
    payload["result_sha256"] = independent._result_sha256(payload)
    independent.validate_independent_source_prediction_seal(
        payload, verify_archive=True
    )
    arrays["prediction_m"][1, 0, 0] = 1.0
    np.savez_compressed(archive, **arrays)
    with pytest.raises(ValueError, match="archive checksum changed"):
        independent.validate_independent_source_prediction_seal(
            payload, verify_archive=True
        )


def test_conjunctive_gate_passes_only_complete_transferring_panel() -> None:
    evaluations = [
        _evaluation(object_id, episode_id, 0.08)
        for object_id, episode_ids in (
            independent.EXPECTED_INDEPENDENT_SOURCE_EPISODES.items()
        )
        for episode_id in episode_ids
    ]
    passed = independent.aggregate_independent_source_gate(evaluations, lock_path=LOCK)
    assert passed["passed"] is True
    assert passed["episode_count"] == 27

    failed = json.loads(json.dumps(evaluations))
    failed[0]["metrics"]["future"]["track_rmse_m"] *= 1.5
    failed[0]["metrics"]["future"]["track_improvement_fraction"] = -0.38
    failed[0]["joint_future_win"] = False
    failed[0]["result_sha256"] = independent._result_sha256(failed[0])
    result = independent.aggregate_independent_source_gate(failed, lock_path=LOCK)
    assert result["passed"] is False
    assert result["gates"]["maximum_per_episode_track_degradation"] is False
