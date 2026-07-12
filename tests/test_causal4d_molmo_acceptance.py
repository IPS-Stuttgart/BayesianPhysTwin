import json
from pathlib import Path

import numpy as np
import pytest

from causal4d.molmo_acceptance import (
    MolmoAcceptanceThresholds,
    aggregate_molmo_acceptance,
    evaluate_molmo_acceptance_case,
    gate_beta_candidates,
    load_molmo_acceptance_result,
    molmo_acceptance_result_id,
)
from causal4d.molmo_adapter import MolmoForecastBundle, MolmoPhysTwinQuery
from causal4d.rollout_bank import JointRolloutBank


def _fixture(tmp_path: Path, *, static: bool = False):
    point_count = 3
    frame_count = 11
    t0 = 4
    nodes = np.arange(point_count)
    points = np.zeros((frame_count, point_count, 3), dtype=float)
    anchors = np.column_stack(
        (np.arange(point_count) * 0.01, np.zeros(point_count), np.ones(point_count))
    )
    points[:] = anchors
    future_frames = np.asarray([6, 8, 10])
    displacements = np.asarray([0.01, 0.03, 0.06])
    for frame, displacement in zip(future_frames, displacements, strict=True):
        points[frame:, :, 1] = displacement
    image_paths = []
    for frame in (0, 2, 4):
        path = tmp_path / f"{frame}.png"
        path.write_bytes(b"test")
        image_paths.append(path)
    query = MolmoPhysTwinQuery(
        case_name="synthetic",
        raw_case_dir=tmp_path,
        camera_index=0,
        t0_frame=t0,
        history_frame_indices=np.asarray([0, 2, 4]),
        image_paths=tuple(image_paths),
        node_indices=nodes,
        raw_track_indices=nodes,
        points_2d_xy=np.zeros((point_count, 2)),
        points_3d_world_history_m=points[[0, 2, 4]],
        camera_to_world=np.eye(4),
        intrinsics=np.eye(3),
        source_fps=30.0,
        forecast_fps=15.0,
        frame_stride=2,
    )
    exact = np.transpose(points[future_frames][:, nodes], (1, 0, 2))
    forecast = np.repeat(anchors[:, None], len(future_frames), axis=1) if static else exact
    bundle = MolmoForecastBundle(
        query=query,
        forecast_ids=("instruction", "paraphrase_one", "paraphrase_two"),
        captions=("Lift it.", "Raise it.", "Move it upward."),
        future_camera_m=np.repeat(forecast[None], 3, axis=0),
        future_world_m=np.repeat(forecast[None], 3, axis=0),
        raw_text=("tracks", "tracks", "tracks"),
        checkpoint="synthetic-checkpoint",
    )

    trajectories = np.repeat(anchors[None, None, None], 3 * 1 * 7, axis=0).reshape(
        3, 1, 7, point_count, 3
    )
    for offset, displacement in zip((2, 4, 6), displacements, strict=True):
        trajectories[0, 0, offset:, :, 1] = displacement
        trajectories[1, 0, offset:, :, 1] = -displacement
    metadata = tuple(
        {
            "action": {
                "proposal_id": action,
                "future_action_observed": action == "lift",
            }
        }
        for action in ("lift", "reverse", "persist")
    )
    bank = JointRolloutBank(
        hypothesis_ids=("lift", "reverse", "persist"),
        hypothesis_metadata=metadata,
        hypothesis_prior_weights=np.ones(3),
        parameter_particles=np.zeros((1, 1)),
        parameter_weights=np.ones(1),
        trajectories=trajectories,
    )
    manifest = {
        "action_proposals": [
            {"proposal_id": action, "future_action_observed": action == "lift"}
            for action in ("lift", "reverse", "persist")
        ]
    }
    return bundle, points, np.ones(points.shape[:2], dtype=bool), bank, manifest


def _evaluate(tmp_path: Path, *, static: bool = False):
    bundle, points, valid, bank, manifest = _fixture(tmp_path, static=static)
    thresholds = MolmoAcceptanceThresholds(minimum_independent_cases=1)
    case = evaluate_molmo_acceptance_case(
        case_id="synthetic",
        bundle=bundle,
        object_points_m=points,
        validity=valid,
        bank=bank,
        bank_manifest=manifest,
        primary_forecast_id="instruction",
        paraphrase_forecast_ids=(
            "instruction",
            "paraphrase_one",
            "paraphrase_two",
        ),
        thresholds=thresholds,
    )
    return case, aggregate_molmo_acceptance((case,), thresholds)


def test_competent_forecast_passes_every_gate(tmp_path: Path) -> None:
    case, decision = _evaluate(tmp_path)
    assert case["passed"]
    assert all(case["gates"].values())
    assert case["direct_forecast"]["target_frame_indices"] == [6, 8, 10]
    assert decision["accepted_for_semantic_reweighting"]
    assert decision["blocking_reasons"] == []


def test_static_forecast_is_rejected_before_beta_selection(tmp_path: Path) -> None:
    case, decision = _evaluate(tmp_path, static=True)
    assert not case["gates"]["beats_zero_motion"]
    assert not case["gates"]["motion_scale"]
    assert not case["gates"]["correct_rollout_ranking"]
    assert not decision["accepted_for_semantic_reweighting"]
    assert decision["decision"] == "keep_beta_zero_and_exclude_semantic_improvement_claim"
    assert decision["safe_fallback_frequency"] == 1.0


def test_positive_beta_is_locked_behind_acceptance_artifact(tmp_path: Path) -> None:
    def write_result(path: Path, accepted: bool) -> None:
        payload = {
            "schema_version": 1,
            "decision": {"accepted_for_semantic_reweighting": accepted},
        }
        payload["acceptance_result_id"] = molmo_acceptance_result_id(payload)
        path.write_text(json.dumps(payload), encoding="utf-8")

    rejected_path = tmp_path / "rejected.json"
    write_result(rejected_path, False)
    rejected = load_molmo_acceptance_result(rejected_path)
    assert gate_beta_candidates((0.0, 1.0, 12.0), rejected) == (0.0,)

    accepted_path = tmp_path / "accepted.json"
    write_result(accepted_path, True)
    accepted = load_molmo_acceptance_result(accepted_path)
    assert gate_beta_candidates((12.0, 0.0, 1.0), accepted) == (0.0, 1.0, 12.0)

    tampered = json.loads(accepted_path.read_text(encoding="utf-8"))
    tampered["decision"]["accepted_for_semantic_reweighting"] = False
    accepted_path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="digest"):
        load_molmo_acceptance_result(accepted_path)
