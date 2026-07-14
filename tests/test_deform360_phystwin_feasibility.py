from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from causal4d_public.deform360_phystwin_feasibility import (
    WarpRopeCandidate,
    WarpRopeFeasibilityConfig,
    _source_forecast_case,
    _summarize_candidate_scores,
    deform360_xyz_to_warp_xzy,
    validate_official_warp_feasibility_artifact,
    warp_rope_candidates,
)
from causal4d_public.deform360_rope_dynamics import RopeDynamicsObservation


def test_official_warp_grid_has_locked_two_hundred_candidates() -> None:
    candidates = warp_rope_candidates(WarpRopeFeasibilityConfig())
    assert len(candidates) == 200
    assert candidates[0] == WarpRopeCandidate(100.0, 1e-6, 100.0, 0.0)
    assert candidates[-1] == WarpRopeCandidate(10000.0, 1000.0, 10000.0, 0.3)


def test_coordinate_transform_is_rigid_and_places_support_above_ground() -> None:
    points = np.asarray([[[0.1, -0.02, 0.3], [0.4, 0.05, 0.7]]], dtype=np.float64)
    transformed = deform360_xyz_to_warp_xzy(
        points, initial_support_height_m=-0.02, clearance_m=0.001
    )
    assert transformed[0, 0, 2] == pytest.approx(0.001)
    assert np.linalg.norm(points[0, 1] - points[0, 0]) == pytest.approx(
        np.linalg.norm(transformed[0, 1] - transformed[0, 0])
    )


def test_source_case_starts_after_six_contact_prefix_frames() -> None:
    frame_count = 12
    positions = np.zeros((frame_count, 4, 3), dtype=np.float64)
    controllers = np.zeros((frame_count, 1, 3), dtype=np.float64)
    active = np.zeros((frame_count, 1), dtype=bool)
    active[2:] = True
    observation = RopeDynamicsObservation(
        episode_id="001-rope/episode_0000",
        positions_m=positions,
        controller_positions_m=controllers,
        contact_active=active,
        contact_node_indices=(0,),
        contact_offsets_m=np.asarray([[0.01, 0.0, 0.0]]),
        dt_seconds=1.0 / 15.0,
    )
    case = _source_forecast_case(0, observation, WarpRopeFeasibilityConfig())
    assert case.prefix_start_index == 2
    assert case.prefix_end_index_exclusive == 8
    assert len(case.positions_m) == 5


def test_source_gate_summary_requires_transfer_not_only_pooled_fit() -> None:
    config = WarpRopeFeasibilityConfig(
        stretch_spring_y_grid=(100.0, 300.0),
        bend_spring_y_grid=(1.0,),
        controller_spring_y_grid=(100.0,),
        ground_friction_grid=(0.0,),
    )
    candidates = warp_rope_candidates(config)
    scores = np.asarray(
        [
            [0.5, 0.5, 0.5, 0.5, 0.5],
            [0.4, 0.4, 0.4, 0.4, 0.4],
        ]
    )
    tracks = scores + 0.1
    persistence = np.full(5, 0.6)
    summary = _summarize_candidate_scores(
        candidates,
        [f"episode-{index}" for index in range(5)],
        scores,
        tracks,
        persistence,
        config=config,
    )
    assert summary["selected_candidate_index"] == 1
    assert summary["observed_leave_one_source_win_fraction"] == 1.0
    assert summary["competence_passed"] is True


def test_source_gate_summary_rejects_nonfinite_candidates() -> None:
    config = WarpRopeFeasibilityConfig(
        stretch_spring_y_grid=(100.0, 300.0),
        bend_spring_y_grid=(1.0,),
        controller_spring_y_grid=(100.0,),
        ground_friction_grid=(0.0,),
    )
    candidates = warp_rope_candidates(config)
    scores = np.asarray(
        [
            [np.inf, np.inf, np.inf, np.inf, np.inf],
            [0.4, 0.4, 0.4, 0.4, 0.4],
        ]
    )
    summary = _summarize_candidate_scores(
        candidates,
        [f"episode-{index}" for index in range(5)],
        scores,
        scores.copy(),
        np.full(5, 0.6),
        config=config,
    )
    assert summary["selected_candidate_index"] == 1


def test_archived_official_warp_gate_is_internally_and_file_hash_locked() -> None:
    root = (
        Path(__file__).resolve().parents[1]
        / "milestones"
        / "deform360-official-warp-source-gate-v1"
    )
    manifest = json.loads((root / "artifact-manifest.json").read_text())
    result_path = root / manifest["artifacts"][0]["path"]
    payload = json.loads(result_path.read_text())
    validated = validate_official_warp_feasibility_artifact(
        payload, verify_archive=False
    )
    assert validated["gate_passed"] is True
    assert validated["result_sha256"] == manifest["result_sha256"]
    for record in manifest["artifacts"]:
        path = root / record["path"]
        assert path.stat().st_size == record["bytes"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == record["sha256"]
