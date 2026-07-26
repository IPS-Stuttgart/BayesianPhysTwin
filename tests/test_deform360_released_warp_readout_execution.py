from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from causal4d_public.deform360_released_warp_readout import (
    load_released_warp_readout_protocol,
)
from causal4d_public.deform360_released_warp_readout_execution import (
    associate_particles_to_polyline,
    build_matched_origin_warp_cases,
    lift_sparse_polyline_to_particles,
    minimum_rotation_matrix,
    symmetric_chamfer_distance_m,
    validate_released_warp_prediction_artifact,
    validate_released_warp_score_artifact,
)

ROOT = Path(__file__).resolve().parents[1]


def test_polyline_association_reconstructs_origin_particles() -> None:
    polyline = np.asarray(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]
    )
    particles = np.asarray(
        [[0.25, 1.0, 0.0], [1.75, 0.0, 1.0], [1.0, 0.5, 0.0]]
    )

    association = associate_particles_to_polyline(particles, polyline)
    lifted = lift_sparse_polyline_to_particles(
        polyline[None],
        polyline,
        association,
        rotate_offsets=True,
    )

    np.testing.assert_array_equal(association.segment_indices, [0, 1, 0])
    np.testing.assert_allclose(
        association.barycentric_coordinates,
        [0.25, 0.75, 1.0],
    )
    np.testing.assert_allclose(lifted[0], particles)


@pytest.mark.parametrize(
    "target",
    [
        np.asarray([0.0, 1.0, 0.0]),
        np.asarray([-1.0, 0.0, 0.0]),
        np.asarray([1.0, 0.0, 0.0]),
    ],
)
def test_minimum_rotation_maps_parallel_and_antiparallel_vectors(
    target: np.ndarray,
) -> None:
    source = np.asarray([1.0, 0.0, 0.0])

    rotation = minimum_rotation_matrix(source, target)

    np.testing.assert_allclose(rotation @ source, target, atol=1e-12)
    np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-12)
    assert np.linalg.det(rotation) == pytest.approx(1.0)


def _write_source(path: Path, *, future_shift: float) -> None:
    frames = np.asarray([0, 2, 4, 6], dtype=np.int32)
    positions = np.zeros((4, 4, 3), dtype=np.float64)
    positions[:, :, 0] = np.arange(4)[None]
    positions[1, :, 1] = 0.1
    positions[2:, :, 1] = future_shift
    controllers = np.zeros((4, 1, 3), dtype=np.float64)
    controllers[:, 0, 0] = np.arange(4)
    contact = np.asarray([[False], [True], [False], [False]])
    np.savez_compressed(
        path,
        frame_indices=frames,
        positions_m=positions,
        controller_positions_m=controllers,
        contact_active=contact,
        contact_node_indices=np.asarray([0], dtype=np.int32),
        contact_offsets_m=np.asarray([[0.0, 0.001, 0.0]]),
    )


def test_matched_case_ignores_future_object_state_and_contact(tmp_path: Path) -> None:
    first_path = tmp_path / "first.npz"
    second_path = tmp_path / "second.npz"
    _write_source(first_path, future_shift=10.0)
    _write_source(second_path, future_shift=-20.0)
    record = {
        "episode_id": 0,
        "matched_origin_frame": 2,
        "previous_state_frame": 0,
        "evaluation_frames": [4, 6],
    }

    first, first_zero, first_diagnostics = build_matched_origin_warp_cases(
        first_path,
        record,
        dt_seconds=0.5,
    )
    second, second_zero, second_diagnostics = build_matched_origin_warp_cases(
        second_path,
        record,
        dt_seconds=0.5,
    )

    np.testing.assert_array_equal(first.graph.positions_m, second.graph.positions_m)
    np.testing.assert_array_equal(
        first.initial_velocities_m_s,
        second.initial_velocities_m_s,
    )
    np.testing.assert_array_equal(first.contact_active, [[True], [True], [True]])
    np.testing.assert_array_equal(second.contact_active, first.contact_active)
    np.testing.assert_array_equal(
        first_zero.initial_velocities_m_s,
        np.zeros((4, 3)),
    )
    np.testing.assert_array_equal(
        second_zero.initial_velocities_m_s,
        first_zero.initial_velocities_m_s,
    )
    assert first_diagnostics == second_diagnostics
    assert first_diagnostics["future_contact_active_used"] is False


def test_rotated_offset_follows_segment_orientation() -> None:
    origin = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    particles = np.asarray([[0.5, 1.0, 0.0]])
    association = associate_particles_to_polyline(particles, origin)
    trajectory = np.asarray(
        [
            origin,
            [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        ]
    )

    rotated = lift_sparse_polyline_to_particles(
        trajectory,
        origin,
        association,
        rotate_offsets=True,
    )
    fixed = lift_sparse_polyline_to_particles(
        trajectory,
        origin,
        association,
        rotate_offsets=False,
    )

    np.testing.assert_allclose(rotated[1, 0], [-1.0, 0.5, 0.0])
    np.testing.assert_allclose(fixed[1, 0], [0.0, 1.5, 0.0])


def test_chunked_chamfer_matches_explicit_reference() -> None:
    reference = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    prediction = np.asarray(
        [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [1.0, 0.0, 0.0]]
    )

    observed = symmetric_chamfer_distance_m(
        reference,
        prediction,
        chunk_size=1,
    )

    assert observed == pytest.approx(1.0 / 12.0)


def test_frozen_released_warp_readout_milestone_validates() -> None:
    protocol = load_released_warp_readout_protocol(
        ROOT
        / "configs"
        / "causal4d_public"
        / "deform360_released_warp_readout_source_v1.json"
    )
    milestone = (
        ROOT / "milestones" / "deform360-released-warp-readout-source-v1"
    )
    artifact_root = milestone / "artifacts"
    prediction = json.loads(
        (
            artifact_root / "deform360_released_warp_readout_prediction_v1.json"
        ).read_text(encoding="utf-8")
    )
    score = json.loads(
        (
            artifact_root / "deform360_released_warp_readout_score_v1.json"
        ).read_text(encoding="utf-8")
    )
    manifest = json.loads(
        (milestone / "artifact-manifest.json").read_text(encoding="utf-8")
    )

    prediction_validation = validate_released_warp_prediction_artifact(
        prediction,
        protocol=protocol,
        artifact_directory=artifact_root,
    )
    score_validation = validate_released_warp_score_artifact(
        score,
        protocol=protocol,
    )

    assert prediction_validation["passed"] is True
    assert score_validation["passed"] is True
    assert score_validation["transfer_gate_passed"] is False
    assert manifest["decision"] == "stop_released_particle_readout_route"
    for record in manifest["files"]:
        path = milestone / record["path"]
        assert path.stat().st_size == record["bytes"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == record["sha256"]
