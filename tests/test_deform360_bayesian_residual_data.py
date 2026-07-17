from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from causal4d_public.deform360_bayesian_residual_data import (
    contact_probabilities_from_state,
    load_deform360_residual_source_episode,
)


def _episode_fixture(root: Path) -> Path:
    episode = root / "episode_0000"
    pcd = episode / "pcd_clean"
    robot = episode / "robot"
    pcd.mkdir(parents=True)
    robot.mkdir()
    points = np.column_stack(
        (np.linspace(0.0, 0.25, 6), np.zeros(6), np.zeros(6))
    ).astype(np.float32)
    for frame in range(4):
        shifted = points.copy()
        shifted[:, 1] += 0.002 * frame
        np.savez(
            pcd / f"{frame:06d}.npz",
            pts=shifted,
            vels=np.full_like(points, 0.01),
            visibility_matrix=np.ones((6, 3), dtype=np.uint8),
        )
    (pcd / "pcd_clean.meta.json").write_text(
        json.dumps({"parameters": {"frame_rate_hz": 20.0}}), encoding="utf-8"
    )
    transforms = np.repeat(np.eye(4)[None], 4, axis=0)
    transforms[:, 0, 3] = np.linspace(0.0, 0.03, 4)
    np.savez(
        robot / "robot.npz",
        T_worlds=transforms,
        openings=np.array([0.10, 0.08, 0.04, 0.03]),
    )
    return episode


def test_source_loader_builds_residual_inputs(tmp_path: Path) -> None:
    episode_dir = _episode_fixture(tmp_path)

    episode = load_deform360_residual_source_episode(
        episode_dir,
        object_id="002-rope-silk",
        episode_id=2,
        maximum_node_count=4,
        neighbor_count=2,
    )

    assert episode.episode_key == "002-rope-silk/2"
    assert episode.positions_m.shape == (4, 4, 3)
    assert episode.controller_positions_m.shape == (4, 1, 3)
    assert episode.controller_geometry == "end_effector_origins"
    np.testing.assert_array_equal(episode.controller_group_ids, [0])
    assert episode.edge_index.shape == (2, 8)
    assert episode.frame_interval_s == 0.05
    assert not episode.positions_m.flags.writeable
    contact = episode.contact_probabilities(2, episode.positions_m[2])
    assert contact.shape == (4, 1)
    assert np.all((contact >= 0.0) & (contact <= 1.0))


def test_source_loader_rejects_sealed_penguin_episode(tmp_path: Path) -> None:
    episode_dir = _episode_fixture(tmp_path)

    with pytest.raises(ValueError, match="outside the already-open"):
        load_deform360_residual_source_episode(
            episode_dir,
            object_id="171-penguin",
            episode_id=0,
        )


def test_contact_probability_uses_only_closure_and_geometry() -> None:
    positions = np.array([[0.0, 0.0, 0.0], [0.10, 0.0, 0.0]])
    controllers = np.array([[0.0, 0.0, 0.0]])

    contact = contact_probabilities_from_state(
        positions, controllers, np.array([0.8]), proximity_scale_m=0.03
    )

    assert contact[0, 0] == pytest.approx(0.8)
    assert contact[1, 0] < 0.01


def test_source_loader_uses_fixed_gripper_surface_identities(tmp_path: Path) -> None:
    episode_dir = _episode_fixture(tmp_path)

    def surface(opening_m: float, transform: np.ndarray) -> np.ndarray:
        root = np.array(
            [[0.0, 0.0, 0.0], [0.04 + opening_m, 0.0, 0.0], [0.0, 0.04, 0.0]]
        )
        return root @ transform[:3, :3].T + transform[:3, 3]

    episode = load_deform360_residual_source_episode(
        episode_dir,
        object_id="002-rope-silk",
        episode_id=2,
        maximum_node_count=4,
        neighbor_count=2,
        controller_surface_provider=surface,
        controller_points_per_gripper=2,
    )

    assert episode.controller_positions_m.shape == (4, 2, 3)
    assert episode.controller_geometry == "gripper_surface"
    np.testing.assert_array_equal(episode.controller_group_ids, [0, 0])
    assert not np.allclose(
        episode.controller_positions_m[0], episode.controller_positions_m[-1]
    )


def test_source_loader_aligns_sealed_physics_by_material_identity(
    tmp_path: Path,
) -> None:
    episode_dir = _episode_fixture(tmp_path)
    frames = [
        np.load(path)["pts"]
        for path in sorted((episode_dir / "pcd_clean").glob("*.npz"))
    ]
    physics = np.stack(frames)
    physics[1:, :, 2] += 0.01
    prediction_path = tmp_path / "prediction.npz"
    np.savez(
        prediction_path,
        prediction_m=physics,
        frame_zero_points_m=frames[0],
    )

    episode = load_deform360_residual_source_episode(
        episode_dir,
        object_id="002-rope-silk",
        episode_id=2,
        maximum_node_count=4,
        neighbor_count=2,
        physics_prediction_path=prediction_path,
    )

    assert episode.physics_prior_kind == "sealed_graph_action_support"
    np.testing.assert_array_equal(episode.physics_positions_m[0], episode.positions_m[0])
    np.testing.assert_allclose(
        episode.physics_positions_m[1:, :, 2]
        - episode.positions_m[0:1, :, 2],
        0.01,
    )


def test_source_loader_scales_sealed_response_around_persistence(
    tmp_path: Path,
) -> None:
    episode_dir = _episode_fixture(tmp_path)
    frames = [
        np.load(path)["pts"]
        for path in sorted((episode_dir / "pcd_clean").glob("*.npz"))
    ]
    persistence = np.repeat(frames[0][None], len(frames), axis=0)
    physics = persistence.copy()
    physics[1:, :, 2] += 0.009
    prediction_path = tmp_path / "prediction.npz"
    np.savez(
        prediction_path,
        prediction_m=physics,
        frame_zero_points_m=frames[0],
    )

    episode = load_deform360_residual_source_episode(
        episode_dir,
        object_id="002-rope-silk",
        episode_id=2,
        maximum_node_count=4,
        neighbor_count=2,
        physics_prediction_path=prediction_path,
        physics_response_scale=0.3,
        physics_reference_response_scale=0.9,
    )

    assert episode.physics_prior_kind == "trusted_sealed_graph_action_support"
    assert episode.physics_response_scale == pytest.approx(0.3)
    np.testing.assert_allclose(
        episode.physics_positions_m[1:, :, 2]
        - episode.positions_m[0:1, :, 2],
        0.003,
    )


def test_zero_trust_is_byte_identical_to_persistence(tmp_path: Path) -> None:
    episode_dir = _episode_fixture(tmp_path)
    frames = [
        np.load(path)["pts"]
        for path in sorted((episode_dir / "pcd_clean").glob("*.npz"))
    ]
    physics = np.stack(frames)
    physics[1:, :, 2] += 0.02
    prediction_path = tmp_path / "prediction.npz"
    np.savez(
        prediction_path,
        prediction_m=physics,
        frame_zero_points_m=frames[0],
    )

    episode = load_deform360_residual_source_episode(
        episode_dir,
        object_id="002-rope-silk",
        episode_id=2,
        maximum_node_count=4,
        neighbor_count=2,
        physics_prediction_path=prediction_path,
        physics_response_scale=0.0,
    )

    persistence = np.repeat(
        episode.positions_m[0:1], len(episode.positions_m), axis=0
    )
    assert episode.physics_positions_m.tobytes() == persistence.tobytes()
