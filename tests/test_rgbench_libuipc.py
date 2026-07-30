from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from bayesian_phystwin.rgbench_libuipc import (
    FlingPinController,
    LibuIPCClothParameters,
    PositionTrajectory,
    libuipc_vector_values,
    load_rgbbench_position_trajectory,
    transform_vertices_wxyz,
)
from scripts.held.run_rgbbench_libuipc_competence_v3 import SOURCE_DIGEST_KEYS


def _trajectory(offset: float = 0.0) -> PositionTrajectory:
    return PositionTrajectory(
        np.asarray([10.0, 11.0, 12.0]),
        np.asarray(
            [
                [offset, 0.0, 0.0],
                [offset + 1.0, 0.0, 0.0],
                [offset + 2.0, 0.0, 0.0],
            ]
        ),
    )


def test_position_trajectory_interpolates_and_clamps() -> None:
    trajectory = _trajectory()
    np.testing.assert_allclose(trajectory.position_at(9.0), [0.0, 0.0, 0.0])
    np.testing.assert_allclose(trajectory.position_at(10.5), [0.5, 0.0, 0.0])
    np.testing.assert_allclose(trajectory.position_at(13.0), [2.0, 0.0, 0.0])


def test_rgbbench_csv_loader_applies_only_declared_base_offset(
    tmp_path: Path,
) -> None:
    path = tmp_path / "arm.csv"
    path.write_text(
        "time,pos_x,pos_y,pos_z,unused\n"
        "1.0,0.1,0.2,0.3,x\n"
        "2.0,0.4,0.5,0.6,y\n",
        encoding="utf-8",
    )
    trajectory = load_rgbbench_position_trajectory(
        path,
        base_translation_m=(1.0, -2.0, 3.0),
    )
    np.testing.assert_allclose(trajectory.times_s, [1.0, 2.0])
    np.testing.assert_allclose(
        trajectory.positions_m,
        [[1.1, -1.8, 3.3], [1.4, -1.5, 3.6]],
    )


def test_transform_vertices_uses_wxyz_pose() -> None:
    vertices = np.asarray([[1.0, 0.0, 0.0]])
    half = np.sqrt(0.5)
    transformed = transform_vertices_wxyz(
        vertices,
        (1.0, 2.0, 3.0, half, 0.0, 0.0, half),
    )
    np.testing.assert_allclose(transformed, [[1.0, 3.0, 3.0]], atol=1e-12)


def test_libuipc_vector_values_adds_binding_column_dimension() -> None:
    converted = libuipc_vector_values(
        np.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    )
    assert converted.shape == (2, 3, 1)
    np.testing.assert_allclose(converted[..., 0], [[1, 2, 3], [4, 5, 6]])
    assert converted.flags.c_contiguous


def test_fling_controller_matches_prepare_wait_and_playback_phases() -> None:
    controller = FlingPinController(
        pin_indices=(2, 4),
        initial_positions_m=np.asarray([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
        left=_trajectory(1.0),
        right=_trajectory(3.0),
        prepare_time_s=2.0,
        wait_time_s=3.0,
    )
    np.testing.assert_allclose(
        controller.targets_at(1.0),
        [[0.5, 0.0, 0.0], [2.5, 0.0, 0.0]],
    )
    np.testing.assert_allclose(
        controller.targets_at(4.0),
        [[1.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
    )
    np.testing.assert_allclose(
        controller.targets_at(5.5),
        [[1.5, 0.0, 0.0], [3.5, 0.0, 0.0]],
    )


def test_physical_parameters_reject_nonphysical_values() -> None:
    with pytest.raises(ValueError, match="poisson_ratio"):
        LibuIPCClothParameters(
            timestep_s=0.01,
            youngs_modulus_pa=1e5,
            poisson_ratio=0.5,
            volume_density_kg_m3=200.0,
            thickness_m=0.001,
            bending_stiffness=1000.0,
            friction_coefficient=0.5,
            contact_distance_m=0.003,
            contact_resistance=1e9,
            constraint_strength_ratio=100.0,
        )


def test_competence_protocol_exposes_every_runner_source_digest() -> None:
    root = Path(__file__).resolve().parents[1]
    payload = json.loads(
        (
            root / "configs" / "sota" / "rgbbench_libuipc_competence_v3.json"
        ).read_text(encoding="utf-8")
    )
    case = payload["competence_case"]
    assert set(SOURCE_DIGEST_KEYS.values()) <= set(case)
