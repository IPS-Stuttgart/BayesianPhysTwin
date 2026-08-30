from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from bayesian_phystwin.rgbench_matphys_protocol_v1 import (
    load_rgbench_matphys_preaccess_amendment_v1,
)
from bayesian_phystwin.rgbench_matphys_source_v1 import (
    LEFT_BASE_OFFSET_M,
    RIGHT_BASE_OFFSET_M,
    build_rgbench_matphys_graph_v1,
    camera_points_to_world_v1,
    deterministic_farthest_points_v1,
    interpolate_pose_positions_v1,
    load_episode_world_points_v1,
    load_rgbench_source_episode_index_v1,
    pcd_timestamp_s_v1,
    read_binary_xyzrgb_pcd_v1,
    read_pose_trajectory_csv_v1,
    resolve_amended_source_episode_dir_v1,
    spring_graph_component_count_v1,
)

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "protocols/locks/rgbench_matphys_selective_risk_v1.json"
AMENDMENT = (
    ROOT / "protocols/amendments/rgbench_matphys_selective_risk_v1_preaccess.json"
)


def _amended():
    return load_rgbench_matphys_preaccess_amendment_v1(PROTOCOL, AMENDMENT)


def _write_pcd(path: Path, points: np.ndarray) -> None:
    points = np.asarray(points, dtype=np.float32)
    records = np.empty(
        len(points),
        dtype=np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("rgb", "<f4")]),
    )
    records["x"], records["y"], records["z"] = points.T
    records["rgb"] = 0.0
    header = "\n".join(
        (
            "# .PCD v0.7 - Point Cloud Data file format",
            "VERSION 0.7",
            "FIELDS x y z rgb",
            "SIZE 4 4 4 4",
            "TYPE F F F F",
            "COUNT 1 1 1 1",
            f"WIDTH {len(points)}",
            "HEIGHT 1",
            "VIEWPOINT 0 0 0 1 0 0 0",
            f"POINTS {len(points)}",
            "DATA binary",
            "",
        )
    ).encode("ascii")
    path.write_bytes(header + records.tobytes())


def _write_pose(path: Path, *, y: float) -> None:
    columns = [
        "time",
        "pos_1",
        "pos_2",
        "pos_3",
        "pos_4",
        "pos_5",
        "pos_6",
        "pos_gripper",
        "pos_x",
        "pos_y",
        "pos_z",
        "orn_w",
        "orn_x",
        "orn_y",
        "orn_z",
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for index, timestamp in enumerate((9.5, 10.5, 11.5, 12.5)):
            row = {column: 0.0 for column in columns}
            row.update(
                {
                    "time": timestamp,
                    "pos_x": 0.1 * index,
                    "pos_y": y,
                    "pos_z": 0.7,
                    "orn_w": 1.0,
                }
            )
            writer.writerow(row)


def _source_tree(tmp_path: Path) -> tuple[Path, Path]:
    amended = _amended()
    cell = next(
        item
        for item in amended.source_cells
        if item.garment_id == "beige_hoodie" and item.action == "fling"
    )
    episode = tmp_path.joinpath(*Path(cell.data_subfolder).parts)
    (episode / "segment_pcds").mkdir(parents=True)
    (episode / "calibration").mkdir()
    (episode / "joints").mkdir()
    points = np.asarray(
        [
            (0.0, 0.0, 0.5),
            (0.1, 0.0, 0.5),
            (0.0, 0.1, 0.5),
            (0.1, 0.1, 0.5),
        ],
        dtype=np.float32,
    )
    for timestamp in (10.0, 11.0, 12.0):
        _write_pcd(
            episode / "segment_pcds" / f"pointcloud_{timestamp:.1f}_segmented.pcd",
            points + np.asarray((timestamp - 10.0, 0.0, 0.0), dtype=np.float32),
        )
    transform = np.eye(4)
    transform[:3, 3] = (1.0, 2.0, 3.0)
    (episode / "calibration" / "world_to_camera_transform.json").write_text(
        json.dumps(transform.tolist()), encoding="utf-8"
    )
    _write_pose(episode / "joints" / "left_arm_joint_states_and_end_pose.csv", y=0.0)
    _write_pose(episode / "joints" / "right_arm_joint_states_and_end_pose.csv", y=0.0)
    return tmp_path, episode


def test_reserved_target_is_rejected_before_dataset_root_resolution() -> None:
    with pytest.raises(PermissionError, match="target cell access is forbidden"):
        resolve_amended_source_episode_dir_v1(
            _amended(),
            "/this/path/must/not/be-resolved",
            garment_id="grey_sunwear",
            action="fling",
            sample_id="01",
        )


def test_source_episode_index_uses_exact_roster_path_and_bilateral_offsets(
    tmp_path: Path,
) -> None:
    root, episode_dir = _source_tree(tmp_path)

    episode = load_rgbench_source_episode_index_v1(
        _amended(),
        root,
        garment_id="beige_hoodie",
        action="fling",
        sample_id="01",
        camera_delay_s=0.25,
    )

    assert episode.episode_dir == episode_dir.resolve()
    np.testing.assert_array_equal(episode.frame_times_s, (10.0, 11.0, 12.0))
    np.testing.assert_allclose(
        episode.controller_points_m[:, 0, 1], LEFT_BASE_OFFSET_M[1]
    )
    np.testing.assert_allclose(
        episode.controller_points_m[:, 1, 1], RIGHT_BASE_OFFSET_M[1]
    )
    np.testing.assert_allclose(
        episode.controller_points_m[:, :, 0],
        np.asarray(((0.075, 0.075), (0.175, 0.175), (0.275, 0.275))),
    )

    world = load_episode_world_points_v1(episode, 0)
    np.testing.assert_allclose(world[0], (-1.0, -2.0, -2.5), atol=1e-7)


def test_binary_pcd_reader_is_strict_and_accounts_for_every_byte(
    tmp_path: Path,
) -> None:
    path = tmp_path / "pointcloud_1.0_segmented.pcd"
    expected = np.asarray(((1.0, 2.0, 3.0), (4.0, 5.0, 6.0)), dtype=np.float32)
    _write_pcd(path, expected)

    np.testing.assert_array_equal(read_binary_xyzrgb_pcd_v1(path), expected)
    assert pcd_timestamp_s_v1(path) == 1.0

    path.write_bytes(path.read_bytes() + b"x")
    with pytest.raises(ValueError, match="payload length changed"):
        read_binary_xyzrgb_pcd_v1(path)


def test_pose_reader_and_interpolator_reject_extrapolation(tmp_path: Path) -> None:
    path = tmp_path / "pose.csv"
    _write_pose(path, y=0.2)
    trajectory = read_pose_trajectory_csv_v1(path, base_offset_m=np.zeros(3))

    points = interpolate_pose_positions_v1(trajectory, np.asarray((10.0, 11.0)))
    np.testing.assert_allclose(points[:, 0], (0.05, 0.15))
    np.testing.assert_allclose(points[:, 1], 0.2)

    with pytest.raises(ValueError, match="would extrapolate"):
        interpolate_pose_positions_v1(trajectory, np.asarray((9.0, 10.0)))


def test_camera_transform_uses_inverse_world_to_camera_convention() -> None:
    transform = np.eye(4)
    transform[:3, 3] = (1.0, 2.0, 3.0)

    actual = camera_points_to_world_v1(np.asarray(((1.0, 2.0, 3.0),)), transform)

    np.testing.assert_allclose(actual, ((0.0, 0.0, 0.0),))


def test_farthest_sampling_is_permutation_invariant_and_graph_is_connected() -> None:
    grid = np.asarray(
        [
            (x, y, 0.5)
            for x in np.linspace(0.0, 0.3, 7)
            for y in np.linspace(-0.2, 0.2, 9)
        ],
        dtype=np.float32,
    )
    first = deterministic_farthest_points_v1(grid, count=32)
    second = deterministic_farthest_points_v1(grid[::-1], count=32)
    np.testing.assert_array_equal(first, second)

    graph = build_rgbench_matphys_graph_v1(
        grid,
        np.asarray(((0.0, 0.2, 0.5), (0.0, -0.2, 0.5))),
        node_count=32,
        total_mass_kg=0.255,
        object_radius_m=0.13,
        object_max_neighbours=12,
        controller_radius_m=0.08,
        controller_max_neighbours=4,
    )

    assert spring_graph_component_count_v1(graph) == 1
    assert graph.num_object_points == 32
    assert graph.num_object_springs > 31
    np.testing.assert_allclose(np.sum(graph.masses[:32]), 0.255)
    np.testing.assert_array_equal(graph.masses[32:], (1.0, 1.0))
