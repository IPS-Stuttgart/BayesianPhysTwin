from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from bayesian_phystwin.rgbench_online_belief import (
    evaluation_pcd_paths,
    load_binary_pcd_xyz,
    load_obj_triangles,
    load_rgbbench_world_cloud,
    real_to_sim_l1_chamfer_m,
)
from bayesian_phystwin.rgbench_protocol import (
    CALIBRATION_GARMENTS,
    DATASET_REVISION,
    PAPER_GARMENTS,
    PRIMARY_SAMPLES,
    SOURCE_GARMENTS,
    TARGET_GARMENTS,
    build_rgbbench_dataset_manifest,
    garment_hash,
    garment_split,
)


def _write_binary_pcd(path: Path, points: np.ndarray) -> None:
    values = np.zeros(
        len(points),
        dtype=np.dtype(
            [
                ("x", "<f4"),
                ("y", "<f4"),
                ("z", "<f4"),
                ("rgb", "<f4"),
            ]
        ),
    )
    values["x"] = points[:, 0]
    values["y"] = points[:, 1]
    values["z"] = points[:, 2]
    header = (
        "# .PCD v0.7 - Point Cloud Data file format\n"
        "VERSION 0.7\n"
        "FIELDS x y z rgb\n"
        "SIZE 4 4 4 4\n"
        "TYPE F F F F\n"
        "COUNT 1 1 1 1\n"
        f"WIDTH {len(points)}\n"
        "HEIGHT 1\n"
        "VIEWPOINT 0 0 0 1 0 0 0\n"
        f"POINTS {len(points)}\n"
        "DATA binary\n"
    ).encode("ascii")
    path.write_bytes(header + values.tobytes())


def test_frozen_garment_split_is_salted_hash_partition() -> None:
    ordered = tuple(sorted(PAPER_GARMENTS, key=garment_hash))
    assert ordered[:3] == SOURCE_GARMENTS
    assert ordered[3:5] == CALIBRATION_GARMENTS
    assert ordered[5:] == TARGET_GARMENTS
    assert {garment_split(garment) for garment in PAPER_GARMENTS} == {
        "source",
        "calibration",
        "target",
    }


def test_binary_pcd_loader_and_world_transform(tmp_path: Path) -> None:
    pcd = tmp_path / "cloud.pcd"
    points = np.asarray([[1.0, 2.0, 3.0], [-1.0, 0.5, 4.0]])
    _write_binary_pcd(pcd, points)
    np.testing.assert_allclose(load_binary_pcd_xyz(pcd), points, atol=1e-6)

    world_to_camera = np.eye(4)
    world_to_camera[:3, 3] = [1.0, -2.0, 0.5]
    transform_path = tmp_path / "world_to_camera_transform.json"
    transform_path.write_text(json.dumps(world_to_camera.tolist()), encoding="utf-8")
    expected = points - world_to_camera[:3, 3]
    np.testing.assert_allclose(
        load_rgbbench_world_cloud(pcd, transform_path),
        expected,
        atol=1e-6,
    )


def test_binary_pcd_loader_rejects_ascii(tmp_path: Path) -> None:
    pcd = tmp_path / "ascii.pcd"
    pcd.write_text(
        "FIELDS x y z\nSIZE 4 4 4\nTYPE F F F\nCOUNT 1 1 1\n"
        "POINTS 1\nDATA ascii\n0 0 0\n",
        encoding="ascii",
    )
    with pytest.raises(ValueError, match="not binary PCD"):
        load_binary_pcd_xyz(pcd)


def test_obj_loader_triangulates_polygons(tmp_path: Path) -> None:
    obj = tmp_path / "square.obj"
    obj.write_text(
        "v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nf 1 2 3 4\n",
        encoding="utf-8",
    )
    vertices, faces = load_obj_triangles(obj)
    assert vertices.shape == (4, 3)
    np.testing.assert_array_equal(faces, [[0, 1, 2], [0, 2, 3]])


def test_primary_metric_uses_real_to_sim_l1_direction() -> None:
    real = np.asarray([[0.0, 0.0, 0.0], [100.0, 0.0, 0.0]])
    simulated = np.asarray([[0.0, 0.0, 0.0]])
    assert real_to_sim_l1_chamfer_m(real, simulated) == pytest.approx(50.0)
    assert real_to_sim_l1_chamfer_m(simulated, real) == pytest.approx(0.0)


def test_evaluation_paths_verify_name_digest(tmp_path: Path) -> None:
    capture = tmp_path / "capture"
    clouds = capture / "segment_pcds"
    clouds.mkdir(parents=True)
    names = [
        "pointcloud_100.000000_segmented.pcd",
        "pointcloud_100.100000_segmented.pcd",
        "pointcloud_100.200000_segmented.pcd",
    ]
    for name in names:
        (clouds / name).write_bytes(b"x")
    import hashlib

    digest = hashlib.sha256(
        b"rgbbench-evaluation-point-cloud-names-v1\0"
        + "\n".join(names).encode("ascii")
    ).hexdigest()
    paths = evaluation_pcd_paths(
        capture,
        master_start_time_s=100.0,
        camera_delay_s=0.0,
        start_calculate_time_s=0.0,
        end_calculate_time_s=0.2,
        expected_count=3,
        expected_name_sha256=digest,
    )
    assert tuple(path.name for path in paths) == tuple(names)


def _write_mesh(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"v {index} 0 0\n" for index in range(128)]
    lines.append("f 1 2 3\n")
    path.write_text("".join(lines), encoding="utf-8")


def _write_capture(path: Path) -> None:
    joints = path / "joints"
    calibration = path / "calibration"
    clouds = path / "segment_pcds"
    joints.mkdir(parents=True)
    calibration.mkdir()
    clouds.mkdir()
    for arm in ("left", "right"):
        (joints / f"{arm}_arm_joint_states_and_end_pose.csv").write_text(
            "time,pos_x\n100.0,0.0\n",
            encoding="utf-8",
        )
    (calibration / "world_to_camera_transform.json").write_text(
        json.dumps(np.eye(4).tolist()),
        encoding="utf-8",
    )
    for index in range(24):
        (clouds / f"pointcloud_{100.0 + 0.1 * index:.6f}_segmented.pcd").write_bytes(
            b"outcome-not-parsed"
        )


def test_manifest_builder_reads_only_names_sizes_and_required_streams(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "dataset"
    benchmark = tmp_path / "benchmark"
    (benchmark / "configs").mkdir(parents=True)
    (benchmark / "results").mkdir()
    (benchmark / "configs/experiment_library.yaml").write_text(
        "experiments: {}\n",
        encoding="utf-8",
    )
    (benchmark / "results/paper_baselines.csv").write_text(
        "garment,action\n",
        encoding="utf-8",
    )
    mesh_paths: dict[str, str] = {}
    experiments: dict[str, object] = {"experiments": {}}
    experiment_root = experiments["experiments"]
    assert isinstance(experiment_root, dict)
    for garment in PAPER_GARMENTS:
        mesh_relative = f"{garment}/{garment}.obj"
        mesh_paths[garment] = mesh_relative
        _write_mesh(dataset / "meshes" / mesh_relative)
        garment_entry: dict[str, object] = {}
        experiment_root[garment] = garment_entry
        for action in ("fling", "fold", "grasp"):
            samples: dict[str, object] = {}
            garment_entry[action] = {"piper": samples}
            for sample in PRIMARY_SAMPLES:
                subfolder = f"{garment}/{garment}_{action}_{sample}"
                _write_capture(dataset / subfolder)
                samples[sample] = {
                    "camera_delay": 0.0,
                    "data_subfolder": subfolder,
                    "evaluate": {
                        "start_calculate_time": 0.0,
                        "end_calculate_time": 2.3,
                    },
                }
    manifest = build_rgbbench_dataset_manifest(
        dataset,
        benchmark,
        experiment_library=experiments,
        mesh_relative_paths=mesh_paths,
        dataset_revision=DATASET_REVISION,
    )
    assert len(manifest.cases) == 63
    assert sum(case.split == "source" for case in manifest.cases) == 27
    assert sum(case.split == "calibration" for case in manifest.cases) == 18
    assert sum(case.split == "target" for case in manifest.cases) == 18
    assert all(case.evaluation_frame_count == 24 for case in manifest.cases)
    assert manifest.descriptor()["information_boundary"]["point_coordinates_read"] is False
