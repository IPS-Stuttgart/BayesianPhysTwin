from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import numpy as np


def _load_runner() -> ModuleType:
    script = (
        Path(__file__).parents[1]
        / "scripts"
        / "remote"
        / "run_deform360_action_response_source_v1.py"
    )
    spec = importlib.util.spec_from_file_location(
        "run_deform360_action_response_source_v1",
        script,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load action-response source runner")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_prefix_calibration_excludes_incomplete_camera_assets(
    tmp_path: Path,
) -> None:
    module = _load_runner()
    intrinsics = {
        "complete-a": object(),
        "complete-b": object(),
        "missing-depth": object(),
    }
    extrinsics = {
        "complete-a": object(),
        "complete-b": object(),
        "missing-depth": object(),
    }
    for camera in intrinsics:
        camera_dir = tmp_path / camera
        camera_dir.mkdir()
        (camera_dir / "undistorted.mp4").touch()
        (camera_dir / "mask_refined.h5").touch()
    (tmp_path / "complete-a" / "rendered_depth.h5").touch()
    (tmp_path / "complete-b" / "rendered_depth.h5").touch()

    filtered_intrinsics, filtered_extrinsics, cameras = (
        module._eligible_prefix_calibration(
            tmp_path,
            intrinsics,
            extrinsics,
        )
    )

    assert cameras == ("complete-a", "complete-b")
    assert tuple(filtered_intrinsics) == cameras
    assert tuple(filtered_extrinsics) == cameras


def test_legacy_planner_remains_the_default() -> None:
    module = _load_runner()

    parsed = module._parser().parse_args(
        [
            "--physical-dir",
            "physical",
            "--processed-dir",
            "processed",
            "--alltracker-source",
            "alltracker",
            "--checkpoint",
            "checkpoint",
            "--output-dir",
            "output",
            "--repository-revision",
            "revision",
        ]
    )

    assert parsed.planner == module.LEGACY_PLANNER


def test_projected_view_planner_is_explicitly_opt_in() -> None:
    module = _load_runner()

    parsed = module._parser().parse_args(
        [
            "--physical-dir",
            "physical",
            "--processed-dir",
            "processed",
            "--alltracker-source",
            "alltracker",
            "--checkpoint",
            "checkpoint",
            "--output-dir",
            "output",
            "--repository-revision",
            "revision",
            "--planner",
            module.PROJECTED_VIEW_PLANNER,
        ]
    )

    assert parsed.planner == module.PROJECTED_VIEW_PLANNER
    assert module.PROTOCOL_IDS[parsed.planner].endswith("-v3")


def test_named_array_digest_binds_values_and_labels() -> None:
    module = _load_runner()
    first = module._named_arrays_sha256(
        {
            "a": np.asarray([1.0, 2.0]),
            "b": np.asarray([3], dtype=np.int64),
        }
    )
    reordered = module._named_arrays_sha256(
        {
            "b": np.asarray([3], dtype=np.int64),
            "a": np.asarray([1.0, 2.0]),
        }
    )
    changed_value = module._named_arrays_sha256(
        {
            "a": np.asarray([1.0, 2.1]),
            "b": np.asarray([3], dtype=np.int64),
        }
    )
    changed_label = module._named_arrays_sha256(
        {
            "a": np.asarray([1.0, 2.0]),
            "c": np.asarray([3], dtype=np.int64),
        }
    )

    assert first == reordered
    assert first != changed_value
    assert first != changed_label


def test_group_support_count_uses_cartesian_node_camera_indexing() -> None:
    module = _load_runner()
    support = np.asarray(
        [
            [True, True, False, False],
            [True, True, True, True],
            [False, False, True, True],
        ],
        dtype=bool,
    )

    counts = module._group_frame_zero_support_counts(
        support,
        np.asarray([0, 1, 2]),
        ("a", "b", "c", "d"),
        (("a", "b"), ("c", "d")),
        minimum_cameras_per_group=2,
    )

    assert counts == [2, 2]
