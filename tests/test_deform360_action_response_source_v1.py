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


def test_projected_observability_panel_is_opt_in_and_excludes_v3_case() -> None:
    module = _load_runner()
    case = module.PROJECTED_SOURCE_PANEL_CASES[0]

    parsed = module._parser().parse_args(
        [
            "--case",
            case,
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
            module.PROJECTED_OBSERVABILITY_PLANNER,
        ]
    )

    assert parsed.case == case
    assert parsed.planner == module.PROJECTED_OBSERVABILITY_PLANNER
    assert module.EXPECTED_CASE not in module.PROJECTED_SOURCE_PANEL_CASES
    assert len(module.PROJECTED_SOURCE_PANEL_CASES) == 7
    assert len(set(module.PROJECTED_SOURCE_PANEL_CASES)) == 7
    lock_path, lock = module._load_projected_source_lock()
    assert lock_path.is_file()
    assert tuple(
        entry["case"] for entry in lock["source_cases"]
    ) == module.PROJECTED_SOURCE_PANEL_CASES


def test_project_physical_prefix_preserves_metric_depth_and_shape() -> None:
    module = _load_runner()
    physical = np.asarray(
        [
            [[0.0, 0.0, 2.0], [0.2, 0.0, 2.0]],
            [[0.0, 0.1, 2.0], [0.2, 0.1, 2.0]],
            [[0.0, 0.2, 2.0], [0.2, 0.2, 2.0]],
        ]
    )
    intrinsic = np.asarray(
        [[500.0, 0.0, 320.0], [0.0, 400.0, 240.0], [0.0, 0.0, 1.0]]
    )
    extrinsic = np.eye(4)

    pixels, depth, focal = module._project_physical_prefix(
        physical,
        np.asarray([0, 1, 2]),
        ("camera",),
        {"camera": intrinsic},
        {"camera": extrinsic},
        minimum_initial_depth_m=0.05,
    )

    assert pixels.shape == (1, 3, 2, 2)
    np.testing.assert_allclose(depth, 2.0)
    np.testing.assert_allclose(focal, [[500.0, 400.0]])
    np.testing.assert_allclose(pixels[0, :, 0, 1], [240.0, 260.0, 280.0])


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
