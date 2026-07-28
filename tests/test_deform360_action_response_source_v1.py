from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType


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
