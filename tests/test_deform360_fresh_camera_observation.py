from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np

from bayesian_phystwin.deform360_fresh_camera_observation import (
    build_fresh_raw_camera_measurement_case_with_contract,
    materialized_calibrated_camera_names,
)


def test_materialized_camera_panel_is_sorted_calibrated_intersection(
    tmp_path: Path,
) -> None:
    cameras = ("camera-2", "camera-0", "camera-1", "camera-uncalibrated")
    np.save(
        tmp_path / "undistorted_intrinsics.npy",
        {camera: np.eye(3) for camera in cameras[:-1]},
    )
    np.save(
        tmp_path / "extrinsics.npy",
        {camera: np.eye(4) for camera in cameras[:-1]},
    )
    for camera in cameras:
        camera_dir = tmp_path / camera
        camera_dir.mkdir()
        (camera_dir / "undistorted.mp4").touch()
        (camera_dir / "mask_refined.h5").touch()
        (camera_dir / "rendered_depth.h5").touch()
    (tmp_path / "camera-1" / "rendered_depth.h5").unlink()

    result = materialized_calibrated_camera_names(tmp_path)

    assert result == ("camera-0", "camera-2")


def test_fresh_camera_adapter_has_no_target_or_outcome_argument() -> None:
    parameters = inspect.signature(
        build_fresh_raw_camera_measurement_case_with_contract
    ).parameters

    assert "target" not in parameters
    assert "outcome" not in parameters
