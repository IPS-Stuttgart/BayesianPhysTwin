from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np


SCRIPT = (
    Path(__file__).parents[1]
    / "scripts"
    / "science"
    / "inventory_deform360_source_mask_carriers_v1.py"
)


def _write_calibration(root: Path, cameras: list[str]) -> None:
    intrinsics = {
        camera: np.array(
            [[100.0, 0.0, 4.0], [0.0, 100.0, 4.0], [0.0, 0.0, 1.0]]
        )
        for camera in cameras
    }
    extrinsics = {camera: np.eye(4) for camera in cameras}
    np.save(root / "undistorted_intrinsics.npy", intrinsics)
    np.save(root / "extrinsics.npy", extrinsics)


def _write_carrier(
    path: Path,
    *,
    shape: tuple[int, int, int] = (17, 8, 9),
    dtype: np.dtype | type = np.uint8,
) -> None:
    with h5py.File(path, "w") as handle:
        handle.create_dataset(
            "data",
            shape=shape,
            chunks=(1, shape[1], shape[2]),
            dtype=dtype,
            compression="gzip",
        )


def _make_episode(tmp_path: Path, camera_count: int) -> tuple[Path, list[str]]:
    root = tmp_path / "038-mat-cloth" / "episode_0003"
    root.mkdir(parents=True)
    cameras = [f"camera_{index:02d}" for index in range(camera_count)]
    _write_calibration(root, cameras)
    for camera in cameras:
        directory = root / camera
        directory.mkdir()
        (directory / "undistorted.mp4").write_bytes(b"header-only-test")
    return root, cameras


def _run_inventory(
    root: Path,
    output: Path,
    *,
    minimum_object_masks: int,
    minimum_heldout: int,
) -> dict:
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--source-episode-root",
            str(root),
            "--source-object",
            "038-mat-cloth",
            "--source-episode",
            "3",
            "--output-dir",
            str(output),
            "--minimum-object-mask-cameras",
            str(minimum_object_masks),
            "--minimum-heldout-cameras",
            str(minimum_heldout),
            "--repository",
            "test/repository",
            "--revision",
            "0" * 40,
            "--workflow-run-id",
            "1",
            "--workflow-run-attempt",
            "1",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads((output / "result.json").read_text())


def test_inventory_qualifies_compatible_headers_without_opening_frames(
    tmp_path: Path,
) -> None:
    root, cameras = _make_episode(tmp_path, 6)
    for camera in cameras[:5]:
        _write_carrier(root / camera / "mask_refined.h5")
    for camera in cameras[:2]:
        _write_carrier(
            root / camera / "rendered_urdf.h5", dtype=np.bool_
        )
    result = _run_inventory(
        root,
        tmp_path / "output",
        minimum_object_masks=3,
        minimum_heldout=2,
    )
    assert result["decision"] == "source-visual-hull-inputs-ready"
    assert result["aligned_calibrated_camera_count"] == 6
    assert result["object_mask_inventory"]["valid_header_count"] == 5
    assert result["gripper_mask_inventory"]["valid_header_count"] == 2
    assert result["information_boundary"][
        "source_mask_dataset_payload_frames_opened"
    ] == 0
    assert result["information_boundary"]["source_camera_pixels_opened"] is False
    assert result["information_boundary"]["target_numeric_payload_opened"] is False
    assert all(
        record["object_mask"]["dataset_payload_frames_opened"] == 0
        for record in result["camera_records"]
    )


def test_inventory_preserves_invalid_header_as_negative_evidence(
    tmp_path: Path,
) -> None:
    root, cameras = _make_episode(tmp_path, 4)
    _write_carrier(root / cameras[0] / "mask_refined.h5")
    with h5py.File(root / cameras[1] / "mask_refined.h5", "w") as handle:
        handle.create_dataset("wrong_name", shape=(17, 8, 9), dtype=np.uint8)
    _write_carrier(
        root / cameras[2] / "mask_refined.h5", dtype=np.float32
    )
    result = _run_inventory(
        root,
        tmp_path / "output",
        minimum_object_masks=3,
        minimum_heldout=1,
    )
    assert result["decision"] == "source-visual-hull-inputs-not-ready"
    assert result["object_mask_inventory"]["present_count"] == 3
    assert result["object_mask_inventory"]["valid_header_count"] == 1
    errors = [
        record["object_mask"].get("error")
        for record in result["camera_records"]
        if record["object_mask"].get("error")
    ]
    assert len(errors) == 2
    assert result["information_boundary"][
        "source_mask_dataset_payload_frames_opened"
    ] == 0
