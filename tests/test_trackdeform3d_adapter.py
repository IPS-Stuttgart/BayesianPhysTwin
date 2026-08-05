from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from bayesian_phystwin.cli.trackdeform3d_sample import build_sample_manifest
from bayesian_phystwin.trackdeform3d_adapter import (
    deterministic_observed_identity_ids,
    inspect_trackdeform3d_chunk,
)


def _write_pose_archive(path: Path, frame_count: int) -> None:
    np.savez(
        path,
        **{
            f"arr_{index}": np.asarray(
                [index, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                dtype=np.float64,
            )
            for index in range(frame_count)
        },
    )


def _write_chunk(root: Path, *, pose_frame_count: int = 4) -> tuple[Path, Path]:
    chunk = root / "dlo" / "chunk_1"
    (chunk / "masks").mkdir(parents=True)
    calibration = root / "dlo" / "calibration" / "transform_ee_cam_world.npz"
    calibration.parent.mkdir(parents=True)
    np.savez(
        chunk / "rgbd.npz",
        color=np.zeros((4, 5, 6, 3), dtype=np.uint8),
        depth=np.zeros((4, 5, 6), dtype=np.uint16),
    )
    np.savez(
        chunk / "masks" / "masks.npz",
        masks=np.ones((4, 5, 6), dtype=np.uint8),
    )
    _write_pose_archive(chunk / "left_arm_poses.npz", pose_frame_count)
    _write_pose_archive(chunk / "right_arm_poses.npz", pose_frame_count)
    np.savez(
        calibration,
        K=np.eye(3, dtype=np.float64),
        T_left_base2cam=np.eye(4, dtype=np.float64),
        T_right_base2cam=np.eye(4, dtype=np.float64),
    )
    return chunk, calibration


def test_trackdeform3d_admission_uses_headers_and_hashes(tmp_path: Path) -> None:
    chunk, calibration = _write_chunk(tmp_path)

    result = inspect_trackdeform3d_chunk(
        chunk,
        calibration,
        object_kind="dlo",
    )

    assert result.frame_count == 4
    assert result.image_height == 5
    assert result.image_width == 6
    assert result.mask_relative_path == "masks/masks.npz"
    assert len(result.rgbd_sha256) == 64
    assert result.information_boundary == {
        "rgbd_values_decoded": False,
        "mask_values_decoded": False,
        "pose_values_decoded": False,
        "keypoint_trajectories_read": False,
        "future_outcomes_read": False,
    }
    assert {member.name for member in result.rgbd_members} == {"color", "depth"}


def test_trackdeform3d_admission_rejects_frame_count_mismatch(tmp_path: Path) -> None:
    chunk, calibration = _write_chunk(tmp_path, pose_frame_count=3)

    with pytest.raises(ValueError, match="left pose frame count changed"):
        inspect_trackdeform3d_chunk(chunk, calibration, object_kind="dlo")


def test_observed_identity_split_is_deterministic_and_disjoint() -> None:
    points = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [0.1, 0.0, 0.0],
            [0.2, 0.0, 0.0],
            [0.3, 0.0, 0.0],
            [0.4, 0.0, 0.0],
        ]
    )

    first = deterministic_observed_identity_ids(points, 2)
    second = deterministic_observed_identity_ids(points, 2)
    hidden = np.setdiff1d(np.arange(len(points)), first)

    np.testing.assert_array_equal(first, second)
    assert len(np.unique(first)) == 2
    assert not set(first.tolist()) & set(hidden.tolist())


def test_sample_manifest_binds_all_four_chunks(tmp_path: Path) -> None:
    for object_kind, chunk_name in (
        ("dlo", "chunk_1"),
        ("bdlo", "chunk_7"),
        ("fabric", "chunk_14"),
        ("cloth", "chunk_0"),
    ):
        source_chunk, source_calibration = _write_chunk(tmp_path / object_kind)
        target_chunk = tmp_path / object_kind / chunk_name
        target_chunk.parent.mkdir(parents=True, exist_ok=True)
        source_chunk.rename(target_chunk)
        target_calibration = (
            tmp_path / object_kind / "calibration" / "transform_ee_cam_world.npz"
        )
        target_calibration.parent.mkdir(parents=True, exist_ok=True)
        source_calibration.rename(target_calibration)
        if object_kind == "fabric":
            (target_chunk / "masks" / "masks.npz").rename(target_chunk / "fg_mask.npz")
            with np.load(target_chunk / "fg_mask.npz", allow_pickle=False) as stored:
                masks = np.asarray(stored["masks"])
            np.savez(target_chunk / "fg_mask.npz", fg_mask=masks)
        elif object_kind == "cloth":
            (target_chunk / "fg_masks").mkdir()
            (target_chunk / "masks" / "masks.npz").rename(
                target_chunk / "fg_masks" / "masks.npz"
            )

    manifest = build_sample_manifest(tmp_path, upstream_revision="a" * 40)

    assert manifest["admitted_count"] == 4
    assert len(manifest["manifest_sha256"]) == 64
    assert manifest["information_boundary"]["future_outcomes_read"] is False
