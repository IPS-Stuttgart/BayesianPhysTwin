from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from bayesian_phystwin import deform360_dynamic_tapnextpp_provider as provider
from bayesian_phystwin.deform360_dynamic_tapnextpp_admission_v2 import (
    CAUSAL_FRAME_COUNT,
    CameraPrefixEvidence,
    load_complete_camera_geometry,
    load_selected_complete_causal_inputs,
)


def _digest(character: str) -> str:
    return character * 64


def _episode(tmp_path: Path, camera_count: int = 9) -> tuple[Path, tuple[str, ...]]:
    names = tuple(f"camera-{index:02d}" for index in range(camera_count))
    intrinsics = {name: np.eye(3) for name in names}
    poses = {name: np.eye(4) for name in names}
    np.save(
        tmp_path / "undistorted_intrinsics.npy",
        np.asarray(intrinsics, dtype=object),
        allow_pickle=True,
    )
    np.save(
        tmp_path / "extrinsics.npy",
        np.asarray(poses, dtype=object),
        allow_pickle=True,
    )
    for name in names:
        camera = tmp_path / name
        camera.mkdir()
        for filename in (
            "undistorted.mp4",
            "rendered_depth.h5",
            "mask_refined.h5",
        ):
            (camera / filename).write_bytes(filename.encode("ascii"))
    return tmp_path, names


def _probe(camera_dir: Path, frame_count: int) -> CameraPrefixEvidence:
    assert frame_count == CAUSAL_FRAME_COUNT
    index = int(camera_dir.name.rsplit("-", maxsplit=1)[-1])
    token = format(index % 16, "x")
    return CameraPrefixEvidence(
        image_shape_hw=(48, 64),
        rgb_prefix_sha256=_digest(token),
        depth_prefix_sha256=_digest("a"),
        mask_prefix_sha256=_digest("b"),
    )


def test_loader_uses_complete_camera_subset_and_preserves_panel_indices(
    tmp_path: Path,
) -> None:
    episode, names = _episode(tmp_path)
    (episode / names[4] / "rendered_depth.h5").unlink()

    geometry = load_complete_camera_geometry(
        episode,
        candidate_camera_names=names,
        prefix_probe=_probe,
    )

    assert geometry.camera_names == names[:4] + names[5:]
    np.testing.assert_array_equal(
        geometry.frozen_panel_indices,
        np.asarray([0, 1, 2, 3, 5, 6, 7, 8]),
    )
    assert geometry.rejected_cameras == {names[4]: "missing_causal_stream"}
    assert geometry.intrinsics.shape == (8, 3, 3)
    assert geometry.camera_to_world.shape == (8, 4, 4)
    assert geometry.image_shapes_hw.tolist() == [[48, 64]] * 8
    assert geometry.descriptor()["information_boundary"] == {
        "maximum_rgb_depth_mask_frame_read": 57,
        "future_object_observation_read": False,
        "target_metric_read": False,
    }


def test_loader_rejects_fewer_than_eight_complete_cameras(tmp_path: Path) -> None:
    episode, names = _episode(tmp_path, camera_count=8)
    (episode / names[-1] / "mask_refined.h5").unlink()
    with pytest.raises(ValueError, match="too few complete"):
        load_complete_camera_geometry(
            episode,
            candidate_camera_names=names,
            prefix_probe=_probe,
        )


def test_loader_records_invalid_prefix_without_using_future_frames(
    tmp_path: Path,
) -> None:
    episode, names = _episode(tmp_path)

    def rejecting_probe(
        camera_dir: Path,
        frame_count: int,
    ) -> CameraPrefixEvidence:
        if camera_dir.name == names[0]:
            raise ValueError("prefix is short")
        return _probe(camera_dir, frame_count)

    geometry = load_complete_camera_geometry(
        episode,
        candidate_camera_names=names,
        prefix_probe=rejecting_probe,
    )
    assert geometry.rejected_cameras == {
        names[0]: "invalid_causal_prefix:ValueError"
    }
    assert names[0] not in geometry.camera_names


def test_selected_inputs_decode_only_certified_cameras(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    episode, names = _episode(tmp_path)
    geometry = load_complete_camera_geometry(
        episode,
        candidate_camera_names=names,
        prefix_probe=_probe,
    )

    def decode_rgb(path: Path, frame_count: int) -> np.ndarray:
        assert path.parent.name in geometry.camera_names
        assert frame_count == CAUSAL_FRAME_COUNT
        return np.zeros((frame_count, 4, 5, 3), dtype=np.uint8)

    def read_h5(path: Path, frame_count: int) -> np.ndarray:
        assert path.parent.name in geometry.camera_names
        assert frame_count == CAUSAL_FRAME_COUNT
        return np.ones((frame_count, 4, 5), dtype=np.uint16)

    monkeypatch.setattr(provider, "_decode_rgb_prefix", decode_rgb)
    monkeypatch.setattr(provider, "_read_h5_prefix", read_h5)
    inputs = load_selected_complete_causal_inputs(
        episode,
        geometry,
        np.arange(8),
    )

    assert inputs.camera_names == geometry.camera_names[:8]
    assert inputs.rgbs.shape == (8, 58, 4, 5, 3)
    assert inputs.depths_m.shape == (8, 58, 4, 5)
    assert inputs.provenance["maximum_frame_read"] == 57
    assert (
        inputs.provenance["complete_camera_certificate_sha256"]
        == geometry.artifact_sha256
    )
