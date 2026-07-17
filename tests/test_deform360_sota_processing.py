from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from causal4d_public.deform360_reusable_sota_protocol import (
    load_reusable_sota_config,
)
from causal4d_public.deform360_sota_processing import (
    DEVELOPMENT_MASK_PANEL_KIND,
    authorize_development_processing,
    propagate_development_masks,
    stage_development_processing_episode,
)


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs/causal4d_public/deform360_reusable_sota_v1.json"


def _canonical_sha256(payload: dict[str, object]) -> str:
    canonical = dict(payload)
    canonical.pop("result_sha256", None)
    encoded = json.dumps(
        canonical, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_h5(path: Path, masks: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as stream:
        stream.create_dataset("data", data=masks.astype(np.uint8))


class _FakePredictor:
    def select_initial_mask_with_reference(
        self, video_path, reference_rgb, reference_mask, *, reference_camera
    ):
        assert video_path.name == "undistorted.mp4"
        assert reference_rgb.shape == (4, 5, 3)
        return reference_mask.copy(), {"camera": reference_camera, "score": 0.9}

    def segment_from_initial_mask(
        self, video_path, initial_mask, *, initialization=None
    ):
        assert initialization["future_object_observations_used"] is False
        for frame in range(3):
            yield frame, np.roll(initial_mask, frame, axis=1)


def test_development_processing_refuses_confirmatory_and_split_mismatch() -> None:
    protocol = load_reusable_sota_config(PROTOCOL)
    with pytest.raises(ValueError, match="confirmatory"):
        authorize_development_processing(
            protocol, object_id="090-sloth", episode_id=1, role="fit"
        )
    with pytest.raises(ValueError, match="outside the fit split"):
        authorize_development_processing(
            protocol, object_id="004-rubber-band", episode_id=0, role="fit"
        )


def test_propagate_and_stage_development_masks(tmp_path: Path) -> None:
    protocol = load_reusable_sota_config(PROTOCOL)
    authorization = authorize_development_processing(
        protocol, object_id="004-rubber-band", episode_id=3, role="fit"
    )
    aligned = tmp_path / "aligned" / "004-rubber-band"
    annotations = tmp_path / "annotations"
    reference_annotations = tmp_path / "reference"
    cameras = [f"camera-{index}" for index in range(3)]
    for episode_id in (1, 3):
        episode = aligned / f"episode_{episode_id:04d}"
        for camera in cameras:
            (episode / camera).mkdir(parents=True)
            (episode / camera / "undistorted.mp4").write_bytes(b"video")
    reference_mask = np.zeros((4, 5), dtype=np.uint8)
    reference_mask[1:3, 1:4] = 1
    reference_files = {}
    for camera in cameras:
        reference_h5 = (
            reference_annotations
            / "004-rubber-band"
            / "episode_0001"
            / camera
            / "mask_refined.h5"
        )
        _write_h5(reference_h5, reference_mask[None])
        reference_files[camera] = reference_h5
    panel = {
        "schema_version": 1,
        "artifact_kind": "Deform360DevelopmentSam2MaskPanel",
        "protocol_id": "deform360-reusable-sota-v1",
        "object_id": "004-rubber-band",
        "episode_id": 1,
        "role": "development-fit",
        "accepted_camera_count": 3,
        "records": [
            {
                "camera": camera,
                "output_sha256": _file_sha256(reference_files[camera]),
            }
            for camera in cameras
        ],
    }
    panel["result_sha256"] = _canonical_sha256(panel)
    panel_path = tmp_path / "reference-panel.json"
    panel_path.write_text(json.dumps(panel))

    result = propagate_development_masks(
        authorization=authorization,
        aligned_object_root=aligned,
        reference_annotation_root=reference_annotations,
        reference_panel_path=panel_path,
        output_annotation_root=annotations,
        predictor=_FakePredictor(),
        first_frame_reader=lambda _: np.zeros((4, 5, 3), dtype=np.uint8),
    )
    assert result["artifact_kind"] == DEVELOPMENT_MASK_PANEL_KIND
    assert result["frame_count"] == 3
    assert result["information_boundary"]["future_object_outcome_metric_used"] is False

    episode = aligned / "episode_0003"
    for name in ("alignment.json", "extrinsics.npy", "undistorted_intrinsics.npy"):
        (episode / name).write_bytes(name.encode())
    (episode / "robot").mkdir()
    for camera in cameras:
        for name in ("aligned_timestamps.txt", "alignment.json", "metadata.json"):
            (episode / camera / name).write_bytes(name.encode())

    staging = stage_development_processing_episode(
        authorization=authorization,
        aligned_object_root=aligned,
        annotation_root=annotations,
        processing_root=tmp_path / "processing",
    )
    output = tmp_path / "processing" / "004-rubber-band" / "episode_0003"
    assert staging["camera_count"] == 3
    assert (output / cameras[0] / "mask_refined.h5").is_symlink()
    assert (output / "robot").is_symlink()


def test_staging_rejects_authorization_substitution(tmp_path: Path) -> None:
    protocol = load_reusable_sota_config(PROTOCOL)
    authorization = authorize_development_processing(
        protocol, object_id="004-rubber-band", episode_id=3, role="fit"
    )
    altered = copy.deepcopy(authorization)
    altered["episode_id"] = 4
    with pytest.raises((FileNotFoundError, ValueError)):
        stage_development_processing_episode(
            authorization=altered,
            aligned_object_root=tmp_path,
            annotation_root=tmp_path,
            processing_root=tmp_path / "processing",
        )
