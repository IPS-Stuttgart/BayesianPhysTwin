from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import pytest

import causal4d_public.deform360_sota_processing as sota_processing
from causal4d_public.deform360_reusable_sota_protocol import (
    load_reusable_sota_config,
)
from causal4d_public.deform360_reusable_sota_window import (
    authorize_development_fit_window,
    authorize_development_held_prediction_window,
    load_reusable_sota_window,
)
from causal4d_public.deform360_sota_processing import (
    DEVELOPMENT_MASK_PANEL_KIND,
    DEVELOPMENT_PROCESSING_STAGE_KIND,
    LEGACY_DEVELOPMENT_MASK_PANEL_KIND,
    authorize_development_processing,
    build_development_observations_manifest,
    load_development_reference_mask_panel,
    load_development_source_mask_panel,
    material_identity_sha256,
    propagate_development_masks,
    stage_development_processing_episode,
    validate_development_final_data_input,
    validate_development_held_prediction_stage,
    write_development_action_window_stage,
)


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs/causal4d_public/deform360_reusable_sota_v1.json"
WINDOW = ROOT / "configs/causal4d_public/deform360_reusable_sota_window_v1.json"


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

    held_authorization = authorize_development_processing(
        protocol,
        object_id="004-rubber-band",
        episode_id=0,
        role="held-development",
    )
    loaded_panel = load_development_reference_mask_panel(
        panel_path,
        authorization=held_authorization,
    )
    assert [record["camera"] for record in loaded_panel["records"]] == cameras

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


def test_source_mask_panel_slice_is_fit_only_and_matches_frozen_cameras(
    tmp_path: Path,
) -> None:
    protocol = load_reusable_sota_config(PROTOCOL)
    authorization = authorize_development_processing(
        protocol, object_id="004-rubber-band", episode_id=1, role="fit"
    )
    cameras = [f"camera-{index}" for index in range(3)]
    panel = {
        "schema_version": 1,
        "artifact_kind": LEGACY_DEVELOPMENT_MASK_PANEL_KIND,
        "protocol_id": authorization["protocol_id"],
        "object_id": authorization["object_id"],
        "episode_id": authorization["episode_id"],
        "role": "development-fit",
        "accepted_camera_count": len(cameras),
        "records": [
            {
                "camera": camera,
                "frame_count": 317,
                "output_sha256": f"{index + 1:064x}",
            }
            for index, camera in enumerate(cameras)
        ],
        "information_boundary": {
            "confirmatory_object_opened": False,
            "held_episode_opened": False,
            "fit_episode_future_frames_used_only_for_source_annotation": True,
        },
    }
    panel["result_sha256"] = _canonical_sha256(panel)
    panel_path = tmp_path / "mask_panel.json"
    panel_path.write_text(json.dumps(panel))

    loaded = load_development_source_mask_panel(
        panel_path,
        authorization=authorization,
        reference_cameras=cameras,
        start_frame=122,
        frame_count=81,
    )
    assert loaded["result_sha256"] == panel["result_sha256"]

    held = authorize_development_processing(
        protocol,
        object_id="004-rubber-band",
        episode_id=0,
        role="held-development",
    )
    with pytest.raises(ValueError, match="fit episodes"):
        load_development_source_mask_panel(
            panel_path,
            authorization=held,
            reference_cameras=cameras,
            start_frame=122,
            frame_count=81,
        )
    with pytest.raises(ValueError, match="frozen reference camera"):
        load_development_source_mask_panel(
            panel_path,
            authorization=authorization,
            reference_cameras=[*cameras[:-1], "other-camera"],
            start_frame=122,
            frame_count=81,
        )


def test_action_window_stage_is_compatible_with_sota_observation_runner(
    tmp_path: Path,
) -> None:
    protocol = load_reusable_sota_config(PROTOCOL)
    window = load_reusable_sota_window(WINDOW)
    authorization = authorize_development_processing(
        protocol, object_id="004-rubber-band", episode_id=3, role="fit"
    )
    window_authorization = authorize_development_fit_window(
        protocol, window, object_id="004-rubber-band", episode_id=3
    )
    output = tmp_path / "development_staging.json"
    result = write_development_action_window_stage(
        output,
        authorization=authorization,
        window_authorization=window_authorization,
        selected_raw_frame_range_half_open=(26, 107),
        camera_count=3,
        frame_count=81,
        window_config_sha256=window["config_sha256"],
        mask_diagnostics_sha256="a" * 64,
        initialization_diagnostics_sha256="b" * 64,
    )
    assert output.is_file()
    assert result["artifact_kind"] == DEVELOPMENT_PROCESSING_STAGE_KIND
    assert result["authorization"] == authorization
    assert result["temporal_staging"]["selected_raw_frame_range_half_open"] == [
        26,
        107,
    ]
    assert (
        result["information_boundary"][
            "window_selection_used_object_geometry_or_tactile"
        ]
        is False
    )

    altered = copy.deepcopy(window_authorization)
    altered["held_outcome_read"] = True
    with pytest.raises(ValueError, match="window authorization"):
        write_development_action_window_stage(
            tmp_path / "invalid.json",
            authorization=authorization,
            window_authorization=altered,
            selected_raw_frame_range_half_open=(26, 107),
            camera_count=3,
            frame_count=81,
            window_config_sha256=window["config_sha256"],
            mask_diagnostics_sha256="a" * 64,
            initialization_diagnostics_sha256="b" * 64,
        )


def test_held_action_window_stage_exposes_only_frame_zero(tmp_path: Path) -> None:
    protocol = load_reusable_sota_config(PROTOCOL)
    window = load_reusable_sota_window(WINDOW)
    authorization = authorize_development_processing(
        protocol,
        object_id="004-rubber-band",
        episode_id=0,
        role="held-development",
    )
    window_authorization = authorize_development_held_prediction_window(
        protocol, window, object_id="004-rubber-band", episode_id=0
    )
    output = tmp_path / "development_staging.json"
    result = write_development_action_window_stage(
        output,
        authorization=authorization,
        window_authorization=window_authorization,
        selected_raw_frame_range_half_open=(122, 203),
        camera_count=3,
        frame_count=1,
        known_robot_action_frame_count=81,
        window_config_sha256=window["config_sha256"],
        mask_diagnostics_sha256="a" * 64,
        initialization_diagnostics_sha256="b" * 64,
    )
    assert result["role"] == "held-development"
    assert result["frame_count"] == 1
    assert result["information_boundary"]["object_observation_frames_used"] == [0]
    assert result["information_boundary"]["future_object_outcome_read"] is False
    validation = validate_development_held_prediction_stage(
        output,
        authorization=authorization,
        window_authorization=window_authorization,
    )
    assert validation["passed"] is True
    assert validation["known_robot_action_frame_count"] == 81

    with pytest.raises(ValueError, match="invalid action window"):
        write_development_action_window_stage(
            tmp_path / "invalid-held.json",
            authorization=authorization,
            window_authorization=window_authorization,
            selected_raw_frame_range_half_open=(122, 203),
            camera_count=3,
            frame_count=81,
            known_robot_action_frame_count=81,
            window_config_sha256=window["config_sha256"],
            mask_diagnostics_sha256="a" * 64,
            initialization_diagnostics_sha256="b" * 64,
        )


def test_development_observation_manifest_binds_ordered_material_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    protocol = load_reusable_sota_config(PROTOCOL)
    authorization = authorize_development_processing(
        protocol, object_id="004-rubber-band", episode_id=3, role="fit"
    )
    processing = tmp_path / "processing"
    episode = processing / "004-rubber-band" / "episode_0003"
    episode.mkdir(parents=True)
    staging = {
        "schema_version": 1,
        "artifact_kind": DEVELOPMENT_PROCESSING_STAGE_KIND,
        "authorization": authorization,
        "object_id": "004-rubber-band",
        "episode_id": 3,
        "role": "fit",
        "camera_count": 2,
        "frame_count": 7,
        "mask_panel_result_sha256": "a" * 64,
        "mask_panel_file_sha256": "b" * 64,
        "information_boundary": {
            "development_only": True,
            "confirmatory_object_opened": False,
            "target_metric_read": False,
        },
        "claim_boundary": "fixture",
    }
    staging["result_sha256"] = _canonical_sha256(staging)
    (episode / "development_staging.json").write_text(json.dumps(staging))

    splat_dir = episode / "splatfacto"
    splat_dir.mkdir()
    for frame in range(7):
        (splat_dir / f"splat_{frame}.ply").write_bytes(f"splat-{frame}".encode())
    (splat_dir / "splatfacto.meta.json").write_text("{}")
    for camera in ("cam-a", "cam-b"):
        camera_dir = episode / camera
        tracking = camera_dir / "tracking"
        tracking.mkdir(parents=True)
        (camera_dir / "mask_refined.h5").write_bytes(b"mask")
        (camera_dir / "rendered_urdf.h5").write_bytes(b"gripper-mask")
        (camera_dir / "rendered_urdf.meta.json").write_text("{}")
        (camera_dir / "rendered_depth.h5").write_bytes(b"depth")
        (camera_dir / "rendered_depth.meta.json").write_text("{}")
        (tracking / "vel.h5").write_bytes(b"velocity")
        (tracking / "visibility.h5").write_bytes(b"visibility")
        (tracking / "tracking.meta.json").write_text("{}")

    point_dir = episode / "pcd_clean"
    point_dir.mkdir()
    points = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]], dtype=np.float32)
    np.savez(point_dir / "000000.npz", pts=points)
    np.savez(point_dir / "000001.npz", pts=points + 0.1)
    (point_dir / "pcd_clean.meta.json").write_text("{}")
    for name in (
        "calibrate.pkl",
        "start_obj_pcd.ply",
        "split.json",
        "final_data.pkl",
        "control_points.meta.json",
    ):
        (episode / name).write_bytes(name.encode())

    checkpoint = tmp_path / "scaled_offline.pth"
    checkpoint.write_bytes(b"checkpoint")
    checkpoint_sha = _file_sha256(checkpoint)
    monkeypatch.setattr(
        sota_processing, "PINNED_DEFORM360_PROCESSING_REVISION", "deform-revision"
    )
    monkeypatch.setattr(
        sota_processing, "PINNED_COTRACKER_REVISION", "cotracker-revision"
    )
    monkeypatch.setattr(
        sota_processing, "PINNED_COTRACKER_CHECKPOINT_SHA256", checkpoint_sha
    )

    result = build_development_observations_manifest(
        authorization=authorization,
        processing_root=processing,
        deform360_processing_revision="deform-revision",
        cotracker_revision="cotracker-revision",
        cotracker_checkpoint=checkpoint,
    )

    assert result["point_frame_count"] == 2
    assert result["material_point_count"] == 2
    assert result["material_identity_sha256"] == material_identity_sha256(points)
    assert set(result["output_sha256"]["gripper_masks"]) == {"cam-a", "cam-b"}

    final_data = episode / "final_data.pkl"
    validation = validate_development_final_data_input(
        result,
        authorization=authorization,
        final_data_path=final_data,
    )
    assert validation["passed"] is True
    assert validation["point_frame_count"] == 2
    assert validation["held_outcome_read"] is False

    final_data.write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksum changed"):
        validate_development_final_data_input(
            result,
            authorization=authorization,
            final_data_path=final_data,
        )
    assert material_identity_sha256(points[::-1]) != material_identity_sha256(points)
    assert result["information_boundary"]["prediction_metric_computed"] is False

    (point_dir / "000001.npz").unlink()
    with pytest.raises(ValueError, match="point-cloud output is incomplete"):
        build_development_observations_manifest(
            authorization=authorization,
            processing_root=processing,
            deform360_processing_revision="deform-revision",
            cotracker_revision="cotracker-revision",
            cotracker_checkpoint=checkpoint,
        )
