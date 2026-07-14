from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from causal4d_public.deform360 import (
    PINNED_DEFORM360_CODE_COMMIT,
    PINNED_DEFORM360_DATASET_REVISION,
    load_deform360_protocol_config,
    preflight_deform360_001_rope,
    validate_deform360_preflight,
)


def _config_path() -> Path:
    return (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "causal4d_public"
        / "deform360_001_rope_v1.json"
    )


def _write_fixture(root: Path) -> None:
    root.mkdir(parents=True)
    sequences = {
        str(index): {
            "action": f"fixture action {index}",
            "bimanual": "yes" if index >= 5 else "no",
            "nonprehensile": "yes" if index == 7 else "no",
        }
        for index in range(10)
    }
    (root / "metadata.json").write_text(
        json.dumps({"object": "001-rope", "sequences": sequences}) + "\n",
        encoding="utf-8",
    )
    camera_names = [f"brics-odroid-{index:03d}_cam0" for index in range(41)]
    for camera in camera_names:
        stream = root / camera
        stream.mkdir()
        for episode in range(10):
            stem = f"{camera}_{1_000_000 + episode}"
            (stream / f"{stem}.mp4").write_bytes(b"")
            (stream / f"{stem}.txt").write_text("0 0\n", encoding="ascii")
    sensor_names = (
        "brics-odroid_tactilel_left",
        "brics-odroid_tactilel_right",
        "brics-odroid_tactiler_left",
        "brics-odroid_tactiler_right",
    )
    for sensor in sensor_names:
        stream = root / sensor
        stream.mkdir()
        np.save(stream / "median_999.npy", np.zeros((16, 32), dtype=np.float32))
        for episode in range(10):
            stem = f"{sensor}_{1_000_000 + episode}"
            np.save(stream / f"{stem}.npy", np.zeros((2, 16, 32), dtype=np.float32))
            (stream / f"{stem}.txt").write_text("0 0\n1 1\n", encoding="ascii")
    calibration = root / "calibration_refined"
    calibration.mkdir()
    calibrated = camera_names[:36]
    np.save(
        calibration / "intrinsics.npy",
        {camera: np.eye(3) for camera in calibrated},
        allow_pickle=True,
    )
    np.save(
        calibration / "extrinsics.npy",
        {camera: np.eye(4) for camera in calibrated},
        allow_pickle=True,
    )
    np.save(
        calibration / "dist.npy",
        {camera: np.zeros(4) for camera in calibrated},
        allow_pickle=True,
    )


def _write_target_processed_fixture(root: Path) -> None:
    episode = root / "episode_0006"
    episode.mkdir(parents=True)
    (episode / "alignment.json").write_text(
        json.dumps(
            {
                "anchor_camera": "fixture_cam0",
                "frame_count": 8,
                "timeline_sha256": "0" * 64,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    for sensor in (
        "brics-odroid_tactilel_left",
        "brics-odroid_tactilel_right",
        "brics-odroid_tactiler_left",
        "brics-odroid_tactiler_right",
    ):
        sensor_dir = episode / sensor
        sensor_dir.mkdir()
        values = np.zeros((8, 16, 32), dtype=np.float32)
        values[3:6, 0, 0] = 1.0
        np.save(sensor_dir / "synced_tactile.npy", values)


def test_protocol_is_revision_and_split_locked() -> None:
    config = load_deform360_protocol_config(_config_path())

    assert config.code_commit == PINNED_DEFORM360_CODE_COMMIT
    assert config.dataset_revision == PINNED_DEFORM360_DATASET_REVISION
    assert config.source_episode_ids == (0, 2, 3, 4, 5, 7, 8)
    assert config.calibration_episode_ids == (1, 9)
    assert config.target_episode_ids == (6,)


def test_raw_fixture_passes_and_holds_out_action_six(tmp_path: Path) -> None:
    raw = tmp_path / "001-rope"
    _write_fixture(raw)
    config = load_deform360_protocol_config(_config_path())

    result = preflight_deform360_001_rope(raw, config)
    validation = validate_deform360_preflight(result)

    assert validation["passed"] is True
    assert result["preflight_passed"] is True
    assert result["raw_inventory"]["file_count"] == 908
    assert result["raw_inventory"]["camera_recordings_complete"] is True
    assert result["raw_inventory"]["tactile_recordings_complete"] is True
    assert result["calibration"]["camera_count"] == 36
    assert result["split"]["counts"] == {
        "source": 7,
        "calibration": 2,
        "target": 1,
    }
    assert result["split"]["held_out_action"] == "fixture action 6"
    assert result["information_boundary"]["prediction_metrics_computed"] is False


def test_target_tactile_values_remain_sealed_by_default(tmp_path: Path) -> None:
    raw = tmp_path / "001-rope"
    processed = tmp_path / "processed"
    _write_fixture(raw)
    _write_target_processed_fixture(processed)
    config = load_deform360_protocol_config(_config_path())

    sealed = preflight_deform360_001_rope(raw, config, processed_root=processed)
    prefix = preflight_deform360_001_rope(
        raw,
        config,
        processed_root=processed,
        unlock_target_prefix=True,
        target_prefix_start_frame=2,
    )
    opened = preflight_deform360_001_rope(
        raw,
        config,
        processed_root=processed,
        unlock_target_oracle=True,
    )
    sealed_target = next(
        episode
        for episode in sealed["processed_episodes"]
        if episode["split"] == "target"
    )
    opened_target = next(
        episode
        for episode in opened["processed_episodes"]
        if episode["split"] == "target"
    )
    prefix_target = next(
        episode
        for episode in prefix["processed_episodes"]
        if episode["split"] == "target"
    )

    assert sealed["information_boundary"]["target_tactile_values_read"] is False
    assert sealed_target["tactile"]["values_read"] is False
    assert "contact_first_frame" not in sealed_target["tactile"]["sensors"][0]
    assert prefix["information_boundary"]["target_prefix_values_read"] is True
    assert prefix["information_boundary"]["target_oracle_values_read"] is False
    assert prefix_target["tactile"]["value_scope"] == "prefix"
    assert prefix_target["tactile"]["value_frame_start"] == 2
    assert prefix_target["tactile"]["value_frame_limit"] == 6
    assert prefix_target["tactile"]["sensors"][0]["contact_first_frame"] == 3
    assert opened["information_boundary"]["target_tactile_values_read"] is True
    assert opened["information_boundary"]["target_oracle_values_read"] is True
    assert opened_target["tactile"]["value_scope"] == "full"
    assert opened_target["tactile"]["sensors"][0]["contact_first_frame"] == 3


def test_missing_one_recording_fails_raw_gate(tmp_path: Path) -> None:
    raw = tmp_path / "001-rope"
    _write_fixture(raw)
    missing = next(raw.glob("*_cam0/*.mp4"))
    missing.unlink()
    config = load_deform360_protocol_config(_config_path())

    result = preflight_deform360_001_rope(raw, config)

    assert result["preflight_passed"] is False
    assert result["raw_inventory"]["file_count_matches"] is False
    assert result["raw_inventory"]["camera_recordings_complete"] is False
