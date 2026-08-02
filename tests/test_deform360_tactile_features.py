from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from bayesian_phystwin.deform360_tactile_features import (
    TACTILE_SENSOR_NAMES,
    build_tactile_feature_artifact,
    canonical_artifact_sha256,
    extract_case_tactile_features,
    load_raw_tactile_frames,
    read_frame_timestamps_us,
)


def _write_timeline(path: Path, timestamps: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            f"frame_{int(timestamp)}_{index:012d}\n"
            for index, timestamp in enumerate(timestamps)
        ),
        encoding="utf-8",
    )


def _fixture(tmp_path: Path) -> tuple[Path, Path, np.ndarray]:
    episode = tmp_path / "windows" / "case-a" / "episode_0000"
    raw_object = tmp_path / "raw" / "object-a"
    target = 1_000_000 + np.arange(81, dtype=np.int64) * 33_333
    for camera in ("brics-odroid-001_cam0", "brics-odroid-002_cam0"):
        _write_timeline(episode / camera / "aligned_timestamps.txt", target)
    source = target[0] - 10_000 + np.arange(83, dtype=np.int64) * 33_333
    for sensor_index, sensor in enumerate(TACTILE_SENSOR_NAMES):
        sensor_root = raw_object / sensor
        sensor_root.mkdir(parents=True)
        baseline = np.full((16, 32), sensor_index, dtype=np.float64)
        np.save(sensor_root / "median_900000.npy", baseline)
        frames = np.full((83, 16, 32), sensor_index, dtype=np.float32)
        frames[:, 0, 0] += np.arange(83, dtype=np.float32)
        frames.tofile(sensor_root / f"{sensor}_950000.npy")
        _write_timeline(sensor_root / f"{sensor}_950000.txt", source)
    return episode, raw_object, target


def test_raw_tactile_loader_supports_binary_and_numpy(tmp_path: Path) -> None:
    values = np.arange(2 * 16 * 32, dtype=np.float32).reshape(2, 16, 32)
    binary = tmp_path / "binary.npy"
    archive = tmp_path / "archive.npy"
    values.tofile(binary)
    np.save(archive, values)
    assert np.array_equal(load_raw_tactile_frames(binary, frame_count=2), values)
    assert np.array_equal(load_raw_tactile_frames(archive, frame_count=2), values)


def test_timestamp_reader_rejects_malformed_rows(tmp_path: Path) -> None:
    path = tmp_path / "timestamps.txt"
    path.write_text("not-a-frame\n", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid timestamp token"):
        read_frame_timestamps_us(path)


def test_case_extraction_hashes_every_raw_input(tmp_path: Path) -> None:
    episode, raw_object, _ = _fixture(tmp_path)
    result = extract_case_tactile_features(
        case_name="object-a-ep0000",
        object_id="object-a",
        episode_index=0,
        episode_root=episode,
        raw_object_root=raw_object,
    )
    assert result["episode_frame_count"] == 81
    assert result["available_frame_count"] == 58
    assert result["sensor_count"] == 4
    assert len(result["target_timeline_records"]) == 2
    assert [row["update_frame"] for row in result["updates"]] == [19, 38, 57]
    for record in result["source_records"]:
        assert set(record) >= {
            "data_sha256",
            "timestamps_sha256",
            "baseline_sha256",
        }
        assert all(len(record[name]) == 64 for name in record if name.endswith("sha256"))


def test_artifact_is_canonical_and_target_free(tmp_path: Path) -> None:
    _fixture(tmp_path)
    cases = [
        {
            "case": "object-a-ep0000",
            "object": "object-a",
            "episode_index": 0,
            "episode_path": "case-a/episode_0000",
            "raw_object_path": "object-a",
        }
    ]
    artifact = build_tactile_feature_artifact(
        cases,
        window_root=tmp_path / "windows",
        raw_root=tmp_path / "raw",
    )
    assert artifact["artifact_sha256"] == canonical_artifact_sha256(artifact)
    assert artifact["information_boundary"]["target_outcomes_read"] is False
    assert artifact["information_boundary"]["future_tactile_used_for_update"] is False


def test_values_after_latest_update_do_not_change_features(tmp_path: Path) -> None:
    episode, raw_object, _ = _fixture(tmp_path)
    arguments = {
        "case_name": "object-a-ep0000",
        "object_id": "object-a",
        "episode_index": 0,
        "episode_root": episode,
        "raw_object_root": raw_object,
    }
    before = extract_case_tactile_features(**arguments)
    sensor = TACTILE_SENSOR_NAMES[0]
    data_path = raw_object / sensor / f"{sensor}_950000.npy"
    frames = np.memmap(data_path, dtype="<f4", mode="r+", shape=(83, 16, 32))
    frames[70:] = 1_000_000.0
    frames.flush()
    del frames
    after = extract_case_tactile_features(**arguments)
    assert before["updates"] == after["updates"]
    assert (
        before["source_records"][0]["data_sha256"]
        != after["source_records"][0]["data_sha256"]
    )


def test_camera_timeline_disagreement_is_rejected(tmp_path: Path) -> None:
    episode, raw_object, target = _fixture(tmp_path)
    target[-1] += 1
    _write_timeline(
        episode / "brics-odroid-002_cam0" / "aligned_timestamps.txt",
        target,
    )
    with pytest.raises(ValueError, match="camera timelines disagree"):
        extract_case_tactile_features(
            case_name="object-a-ep0000",
            object_id="object-a",
            episode_index=0,
            episode_root=episode,
            raw_object_root=raw_object,
        )
