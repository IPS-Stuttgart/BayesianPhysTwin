from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any

import pytest

import bayesian_phystwin.deform360_adaptive_covariance_confirmation_source_custody as custody
from bayesian_phystwin.deform360_adaptive_covariance_confirmation_lock import (
    PROTOCOL_ID,
    write_confirmation_cohort_lock,
)


H1 = "a" * 40
H2 = "b" * 40
CAMERAS = tuple(f"camera-{index:02d}" for index in range(8))
SOURCE_START = 8


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _result_sha256(value: dict[str, Any]) -> str:
    unsigned = dict(value)
    unsigned.pop("result_sha256", None)
    return hashlib.sha256(
        json.dumps(
            unsigned,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _write(path: Path, value: bytes | str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, str):
        path.write_text(value, encoding="utf-8")
    else:
        path.write_bytes(value)


def _write_json(path: Path, value: dict[str, Any]) -> None:
    _write(
        path,
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )


def _locked_identity(lock: dict[str, Any], case_id: str) -> dict[str, Any]:
    matches = [
        {
            "case": case_id,
            "object_id": record["object_id"],
            "episode_id": episode["episode_id"],
            "episode_key": f"{record['object_id']}/{episode['episode_id']}",
            "stratum": stratum,
            "role": "calibration",
        }
        for stratum, records in lock["cohort"].items()
        for record in records
        for episode in record["episodes"]
        if episode["case_id"] == case_id
    ]
    assert len(matches) == 1
    return matches[0]


def _fake_decoded_prefix_sha256(
    video_path: Path,
    *,
    frame_count: int = custody.PREFIX_FRAME_COUNT,
) -> str:
    return hashlib.sha256(
        b"raw-rgb24-test\0"
        + str(frame_count).encode("ascii")
        + b"\0"
        + video_path.read_bytes()
    ).hexdigest()


def _build_input_trees(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, Any]:
    monkeypatch.setattr(
        custody,
        "_decoded_rgb24_prefix_sha256",
        _fake_decoded_prefix_sha256,
    )
    lock_path = tmp_path / "lock" / "cohort.json"
    lock = write_confirmation_cohort_lock(lock_path, H1)
    case_id = lock["selected_case_ids"][0]
    identity = _locked_identity(lock, case_id)
    source = (
        tmp_path
        / "aligned"
        / identity["object_id"]
        / f"episode_{identity['episode_id']:04d}"
    )
    staged = tmp_path / "staged" / case_id
    source.mkdir(parents=True)
    staged.mkdir(parents=True)

    _write(source / "alignment.json", '{"frame_count":120}\n')
    _write(source / "undistorted_intrinsics.npy", b"source-intrinsics")
    _write(source / "extrinsics.npy", b"source-extrinsics")
    _write(source / "robot" / "robot.npz", b"source-robot")
    _write(source / "robot" / "robot.meta.json", '{"robot":true}\n')
    source_timestamp_lines = [f"{index:06d}" for index in range(120)]
    source_timestamps = "\n".join(source_timestamp_lines) + "\n"
    for camera in CAMERAS:
        camera_root = source / camera
        _write(camera_root / "undistorted.mp4", f"full-video-{camera}".encode())
        _write(camera_root / "aligned_timestamps.txt", source_timestamps)
        _write(camera_root / "alignment.json", '{"aligned":true}\n')
        _write(camera_root / "metadata.json", f'{{"camera":"{camera}"}}\n')
        _write(camera_root / "undistorted_000000.png", b"png")

    source_manifest: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": "Deform360BiasAwareSourcePreparation",
        "protocol_id": PROTOCOL_ID,
        "protocol_config_sha256": lock["artifact_sha256"],
        **identity,
        "cameras": list(CAMERAS),
        "camera_count": len(CAMERAS),
        "inputs_sha256": {"protocol": _file_sha256(lock_path)},
        "outputs_sha256": {
            "alignment": _file_sha256(source / "alignment.json"),
            "undistorted_intrinsics": _file_sha256(
                source / "undistorted_intrinsics.npy"
            ),
            "extrinsics": _file_sha256(source / "extrinsics.npy"),
            "robot": _file_sha256(source / "robot" / "robot.npz"),
            "robot_metadata": _file_sha256(source / "robot" / "robot.meta.json"),
            "camera_metadata": {
                camera: _file_sha256(source / camera / "metadata.json")
                for camera in CAMERAS
            },
        },
        "target_access_authorization": None,
    }
    source_manifest["result_sha256"] = _result_sha256(source_manifest)
    source_manifest_path = source / custody.SOURCE_PREPARATION_FILENAME
    _write_json(source_manifest_path, source_manifest)

    prefix_episode = staged / "prefix" / "episode_0000"
    frame_episode = staged / "frame-zero" / "episode_0000"
    for episode in (prefix_episode, frame_episode):
        _write(episode / "undistorted_intrinsics.npy", b"staged-intrinsics")
        _write(episode / "extrinsics.npy", b"staged-extrinsics")
        _write(episode / "robot" / "robot.npz", b"robot-slice")
    _write(staged / "known-action" / "robot.npz", b"known-action")
    _write(staged / custody.FRAME_ZERO_ARCHIVE_FILENAME, b"frame-zero-points")
    _write(frame_episode / "splatfacto" / "splat_0.ply", b"splat")
    _write(
        frame_episode / "splatfacto" / "splatfacto.meta.json",
        '{"splat":true}\n',
    )

    prefix_timestamp_text = (
        "\n".join(
            source_timestamp_lines[
                SOURCE_START : SOURCE_START + custody.PREFIX_FRAME_COUNT
            ]
        )
        + "\n"
    )
    frame_timestamp_text = source_timestamp_lines[SOURCE_START] + "\n"
    camera_records = []
    for camera in CAMERAS:
        prefix_camera = prefix_episode / camera
        frame_camera = frame_episode / camera
        _write(prefix_camera / "undistorted.mp4", f"prefix-{camera}".encode())
        _write(prefix_camera / "aligned_timestamps.txt", prefix_timestamp_text)
        _write(prefix_camera / "mask_refined.h5", f"mask-{camera}".encode())
        _write(prefix_camera / "rendered_depth.h5", f"depth-{camera}".encode())
        _write(
            prefix_camera / "rendered_depth.meta.json",
            f'{{"depth":"{camera}"}}\n',
        )
        _write(frame_camera / "undistorted.mp4", f"frame-zero-{camera}".encode())
        _write(frame_camera / "aligned_timestamps.txt", frame_timestamp_text)
        _write(frame_camera / "mask_refined.h5", f"frame-mask-{camera}".encode())
        shutil.copyfile(
            prefix_camera / "rendered_depth.h5",
            frame_camera / "rendered_depth.h5",
        )
        shutil.copyfile(
            prefix_camera / "rendered_depth.meta.json",
            frame_camera / "rendered_depth.meta.json",
        )
        _write(frame_camera / "rendered_urdf.h5", f"urdf-{camera}".encode())
        _write(
            frame_camera / "rendered_urdf.meta.json",
            f'{{"urdf":"{camera}"}}\n',
        )
        camera_records.append(
            {
                "camera": camera,
                "prefix_video_sha256": _file_sha256(prefix_camera / "undistorted.mp4"),
                "frame_zero_video_sha256": _file_sha256(
                    frame_camera / "undistorted.mp4"
                ),
                "frame_zero_mask_sha256": _file_sha256(
                    prefix_camera / "mask_refined.h5"
                ),
            }
        )

    prefix_manifest: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": "Deform360BiasAwarePredictionPrefix",
        "protocol_id": PROTOCOL_ID,
        "protocol_config_sha256": lock["artifact_sha256"],
        **identity,
        "action_window": {
            "selected_raw_frame_range_half_open": [
                SOURCE_START,
                SOURCE_START + custody.STAGING_FRAME_COUNT,
            ],
            "prediction_raw_frame_range_half_open": [
                SOURCE_START,
                SOURCE_START + custody.KNOWN_ACTION_FRAME_COUNT,
            ],
            "prefix_raw_frame_range_half_open": [
                SOURCE_START,
                SOURCE_START + custody.PREFIX_FRAME_COUNT,
            ],
        },
        "staged_prefix_frame_count": custody.PREFIX_FRAME_COUNT,
        "staged_frame_zero_frame_count": custody.FRAME_ZERO_FRAME_COUNT,
        "known_action_frame_count": custody.KNOWN_ACTION_FRAME_COUNT,
        "camera_count": len(CAMERAS),
        "camera_records": camera_records,
        "staged_robot_sha256": {
            "prefix": _file_sha256(prefix_episode / "robot" / "robot.npz"),
            "frame_zero": _file_sha256(frame_episode / "robot" / "robot.npz"),
            "known_action": _file_sha256(staged / "known-action" / "robot.npz"),
        },
        "inputs_sha256": {
            "protocol": _file_sha256(lock_path),
            "source_preparation_manifest": _file_sha256(source_manifest_path),
            "source_robot": _file_sha256(source / "robot" / "robot.npz"),
            "source_intrinsics": _file_sha256(source / "undistorted_intrinsics.npy"),
            "source_extrinsics": _file_sha256(source / "extrinsics.npy"),
        },
        "target_access_authorization": None,
    }
    prefix_manifest["result_sha256"] = _result_sha256(prefix_manifest)
    prefix_manifest_path = staged / custody.PREDICTION_PREFIX_MANIFEST_FILENAME
    _write_json(prefix_manifest_path, prefix_manifest)

    frame_manifest: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": "Deform360BiasAwareFrameZeroReconstruction",
        "protocol_id": PROTOCOL_ID,
        "protocol_config_sha256": lock["artifact_sha256"],
        **identity,
        "cameras": list(CAMERAS),
        "camera_count": len(CAMERAS),
        "inputs_sha256": {
            "prediction_prefix_manifest": _file_sha256(prefix_manifest_path),
        },
        "outputs_sha256": {
            "frame_zero_splat": _file_sha256(
                frame_episode / "splatfacto" / "splat_0.ply"
            ),
            "frame_zero_points": _file_sha256(
                staged / custody.FRAME_ZERO_ARCHIVE_FILENAME
            ),
            "depth_by_camera": {
                camera: _file_sha256(prefix_episode / camera / "rendered_depth.h5")
                for camera in CAMERAS
            },
            "gripper_mask_by_camera": {
                camera: _file_sha256(frame_episode / camera / "rendered_urdf.h5")
                for camera in CAMERAS
            },
        },
    }
    frame_manifest["result_sha256"] = _result_sha256(frame_manifest)
    _write_json(staged / custody.FRAME_ZERO_MANIFEST_FILENAME, frame_manifest)
    return {
        "lock_path": lock_path,
        "case_id": case_id,
        "source": source,
        "staged": staged,
        "output": tmp_path / "custody" / f"{case_id}.json",
    }


def _build(paths: dict[str, Any]) -> dict[str, Any]:
    return custody.build_confirmation_source_custody_seal(
        paths["lock_path"],
        H2,
        paths["case_id"],
        paths["source"],
        paths["staged"],
        paths["output"],
        expected_h1=H1,
    )


def _validate(paths: dict[str, Any]) -> dict[str, Any]:
    return custody.validate_confirmation_source_custody_seal(
        paths["output"],
        paths["lock_path"],
        H2,
        paths["case_id"],
        paths["source"],
        paths["staged"],
        expected_h1=H1,
    )


def test_write_once_custody_seal_replays_complete_source_and_stage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _build_input_trees(tmp_path, monkeypatch)
    payload = _build(paths)

    assert payload["artifact_kind"] == custody.ARTIFACT_KIND
    assert payload["lock_binding"]["cohort_lock_commit_h2"] == H2
    assert payload["case_identity"]["case"] == paths["case_id"]
    assert payload["camera_panel"] == list(CAMERAS)
    assert payload["raw_rgb24_prefix"] == {
        "algorithm": custody.RAW_RGB24_PREFIX_ALGORITHM,
        "frame_count": custody.PREFIX_FRAME_COUNT,
        "by_camera": {
            camera: _fake_decoded_prefix_sha256(
                paths["staged"] / "prefix" / "episode_0000" / camera / "undistorted.mp4"
            )
            for camera in CAMERAS
        },
        "direct_source_vs_reencoded_prefix_equality_required": False,
        "authorized_future_must_reuse_this_digest": True,
    }
    source_inventory = payload["inventories"]["aligned_source_episode"]
    staged_inventory = payload["inventories"]["staged_prediction_case"]
    assert source_inventory["regular_file_count"] == 6 + 5 * len(CAMERAS)
    assert staged_inventory["regular_file_count"] == 12 + 12 * len(CAMERAS)
    assert all(
        payload["camera_custody"][camera]["timestamp_prefix_exact_source_slice"]
        for camera in CAMERAS
    )
    assert _validate(paths) == payload
    assert (
        custody.build_confirmation_source_custody_seal(
            paths["lock_path"],
            H2,
            paths["case_id"],
            paths["source"],
            paths["staged"],
            paths["output"],
            expected_h1=H1,
            replay=True,
        )
        == payload
    )
    with pytest.raises(ValueError, match="already exists"):
        _build(paths)
    source_moved = paths["source"].with_name(paths["source"].name + "-moved")
    staged_moved = paths["staged"].with_name(paths["staged"].name + "-moved")
    paths["source"].rename(source_moved)
    paths["staged"].rename(staged_moved)
    assert (
        custody.validate_confirmation_source_custody_envelope(
            paths["output"],
            paths["lock_path"],
            H2,
            paths["case_id"],
            expected_h1=H1,
        )
        == payload
    )


def test_replay_rejects_changed_full_source_video_not_hashed_by_old_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _build_input_trees(tmp_path, monkeypatch)
    _build(paths)
    _write(paths["source"] / CAMERAS[0] / "undistorted.mp4", b"changed-full-video")

    with pytest.raises(ValueError, match="replay differs"):
        _validate(paths)


@pytest.mark.parametrize("kind", ["extra", "symlink", "hardlink"])
def test_tree_replay_rejects_extras_symlinks_and_hardlinks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
) -> None:
    paths = _build_input_trees(tmp_path, monkeypatch)
    _build(paths)
    camera = paths["staged"] / "prefix" / "episode_0000" / CAMERAS[0]
    if kind == "extra":
        _write(camera / "extra.bin", b"extra")
        match = "file inventory changed"
    elif kind == "symlink":
        os.symlink(camera / "undistorted.mp4", camera / "extra-link")
        match = "symlink"
    else:
        linked = camera / "aligned_timestamps.txt"
        linked.unlink()
        os.link(camera / "undistorted.mp4", linked)
        match = "hard-linked"

    with pytest.raises(ValueError, match=match):
        _validate(paths)


def test_build_rejects_timestamp_window_that_does_not_replay_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _build_input_trees(tmp_path, monkeypatch)
    timestamps = (
        paths["staged"]
        / "prefix"
        / "episode_0000"
        / CAMERAS[0]
        / "aligned_timestamps.txt"
    )
    lines = timestamps.read_text(encoding="utf-8").splitlines()
    lines[0] = "not-the-source-timestamp"
    _write(timestamps, "\n".join(lines) + "\n")

    with pytest.raises(ValueError, match="do not replay the source window"):
        _build(paths)


def test_seal_is_h2_and_absolute_path_bound(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _build_input_trees(tmp_path, monkeypatch)
    _build(paths)
    copied = paths["output"].with_name("moved.json")
    shutil.copyfile(paths["output"], copied)

    with pytest.raises(ValueError, match="path binding|replay differs"):
        custody.validate_confirmation_source_custody_seal(
            copied,
            paths["lock_path"],
            H2,
            paths["case_id"],
            paths["source"],
            paths["staged"],
            expected_h1=H1,
        )
    with pytest.raises(ValueError, match="lock binding|replay differs"):
        custody.validate_confirmation_source_custody_seal(
            paths["output"],
            paths["lock_path"],
            "c" * 40,
            paths["case_id"],
            paths["source"],
            paths["staged"],
            expected_h1=H1,
        )


def test_envelope_rejects_resigned_noninteger_prefix_range(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _build_input_trees(tmp_path, monkeypatch)
    payload = _build(paths)
    payload["camera_custody"][CAMERAS[0]]["source_prefix_frame_range_half_open"][0] = (
        str(SOURCE_START)
    )
    payload["artifact_sha256"] = custody.artifact_sha256(payload)
    paths["output"].chmod(0o644)
    _write_json(paths["output"], payload)

    with pytest.raises(ValueError, match="camera-custody record changed"):
        custody.validate_confirmation_source_custody_envelope(
            paths["output"],
            paths["lock_path"],
            H2,
            paths["case_id"],
            expected_h1=H1,
        )
