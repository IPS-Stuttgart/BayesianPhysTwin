from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from bayesian_phystwin.deform360_pairwise_regret_guard_fresh_processing import (
    PROTOCOL_KIND,
    build_fresh_processing_protocol,
    build_fresh_source_admission,
    canonical_sha256,
    fresh_processing_case,
    select_fresh_source_window,
    validate_fresh_processing_protocol,
    validate_fresh_processing_sources,
    validate_fresh_source_admission,
)
from bayesian_phystwin.deform360_pairwise_regret_guard_fresh_protocol import (
    file_sha256,
)

ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "configs/sota/deform360_pairwise_regret_guard_fresh_technical_v1.json"
PLAN = ROOT / "configs/sota/deform360_pairwise_regret_guard_fresh_source_plan_v1.json"


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _fake_download(tmp_path: Path) -> Path:
    plan = _read(PLAN)
    rows = [
        {
            "path": row["path"],
            "size": row["size"],
            "sha256": row["lfs_sha256"] or "0" * 64,
        }
        for row in plan["download"]["files"]
    ]
    payload = {
        "schema_version": 1,
        "artifact_kind": "Deform360PairwiseRegretGuardFreshSourceDownload",
        "source_plan_sha256": plan["source_plan_sha256"],
        "source_plan_file_sha256": file_sha256(PLAN),
        "repository": plan["repository"],
        "revision": plan["revision"],
        "object_id": plan["object_id"],
        "file_count": len(rows),
        "total_bytes": sum(row["size"] for row in rows),
        "source_tree_sha256": "1" * 64,
        "files": rows,
        "information_boundary": {
            "raw_source_bytes_read": True,
            "future_object_positions_deserialized": False,
            "processed_geometry_read": False,
            "outcome_or_metric_read": False,
            "held_v8_runtime_or_target_artifact_access": False,
        },
    }
    payload["download_sha256"] = canonical_sha256(payload, digest_key="download_sha256")
    path = tmp_path / "download.json"
    _write(path, payload)
    return path


def _protocol(tmp_path: Path) -> tuple[dict, Path, Path]:
    download = _fake_download(tmp_path)
    payload = build_fresh_processing_protocol(
        LOCK,
        PLAN,
        download,
        implementation_commit="a" * 40,
    )
    path = tmp_path / "protocol.json"
    _write(path, payload)
    return payload, path, download


def test_processing_protocol_binds_exact_source_inventory(tmp_path: Path) -> None:
    protocol, path, download = _protocol(tmp_path)
    assert protocol["artifact_kind"] == PROTOCOL_KIND
    validate_fresh_processing_protocol(protocol)
    validate_fresh_processing_sources(path, LOCK, PLAN, download)

    changed = _read(download)
    changed["files"][0]["size"] += 1
    changed["total_bytes"] += 1
    changed["download_sha256"] = canonical_sha256(changed, digest_key="download_sha256")
    _write(download, changed)
    with pytest.raises(ValueError, match="download inventory dimensions differ"):
        validate_fresh_processing_sources(path, LOCK, PLAN, download)


def test_only_metadata_valid_episodes_are_processing_cases() -> None:
    lock = _read(LOCK)
    case = fresh_processing_case(lock, "197-hand-sanitizer", 5)
    assert case["case"] == "197-hand-sanitizer-ep0005"
    with pytest.raises(ValueError, match="outside the valid lock"):
        fresh_processing_case(lock, "197-hand-sanitizer", 6)


def test_action_window_is_deterministic_and_observation_free() -> None:
    actions = np.zeros((110, 5, 3), dtype=np.float64)
    actions[20:, 0, 0] = np.linspace(0.0, 0.2, 90)
    openings = np.linspace(1.0, 0.0, 110)
    first = select_fresh_source_window(actions, openings)
    second = select_fresh_source_window(actions, openings)
    assert first == second
    assert first["object_geometry_read"] is False
    assert first["object_response_read"] is False
    assert first["target_metric_read"] is False
    start, stop = first["selected_raw_frame_range_half_open"]
    assert stop - start == 81


def _write_admission_fixture(tmp_path: Path, point_count: int) -> tuple[Path, Path]:
    episode = tmp_path / "episode_0000"
    episode.mkdir()
    metadata = tmp_path / "metadata.json"
    _write(
        metadata,
        {
            "object": "197-hand-sanitizer",
            "sequences": {"0": {"bimanual": "no"}},
        },
    )
    (episode / "calibrate.pkl").write_bytes(b"calibration")
    (episode / "start_obj_pcd.ply").write_bytes(
        (
            "ply\nformat ascii 1.0\n"
            f"element vertex {point_count}\n"
            "property float x\nproperty float y\nproperty float z\n"
            "end_header\n"
        ).encode("ascii")
    )
    # Invalid pickle bytes prove that admission hashes but never deserializes it.
    (episode / "final_data.pkl").write_bytes(b"not-a-pickle")
    _write(
        episode / "split.json",
        {"frame_len": 76, "train": [0, 60], "test": [60, 76]},
    )
    outputs = {
        "calibrate_sha256": file_sha256(episode / "calibrate.pkl"),
        "start_ply_sha256": file_sha256(episode / "start_obj_pcd.ply"),
        "split_sha256": file_sha256(episode / "split.json"),
        "final_data_sha256": file_sha256(episode / "final_data.pkl"),
        "num_active_frames": 76,
        "contact_start_frame": 0,
        "contact_end_frame": 75,
    }
    _write(
        episode / "control_points.meta.json",
        {
            "schema": "deform360.processing/control-points/v1",
            "inputs": {
                "robot_sha256": "2" * 64,
                "pcd_sha256": "3" * 64,
                "tactile_sha256": {},
            },
            "outputs": outputs,
            "parameters": {
                "cameras": ["camera-a", "camera-b", "camera-c"],
                "train_fraction": 0.8,
            },
        },
    )
    return episode, metadata


def test_admission_never_deserializes_future_payload(tmp_path: Path) -> None:
    protocol, _, _ = _protocol(tmp_path)
    episode, metadata = _write_admission_fixture(tmp_path, 128)
    artifact = build_fresh_source_admission(
        episode,
        metadata,
        protocol=protocol,
        case={
            "case": "197-hand-sanitizer-ep0000",
            "object_id": "197-hand-sanitizer",
            "episode_id": 0,
            "action": "test",
            "bimanual": "no",
            "nonprehensile": "no",
        },
    )
    assert artifact["accepted"] is True
    validate_fresh_source_admission(
        artifact,
        protocol=protocol,
        case={
            "case": "197-hand-sanitizer-ep0000",
            "object_id": "197-hand-sanitizer",
            "episode_id": 0,
            "action": "test",
            "bimanual": "no",
            "nonprehensile": "no",
        },
    )
    assert (
        artifact["information_boundary"]["future_object_positions_deserialized"]
        is False
    )


def test_backend_minimum_rejects_54_point_geometry(tmp_path: Path) -> None:
    protocol, _, _ = _protocol(tmp_path)
    episode, metadata = _write_admission_fixture(tmp_path, 54)
    artifact = build_fresh_source_admission(
        episode,
        metadata,
        protocol=protocol,
        case={
            "case": "197-hand-sanitizer-ep0000",
            "object_id": "197-hand-sanitizer",
            "episode_id": 0,
            "action": "test",
            "bimanual": "no",
            "nonprehensile": "no",
        },
    )
    assert artifact["accepted"] is False
    assert (
        "frame-zero point count is outside backend admission"
        in artifact["rejection_reasons"]
    )
