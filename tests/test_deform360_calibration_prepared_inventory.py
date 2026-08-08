"""Prepared-source inventory contracts for the successful Deform360 calibration."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import test_deform360_calibration_observability_batch as batch_cases

ROOT = Path(__file__).resolve().parents[1]
CLI_PATH = ROOT / "scripts/science/inventory_deform360_calibration_prepared_source.py"
SPEC = importlib.util.spec_from_file_location("deform360_prepared_inventory", CLI_PATH)
assert SPEC is not None and SPEC.loader is not None
CLI = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = CLI
SPEC.loader.exec_module(CLI)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, content: bytes) -> dict[str, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return {
        "path": path,
        "sha256": _sha256(path),
        "byte_count": len(content),
    }


def _camera(episode: Path, camera: str, frame_count: int) -> dict[str, str]:
    root = episode / camera
    video = _write(root / "undistorted.mp4", (camera + "-video").encode())
    preview = _write(root / "undistorted_000000.png", (camera + "-preview").encode())
    timestamps = _write(
        root / "aligned_timestamps.txt",
        "".join(f"{index} {index}\n" for index in range(frame_count)).encode(),
    )
    alignment_value = {"schema": "deform360.alignment/v1", "matches": []}
    alignment_path = root / "alignment.json"
    alignment_path.write_text(json.dumps(alignment_value) + "\n", encoding="utf-8")
    alignment = {"sha256": _sha256(alignment_path)}
    metadata = {
        "schema": "deform360.camera-alignment/v1",
        "target_timeline": {
            "stream": "camera-0",
            "count": frame_count,
            "sha256": "a" * 64,
        },
        "output": {
            "video_sha256": video["sha256"],
            "preview_sha256": preview["sha256"],
            "timestamp_sha256": timestamps["sha256"],
            "alignment_sha256": alignment["sha256"],
            "frame_count": frame_count,
            "width": 640,
            "height": 320,
            "fps": 30.0,
        },
    }
    metadata_path = root / "metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "video": str(video["sha256"]),
        "preview": str(preview["sha256"]),
        "timestamps": str(timestamps["sha256"]),
        "alignment": str(alignment["sha256"]),
        "metadata": _sha256(metadata_path),
    }


def _prepared_inputs(tmp_path: Path):
    inputs = batch_cases._batch_inputs(tmp_path / "chain")
    source = batch_cases.case_inputs.source_run_cases
    processed = tmp_path / "aligned"
    result = json.loads(inputs.chain.result_path.read_text(encoding="utf-8"))

    for row in result["objects"]:
        if row["status"] != "source_prepared":
            continue
        object_id = row["object_id"]
        episode = processed / object_id / "episode_0000"
        episode.mkdir(parents=True)
        alignment_path = episode / "alignment.json"
        alignment_path.write_text(
            json.dumps(
                {
                    "schema": "deform360.episode-alignment/v1",
                    "frame_count": 81,
                    "cameras": row["cameras"],
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        intrinsics_path = episode / "undistorted_intrinsics.npy"
        extrinsics_path = episode / "extrinsics.npy"
        intrinsics_path.write_bytes(b"synthetic-intrinsics")
        extrinsics_path.write_bytes(b"synthetic-extrinsics")

        robot_path = episode / "robot" / "robot.npz"
        robot_path.parent.mkdir()
        np.savez(
            robot_path,
            actions=np.zeros((81, 5, 3), dtype=np.float64),
            T_worlds=np.tile(np.eye(4), (81, 1, 1)),
            openings=np.zeros(81, dtype=np.float64),
            bimanual=np.asarray(False),
        )
        tactile_hashes: dict[str, str] = {}
        for sensor in row["tactile_sensors"]:
            tactile_path = episode / sensor / "synced_tactile.npy"
            tactile_path.parent.mkdir()
            np.save(
                tactile_path,
                np.zeros((81, 16, 32), dtype=np.float32),
                allow_pickle=False,
            )
            tactile_hashes[sensor] = _sha256(tactile_path)
        for camera in row["cameras"]:
            _camera(episode, camera, 81)

        row["aligned_frame_count"] = 81
        row["action_window"] = {
            "candidate_count": 1,
            "candidate_first_frame": 0,
            "candidate_stride_frames": 1,
            "input_fields": ["robot.actions", "robot.openings"],
            "known_future_action_is_conditioning_input": True,
            "mean_closed_weighted_path_length_m": 0.1,
            "object_geometry_read": False,
            "object_tracks_read": False,
            "prediction_raw_frame_range_half_open": [0, 76],
            "prefix_raw_frame_range_half_open": [0, 58],
            "selected_raw_frame_range_half_open": [0, 81],
            "selection_rule": "maximum_mean_closed_weighted_gripper_path",
            "tactile_read": False,
            "tie_break": "earliest start",
        }
        row["outputs_sha256"] = {
            "alignment": _sha256(alignment_path),
            "undistorted_intrinsics": _sha256(intrinsics_path),
            "extrinsics": _sha256(extrinsics_path),
            "robot": _sha256(robot_path),
            "tactile": tactile_hashes,
        }

    result["result_sha256"] = source.canonical_sha256(
        result,
        digest_key="result_sha256",
    )
    source._write(inputs.chain.result_path, result)
    inputs.chain.result = result
    inputs.run_record_path.unlink()
    batch_cases.case_inputs.save_deform360_calibration_source_run_record(
        source._record(inputs.chain),
        inputs.run_record_path,
    )
    return inputs, processed


def _arguments(inputs: Any, processed: Path, output: Path) -> list[str]:
    return [
        "--source-protocol",
        str(inputs.chain.source_protocol_path),
        "--stage0-protocol",
        str(inputs.chain.stage0_protocol_path),
        "--selection-lock",
        str(inputs.chain.selection_path),
        "--visual-provider-lock",
        str(inputs.chain.provider_path),
        "--plan",
        str(inputs.chain.plan_path),
        "--download-manifest",
        str(inputs.chain.download_path),
        "--result",
        str(inputs.chain.result_path),
        "--run-record",
        str(inputs.run_record_path),
        "--processed-root",
        str(processed),
        "--implementation-revision",
        batch_cases.case_inputs.IMPLEMENTATION_REVISION,
        "--output",
        str(output),
    ]


def _build(inputs: Any, processed: Path) -> dict[str, object]:
    return CLI.build_inventory(
        source_protocol_path=inputs.chain.source_protocol_path,
        stage0_protocol_path=inputs.chain.stage0_protocol_path,
        selection_lock_path=inputs.chain.selection_path,
        visual_provider_lock_path=inputs.chain.provider_path,
        plan_path=inputs.chain.plan_path,
        download_path=inputs.chain.download_path,
        result_path=inputs.chain.result_path,
        run_record_path=inputs.run_record_path,
        processed_root=processed,
        implementation_revision=batch_cases.case_inputs.IMPLEMENTATION_REVISION,
    )


def test_inventory_binds_all_prepared_bytes_without_local_paths(tmp_path: Path) -> None:
    inputs, processed = _prepared_inputs(tmp_path)
    output = tmp_path / "inventory.json"

    assert CLI.main(_arguments(inputs, processed, output)) == 0
    value = json.loads(output.read_text(encoding="utf-8"))

    assert value["object_count"] == 10
    assert len(value["objects"]) == 10
    assert value["information_boundary"]["confirmation_payloads_opened"] is False
    assert value["information_boundary"]["target_outcomes_used"] is False
    assert all(len(row["cameras"]) == 8 for row in value["objects"])
    assert all(
        row["episode_files"]["robot"]["arrays"]["actions"]["shape"]
        == [81, 5, 3]
        for row in value["objects"]
    )
    serialized = output.read_text(encoding="utf-8")
    assert str(tmp_path) not in serialized
    assert "confirm-sheet" not in serialized


def test_inventory_rejects_tampering_and_confirmation_directories(
    tmp_path: Path,
) -> None:
    inputs, processed = _prepared_inputs(tmp_path)
    tactile = next(processed.rglob("synced_tactile.npy"))
    with tactile.open("ab") as stream:
        stream.write(b"tamper")
    with pytest.raises(ValueError, match="prepared array changed"):
        _build(inputs, processed)

    inputs, processed = _prepared_inputs(tmp_path / "confirmation")
    (processed / "confirm-sheet-0").mkdir()
    with pytest.raises(ValueError, match="confirmation objects appear"):
        _build(inputs, processed)


def test_cli_refuses_overwrite_and_substituted_terminal_artifacts(tmp_path: Path) -> None:
    inputs, processed = _prepared_inputs(tmp_path)
    output = tmp_path / "inventory.json"
    arguments = _arguments(inputs, processed, output)
    assert CLI.main(arguments) == 0
    assert CLI.main(arguments) == 2

    inputs, processed = _prepared_inputs(tmp_path / "substitution")
    with inputs.chain.result_path.open("a", encoding="utf-8") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="result"):
        _build(inputs, processed)
