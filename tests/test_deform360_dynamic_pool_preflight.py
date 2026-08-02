from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np

import bayesian_phystwin.deform360_dynamic_pool_preflight as preflight
from bayesian_phystwin.cli.deform360_dynamic_pool_preflight import build_parser


def test_preflight_interface_has_no_outcome_or_runtime_input() -> None:
    parameters = inspect.signature(preflight.preflight_dynamic_pool_case).parameters

    assert set(parameters) == {
        "panel_case_dir",
        "processed_episode_dir",
        "config",
    }
    assert "target" not in parameters
    assert "outcome" not in parameters
    assert "runtime" not in parameters


def test_preflight_cli_has_only_source_paths() -> None:
    args = build_parser().parse_args(["panel", "processed", "output"])

    assert args.panel_root == "panel"
    assert args.processed_root == "processed"
    assert args.output_dir == "output"


def test_case_preflight_uses_frame_zero_and_emits_staging_plan(
    tmp_path: Path,
    monkeypatch,
) -> None:
    case = preflight.expected_open_case_names()[0]
    panel_case = tmp_path / "panel" / case
    processed = tmp_path / "processed" / case / "episode_0000"
    panel_case.mkdir(parents=True)
    processed.mkdir(parents=True)
    seal = {
        "object_id": "object",
        "episode_id": 1,
        "episode_key": "episode_0000",
    }
    (panel_case / "prediction_seal.json").write_text(
        json.dumps(seal), encoding="utf-8"
    )
    archive = panel_case / "prediction.npz"
    np.savez_compressed(archive, frame_zero_points_m=np.zeros((128, 3)))
    np.save(processed / "undistorted_intrinsics.npy", {"camera-0": np.eye(3)})
    np.save(processed / "extrinsics.npy", {"camera-0": np.eye(4)})
    cameras = tuple(f"camera-{index}" for index in range(8))
    for camera in cameras:
        directory = processed / camera
        directory.mkdir()
        for filename in preflight.CAMERA_STAGING_FILENAMES:
            (directory / filename).write_bytes(b"fixture")

    monkeypatch.setattr(preflight, "_validate_prediction_seal", lambda _seal: None)
    monkeypatch.setattr(
        preflight, "_resolve_prediction_archive", lambda _case, _seal: archive
    )
    monkeypatch.setattr(
        preflight,
        "_load_calibration",
        lambda _processed: (
            {camera: np.eye(3) for camera in cameras},
            {camera: np.eye(4) for camera in cameras},
        ),
    )
    monkeypatch.setattr(
        preflight,
        "frame_zero_camera_support",
        lambda *_args, **_kwargs: (
            cameras,
            np.ones((128, 8), dtype=bool),
            {camera: np.zeros((128, 2)) for camera in cameras},
        ),
    )
    monkeypatch.setattr(
        preflight,
        "select_frame_zero_observation_plan",
        lambda *_args, **_kwargs: {
            "candidate_ids": np.arange(100),
            "center_ids": np.arange(64),
            "selected_cameras": cameras,
            "selection_score": (64, 64, 512, 20.0),
        },
    )
    read_indices: list[str] = []
    monkeypatch.setattr(
        preflight,
        "_read_h5_frame_zero",
        lambda path: read_indices.append(Path(path).name) or np.zeros((2, 2)),
    )

    record = preflight.preflight_dynamic_pool_case(panel_case, processed)

    assert record["status"] == "passed"
    assert record["candidate_count"] == 100
    assert len(record["center_ids"]) == 64
    assert len(record["selected_cameras"]) == 8
    assert len(record["staging_relative_paths"]) == 50
    assert read_indices == ["mask_refined.h5", "rendered_depth.h5"] * 8
    assert record["information_boundary"] == {
        "target_data_read": False,
        "outcome_manifest_read": False,
        "rgb_frame_read": False,
        "future_reconstruction_after_frame_zero_read": False,
        "hdf5_indices_read": [0],
    }


def test_cohort_preflight_records_failure_and_closes_gate(
    tmp_path: Path,
    monkeypatch,
) -> None:
    cases = ("case-a", "case-b")
    monkeypatch.setattr(preflight, "expected_open_case_names", lambda: cases)

    def fake_case(panel_case, _processed_case, *, config):
        if Path(panel_case).name == "case-b":
            raise ValueError("too few multiview-visible frame-zero candidates")
        return {
            "case": "case-a",
            "status": "passed",
            "staging_relative_paths": ["case-a/episode_0000/extrinsics.npy"],
        }

    monkeypatch.setattr(preflight, "preflight_dynamic_pool_case", fake_case)
    output = tmp_path / "result"

    result = preflight.run_dynamic_pool_preflight(
        tmp_path / "panel",
        tmp_path / "processed",
        output,
    )

    assert result["preflight_gate_passed"] is False
    assert result["passed_case_count"] == 1
    assert result["failed_case_count"] == 1
    assert result["cases"][1]["error_type"] == "ValueError"
    persisted = json.loads(
        (output / preflight.PREFLIGHT_FILENAME).read_text(encoding="utf-8")
    )
    assert persisted["result_sha256"] == result["result_sha256"]
