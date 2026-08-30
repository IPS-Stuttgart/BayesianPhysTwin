from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/science/run_deform360_gpuserver4090_real_evaluation_v1.py"
SPEC = importlib.util.spec_from_file_location("deform360_real_eval_v1", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _protocol(tmp_path: Path) -> tuple[Path, Path]:
    dataset = tmp_path / "deform360"
    dataset.mkdir()
    protocol = json.loads(
        (ROOT / "protocols/deform360_gpuserver4090_real_evaluation_v1.json").read_text()
    )
    protocol["dataset"]["root"] = str(dataset)
    protocol["development_object_ids"] = ["001-rope"]
    protocol["forbidden_reserved_object_ids"] = ["999-reserved"]
    protocol["limits"].update(
        {
            "maximum_objects": 1,
            "maximum_sensor_groups_per_object": 1,
            "maximum_source_recordings_per_group": 2,
            "maximum_frames_per_recording": 96,
            "minimum_frames_per_recording": 24,
            "forecast_horizon_frames": 4,
            "window_stride": 2,
            "bootstrap_repetitions": 100,
        }
    )
    protocol["joint_uncertainty"].update(
        {"maximum_low_rank": 3, "energy_score_samples": 8}
    )
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(protocol))
    return dataset, path


def _write_recording(path: Path, phase: float, *, altered_future: bool = False) -> None:
    frames = 80
    values = np.zeros((frames, 16, 32), dtype=np.float32)
    time = np.linspace(0.0, 4.0 * np.pi, frames)
    center = 8.0 + 4.0 * np.sin(time + phase)
    amplitude = 0.4 + 0.35 * (1.0 + np.sin(0.5 * time + phase))
    for index in range(frames):
        column = int(np.clip(round(center[index]), 1, 29))
        values[index, 4:9, column - 1 : column + 2] = amplitude[index]
    if altered_future:
        values[45:, 2:11, 20:27] = 1.2
    values.tofile(path)
    stamp = path.stem.rsplit("_", 1)[-1]
    np.save(path.parent / f"median_{stamp}.npy", np.zeros((16, 32), dtype=np.float32))


def _fixture(tmp_path: Path) -> tuple[Path, Path]:
    dataset, protocol = _protocol(tmp_path)
    tactile = dataset / "raw-repository" / "raw" / "001-rope" / "tactile_left"
    tactile.mkdir(parents=True)
    for index, phase in enumerate((0.0, 0.25, 0.5), start=1):
        _write_recording(tactile / f"sensor_{index:03d}.npy", phase)
    reserved = dataset / "raw-repository" / "raw" / "999-reserved" / "tactile_left"
    reserved.mkdir(parents=True)
    for index in range(3):
        _write_recording(reserved / f"reserved_{index:03d}.npy", float(index))
    return dataset, protocol


def test_tactile_fallback_is_real_scored_and_object_balanced(tmp_path: Path) -> None:
    dataset, protocol = _fixture(tmp_path)
    result = MODULE.run(protocol, dataset)
    assert result["status"] == "complete"
    assert result["summary"]["primary_modality"] == "tactile_response"
    assert result["summary"]["primary_object_count"] == 1
    assert result["summary"]["completed_group_count"] == 1
    assert result["carrier_inventory"]["tactile_descriptor_recordings"] == 3
    assert result["carrier_inventory"]["reserved_object_payloads_opened"] is False
    row = result["completed_groups"][0]
    assert row["target_recording"] == "sensor_003"
    assert row["source_recordings"] == ["sensor_001", "sensor_002"]
    assert row["source_fit_frozen_before_target_open"] is True
    assert row["target_opened_after_source_fit_id_created"] is True
    assert set(row["methods"]) == set(MODULE.METHODS)
    assert row["forecast_window_count"] > 0
    assert np.isfinite(row["bayesian_uncertainty"]["joint_nanees"])
    assert result["paper_claim_authorized"] is False


def test_target_change_cannot_change_the_source_fit(tmp_path: Path) -> None:
    dataset, protocol = _fixture(tmp_path)
    before = MODULE.run(protocol, dataset)
    target = (
        dataset
        / "raw-repository"
        / "raw"
        / "001-rope"
        / "tactile_left"
        / "sensor_003.npy"
    )
    _write_recording(target, 0.5, altered_future=True)
    after = MODULE.run(protocol, dataset)
    before_row = before["completed_groups"][0]
    after_row = after["completed_groups"][0]
    assert before_row["source_fit_id"] == after_row["source_fit_id"]
    assert before_row["target_fingerprint"] != after_row["target_fingerprint"]
    assert (
        before_row["methods"]["persistence"]["tactile_field_rmse"]
        != after_row["methods"]["persistence"]["tactile_field_rmse"]
    )


def test_report_retains_claim_boundary(tmp_path: Path) -> None:
    dataset, protocol = _fixture(tmp_path)
    result = MODULE.run(protocol, dataset)
    report = MODULE.make_report(result)
    assert "retrospective public-real-data" in report
    assert "does not by itself validate dense 4-D geometry" in report
    assert "paper_claim_authorized" in report
