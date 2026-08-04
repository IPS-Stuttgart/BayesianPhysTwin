from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from bayesian_phystwin.deform360_pairwise_regret_guard_fresh_processing import (
    PROCESSING_KIND,
    fresh_processing_case,
    seal_case_artifact,
)

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/remote/run_deform360_pairwise_regret_guard_fresh_prediction.py"
LOCK = ROOT / "configs/sota/deform360_pairwise_regret_guard_fresh_technical_v1.json"
PROTOCOL = (
    ROOT / "configs/sota/deform360_pairwise_regret_guard_fresh_processing_v1.json"
)


def _module():
    spec = importlib.util.spec_from_file_location("fresh_prediction_runner", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fixture(tmp_path: Path) -> tuple[Path, Path, dict, dict]:
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    case = fresh_processing_case(lock, "197-hand-sanitizer", 0)
    cameras = [f"camera-{index:02d}" for index in range(8)]
    extra = "camera-extra"
    processed = tmp_path / "processed"
    processed.mkdir()
    intrinsics = {camera: np.eye(3, dtype=np.float64) for camera in [*cameras, extra]}
    extrinsics = {camera: np.eye(4, dtype=np.float64) for camera in [*cameras, extra]}
    np.save(processed / "undistorted_intrinsics.npy", intrinsics)
    np.save(processed / "extrinsics.npy", extrinsics)
    for camera in cameras:
        camera_dir = processed / camera
        camera_dir.mkdir()
        for filename in ("undistorted.mp4", "mask_refined.h5", "rendered_depth.h5"):
            (camera_dir / filename).write_bytes(f"{camera}:{filename}".encode())
    processing = seal_case_artifact(
        PROCESSING_KIND,
        protocol=protocol,
        case=case,
        payload={"status": "admitted", "cameras": cameras},
    )
    (processed / "fresh_pairwise_processing.json").write_text(
        json.dumps(processing), encoding="utf-8"
    )
    physical_seal = {
        **case,
        "protocol_id": lock["protocol_id"],
        "technical_lock_sha256": lock["lock_sha256"],
        "source_processing_result_sha256": processing["result_sha256"],
    }
    return processed, tmp_path / "staged", physical_seal, protocol


def test_measurement_panel_stages_only_admitted_cameras(tmp_path: Path) -> None:
    processed, staged, physical_seal, protocol = _fixture(tmp_path)
    manifest = _module()._stage_measurement_panel(
        processed, staged, physical_seal, protocol
    )
    assert manifest["camera_count"] == 8
    assert set(manifest["cameras"]) == {f"camera-{index:02d}" for index in range(8)}
    staged_intrinsics = np.load(
        staged / "undistorted_intrinsics.npy", allow_pickle=True
    ).item()
    assert set(staged_intrinsics) == set(manifest["cameras"])
    assert not (staged / "camera-extra").exists()
    for camera in manifest["cameras"]:
        assert (staged / camera / "undistorted.mp4").is_symlink()
    boundary = manifest["information_boundary"]
    assert boundary["future_video_or_hdf5_bytes_read_during_staging"] is False
    assert boundary["target_or_metric_read"] is False


def test_measurement_panel_rejects_missing_admitted_asset(tmp_path: Path) -> None:
    processed, staged, physical_seal, protocol = _fixture(tmp_path)
    (processed / "camera-07" / "mask_refined.h5").unlink()
    with pytest.raises(ValueError, match="admitted camera asset is missing"):
        _module()._stage_measurement_panel(processed, staged, physical_seal, protocol)
