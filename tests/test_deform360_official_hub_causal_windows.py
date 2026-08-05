from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from bayesian_phystwin._portable_contracts import content_id
from bayesian_phystwin.deform360_official_hub_causal_windows import (
    Deform360CustodyError,
    build_deform360_official_hub_causal_window_manifest,
    derive_deform360_causal_window_v2,
    load_deform360_official_hub_causal_window_manifest,
    load_deform360_official_hub_causal_window_manifest_v2,
    save_deform360_official_hub_causal_window_manifest,
    save_deform360_official_hub_causal_window_manifest_v2,
    validate_deform360_causal_schedule_recovery_lock,
    validate_deform360_visual_execution_lock,
)
from bayesian_phystwin.deform360_visual_provider_recovery_lock import (
    load_deform360_visual_provider_recovery_lock,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _repository() -> Path:
    return Path(__file__).resolve().parents[1]


def _add_file(
    object_root: Path,
    files: list[dict[str, object]],
    relative: str,
    payload: bytes,
) -> None:
    path = object_root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    files.append(
        {
            "path": relative,
            "size": len(payload),
            "sha256": _sha256(path),
        }
    )


def _locks() -> tuple[object, dict[str, object]]:
    repository = _repository()
    provider = load_deform360_visual_provider_recovery_lock(
        repository / "protocols/locks/"
        "deform360_official_hub_visuotactile_v1_visual_provider_recovery_v1.json"
    )
    execution = json.loads(
        (
            repository / "protocols/locks/"
            "deform360_official_hub_visuotactile_v1_visual_execution_lock_v1.json"
        ).read_text(encoding="utf-8")
    )
    return provider, execution


def _schedule_lock() -> dict[str, object]:
    return json.loads(
        (
            _repository() / "protocols/locks/"
            "deform360_official_hub_visuotactile_v2_causal_schedule_recovery.json"
        ).read_text(encoding="utf-8")
    )


def _fixture(
    root: Path,
    *,
    no_contact_object: int | None = None,
    contact_frame: int = 50,
) -> tuple[
    dict[str, object],
    dict[str, tuple[dict[str, np.ndarray], dict[str, np.ndarray]]],
]:
    rows: list[dict[str, object]] = []
    calibration: dict[str, tuple[dict[str, np.ndarray], dict[str, np.ndarray]]] = {}
    sensors = (
        "brics-odroid_tactilel_left",
        "brics-odroid_tactilel_right",
        "brics-odroid_tactiler_left",
        "brics-odroid_tactiler_right",
    )
    camera_centers = {
        "camera-a": (1.0, 0.0, 0.0),
        "camera-b": (0.0, 1.0, 0.0),
        "camera-c": (-1.0, 0.0, 0.0),
        "camera-d": (0.0, -1.0, 0.0),
    }
    for index in range(10):
        object_id = f"calibration-{index:02d}"
        object_root = root / object_id
        episode_root = object_root / "episode_0000"
        files: list[dict[str, object]] = []

        for sensor_index, sensor in enumerate(sensors):
            tactile = np.zeros((100, 16, 32), dtype=np.float32)
            if index != no_contact_object and sensor_index == 0:
                tactile[contact_frame, 0, :2] = 1.0
            path = episode_root / sensor / "synced_tactile.npy"
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, tactile, allow_pickle=False)
            relative = f"episode_0000/{sensor}/synced_tactile.npy"
            files.append(
                {
                    "path": relative,
                    "size": path.stat().st_size,
                    "sha256": _sha256(path),
                }
            )
        _add_file(
            object_root,
            files,
            "episode_0000/extrinsics.npy",
            b"trusted-extrinsics",
        )
        _add_file(
            object_root,
            files,
            "episode_0000/undistorted_intrinsics.npy",
            b"trusted-intrinsics",
        )
        intrinsics: dict[str, np.ndarray] = {}
        extrinsics: dict[str, np.ndarray] = {}
        for camera, center in camera_centers.items():
            pose = np.eye(4)
            pose[:3, 3] = center
            extrinsics[camera] = pose
            intrinsics[camera] = np.eye(3)
            for filename in (
                "aligned_timestamps.txt",
                "alignment.json",
                "metadata.json",
                "undistorted.mp4",
            ):
                _add_file(
                    object_root,
                    files,
                    f"episode_0000/{camera}/{filename}",
                    f"{object_id}:{camera}:{filename}".encode(),
                )
        calibration[str(episode_root)] = (intrinsics, extrinsics)
        rows.append(
            {
                "object_id": object_id,
                "stratum": "fixture",
                "source_episode_id": index,
                "processing_episode_index": 0,
                "action": "fixture action",
                "status": "success",
                "camera_count": 4,
                "frame_count": 100,
                "tactile_sensor_count": 4,
                "tactile_frame_count": 100,
                "tactile_outputs": list(sensors),
                "output_file_count": len(files),
                "output_total_bytes": sum(int(file["size"]) for file in files),
                "output_tree_sha256": "e" * 64,
                "output_files": files,
            }
        )
    report: dict[str, object] = {
        "schema": (
            "bayesian-phystwin/deform360-official-hub-stage1-processing-report-v1"
        ),
        "schema_version": 1,
        "protocol_id": "deform360-official-hub-visuotactile-v1",
        "protocol_sha256": "1" * 64,
        "preflight_sha256": "2" * 64,
        "download_sha256": "3" * 64,
        "processing_view_sha256": "4" * 64,
        "implementation_revision": "5" * 40,
        "official_processing": {
            "repository": "lhy0807/deform360",
            "revision": "6" * 40,
        },
        "role": "calibration",
        "status": "complete",
        "object_count": 10,
        "success_count": 10,
        "retained_technical_failure_count": 0,
        "processing_parameters": {},
        "objects": rows,
        "physical_backend_contract": {
            "minimum_node_count": 128,
            "status": "pending-reconstruction",
        },
        "information_boundary": {
            "calibration_payload_opened": True,
            "confirmation_payload_opened": False,
            "future_target_opened": False,
            "replacement_performed": False,
            "technical_failures_retained": True,
        },
    }
    report["processing_report_sha256"] = content_id(report)
    return report, calibration


def _build(
    root: Path,
    report: dict[str, object],
    calibration: dict[str, tuple[dict[str, np.ndarray], dict[str, np.ndarray]]],
    *,
    schedule_recovery_lock: dict[str, object] | None = None,
) -> dict[str, object]:
    provider, execution = _locks()
    return build_deform360_official_hub_causal_window_manifest(
        processing_report=report,
        processed_root=root,
        provider_lock=provider,  # type: ignore[arg-type]
        execution_lock=execution,
        implementation_revision="a" * 40,
        camera_calibration_loader=lambda path: calibration[str(path)],
        schedule_recovery_lock=schedule_recovery_lock,
    )


def test_builds_content_addressed_causal_windows_and_camera_panels(
    tmp_path: Path,
) -> None:
    report, calibration = _fixture(tmp_path)

    manifest = _build(tmp_path, report, calibration)

    assert manifest["status"] == "complete"
    assert manifest["success_count"] == 10
    assert manifest["retained_technical_failure_count"] == 0
    first = manifest["cases"][0]
    assert first["camera_panel"] == ["camera-a", "camera-b", "camera-c"]
    assert first["reference_camera"] == "camera-a"
    assert first["causal_window"] == {
        "contact_start_frame": 50,
        "source_start_frame": 14,
        "causal_cutoff_frame": 56,
        "future_stop_frame": 80,
        "total_episode_frames": 100,
        "observed_frame_count": 42,
        "future_frame_count": 24,
        "processing_frame_count": 66,
    }
    assert first["provider_windows"] == [
        {"window_index": 0, "frame_start": 14, "frame_stop_exclusive": 39},
        {"window_index": 1, "frame_start": 31, "frame_stop_exclusive": 56},
    ]
    assert first["untouched_future"] == {
        "frame_start": 56,
        "frame_stop_exclusive": 80,
    }
    assert len(first["bound_input_files"]) == 18
    assert manifest["information_boundary"]["calibration_scores_opened"] is False

    output = tmp_path / "manifest.json"
    save_deform360_official_hub_causal_window_manifest(output, manifest)
    assert load_deform360_official_hub_causal_window_manifest(output) == manifest


def test_no_contact_is_retained_without_replacement(tmp_path: Path) -> None:
    report, calibration = _fixture(tmp_path, no_contact_object=3)

    manifest = _build(tmp_path, report, calibration)

    assert manifest["status"] == "complete_with_retained_technical_failures"
    assert manifest["success_count"] == 9
    assert manifest["retained_technical_failure_count"] == 1
    failure = next(case for case in manifest["cases"] if case["status"] != "success")
    assert failure["object_id"] == "calibration-03"
    assert failure["error_type"] == "ValueError"
    assert "no tactile contact" in failure["error_message"]
    assert manifest["replacement_performed"] is False


def test_v2_schedule_waits_for_full_history_after_early_contact() -> None:
    tactile = np.zeros((80, 16, 32), dtype=np.float32)
    tactile[3, 0, :2] = 1.0

    window = derive_deform360_causal_window_v2(
        {"left_left": tactile},
        total_episode_frames=80,
    )

    assert window.contact_start_frame == 3
    assert window.source_start_frame == 0
    assert window.causal_cutoff_frame == 42
    assert window.future_stop_frame == 66


def test_v2_manifest_is_separately_locked_and_supports_early_contact(
    tmp_path: Path,
) -> None:
    report, calibration = _fixture(tmp_path, contact_frame=3)
    schedule_lock = _schedule_lock()

    manifest = _build(
        tmp_path,
        report,
        calibration,
        schedule_recovery_lock=schedule_lock,
    )

    assert (
        validate_deform360_causal_schedule_recovery_lock(schedule_lock)
        == (schedule_lock["artifact_id"])
    )
    assert manifest["status"] == "complete"
    assert manifest["success_count"] == 10
    assert manifest["causal_schedule_recovery_lock_id"] == schedule_lock["artifact_id"]
    first = manifest["cases"][0]
    assert first["causal_window"]["contact_start_frame"] == 3
    assert first["causal_window"]["source_start_frame"] == 0
    assert first["causal_window"]["causal_cutoff_frame"] == 42

    output = tmp_path / "manifest-v2.json"
    save_deform360_official_hub_causal_window_manifest_v2(output, manifest)
    assert load_deform360_official_hub_causal_window_manifest_v2(output) == manifest


def test_file_drift_is_fatal_custody_error(tmp_path: Path) -> None:
    report, calibration = _fixture(tmp_path)
    path = (
        tmp_path / "calibration-00/episode_0000/"
        "brics-odroid_tactilel_left/synced_tactile.npy"
    )
    path.write_bytes(path.read_bytes() + b"drift")

    with pytest.raises(Deform360CustodyError, match="file drift"):
        _build(tmp_path, report, calibration)


def test_execution_lock_is_bound_to_provider_and_camera_policy() -> None:
    _, execution = _locks()
    assert (
        validate_deform360_visual_execution_lock(execution) == execution["artifact_id"]
    )

    mutated = json.loads(json.dumps(execution))
    mutated["camera_panel_policy"]["artifact_id"] = "f" * 64
    canonical = dict(mutated)
    canonical.pop("artifact_id")
    mutated["artifact_id"] = content_id(canonical)
    with pytest.raises(ValueError, match="execution lock identity"):
        validate_deform360_visual_execution_lock(mutated)
