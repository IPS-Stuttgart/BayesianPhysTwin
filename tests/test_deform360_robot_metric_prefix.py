from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from bayesian_phystwin._portable_contracts import content_id
from bayesian_phystwin.deform360_calibration_visual_execution_admission import (
    DEFORM360_PREPARED_SOURCE_INVENTORY_CLAIM_BOUNDARY,
    DEFORM360_PREPARED_SOURCE_INVENTORY_SCHEMA,
    DEFORM360_PREPARED_SOURCE_INVENTORY_SEMANTICS,
    DEFORM360_PREPARED_SOURCE_INVENTORY_STATUS,
    DEFORM360_PREPARED_SOURCE_INVENTORY_VERSION,
)
from bayesian_phystwin.deform360_robot_metric_prefix import (
    METRIC_PREFIX_FILENAME,
    materialize_deform360_robot_metric_prefix,
    validate_deform360_robot_metric_prefix,
)

OBJECT_ID = "object-00"
CAMERA_ID = "camera-0"
PROCESSING_REVISION = "3" * 40
INFORMATION_BOUNDARY = {
    "calibration_camera_payloads_opened": True,
    "calibration_tactile_payloads_opened": True,
    "calibration_robot_state_opened": True,
    "calibration_target_metrics_computed": False,
    "confirmation_payloads_opened": False,
    "target_outcomes_used": False,
    "replacement_allowed": False,
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _record(path: Path, *, root: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": _sha256(path),
        "byte_count": path.stat().st_size,
    }


def _dummy_record(name: str = "unused") -> dict[str, Any]:
    return {"path": name, "sha256": "a" * 64, "byte_count": 1}


def _camera(camera_id: str = CAMERA_ID) -> dict[str, Any]:
    record = _dummy_record()
    return {
        "camera": camera_id,
        "video": record,
        "preview": record,
        "timestamps": record,
        "alignment": record,
        "metadata": record,
        "frame_count": 81,
        "width": 640,
        "height": 320,
        "fps": 30.0,
        "timeline_sha256": "b" * 64,
    }


def _write_sources(root: Path) -> dict[str, dict[str, Any]]:
    episode = root / OBJECT_ID / "episode_0000"
    robot_path = episode / "robot" / "robot.npz"
    robot_path.parent.mkdir(parents=True)
    poses = np.tile(np.eye(4, dtype=np.float64), (81, 1, 1))
    poses[:, 2, 3] = 1.0
    # The reserved future is deliberately non-finite. A causal exporter must
    # not use it.
    poses[58:] = np.nan
    openings = np.full(81, 0.06, dtype=np.float64)
    openings[58:] = np.nan
    np.savez(
        robot_path,
        actions=np.zeros((81, 5, 3), dtype=np.float64),
        T_worlds=poses,
        openings=openings,
        bimanual=np.asarray(False, dtype=np.bool_),
    )
    intrinsics_path = episode / "undistorted_intrinsics.npy"
    extrinsics_path = episode / "extrinsics.npy"
    intrinsics = np.asarray(
        [[900.0, 0.0, 320.0], [0.0, 900.0, 160.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    np.save(intrinsics_path, {CAMERA_ID: intrinsics}, allow_pickle=True)
    np.save(extrinsics_path, {CAMERA_ID: np.eye(4)}, allow_pickle=True)
    return {
        "robot": _record(robot_path, root=root),
        "undistorted_intrinsics": _record(intrinsics_path, root=root),
        "extrinsics": _record(extrinsics_path, root=root),
    }


def _fixture(tmp_path: Path) -> tuple[Path, Path]:
    processed = tmp_path / "processed"
    processed.mkdir()
    episode_files = _write_sources(processed)
    objects = []
    for index in range(10):
        object_id = f"object-{index:02d}"
        objects.append(
            {
                "object_id": object_id,
                "episode_id": 0,
                "stratum": "sheet" if index < 5 else "volumetric",
                "synthetic_episode_index": 0,
                "aligned_frame_count": 81,
                "action_window": {
                    "selected_raw_frame_range_half_open": [0, 81],
                    "prediction_raw_frame_range_half_open": [0, 76],
                    "prefix_raw_frame_range_half_open": [0, 58],
                },
                "episode_files": (
                    episode_files
                    if object_id == OBJECT_ID
                    else {
                        "robot": _dummy_record(f"{object_id}/robot.npz"),
                        "undistorted_intrinsics": _dummy_record(
                            f"{object_id}/intrinsics.npy"
                        ),
                        "extrinsics": _dummy_record(f"{object_id}/extrinsics.npy"),
                    }
                ),
                "cameras": [_camera()],
                "tactile": [],
            }
        )
    identity = {
        "schema": DEFORM360_PREPARED_SOURCE_INVENTORY_SCHEMA,
        "schema_version": DEFORM360_PREPARED_SOURCE_INVENTORY_VERSION,
        "semantics": DEFORM360_PREPARED_SOURCE_INVENTORY_SEMANTICS,
        "status": DEFORM360_PREPARED_SOURCE_INVENTORY_STATUS,
        "implementation_revision": "1" * 40,
        "calibration_source_revision": "2" * 40,
        "processing_revision": PROCESSING_REVISION,
        "selection_artifact_sha256": "4" * 64,
        "visual_provider_lock_id": "5" * 64,
        "calibration_source_run_record_sha256": "6" * 64,
        "object_count": 10,
        "objects": objects,
        "source_artifacts": {"sources/calibration-source/result.json": "7" * 64},
        "information_boundary": INFORMATION_BOUNDARY,
        "claim_boundary": DEFORM360_PREPARED_SOURCE_INVENTORY_CLAIM_BOUNDARY,
    }
    inventory = {**identity, "inventory_id": content_id(identity)}
    inventory_path = tmp_path / "prepared-source-inventory.json"
    inventory_path.write_text(
        json.dumps(inventory, sort_keys=True) + "\n", encoding="utf-8"
    )
    return inventory_path, processed


def _materialize(tmp_path: Path) -> tuple[dict[str, Any], Path]:
    inventory, processed = _fixture(tmp_path)
    output = tmp_path / "metric-prefix"
    result = materialize_deform360_robot_metric_prefix(
        prepared_source_inventory_path=inventory,
        processed_root=processed,
        object_id=OBJECT_ID,
        camera_id=CAMERA_ID,
        expected_processing_revision=PROCESSING_REVISION,
        target_height=160,
        target_width=320,
        output_directory=output,
    )
    return result, output


def test_robot_metric_prefix_is_causal_metric_and_camera_specific(
    tmp_path: Path,
) -> None:
    result, output = _materialize(tmp_path)

    assert result["camera_id"] == CAMERA_ID
    assert result["causal_frame_range_half_open"] == [0, 58]
    assert result["information_boundary"]["future_frames_used"] is False
    assert result["information_boundary"]["rendered_depth_opened"] is False
    assert result["information_boundary"]["human_approval_required"] is False
    assert result["projected_point_count"] > 0
    with np.load(output / METRIC_PREFIX_FILENAME, allow_pickle=False) as stored:
        np.testing.assert_array_equal(stored["frame_indices"], np.arange(58))
        assert stored["points_world_m"].shape == (58, 160, 320, 3)
        assert stored["valid_mask"].shape == (58, 160, 320)
        selected = np.asarray(stored["points_world_m"])[stored["valid_mask"]]
        assert np.all(selected[:, 2] < 1.1)
    validate_deform360_robot_metric_prefix(output)


def test_robot_metric_prefix_matches_motioncrafter_cover_crop(tmp_path: Path) -> None:
    result, output = _materialize(tmp_path)
    calibration = json.loads(
        (output / "metric-calibration.json").read_text(encoding="utf-8")
    )

    assert result["source_image_shape"] == [320, 640]
    assert result["target_image_shape"] == [160, 320]
    assert calibration["cover_resize"]["scale"] == 0.5
    assert calibration["cover_resize"]["crop_top"] == 0
    assert calibration["cover_resize"]["crop_left"] == 0


def test_robot_metric_prefix_rejects_processing_revision_drift(
    tmp_path: Path,
) -> None:
    inventory, processed = _fixture(tmp_path)
    with pytest.raises(ValueError, match="processing revision changed"):
        materialize_deform360_robot_metric_prefix(
            prepared_source_inventory_path=inventory,
            processed_root=processed,
            object_id=OBJECT_ID,
            camera_id=CAMERA_ID,
            expected_processing_revision="f" * 40,
            target_height=160,
            target_width=320,
            output_directory=tmp_path / "metric-prefix",
        )


def test_robot_metric_prefix_detects_array_tampering(tmp_path: Path) -> None:
    _result, output = _materialize(tmp_path)
    (output / METRIC_PREFIX_FILENAME).write_bytes(b"changed")
    with pytest.raises(ValueError, match="digest changed"):
        validate_deform360_robot_metric_prefix(output)
