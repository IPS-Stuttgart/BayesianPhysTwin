from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "science"
    / "run_deform360_public_evaluation.py"
)
_SPEC = importlib.util.spec_from_file_location("_deform360_public_evaluation", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)

EvaluationLimits = _MODULE.EvaluationLimits
evaluate_deform360_public_data = _MODULE.evaluate_deform360_public_data
write_evaluation = _MODULE.write_evaluation


def _constant_velocity_trajectory() -> np.ndarray:
    base = np.array(
        [
            [0.00, 0.00, 0.00],
            [0.02, 0.00, 0.00],
            [0.00, 0.02, 0.00],
            [0.02, 0.02, 0.00],
        ],
        dtype=np.float64,
    )
    velocity = np.array([0.003, -0.001, 0.002], dtype=np.float64)
    return np.stack([base + frame * velocity for frame in range(8)])


def test_fixed_identity_trajectory_is_evaluated(tmp_path: Path) -> None:
    episode = tmp_path / "processed" / "001-rope" / "episode_0000"
    episode.mkdir(parents=True)
    trajectory = _constant_velocity_trajectory()
    np.savez_compressed(
        episode / "control_points.npz",
        positions_world_m=trajectory,
        valid_mask=np.ones(trajectory.shape[:2], dtype=bool),
    )

    result = evaluate_deform360_public_data(
        tmp_path,
        limits=EvaluationLimits(
            max_archives=4,
            max_frames_per_archive=16,
            max_tracks=16,
        ),
        revision="revision-test",
    )

    assert result["inventory"]["archives_evaluated"] == 1
    assert result["cases"][0]["object_id"] == "001-rope"
    assert result["cases"][0]["representation"] == "fixed_identity_trajectory"
    summary = result["summary"]["fixed_identity_trajectory"]
    assert summary["identity_rmse_m"]["last_residual"] < 1e-12
    assert (
        summary["identity_rmse_m"]["last_residual"]
        < summary["identity_rmse_m"]["persistence"]
    )
    assert len(result["result_sha256"]) == 64


def test_declared_millimeter_trajectory_is_scaled(tmp_path: Path) -> None:
    trajectory_mm = _constant_velocity_trajectory() * 1000.0
    np.savez_compressed(tmp_path / "positions.npz", positions_mm=trajectory_mm)

    result = evaluate_deform360_public_data(
        tmp_path,
        limits=EvaluationLimits(max_archives=1, max_tracks=2),
    )

    case = result["cases"][0]
    assert case["unit_source"] == "declared_mm"
    assert case["track_count"] == 2
    assert (
        result["summary"]["fixed_identity_trajectory"]["identity_rmse_m"][
            "last_residual"
        ]
        < 1e-12
    )


def test_packed_hulls_use_correspondence_free_scoring(tmp_path: Path) -> None:
    archive = tmp_path / "observations" / "002-rope-silk" / "sampled_hulls.npz"
    archive.parent.mkdir(parents=True)
    base = np.array(
        [
            [0.00, 0.00, 0.00],
            [0.02, 0.00, 0.00],
            [0.00, 0.02, 0.00],
            [0.02, 0.02, 0.00],
        ],
        dtype=np.float64,
    )
    translation = np.array([0.002, 0.0, 0.001], dtype=np.float64)
    hulls = tuple(base + frame * translation for frame in range(7))
    offsets = np.arange(
        0,
        (len(hulls) + 1) * len(base),
        len(base),
        dtype=np.int64,
    )
    np.savez_compressed(
        archive,
        frame_indices=np.arange(len(hulls), dtype=np.int32),
        point_offsets=offsets,
        points_world_m=np.concatenate(hulls, axis=0),
    )

    result = evaluate_deform360_public_data(
        tmp_path,
        limits=EvaluationLimits(max_archives=4, max_frames_per_archive=16),
    )

    assert result["inventory"]["archives_evaluated"] == 1
    case = result["cases"][0]
    assert case["representation"] == "packed_visual_hulls"
    summary = result["summary"]["packed_visual_hulls"]
    assert summary["centroid_error_m"]["last_residual"] < 1e-12
    assert summary["chamfer_rmse_m"]["last_residual"] < 1e-12


def test_unsupported_and_invalid_archives_are_reported(tmp_path: Path) -> None:
    np.savez_compressed(
        tmp_path / "unknown.npz",
        actions=np.zeros((5, 2, 3), dtype=np.float64),
    )
    (tmp_path / "hull-invalid.npz").write_bytes(b"not an npz file")

    result = evaluate_deform360_public_data(tmp_path)

    assert result["inventory"]["archives_evaluated"] == 0
    statuses = {
        record["path"]: record["status"]
        for record in result["inventory"]["inspection"]
    }
    assert statuses["hull-invalid.npz"] == "invalid"
    assert statuses["unknown.npz"] == "unsupported"


def test_empty_visibility_does_not_emit_nan_metrics(tmp_path: Path) -> None:
    trajectory = _constant_velocity_trajectory()
    np.savez_compressed(
        tmp_path / "tracks.npz",
        tracks=trajectory,
        valid=np.zeros(trajectory.shape[:2], dtype=bool),
    )

    result = evaluate_deform360_public_data(tmp_path)

    assert result["inventory"]["archives_evaluated"] == 0
    assert result["summary"] == {}


def test_result_is_stable_and_writable(tmp_path: Path) -> None:
    trajectory = _constant_velocity_trajectory()
    np.savez_compressed(tmp_path / "trajectory.npz", trajectory=trajectory)
    limits = EvaluationLimits(max_archives=1, max_frames_per_archive=8)

    first = evaluate_deform360_public_data(
        tmp_path,
        limits=limits,
        revision="revision-test",
    )
    second = evaluate_deform360_public_data(
        tmp_path,
        limits=limits,
        revision="revision-test",
    )
    assert first["result_sha256"] == second["result_sha256"]

    output = tmp_path / "result.json"
    write_evaluation(output, first)
    assert output.is_file()
    assert output.read_text(encoding="utf-8").endswith("\n")


@pytest.mark.parametrize(
    "field",
    ("max_archives", "max_frames_per_archive", "max_tracks"),
)
def test_limits_must_be_positive_integers(field: str) -> None:
    values = {
        "max_archives": 1,
        "max_frames_per_archive": 1,
        "max_tracks": 1,
    }
    values[field] = 0
    with pytest.raises(ValueError, match="positive integer"):
        EvaluationLimits(**values)


def test_missing_dataset_root_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="data root is missing"):
        evaluate_deform360_public_data(tmp_path / "missing")
