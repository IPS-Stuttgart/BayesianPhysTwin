import csv
import json
import pickle
from pathlib import Path

import numpy as np

from bayesian_phystwin import (
    PhysTwinExportConfig,
    export_phystwin_residuals,
    replay_residual_csv,
    write_export_summary,
)


def _write_fixture(directory: Path) -> tuple[Path, Path, np.ndarray, np.ndarray]:
    frame_count = 4
    track_count = 3
    observed = np.zeros((frame_count, track_count, 3), dtype=float)
    observed[0, :, 0] = [0.0, 1.0, 2.0]
    for frame in range(1, frame_count):
        observed[frame] = observed[0] + np.array([0.1 * frame, 0.02 * frame, 0.0])

    trajectory = np.zeros((frame_count, 5, 3), dtype=float)
    trajectory[:, :track_count] = observed
    trajectory[1:, 1, 1] += 0.01
    visible = np.ones((frame_count, track_count), dtype=bool)
    visible[2, 1] = False
    motion_valid = np.array(
        [
            [True, True, False],
            [True, False, True],
            [True, True, True],
            [False, False, False],
        ]
    )
    final_data = {
        "object_points": observed,
        "object_visibilities": visible,
        "object_motions_valid": motion_valid,
    }
    final_data_path = directory / "final_data.pkl"
    trajectory_path = directory / "inference.pkl"
    with final_data_path.open("wb") as handle:
        pickle.dump(final_data, handle)
    with trajectory_path.open("wb") as handle:
        pickle.dump(trajectory, handle)
    return final_data_path, trajectory_path, observed, trajectory


def test_direct_export_matches_phystwin_motion_valid_contract(tmp_path: Path) -> None:
    final_data_path, trajectory_path, _, _ = _write_fixture(tmp_path)
    output_csv = tmp_path / "residuals.csv"

    summary = export_phystwin_residuals(
        final_data_path,
        trajectory_path,
        output_csv,
    )
    with output_csv.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert summary["exported_measurement_count"] == 7
    assert summary["skipped_invalid_count"] == 2
    assert summary["initial_alignment_rmse"] == 0.0
    assert {(row["frame"], row["track_id"]) for row in rows} == {
        ("1", "0"),
        ("1", "1"),
        ("2", "0"),
        ("2", "2"),
        ("3", "0"),
        ("3", "1"),
        ("3", "2"),
    }
    assert all(row["track_valid"] == "true" for row in rows)

    replay = replay_residual_csv(output_csv)
    assert replay.summary["measurement_count"] == 7
    assert replay.summary["measurement_dimension"] == 3


def test_export_includes_invalid_rows_and_sidecar_cues(tmp_path: Path) -> None:
    final_data_path, trajectory_path, observed, _ = _write_fixture(tmp_path)
    cues_path = tmp_path / "cues.npz"
    confidence = np.full((observed.shape[0] - 1, observed.shape[1]), 0.8)
    boundary = np.full_like(confidence, 0.03)
    flow = np.full_like(confidence, 0.02)
    np.savez(
        cues_path,
        confidence=confidence,
        boundary_distance=boundary,
        flow_inconsistency=flow,
    )
    output_csv = tmp_path / "residuals_all.csv"

    summary = export_phystwin_residuals(
        final_data_path,
        trajectory_path,
        output_csv,
        cues_path=cues_path,
        config=PhysTwinExportConfig(include_invalid=True),
    )
    with output_csv.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 9
    assert summary["cue_fields"] == [
        "boundary_distance",
        "confidence",
        "flow_inconsistency",
    ]
    assert rows[0]["confidence"] == "0.8"
    assert rows[0]["boundary_distance"] == "0.03"
    occluded_row = next(row for row in rows if row["frame"] == "2" and row["track_id"] == "1")
    assert occluded_row["occluded"] == "true"
    assert occluded_row["track_valid"] == "false"


def test_nearest_correspondence_records_vertex_mapping(tmp_path: Path) -> None:
    final_data_path, trajectory_path, _, trajectory = _write_fixture(tmp_path)
    permutation = [2, 0, 1, 3, 4]
    permuted = trajectory[:, permutation]
    with trajectory_path.open("wb") as handle:
        pickle.dump(permuted, handle)
    output_csv = tmp_path / "nearest.csv"

    summary = export_phystwin_residuals(
        final_data_path,
        trajectory_path,
        output_csv,
        config=PhysTwinExportConfig(correspondence="nearest"),
    )
    with output_csv.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    mapping = {int(row["track_id"]): int(row["vertex_id"]) for row in rows}
    assert mapping == {0: 1, 1: 2, 2: 0}
    assert summary["initial_alignment_max_norm"] == 0.0


def test_export_summary_is_json_serializable(tmp_path: Path) -> None:
    final_data_path, trajectory_path, _, _ = _write_fixture(tmp_path)
    summary_path = tmp_path / "export.json"
    summary = export_phystwin_residuals(
        final_data_path,
        trajectory_path,
        tmp_path / "residuals.csv",
    )

    write_export_summary(summary, summary_path)

    loaded = json.loads(summary_path.read_text(encoding="utf-8"))
    assert loaded["schema_version"] == 1
    assert loaded["config"]["correspondence"] == "direct"
