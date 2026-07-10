import json
import pickle
from pathlib import Path

import numpy as np

from bayesian_phystwin.phystwin_official_evaluation import (
    evaluate_official_phystwin_arrays,
    evaluate_official_phystwin_files,
    evaluate_official_phystwin_interval,
    write_official_evaluation,
)


def _official_fixture() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    vertices = np.array(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            [[0.1, 0.0, 0.0], [1.1, 0.0, 0.0]],
            [[0.2, 0.0, 0.0], [1.2, 0.0, 0.0]],
            [[0.3, 0.0, 0.0], [1.3, 0.0, 0.0]],
        ]
    )
    object_points = vertices.copy()
    object_points[2, 0, 1] = 0.2
    visibility = np.ones((4, 2), dtype=bool)
    visibility[2, 1] = False
    gt_track_3d = vertices.copy()
    gt_track_3d[2, 0, 1] = 0.2
    gt_track_3d[2, 1] = np.nan
    return vertices, object_points, visibility, gt_track_3d


def test_official_metrics_match_released_averaging_contract() -> None:
    vertices, object_points, visibility, gt_track_3d = _official_fixture()

    result = evaluate_official_phystwin_arrays(
        vertices,
        object_points,
        visibility,
        gt_track_3d,
        num_surface_points=2,
        train_frame=3,
        test_frame=4,
    )

    assert np.isclose(result["train"]["chamfer_distance_m"], 0.1)
    assert np.isclose(result["train"]["track_error_m"], 0.1)
    assert result["test"]["chamfer_distance_m"] == 0.0
    assert result["test"]["track_error_m"] == 0.0


def test_official_interval_supports_causal_validation_slice() -> None:
    vertices, object_points, visibility, gt_track_3d = _official_fixture()

    result = evaluate_official_phystwin_interval(
        vertices,
        object_points,
        visibility,
        gt_track_3d,
        num_surface_points=2,
        start_frame=2,
        end_frame=3,
    )

    assert result["frame_count"] == 1
    assert np.isclose(result["chamfer_distance_m"], 0.2)
    assert np.isclose(result["track_error_m"], 0.2)


def test_file_evaluation_records_hashes_and_split(tmp_path: Path) -> None:
    vertices, object_points, visibility, gt_track_3d = _official_fixture()
    paths = {
        "trajectory": tmp_path / "trajectory.pkl",
        "final_data": tmp_path / "final_data.pkl",
        "gt_track": tmp_path / "gt_track_3d.pkl",
        "split": tmp_path / "split.json",
    }
    with paths["trajectory"].open("wb") as handle:
        pickle.dump(vertices, handle)
    with paths["final_data"].open("wb") as handle:
        pickle.dump(
            {
                "object_points": object_points,
                "object_visibilities": visibility,
                "surface_points": np.empty((0, 3)),
            },
            handle,
        )
    with paths["gt_track"].open("wb") as handle:
        pickle.dump(gt_track_3d, handle)
    paths["split"].write_text(
        json.dumps({"frame_len": 4, "train": [0, 3], "test": [3, 4]}),
        encoding="utf-8",
    )

    summary = evaluate_official_phystwin_files(
        paths["trajectory"],
        paths["final_data"],
        paths["gt_track"],
        paths["split"],
    )
    output_path = tmp_path / "evaluation.json"
    write_official_evaluation(summary, output_path)
    loaded = json.loads(output_path.read_text(encoding="utf-8"))

    assert loaded["schema_version"] == 1
    assert loaded["split"]["test"] == [3, 4]
    assert len(loaded["inputs"]["gt_track_3d"]["sha256"]) == 64
    assert loaded["evaluation"]["train"]["track_error_m"] == summary[
        "evaluation"
    ]["train"]["track_error_m"]
