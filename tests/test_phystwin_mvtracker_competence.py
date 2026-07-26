from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pytest

from bayesian_phystwin.phystwin_mvtracker_competence import (
    PREDICTION_FILENAME,
    PREDICTION_REPORT_FILENAME,
    PhysTwinMVTrackerCompetenceConfig,
    evaluate_competence,
    file_sha256,
    metric_observation_variance_m2,
    prepare_source_artifacts,
    seal_prediction,
    validate_query_input,
    write_prediction_artifact,
)


def _write_source(
    root: Path,
    tracks: np.ndarray,
) -> tuple[Path, Path]:
    root.mkdir(parents=True, exist_ok=True)
    manual = root / "gt_track_3d.pkl"
    split = root / "split.json"
    with manual.open("wb") as stream:
        pickle.dump(tracks, stream)
    split.write_text(
        json.dumps({"train": [0, 121], "test": [121, len(tracks)]}),
        encoding="utf-8",
    )
    return manual, split


def test_source_artifact_retains_only_frozen_prefix(tmp_path: Path) -> None:
    tracks = np.zeros((140, 9, 3), dtype=float)
    tracks[90:121, :, 0] = np.arange(31)[:, None] * 0.001
    manual, split = _write_source(tmp_path, tracks)
    first = prepare_source_artifacts(manual, split, tmp_path / "first")

    tracks[121:] = 999.0
    manual_mutated, split_mutated = _write_source(tmp_path / "mutated", tracks)
    second = prepare_source_artifacts(
        manual_mutated,
        split_mutated,
        tmp_path / "second",
    )
    assert (
        first["prediction_input"]["query_array_sha256"]
        == second["prediction_input"]["query_array_sha256"]
    )
    assert (
        first["withheld_evaluation"]["target_array_sha256"]
        == second["withheld_evaluation"]["target_array_sha256"]
    )


def test_validate_query_rejects_hash_change(tmp_path: Path) -> None:
    tracks = np.zeros((140, 9, 3), dtype=float)
    manual, split = _write_source(tmp_path, tracks)
    report = prepare_source_artifacts(manual, split, tmp_path / "source")
    query_path = Path(report["prediction_input"]["path"])
    query, ids = validate_query_input(
        query_path,
        report["prediction_input"]["sha256"],
    )
    assert query.shape == (4, 3)
    assert np.array_equal(ids, [3, 4, 6, 8])
    with query_path.open("ab") as stream:
        stream.write(b"x")
    with pytest.raises(ValueError, match="hash changed"):
        validate_query_input(
            query_path,
            report["prediction_input"]["sha256"],
        )


def test_variance_does_not_use_state_innovation() -> None:
    visibility = np.asarray([[1.0, 0.5], [1.0, 0.5]])
    correction = np.asarray([[0.0, 0.0, 0.0], [0.003, 0.0, 0.0]])
    variance = metric_observation_variance_m2(
        visibility,
        correction,
        standard_deviation_floor_m=0.005,
    )
    assert variance.shape == (2, 2)
    assert np.array_equal(variance[0], variance[1])
    assert variance[0, 1] > variance[0, 0]


def _write_prediction(
    root: Path,
    target: np.ndarray,
    visibility: np.ndarray,
    config: PhysTwinMVTrackerCompetenceConfig,
) -> Path:
    prediction = root / "prediction"
    write_prediction_artifact(
        prediction,
        raw_tracker_m=target,
        visibility_probability=visibility,
        query_points_world_m=target[0],
        identity_ids=np.asarray(config.selected_identity_ids),
        input_provenance={"query_input_sha256": "q"},
        runtime_provenance={"device": "test"},
        implementation_sha256={"runner": "r", "adapter": "a", "protocol": "p"},
        config=config,
    )
    seal_prediction(prediction)
    return prediction


def test_competence_gate_passes_accurate_motion(tmp_path: Path) -> None:
    config = PhysTwinMVTrackerCompetenceConfig(selected_identity_ids=(0, 1, 2))
    target = np.zeros((config.prefix_frame_count, 3, 3), dtype=np.float32)
    target[:, :, 0] = np.arange(config.prefix_frame_count)[:, None] * 0.001
    visibility = np.ones(target.shape[:2], dtype=np.float32)
    prediction = _write_prediction(tmp_path, target.copy(), visibility, config)
    withheld = tmp_path / "withheld.npz"
    np.savez_compressed(
        withheld,
        target_tracks_world_m=target,
        identity_ids=np.arange(3),
        source_frame_start=np.asarray(config.source_frame_start),
        source_frame_end_exclusive=np.asarray(
            config.source_frame_end_exclusive
        ),
    )
    result = evaluate_competence(
        prediction,
        withheld,
        file_sha256(withheld),
        tmp_path / "result.json",
        config=config,
    )
    assert result["competence_gate_passed"] is True
    assert result["decision"] == "advance-to-separately-locked-assimilation-smoke"


def test_competence_gate_stops_persistence_copy(tmp_path: Path) -> None:
    config = PhysTwinMVTrackerCompetenceConfig(selected_identity_ids=(0, 1, 2))
    target = np.zeros((config.prefix_frame_count, 3, 3), dtype=np.float32)
    target[:, :, 0] = np.arange(config.prefix_frame_count)[:, None] * 0.001
    candidate = np.repeat(target[:1], config.prefix_frame_count, axis=0)
    visibility = np.ones(target.shape[:2], dtype=np.float32)
    prediction = _write_prediction(tmp_path, candidate, visibility, config)
    withheld = tmp_path / "withheld.npz"
    np.savez_compressed(
        withheld,
        target_tracks_world_m=target,
        identity_ids=np.arange(3),
        source_frame_start=np.asarray(config.source_frame_start),
        source_frame_end_exclusive=np.asarray(
            config.source_frame_end_exclusive
        ),
    )
    result = evaluate_competence(
        prediction,
        withheld,
        file_sha256(withheld),
        tmp_path / "result.json",
        config=config,
    )
    assert result["competence_gate_passed"] is False
    assert result["gates"]["relative_gain_over_persistence"] is False


def test_seal_detects_prediction_archive_mutation(tmp_path: Path) -> None:
    config = PhysTwinMVTrackerCompetenceConfig(selected_identity_ids=(0, 1, 2))
    target = np.zeros((config.prefix_frame_count, 3, 3), dtype=np.float32)
    visibility = np.ones(target.shape[:2], dtype=np.float32)
    prediction = tmp_path / "prediction"
    write_prediction_artifact(
        prediction,
        raw_tracker_m=target,
        visibility_probability=visibility,
        query_points_world_m=target[0],
        identity_ids=np.arange(3),
        input_provenance={},
        runtime_provenance={},
        implementation_sha256={},
        config=config,
    )
    with (prediction / PREDICTION_FILENAME).open("ab") as stream:
        stream.write(b"x")
    with pytest.raises(ValueError, match="archive hash"):
        seal_prediction(prediction)
    assert (prediction / PREDICTION_REPORT_FILENAME).is_file()
