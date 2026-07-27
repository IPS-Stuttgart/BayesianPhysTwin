from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pytest

from bayesian_phystwin.phystwin_conditioned_dino_competence import (
    PREDICTION_FILENAME,
    PhysTwinConditionedDinoCompetenceConfig,
    evaluate_competence,
    prepare_source_artifacts,
    seal_prediction,
    validate_prediction_input,
    write_prediction_artifact,
)
from bayesian_phystwin.phystwin_mvtracker_competence import file_sha256


def _config() -> PhysTwinConditionedDinoCompetenceConfig:
    return PhysTwinConditionedDinoCompetenceConfig(
        reference_frame=2,
        source_frame_end_exclusive=7,
        selected_identity_ids=(0, 1, 2),
        endpoint_frame_count=2,
    )


def _write_pickle(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        pickle.dump(value, stream)


def _write_source(
    root: Path,
    tracks: np.ndarray,
    physical: np.ndarray,
) -> tuple[Path, Path, Path]:
    manual_path = root / "gt_track_3d.pkl"
    physical_path = root / "inference.pkl"
    split_path = root / "split.json"
    _write_pickle(manual_path, tracks)
    _write_pickle(physical_path, physical)
    split_path.write_text(
        json.dumps({"train": [0, 7], "test": [7, len(tracks)]}),
        encoding="utf-8",
    )
    return manual_path, split_path, physical_path


def _source_arrays() -> tuple[np.ndarray, np.ndarray]:
    tracks = np.zeros((9, 3, 3), dtype=np.float32)
    tracks[:, :, 0] = np.arange(9)[:, None] * 0.002
    tracks[:, :, 1] = np.arange(3)[None] * 0.020
    physical = tracks.copy()
    physical[:, :, 0] += 0.004
    return tracks, physical


def test_source_artifact_hides_manual_frames_after_reference(
    tmp_path: Path,
) -> None:
    config = _config()
    tracks, physical = _source_arrays()
    manual, split, trajectory = _write_source(
        tmp_path / "first_input",
        tracks,
        physical,
    )
    first = prepare_source_artifacts(
        manual,
        split,
        trajectory,
        tmp_path / "first",
        config=config,
    )

    mutated = tracks.copy()
    mutated[config.reference_frame + 1 :] += 100.0
    manual, split, trajectory = _write_source(
        tmp_path / "second_input",
        mutated,
        physical,
    )
    second = prepare_source_artifacts(
        manual,
        split,
        trajectory,
        tmp_path / "second",
        config=config,
    )

    assert (
        first["prediction_input"]["query_array_sha256"]
        == second["prediction_input"]["query_array_sha256"]
    )
    assert (
        first["prediction_input"]["physical_array_sha256"]
        == second["prediction_input"]["physical_array_sha256"]
    )
    assert (
        first["withheld_evaluation"]["target_array_sha256"]
        != second["withheld_evaluation"]["target_array_sha256"]
    )


def test_validate_prediction_input_rejects_hash_mutation(tmp_path: Path) -> None:
    config = _config()
    tracks, physical = _source_arrays()
    manual, split, trajectory = _write_source(
        tmp_path / "input",
        tracks,
        physical,
    )
    report = prepare_source_artifacts(
        manual,
        split,
        trajectory,
        tmp_path / "source",
        config=config,
    )
    input_path = Path(report["prediction_input"]["path"])
    query, baseline, vertices, identities = validate_prediction_input(
        input_path,
        report["prediction_input"]["sha256"],
        config=config,
    )
    assert query.shape == (3, 3)
    assert baseline.shape == (5, 3, 3)
    np.testing.assert_array_equal(vertices, identities)

    with input_path.open("ab") as stream:
        stream.write(b"x")
    with pytest.raises(ValueError, match="hash changed"):
        validate_prediction_input(
            input_path,
            report["prediction_input"]["sha256"],
            config=config,
        )


def _write_prediction(
    root: Path,
    observed: np.ndarray,
    physical: np.ndarray,
    accepted: np.ndarray,
    config: PhysTwinConditionedDinoCompetenceConfig,
) -> Path:
    prediction = root / "prediction"
    shape = accepted.shape
    write_prediction_artifact(
        prediction,
        observed_points_world_m=observed,
        observation_covariance_world_m2=np.repeat(
            (1e-4 * np.eye(3, dtype=np.float32))[None, None],
            shape[0],
            axis=0,
        ).repeat(shape[1], axis=1),
        prior_reliability=np.where(accepted, 0.8, 0.0),
        accepted=accepted,
        accepted_view_count=np.where(accepted, 2, 0),
        physical_points_world_m=physical,
        identity_ids=np.arange(shape[1]),
        input_provenance={"prediction_input_sha256": "input"},
        runtime_provenance={"device": "test"},
        implementation_sha256={"runner": "runner", "adapter": "adapter"},
        config=config,
    )
    seal_prediction(prediction)
    return prediction


def _write_withheld(
    path: Path,
    target: np.ndarray,
    config: PhysTwinConditionedDinoCompetenceConfig,
) -> None:
    np.savez_compressed(
        path,
        target_tracks_world_m=target,
        identity_ids=np.arange(target.shape[1]),
        reference_frame=np.asarray(config.reference_frame),
        source_frame_end_exclusive=np.asarray(
            config.source_frame_end_exclusive
        ),
    )


def test_competence_gate_passes_accurate_supported_observations(
    tmp_path: Path,
) -> None:
    config = _config()
    target = np.zeros((config.prefix_frame_count, 3, 3), dtype=np.float32)
    target[:, :, 0] = np.arange(config.prefix_frame_count)[:, None] * 0.002
    physical = target.copy()
    physical[1:, :, 0] += 0.008
    accepted = np.ones(target.shape[:2], dtype=bool)
    prediction = _write_prediction(
        tmp_path,
        target.copy(),
        physical,
        accepted,
        config,
    )
    withheld = tmp_path / "withheld.npz"
    _write_withheld(withheld, target, config)

    result = evaluate_competence(
        prediction,
        withheld,
        file_sha256(withheld),
        tmp_path / "result.json",
        config=config,
    )

    assert result["competence_gate_passed"] is True
    assert result["decision"] == "advance-to-separately-locked-source-panel"


def test_zero_support_fails_cleanly_with_exact_physical_fallback(
    tmp_path: Path,
) -> None:
    config = _config()
    target = np.zeros((config.prefix_frame_count, 3, 3), dtype=np.float32)
    physical = target.copy()
    observed = np.full_like(target, np.nan)
    accepted = np.zeros(target.shape[:2], dtype=bool)
    prediction = _write_prediction(
        tmp_path,
        observed,
        physical,
        accepted,
        config,
    )
    withheld = tmp_path / "withheld.npz"
    _write_withheld(withheld, target, config)

    result = evaluate_competence(
        prediction,
        withheld,
        file_sha256(withheld),
        tmp_path / "result.json",
        config=config,
    )

    assert result["competence_gate_passed"] is False
    assert result["metrics"]["supported_candidate_identity_rmse_m"] is None
    assert result["gates"]["supported_fraction"] is False
    with np.load(prediction / PREDICTION_FILENAME, allow_pickle=False) as stored:
        candidate = stored["candidate_points_world_m"]
        fallback = stored["physical_points_world_m"]
    assert candidate.tobytes() == fallback.tobytes()


def test_seal_detects_prediction_archive_mutation(tmp_path: Path) -> None:
    config = _config()
    target = np.zeros((config.prefix_frame_count, 3, 3), dtype=np.float32)
    accepted = np.ones(target.shape[:2], dtype=bool)
    prediction = tmp_path / "prediction"
    shape = accepted.shape
    write_prediction_artifact(
        prediction,
        observed_points_world_m=target,
        observation_covariance_world_m2=np.repeat(
            (1e-4 * np.eye(3, dtype=np.float32))[None, None],
            shape[0],
            axis=0,
        ).repeat(shape[1], axis=1),
        prior_reliability=np.ones(shape, dtype=np.float32),
        accepted=accepted,
        accepted_view_count=np.full(shape, 2),
        physical_points_world_m=target,
        identity_ids=np.arange(shape[1]),
        input_provenance={},
        runtime_provenance={},
        implementation_sha256={},
        config=config,
    )
    with (prediction / PREDICTION_FILENAME).open("ab") as stream:
        stream.write(b"x")

    with pytest.raises(ValueError, match="archive hash"):
        seal_prediction(prediction)
