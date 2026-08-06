from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pytest

from bayesian_phystwin.phystwin_tapnextpp_competence import (
    PREDICTION_FILENAME,
    PREDICTION_REPORT_FILENAME,
    PhysTwinTAPNextPPCompetenceConfig,
    evaluate_competence,
    file_sha256,
    prepare_source_artifacts,
    seal_prediction,
    validate_prediction_input,
    write_prediction_artifact,
)


def _write_source(
    root: Path,
    tracks: np.ndarray,
    config: PhysTwinTAPNextPPCompetenceConfig,
) -> tuple[Path, Path, Path]:
    root.mkdir(parents=True, exist_ok=True)
    manual = root / "gt_track_3d.pkl"
    split = root / "split.json"
    masks = root / "processed_masks.pkl"
    with manual.open("wb") as stream:
        pickle.dump(tracks, stream)
    split.write_text(
        json.dumps({"train": [0, 121], "test": [121, len(tracks)]}),
        encoding="utf-8",
    )
    mask_payload = {
        frame: {
            camera: {
                "object": np.ones((8, 10), dtype=bool),
                "controller": np.zeros((8, 10), dtype=bool),
            }
            for camera in config.selected_cameras
        }
        for frame in range(
            config.source_frame_start,
            config.source_frame_end_exclusive,
        )
    }
    with masks.open("wb") as stream:
        pickle.dump(mask_payload, stream)
    return manual, split, masks


def test_source_artifacts_retain_only_frozen_prefix(tmp_path: Path) -> None:
    config = PhysTwinTAPNextPPCompetenceConfig()
    tracks = np.zeros((140, 9, 3), dtype=float)
    tracks[
        config.source_frame_start : config.source_frame_end_exclusive,
        :,
        0,
    ] = np.arange(config.prefix_frame_count)[:, None] * 0.001
    source = _write_source(tmp_path / "source", tracks, config)
    first = prepare_source_artifacts(
        *source,
        tmp_path / "first",
        config=config,
    )

    tracks[config.source_frame_end_exclusive :] = 999.0
    mutated = _write_source(tmp_path / "mutated", tracks, config)
    second = prepare_source_artifacts(
        *mutated,
        tmp_path / "second",
        config=config,
    )
    assert (
        first["prediction_input"]["query_array_sha256"]
        == second["prediction_input"]["query_array_sha256"]
    )
    assert (
        first["withheld_evaluation"]["target_array_sha256"]
        == second["withheld_evaluation"]["target_array_sha256"]
    )


def test_prediction_input_carries_only_prefix_masks(tmp_path: Path) -> None:
    config = PhysTwinTAPNextPPCompetenceConfig()
    tracks = np.zeros((140, 9, 3), dtype=float)
    source = _write_source(tmp_path, tracks, config)
    report = prepare_source_artifacts(
        *source,
        tmp_path / "artifacts",
        config=config,
    )
    query, identity_ids, masks = validate_prediction_input(
        report["prediction_input"]["path"],
        report["prediction_input"]["sha256"],
        config=config,
    )
    assert query.shape == (4, 3)
    assert np.array_equal(identity_ids, [3, 4, 6, 8])
    assert masks.shape == (3, config.prefix_frame_count, 8, 10)


def test_dynamic_case_name_propagates_through_sealed_artifacts(
    tmp_path: Path,
) -> None:
    config = PhysTwinTAPNextPPCompetenceConfig(
        case_name="transfer_case",
        selected_identity_ids=(0, 1, 2),
    )
    tracks = np.zeros((140, 3, 3), dtype=np.float32)
    source = _write_source(tmp_path / "source", tracks, config)
    source_report = prepare_source_artifacts(
        *source,
        tmp_path / "artifacts",
        config=config,
    )
    assert source_report["case"] == "transfer_case"

    support = np.ones((config.prefix_frame_count, 3), dtype=bool)
    prediction = _write_prediction(tmp_path, tracks[: config.prefix_frame_count], support, config)
    prediction_report = json.loads(
        (prediction / PREDICTION_REPORT_FILENAME).read_text(encoding="utf-8")
    )
    seal = json.loads(
        (prediction / "tapnextpp_prediction_seal.json").read_text(
            encoding="utf-8"
        )
    )
    assert prediction_report["case"] == "transfer_case"
    assert seal["case"] == "transfer_case"

    withheld = tmp_path / "withheld.npz"
    _write_withheld(withheld, tracks[: config.prefix_frame_count], config)
    result = evaluate_competence(
        prediction,
        withheld,
        file_sha256(withheld),
        tmp_path / "result.json",
        config=config,
    )
    assert result["case"] == "transfer_case"


def _write_prediction(
    root: Path,
    candidate: np.ndarray,
    support: np.ndarray,
    config: PhysTwinTAPNextPPCompetenceConfig,
) -> Path:
    prediction = root / "prediction"
    frame_count, point_count = candidate.shape[:2]
    camera_count = len(config.selected_cameras)
    write_prediction_artifact(
        prediction,
        raw_tracker_m=candidate,
        accepted_support=support,
        observation_reliability=np.where(support, 0.8, 0.0),
        observation_covariance_m2=np.repeat(
            np.eye(3, dtype=np.float32)[None, None],
            frame_count * point_count,
            axis=0,
        ).reshape(frame_count, point_count, 3, 3)
        * 2.5e-5,
        support_view_count=np.where(support, 2, 0),
        reprojection_rmse_px=np.where(support, 0.5, np.nan),
        depth_residual_rmse_m=np.where(support, 0.002, np.nan),
        per_camera_tracks_xy=np.zeros(
            (camera_count, frame_count, point_count, 2),
            dtype=np.float32,
        ),
        per_camera_visibility_probability=np.ones(
            (camera_count, frame_count, point_count),
            dtype=np.float32,
        ),
        query_points_world_m=candidate[0],
        identity_ids=np.asarray(config.selected_identity_ids),
        input_provenance={"prediction_input_sha256": "q"},
        runtime_provenance={"device": "test"},
        implementation_sha256={"runner": "r", "adapter": "a"},
        config=config,
    )
    seal_prediction(prediction)
    return prediction


def _write_withheld(
    path: Path,
    target: np.ndarray,
    config: PhysTwinTAPNextPPCompetenceConfig,
) -> None:
    np.savez_compressed(
        path,
        target_tracks_world_m=target,
        identity_ids=np.asarray(config.selected_identity_ids),
        source_frame_start=np.asarray(config.source_frame_start),
        source_frame_end_exclusive=np.asarray(
            config.source_frame_end_exclusive
        ),
    )


def test_competence_gate_passes_accurate_motion(tmp_path: Path) -> None:
    config = PhysTwinTAPNextPPCompetenceConfig(
        selected_identity_ids=(0, 1, 2)
    )
    target = np.zeros((config.prefix_frame_count, 3, 3), dtype=np.float32)
    target[:, :, 0] = np.arange(config.prefix_frame_count)[:, None] * 0.001
    support = np.ones(target.shape[:2], dtype=bool)
    prediction = _write_prediction(
        tmp_path,
        target.copy(),
        support,
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
    assert result["decision"] == (
        "advance-to-separately-locked-guarded-assimilation-smoke"
    )


def test_competence_gate_stops_persistence_copy(tmp_path: Path) -> None:
    config = PhysTwinTAPNextPPCompetenceConfig(
        selected_identity_ids=(0, 1, 2)
    )
    target = np.zeros((config.prefix_frame_count, 3, 3), dtype=np.float32)
    target[:, :, 0] = np.arange(config.prefix_frame_count)[:, None] * 0.001
    candidate = np.repeat(target[:1], config.prefix_frame_count, axis=0)
    support = np.ones(target.shape[:2], dtype=bool)
    prediction = _write_prediction(tmp_path, candidate, support, config)
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
    assert result["gates"]["relative_gain_over_persistence"] is False


def test_prediction_archive_preserves_metric_covariance(tmp_path: Path) -> None:
    config = PhysTwinTAPNextPPCompetenceConfig(
        selected_identity_ids=(0, 1, 2)
    )
    target = np.zeros((config.prefix_frame_count, 3, 3), dtype=np.float32)
    support = np.ones(target.shape[:2], dtype=bool)
    prediction = _write_prediction(tmp_path, target, support, config)
    with np.load(prediction / PREDICTION_FILENAME, allow_pickle=False) as stored:
        covariance = np.asarray(stored["observation_covariance_m2"])
        reliability = np.asarray(stored["observation_reliability"])
    assert covariance.shape == (*target.shape[:2], 3, 3)
    assert np.all(np.linalg.eigvalsh(covariance) > 0.0)
    assert np.all(reliability == np.float32(0.8))


def test_seal_detects_prediction_archive_mutation(tmp_path: Path) -> None:
    config = PhysTwinTAPNextPPCompetenceConfig(
        selected_identity_ids=(0, 1, 2)
    )
    target = np.zeros((config.prefix_frame_count, 3, 3), dtype=np.float32)
    support = np.ones(target.shape[:2], dtype=bool)
    prediction = tmp_path / "prediction"
    frame_count, point_count = target.shape[:2]
    camera_count = len(config.selected_cameras)
    write_prediction_artifact(
        prediction,
        raw_tracker_m=target,
        accepted_support=support,
        observation_reliability=np.ones(support.shape),
        observation_covariance_m2=np.repeat(
            np.eye(3)[None, None],
            frame_count * point_count,
            axis=0,
        ).reshape(frame_count, point_count, 3, 3),
        support_view_count=np.full(support.shape, 3),
        reprojection_rmse_px=np.zeros(support.shape),
        depth_residual_rmse_m=np.zeros(support.shape),
        per_camera_tracks_xy=np.zeros(
            (camera_count, frame_count, point_count, 2)
        ),
        per_camera_visibility_probability=np.ones(
            (camera_count, frame_count, point_count)
        ),
        query_points_world_m=target[0],
        identity_ids=np.asarray(config.selected_identity_ids),
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
