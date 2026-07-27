import json
import pickle
from pathlib import Path

import numpy as np

from bayesian_phystwin.phystwin_bayesian_anchor import (
    BayesianResidualAnchorConfig,
    fit_bayesian_residual_anchor,
    robust_random_walk_endpoint,
)


def test_robust_endpoint_downweights_a_terminal_outlier() -> None:
    residual = np.full((8, 1, 3), 0.002)
    residual[-1, 0] = 0.1
    posterior = robust_random_walk_endpoint(
        residual,
        np.ones((8, 1), dtype=bool),
        end_frame=8,
        process_variance=1e-7,
        observation_variance=1e-6,
        initial_variance=1e-4,
        inlier_prior=0.95,
        outlier_variance_multiplier=100.0,
    )

    assert np.linalg.norm(posterior.mean[0] - 0.002) < 0.01
    assert posterior.final_inlier_probability[0] < 0.1
    assert posterior.updated_mask[0]
    assert posterior.updated_mask.flags.writeable is False


def test_bayesian_anchor_improves_constant_held_out_discrepancy(
    tmp_path: Path,
) -> None:
    frame_count = 12
    original = np.array([[0.0, 0.0, 0.0], [0.02, 0.0, 0.0]])
    observed = np.repeat(original[None], frame_count, axis=0)
    observed[1:, :, 0] += 0.01
    baseline = np.repeat(original[None], frame_count, axis=0)
    data = {
        "object_points": observed.astype(np.float32),
        "object_visibilities": np.ones((frame_count, 2), dtype=bool),
        "object_motions_valid": np.ones((frame_count - 1, 2), dtype=bool),
        "controller_points": np.zeros((frame_count, 1, 3), dtype=np.float32),
        "surface_points": np.empty((0, 3), dtype=np.float32),
        "interior_points": np.empty((0, 3), dtype=np.float32),
    }
    paths = {
        "final": tmp_path / "final.pkl",
        "baseline": tmp_path / "baseline.pkl",
        "tracks": tmp_path / "tracks.pkl",
    }
    for path, value in (
        (paths["final"], data),
        (paths["baseline"], baseline.astype(np.float32)),
        (paths["tracks"], observed[:, :1].astype(np.float32)),
    ):
        with path.open("wb") as handle:
            pickle.dump(value, handle)

    summary = fit_bayesian_residual_anchor(
        paths["final"],
        paths["baseline"],
        paths["tracks"],
        tmp_path / "output",
        config=BayesianResidualAnchorConfig(
            fit_end_frame=6,
            train_end_frame=9,
            process_std_candidates_m=(0.001,),
            observation_std_candidates_m=(0.001,),
            interpolation_neighbors=1,
            maximum_residual_m=0.02,
        ),
    )

    assert summary["selection"]["accepted"]
    assert summary["test"]["selection_score_relative_to_baseline"] < 0.1
    assert summary["posterior"]["median_std_m"] > 0.0


def test_bayesian_anchor_can_predict_without_future_observations(
    tmp_path: Path,
) -> None:
    frame_count = 12
    train_end = 9
    original = np.array([[0.0, 0.0, 0.0], [0.02, 0.0, 0.0]])
    observed = np.repeat(original[None], train_end, axis=0)
    observed[1:, :, 0] += 0.01
    baseline = np.repeat(original[None], frame_count, axis=0)
    data = {
        "object_points": observed.astype(np.float32),
        "object_visibilities": np.ones((train_end, 2), dtype=bool),
        "object_motions_valid": np.ones((train_end - 1, 2), dtype=bool),
        "controller_points": np.zeros((frame_count, 1, 3), dtype=np.float32),
        "surface_points": np.empty((0, 3), dtype=np.float32),
        "interior_points": np.empty((0, 3), dtype=np.float32),
    }
    paths = {
        "final": tmp_path / "final.pkl",
        "baseline": tmp_path / "baseline.pkl",
        "tracks": tmp_path / "tracks.pkl",
    }
    for path, value in (
        (paths["final"], data),
        (paths["baseline"], baseline.astype(np.float32)),
        (paths["tracks"], observed[:, :1].astype(np.float32)),
    ):
        with path.open("wb") as handle:
            pickle.dump(value, handle)

    summary = fit_bayesian_residual_anchor(
        paths["final"],
        paths["baseline"],
        paths["tracks"],
        tmp_path / "sealed",
        config=BayesianResidualAnchorConfig(
            fit_end_frame=6,
            train_end_frame=train_end,
            process_std_candidates_m=(0.001,),
            observation_std_candidates_m=(0.001,),
            interpolation_neighbors=1,
            maximum_residual_m=0.02,
        ),
        evaluate_future=False,
    )

    with Path(summary["outputs"]["trajectory"]).open("rb") as handle:
        trajectory = pickle.load(handle)
    assert trajectory.shape == baseline.shape
    assert summary["future_metrics_opened"] is False
    assert "test" not in summary


def test_bayesian_anchor_reports_null_summary_without_updates(
    tmp_path: Path,
) -> None:
    frame_count = 12
    train_end = 9
    original = np.array([[0.0, 0.0, 0.0], [0.02, 0.0, 0.0]])
    observed = np.repeat(original[None], frame_count, axis=0)
    observed[1:, :, 0] += 0.01
    baseline = np.repeat(original[None], frame_count, axis=0)
    visibility = np.ones((frame_count, 2), dtype=bool)
    visibility[0] = False
    data = {
        "object_points": observed.astype(np.float32),
        "object_visibilities": visibility,
        "object_motions_valid": np.zeros(
            (frame_count - 1, 2),
            dtype=bool,
        ),
        "controller_points": np.zeros(
            (frame_count, 1, 3),
            dtype=np.float32,
        ),
        "surface_points": np.empty((0, 3), dtype=np.float32),
        "interior_points": np.empty((0, 3), dtype=np.float32),
    }
    paths = {
        "final": tmp_path / "final.pkl",
        "baseline": tmp_path / "baseline.pkl",
        "tracks": tmp_path / "tracks.pkl",
    }
    for path, value in (
        (paths["final"], data),
        (paths["baseline"], baseline.astype(np.float32)),
        (paths["tracks"], observed[:, :1].astype(np.float32)),
    ):
        with path.open("wb") as handle:
            pickle.dump(value, handle)

    summary = fit_bayesian_residual_anchor(
        paths["final"],
        paths["baseline"],
        paths["tracks"],
        tmp_path / "no-updates",
        config=BayesianResidualAnchorConfig(
            fit_end_frame=6,
            train_end_frame=train_end,
            process_std_candidates_m=(0.001,),
            observation_std_candidates_m=(0.001,),
            interpolation_neighbors=1,
            maximum_residual_m=0.02,
        ),
    )

    assert summary["selection"]["accepted"] is False
    posterior_summary = summary["posterior"]
    assert posterior_summary["updated_track_count"] == 0
    for key in (
        "median_std_m",
        "upper_95_std_m",
        "median_final_inlier_probability",
        "median_final_future_predictive_std_m",
    ):
        assert posterior_summary[key] is None

    written = json.loads(
        Path(summary["outputs"]["summary"]).read_text(encoding="utf-8")
    )
    assert written["posterior"] == posterior_summary
