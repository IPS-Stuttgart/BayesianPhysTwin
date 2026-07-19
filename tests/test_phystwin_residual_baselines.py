import pickle
from pathlib import Path

import numpy as np

from bayesian_phystwin.phystwin_residual_baselines import (
    BASELINE_METHODS,
    fit_residual_dynamics_baselines,
)
from bayesian_phystwin.phystwin_residual_dynamics import (
    PhysTwinResidualDynamicsConfig,
)


def test_matched_residual_baselines_write_all_trajectories(tmp_path: Path) -> None:
    frame_count = 12
    original = np.array([[0.0, 0.0, 0.0], [0.02, 0.0, 0.0]])
    displacement = 0.001 * np.square(np.arange(frame_count))
    observed = np.repeat(original[None], frame_count, axis=0)
    observed[:, :, 0] += displacement[:, None]
    baseline = np.repeat(original[None], frame_count, axis=0)
    controllers = np.zeros((frame_count, 1, 3))
    controllers[:, 0, 0] = displacement
    data = {
        "object_points": observed.astype(np.float32),
        "object_visibilities": np.ones((frame_count, 2), dtype=bool),
        "object_motions_valid": np.ones((frame_count - 1, 2), dtype=bool),
        "controller_points": controllers.astype(np.float32),
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

    summary = fit_residual_dynamics_baselines(
        paths["final"],
        paths["baseline"],
        paths["tracks"],
        tmp_path / "output",
        config=PhysTwinResidualDynamicsConfig(
            fit_end_frame=6,
            train_end_frame=9,
            rank_candidates=(1,),
            persistence_candidates=(0.0, 1.0),
            ridge_candidates=(1e-3,),
            interpolation_neighbors=1,
            maximum_residual_m=0.2,
        ),
    )

    assert tuple(summary["methods"]) == BASELINE_METHODS
    for method in BASELINE_METHODS:
        assert Path(summary["methods"][method]["outputs"]["trajectory"]).is_file()
    assert summary["methods"]["dmdc"]["selection"]["accepted"]
    assert (
        summary["methods"]["dmdc"]["test"]["selection_score_relative_to_baseline"]
        < 0.2
    )


def test_residual_baselines_can_predict_without_future_observations(
    tmp_path: Path,
) -> None:
    frame_count = 12
    train_end = 9
    original = np.array([[0.0, 0.0, 0.0], [0.02, 0.0, 0.0]])
    displacement = 0.001 * np.square(np.arange(frame_count))
    observed = np.repeat(original[None], train_end, axis=0)
    observed[:, :, 0] += displacement[:train_end, None]
    baseline = np.repeat(original[None], frame_count, axis=0)
    controllers = np.zeros((frame_count, 1, 3))
    controllers[:, 0, 0] = displacement
    data = {
        "object_points": observed.astype(np.float32),
        "object_visibilities": np.ones((train_end, 2), dtype=bool),
        "object_motions_valid": np.ones((train_end - 1, 2), dtype=bool),
        "controller_points": controllers.astype(np.float32),
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

    summary = fit_residual_dynamics_baselines(
        paths["final"],
        paths["baseline"],
        paths["tracks"],
        tmp_path / "sealed",
        config=PhysTwinResidualDynamicsConfig(
            fit_end_frame=6,
            train_end_frame=train_end,
            rank_candidates=(1,),
            persistence_candidates=(0.0, 1.0),
            ridge_candidates=(1e-3,),
            interpolation_neighbors=1,
            maximum_residual_m=0.2,
        ),
        evaluate_future=False,
    )

    assert summary["future_metrics_opened"] is False
    for method in BASELINE_METHODS:
        result = summary["methods"][method]
        assert "test" not in result
        with Path(result["outputs"]["trajectory"]).open("rb") as handle:
            trajectory = pickle.load(handle)
        assert trajectory.shape == baseline.shape
