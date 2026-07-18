import pickle
from pathlib import Path

import numpy as np

from bayesian_phystwin.phystwin_residual_velocity import (
    PhysTwinResidualVelocityConfig,
    fit_recurrent_residual_velocity,
    physical_rollout_features,
    residual_velocity_features,
    rollout_latent_residual_velocity,
)


def _write_case(
    root: Path,
    displacement: np.ndarray,
    *,
    future_observation_offset: float = 0.0,
    train_end_frame: int = 8,
) -> tuple[Path, Path, Path, np.ndarray]:
    frame_count = len(displacement)
    original = np.array([[0.0, 0.0, 0.0], [0.02, 0.0, 0.0]])
    baseline = np.repeat(original[None], frame_count, axis=0)
    observed = baseline.copy()
    observed[:, :, 0] += displacement[:, None]
    observed[train_end_frame:, :, 1] += future_observation_offset
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
    gt_track = observed[:, :1].copy()
    final_path = root / "final.pkl"
    baseline_path = root / "baseline.pkl"
    track_path = root / "track.pkl"
    root.mkdir(parents=True, exist_ok=True)
    for path, value in (
        (final_path, data),
        (baseline_path, baseline.astype(np.float32)),
        (track_path, gt_track.astype(np.float32)),
    ):
        with path.open("wb") as handle:
            pickle.dump(value, handle)
    return final_path, baseline_path, track_path, observed


def _config() -> PhysTwinResidualVelocityConfig:
    return PhysTwinResidualVelocityConfig(
        fit_end_frame=5,
        train_end_frame=8,
        rank_candidates=(1,),
        velocity_persistence_candidates=(1.0,),
        ridge_candidates=(1e-6,),
        interpolation_neighbors=1,
        maximum_state_multiplier=3.0,
        maximum_velocity_multiplier=3.0,
        maximum_residual_m=0.1,
        minimum_dynamic_improvement=0.001,
    )


def test_rollout_features_use_physical_motion() -> None:
    baseline = np.zeros((3, 2, 3))
    baseline[1:, :, 0] = [[0.01], [0.03]]
    controllers = np.zeros((3, 1, 3))
    controllers[:, 0, 0] = [0.0, 0.02, 0.04]

    physical = physical_rollout_features(baseline)
    combined = residual_velocity_features(controllers, baseline)

    assert physical.shape == (3, 9)
    assert combined.shape == (3, 28)
    assert physical[2, -1] > physical[1, -1]


def test_recursive_velocity_rollout_integrates_its_own_state() -> None:
    features = np.zeros((5, 1))
    coefficients = np.array([[0.0], [0.0], [1.0]])

    result = rollout_latent_residual_velocity(
        np.array([0.0]),
        np.array([0.0]),
        features,
        coefficients,
        np.ones(1),
        np.ones(1),
        start_frame=1,
        end_frame=4,
        velocity_persistence=1.0,
        state_norm_cap=100.0,
        velocity_norm_cap=100.0,
    )

    np.testing.assert_allclose(result[:, 0], [1.0, 3.0, 6.0])


def test_recurrent_velocity_beats_persistence_for_constant_residual_motion(
    tmp_path: Path,
) -> None:
    displacement = 0.002 * np.arange(12)
    final_path, baseline_path, track_path, _ = _write_case(
        tmp_path / "case", displacement
    )

    summary = fit_recurrent_residual_velocity(
        final_path,
        baseline_path,
        track_path,
        tmp_path / "output",
        config=_config(),
    )

    assert summary["selection"]["selected_method"] == "residual_velocity"
    assert summary["selection"]["dynamic_accepted"]
    assert summary["test"]["selection_score_relative_to_persistence"] < 0.1


def test_future_observation_mutation_does_not_change_prediction(tmp_path: Path) -> None:
    displacement = 0.002 * np.arange(12)
    first = _write_case(tmp_path / "first", displacement)
    second = _write_case(
        tmp_path / "second",
        displacement,
        future_observation_offset=10.0,
    )

    first_summary = fit_recurrent_residual_velocity(
        first[0], first[1], first[2], tmp_path / "first_output", config=_config()
    )
    second_summary = fit_recurrent_residual_velocity(
        second[0], second[1], second[2], tmp_path / "second_output", config=_config()
    )
    with Path(first_summary["outputs"]["trajectory"]).open("rb") as handle:
        first_trajectory = pickle.load(handle)
    with Path(second_summary["outputs"]["trajectory"]).open("rb") as handle:
        second_trajectory = pickle.load(handle)

    assert first_summary["selection"] == second_summary["selection"]
    np.testing.assert_array_equal(first_trajectory, second_trajectory)


def test_static_residual_falls_back_to_exact_persistence(tmp_path: Path) -> None:
    displacement = np.full(12, 0.004)
    final_path, baseline_path, track_path, observed = _write_case(
        tmp_path / "case", displacement
    )

    summary = fit_recurrent_residual_velocity(
        final_path,
        baseline_path,
        track_path,
        tmp_path / "output",
        config=_config(),
    )
    with Path(summary["outputs"]["trajectory"]).open("rb") as handle:
        trajectory = pickle.load(handle)

    assert summary["selection"]["selected_method"] == "persistence"
    np.testing.assert_array_equal(trajectory[8:], observed[8:].astype(np.float32))
