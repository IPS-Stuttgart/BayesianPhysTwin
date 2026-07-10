import pickle
from pathlib import Path

import numpy as np

from bayesian_phystwin.phystwin_residual_dynamics import (
    PhysTwinResidualDynamicsConfig,
    controller_action_features,
    fit_action_conditioned_residual_dynamics,
    fit_residual_basis,
)


def test_controller_features_include_position_and_motion() -> None:
    points = np.zeros((3, 2, 3))
    points[1:, :, 0] = [[0.01], [0.03]]

    features = controller_action_features(points)

    assert features.shape == (3, 13)
    assert features[0, -1] == 0.0
    assert features[2, -1] > features[1, -1]


def test_residual_basis_does_not_use_future_frames() -> None:
    residual = np.zeros((6, 2, 3))
    residual[1:3, :, 0] = [[0.01], [0.02]]
    valid = np.ones((6, 2), dtype=bool)
    changed = residual.copy()
    changed[3:] = 100.0

    first = fit_residual_basis(residual, valid, end_frame=3, maximum_rank=1)
    second = fit_residual_basis(changed, valid, end_frame=3, maximum_rank=1)

    np.testing.assert_allclose(np.abs(first), np.abs(second))


def test_action_conditioned_residual_improves_held_out_translation(
    tmp_path: Path,
) -> None:
    frame_count = 9
    original = np.array([[0.0, 0.0, 0.0], [0.02, 0.0, 0.0]])
    displacement = 0.003 * np.arange(frame_count)
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
    gt_track = observed[:, :1].copy()
    final_path = tmp_path / "final.pkl"
    baseline_path = tmp_path / "baseline.pkl"
    track_path = tmp_path / "track.pkl"
    for path, value in (
        (final_path, data),
        (baseline_path, baseline.astype(np.float32)),
        (track_path, gt_track.astype(np.float32)),
    ):
        with path.open("wb") as handle:
            pickle.dump(value, handle)

    summary = fit_action_conditioned_residual_dynamics(
        final_path,
        baseline_path,
        track_path,
        tmp_path / "output",
        config=PhysTwinResidualDynamicsConfig(
            fit_end_frame=4,
            train_end_frame=6,
            rank_candidates=(1,),
            persistence_candidates=(0.0,),
            ridge_candidates=(1e-6,),
            interpolation_neighbors=1,
            maximum_residual_m=0.1,
        ),
    )

    assert summary["selection"]["accepted"]
    assert summary["test"]["selection_score_relative_to_baseline"] < 0.1
