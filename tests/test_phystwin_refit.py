import numpy as np
import pytest

from bayesian_phystwin.cli.phystwin_refit import build_parser
from bayesian_phystwin.phystwin_refit import (
    PhysTwinRefitReliabilityConfig,
    build_phystwin_track_objective,
    evaluate_phystwin_trajectory,
    evaluate_phystwin_trajectory_splits,
    phystwin_tracking_metrics,
)


def _masks():
    visible = np.array(
        [[True, True], [True, False], [True, True], [True, True]]
    )
    motion_valid = np.array(
        [[True, False], [False, False], [True, False], [False, False]]
    )
    return visible, motion_valid


def test_hard_objective_uses_previous_motion_gate_for_target_frame():
    visible, motion_valid = _masks()

    objective = build_phystwin_track_objective(
        visible,
        motion_valid,
        variant="hard",
    )

    np.testing.assert_array_equal(objective.support[1:], motion_valid[:-1])
    np.testing.assert_allclose(objective.normalizer, [2.0, 1.0, 1.0, 1.0])


def test_cue_objective_aligns_interframe_flow_to_target_frame():
    visible, motion_valid = _masks()
    flow = np.array([[0.0, 0.01], [0.005, 0.0], [0.02, 0.0]])

    objective = build_phystwin_track_objective(
        visible,
        motion_valid,
        cues={"flow_inconsistency": flow},
        variant="cue",
        config=PhysTwinRefitReliabilityConfig(flow_scale=0.005),
    )

    assert objective.weights[1, 0] == pytest.approx(0.999)
    assert objective.weights[1, 1] == 0.0
    assert objective.weights[2, 0] == pytest.approx(np.exp(-1.0), rel=1e-6)
    assert objective.weights[3, 0] == pytest.approx(np.exp(-4.0), rel=1e-6)


def test_mixture_normalizes_by_visible_support_not_prior_mass():
    visible, motion_valid = _masks()
    objective = build_phystwin_track_objective(
        visible,
        motion_valid,
        cues={"flow_inconsistency": np.full((3, 2), 0.01)},
        variant="mixture",
    )

    np.testing.assert_allclose(objective.normalizer, visible.sum(axis=1))
    assert np.all(objective.prior_inlier_probability > 0.0)
    assert np.all(objective.prior_inlier_probability < 1.0)


def test_tracking_metrics_and_split_evaluation_use_direct_correspondence():
    observed = np.zeros((4, 2, 3))
    trajectory = np.zeros((4, 3, 3))
    trajectory[2, 0, 0] = 0.03
    trajectory[3, 1, 1] = 0.04
    visible, motion_valid = _masks()

    metrics = phystwin_tracking_metrics(
        observed,
        trajectory,
        np.array(
            [[False, False], [False, False], [True, False], [False, True]]
        ),
    )
    evaluation = evaluate_phystwin_trajectory(
        observed,
        trajectory,
        visible,
        motion_valid,
        train_end_frame=3,
    )

    assert metrics["count"] == 2
    assert metrics["vector_rmse_m"] == pytest.approx(0.0353553391)
    assert evaluation["test"]["visible"]["count"] == 2


def test_split_evaluation_supports_fit_validation_and_test_ranges():
    observed = np.zeros((4, 2, 3))
    trajectory = np.zeros((4, 2, 3))
    visible, motion_valid = _masks()

    evaluation = evaluate_phystwin_trajectory_splits(
        observed,
        trajectory,
        visible,
        motion_valid,
        splits={"fit": (1, 2), "validation": (2, 3), "test": (3, 4)},
    )

    assert evaluation["fit"]["visible"]["count"] == 1
    assert evaluation["validation"]["visible"]["count"] == 2
    assert evaluation["test"]["visible"]["count"] == 2


def test_refit_objective_rejects_unknown_variant():
    visible, motion_valid = _masks()
    with pytest.raises(ValueError):
        build_phystwin_track_objective(
            visible,
            motion_valid,
            variant="residual-gated",
        )


def test_refit_cli_accepts_grouped_spring_parameterization():
    args = build_parser().parse_args(
        [
            "official",
            "final.pkl",
            "optimal.pkl",
            "checkpoint.pt",
            "cues.npz",
            "output",
            "--variant",
            "mixture",
            "--train-end-frame",
            "64",
            "--spring-parameterization",
            "grouped",
        ]
    )

    assert args.spring_parameterization == "grouped"
