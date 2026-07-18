import numpy as np
import pytest

from bayesian_phystwin.cli.phystwin_refit import build_parser
from bayesian_phystwin.phystwin_prior_evaluation import (
    evaluate_phystwin_prior_arrays,
)
from bayesian_phystwin.phystwin_refit import (
    PhysTwinRefitReliabilityConfig,
    build_phystwin_track_objective,
    causal_markov_cue_reliability,
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


def test_cue_objective_composes_regenerated_probabilities_and_residuals():
    visible = np.ones((2, 1), dtype=bool)
    motion_valid = np.ones((2, 1), dtype=bool)
    cues = {
        "confidence": np.array([[0.8], [1.0]]),
        "visibility_probability": np.array([[0.5], [1.0]]),
        "forward_backward_error_px": np.array([[2.0], [100.0]]),
        "forward_backward_valid": np.array([[True], [False]]),
        "multiview_reprojection_error_px": np.array([[1.0], [2.0]]),
        "multiview_valid": np.array([[True], [True]]),
    }

    objective = build_phystwin_track_objective(
        visible,
        motion_valid,
        cues=cues,
        variant="cue",
        config=PhysTwinRefitReliabilityConfig(
            boundary_scale=None,
            flow_scale=None,
            forward_backward_scale_px=2.0,
            multiview_scale_px=1.0,
        ),
    )

    assert objective.weights[0, 0] == pytest.approx(0.4 * np.exp(-2.0))
    assert objective.weights[1, 0] == pytest.approx(np.exp(-2.0))


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


def test_causal_markov_cue_does_not_use_future_values():
    prior = np.full((6, 2), 0.9)
    changed = prior.copy()
    changed[4:, 0] = 0.01

    original_filtered = causal_markov_cue_reliability(prior)
    changed_filtered = causal_markov_cue_reliability(changed)

    np.testing.assert_allclose(original_filtered[:4], changed_filtered[:4])
    assert changed_filtered[-1, 0] < original_filtered[-1, 0]


def test_markov_mixture_uses_persistent_prior_and_visible_normalizer():
    visible, motion_valid = _masks()
    flow = np.zeros((3, 2))
    flow[1, 0] = 0.02

    static = build_phystwin_track_objective(
        visible,
        motion_valid,
        cues={"flow_inconsistency": flow},
        variant="mixture",
    )
    markov = build_phystwin_track_objective(
        visible,
        motion_valid,
        cues={"flow_inconsistency": flow},
        variant="markov_mixture",
    )

    np.testing.assert_allclose(markov.normalizer, visible.sum(axis=1))
    assert markov.prior_inlier_probability[2, 0] != pytest.approx(
        static.prior_inlier_probability[2, 0]
    )


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
            "--selection-metric",
            "official_3d",
            "--gt-track-3d",
            "manual.pkl",
            "--profile-prediction-mass",
            "0.999",
            "--boundary-scale",
            "0.004",
        ]
    )

    assert args.spring_parameterization == "grouped"
    assert args.selection_metric == "official_3d"
    assert args.gt_track_3d == "manual.pkl"
    assert args.profile_prediction_mass == pytest.approx(0.999)
    assert args.boundary_scale == pytest.approx(0.004)
    assert not args.atomic_spring_forces


def test_refit_cli_accepts_regenerated_cue_controls():
    args = build_parser().parse_args(
        [
            "official",
            "final.pkl",
            "optimal.pkl",
            "checkpoint.pt",
            "cues.npz",
            "output",
            "--variant",
            "cue",
            "--train-end-frame",
            "64",
            "--disable-flow-cue",
            "--disable-boundary-cue",
            "--forward-backward-scale-px",
            "16",
            "--multiview-scale-px",
            "2",
        ]
    )

    assert args.disable_flow_cue
    assert args.disable_boundary_cue
    assert args.forward_backward_scale_px == pytest.approx(16.0)
    assert args.multiview_scale_px == pytest.approx(2.0)


def test_refit_cli_accepts_regularized_regional_springs():
    args = build_parser().parse_args(
        [
            "official",
            "final.pkl",
            "optimal.pkl",
            "checkpoint.pt",
            "cues.npz",
            "output",
            "--variant",
            "hard",
            "--train-end-frame",
            "64",
            "--spring-parameterization",
            "regional",
            "--spring-region-count",
            "4",
            "--spring-scale-weight-decay",
            "0.1",
            "--dashpot-log-scale",
            "-0.2",
            "--drag-log-scale",
            "0.1",
        ]
    )

    assert args.spring_parameterization == "regional"
    assert args.spring_region_count == 4
    assert args.spring_scale_weight_decay == pytest.approx(0.1)
    assert args.dashpot_log_scale == pytest.approx(-0.2)
    assert args.drag_log_scale == pytest.approx(0.1)


def test_refit_cli_accepts_part_pair_partition():
    args = build_parser().parse_args(
        [
            "official",
            "final.pkl",
            "optimal.pkl",
            "checkpoint.pt",
            "cues.npz",
            "output",
            "--variant",
            "hard",
            "--train-end-frame",
            "64",
            "--spring-parameterization",
            "part_pair",
            "--spring-partition",
            "node_sem.npz",
            "--spring-topology",
            "topology.npz",
        ]
    )

    assert args.spring_parameterization == "part_pair"
    assert args.spring_partition == "node_sem.npz"
    assert args.spring_topology == "topology.npz"


def test_refit_cli_accepts_canonical_spring_basis():
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
            "canonical_basis",
            "--spring-basis-rank",
            "12",
            "--spring-basis-length-scale-multiplier",
            "1.5",
        ]
    )

    assert args.spring_parameterization == "canonical_basis"
    assert args.spring_basis_rank == 12
    assert args.spring_basis_length_scale_multiplier == pytest.approx(1.5)


def test_prior_evaluation_uses_target_visible_refit_support():
    visible, motion_valid = _masks()
    cues = {
        "flow_inconsistency": np.array(
            [[0.0, 0.1], [0.02, 0.1], [0.0, 0.1]]
        )
    }

    result = evaluate_phystwin_prior_arrays(
        visible,
        motion_valid,
        cues,
    )

    assert result["measurement_count"] == int(np.sum(visible[1:]))
    assert set(result["variants"]) == {"mixture", "markov_mixture"}
