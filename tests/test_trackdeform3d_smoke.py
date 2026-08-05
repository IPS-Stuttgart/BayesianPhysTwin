from __future__ import annotations

from dataclasses import fields

import numpy as np
import pytest

from bayesian_phystwin.trackdeform3d_smoke import (
    TrackDeform3DEvaluatorTarget,
    TrackDeform3DSmokeConfig,
    evaluate_trackdeform3d_smoke,
    predict_trackdeform3d_smoke,
    rollout_known_action_chain,
    split_trackdeform3d_carriers,
)


def _synthetic_case(
    *,
    with_action_conditioned_discrepancy: bool,
) -> tuple[
    TrackDeform3DSmokeConfig,
    object,
    TrackDeform3DEvaluatorTarget,
]:
    config = TrackDeform3DSmokeConfig(
        prefix_frames=12,
        future_frames=6,
        observed_identity_fraction=0.4,
        pbd_iterations=10,
        graph_rank=3,
        validation_frames=4,
        minimum_validation_improvement=0.10,
        constant_velocity_history=4,
    )
    node_count = 7
    frame_zero = np.column_stack(
        [
            np.linspace(0.0, 0.6, node_count),
            np.zeros(node_count),
            np.ones(node_count),
        ]
    )
    edges = np.column_stack([np.arange(node_count - 1), np.arange(1, node_count)])
    rest = np.full(node_count - 1, 0.1)
    time = np.arange(config.prefix_frames + config.future_frames, dtype=float)
    left = np.column_stack([np.zeros_like(time), 0.002 * time, np.ones_like(time)])
    right = np.column_stack(
        [
            np.full_like(time, 0.6),
            0.001 * time,
            np.ones_like(time) + 0.0015 * time,
        ]
    )
    action = np.stack([left, right], axis=1)
    physical = rollout_known_action_chain(
        frame_zero,
        edges,
        rest,
        action,
        pbd_iterations=config.pbd_iterations,
    )
    truth = physical.copy()
    if with_action_conditioned_discrepancy:
        graph_coordinate = np.linspace(-1.0, 1.0, node_count)
        correction = 0.45 * (left[:, 1] - left[0, 1])[:, None] * graph_coordinate[None]
        truth[:, :, 2] += correction
    prediction_input, target = split_trackdeform3d_carriers(
        truth,
        edges,
        rest,
        action,
        config=config,
    )
    return config, prediction_input, target


def test_predictor_type_cannot_accept_hidden_future() -> None:
    _, prediction_input, _ = _synthetic_case(with_action_conditioned_discrepancy=True)

    field_names = {field.name for field in fields(type(prediction_input))}

    assert "hidden_future_points_m" not in field_names
    assert "future_points_m" not in field_names


def test_action_conditioned_belief_passes_prefix_gate_and_improves_hidden() -> None:
    config, prediction_input, target = _synthetic_case(
        with_action_conditioned_discrepancy=True
    )

    prediction = predict_trackdeform3d_smoke(prediction_input, config=config)
    result = evaluate_trackdeform3d_smoke(
        prediction,
        target,
        nominal_coverage=config.nominal_coverage,
    )

    assert prediction.gate["admitted"] is True
    assert result["guarded_improvement_vs_physical_fraction"] > 0.10
    assert (
        result["arms"]["guarded_bayesian"]["hidden_identity_rmse_m"]
        < result["arms"]["physical"]["hidden_identity_rmse_m"]
    )
    assert (
        result["arms"]["guarded_bayesian"]["marginal_coordinate_coverage"] is not None
    )


def test_mutating_evaluator_future_cannot_change_prediction() -> None:
    config, prediction_input, target = _synthetic_case(
        with_action_conditioned_discrepancy=True
    )
    first = predict_trackdeform3d_smoke(prediction_input, config=config)
    mutated = TrackDeform3DEvaluatorTarget(
        hidden_identity_ids=target.hidden_identity_ids,
        hidden_future_points_m=target.hidden_future_points_m + 1000.0,
    )

    second = predict_trackdeform3d_smoke(prediction_input, config=config)

    np.testing.assert_array_equal(first.physical_m, second.physical_m)
    np.testing.assert_array_equal(first.guarded_bayesian_m, second.guarded_bayesian_m)
    assert not np.array_equal(
        target.hidden_future_points_m,
        mutated.hidden_future_points_m,
    )


def test_failed_gate_is_bit_exact_physical_fallback() -> None:
    config, prediction_input, _ = _synthetic_case(
        with_action_conditioned_discrepancy=False
    )

    prediction = predict_trackdeform3d_smoke(prediction_input, config=config)

    assert prediction.gate["admitted"] is False
    assert prediction.guarded_variance_m2 is None
    np.testing.assert_array_equal(
        prediction.guarded_bayesian_m,
        prediction.physical_m,
    )


def test_evaluator_rejects_observed_hidden_identity_overlap() -> None:
    config, prediction_input, target = _synthetic_case(
        with_action_conditioned_discrepancy=True
    )
    prediction = predict_trackdeform3d_smoke(prediction_input, config=config)
    invalid = TrackDeform3DEvaluatorTarget(
        hidden_identity_ids=np.asarray(
            [prediction.observed_identity_ids[0], target.hidden_identity_ids[0]]
        ),
        hidden_future_points_m=target.hidden_future_points_m[:, :2],
    )

    with pytest.raises(ValueError, match="overlap"):
        evaluate_trackdeform3d_smoke(
            prediction,
            invalid,
            nominal_coverage=config.nominal_coverage,
        )
