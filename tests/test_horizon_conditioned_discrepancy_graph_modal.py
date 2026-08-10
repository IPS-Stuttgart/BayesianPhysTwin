from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from bayesian_phystwin.graph_dynamic_discrepancy import (
    GraphDynamicDiscrepancyBeliefV1,
    GraphDynamicDiscrepancyConfigV1,
    fit_graph_dynamic_discrepancy,
)


def test_last_residual_is_an_exact_nested_special_case() -> None:
    residual = np.asarray(
        [
            [0.010, -0.002, 0.003],
            [-0.004, 0.005, 0.001],
        ]
    )
    belief = GraphDynamicDiscrepancyBeliefV1.from_last_residual(residual)

    forecast = belief.forecast([1, 4])

    np.testing.assert_array_equal(forecast.mean_m[0], residual)
    np.testing.assert_array_equal(forecast.mean_m[1], residual)
    np.testing.assert_array_equal(
        forecast.joint_covariance_m2,
        np.zeros((12, 12)),
    )


def test_independent_anchor_matches_random_walk_mean_and_variance() -> None:
    mean = np.asarray([[0.003, -0.001, 0.002], [0.006, 0.004, -0.002]])
    variance = np.asarray([4e-6, 9e-6])
    process_std = 0.001
    belief = GraphDynamicDiscrepancyBeliefV1.from_independent_endpoint_posterior(
        mean,
        variance,
        process_std_m=process_std,
    )

    forecast = belief.forecast([1, 3])

    np.testing.assert_allclose(forecast.mean_m, np.repeat(mean[None], 2, axis=0))
    marginal = forecast.marginal_covariance_m2
    for node in range(2):
        np.testing.assert_allclose(
            marginal[0, node],
            np.eye(3) * (variance[node] + process_std**2),
        )
        np.testing.assert_allclose(
            marginal[1, node],
            np.eye(3) * (variance[node] + 3 * process_std**2),
        )
    np.testing.assert_allclose(
        forecast.joint_covariance_m2[:6, 6:],
        forecast.joint_covariance_m2[:6, :6],
    )


def test_constant_velocity_and_causal_acceleration_are_propagated() -> None:
    basis = np.eye(1)
    state = np.zeros((2, 1, 3))
    state[1, 0, 0] = 2.0
    belief = GraphDynamicDiscrepancyBeliefV1(
        graph_basis=basis,
        state_mean=state,
        state_covariance=np.zeros((6, 6)),
        frame_dt_s=0.5,
        velocity_retention=1.0,
        process_position_std_m=0.0,
        process_acceleration_std_mps2=0.0,
        last_frame_index=0,
    )

    forecast = belief.forecast(
        [1, 2],
        modal_acceleration_mps2=np.asarray([[4.0, 0.0, 0.0]]),
    )

    np.testing.assert_allclose(forecast.mean_m[:, 0, 0], [1.5, 4.0])
    np.testing.assert_array_equal(forecast.mean_m[:, 0, 1:], 0.0)


def test_joint_covariance_retains_cross_node_and_cross_horizon_terms() -> None:
    basis = np.asarray([[1.0], [1.0]]) / np.sqrt(2.0)
    state = np.zeros((2, 1, 3))
    covariance = np.diag([4e-6, 1e-6, 1e-6, 9e-6, 1e-6, 1e-6])
    belief = GraphDynamicDiscrepancyBeliefV1(
        graph_basis=basis,
        state_mean=state,
        state_covariance=covariance,
        frame_dt_s=0.1,
        velocity_retention=1.0,
        process_position_std_m=0.0,
        process_acceleration_std_mps2=0.0,
        last_frame_index=0,
    )

    forecast = belief.forecast([1, 3])

    covariance = forecast.joint_covariance_m2
    assert covariance[0, 3] > 0.0
    assert covariance[0, 6] > 0.0
    assert np.min(np.linalg.eigvalsh(covariance)) >= -1e-12
    assert not covariance.flags.writeable


def test_filter_recovers_a_shared_graph_mode_and_downweights_outlier_frame() -> None:
    frame_count = 8
    node_count = 4
    basis = np.ones((node_count, 1)) / np.sqrt(node_count)
    true_field = np.asarray([0.008, -0.004, 0.002])
    residual = np.repeat(true_field[None, None], frame_count, axis=0)
    residual = np.repeat(residual, node_count, axis=1)
    residual[4] += np.asarray([0.20, -0.15, 0.10])
    valid = np.ones((frame_count, node_count), dtype=bool)
    config = GraphDynamicDiscrepancyConfigV1(
        initial_position_std_m=0.05,
        initial_velocity_std_mps=0.01,
        process_position_std_m=0.0,
        process_acceleration_std_mps2=0.0,
        velocity_retention=0.0,
        observation_std_m=0.001,
        maximum_node_position_m=0.5,
        maximum_node_velocity_mps=1.0,
    )

    belief = fit_graph_dynamic_discrepancy(
        residual,
        valid,
        basis,
        frame_dt_s=1.0,
        config=config,
    )

    np.testing.assert_allclose(
        belief.position_field_m,
        np.repeat(true_field[None], node_count, axis=0),
        atol=4e-4,
    )
    assert belief.accepted_update_count == frame_count
    outlier_weights = belief.diagnostics["frame_diagnostics"][4][
        "correlation_group_robust_weight"
    ]
    assert outlier_weights[0] < 0.01


def test_nonconverged_robust_update_retains_exact_predicted_belief() -> None:
    basis = np.eye(1)
    residual = np.asarray([[[0.08, 0.0, 0.0]]])
    valid = np.ones((1, 1), dtype=bool)
    config = GraphDynamicDiscrepancyConfigV1(
        maximum_iterations=1,
        convergence_tolerance=1e-15,
        maximum_node_position_m=1.0,
    )

    belief = fit_graph_dynamic_discrepancy(
        residual,
        valid,
        basis,
        frame_dt_s=1.0,
        config=config,
    )

    np.testing.assert_array_equal(belief.state_mean, np.zeros((2, 1, 3)))
    expected = np.diag(
        [config.initial_position_std_m**2] * 3
        + [config.initial_velocity_std_mps**2] * 3
    )
    np.testing.assert_array_equal(belief.state_covariance, expected)
    assert not belief.update_accepted[0]
    assert belief.update_reasons == ("robust-fixed-point-not-converged",)


def test_no_support_propagates_without_inventing_an_update() -> None:
    basis = np.eye(1)
    residual = np.zeros((3, 1, 3))
    valid = np.zeros((3, 1), dtype=bool)
    config = GraphDynamicDiscrepancyConfigV1(
        process_position_std_m=0.002,
        process_acceleration_std_mps2=0.0,
        velocity_retention=0.0,
    )

    belief = fit_graph_dynamic_discrepancy(
        residual,
        valid,
        basis,
        frame_dt_s=1.0,
        config=config,
    )

    assert belief.accepted_update_count == 0
    assert belief.update_reasons == ("no-observation-support",) * 3
    assert np.trace(belief.state_covariance) > 0.0


def test_invalid_basis_and_covariance_budget_fail_closed() -> None:
    with pytest.raises(ValueError, match="orthonormal"):
        GraphDynamicDiscrepancyBeliefV1(
            graph_basis=np.ones((2, 1)),
            state_mean=np.zeros((2, 1, 3)),
            state_covariance=np.eye(6),
            frame_dt_s=1.0,
            velocity_retention=1.0,
            process_position_std_m=0.0,
            process_acceleration_std_mps2=0.0,
            last_frame_index=0,
        )

    belief = GraphDynamicDiscrepancyBeliefV1.from_last_residual(np.zeros((3, 3)))
    with pytest.raises(MemoryError, match="budget"):
        belief.forecast([1, 2], maximum_covariance_bytes=1)


def test_configuration_rejects_boolean_and_out_of_range_controls() -> None:
    with pytest.raises(TypeError, match="maximum_iterations"):
        GraphDynamicDiscrepancyConfigV1(
            maximum_iterations=True,  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="velocity_retention"):
        replace(GraphDynamicDiscrepancyConfigV1(), velocity_retention=1.1)
