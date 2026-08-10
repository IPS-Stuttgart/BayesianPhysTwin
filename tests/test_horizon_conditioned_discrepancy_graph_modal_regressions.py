from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from bayesian_phystwin.graph_dynamic_discrepancy import (
    GraphDynamicDiscrepancyBeliefV1,
    GraphDynamicDiscrepancyConfigV1,
    fit_graph_dynamic_discrepancy,
)


def test_group_cap_changes_power_but_not_outlier_classification() -> None:
    residual = np.asarray(
        [[[0.004, 0.0, 0.0], [0.004, 0.0, 0.0], [0.03, 0.0, 0.0]]]
    )
    valid = np.ones((1, 3), dtype=bool)
    groups = np.asarray([["shared", "shared", "shared"]])
    common = dict(
        initial_velocity_std_mps=0.0,
        process_position_std_m=0.0,
        process_acceleration_std_mps2=0.0,
        velocity_retention=0.0,
        observation_std_m=0.002,
        maximum_node_position_m=1.0,
    )
    uncapped = fit_graph_dynamic_discrepancy(
        residual,
        valid,
        np.eye(3),
        frame_dt_s=1.0,
        correlation_group_ids=groups,
        config=GraphDynamicDiscrepancyConfigV1(
            effective_samples_per_correlation_group=3.0,
            **common,
        ),
    )
    capped = fit_graph_dynamic_discrepancy(
        residual,
        valid,
        np.eye(3),
        frame_dt_s=1.0,
        correlation_group_ids=groups,
        config=GraphDynamicDiscrepancyConfigV1(
            effective_samples_per_correlation_group=1.0,
            **common,
        ),
    )

    uncapped_diagnostics = uncapped.diagnostics["frame_diagnostics"][0]
    capped_diagnostics = capped.diagnostics["frame_diagnostics"][0]
    np.testing.assert_allclose(
        capped_diagnostics["correlation_group_robust_weight"],
        uncapped_diagnostics["correlation_group_robust_weight"],
    )
    assert capped_diagnostics["correlation_group_power"] == [pytest.approx(1 / 3)]
    assert uncapped_diagnostics["correlation_group_power"] == [1.0]
    assert not np.allclose(capped.position_field_m, uncapped.position_field_m)


def test_public_contracts_reject_numeric_strings_and_falsey_config() -> None:
    with pytest.raises(ValueError, match="real numeric"):
        GraphDynamicDiscrepancyBeliefV1.from_last_residual(
            np.asarray([["0", "0", "0"]])
        )
    with pytest.raises(ValueError, match="real numeric"):
        GraphDynamicDiscrepancyBeliefV1.from_independent_endpoint_posterior(
            np.zeros((1, 3)),
            np.asarray(["0.1"]),
        )
    with pytest.raises(ValueError, match="real numeric"):
        fit_graph_dynamic_discrepancy(
            np.asarray([[["0", "0", "0"]]]),
            np.ones((1, 1), dtype=bool),
            np.eye(1),
            frame_dt_s=1.0,
        )

    class FalseyConfig:
        def __bool__(self) -> bool:
            return False

    with pytest.raises(TypeError, match="GraphDynamicDiscrepancyConfigV1"):
        fit_graph_dynamic_discrepancy(
            np.zeros((1, 1, 3)),
            np.ones((1, 1), dtype=bool),
            np.eye(1),
            frame_dt_s=1.0,
            config=FalseyConfig(),  # type: ignore[arg-type]
        )


def test_forecast_result_rejects_lossy_manual_indices() -> None:
    from bayesian_phystwin.graph_dynamic_discrepancy import (
        GraphDynamicDiscrepancyForecastV1,
    )

    with pytest.raises(ValueError, match="integers"):
        GraphDynamicDiscrepancyForecastV1(
            horizon_steps=np.asarray([1.5]),
            node_indices=np.asarray([0]),
            mean_m=np.zeros((1, 1, 3)),
            joint_covariance_m2=np.eye(3),
        )
    with pytest.raises(ValueError, match="strictly increasing"):
        GraphDynamicDiscrepancyForecastV1(
            horizon_steps=np.asarray([1, 1]),
            node_indices=np.asarray([0]),
            mean_m=np.zeros((2, 1, 3)),
            joint_covariance_m2=np.eye(6),
        )
    with pytest.raises(ValueError, match="unique"):
        GraphDynamicDiscrepancyForecastV1(
            horizon_steps=np.asarray([1]),
            node_indices=np.asarray([0, 0]),
            mean_m=np.zeros((1, 2, 3)),
            joint_covariance_m2=np.eye(6),
        )


def test_configuration_requires_valid_condition_limit() -> None:
    with pytest.raises(ValueError, match="at least one"):
        GraphDynamicDiscrepancyConfigV1(maximum_condition_number=0.5)


def test_unit_weight_limit_matches_dense_linear_gaussian_filter() -> None:
    from bayesian_phystwin import _graph_dynamic_discrepancy_common as module

    basis = np.asarray([[1.0], [1.0]]) / np.sqrt(2.0)
    residual = np.asarray(
        [
            [[0.0010, -0.0005, 0.0002], [0.0008, -0.0004, 0.0001]],
            [[0.0011, -0.0005, 0.0002], [0.0009, -0.0003, 0.0002]],
            [[0.0012, -0.0004, 0.0003], [0.0010, -0.0002, 0.0002]],
        ]
    )
    frame_dt = 0.2
    observation_variance = 1e-4
    config = GraphDynamicDiscrepancyConfigV1(
        initial_position_std_m=0.02,
        initial_velocity_std_mps=0.01,
        process_position_std_m=0.001,
        process_acceleration_std_mps2=0.01,
        velocity_retention=0.9,
        observation_std_m=np.sqrt(observation_variance),
        maximum_node_position_m=1.0,
        maximum_node_velocity_mps=1.0,
    )
    groups = np.asarray(
        [[f"frame-{frame}-node-{node}" for node in range(2)] for frame in range(3)]
    )

    belief = fit_graph_dynamic_discrepancy(
        residual,
        np.ones((3, 2), dtype=bool),
        basis,
        frame_dt_s=frame_dt,
        correlation_group_ids=groups,
        config=config,
    )

    transition, process_noise, _ = module._transition_and_noise(
        1,
        frame_dt_s=frame_dt,
        velocity_retention=config.velocity_retention,
        process_position_std_m=config.process_position_std_m,
        process_acceleration_std_mps2=config.process_acceleration_std_mps2,
    )
    design = np.concatenate(
        (
            np.kron(basis, np.eye(3)),
            np.zeros((6, 3)),
        ),
        axis=1,
    )
    observation_precision = np.eye(6) / observation_variance
    mean = np.zeros(6)
    covariance = np.diag(
        [config.initial_position_std_m**2] * 3
        + [config.initial_velocity_std_mps**2] * 3
    )
    for frame in range(3):
        if frame:
            mean = transition @ mean
            covariance = transition @ covariance @ transition.T + process_noise
        prior_precision = np.linalg.inv(covariance)
        posterior_precision = (
            prior_precision + design.T @ observation_precision @ design
        )
        covariance = np.linalg.inv(posterior_precision)
        mean = covariance @ (
            prior_precision @ mean
            + design.T @ observation_precision @ residual[frame].reshape(-1)
        )

    assert all(
        diagnostics["correlation_group_robust_weight"] == [1.0, 1.0]
        for diagnostics in belief.diagnostics["frame_diagnostics"]
    )
    np.testing.assert_allclose(
        belief.state_mean.reshape(2, -1).reshape(-1),
        mean,
        atol=1e-12,
        rtol=1e-10,
    )
    np.testing.assert_allclose(
        belief.state_covariance,
        covariance,
        atol=1e-12,
        rtol=1e-10,
    )
