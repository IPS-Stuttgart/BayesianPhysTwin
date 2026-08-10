from __future__ import annotations

import numpy as np
import pytest

from bayesian_phystwin.graph_dynamic_discrepancy import (
    GraphDynamicDiscrepancyBeliefV1,
    GraphDynamicDiscrepancyConfigV1,
    fit_graph_dynamic_discrepancy,
)


def test_contracts_reject_lossy_boolean_and_reason_coercion() -> None:
    with pytest.raises(ValueError, match="Boolean vector"):
        GraphDynamicDiscrepancyBeliefV1(
            graph_basis=np.eye(1),
            state_mean=np.zeros((2, 1, 3)),
            state_covariance=np.eye(6),
            frame_dt_s=1.0,
            velocity_retention=1.0,
            process_position_std_m=0.0,
            process_acceleration_std_mps2=0.0,
            last_frame_index=0,
            update_accepted=np.asarray([1]),
            update_reasons=("accepted",),
        )
    with pytest.raises(ValueError, match="nonempty strings"):
        GraphDynamicDiscrepancyBeliefV1(
            graph_basis=np.eye(1),
            state_mean=np.zeros((2, 1, 3)),
            state_covariance=np.eye(6),
            frame_dt_s=1.0,
            velocity_retention=1.0,
            process_position_std_m=0.0,
            process_acceleration_std_mps2=0.0,
            last_frame_index=0,
            update_accepted=np.asarray([True]),
            update_reasons=("",),
        )
    with pytest.raises(ValueError, match="valid must be a Boolean array"):
        fit_graph_dynamic_discrepancy(
            np.zeros((1, 1, 3)),
            np.ones((1, 1), dtype=np.int64),
            np.eye(1),
            frame_dt_s=1.0,
        )


def test_private_numerical_contracts_reject_invalid_factors() -> None:
    from bayesian_phystwin import _graph_dynamic_discrepancy_common as module

    with pytest.raises(ValueError, match="positive definite"):
        module._positive_definite_precision(
            np.diag([1.0, 0.0]),
            name="covariance",
        )
    with pytest.raises(ValueError, match="positive definite"):
        module._covariance_from_precision(
            np.diag([1.0, 0.0]),
            name="precision",
        )
    root = module._positive_semidefinite_root(
        np.zeros((2, 2)),
        name="zero covariance",
    )
    assert root.shape == (2, 0)
    with pytest.raises(ValueError, match="positive semidefinite"):
        module._positive_semidefinite_root(
            np.diag([1.0, -1.0]),
            name="indefinite covariance",
        )


def test_belief_rejects_nonfinite_diagnostics_and_real_controls() -> None:
    with pytest.raises(ValueError, match="finite JSON"):
        GraphDynamicDiscrepancyBeliefV1(
            graph_basis=np.eye(1),
            state_mean=np.zeros((2, 1, 3)),
            state_covariance=np.eye(6),
            frame_dt_s=1.0,
            velocity_retention=1.0,
            process_position_std_m=0.0,
            process_acceleration_std_mps2=0.0,
            last_frame_index=0,
            diagnostics={"invalid": float("nan")},
        )
    with pytest.raises(TypeError, match="real number"):
        GraphDynamicDiscrepancyConfigV1(
            initial_position_std_m=True,  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="finite"):
        GraphDynamicDiscrepancyConfigV1(initial_position_std_m=float("inf"))


def test_endpoint_posterior_accepts_coordinate_variance_and_rejects_shape() -> None:
    mean = np.zeros((2, 3))
    variance = np.asarray([[1e-6, 2e-6, 3e-6], [4e-6, 5e-6, 6e-6]])

    belief = GraphDynamicDiscrepancyBeliefV1.from_independent_endpoint_posterior(
        mean,
        variance,
    )

    np.testing.assert_array_equal(
        np.diag(belief.state_covariance)[:6],
        variance.reshape(-1),
    )
    with pytest.raises(ValueError, match="variance_m2 must have shape"):
        GraphDynamicDiscrepancyBeliefV1.from_independent_endpoint_posterior(
            mean,
            np.zeros((2, 2)),
        )
    with pytest.raises(ValueError, match="nonnegative"):
        GraphDynamicDiscrepancyBeliefV1.from_independent_endpoint_posterior(
            mean,
            -np.ones(2),
        )


def test_forecast_supports_registered_node_subset_and_acceleration_sequence() -> None:
    belief = GraphDynamicDiscrepancyBeliefV1.from_last_residual(
        np.zeros((3, 3)),
        frame_dt_s=0.5,
    )
    acceleration = np.zeros((2, 3, 3))
    acceleration[:, 1, 1] = 2.0

    forecast = belief.forecast(
        [1, 2],
        node_indices=np.asarray([1], dtype=np.int64),
        modal_acceleration_mps2=acceleration,
    )

    assert forecast.node_indices.tolist() == [1]
    np.testing.assert_allclose(forecast.mean_m[:, 0, 1], [0.25, 1.0])
    assert belief.velocity_coefficients_mps.shape == (3, 3)
    assert belief.velocity_field_mps.shape == (3, 3)


@pytest.mark.parametrize(
    ("horizons", "message"),
    [
        ([], "empty"),
        ([True], "integers"),
        ([0], "positive"),
        ([2, 1], "strictly increasing"),
    ],
)
def test_forecast_rejects_invalid_horizon_contracts(
    horizons: list[object],
    message: str,
) -> None:
    belief = GraphDynamicDiscrepancyBeliefV1.from_last_residual(
        np.zeros((2, 3))
    )
    with pytest.raises(ValueError, match=message):
        belief.forecast(horizons)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("nodes", "message"),
    [
        ([], "empty"),
        ([0.0], "integers"),
        ([True], "integers"),
        ([-1], "outside"),
        ([2], "outside"),
        ([0, 0], "unique"),
    ],
)
def test_forecast_rejects_invalid_node_contracts(
    nodes: list[object],
    message: str,
) -> None:
    belief = GraphDynamicDiscrepancyBeliefV1.from_last_residual(
        np.zeros((2, 3))
    )
    with pytest.raises(ValueError, match=message):
        belief.forecast([1], node_indices=nodes)  # type: ignore[arg-type]


def test_forecast_rejects_invalid_acceleration_and_budget_types() -> None:
    belief = GraphDynamicDiscrepancyBeliefV1.from_last_residual(
        np.zeros((2, 3))
    )
    with pytest.raises(ValueError, match="must have shape"):
        belief.forecast([1, 2], modal_acceleration_mps2=np.zeros((1, 3)))
    invalid = np.zeros((2, 2, 3))
    invalid[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        belief.forecast([1, 2], modal_acceleration_mps2=invalid)
    with pytest.raises(TypeError, match="integer"):
        belief.forecast([1], maximum_covariance_bytes=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="at least 1"):
        belief.forecast([1], maximum_covariance_bytes=0)


@pytest.mark.parametrize("covariance_mode", ["global", "node", "complete"])
def test_filter_accepts_all_declared_covariance_shapes_and_group_metadata(
    covariance_mode: str,
) -> None:
    frame_count = 2
    node_count = 2
    residual = np.full((frame_count, node_count, 3), 0.001)
    valid = np.ones((frame_count, node_count), dtype=bool)
    base = np.eye(3) * 1e-4
    if covariance_mode == "global":
        covariance = base
    elif covariance_mode == "node":
        covariance = np.repeat(base[None], node_count, axis=0)
    else:
        covariance = np.repeat(
            base[None, None],
            frame_count * node_count,
            axis=0,
        ).reshape(frame_count, node_count, 3, 3)
    reliability = np.asarray([[1.0, 0.5], [0.8, 1.0]])
    groups = np.asarray([["left", "right"], ["left", "right"]])

    belief = fit_graph_dynamic_discrepancy(
        residual,
        valid,
        np.eye(node_count),
        frame_dt_s=0.1,
        observation_covariance_m2=covariance,
        prior_reliability=reliability,
        correlation_group_ids=groups,
    )

    assert belief.accepted_update_count == frame_count
    assert belief.diagnostics["frame_diagnostics"][0][
        "correlation_group_ids"
    ] == ["left", "right"]


@pytest.mark.parametrize(
    ("covariance", "message"),
    [
        (np.zeros((2, 2)), "must have shape"),
        (np.asarray([[1.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]), "symmetric"),
        (np.diag([1.0, 1.0, 0.0]), "positive definite"),
        (np.eye(3) * np.nan, "finite"),
    ],
)
def test_filter_rejects_invalid_observation_covariance(
    covariance: np.ndarray,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        fit_graph_dynamic_discrepancy(
            np.zeros((1, 1, 3)),
            np.ones((1, 1), dtype=bool),
            np.eye(1),
            frame_dt_s=1.0,
            observation_covariance_m2=covariance,
        )


def test_filter_rejects_invalid_reliability_and_group_contracts() -> None:
    residual = np.zeros((1, 1, 3))
    valid = np.ones((1, 1), dtype=bool)
    with pytest.raises(ValueError, match="shape changed"):
        fit_graph_dynamic_discrepancy(
            residual,
            valid,
            np.eye(1),
            frame_dt_s=1.0,
            prior_reliability=np.ones((2, 1)),
        )
    with pytest.raises(ValueError, match="lie in"):
        fit_graph_dynamic_discrepancy(
            residual,
            valid,
            np.eye(1),
            frame_dt_s=1.0,
            prior_reliability=np.asarray([[1.1]]),
        )
    with pytest.raises(ValueError, match="shape changed"):
        fit_graph_dynamic_discrepancy(
            residual,
            valid,
            np.eye(1),
            frame_dt_s=1.0,
            correlation_group_ids=np.asarray(["group"]),
        )
    with pytest.raises(ValueError, match="nonempty"):
        fit_graph_dynamic_discrepancy(
            residual,
            valid,
            np.eye(1),
            frame_dt_s=1.0,
            correlation_group_ids=np.asarray([[""]]),
        )


def test_filter_rejects_unidentifiable_support_and_nonstring_groups() -> None:
    residual = np.asarray([[[0.01, 0.0, 0.0], [0.0, 0.0, 0.0]]])
    valid = np.asarray([[True, False]])
    basis = np.asarray([[0.0], [1.0]])

    belief = fit_graph_dynamic_discrepancy(
        residual,
        valid,
        basis,
        frame_dt_s=1.0,
    )

    assert belief.update_reasons == ("no-identifiable-graph-support",)
    with pytest.raises(ValueError, match="literal nonempty strings"):
        fit_graph_dynamic_discrepancy(
            residual,
            valid,
            basis,
            frame_dt_s=1.0,
            correlation_group_ids=np.asarray([[1, 1]], dtype=object),
        )


def test_filter_rejects_ill_conditioned_and_implausible_updates() -> None:
    residual = np.asarray([[[0.01, 0.0, 0.0]]])
    valid = np.ones((1, 1), dtype=bool)
    ill_conditioned = GraphDynamicDiscrepancyConfigV1(
        maximum_condition_number=1.000001,
        observation_std_m=0.001,
        maximum_node_position_m=1.0,
    )

    rejected_condition = fit_graph_dynamic_discrepancy(
        residual,
        valid,
        np.eye(1),
        frame_dt_s=1.0,
        config=ill_conditioned,
    )

    assert rejected_condition.update_reasons == ("ill-conditioned-posterior",)
    implausible = GraphDynamicDiscrepancyConfigV1(
        observation_std_m=0.1,
        maximum_node_position_m=1e-6,
        maximum_node_velocity_mps=1.0,
    )
    rejected_plausibility = fit_graph_dynamic_discrepancy(
        residual,
        valid,
        np.eye(1),
        frame_dt_s=1.0,
        config=implausible,
    )
    assert rejected_plausibility.update_reasons == (
        "implausible-discrepancy-update",
    )


def test_zero_initial_velocity_uncertainty_is_supported_without_jitter() -> None:
    residual = np.asarray([[[0.003, -0.001, 0.002]]] * 3)
    config = GraphDynamicDiscrepancyConfigV1(
        initial_velocity_std_mps=0.0,
        process_position_std_m=0.0,
        process_acceleration_std_mps2=0.0,
        velocity_retention=0.0,
        observation_std_m=0.001,
    )

    belief = fit_graph_dynamic_discrepancy(
        residual,
        np.ones((3, 1), dtype=bool),
        np.eye(1),
        frame_dt_s=1.0,
        config=config,
    )

    assert belief.accepted_update_count == 3
    np.testing.assert_array_equal(belief.velocity_coefficients_mps, 0.0)
    np.testing.assert_array_equal(belief.state_covariance[3:, :], 0.0)
    np.testing.assert_array_equal(belief.state_covariance[:, 3:], 0.0)
