import numpy as np

from causal4d.benchmark import CounterfactualBenchmarkConfig, build_protocol
from causal4d.simulator import graph_laplacian, simulate


def _config() -> CounterfactualBenchmarkConfig:
    return CounterfactualBenchmarkConfig(
        frame_count=20,
        training_repeats=2,
        parameter_grid_count=3,
        fit_frame_stride=2,
    )


def test_protocol_contains_three_distinct_graph_objects() -> None:
    protocols = build_protocol(_config())

    assert [protocol.graph_object.name for protocol in protocols] == [
        "rope",
        "cloth",
        "soft_block",
    ]
    assert len({protocol.graph_object.node_count for protocol in protocols}) >= 2
    for protocol in protocols:
        laplacian = graph_laplacian(protocol.graph_object)
        assert laplacian.shape == (
            protocol.graph_object.node_count,
            protocol.graph_object.node_count,
        )
        assert np.allclose(laplacian, laplacian.T)
        assert np.allclose(np.sum(laplacian, axis=1), 0.0)


def test_simulation_is_deterministic_and_parameter_dependent() -> None:
    config = _config()
    protocol = build_protocol(config)[0]
    action = protocol.test_action
    condition = protocol.test_conditions[0]
    parameters = protocol.graph_object.true_parameters

    first = simulate(
        protocol.graph_object, action, parameters, condition, config.simulator
    )
    second = simulate(
        protocol.graph_object, action, parameters, condition, config.simulator
    )
    stiffer = simulate(
        protocol.graph_object,
        action,
        type(parameters)(
            stiffness=parameters.stiffness * 1.3,
            damping=parameters.damping,
            contact_gain=parameters.contact_gain,
        ),
        condition,
        config.simulator,
    )

    assert first.shape == (config.frame_count, protocol.graph_object.node_count, 2)
    assert np.array_equal(first, second)
    assert np.all(np.isfinite(first))
    assert not np.allclose(first, stiffer)


def test_shifted_contact_is_a_world_change_hidden_from_plan_model() -> None:
    config = _config()
    protocol = build_protocol(config)[1]
    action = protocol.test_action
    parameters = protocol.graph_object.true_parameters
    matched, shifted = protocol.test_conditions

    matched_truth = simulate(
        protocol.graph_object, action, parameters, matched, config.simulator
    )
    shifted_truth = simulate(
        protocol.graph_object, action, parameters, shifted, config.simulator
    )
    matched_plan = simulate(
        protocol.graph_object,
        action,
        parameters,
        matched.plan_model(),
        config.simulator,
    )
    shifted_plan = simulate(
        protocol.graph_object,
        action,
        parameters,
        shifted.plan_model(),
        config.simulator,
    )

    assert not np.allclose(matched_truth, shifted_truth)
    assert np.array_equal(matched_plan, shifted_plan)
    assert shifted.shift_contact_nodes
    assert shifted.contact_delay_steps > 0


def test_contact_spread_is_physical_and_oracle_view_preserves_it() -> None:
    config = _config()
    protocol = build_protocol(config)[1]
    action = protocol.test_action
    parameters = protocol.graph_object.true_parameters
    concentrated = protocol.test_conditions[0]
    spread = type(concentrated)(
        name="spread",
        contact_gain_multiplier=0.8,
        contact_delay_steps=1,
        contact_spread=0.3,
        control_rotation_radians=0.1,
        nonlinear_stiffening=0.2,
    )

    concentrated_trajectory = simulate(
        protocol.graph_object,
        action,
        parameters,
        concentrated,
        config.simulator,
    )
    spread_trajectory = simulate(
        protocol.graph_object,
        action,
        parameters,
        spread,
        config.simulator,
    )
    oracle = spread.oracle_contact_model()

    assert not np.allclose(concentrated_trajectory, spread_trajectory)
    assert oracle.contact_spread == spread.contact_spread
    assert oracle.contact_delay_steps == spread.contact_delay_steps
    assert oracle.control_rotation_radians == spread.control_rotation_radians
    assert oracle.nonlinear_stiffening == 0.0
    assert spread.plan_model().contact_spread == 0.0
