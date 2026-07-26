from __future__ import annotations

import json

import numpy as np

from bayesian_phystwin.propagated_state_belief import PropagatedStateBeliefConfig
from bayesian_phystwin.propagated_state_correction import (
    PropagatedStateCorrection,
    PropagatedStateSelectionConfig,
    decode_limited_state_weights,
    modal_state_parameter_fields,
    scale_posterior_covariance_for_state_limits,
    select_propagated_state_update,
    write_propagated_state_correction,
)
def _orthonormal_basis(point_count: int = 12, rank: int = 4) -> np.ndarray:
    coordinate = np.linspace(-1.0, 1.0, point_count)
    values = np.column_stack([coordinate**degree for degree in range(rank)])
    return np.linalg.qr(values)[0]


def _selection_problem(*, state_is_useful: bool):
    rng = np.random.default_rng(4)
    frame_count = 7
    point_count = 12
    basis = _orthonormal_basis(point_count)
    position_fields, velocity_fields, position_steps, velocity_steps = (
        modal_state_parameter_fields(
            basis,
            position_step_m=0.005,
            velocity_step_mps=0.05,
        )
    )
    parameter_count = position_fields.shape[2] + velocity_fields.shape[2]
    response = rng.normal(
        scale=0.001,
        size=(frame_count, point_count, 3, parameter_count),
    )
    response *= np.linspace(0.0, 2.0, frame_count)[:, None, None, None]
    weights = np.zeros(parameter_count)
    weights[1] = 1.5 if state_is_useful else 0.0
    persistent_coefficients = np.zeros((basis.shape[1], 3))
    persistent_coefficients[0, 0] = 0.003
    persistent_coefficients[1, 1] = 0.001
    persistent = basis @ persistent_coefficients
    innovation = np.einsum("tnck,k->tnc", response, weights) + persistent[None]
    return (
        innovation,
        np.ones((frame_count, point_count), dtype=bool),
        response,
        basis,
        position_steps,
        velocity_steps,
    )


def test_modal_parameter_fields_have_declared_node_steps() -> None:
    basis = _orthonormal_basis()
    position, velocity, _, _ = modal_state_parameter_fields(
        basis,
        position_step_m=0.004,
        velocity_step_mps=0.03,
    )

    np.testing.assert_allclose(
        np.max(np.linalg.norm(position, axis=1), axis=0),
        0.004,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        np.max(np.linalg.norm(velocity, axis=1), axis=0),
        0.03,
        atol=1e-12,
    )


def test_state_weight_limit_is_radial_within_each_state_family() -> None:
    basis = _orthonormal_basis()
    _, _, position_steps, velocity_steps = modal_state_parameter_fields(
        basis,
        position_step_m=0.005,
        velocity_step_mps=0.05,
    )
    weights = np.full(6 * basis.shape[1], 20.0)

    limited, position, velocity, diagnostics = decode_limited_state_weights(
        weights,
        basis,
        position_steps,
        velocity_steps,
        maximum_position_update_m=0.01,
        maximum_velocity_update_mps=0.1,
    )

    assert np.max(np.linalg.norm(position, axis=1)) <= 0.01 + 1e-12
    assert np.max(np.linalg.norm(velocity, axis=1)) <= 0.1 + 1e-12
    assert diagnostics["position"]["limit_applied"]
    assert diagnostics["velocity"]["limit_applied"]
    assert np.all(limited < weights)


def test_prefix_guard_accepts_predictive_action_propagated_state() -> None:
    innovation, available, response, basis, position_steps, velocity_steps = (
        _selection_problem(state_is_useful=True)
    )
    result = select_propagated_state_update(
        innovation,
        available,
        response,
        basis,
        basis,
        position_steps,
        velocity_steps,
        observation_variance_m2=np.full(available.shape, 1e-8),
        belief_config=PropagatedStateBeliefConfig(
            state_weight_prior_std=100.0,
            shared_bias_prior_std_m=100.0,
        ),
        selection_config=PropagatedStateSelectionConfig(
            minimum_validation_improvement_fraction=0.01,
        ),
    )

    assert result.accepted
    assert result.diagnostics["validation_improvement_fraction"] > 0.01
    assert np.linalg.norm(result.position_update_m) > 0.0
    assert np.max(np.linalg.norm(basis @ result.shared_bias_coefficients_m, axis=1)) <= (
        0.05 + 1e-12
    )


def test_prefix_guard_returns_exact_zero_state_when_persistence_wins() -> None:
    innovation, available, response, basis, position_steps, velocity_steps = (
        _selection_problem(state_is_useful=False)
    )
    result = select_propagated_state_update(
        innovation,
        available,
        response,
        basis,
        basis,
        position_steps,
        velocity_steps,
        observation_variance_m2=np.full(available.shape, 1e-8),
        belief_config=PropagatedStateBeliefConfig(
            state_weight_prior_std=100.0,
            shared_bias_prior_std_m=100.0,
        ),
    )

    assert not result.accepted
    assert result.reason == "prefix-validation-regret-guard"
    assert (
        result.state_weights.tobytes() == np.zeros_like(result.state_weights).tobytes()
    )
    assert (
        result.position_update_m.tobytes()
        == np.zeros_like(result.position_update_m).tobytes()
    )
    assert (
        result.shared_bias_coefficients_m.tobytes()
        == np.zeros_like(result.shared_bias_coefficients_m).tobytes()
    )


def test_typed_artifact_is_checksummed_and_records_information_boundary(
    tmp_path,
) -> None:
    basis = _orthonormal_basis()
    rank = basis.shape[1]
    correction = PropagatedStateCorrection(
        case_id="development_case",
        graph_basis=basis,
        graph_eigenvalues=np.arange(rank, dtype=float),
        position_coefficient_steps_m=np.full(rank, 0.01),
        velocity_coefficient_steps_mps=np.full(rank, 0.1),
        state_weights=np.zeros(6 * rank),
        shared_bias_coefficients_m=np.zeros((rank, 3)),
        posterior_covariance=np.eye(9 * rank),
        accepted_state_update=False,
        selection_reason="fallback",
        prefix_frame_start=4,
        fit_frame_stop=8,
        prefix_frame_stop=11,
        information_boundary={
            "forecast_frames_used_for_fit_or_selection": False,
            "released_case_role": "implementation_diagnostic_only",
        },
        source_checksums={"source": "0" * 64},
    )

    record = write_propagated_state_correction(tmp_path / "correction", correction)
    manifest = json.loads((tmp_path / "correction.json").read_text())

    assert manifest["artifact_id"] == correction.artifact_id
    assert manifest["arrays_sha256"] == record["arrays_sha256"]
    assert not manifest["information_boundary"][
        "forecast_frames_used_for_fit_or_selection"
    ]


def test_state_limit_scaling_transforms_covariance_and_cross_terms() -> None:
    rank = 1
    covariance = np.ones((9, 9))

    scaled = scale_posterior_covariance_for_state_limits(
        covariance,
        graph_rank=rank,
        position_scale=0.5,
        velocity_scale=0.25,
        shared_bias_scale=0.1,
    )

    assert scaled[0, 0] == 0.25
    assert scaled[3, 3] == 0.0625
    assert scaled[0, 3] == 0.125
    assert scaled[0, 6] == 0.05
    assert np.isclose(scaled[6, 6], 0.01)
