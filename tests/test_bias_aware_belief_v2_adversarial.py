from __future__ import annotations

import numpy as np
import pytest

import bayesian_phystwin.bias_aware_belief_v2 as belief_v2
from bayesian_phystwin.bias_aware_belief_v2 import (
    BiasAwareStateUpdateConfigV2,
    update_bias_aware_state_v2,
)
from bayesian_phystwin.spd_system import SPDSolveError
from test_bias_aware_belief_v2 import _centered_problem, _legacy_config, _v2_config


def test_posterior_failure_reason_distinguishes_validation_and_solve() -> None:
    diagnostics: dict[str, object] = {}
    reason = belief_v2._posterior_failure(
        belief_v2.SPDValidationError("not positive definite"),
        diagnostics=diagnostics,
    )
    assert reason == "non-positive-definite-posterior"
    assert diagnostics["numerical_failure_type"] == "SPDValidationError"

    diagnostics = {}
    reason = belief_v2._posterior_failure(
        SPDSolveError("triangular residual failed"),
        diagnostics=diagnostics,
    )
    assert reason == "unstable-posterior-solve"
    assert diagnostics["numerical_failure_type"] == "SPDSolveError"


def test_v2_rejects_a_v1_configuration_explicitly() -> None:
    innovation, available, state_basis, bias_basis = _centered_problem()
    with pytest.raises(TypeError, match="BiasAwareStateUpdateConfigV2"):
        update_bias_aware_state_v2(
            innovation,
            available,
            state_basis,
            bias_basis,
            config=_legacy_config(),  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"symmetry_absolute_tolerance": -1.0}, "symmetry tolerances"),
        ({"symmetry_relative_tolerance": float("inf")}, "symmetry tolerances"),
        ({"solve_residual_tolerance": 0.0}, "residual tolerances"),
        ({"inverse_residual_tolerance": float("nan")}, "residual tolerances"),
    ],
)
def test_v2_config_rejects_invalid_numerical_tolerances(
    overrides: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _v2_config(**overrides)


def test_v2_accepts_custom_observation_and_anchor_variances() -> None:
    innovation, available, state_basis, bias_basis = _centered_problem()
    observation_variance = np.full(available.shape, 2.5e-6, dtype=np.float64)
    anchor = np.asarray(
        [[0.01, -0.002, 0.0], [np.nan, 0.0, 0.0]],
        dtype=np.float64,
    )
    anchor_design = np.ones((2, 1), dtype=np.float64)
    anchor_variance = np.asarray([4e-8, 9e-8], dtype=np.float64)

    result = update_bias_aware_state_v2(
        innovation,
        available,
        state_basis,
        bias_basis,
        observation_variance_m2=observation_variance,
        anchor_innovation_m=anchor,
        anchor_state_basis=anchor_design,
        anchor_variance_m2=anchor_variance,
        config=_v2_config(),
    )

    assert result.accepted
    assert result.diagnostics["independent_anchor_count"] == 1
    assert result.anchor_robust_weights.shape == (1,)
    assert result.diagnostics["final_solve_relative_residual"] < 1e-10


def test_v2_uses_default_anchor_variance() -> None:
    innovation, available, state_basis, bias_basis = _centered_problem()
    result = update_bias_aware_state_v2(
        innovation,
        available,
        state_basis,
        bias_basis,
        anchor_innovation_m=np.asarray([[0.01, 0.0, 0.0]], dtype=np.float64),
        anchor_state_basis=np.ones((1, 1), dtype=np.float64),
        config=_v2_config(),
    )

    assert result.accepted
    assert result.anchor_robust_weights.shape == (1,)


def test_v2_returns_no_support_fallback() -> None:
    point_count = 5
    result = update_bias_aware_state_v2(
        np.full((2, point_count, 3), np.nan, dtype=np.float64),
        np.zeros((2, point_count), dtype=bool),
        np.ones((point_count, 1), dtype=np.float64),
        np.zeros((point_count, 0), dtype=np.float64),
    )

    assert not result.accepted
    assert result.reason == "no-observation-support"
    assert result.diagnostics["active_view_count"] == 0
    np.testing.assert_array_equal(result.prior_reliability, 0.0)


def test_v2_accepts_a_non_diagonal_spd_state_prior() -> None:
    point_count = 16
    coordinate = np.linspace(-1.0, 1.0, point_count)
    state_basis = np.column_stack(
        (
            coordinate,
            np.square(coordinate) - np.mean(np.square(coordinate)),
        )
    )
    innovation = np.zeros((2, point_count, 3), dtype=np.float64)
    innovation[..., 0] = 0.01 * state_basis[:, 0] - 0.004 * state_basis[:, 1]
    prior = np.asarray([[0.04, 0.006], [0.006, 0.02]], dtype=np.float64)

    result = update_bias_aware_state_v2(
        innovation,
        np.ones((2, point_count), dtype=bool),
        state_basis,
        np.zeros((point_count, 0), dtype=np.float64),
        state_prior_covariance_m2=prior,
        config=_v2_config(),
    )

    assert result.accepted
    assert result.diagnostics["state_prior_numerical_path"] == "cholesky"
    state_prior_spd = result.diagnostics["state_prior_spd"]
    assert isinstance(state_prior_spd, dict)
    assert state_prior_spd["dimension"] == 2


def test_v2_anchor_only_zero_view_update_has_no_bias_block() -> None:
    point_count = 4
    result = update_bias_aware_state_v2(
        np.zeros((0, point_count, 3), dtype=np.float64),
        np.zeros((0, point_count), dtype=bool),
        np.ones((point_count, 1), dtype=np.float64),
        np.zeros((point_count, 0), dtype=np.float64),
        anchor_innovation_m=np.asarray([[0.008, 0.0, 0.0]], dtype=np.float64),
        anchor_state_basis=np.ones((1, 1), dtype=np.float64),
        config=_v2_config(),
    )

    assert result.accepted
    assert result.camera_biases_m.shape == (0, 3)
    assert result.shared_bias_coefficients_m.shape == (0, 3)
    assert result.diagnostics["active_view_count"] == 0
    assert result.diagnostics["maximum_state_bias_posterior_correlation"] == 0.0


def test_v2_fails_closed_during_an_irls_iteration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    innovation, available, state_basis, bias_basis = _centered_problem()
    original = belief_v2._factor_system

    def fail_first(
        matrix: np.ndarray,
        *,
        name: str,
        config: BiasAwareStateUpdateConfigV2,
    ) -> object:
        if name == "posterior normal iteration 1":
            raise SPDSolveError("deliberate iteration solve failure")
        return original(matrix, name=name, config=config)

    monkeypatch.setattr(belief_v2, "_factor_system", fail_first)
    result = update_bias_aware_state_v2(
        innovation,
        available,
        state_basis,
        bias_basis,
        config=_v2_config(),
    )

    assert not result.accepted
    assert result.reason == "unstable-posterior-solve"
    assert result.diagnostics["numerical_failure_type"] == "SPDSolveError"


def test_v2_falls_back_on_implausible_state_update() -> None:
    innovation, available, state_basis, bias_basis = _centered_problem()
    result = update_bias_aware_state_v2(
        innovation,
        available,
        state_basis,
        bias_basis,
        config=_v2_config(maximum_state_update_m=1e-6),
    )

    assert not result.accepted
    assert result.reason == "implausible-state-update"
    assert result.diagnostics["maximum_state_update_m"] > 1e-6
    assert result.state_coefficients_m.tobytes() == np.zeros((1, 3)).tobytes()


def test_v2_falls_back_when_covariance_reconstruction_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    innovation, available, state_basis, bias_basis = _centered_problem()

    def fail_reconstruction(self: object) -> np.ndarray:
        raise SPDSolveError("deliberate covariance failure")

    monkeypatch.setattr(
        belief_v2.SPDSystem,
        "reconstruct_inverse",
        fail_reconstruction,
    )
    result = update_bias_aware_state_v2(
        innovation,
        available,
        state_basis,
        bias_basis,
        config=_v2_config(),
    )

    assert not result.accepted
    assert result.reason == "unstable-posterior-covariance"
    assert result.diagnostics["numerical_failure_type"] == "SPDSolveError"


def test_v2_falls_back_when_exported_inverse_residual_exceeds_limit() -> None:
    innovation, available, state_basis, bias_basis = _centered_problem()
    result = update_bias_aware_state_v2(
        innovation,
        available,
        state_basis,
        bias_basis,
        config=_v2_config(inverse_residual_tolerance=1e-30),
    )

    assert not result.accepted
    assert result.reason == "unstable-posterior-covariance"
    assert result.diagnostics["inverse_relative_residual"] > 1e-30


@pytest.mark.parametrize(
    ("keyword", "value", "message"),
    [
        ("prior_reliability", np.ones((1, 1)), "prior reliability changed"),
        ("observation_variance_m2", np.ones((1, 1)), "camera variance changed"),
        (
            "observation_variance_m2",
            np.full((4, 12), -1.0),
            "camera variance must be positive",
        ),
        ("anchor_state_basis", np.ones((1, 1)), "anchor innovation is missing"),
        ("anchor_variance_m2", np.ones(1), "anchor innovation is missing"),
    ],
)
def test_v2_rejects_malformed_optional_inputs(
    keyword: str,
    value: np.ndarray,
    message: str,
) -> None:
    innovation, available, state_basis, bias_basis = _centered_problem()
    with pytest.raises(ValueError, match=message):
        update_bias_aware_state_v2(
            innovation,
            available,
            state_basis,
            bias_basis,
            config=_v2_config(),
            **{keyword: value},
        )
