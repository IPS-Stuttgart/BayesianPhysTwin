from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

import bayesian_phystwin.bias_aware_belief_v2 as belief_v2
from bayesian_phystwin.bias_aware_belief import (
    BiasAwareStateUpdateConfig,
    update_bias_aware_state,
)
from bayesian_phystwin.bias_aware_belief_v2 import (
    BIAS_AWARE_BELIEF_V2_IMPLEMENTATION,
    BIAS_AWARE_BELIEF_V2_VERSION,
    BiasAwareStateUpdateConfigV2,
    update_bias_aware_state_v2,
)
from bayesian_phystwin.spd_system import SPDConditionError, SPD_SYSTEM_SCHEMA

FROZEN_V1_GIT_BLOB_SHA1 = "80994687a44b798c6b33089bfd4f1858911e0837"


def _git_blob_sha1(path: Path) -> str:
    content = path.read_bytes()
    header = f"blob {len(content)}\0".encode("ascii")
    return hashlib.sha1(header + content).hexdigest()  # noqa: S324


def _centered_problem(
    *,
    view_count: int = 4,
    point_count: int = 12,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    state_basis = np.linspace(-1.0, 1.0, point_count)[:, None]
    shared_bias_basis = np.ones((point_count, 1), dtype=np.float64)
    innovation = np.zeros((view_count, point_count, 3), dtype=np.float64)
    innovation[..., 0] = 0.012 * state_basis[:, 0] + 0.018
    innovation[..., 1] = -0.004 * state_basis[:, 0] + 0.006
    available = np.ones((view_count, point_count), dtype=bool)
    return innovation, available, state_basis, shared_bias_basis


def _legacy_config() -> BiasAwareStateUpdateConfig:
    return BiasAwareStateUpdateConfig(
        observation_std_m=0.001,
        anchor_std_m=0.0002,
        state_prior_std_m=0.1,
        shared_bias_prior_std_m=0.1,
        camera_bias_prior_std_m=0.1,
        effective_samples_per_view=10.0,
        maximum_iterations=6,
    )


def _v2_config(**overrides: object) -> BiasAwareStateUpdateConfigV2:
    values: dict[str, object] = {
        "observation_std_m": 0.001,
        "anchor_std_m": 0.0002,
        "state_prior_std_m": 0.1,
        "shared_bias_prior_std_m": 0.1,
        "camera_bias_prior_std_m": 0.1,
        "effective_samples_per_view": 10.0,
        "maximum_iterations": 6,
    }
    values.update(overrides)
    return BiasAwareStateUpdateConfigV2(**values)  # type: ignore[arg-type]


def test_frozen_v1_implementation_bytes_are_unchanged() -> None:
    source = (
        Path(__file__).parents[1]
        / "src"
        / "bayesian_phystwin"
        / "bias_aware_belief.py"
    )

    assert _git_blob_sha1(source) == FROZEN_V1_GIT_BLOB_SHA1


def test_v2_matches_v1_on_well_conditioned_fixture() -> None:
    innovation, available, state_basis, bias_basis = _centered_problem()
    reliability = np.linspace(0.5, 1.0, state_basis.shape[0])[None]
    reliability = np.repeat(reliability, innovation.shape[0], axis=0)

    legacy = update_bias_aware_state(
        innovation,
        available,
        state_basis,
        bias_basis,
        prior_reliability=reliability,
        config=_legacy_config(),
    )
    prospective = update_bias_aware_state_v2(
        innovation,
        available,
        state_basis,
        bias_basis,
        prior_reliability=reliability,
        config=_v2_config(),
    )

    assert legacy.accepted and prospective.accepted
    assert legacy.reason == prospective.reason
    np.testing.assert_allclose(
        prospective.state_coefficients_m,
        legacy.state_coefficients_m,
        atol=1e-11,
        rtol=1e-10,
    )
    np.testing.assert_allclose(
        prospective.shared_bias_coefficients_m,
        legacy.shared_bias_coefficients_m,
        atol=1e-11,
        rtol=1e-10,
    )
    np.testing.assert_allclose(
        prospective.camera_biases_m,
        legacy.camera_biases_m,
        atol=1e-11,
        rtol=1e-10,
    )
    np.testing.assert_allclose(
        prospective.posterior_covariance_m2,
        legacy.posterior_covariance_m2,
        atol=1e-13,
        rtol=1e-10,
    )
    np.testing.assert_allclose(
        prospective.robust_weights,
        legacy.robust_weights,
        atol=1e-12,
        rtol=1e-10,
    )
    np.testing.assert_array_equal(
        prospective.prior_reliability,
        legacy.prior_reliability,
    )
    assert prospective.implementation_version == BIAS_AWARE_BELIEF_V2_VERSION
    assert prospective.implementation_id == BIAS_AWARE_BELIEF_V2_IMPLEMENTATION
    assert prospective.diagnostics["numerical_backend_schema"] == SPD_SYSTEM_SCHEMA
    assert prospective.diagnostics["implicit_jitter"] is False
    assert prospective.diagnostics["pseudoinverse_fallback"] is False
    assert prospective.diagnostics["final_solve_relative_residual"] < 1e-10
    assert prospective.diagnostics["inverse_relative_residual"] < 1e-9


def test_v2_preserves_exact_fallback_for_unanchored_common_mode() -> None:
    view_count = 3
    point_count = 8
    innovation = np.zeros((view_count, point_count, 3), dtype=np.float64)
    innovation[..., 0] = 0.03
    available = np.ones((view_count, point_count), dtype=bool)
    state_basis = np.ones((point_count, 1), dtype=np.float64)
    bias_basis = np.ones((point_count, 1), dtype=np.float64)

    result = update_bias_aware_state_v2(
        innovation,
        available,
        state_basis,
        bias_basis,
    )

    assert not result.accepted
    assert result.reason == "unanchored-common-mode-ambiguity"
    for value in (
        result.state_coefficients_m,
        result.shared_bias_coefficients_m,
        result.camera_biases_m,
        result.posterior_covariance_m2,
        result.robust_weights,
    ):
        assert value.tobytes() == np.zeros_like(value).tobytes()


def test_v2_is_invariant_to_point_row_permutation() -> None:
    innovation, available, state_basis, bias_basis = _centered_problem()
    baseline = update_bias_aware_state_v2(
        innovation,
        available,
        state_basis,
        bias_basis,
        config=_v2_config(),
    )
    permutation = np.asarray([7, 1, 10, 3, 0, 8, 5, 11, 2, 9, 6, 4])
    permuted = update_bias_aware_state_v2(
        innovation[:, permutation],
        available[:, permutation],
        state_basis[permutation],
        bias_basis[permutation],
        config=_v2_config(),
    )

    assert baseline.accepted and permuted.accepted
    np.testing.assert_allclose(
        permuted.state_coefficients_m,
        baseline.state_coefficients_m,
        atol=1e-12,
        rtol=1e-11,
    )
    np.testing.assert_allclose(
        permuted.shared_bias_coefficients_m,
        baseline.shared_bias_coefficients_m,
        atol=1e-12,
        rtol=1e-11,
    )
    np.testing.assert_allclose(
        permuted.camera_biases_m,
        baseline.camera_biases_m,
        atol=1e-12,
        rtol=1e-11,
    )
    np.testing.assert_allclose(
        permuted.posterior_covariance_m2,
        baseline.posterior_covariance_m2,
        atol=1e-13,
        rtol=1e-11,
    )
    inverse = np.argsort(permutation)
    np.testing.assert_allclose(
        permuted.robust_weights[:, inverse],
        baseline.robust_weights,
        atol=1e-12,
        rtol=1e-11,
    )


def test_v2_is_invariant_to_coordinate_permutation() -> None:
    innovation, available, state_basis, bias_basis = _centered_problem()
    baseline = update_bias_aware_state_v2(
        innovation,
        available,
        state_basis,
        bias_basis,
        config=_v2_config(),
    )
    coordinate_permutation = np.asarray([2, 0, 1])
    permuted = update_bias_aware_state_v2(
        innovation[..., coordinate_permutation],
        available,
        state_basis,
        bias_basis,
        config=_v2_config(),
    )

    assert baseline.accepted and permuted.accepted
    np.testing.assert_allclose(
        permuted.state_coefficients_m,
        baseline.state_coefficients_m[:, coordinate_permutation],
        atol=1e-12,
        rtol=1e-11,
    )
    np.testing.assert_allclose(
        permuted.shared_bias_coefficients_m,
        baseline.shared_bias_coefficients_m[:, coordinate_permutation],
        atol=1e-12,
        rtol=1e-11,
    )
    np.testing.assert_allclose(
        permuted.camera_biases_m,
        baseline.camera_biases_m[:, coordinate_permutation],
        atol=1e-12,
        rtol=1e-11,
    )
    np.testing.assert_allclose(
        permuted.posterior_covariance_m2,
        baseline.posterior_covariance_m2,
        atol=1e-13,
        rtol=1e-11,
    )
    np.testing.assert_allclose(
        permuted.robust_weights,
        baseline.robust_weights,
        atol=1e-12,
        rtol=1e-11,
    )


def test_v2_fails_closed_on_singular_state_prior_covariance() -> None:
    point_count = 12
    coordinate = np.linspace(-1.0, 1.0, point_count)
    state_basis = np.column_stack(
        (coordinate, np.square(coordinate) - np.mean(np.square(coordinate)))
    )
    innovation = np.zeros((2, point_count, 3), dtype=np.float64)
    innovation[..., 0] = 0.01 * state_basis[:, 0]
    singular_prior = np.asarray([[1.0, 1.0], [1.0, 1.0]], dtype=np.float64)

    result = update_bias_aware_state_v2(
        innovation,
        np.ones((2, point_count), dtype=bool),
        state_basis,
        np.zeros((point_count, 0), dtype=np.float64),
        state_prior_covariance_m2=singular_prior,
        config=_v2_config(),
    )

    assert not result.accepted
    assert result.reason == "invalid-state-prior-covariance"
    assert result.diagnostics["numerical_failure_type"] == "SPDValidationError"
    assert result.state_coefficients_m.tobytes() == np.zeros((2, 3)).tobytes()


def test_v2_rechecks_the_final_irls_system_and_falls_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    innovation, available, state_basis, bias_basis = _centered_problem()
    original = belief_v2._factor_system

    def fail_final(
        matrix: np.ndarray,
        *,
        name: str,
        config: BiasAwareStateUpdateConfigV2,
    ) -> object:
        if name == "final posterior normal":
            raise SPDConditionError("deliberate final-system failure")
        return original(matrix, name=name, config=config)

    monkeypatch.setattr(belief_v2, "_factor_system", fail_final)
    result = update_bias_aware_state_v2(
        innovation,
        available,
        state_basis,
        bias_basis,
        config=_v2_config(maximum_iterations=1),
    )

    assert not result.accepted
    assert result.reason == "ill-conditioned-posterior"
    assert result.diagnostics["numerical_failure_type"] == "SPDConditionError"
    assert result.state_coefficients_m.tobytes() == np.zeros((1, 3)).tobytes()
