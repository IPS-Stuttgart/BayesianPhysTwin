from __future__ import annotations

from typing import Any

import numpy as np
import pytest

import bayesian_phystwin._phystwin_directional_endpoint_v2_solver as solver
import bayesian_phystwin.phystwin_directional_endpoint_v2 as endpoint_v2
from bayesian_phystwin.phystwin_directional_endpoint import (
    robust_directional_endpoint,
)
from bayesian_phystwin.phystwin_directional_endpoint_v2 import (
    PHYSTWIN_DIRECTIONAL_ENDPOINT_VERSION,
    DirectionalEndpointConfigV2,
    DirectionalEndpointNumericalError,
    robust_directional_endpoint_v2,
)


def _tangent_projectors(point_count: int) -> np.ndarray:
    projector = np.diag([1.0, 1.0, 0.0])
    return np.repeat(projector[None], point_count, axis=0)


def _run_v1(
    source: np.ndarray,
    source_valid: np.ndarray,
    multiview: np.ndarray,
    multiview_valid: np.ndarray,
    priority: np.ndarray,
):
    return robust_directional_endpoint(
        source,
        source_valid,
        multiview,
        multiview_valid,
        _tangent_projectors(source.shape[1]),
        priority,
        end_frame=len(source),
        process_variance=0.0,
        observation_variance=1e-4,
        initial_variance=1e-3,
        inlier_prior=0.95,
        outlier_variance_multiplier=100.0,
    )


def _run_v2(
    source: np.ndarray,
    source_valid: np.ndarray,
    multiview: np.ndarray,
    multiview_valid: np.ndarray,
    priority: np.ndarray,
    *,
    projectors: np.ndarray | None = None,
    config: DirectionalEndpointConfigV2 | None = None,
    observation_variance: float = 1e-4,
    initial_variance: float = 1e-3,
):
    return robust_directional_endpoint_v2(
        source,
        source_valid,
        multiview,
        multiview_valid,
        (_tangent_projectors(source.shape[1]) if projectors is None else projectors),
        priority,
        end_frame=len(source),
        process_variance=0.0,
        observation_variance=observation_variance,
        initial_variance=initial_variance,
        inlier_prior=0.95,
        outlier_variance_multiplier=100.0,
        config=config,
    )


def _raw_inputs() -> dict[str, object]:
    source = np.zeros((1, 1, 3), dtype=np.float64)
    valid = np.ones((1, 1), dtype=bool)
    return {
        "source_residual": source,
        "source_valid": valid,
        "multiview_residual": np.zeros_like(source),
        "multiview_valid": valid.copy(),
        "tangent_projectors": _tangent_projectors(1),
        "priority_identities": np.asarray([False]),
        "end_frame": 1,
        "process_variance": 0.0,
        "observation_variance": 1e-4,
        "initial_variance": 1e-3,
        "inlier_prior": 0.95,
        "outlier_variance_multiplier": 100.0,
    }


def _call_raw(**changes: object) -> endpoint_v2.DirectionalEndpointPosteriorV2:
    inputs = _raw_inputs()
    inputs.update(changes)
    return endpoint_v2.robust_directional_endpoint_v2(**inputs)  # type: ignore[arg-type]


def _solver_inputs() -> dict[str, Any]:
    return {
        "mean": np.zeros((1, 3), dtype=np.float64),
        "covariance": np.eye(3, dtype=np.float64)[None],
        "observation": np.zeros((1, 3), dtype=np.float64),
        "observation_matrix": np.eye(3, dtype=np.float64)[None],
        "observation_variance": 1e-4,
        "inlier_prior": 0.95,
        "outlier_variance_multiplier": 100.0,
        "name": "adversarial",
        "config": DirectionalEndpointConfigV2(),
    }


def test_v2_retains_legacy_mean_but_not_anti_conservative_isotropization() -> None:
    source = np.array([[[0.1, 0.0, 0.0]]])
    valid = np.ones((1, 1), dtype=bool)
    multiview = np.zeros_like(source)
    priority = np.array([False])

    legacy = _run_v1(
        source,
        valid,
        multiview,
        np.zeros_like(valid),
        priority,
    )
    prospective = _run_v2(
        source,
        valid,
        multiview,
        np.zeros_like(valid),
        priority,
    )

    np.testing.assert_allclose(prospective.mean, legacy.mean, atol=1e-14)
    np.testing.assert_allclose(
        prospective.final_inlier_probability,
        legacy.final_inlier_probability,
        atol=1e-14,
    )
    assert prospective.variance[0] > legacy.variance[0]
    assert prospective.variance[0] == pytest.approx(
        np.max(np.linalg.eigvalsh(prospective.covariance[0]))
    )
    assert legacy.variance[0] == pytest.approx(
        np.trace(prospective.covariance[0]) / 3.0
    )
    scalar_upper_bound = prospective.variance[0] * np.eye(3) - prospective.covariance[0]
    assert np.min(np.linalg.eigvalsh(scalar_upper_bound)) >= -1e-15


def test_v2_covariance_is_symmetric_positive_definite_after_mixed_updates() -> None:
    source = np.array(
        [
            [[0.0, 0.0, 0.02]],
            [[0.0, 0.0, 0.03]],
            [[0.0, 0.0, 0.04]],
        ]
    )
    multiview = np.array(
        [
            [[0.01, -0.02, 0.0]],
            [[0.02, -0.01, 0.0]],
            [[0.03, 0.00, 0.0]],
        ]
    )
    valid = np.ones((3, 1), dtype=bool)

    result = _run_v2(
        source,
        valid,
        multiview,
        valid,
        np.array([True]),
    )

    np.testing.assert_allclose(
        result.covariance,
        np.swapaxes(result.covariance, 1, 2),
        atol=0.0,
    )
    assert np.all(np.linalg.eigvalsh(result.covariance) > 0.0)
    np.testing.assert_array_equal(result.source_update_count, [3])
    np.testing.assert_array_equal(result.tangent_update_count, [3])
    assert result.maximum_innovation_condition_number[0] >= 1.0
    assert result.maximum_posterior_condition_number[0] >= 1.0


def test_v2_unobserved_identity_retains_the_declared_inlier_prior() -> None:
    source = np.zeros((2, 1, 3), dtype=np.float64)
    invalid = np.zeros((2, 1), dtype=bool)

    result = _run_v2(
        source,
        invalid,
        np.zeros_like(source),
        invalid,
        np.array([True]),
    )

    np.testing.assert_allclose(result.final_inlier_probability, [0.95])
    np.testing.assert_array_equal(result.update_count, [0])
    np.testing.assert_array_equal(result.source_update_count, [0])
    np.testing.assert_array_equal(result.tangent_update_count, [0])
    np.testing.assert_allclose(result.mean, np.zeros((1, 3)))
    np.testing.assert_allclose(result.covariance, 1e-3 * np.eye(3)[None])


def test_v2_is_invariant_to_an_orthogonal_coordinate_change() -> None:
    source = np.array(
        [
            [[0.01, -0.02, 0.03]],
            [[0.02, -0.01, 0.04]],
        ]
    )
    multiview = np.array(
        [
            [[0.03, -0.01, 0.0]],
            [[0.04, 0.01, 0.0]],
        ]
    )
    valid = np.ones((2, 1), dtype=bool)
    priority = np.array([True])
    projectors = _tangent_projectors(1)
    axis = np.asarray([1.0, 2.0, 3.0])
    axis /= np.linalg.norm(axis)
    angle = 0.7
    cross = np.array(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ]
    )
    rotation = (
        np.cos(angle) * np.eye(3)
        + (1.0 - np.cos(angle)) * np.outer(axis, axis)
        + np.sin(angle) * cross
    )

    reference = _run_v2(
        source,
        valid,
        multiview,
        valid,
        priority,
        projectors=projectors,
    )
    rotated = _run_v2(
        source @ rotation.T,
        valid,
        multiview @ rotation.T,
        valid,
        priority,
        projectors=np.einsum(
            "ij,njk,lk->nil",
            rotation,
            projectors,
            rotation,
        ),
    )
    restored_mean = rotated.mean @ rotation
    restored_covariance = np.einsum(
        "ij,njk,kl->nil",
        rotation.T,
        rotated.covariance,
        rotation,
    )

    np.testing.assert_allclose(restored_mean, reference.mean, atol=2e-14)
    np.testing.assert_allclose(
        restored_covariance,
        reference.covariance,
        atol=2e-14,
    )
    np.testing.assert_allclose(rotated.variance, reference.variance, atol=2e-14)


def test_v2_rejects_a_nearly_idempotent_projector_outside_absolute_tolerance() -> None:
    source = np.zeros((1, 1, 3), dtype=np.float64)
    valid = np.ones((1, 1), dtype=bool)
    near_projector = np.diag([1.0 + 5e-6, 1.0, 0.0])[None]

    with pytest.raises(ValueError, match="idempotent"):
        _run_v2(
            source,
            valid,
            np.zeros_like(source),
            np.zeros_like(valid),
            np.array([True]),
            projectors=near_projector,
        )


def test_v2_fails_closed_when_the_posterior_exceeds_condition_limit() -> None:
    source = np.array([[[0.0, 0.0, 0.01]]])
    multiview = np.zeros_like(source)
    valid = np.ones((1, 1), dtype=bool)

    with pytest.raises(DirectionalEndpointNumericalError, match="SPD admission"):
        _run_v2(
            source,
            valid,
            multiview,
            np.zeros_like(valid),
            np.array([True]),
            config=DirectionalEndpointConfigV2(
                maximum_condition_number=1e8,
            ),
            observation_variance=1e-16,
            initial_variance=1.0,
        )


def test_v2_results_are_immutable_and_report_numerical_semantics() -> None:
    source = np.zeros((1, 1, 3))
    valid = np.ones((1, 1), dtype=bool)
    result = _run_v2(
        source,
        valid,
        np.zeros_like(source),
        np.zeros_like(valid),
        np.array([False]),
    )

    for array in (
        result.mean,
        result.covariance,
        result.variance,
        result.final_inlier_probability,
        result.update_count,
        result.source_update_count,
        result.tangent_update_count,
        result.maximum_innovation_condition_number,
        result.maximum_posterior_condition_number,
    ):
        assert not array.flags.writeable
    with pytest.raises(ValueError):
        result.mean[0, 0] = 1.0

    diagnostics = result.diagnostics()
    assert diagnostics["schema_version"] == PHYSTWIN_DIRECTIONAL_ENDPOINT_VERSION
    assert diagnostics["component_covariance_update"] == "joseph-form"
    assert diagnostics["mixture_covariance_update"] == "exact-moment-matching"
    assert diagnostics["full_source_covariance_retained"] is True
    assert diagnostics["trace_average_isotropization"] is False
    assert diagnostics["implicit_jitter"] is False
    assert diagnostics["eigenvalue_clipping"] is False
    assert diagnostics["pseudoinverse_fallback"] is False


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"source_residual": np.zeros((1, 3))}, "source residual"),
        ({"multiview_residual": np.zeros((2, 1, 3))}, "multiview residual"),
        ({"source_valid": np.zeros((1, 2), dtype=bool)}, "source validity"),
        ({"multiview_valid": np.zeros((1, 2), dtype=bool)}, "multiview validity"),
        ({"tangent_projectors": np.zeros((2, 3, 3))}, "tangent projectors"),
        ({"priority_identities": np.zeros(2, dtype=bool)}, "priority identities"),
        (
            {"source_residual": np.full((1, 1, 3), np.nan)},
            "valid source residuals",
        ),
        (
            {"multiview_residual": np.full((1, 1, 3), np.nan)},
            "valid multiview residuals",
        ),
        (
            {"tangent_projectors": np.full((1, 3, 3), np.nan)},
            "tangent projectors must be finite",
        ),
        (
            {
                "tangent_projectors": np.asarray(
                    [[[1.0, 1.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]]
                )
            },
            "symmetric",
        ),
    ],
)
def test_v2_rejects_malformed_inputs(
    changes: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _call_raw(**changes)


def test_v2_rejects_invalid_projector_spectra() -> None:
    with pytest.raises(ValueError, match="one null direction"):
        _call_raw(tangent_projectors=np.eye(3, dtype=np.float64)[None])
    with pytest.raises(ValueError, match="rank two"):
        _call_raw(tangent_projectors=np.zeros((1, 3, 3), dtype=np.float64))


@pytest.mark.parametrize(
    ("changes", "error", "message"),
    [
        ({"maximum_condition_number": True}, TypeError, "real scalar"),
        ({"maximum_condition_number": 0.5}, ValueError, "at least one"),
        ({"symmetry_absolute_tolerance": -1.0}, ValueError, "nonnegative"),
        ({"symmetry_relative_tolerance": np.inf}, ValueError, "nonnegative"),
        ({"solve_residual_tolerance": 0.0}, ValueError, "positive"),
    ],
)
def test_v2_config_rejects_invalid_numerical_policy(
    changes: dict[str, object],
    error: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error, match=message):
        DirectionalEndpointConfigV2(**changes)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("changes", "error", "message"),
    [
        ({"end_frame": True}, TypeError, "integer"),
        ({"end_frame": 0}, ValueError, "inside"),
        ({"process_variance": -1.0}, ValueError, "nonnegative"),
        ({"observation_variance": 0.0}, ValueError, "positive"),
        ({"initial_variance": 0.0}, ValueError, "positive"),
        ({"inlier_prior": 1.0}, ValueError, "lie in"),
        ({"outlier_variance_multiplier": 1.0}, ValueError, "exceed"),
        (
            {
                "observation_variance": np.finfo(np.float64).max,
                "outlier_variance_multiplier": 2.0,
            },
            ValueError,
            "remain finite",
        ),
    ],
)
def test_filter_parameter_contract_rejects_invalid_values(
    changes: dict[str, object],
    error: type[Exception],
    message: str,
) -> None:
    values: dict[str, object] = {
        "end_frame": 1,
        "frame_count": 1,
        "process_variance": 0.0,
        "observation_variance": 1e-4,
        "initial_variance": 1e-3,
        "inlier_prior": 0.95,
        "outlier_variance_multiplier": 100.0,
    }
    values.update(changes)
    with np.errstate(over="ignore"):
        with pytest.raises(error, match=message):
            solver.validate_filter_parameters(**values)  # type: ignore[arg-type]


def test_component_update_rejects_nonfinite_mean() -> None:
    config = DirectionalEndpointConfigV2()
    prior = solver.admit_spd_system(
        np.eye(3),
        name="prior",
        config=config,
    )
    with pytest.raises(solver.SPDSolveError, match="non-finite"):
        solver._component_update(
            mean=np.zeros(3),
            prior=prior,
            innovation=np.asarray([np.inf]),
            projected_covariance=np.asarray([[1.0]]),
            observation_matrix=np.asarray([[1.0, 0.0, 0.0]]),
            observation_variance=1.0,
            name="component",
            config=config,
        )


def test_robust_update_rejects_overflow_and_nonfinite_innovation() -> None:
    values = _solver_inputs()
    values["observation_variance"] = np.finfo(np.float64).max
    values["outlier_variance_multiplier"] = 2.0
    with np.errstate(over="ignore"):
        with pytest.raises(DirectionalEndpointNumericalError, match="overflowed"):
            solver.robust_linear_update_v2(**values)

    values = _solver_inputs()
    values["observation"] = np.asarray([[np.inf, 0.0, 0.0]])
    with pytest.raises(DirectionalEndpointNumericalError, match="failed SPD admission"):
        solver.robust_linear_update_v2(**values)


def test_robust_update_rejects_nonfinite_mixture_probability(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(solver.np, "logaddexp", lambda _left, _right: np.nan)
    with pytest.raises(
        DirectionalEndpointNumericalError,
        match="non-finite mixture probability",
    ):
        solver.robust_linear_update_v2(**_solver_inputs())


def test_robust_update_rejects_nonfinite_mixture_mean(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def nonfinite_component(**_kwargs: object) -> tuple[object, ...]:
        return np.full(3, np.inf), np.eye(3), 0.0, 0.0, 1.0, 1.0

    monkeypatch.setattr(solver, "_component_update", nonfinite_component)
    with pytest.raises(
        DirectionalEndpointNumericalError,
        match="non-finite mixture mean",
    ):
        solver.robust_linear_update_v2(**_solver_inputs())


def test_robust_update_rejects_singular_mixture_covariance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    call_count = 0

    def singular_component(**_kwargs: object) -> tuple[object, ...]:
        nonlocal call_count
        mean = np.asarray([float(call_count), 0.0, 0.0])
        call_count += 1
        return mean, np.zeros((3, 3)), 0.0, 0.0, 1.0, 1.0

    monkeypatch.setattr(solver, "_component_update", singular_component)
    with pytest.raises(
        DirectionalEndpointNumericalError,
        match="mixture failed SPD admission",
    ):
        solver.robust_linear_update_v2(**_solver_inputs())


def test_v2_fails_closed_on_process_overflow() -> None:
    maximum = np.finfo(np.float64).max
    with np.errstate(over="ignore"):
        with pytest.raises(
            DirectionalEndpointNumericalError,
            match="process covariance overflowed",
        ):
            _call_raw(
                source_valid=np.zeros((1, 1), dtype=bool),
                multiview_valid=np.zeros((1, 1), dtype=bool),
                process_variance=maximum,
                initial_variance=maximum,
            )


def test_v2_fails_closed_when_final_covariance_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def reject_final(*_args: object, **_kwargs: object) -> object:
        raise endpoint_v2.SPDSystemError("forced final rejection")

    monkeypatch.setattr(endpoint_v2, "admit_spd_system", reject_final)
    with pytest.raises(
        DirectionalEndpointNumericalError,
        match="final point 0 covariance failed SPD admission",
    ):
        _call_raw(
            source_valid=np.zeros((1, 1), dtype=bool),
            multiview_valid=np.zeros((1, 1), dtype=bool),
        )
