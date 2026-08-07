from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from bayesian_phystwin.invariant_query_subspace import (
    INVARIANT_QUERY_SUBSPACE_SCHEMA,
    INVARIANT_QUERY_SUBSPACE_VERSION,
    InvariantQuerySubspaceConfigV1,
    InvariantQuerySubspaceResultV1,
    select_invariant_query_subspace,
)


def _query_x() -> np.ndarray:
    query = np.zeros((1, 3, 2), dtype=np.float64)
    query[0, 0, 0] = 1.0
    return query


def _config(**changes: object) -> InvariantQuerySubspaceConfigV1:
    values: dict[str, object] = {
        "minimum_information_fraction": 0.0,
        "minimum_identifiable_fraction": 0.1,
        "minimum_query_sensitivity_fraction": 0.8,
    }
    values.update(changes)
    return InvariantQuerySubspaceConfigV1(**values)


def _result(
    *,
    known: np.ndarray | None = None,
    conditional: np.ndarray | None = None,
    prior: np.ndarray | None = None,
    query: np.ndarray | None = None,
    config: InvariantQuerySubspaceConfigV1 | None = None,
) -> InvariantQuerySubspaceResultV1:
    return select_invariant_query_subspace(
        np.eye(2) if known is None else known,
        np.eye(2) if conditional is None else conditional,
        np.eye(2) if prior is None else prior,
        _query_x() if query is None else query,
        config=_config() if config is None else config,
    )


def test_repeated_information_eigenspace_is_rotation_invariant() -> None:
    original = _result()
    angle = np.pi / 4.0
    transform = np.asarray(
        [
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)],
        ]
    )
    rotated = _result(
        known=transform.T @ transform,
        conditional=transform.T @ transform,
        prior=transform.T @ transform,
        query=np.einsum("qcs,sk->qck", _query_x(), transform),
    )

    assert original.admissible
    assert rotated.admissible
    assert original.retained_rank == rotated.retained_rank == 1
    np.testing.assert_allclose(
        transform @ rotated.state_mapping @ rotated.state_mapping.T @ transform.T,
        original.state_mapping @ original.state_mapping.T,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        transform @ rotated.lift_state_mean(np.asarray([0.3])),
        original.lift_state_mean(np.asarray([0.3])),
        atol=1e-12,
    )
    assert (
        original.diagnostics()["individual_information_eigenvectors_thresholded"]
        is False
    )


def test_nonorthogonal_state_reparameterization_preserves_physical_span() -> None:
    original = _result()
    transform = np.asarray([[2.0, 0.3], [-0.4, 0.7]])
    inverse = np.linalg.inv(transform)
    rotated = _result(
        known=transform.T @ transform,
        conditional=transform.T @ transform,
        prior=inverse @ inverse.T,
        query=np.einsum("qcs,sk->qck", _query_x(), transform),
    )

    physical_mapping = transform @ rotated.state_mapping
    original_basis, _ = np.linalg.qr(original.state_mapping)
    physical_basis, _ = np.linalg.qr(physical_mapping)
    np.testing.assert_allclose(
        physical_basis @ physical_basis.T,
        original_basis @ original_basis.T,
        atol=1.0e-12,
    )


def test_information_projector_respects_relative_threshold() -> None:
    result = _result(
        known=np.diag([10.0, 1.0]),
        conditional=np.diag([10.0, 1.0]),
        config=_config(
            minimum_information_fraction=0.5,
            minimum_query_sensitivity_fraction=0.0,
        ),
    )

    assert result.retained_rank == 1
    np.testing.assert_allclose(
        result.state_mapping @ result.state_mapping.T,
        np.diag([1.0, 0.0]),
    )
    assert result.diagnostics()["information_rank"] == 1


def test_identifiability_projector_uses_generalized_ratio() -> None:
    query = np.zeros((2, 3, 2))
    query[0, 0, 0] = 1.0
    query[1, 1, 1] = 1.0
    result = _result(
        known=np.diag([1.0, 10.0]),
        conditional=np.diag([0.9, 0.5]),
        query=query,
        config=_config(
            minimum_identifiable_fraction=0.5,
            minimum_query_sensitivity_fraction=0.0,
        ),
    )

    assert result.retained_rank == 1
    np.testing.assert_allclose(
        result.state_mapping @ result.state_mapping.T,
        np.diag([1.0, 0.0]),
    )
    assert result.identifiable_fractions[0] == pytest.approx(0.9)


def test_query_projector_can_keep_multiple_query_directions() -> None:
    query = np.zeros((2, 3, 2))
    query[0, 0, 0] = 1.0
    query[1, 1, 1] = 0.9
    result = _result(
        query=query,
        config=_config(minimum_query_sensitivity_fraction=0.8),
    )

    assert result.retained_rank == 2
    np.testing.assert_allclose(
        np.sort(result.query_sensitivity_fractions),
        [0.9, 1.0],
    )
    assert result.diagnostics()["query_rank"] == 2


def test_zero_query_is_rejected_when_positive_query_fraction_is_required() -> None:
    result = _result(query=np.zeros((1, 3, 2)))

    assert not result.admissible
    assert result.reason == "no-query-support"
    assert result.retained_rank == 0


def test_zero_query_is_retained_when_query_filter_is_disabled() -> None:
    result = _result(
        query=np.zeros((1, 3, 2)),
        config=_config(minimum_query_sensitivity_fraction=0.0),
    )

    assert result.admissible
    assert result.retained_rank == 2
    np.testing.assert_array_equal(result.query_sensitivity_fractions, 0.0)


def test_no_information_and_no_identifiable_support_return_empty_results() -> None:
    no_information = _result(
        known=np.zeros((2, 2)),
        conditional=np.zeros((2, 2)),
    )
    no_identifiability = _result(
        known=np.eye(2),
        conditional=np.eye(2) * 0.01,
        config=_config(minimum_identifiable_fraction=0.5),
    )

    assert no_information.reason == "no-information-support"
    assert no_identifiability.reason == "no-identifiable-support"
    assert not no_information.admissible
    assert not no_identifiability.admissible


def test_near_repeated_spectrum_uses_one_stable_query_projector() -> None:
    conditional = np.diag([1.0, 1.0 + 2.0e-10])
    original = _result(known=conditional, conditional=conditional)
    angle = 0.37
    transform = np.asarray(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )
    rotated = _result(
        known=transform.T @ conditional @ transform,
        conditional=transform.T @ conditional @ transform,
        query=np.einsum("qcs,sk->qck", _query_x(), transform),
    )

    np.testing.assert_allclose(
        transform @ rotated.state_mapping @ rotated.state_mapping.T @ transform.T,
        original.state_mapping @ original.state_mapping.T,
        atol=2e-9,
    )


def test_projection_and_lifting_preserve_prior_outside_subspace() -> None:
    result = _result(
        known=np.diag([10.0, 1.0]),
        conditional=np.diag([10.0, 1.0]),
        config=_config(minimum_information_fraction=0.5),
    )
    design = np.asarray([[[2.0, 3.0]]])

    projected = result.project_state_jacobian(design)
    lifted = result.lift_state_covariance(np.asarray([[0.25]]), np.eye(2))

    np.testing.assert_allclose(projected, [[[2.0]]])
    np.testing.assert_allclose(lifted, np.diag([0.25, 1.0]))
    assert not result.state_mapping.flags.writeable
    assert not result.information_projector.flags.writeable


def test_result_diagnostics_and_constants_are_stable() -> None:
    result = _result()
    diagnostics = result.diagnostics()

    assert diagnostics["schema"] == INVARIANT_QUERY_SUBSPACE_SCHEMA
    assert diagnostics["schema_version"] == INVARIANT_QUERY_SUBSPACE_VERSION
    assert diagnostics["retained_rank"] == 1
    assert diagnostics["repeated_eigenspace_projectors_used"] is True


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        (
            {"minimum_information_fraction": -1.0},
            "minimum_information_fraction",
        ),
        (
            {"minimum_identifiable_fraction": 0.0},
            "minimum_identifiable_fraction",
        ),
        (
            {"minimum_query_sensitivity_fraction": 2.0},
            "minimum_query_sensitivity_fraction",
        ),
        ({"eigenvalue_floor": 0.0}, "eigenvalue_floor"),
        (
            {"relative_spectral_tolerance": -1.0},
            "relative_spectral_tolerance",
        ),
        (
            {
                "relative_spectral_tolerance": 0.0,
                "absolute_spectral_tolerance": 0.0,
            },
            "at least one spectral tolerance",
        ),
    ],
)
def test_config_rejects_invalid_controls(
    changes: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        replace(InvariantQuerySubspaceConfigV1(), **changes)


@pytest.mark.parametrize("value", [True, np.bool_(False), "0.1", np.inf])
def test_config_rejects_nonfinite_or_nonreal_tolerance(value: object) -> None:
    with pytest.raises(ValueError, match="relative_spectral_tolerance"):
        InvariantQuerySubspaceConfigV1(relative_spectral_tolerance=value)


def test_selector_rejects_invalid_config_and_matrix_contracts() -> None:
    with pytest.raises(TypeError, match="config"):
        select_invariant_query_subspace(
            np.eye(2),
            np.eye(2),
            np.eye(2),
            _query_x(),
            config=object(),
        )
    with pytest.raises(ValueError, match="square"):
        _result(known=np.ones((2, 3)))
    with pytest.raises(ValueError, match="symmetric"):
        _result(known=np.asarray([[1.0, 2.0], [0.0, 1.0]]))
    with pytest.raises(ValueError, match="finite"):
        _result(conditional=np.asarray([[1.0, np.nan], [np.nan, 1.0]]))
    with pytest.raises(ValueError, match="same shape"):
        select_invariant_query_subspace(
            np.eye(2),
            np.eye(3),
            np.eye(2),
            _query_x(),
        )
    with pytest.raises(ValueError, match="shape"):
        _result(query=np.zeros((1, 2, 2)))
    with pytest.raises(ValueError, match="nonempty and finite"):
        _result(query=np.zeros((0, 3, 2)))


def test_selector_rejects_non_psd_information_and_prior() -> None:
    with pytest.raises(ValueError, match="positive semidefinite"):
        _result(conditional=np.diag([1.0, -1.0]))
    with pytest.raises(ValueError, match="positive semidefinite"):
        _result(prior=np.diag([1.0, -1.0]))
    with pytest.raises(ValueError, match="positive definite"):
        _result(known=np.diag([0.0, 1.0]), conditional=np.eye(2))


def test_result_and_helper_methods_reject_malformed_inputs() -> None:
    result = _result()
    with pytest.raises(ValueError, match="final axis"):
        result.project_state_jacobian(np.ones((2, 3)))
    with pytest.raises(ValueError, match="finite"):
        result.project_state_jacobian(np.asarray([[np.nan, 0.0]]))
    with pytest.raises(ValueError, match="retained-state vector"):
        result.lift_state_mean(np.zeros(2))
    with pytest.raises(ValueError, match="reduced_covariance shape"):
        result.lift_state_covariance(np.eye(2), np.eye(2))
    with pytest.raises(ValueError, match="state_prior_covariance shape"):
        result.lift_state_covariance(np.eye(1), np.eye(3))

    kwargs = dict(
        state_mapping=np.ones((2, 1)),
        identifiable_fractions=np.ones(1),
        query_sensitivity_fractions=np.ones(1),
        information_projector=np.eye(2),
        identifiability_projector=np.eye(2),
        query_projector=np.eye(2),
        information_eigenvalues=np.ones(2),
        identifiability_eigenvalues=np.ones(2),
        query_eigenvalues=np.ones(1),
        maximum_information=1.0,
        maximum_query_sensitivity=1.0,
        reason="admissible",
    )
    with pytest.raises(ValueError, match="fractions"):
        InvariantQuerySubspaceResultV1(
            **{**kwargs, "identifiable_fractions": np.ones(2)}
        )
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        InvariantQuerySubspaceResultV1(
            **{**kwargs, "query_sensitivity_fractions": np.asarray([2.0])}
        )
    with pytest.raises(ValueError, match="reason"):
        InvariantQuerySubspaceResultV1(**{**kwargs, "reason": 3})
    with pytest.raises(ValueError, match="inconsistent"):
        InvariantQuerySubspaceResultV1(**{**kwargs, "reason": "no-query-support"})
    with pytest.raises(ValueError, match="idempotent"):
        InvariantQuerySubspaceResultV1(
            **{
                **kwargs,
                "information_projector": np.asarray([[1.0, 0.0], [0.0, 0.5]]),
            }
        )
