from __future__ import annotations

import numpy as np
import pytest

from bayesian_phystwin.spd_system import (
    SPDConditionError,
    SPDSolveError,
    SPDSystem,
    SPDValidationError,
    SPD_SYSTEM_SCHEMA,
    SPD_SYSTEM_VERSION,
)


def test_spd_system_solves_whitens_and_reconstructs_inverse() -> None:
    matrix = np.asarray([[4.0, 1.0], [1.0, 3.0]], dtype=np.float64)
    right = np.asarray([1.0, 2.0], dtype=np.float64)

    system = SPDSystem.from_matrix(matrix, name="well-conditioned fixture")
    solution = system.solve(right)
    inverse = system.reconstruct_inverse()

    np.testing.assert_allclose(matrix @ solution, right, atol=1e-14, rtol=1e-14)
    np.testing.assert_allclose(
        inverse,
        np.linalg.solve(matrix, np.eye(2)),
        atol=1e-14,
        rtol=1e-14,
    )
    assert system.quadratic_form(right) == pytest.approx(
        float(right @ np.linalg.solve(matrix, right)),
        rel=1e-14,
    )
    assert system.log_determinant == pytest.approx(
        float(np.linalg.slogdet(matrix)[1]),
        rel=1e-14,
    )
    whitened = system.whiten(right)
    np.testing.assert_allclose(system.cholesky @ whitened, right, atol=1e-14)
    assert system.relative_residual(right, solution) < 1e-14

    for array in (system.matrix, system.cholesky, solution, inverse, whitened):
        assert not array.flags.writeable
        with pytest.raises(ValueError):
            array.setflags(write=True)

    diagnostics = system.diagnostics()
    assert diagnostics["schema"] == SPD_SYSTEM_SCHEMA
    assert diagnostics["schema_version"] == SPD_SYSTEM_VERSION
    assert diagnostics["implicit_jitter"] is False
    assert diagnostics["eigenvalue_clipping"] is False
    assert diagnostics["pseudoinverse_fallback"] is False


def test_spd_system_uses_deterministic_symmetrization_within_tolerance() -> None:
    candidate = np.asarray(
        [[2.0, 0.25 + 2e-13], [0.25, 1.0]],
        dtype=np.float64,
    )

    system = SPDSystem.from_matrix(
        candidate,
        name="roundoff-asymmetric fixture",
        symmetry_absolute_tolerance=1e-12,
        symmetry_relative_tolerance=0.0,
    )

    expected = 0.5 * (candidate + candidate.T)
    np.testing.assert_array_equal(system.matrix, expected)
    assert system.symmetry_error == pytest.approx(2e-13)
    assert system.symmetry_tolerance == pytest.approx(1e-12)


def test_spd_system_rejects_material_asymmetry_without_silent_repair() -> None:
    candidate = np.asarray([[2.0, 0.2], [0.3, 1.0]], dtype=np.float64)

    with pytest.raises(SPDValidationError, match="symmetry tolerance"):
        SPDSystem.from_matrix(
            candidate,
            name="asymmetric fixture",
            symmetry_absolute_tolerance=1e-12,
            symmetry_relative_tolerance=0.0,
        )


def test_spd_system_rejects_indefinite_and_singular_matrices_without_jitter() -> None:
    indefinite = np.asarray([[1.0, 2.0], [2.0, 1.0]], dtype=np.float64)
    singular = np.asarray([[1.0, 0.0], [0.0, 0.0]], dtype=np.float64)

    with pytest.raises(SPDValidationError, match="positive definite"):
        SPDSystem.from_matrix(indefinite, name="indefinite fixture")
    with pytest.raises(SPDValidationError, match="positive definite"):
        SPDSystem.from_matrix(singular, name="singular fixture")


def test_spd_system_enforces_the_declared_condition_limit() -> None:
    ill_conditioned = np.diag(np.asarray([1.0, 1e-14], dtype=np.float64))

    with pytest.raises(SPDConditionError, match="declared limit"):
        SPDSystem.from_matrix(
            ill_conditioned,
            name="ill-conditioned fixture",
            maximum_condition_number=1e12,
        )


def test_spd_system_rejects_invalid_right_hand_sides() -> None:
    system = SPDSystem.from_matrix(np.eye(2), name="identity fixture")

    with pytest.raises(SPDSolveError, match="leading dimension"):
        system.solve(np.ones(3))
    with pytest.raises(SPDSolveError, match="finite"):
        system.solve(np.asarray([1.0, np.nan]))
    with pytest.raises(SPDSolveError, match="vector"):
        system.quadratic_form(np.eye(2))
