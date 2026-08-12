from __future__ import annotations

import numpy as np
import pytest

from bayesian_phystwin.numerical_linear_algebra_v1 import (
    SymmetricSolveDiagnostics,
    SymmetricSolveResult,
    solve_psd,
    solve_spd,
)


def _diagnostics(*, dimension: int = 2) -> SymmetricSolveDiagnostics:
    return SymmetricSolveDiagnostics(
        method="cholesky",
        dimension=dimension,
        numerical_rank=dimension,
        condition_number=1.0,
        minimum_eigenvalue=1.0 if dimension else 0.0,
        maximum_eigenvalue=1.0 if dimension else 0.0,
        relative_residual_norm=0.0,
    )


def test_spd_solve_reports_cholesky_semantics_and_covariance() -> None:
    matrix = np.array([[4.0, 1.0], [1.0, 3.0]])
    right = np.array([1.0, 2.0])

    result = solve_spd(matrix, right, compute_covariance=True)

    assert np.allclose(matrix @ result.solution, right)
    assert result.covariance is not None
    assert np.allclose(matrix @ result.covariance, np.eye(2))
    assert result.diagnostics.method == "cholesky"
    assert result.diagnostics.numerical_rank == 2
    assert result.diagnostics.condition_number >= 1.0
    assert result.diagnostics.relative_residual_norm < 1e-14
    assert result.diagnostics.regularization == 0.0
    assert not result.solution.flags.writeable
    assert not result.covariance.flags.writeable


def test_spd_solve_supports_multiple_right_hand_sides() -> None:
    matrix = np.diag([2.0, 4.0, 8.0])
    right = np.eye(3)

    result = solve_spd(matrix, right)

    assert result.solution.shape == right.shape
    assert np.allclose(matrix @ result.solution, right)
    assert result.covariance is None


def test_spd_solve_can_delegate_condition_admission_explicitly() -> None:
    matrix = np.diag([1.0, 1e-16])

    result = solve_spd(
        matrix,
        np.zeros(2),
        maximum_condition_number=None,
    )

    assert result.diagnostics.condition_number == pytest.approx(1e16)
    assert result.diagnostics.relative_residual_norm == 0.0


def test_spd_solve_fails_closed_on_conditioning_and_input_contracts() -> None:
    with pytest.raises(np.linalg.LinAlgError, match="condition number"):
        solve_spd(
            np.diag([1.0, 1e-12]),
            np.ones(2),
            maximum_condition_number=1e6,
        )
    with pytest.raises(np.linalg.LinAlgError):
        solve_spd(np.diag([1.0, 0.0]), np.ones(2))
    with pytest.raises(ValueError, match="symmetric"):
        solve_spd(np.array([[1.0, 1.0], [0.0, 1.0]]), np.ones(2))
    with pytest.raises(ValueError, match="real numeric"):
        solve_spd(np.eye(2), np.array([True, False]))
    with pytest.raises(ValueError, match="literal Boolean"):
        solve_spd(np.eye(1), np.ones(1), compute_covariance=np.bool_(True))


@pytest.mark.parametrize(
    ("matrix", "right", "message"),
    [
        (np.ones(2), np.ones(2), "2 dimensions"),
        (np.ones((2, 3)), np.ones(2), "square"),
        (np.array([[1.0, np.nan], [np.nan, 1.0]]), np.ones(2), "finite"),
        (np.eye(2), np.ones((2, 1, 1)), "1 or 2 dimensions"),
        (np.eye(2), np.ones(3), "leading dimension"),
        (np.eye(2), np.array([1.0, np.inf]), "finite"),
    ],
)
def test_symmetric_solve_rejects_malformed_arrays(
    matrix: np.ndarray,
    right: np.ndarray,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        solve_spd(matrix, right)


@pytest.mark.parametrize(
    ("keyword", "value", "message"),
    [
        ("maximum_condition_number", True, "finite real"),
        ("maximum_condition_number", 0.5, "at least 1.0"),
        ("maximum_condition_number", np.inf, "finite"),
        ("symmetry_tolerance", -1.0, "at least 0.0"),
    ],
)
def test_spd_solve_rejects_invalid_numerical_policy(
    keyword: str,
    value: object,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        solve_spd(np.eye(1), np.ones(1), **{keyword: value})


def test_psd_solve_returns_minimum_norm_consistent_solution() -> None:
    matrix = np.array([[1.0, 1.0], [1.0, 1.0]])
    right = np.array([2.0, 2.0])

    result = solve_psd(matrix, right, compute_covariance=True)

    assert np.allclose(result.solution, np.array([1.0, 1.0]))
    assert np.allclose(matrix @ result.solution, right)
    assert result.covariance is not None
    assert np.allclose(result.covariance, np.linalg.pinv(matrix))
    assert result.diagnostics.method == "eigh-pseudoinverse"
    assert result.diagnostics.numerical_rank == 1
    assert result.diagnostics.relative_residual_norm < 1e-14


def test_psd_solve_supports_multiple_rhs_and_explicit_condition_delegation() -> None:
    matrix = np.diag([1.0, 1e-12, 0.0])
    right = np.array([[1.0, 2.0], [1e-12, 2e-12], [0.0, 0.0]])

    result = solve_psd(
        matrix,
        right,
        maximum_condition_number=None,
        relative_rank_tolerance=0.0,
    )

    assert result.solution.shape == right.shape
    assert np.allclose(matrix @ result.solution, right)
    assert result.diagnostics.numerical_rank == 2
    assert result.diagnostics.condition_number == pytest.approx(1e12)
    assert result.covariance is None


def test_psd_solve_handles_rank_zero_and_zero_dimension() -> None:
    rank_zero = solve_psd(
        np.zeros((2, 2)),
        np.zeros(2),
        compute_covariance=True,
    )
    empty = solve_psd(
        np.zeros((0, 0)),
        np.zeros((0, 2)),
        compute_covariance=True,
    )

    assert np.array_equal(rank_zero.solution, np.zeros(2))
    assert rank_zero.covariance is not None
    assert np.array_equal(rank_zero.covariance, np.zeros((2, 2)))
    assert rank_zero.diagnostics.numerical_rank == 0
    assert empty.solution.shape == (0, 2)
    assert empty.covariance is not None
    assert empty.covariance.shape == (0, 0)


def test_psd_solve_rejects_inconsistent_indefinite_or_ill_conditioned_systems() -> None:
    matrix = np.array([[1.0, 1.0], [1.0, 1.0]])
    with pytest.raises(np.linalg.LinAlgError, match="outside the matrix range"):
        solve_psd(matrix, np.array([1.0, 0.0]))
    with pytest.raises(np.linalg.LinAlgError, match="positive semidefinite"):
        solve_psd(np.diag([1.0, -1.0]), np.ones(2))
    with pytest.raises(np.linalg.LinAlgError, match="condition number"):
        solve_psd(
            np.diag([1.0, 1e-8]),
            np.ones(2),
            relative_rank_tolerance=0.0,
            maximum_condition_number=1e4,
        )


@pytest.mark.parametrize(
    ("keyword", "value", "message"),
    [
        ("compute_covariance", np.bool_(True), "literal Boolean"),
        ("relative_rank_tolerance", -1.0, "at least 0.0"),
        ("consistency_tolerance", -1.0, "at least 0.0"),
        ("symmetry_tolerance", -1.0, "at least 0.0"),
    ],
)
def test_psd_solve_rejects_invalid_numerical_policy(
    keyword: str,
    value: object,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        solve_psd(np.eye(1), np.ones(1), **{keyword: value})


@pytest.mark.parametrize(
    "overrides",
    [
        {"method": "inverse"},
        {"method": 1},
        {"dimension": -1},
        {"dimension": True},
        {"numerical_rank": -1},
        {"numerical_rank": 3},
        {"condition_number": 0.5},
        {"condition_number": np.nan},
        {"minimum_eigenvalue": 2.0, "maximum_eigenvalue": 1.0},
        {"relative_residual_norm": -1.0},
        {"regularization": -1.0},
    ],
)
def test_diagnostics_reject_inconsistent_numerical_semantics(
    overrides: dict[str, object],
) -> None:
    values: dict[str, object] = {
        "method": "cholesky",
        "dimension": 2,
        "numerical_rank": 2,
        "condition_number": 1.0,
        "minimum_eigenvalue": 1.0,
        "maximum_eigenvalue": 1.0,
        "relative_residual_norm": 0.0,
        "regularization": 0.0,
    }
    values.update(overrides)

    with pytest.raises(ValueError):
        SymmetricSolveDiagnostics(**values)


@pytest.mark.parametrize(
    ("solution", "covariance", "diagnostics", "message"),
    [
        (np.zeros(2), None, object(), "diagnostics"),
        (np.zeros((2, 1, 1)), None, _diagnostics(), "1 or 2 dimensions"),
        (np.zeros(3), None, _diagnostics(), "leading dimension"),
        (np.zeros(2), np.eye(3), _diagnostics(), "shape changed"),
        (np.zeros(2), np.diag([1.0, -1.0]), _diagnostics(), "semidefinite"),
    ],
)
def test_result_rejects_inconsistent_arrays(
    solution: np.ndarray,
    covariance: np.ndarray | None,
    diagnostics: object,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        SymmetricSolveResult(
            solution=solution,
            covariance=covariance,
            diagnostics=diagnostics,  # type: ignore[arg-type]
        )


def test_zero_dimensional_spd_system_is_explicit_and_immutable() -> None:
    result = solve_spd(
        np.zeros((0, 0)),
        np.zeros(0),
        compute_covariance=True,
    )

    assert result.solution.shape == (0,)
    assert result.covariance is not None
    assert result.covariance.shape == (0, 0)
    assert result.diagnostics.dimension == 0
    assert result.diagnostics.numerical_rank == 0
    assert not result.solution.flags.writeable
