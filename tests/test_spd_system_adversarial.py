from __future__ import annotations

import numpy as np
import pytest

from bayesian_phystwin.spd_system import (
    SPDConditionError,
    SPDSolveError,
    SPDSystem,
    SPDValidationError,
)


@pytest.mark.parametrize(
    ("kwargs", "error_type", "message"),
    [
        ({"name": ""}, TypeError, "nonempty literal string"),
        (
            {"name": "x", "symmetry_absolute_tolerance": True},
            TypeError,
            "real scalar",
        ),
        (
            {"name": "x", "symmetry_relative_tolerance": object()},
            TypeError,
            "real scalar",
        ),
        (
            {"name": "x", "symmetry_relative_tolerance": -1.0},
            ValueError,
            "nonnegative",
        ),
        (
            {"name": "x", "solve_residual_tolerance": 0.0},
            ValueError,
            "positive",
        ),
        (
            {"name": "x", "maximum_condition_number": 0.5},
            ValueError,
            "at least one",
        ),
    ],
)
def test_spd_system_rejects_invalid_configuration(
    kwargs: dict[str, object],
    error_type: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error_type, match=message):
        SPDSystem.from_matrix(np.eye(2), **kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "candidate",
    [
        "not-a-matrix",
        np.ones(2),
        np.zeros((0, 0)),
        np.asarray([[1.0, np.nan], [np.nan, 1.0]]),
    ],
)
def test_spd_system_rejects_malformed_matrices(candidate: object) -> None:
    with pytest.raises(SPDValidationError):
        SPDSystem.from_matrix(candidate, name="malformed fixture")


def test_spd_system_rejects_symmetrization_overflow() -> None:
    maximum = np.finfo(np.float64).max
    candidate = np.asarray([[maximum, maximum], [maximum, maximum]], dtype=np.float64)

    with np.errstate(over="ignore", invalid="ignore"):
        with pytest.raises(SPDValidationError, match="overflowed"):
            SPDSystem.from_matrix(candidate, name="overflow fixture")


def test_spd_system_rejects_nonfinite_factor_and_condition_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        np.linalg,
        "cholesky",
        lambda matrix: np.full_like(matrix, np.nan),
    )
    with pytest.raises(SPDValidationError, match="non-finite Cholesky"):
        SPDSystem.from_matrix(np.eye(2), name="bad-factor fixture")

    monkeypatch.undo()

    def fail_condition(matrix: np.ndarray) -> float:
        raise np.linalg.LinAlgError("condition failure")

    monkeypatch.setattr(np.linalg, "cond", fail_condition)
    with pytest.raises(SPDConditionError, match="could not be evaluated"):
        SPDSystem.from_matrix(np.eye(2), name="condition-error fixture")

    monkeypatch.setattr(np.linalg, "cond", lambda matrix: float("inf"))
    with pytest.raises(SPDConditionError, match="non-finite"):
        SPDSystem.from_matrix(np.eye(2), name="condition-inf fixture")


def test_spd_system_rejects_invalid_logdet_and_residual_inputs() -> None:
    system = SPDSystem.from_matrix(np.eye(2), name="identity fixture")
    invalid_factor = SPDSystem(
        name=system.name,
        matrix=system.matrix,
        cholesky=np.diag(np.asarray([0.0, 1.0])),
        condition_number=system.condition_number,
        symmetry_error=system.symmetry_error,
        symmetry_tolerance=system.symmetry_tolerance,
        solve_residual_tolerance=system.solve_residual_tolerance,
    )
    with np.errstate(divide="ignore"):
        with pytest.raises(SPDValidationError, match="log determinant"):
            _ = invalid_factor.log_determinant

    with pytest.raises(SPDSolveError, match="numeric"):
        system.solve(object())
    with pytest.raises(SPDSolveError, match="numeric"):
        system.relative_residual(np.ones(2), object())
    with pytest.raises(SPDSolveError, match="shape differs"):
        system.relative_residual(np.ones(2), np.ones((2, 1)))
    with pytest.raises(SPDSolveError, match="finite"):
        system.relative_residual(np.ones(2), np.asarray([np.nan, 0.0]))

    invalid_matrix = SPDSystem(
        name=system.name,
        matrix=np.full((2, 2), np.inf),
        cholesky=system.cholesky,
        condition_number=system.condition_number,
        symmetry_error=system.symmetry_error,
        symmetry_tolerance=system.symmetry_tolerance,
        solve_residual_tolerance=system.solve_residual_tolerance,
    )
    with np.errstate(invalid="ignore"):
        with pytest.raises(SPDSolveError, match="non-finite"):
            invalid_matrix.relative_residual(np.ones(2), np.ones(2))


def test_spd_system_fails_closed_on_backend_solve_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    system = SPDSystem.from_matrix(np.eye(2), name="identity fixture")

    def fail_solve(matrix: np.ndarray, right: np.ndarray) -> np.ndarray:
        raise np.linalg.LinAlgError("deliberate failure")

    monkeypatch.setattr(np.linalg, "solve", fail_solve)
    with pytest.raises(SPDSolveError, match="triangular solve"):
        system.solve(np.ones(2))
    with pytest.raises(SPDSolveError, match="whitening solve"):
        system.whiten(np.ones(2))

    monkeypatch.setattr(
        np.linalg,
        "solve",
        lambda matrix, right: np.full_like(right, np.nan),
    )
    with pytest.raises(SPDSolveError, match="non-finite values"):
        system.solve(np.ones(2))
    with pytest.raises(SPDSolveError, match="non-finite values"):
        system.whiten(np.ones(2))


def test_spd_system_enforces_solve_whitening_and_inverse_residuals(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    system = SPDSystem.from_matrix(np.eye(2), name="identity fixture")

    monkeypatch.setattr(SPDSystem, "relative_residual", lambda self, b, x: 1.0)
    with pytest.raises(SPDSolveError, match="relative solve residual"):
        system.solve(np.ones(2))

    monkeypatch.undo()
    original_norm = np.linalg.norm
    calls = 0

    def bad_norm(value: np.ndarray, ord: object = None) -> float:
        nonlocal calls
        calls += 1
        if calls == 4:
            return 1.0
        return float(original_norm(value, ord=ord))

    monkeypatch.setattr(np.linalg, "norm", bad_norm)
    with pytest.raises(SPDSolveError, match="whitening residual"):
        system.whiten(np.ones(2))

    monkeypatch.undo()
    monkeypatch.setattr(
        SPDSystem,
        "whiten",
        lambda self, value: np.full(2, np.nan),
    )
    with pytest.raises(SPDSolveError, match="quadratic form"):
        system.quadratic_form(np.ones(2))

    monkeypatch.undo()
    monkeypatch.setattr(
        SPDSystem,
        "solve",
        lambda self, right: np.full_like(np.asarray(right), np.nan),
    )
    with pytest.raises(SPDSolveError, match="non-finite values"):
        system.reconstruct_inverse()

    monkeypatch.setattr(
        SPDSystem,
        "solve",
        lambda self, right: 2.0 * np.eye(self.dimension),
    )
    with pytest.raises(SPDSolveError, match="inverse reconstruction residual"):
        system.reconstruct_inverse()


class _ArrayConversionFailure:
    def __array__(self, dtype: object = None) -> np.ndarray:
        raise ValueError("deliberate array conversion failure")


def test_spd_system_rejects_lossy_and_failed_numeric_coercion() -> None:
    with pytest.raises(SPDValidationError, match="numeric float64"):
        SPDSystem.from_matrix(
            [["1.0", "0.0"], ["0.0", "1.0"]],
            name="string matrix",
        )
    with pytest.raises(SPDValidationError, match="numeric float64"):
        SPDSystem.from_matrix(np.eye(2, dtype=bool), name="boolean matrix")
    with pytest.raises(SPDValidationError, match="numeric float64"):
        SPDSystem.from_matrix(_ArrayConversionFailure(), name="failed matrix")

    system = SPDSystem.from_matrix(np.eye(2), name="identity fixture")
    with pytest.raises(SPDSolveError, match="numeric"):
        system.solve(["1.0", "2.0"])
    with pytest.raises(SPDSolveError, match="numeric"):
        system.solve(np.asarray([True, False]))
    with pytest.raises(SPDSolveError, match="numeric"):
        system.solve(_ArrayConversionFailure())
