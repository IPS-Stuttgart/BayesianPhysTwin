"""Fail-closed symmetric-positive-definite linear algebra.

The backend deliberately exposes factorization, solves, whitening, log
determinants, and explicit inverse reconstruction as separate operations. It
never adds jitter, clips eigenvalues, or substitutes a pseudoinverse.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real
from typing import Final

import numpy as np

SPD_SYSTEM_SCHEMA: Final = "bayesian_phystwin.spd_system"
SPD_SYSTEM_VERSION: Final = 1


class SPDSystemError(ValueError):
    """Base class for fail-closed SPD admission or solve failures."""


class SPDValidationError(SPDSystemError):
    """Raised when a matrix is not a finite symmetric positive-definite system."""


class SPDConditionError(SPDSystemError):
    """Raised when a valid SPD matrix exceeds its declared condition limit."""


class SPDSolveError(SPDSystemError):
    """Raised when a triangular solve fails its residual contract."""


def _literal_name(value: object) -> str:
    if type(value) is not str or not value:
        raise TypeError("SPD system name must be a nonempty literal string")
    return value


def _finite_nonnegative(value: object, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real scalar")
    result = float(value)
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return result


def _finite_positive(value: object, *, name: str) -> float:
    result = _finite_nonnegative(value, name=name)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def _immutable_float64(value: np.ndarray) -> np.ndarray:
    array = np.ascontiguousarray(value, dtype=np.float64)
    immutable = np.frombuffer(array.tobytes(order="C"), dtype=np.float64)
    return immutable.reshape(array.shape)


def _matrix_norm(value: np.ndarray) -> float:
    return float(np.linalg.norm(value, ord=np.inf))


@dataclass(frozen=True, slots=True)
class SPDSystem:
    """One validated SPD matrix and its unique retained Cholesky factor."""

    name: str
    matrix: np.ndarray
    cholesky: np.ndarray
    condition_number: float
    symmetry_error: float
    symmetry_tolerance: float
    solve_residual_tolerance: float

    @classmethod
    def from_matrix(
        cls,
        value: object,
        *,
        name: str,
        maximum_condition_number: float | None = None,
        symmetry_absolute_tolerance: float = 1e-12,
        symmetry_relative_tolerance: float = 1e-10,
        solve_residual_tolerance: float = 1e-10,
    ) -> SPDSystem:
        """Validate, deterministically symmetrize, and factor one SPD matrix."""

        system_name = _literal_name(name)
        absolute_tolerance = _finite_nonnegative(
            symmetry_absolute_tolerance,
            name="symmetry_absolute_tolerance",
        )
        relative_tolerance = _finite_nonnegative(
            symmetry_relative_tolerance,
            name="symmetry_relative_tolerance",
        )
        residual_tolerance = _finite_positive(
            solve_residual_tolerance,
            name="solve_residual_tolerance",
        )
        condition_limit = (
            None
            if maximum_condition_number is None
            else _finite_positive(
                maximum_condition_number,
                name="maximum_condition_number",
            )
        )
        if condition_limit is not None and condition_limit < 1.0:
            raise ValueError("maximum_condition_number must be at least one")

        try:
            untyped_candidate = np.asarray(value)
        except (TypeError, ValueError, OverflowError) as error:
            raise SPDValidationError(
                f"{system_name} must be a numeric float64 matrix"
            ) from error
        if untyped_candidate.dtype.kind not in "fiu":
            raise SPDValidationError(
                f"{system_name} must be a numeric float64 matrix"
            )
        candidate = untyped_candidate.astype(np.float64, copy=False)
        if candidate.ndim != 2 or candidate.shape[0] != candidate.shape[1]:
            raise SPDValidationError(f"{system_name} must be a square matrix")
        if candidate.shape[0] < 1:
            raise SPDValidationError(f"{system_name} must be nonempty")
        if not np.all(np.isfinite(candidate)):
            raise SPDValidationError(f"{system_name} must be finite")

        magnitude = max(1.0, float(np.max(np.abs(candidate))))
        symmetry_error = float(np.max(np.abs(candidate - candidate.T)))
        symmetry_tolerance = absolute_tolerance + relative_tolerance * magnitude
        if symmetry_error > symmetry_tolerance:
            raise SPDValidationError(
                f"{system_name} exceeds its declared symmetry tolerance"
            )
        matrix = 0.5 * (candidate + candidate.T)
        if not np.all(np.isfinite(matrix)):
            raise SPDValidationError(
                f"{system_name} overflowed during deterministic symmetrization"
            )
        try:
            cholesky = np.linalg.cholesky(matrix)
        except np.linalg.LinAlgError as error:
            raise SPDValidationError(
                f"{system_name} must be positive definite"
            ) from error
        if not np.all(np.isfinite(cholesky)):
            raise SPDValidationError(
                f"{system_name} produced a non-finite Cholesky factor"
            )
        try:
            condition_number = float(np.linalg.cond(matrix))
        except np.linalg.LinAlgError as error:
            raise SPDConditionError(
                f"{system_name} condition number could not be evaluated"
            ) from error
        if not np.isfinite(condition_number):
            raise SPDConditionError(
                f"{system_name} has a non-finite condition number"
            )
        if condition_limit is not None and condition_number > condition_limit:
            raise SPDConditionError(
                f"{system_name} condition number {condition_number:.17g} exceeds "
                f"the declared limit {condition_limit:.17g}"
            )

        return cls(
            name=system_name,
            matrix=_immutable_float64(matrix),
            cholesky=_immutable_float64(cholesky),
            condition_number=condition_number,
            symmetry_error=symmetry_error,
            symmetry_tolerance=symmetry_tolerance,
            solve_residual_tolerance=residual_tolerance,
        )

    @property
    def dimension(self) -> int:
        """Return the system dimension."""

        return int(self.matrix.shape[0])

    @property
    def log_determinant(self) -> float:
        """Return ``log(det(A))`` from the retained Cholesky factor."""

        diagonal = np.diag(self.cholesky)
        result = 2.0 * float(np.sum(np.log(diagonal)))
        if not np.isfinite(result):
            raise SPDValidationError(
                f"{self.name} has a non-finite log determinant"
            )
        return result

    def _right_hand_side(self, value: object, *, name: str) -> np.ndarray:
        try:
            untyped_right = np.asarray(value)
        except (TypeError, ValueError, OverflowError) as error:
            raise SPDSolveError(f"{name} must be numeric") from error
        if untyped_right.dtype.kind not in "fiu":
            raise SPDSolveError(f"{name} must be numeric")
        right = untyped_right.astype(np.float64, copy=False)
        if right.ndim not in (1, 2) or right.shape[0] != self.dimension:
            raise SPDSolveError(
                f"{name} must have leading dimension {self.dimension}"
            )
        if not np.all(np.isfinite(right)):
            raise SPDSolveError(f"{name} must be finite")
        return right

    def relative_residual(self, right: object, solution: object) -> float:
        """Return a scale-normalized infinity-norm residual for ``A x = b``."""

        right_array = self._right_hand_side(right, name="right-hand side")
        try:
            solution_array = np.asarray(solution, dtype=np.float64)
        except (TypeError, ValueError, OverflowError) as error:
            raise SPDSolveError("solution must be numeric") from error
        if solution_array.shape != right_array.shape:
            raise SPDSolveError("solution shape differs from the right-hand side")
        if not np.all(np.isfinite(solution_array)):
            raise SPDSolveError("solution must be finite")
        residual = self.matrix @ solution_array - right_array
        numerator = _matrix_norm(residual)
        denominator = max(
            np.finfo(np.float64).tiny,
            _matrix_norm(self.matrix) * _matrix_norm(solution_array)
            + _matrix_norm(right_array),
        )
        result = numerator / denominator
        if not np.isfinite(result):
            raise SPDSolveError("relative solve residual is non-finite")
        return result

    def solve(self, right: object) -> np.ndarray:
        """Solve ``A x = b`` through the retained lower-triangular factor."""

        right_array = self._right_hand_side(right, name="right-hand side")
        try:
            intermediate = np.linalg.solve(self.cholesky, right_array)
            solution = np.linalg.solve(self.cholesky.T, intermediate)
        except np.linalg.LinAlgError as error:
            raise SPDSolveError(f"{self.name} triangular solve failed") from error
        if not np.all(np.isfinite(solution)):
            raise SPDSolveError(f"{self.name} solve produced non-finite values")
        residual = self.relative_residual(right_array, solution)
        if residual > self.solve_residual_tolerance:
            raise SPDSolveError(
                f"{self.name} relative solve residual {residual:.17g} exceeds "
                f"the declared tolerance {self.solve_residual_tolerance:.17g}"
            )
        return _immutable_float64(solution)

    def whiten(self, value: object) -> np.ndarray:
        """Apply ``L^-1`` where ``A = L L.T`` without forming an inverse."""

        right = self._right_hand_side(value, name="value to whiten")
        try:
            whitened = np.linalg.solve(self.cholesky, right)
        except np.linalg.LinAlgError as error:
            raise SPDSolveError(f"{self.name} whitening solve failed") from error
        if not np.all(np.isfinite(whitened)):
            raise SPDSolveError(
                f"{self.name} whitening produced non-finite values"
            )
        residual = self.cholesky @ whitened - right
        denominator = max(
            np.finfo(np.float64).tiny,
            _matrix_norm(self.cholesky) * _matrix_norm(whitened)
            + _matrix_norm(right),
        )
        relative = _matrix_norm(residual) / denominator
        if not np.isfinite(relative) or relative > self.solve_residual_tolerance:
            raise SPDSolveError(
                f"{self.name} whitening residual violates its contract"
            )
        return _immutable_float64(whitened)

    def quadratic_form(self, value: object) -> float:
        """Return ``x.T A^-1 x`` through whitening."""

        vector = self._right_hand_side(value, name="quadratic-form vector")
        if vector.ndim != 1:
            raise SPDSolveError("quadratic-form input must be a vector")
        whitened = self.whiten(vector)
        result = float(whitened @ whitened)
        if not np.isfinite(result) or result < 0.0:
            raise SPDSolveError("quadratic form is invalid")
        return result

    def reconstruct_inverse(self) -> np.ndarray:
        """Explicitly reconstruct ``A^-1`` only for an exported contract."""

        identity = np.eye(self.dimension, dtype=np.float64)
        inverse = np.asarray(self.solve(identity))
        inverse = 0.5 * (inverse + inverse.T)
        if not np.all(np.isfinite(inverse)):
            raise SPDSolveError(
                f"{self.name} inverse reconstruction produced non-finite values"
            )
        residual = self.matrix @ inverse - identity
        relative = _matrix_norm(residual) / max(
            1.0,
            _matrix_norm(self.matrix) * _matrix_norm(inverse),
        )
        tolerance = max(
            10.0 * self.solve_residual_tolerance,
            100.0 * self.dimension * np.finfo(np.float64).eps,
        )
        if not np.isfinite(relative) or relative > tolerance:
            raise SPDSolveError(
                f"{self.name} inverse reconstruction residual violates its contract"
            )
        return _immutable_float64(inverse)

    def diagnostics(self) -> dict[str, object]:
        """Return JSON-compatible numerical diagnostics."""

        return {
            "schema": SPD_SYSTEM_SCHEMA,
            "schema_version": SPD_SYSTEM_VERSION,
            "name": self.name,
            "dimension": self.dimension,
            "condition_number": self.condition_number,
            "symmetry_error": self.symmetry_error,
            "symmetry_tolerance": self.symmetry_tolerance,
            "solve_residual_tolerance": self.solve_residual_tolerance,
            "implicit_jitter": False,
            "eigenvalue_clipping": False,
            "pseudoinverse_fallback": False,
        }


__all__ = [
    "SPDConditionError",
    "SPDSolveError",
    "SPDSystem",
    "SPDSystemError",
    "SPDValidationError",
    "SPD_SYSTEM_SCHEMA",
    "SPD_SYSTEM_VERSION",
]
