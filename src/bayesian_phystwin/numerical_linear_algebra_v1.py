"""Versioned fail-closed numerical policy for symmetric information systems.

The helpers in this module deliberately avoid implicit regularization and direct
matrix inversion.  They expose the solver, numerical rank, conditioning, and
residual semantics so callers can bind numerical admission separately from the
scientific model.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

FloatArray: TypeAlias = NDArray[np.float64]


def _finite_real(
    value: object,
    *,
    name: str,
    minimum: float,
    allow_none: bool = False,
) -> float | None:
    if allow_none and value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number")
    result = float(value)
    if not np.isfinite(result) or result < minimum:
        raise ValueError(f"{name} must be finite and at least {minimum}")
    return result


def _finite_numeric_array(
    value: object,
    *,
    name: str,
    dimensions: frozenset[int],
) -> FloatArray:
    raw = np.asarray(value)
    if raw.dtype.kind not in {"i", "u", "f"}:
        raise ValueError(f"{name} must contain real numeric values")
    result: FloatArray = np.asarray(raw, dtype=np.float64)
    if result.ndim not in dimensions:
        expected = " or ".join(str(dimension) for dimension in sorted(dimensions))
        raise ValueError(f"{name} must have {expected} dimensions")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values")
    return result


def _immutable(value: FloatArray) -> FloatArray:
    canonical: FloatArray = np.asarray(
        value,
        dtype=np.dtype("<f8"),
        order="C",
    )
    return np.frombuffer(
        canonical.tobytes(order="C"),
        dtype=np.dtype("<f8"),
    ).reshape(canonical.shape)


def _symmetric_matrix(
    value: object,
    *,
    name: str,
    symmetry_tolerance: float,
) -> FloatArray:
    matrix = _finite_numeric_array(
        value,
        name=name,
        dimensions=frozenset({2}),
    )
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"{name} must be square")
    if not np.allclose(
        matrix,
        matrix.T,
        atol=symmetry_tolerance,
        rtol=symmetry_tolerance,
    ):
        raise ValueError(f"{name} must be symmetric")
    return 0.5 * (matrix + matrix.T)


def _right_hand_side(value: object, *, dimension: int) -> FloatArray:
    right = _finite_numeric_array(
        value,
        name="right_hand_side",
        dimensions=frozenset({1, 2}),
    )
    if right.shape[0] != dimension:
        raise ValueError("right_hand_side leading dimension changed")
    return right


def _as_matrix(value: FloatArray) -> tuple[FloatArray, bool]:
    if value.ndim == 1:
        return value[:, None], True
    return value, False


def _relative_residual(
    matrix: FloatArray,
    solution: FloatArray,
    right: FloatArray,
) -> float:
    residual_norm = float(np.linalg.norm(matrix @ solution - right))
    right_norm = float(np.linalg.norm(right))
    return residual_norm / right_norm if right_norm > 0.0 else residual_norm


@dataclass(frozen=True, slots=True)
class SymmetricSolveDiagnostics:
    """Numerical semantics for one accepted symmetric-system solve."""

    method: str
    dimension: int
    numerical_rank: int
    condition_number: float
    minimum_eigenvalue: float
    maximum_eigenvalue: float
    relative_residual_norm: float
    regularization: float = 0.0

    def __post_init__(self) -> None:
        if type(self.method) is not str or self.method not in {
            "cholesky",
            "eigh-pseudoinverse",
        }:
            raise ValueError("method must identify an implemented solver")
        if type(self.dimension) is not int or self.dimension < 0:
            raise ValueError("dimension must be a nonnegative integer")
        if (
            type(self.numerical_rank) is not int
            or not 0 <= self.numerical_rank <= self.dimension
        ):
            raise ValueError("numerical_rank must lie in [0, dimension]")
        condition = _finite_real(
            self.condition_number,
            name="condition_number",
            minimum=1.0,
        )
        minimum = _finite_real(
            self.minimum_eigenvalue,
            name="minimum_eigenvalue",
            minimum=-float("inf"),
        )
        maximum = _finite_real(
            self.maximum_eigenvalue,
            name="maximum_eigenvalue",
            minimum=-float("inf"),
        )
        residual = _finite_real(
            self.relative_residual_norm,
            name="relative_residual_norm",
            minimum=0.0,
        )
        regularization = _finite_real(
            self.regularization,
            name="regularization",
            minimum=0.0,
        )
        assert condition is not None
        assert minimum is not None
        assert maximum is not None
        assert residual is not None
        assert regularization is not None
        if minimum > maximum:
            raise ValueError("minimum_eigenvalue exceeds maximum_eigenvalue")
        object.__setattr__(self, "condition_number", condition)
        object.__setattr__(self, "minimum_eigenvalue", minimum)
        object.__setattr__(self, "maximum_eigenvalue", maximum)
        object.__setattr__(self, "relative_residual_norm", residual)
        object.__setattr__(self, "regularization", regularization)


@dataclass(frozen=True, slots=True)
class SymmetricSolveResult:
    """Immutable solution, optional covariance, and numerical diagnostics."""

    solution: FloatArray
    covariance: FloatArray | None
    diagnostics: SymmetricSolveDiagnostics

    def __post_init__(self) -> None:
        if not isinstance(self.diagnostics, SymmetricSolveDiagnostics):
            raise ValueError("diagnostics must be a SymmetricSolveDiagnostics")
        solution = _finite_numeric_array(
            self.solution,
            name="solution",
            dimensions=frozenset({1, 2}),
        )
        if solution.shape[0] != self.diagnostics.dimension:
            raise ValueError("solution leading dimension changed")
        covariance = None
        if self.covariance is not None:
            covariance = _symmetric_matrix(
                self.covariance,
                name="covariance",
                symmetry_tolerance=1e-10,
            )
            if covariance.shape != (self.diagnostics.dimension,) * 2:
                raise ValueError("covariance shape changed")
            eigenvalues = np.linalg.eigvalsh(covariance)
            tolerance = 1e-12 * max(
                float(np.max(np.abs(eigenvalues), initial=0.0)),
                1.0,
            )
            if float(np.min(eigenvalues, initial=0.0)) < -tolerance:
                raise ValueError("covariance must be positive semidefinite")
        object.__setattr__(self, "solution", _immutable(solution))
        if covariance is not None:
            object.__setattr__(self, "covariance", _immutable(covariance))


def solve_spd(
    matrix: object,
    right_hand_side: object,
    *,
    compute_covariance: bool = False,
    maximum_condition_number: float | None = 1e14,
    symmetry_tolerance: float = 1e-10,
) -> SymmetricSolveResult:
    """Solve an SPD system with Cholesky and no implicit regularization.

    ``maximum_condition_number=None`` disables condition-number admission for a
    caller that already owns an equivalent frozen guard.  The method still
    reports the exact two-norm condition number implied by the eigenvalues.
    """

    if type(compute_covariance) is not bool:
        raise ValueError("compute_covariance must be a literal Boolean")
    maximum_condition = _finite_real(
        maximum_condition_number,
        name="maximum_condition_number",
        minimum=1.0,
        allow_none=True,
    )
    symmetry = _finite_real(
        symmetry_tolerance,
        name="symmetry_tolerance",
        minimum=0.0,
    )
    assert symmetry is not None
    normal = _symmetric_matrix(
        matrix,
        name="matrix",
        symmetry_tolerance=symmetry,
    )
    right = _right_hand_side(right_hand_side, dimension=len(normal))
    right_matrix, was_vector = _as_matrix(right)

    if not len(normal):
        empty_solution = right_matrix.copy()
        if was_vector:
            empty_solution = empty_solution[:, 0]
        empty_covariance = normal.copy() if compute_covariance else None
        return SymmetricSolveResult(
            solution=empty_solution,
            covariance=empty_covariance,
            diagnostics=SymmetricSolveDiagnostics(
                method="cholesky",
                dimension=0,
                numerical_rank=0,
                condition_number=1.0,
                minimum_eigenvalue=0.0,
                maximum_eigenvalue=0.0,
                relative_residual_norm=0.0,
            ),
        )

    eigenvalues = np.linalg.eigvalsh(normal)
    factor = np.linalg.cholesky(normal)
    minimum_eigenvalue = float(eigenvalues[0])
    maximum_eigenvalue = float(eigenvalues[-1])
    condition_number = maximum_eigenvalue / minimum_eigenvalue
    if maximum_condition is not None and condition_number > maximum_condition:
        raise np.linalg.LinAlgError(
            "matrix condition number exceeds maximum_condition_number"
        )

    intermediate = np.linalg.solve(factor, right_matrix)
    solution_matrix = np.linalg.solve(factor.T, intermediate)
    solution: FloatArray = (
        solution_matrix[:, 0] if was_vector else solution_matrix
    )
    covariance: FloatArray | None = None
    if compute_covariance:
        inverse_factor = np.linalg.solve(
            factor,
            np.eye(len(factor), dtype=np.float64),
        )
        covariance = inverse_factor.T @ inverse_factor
        covariance = 0.5 * (covariance + covariance.T)

    return SymmetricSolveResult(
        solution=solution,
        covariance=covariance,
        diagnostics=SymmetricSolveDiagnostics(
            method="cholesky",
            dimension=len(normal),
            numerical_rank=len(normal),
            condition_number=condition_number,
            minimum_eigenvalue=minimum_eigenvalue,
            maximum_eigenvalue=maximum_eigenvalue,
            relative_residual_norm=_relative_residual(normal, solution, right),
        ),
    )


def solve_psd(
    matrix: object,
    right_hand_side: object,
    *,
    compute_covariance: bool = False,
    maximum_condition_number: float | None = 1e14,
    relative_rank_tolerance: float = 1e-12,
    consistency_tolerance: float = 1e-10,
    symmetry_tolerance: float = 1e-10,
) -> SymmetricSolveResult:
    """Solve a consistent PSD system by an explicit minimum-norm eigensolve."""

    if type(compute_covariance) is not bool:
        raise ValueError("compute_covariance must be a literal Boolean")
    maximum_condition = _finite_real(
        maximum_condition_number,
        name="maximum_condition_number",
        minimum=1.0,
        allow_none=True,
    )
    rank_tolerance = _finite_real(
        relative_rank_tolerance,
        name="relative_rank_tolerance",
        minimum=0.0,
    )
    consistency = _finite_real(
        consistency_tolerance,
        name="consistency_tolerance",
        minimum=0.0,
    )
    symmetry = _finite_real(
        symmetry_tolerance,
        name="symmetry_tolerance",
        minimum=0.0,
    )
    assert rank_tolerance is not None
    assert consistency is not None
    assert symmetry is not None
    normal = _symmetric_matrix(
        matrix,
        name="matrix",
        symmetry_tolerance=symmetry,
    )
    right = _right_hand_side(right_hand_side, dimension=len(normal))
    right_matrix, was_vector = _as_matrix(right)

    if not len(normal):
        empty_solution = right_matrix.copy()
        if was_vector:
            empty_solution = empty_solution[:, 0]
        empty_covariance = normal.copy() if compute_covariance else None
        return SymmetricSolveResult(
            solution=empty_solution,
            covariance=empty_covariance,
            diagnostics=SymmetricSolveDiagnostics(
                method="eigh-pseudoinverse",
                dimension=0,
                numerical_rank=0,
                condition_number=1.0,
                minimum_eigenvalue=0.0,
                maximum_eigenvalue=0.0,
                relative_residual_norm=0.0,
            ),
        )

    eigenvalues, eigenvectors = np.linalg.eigh(normal)
    spectral_scale = max(float(np.max(np.abs(eigenvalues))), np.finfo(float).tiny)
    absolute_rank_tolerance = rank_tolerance * spectral_scale
    if float(eigenvalues[0]) < -absolute_rank_tolerance:
        raise np.linalg.LinAlgError("matrix must be positive semidefinite")
    retained = eigenvalues > absolute_rank_tolerance
    rank = int(np.count_nonzero(retained))
    projected = eigenvectors.T @ right_matrix
    null_component = projected[~retained]
    consistency_limit = consistency * max(1.0, float(np.linalg.norm(right_matrix)))
    if float(np.linalg.norm(null_component)) > consistency_limit:
        raise np.linalg.LinAlgError(
            "right_hand_side has a component outside the matrix range"
        )

    if rank:
        retained_eigenvalues = eigenvalues[retained]
        retained_vectors = eigenvectors[:, retained]
        condition_number = float(
            retained_eigenvalues[-1] / retained_eigenvalues[0]
        )
        if maximum_condition is not None and condition_number > maximum_condition:
            raise np.linalg.LinAlgError(
                "retained matrix condition number exceeds maximum_condition_number"
            )
        solution_matrix = retained_vectors @ (
            projected[retained] / retained_eigenvalues[:, None]
        )
        covariance = (
            (retained_vectors * (1.0 / retained_eigenvalues))
            @ retained_vectors.T
            if compute_covariance
            else None
        )
        if covariance is not None:
            covariance = 0.5 * (covariance + covariance.T)
    else:
        condition_number = 1.0
        solution_matrix = np.zeros_like(right_matrix)
        covariance = np.zeros_like(normal) if compute_covariance else None

    solution = solution_matrix[:, 0] if was_vector else solution_matrix
    return SymmetricSolveResult(
        solution=solution,
        covariance=covariance,
        diagnostics=SymmetricSolveDiagnostics(
            method="eigh-pseudoinverse",
            dimension=len(normal),
            numerical_rank=rank,
            condition_number=condition_number,
            minimum_eigenvalue=float(eigenvalues[0]),
            maximum_eigenvalue=float(eigenvalues[-1]),
            relative_residual_norm=_relative_residual(normal, solution, right),
        ),
    )


__all__ = [
    "SymmetricSolveDiagnostics",
    "SymmetricSolveResult",
    "solve_psd",
    "solve_spd",
]
