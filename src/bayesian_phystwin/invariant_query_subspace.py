"""Basis-invariant query-subspace selection for prospective BPT updates.

Frozen prior-aware solvers threshold individual information eigenvectors. Inside
a repeated eigenspace those vectors are arbitrary. This module thresholds
spectral projectors instead, leaving all historical solvers and evidence intact.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real
from typing import Final

import numpy as np

from ._gauge_aware_contracts import (
    _positive_definite_whitener,
    _positive_semidefinite_square_root,
)

INVARIANT_QUERY_SUBSPACE_SCHEMA: Final = "bayesian_phystwin.invariant_query_subspace"
INVARIANT_QUERY_SUBSPACE_VERSION: Final = 1


def _number(value: object, name: str, *, positive: bool = False) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite nonnegative real number")
    result = float(value)
    if not np.isfinite(result) or result < 0.0 or (positive and result == 0.0):
        qualifier = "positive" if positive else "nonnegative"
        raise ValueError(f"{name} must be a finite {qualifier} real number")
    return result


def _fraction(value: object, name: str, *, positive: bool = False) -> float:
    result = _number(value, name)
    if result > 1.0 or (positive and result == 0.0):
        interval = "(0, 1]" if positive else "[0, 1]"
        raise ValueError(f"{name} must lie in {interval}")
    return result


def _matrix(value: object, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 2 or result.shape[0] != result.shape[1]:
        raise ValueError(f"{name} must be a square matrix")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be finite")
    if not np.allclose(result, result.T, atol=1e-11, rtol=1e-10):
        raise ValueError(f"{name} must be symmetric")
    return 0.5 * (result + result.T)


def _projector(value: object, name: str) -> np.ndarray:
    result = _matrix(value, name)
    if not np.allclose(result @ result, result, atol=1e-10, rtol=1e-10):
        raise ValueError(f"{name} must be idempotent")
    return result


def _readonly(value: np.ndarray) -> np.ndarray:
    source = np.ascontiguousarray(value, dtype=np.float64)
    return np.frombuffer(source.tobytes(), dtype=np.float64).reshape(source.shape)


@dataclass(frozen=True, slots=True)
class InvariantQuerySubspaceConfigV1:
    """Frozen thresholds and numerical clustering for projector selection."""

    minimum_information_fraction: float = 1e-4
    minimum_identifiable_fraction: float = 0.10
    minimum_query_sensitivity_fraction: float = 1e-3
    eigenvalue_floor: float = 1e-12
    relative_spectral_tolerance: float = 1e-9
    absolute_spectral_tolerance: float = 1e-12

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "minimum_information_fraction",
            _fraction(
                self.minimum_information_fraction,
                "minimum_information_fraction",
            ),
        )
        object.__setattr__(
            self,
            "minimum_identifiable_fraction",
            _fraction(
                self.minimum_identifiable_fraction,
                "minimum_identifiable_fraction",
                positive=True,
            ),
        )
        object.__setattr__(
            self,
            "minimum_query_sensitivity_fraction",
            _fraction(
                self.minimum_query_sensitivity_fraction,
                "minimum_query_sensitivity_fraction",
            ),
        )
        object.__setattr__(
            self,
            "eigenvalue_floor",
            _number(self.eigenvalue_floor, "eigenvalue_floor", positive=True),
        )
        relative = _number(
            self.relative_spectral_tolerance,
            "relative_spectral_tolerance",
        )
        absolute = _number(
            self.absolute_spectral_tolerance,
            "absolute_spectral_tolerance",
        )
        if relative == absolute == 0.0:
            raise ValueError("at least one spectral tolerance must be positive")
        object.__setattr__(self, "relative_spectral_tolerance", relative)
        object.__setattr__(self, "absolute_spectral_tolerance", absolute)


@dataclass(frozen=True, slots=True)
class InvariantQuerySubspaceResultV1:
    """Immutable selected mapping, state-coordinate projectors, and spectra."""

    state_mapping: np.ndarray
    identifiable_fractions: np.ndarray
    query_sensitivity_fractions: np.ndarray
    information_projector: np.ndarray
    identifiability_projector: np.ndarray
    query_projector: np.ndarray
    information_eigenvalues: np.ndarray
    identifiability_eigenvalues: np.ndarray
    query_eigenvalues: np.ndarray
    maximum_information: float
    maximum_query_sensitivity: float
    reason: str

    def __post_init__(self) -> None:
        mapping = np.asarray(self.state_mapping, dtype=np.float64)
        if mapping.ndim != 2:
            raise ValueError("state_mapping must be a matrix")
        state_count, rank = mapping.shape
        identifiable = np.asarray(self.identifiable_fractions, dtype=np.float64)
        query = np.asarray(self.query_sensitivity_fractions, dtype=np.float64)
        projectors = (
            _projector(self.information_projector, "information_projector"),
            _projector(
                self.identifiability_projector,
                "identifiability_projector",
            ),
            _projector(self.query_projector, "query_projector"),
        )
        if any(value.shape != (state_count, state_count) for value in projectors):
            raise ValueError("all projectors must use standardized state coordinates")
        if identifiable.shape != (rank,) or query.shape != (rank,):
            raise ValueError("retained fractions must identify every mapping column")
        spectra = tuple(
            np.asarray(value, dtype=np.float64)
            for value in (
                self.information_eigenvalues,
                self.identifiability_eigenvalues,
                self.query_eigenvalues,
            )
        )
        arrays = (mapping, identifiable, query, *spectra)
        if any(not np.all(np.isfinite(value)) for value in arrays):
            raise ValueError("result arrays must be finite")
        if np.any((identifiable < 0.0) | (identifiable > 1.0)):
            raise ValueError("identifiable_fractions must lie in [0, 1]")
        if np.any((query < 0.0) | (query > 1.0 + 1e-10)):
            raise ValueError("query_sensitivity_fractions must lie in [0, 1]")
        maximum_information = _number(
            self.maximum_information,
            "maximum_information",
        )
        maximum_query = _number(
            self.maximum_query_sensitivity,
            "maximum_query_sensitivity",
        )
        if type(self.reason) is not str:
            raise ValueError("reason must be a string")
        if (rank > 0) != (self.reason == "admissible"):
            raise ValueError("reason and retained rank are inconsistent")
        names = (
            "state_mapping",
            "identifiable_fractions",
            "query_sensitivity_fractions",
            "information_eigenvalues",
            "identifiability_eigenvalues",
            "query_eigenvalues",
        )
        for name, value in zip(names, arrays, strict=True):
            object.__setattr__(self, name, _readonly(value))
        for name, value in zip(
            (
                "information_projector",
                "identifiability_projector",
                "query_projector",
            ),
            projectors,
            strict=True,
        ):
            object.__setattr__(self, name, _readonly(value))
        object.__setattr__(self, "maximum_information", maximum_information)
        object.__setattr__(self, "maximum_query_sensitivity", maximum_query)

    @property
    def retained_rank(self) -> int:
        return self.state_mapping.shape[1]

    @property
    def admissible(self) -> bool:
        return self.retained_rank > 0

    def project_state_jacobian(self, values: np.ndarray) -> np.ndarray:
        design = np.asarray(values, dtype=np.float64)
        if design.ndim < 1 or design.shape[-1] != self.state_mapping.shape[0]:
            raise ValueError("state Jacobian final axis does not match the mapping")
        if not np.all(np.isfinite(design)):
            raise ValueError("state Jacobian must be finite")
        return np.einsum("...s,sr->...r", design, self.state_mapping, optimize=True)

    def lift_state_mean(self, reduced_mean: np.ndarray) -> np.ndarray:
        mean = np.asarray(reduced_mean, dtype=np.float64)
        if mean.shape != (self.retained_rank,) or not np.all(np.isfinite(mean)):
            raise ValueError("reduced_mean must be a finite retained-state vector")
        return self.state_mapping @ mean

    def lift_state_covariance(
        self,
        reduced_covariance: np.ndarray,
        state_prior_covariance: np.ndarray,
    ) -> np.ndarray:
        reduced = _matrix(reduced_covariance, "reduced_covariance")
        prior = _matrix(state_prior_covariance, "state_prior_covariance")
        if reduced.shape != (self.retained_rank, self.retained_rank):
            raise ValueError("reduced_covariance shape does not match retained rank")
        if prior.shape != (self.state_mapping.shape[0],) * 2:
            raise ValueError("state_prior_covariance shape does not match state")
        result = (
            prior
            + self.state_mapping
            @ (reduced - np.eye(self.retained_rank))
            @ self.state_mapping.T
        )
        return 0.5 * (result + result.T)

    def diagnostics(self) -> dict[str, object]:
        return {
            "schema": INVARIANT_QUERY_SUBSPACE_SCHEMA,
            "schema_version": INVARIANT_QUERY_SUBSPACE_VERSION,
            "selection": (
                "information-projector/generalized-identifiability-projector/"
                "query-gram-projector-v1"
            ),
            "basis_canonicalization": (
                "projected-coordinate-axes-modified-gram-schmidt-v1"
            ),
            "retained_rank": self.retained_rank,
            "information_rank": int(round(np.trace(self.information_projector))),
            "identifiability_rank": int(
                round(np.trace(self.identifiability_projector))
            ),
            "query_rank": int(round(np.trace(self.query_projector))),
            "maximum_information": self.maximum_information,
            "maximum_query_sensitivity": self.maximum_query_sensitivity,
            "reason": self.reason,
            "repeated_eigenspace_projectors_used": True,
            "individual_information_eigenvectors_thresholded": False,
        }


def _tolerance(values: np.ndarray, config: InvariantQuerySubspaceConfigV1) -> float:
    return config.absolute_spectral_tolerance + (
        config.relative_spectral_tolerance * float(np.max(np.abs(values), initial=0.0))
    )


def _canonical_basis(projector: np.ndarray, rank: int, tolerance: float) -> np.ndarray:
    if rank == 0:
        return np.zeros((len(projector), 0))
    columns: list[np.ndarray] = []
    for coordinate in range(len(projector)):
        vector = projector[:, coordinate].copy()
        for _ in range(2):
            for column in columns:
                vector -= column * float(column @ vector)
        norm = float(np.linalg.norm(vector))
        if norm <= tolerance:
            continue
        vector /= norm
        pivot = int(np.argmax(np.abs(vector)))
        columns.append(-vector if vector[pivot] < 0.0 else vector)
        if len(columns) == rank:
            break
    if len(columns) != rank:
        raise ValueError("selected projector does not expose its declared rank")
    basis = np.column_stack(columns)
    if not np.allclose(basis.T @ basis, np.eye(rank), atol=1e-10, rtol=1e-10):
        raise ValueError("canonical projector basis is not orthonormal")
    return basis


def _select(
    matrix: np.ndarray,
    threshold: float,
    config: InvariantQuerySubspaceConfigV1,
    *,
    positive: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    values, vectors = np.linalg.eigh(0.5 * (matrix + matrix.T))
    tolerance = _tolerance(values, config)
    if float(np.min(values, initial=0.0)) < -tolerance:
        raise ValueError("selection matrix is not positive semidefinite")
    values = np.maximum(values, 0.0)
    selected = values + tolerance >= threshold
    if positive:
        selected &= values > tolerance
    raw = vectors[:, selected]
    projector = raw @ raw.T if raw.shape[1] else np.zeros_like(matrix)
    basis = _canonical_basis(
        projector,
        raw.shape[1],
        max(config.absolute_spectral_tolerance, tolerance),
    )
    return basis, 0.5 * (projector + projector.T), values, values[selected]


def select_invariant_query_subspace(
    known_information: np.ndarray,
    conditional_information: np.ndarray,
    state_prior_covariance: np.ndarray,
    query_state_jacobian: np.ndarray,
    *,
    config: InvariantQuerySubspaceConfigV1 | None = None,
) -> InvariantQuerySubspaceResultV1:
    """Select a state span without thresholding arbitrary repeated eigenvectors."""

    settings = config or InvariantQuerySubspaceConfigV1()
    if not isinstance(settings, InvariantQuerySubspaceConfigV1):
        raise TypeError("config must be an InvariantQuerySubspaceConfigV1")
    known = _matrix(known_information, "known_information")
    conditional = _matrix(conditional_information, "conditional_information")
    prior = _matrix(state_prior_covariance, "state_prior_covariance")
    if known.shape != conditional.shape or known.shape != prior.shape:
        raise ValueError("information and prior matrices must have the same shape")
    state_count = len(prior)
    query = np.asarray(query_state_jacobian, dtype=np.float64)
    if query.ndim != 3 or query.shape[1:] != (3, state_count):
        raise ValueError("query_state_jacobian must have shape (Q, 3, S)")
    if len(query) == 0 or not np.all(np.isfinite(query)):
        raise ValueError("query_state_jacobian must be nonempty and finite")
    state_root = _positive_semidefinite_square_root(
        prior,
        "state prior covariance",
        eigenvalue_floor=settings.eigenvalue_floor,
    )
    standardized = state_root.T @ conditional @ state_root
    raw_values = np.linalg.eigvalsh(0.5 * (standardized + standardized.T))
    maximum_information = float(np.max(np.maximum(raw_values, 0.0), initial=0.0))
    information_basis, information_projector, info_values, _ = _select(
        standardized,
        max(
            settings.eigenvalue_floor,
            settings.minimum_information_fraction * maximum_information,
        ),
        settings,
        positive=True,
    )

    def empty(
        reason: str,
        ident_projector: np.ndarray | None = None,
        ident_values: np.ndarray | None = None,
        maximum_query: float = 0.0,
    ) -> InvariantQuerySubspaceResultV1:
        zero = np.zeros((state_count, state_count))
        return InvariantQuerySubspaceResultV1(
            state_mapping=np.zeros((state_count, 0)),
            identifiable_fractions=np.zeros(0),
            query_sensitivity_fractions=np.zeros(0),
            information_projector=information_projector,
            identifiability_projector=(
                zero if ident_projector is None else ident_projector
            ),
            query_projector=zero,
            information_eigenvalues=info_values,
            identifiability_eigenvalues=(
                np.zeros(state_count) if ident_values is None else ident_values
            ),
            query_eigenvalues=np.zeros(state_count),
            maximum_information=maximum_information,
            maximum_query_sensitivity=maximum_query,
            reason=reason,
        )

    if not information_basis.shape[1]:
        return empty("no-information-support")
    information_mapping = state_root @ information_basis
    known_reduced = information_mapping.T @ known @ information_mapping
    conditional_reduced = information_mapping.T @ conditional @ information_mapping
    whitener = _positive_definite_whitener(
        known_reduced,
        "known information in selected subspace",
    )
    ident_basis_white, _, ident_values, selected_ident_values = _select(
        whitener @ conditional_reduced @ whitener,
        settings.minimum_identifiable_fraction,
        settings,
        positive=False,
    )
    if not ident_basis_white.shape[1]:
        return empty("no-identifiable-support", ident_values=ident_values)
    raw_ident_basis = whitener @ ident_basis_white
    orthonormal_ident_basis, _ = np.linalg.qr(raw_ident_basis, mode="reduced")
    ident_projector_reduced = orthonormal_ident_basis @ orthonormal_ident_basis.T
    ident_basis = _canonical_basis(
        ident_projector_reduced,
        orthonormal_ident_basis.shape[1],
        settings.absolute_spectral_tolerance,
    )
    standardized_ident_basis = information_basis @ ident_basis
    ident_projector = standardized_ident_basis @ standardized_ident_basis.T
    identifiable_mapping = state_root @ standardized_ident_basis
    query_flat = query.reshape(-1, state_count)
    query_design = query_flat @ identifiable_mapping
    query_gram = query_design.T @ query_design
    raw_query_values = np.linalg.eigvalsh(0.5 * (query_gram + query_gram.T))
    maximum_query_power = float(np.max(np.maximum(raw_query_values, 0.0), initial=0.0))
    maximum_query = float(np.sqrt(maximum_query_power))
    if settings.minimum_query_sensitivity_fraction == 0.0:
        query_basis = np.eye(identifiable_mapping.shape[1])
        query_values = np.maximum(raw_query_values, 0.0)
        selected_query_values = query_values.copy()
    else:
        query_basis, _, query_values, selected_query_values = _select(
            query_gram,
            settings.minimum_query_sensitivity_fraction**2 * maximum_query_power,
            settings,
            positive=True,
        )
    if not query_basis.shape[1]:
        return empty(
            "no-query-support",
            ident_projector,
            ident_values,
            maximum_query,
        )
    standardized_query_basis = standardized_ident_basis @ query_basis
    query_projector = standardized_query_basis @ standardized_query_basis.T
    mapping = state_root @ standardized_query_basis
    conservative_identifiability = float(np.min(selected_ident_values))
    identifiable_fractions = np.full(
        mapping.shape[1],
        np.clip(conservative_identifiability, 0.0, 1.0),
    )
    query_fractions = (
        np.sqrt(np.sort(selected_query_values)[::-1]) / maximum_query
        if maximum_query > 0.0
        else np.zeros(mapping.shape[1])
    )
    return InvariantQuerySubspaceResultV1(
        state_mapping=mapping,
        identifiable_fractions=identifiable_fractions,
        query_sensitivity_fractions=query_fractions,
        information_projector=information_projector,
        identifiability_projector=ident_projector,
        query_projector=query_projector,
        information_eigenvalues=info_values,
        identifiability_eigenvalues=ident_values,
        query_eigenvalues=query_values,
        maximum_information=maximum_information,
        maximum_query_sensitivity=maximum_query,
        reason="admissible",
    )


__all__ = [
    "INVARIANT_QUERY_SUBSPACE_SCHEMA",
    "INVARIANT_QUERY_SUBSPACE_VERSION",
    "InvariantQuerySubspaceConfigV1",
    "InvariantQuerySubspaceResultV1",
    "select_invariant_query_subspace",
]
