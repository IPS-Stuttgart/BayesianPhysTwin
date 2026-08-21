"""Query identifiability modulo a declared nuisance design.

For a whitened local model ``r = X z + N nu + epsilon`` and registered
linearized query ``delta_q = B z``, the certificate residualizes the physical
design against the nuisance column space, ``A = P_perp(N) X``. The query is
identifiable from the residualized observation exactly when
``ker(A)`` is contained in ``ker(B)``, equivalently when a linear operator
``M`` exists with ``B = M A``.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from numbers import Real
from typing import Any, Final, cast

import numpy as np

from ._canonical_contracts import (
    frozen_finite_json_mapping,
    immutable_array,
    literal_lower_hex,
    plain_json,
)
from ._portable_contracts import content_id

QUERY_IDENTIFIABILITY_CERTIFICATE_SCHEMA: Final = (
    "bayesian_phystwin.query_identifiability_certificate"
)
QUERY_IDENTIFIABILITY_CERTIFICATE_VERSION: Final = 2
QUERY_IDENTIFIABILITY_CERTIFICATE_SEMANTICS: Final = (
    "query-identifiability-modulo-declared-nuisance-by-kernel-inclusion-v2"
)
QUERY_IDENTIFIABILITY_CERTIFICATE_CLAIM_BOUNDARY: Final = (
    "Local linear query identifiability under the supplied whitening, physical "
    "design, nuisance design, latent coordinates, query map, and numerical "
    "tolerances only. The certificate does not prove a unique physical cause, "
    "global nonlinear identifiability, provider competence, uncertainty "
    "calibration, unseen-object transfer, deployment safety, or Causal4D benefit."
)


class QueryIdentifiabilityStatus(str, Enum):
    """Numerical outcome of the kernel-inclusion test."""

    IDENTIFIABLE = "identifiable"
    NONIDENTIFIABLE = "nonidentifiable"
    TRIVIAL_QUERY = "trivial_query"


def _real_float64_matrix(value: object, *, name: str) -> np.ndarray:
    raw = np.asarray(value)
    if raw.dtype.kind not in "iuf":
        raise ValueError(f"{name} must contain real numeric values")
    matrix = np.ascontiguousarray(raw, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError(f"{name} must be a matrix")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must be finite")
    return matrix


def _immutable_float64(value: object) -> np.ndarray:
    return cast(np.ndarray, immutable_array(value, dtype=np.float64))


def _finite_nonnegative(value: object, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite nonnegative real number")
    result = float(value)
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be a finite nonnegative real number")
    return result


def _array_record(value: np.ndarray) -> dict[str, object]:
    return {
        "shape": list(value.shape),
        "dtype": value.dtype.str,
        "sha256": hashlib.sha256(value.tobytes(order="C")).hexdigest(),
    }


def _singular_value_tolerance(
    singular_values: np.ndarray,
    *,
    relative: float,
    absolute: float,
) -> float:
    scale = float(singular_values[0]) if len(singular_values) else 0.0
    return max(absolute, relative * scale)


def _svd_rank(singular_values: np.ndarray, tolerance: float) -> int:
    return int(np.count_nonzero(singular_values > tolerance))


@dataclass(frozen=True, slots=True)
class QueryIdentifiabilityCertificateV2:
    """Content-addressed certificate for one local physical query.

    The supplied physical and nuisance matrices must already use the same
    whitened observation coordinates. Columns of the physical design are the
    declared reachable latent coordinates; columns of the nuisance design are
    every competing nuisance direction against which the physical query is to be
    distinguished.
    """

    physical_response_id: str
    observation_mapping_id: str
    nuisance_design_id: str
    query_id: str
    whitened_physical_design: np.ndarray
    whitened_nuisance_design: np.ndarray
    query_map: np.ndarray
    relative_rank_tolerance: float = 1e-10
    absolute_rank_tolerance: float = 1e-12
    identifiability_tolerance: float = 1e-8
    metadata: Mapping[str, Any] = field(default_factory=dict)
    artifact_id: str | None = None

    nuisance_projector: np.ndarray = field(init=False, repr=False)
    residualized_physical_design: np.ndarray = field(init=False, repr=False)
    factor_operator: np.ndarray = field(init=False, repr=False)
    factorization_residual: np.ndarray = field(init=False, repr=False)
    nuisance_singular_values: np.ndarray = field(init=False, repr=False)
    physical_singular_values: np.ndarray = field(init=False, repr=False)
    query_singular_values: np.ndarray = field(init=False, repr=False)
    null_query_singular_values: np.ndarray = field(init=False, repr=False)
    nuisance_rank: int = field(init=False)
    physical_rank: int = field(init=False)
    query_rank: int = field(init=False)
    physical_nullity: int = field(init=False)
    rank_increment: int = field(init=False)
    augmented_rank: int = field(init=False)
    physical_rank_tolerance: float = field(init=False)
    nuisance_rank_tolerance: float = field(init=False)
    query_rank_tolerance: float = field(init=False)
    factorization_residual_frobenius: float = field(init=False)
    factorization_residual_spectral: float = field(init=False)
    normalized_factorization_residual: float = field(init=False)
    identifiable_query_energy_fraction: float = field(init=False)
    status: QueryIdentifiabilityStatus = field(init=False)

    def __post_init__(self) -> None:
        for name in (
            "physical_response_id",
            "observation_mapping_id",
            "nuisance_design_id",
            "query_id",
        ):
            object.__setattr__(
                self,
                name,
                literal_lower_hex(getattr(self, name), name=name, lengths={64}),
            )

        physical = _real_float64_matrix(
            self.whitened_physical_design,
            name="whitened_physical_design",
        )
        nuisance = _real_float64_matrix(
            self.whitened_nuisance_design,
            name="whitened_nuisance_design",
        )
        query = _real_float64_matrix(self.query_map, name="query_map")
        if physical.shape[0] == 0 or physical.shape[1] == 0:
            raise ValueError("whitened_physical_design must have nonzero dimensions")
        if nuisance.shape[0] != physical.shape[0]:
            raise ValueError(
                "whitened_nuisance_design must share the observation row count"
            )
        if query.shape[0] == 0 or query.shape[1] != physical.shape[1]:
            raise ValueError(
                "query_map must have nonzero rows and one column per latent coordinate"
            )

        relative = _finite_nonnegative(
            self.relative_rank_tolerance,
            name="relative_rank_tolerance",
        )
        absolute = _finite_nonnegative(
            self.absolute_rank_tolerance,
            name="absolute_rank_tolerance",
        )
        identifiability = _finite_nonnegative(
            self.identifiability_tolerance,
            name="identifiability_tolerance",
        )
        if relative == absolute == 0.0:
            raise ValueError("at least one rank tolerance must be positive")
        if identifiability == 0.0:
            raise ValueError("identifiability_tolerance must be positive")

        row_count, latent_count = physical.shape
        if nuisance.shape[1]:
            nuisance_u, nuisance_s, _ = np.linalg.svd(
                nuisance,
                full_matrices=False,
            )
            nuisance_tol = _singular_value_tolerance(
                nuisance_s,
                relative=relative,
                absolute=absolute,
            )
            nuisance_rank = _svd_rank(nuisance_s, nuisance_tol)
            nuisance_basis = nuisance_u[:, :nuisance_rank]
            nuisance_projector = nuisance_basis @ nuisance_basis.T
        else:
            nuisance_s = np.zeros(0, dtype=np.float64)
            nuisance_tol = absolute
            nuisance_rank = 0
            nuisance_projector = np.zeros((row_count, row_count), dtype=np.float64)

        residualized = physical - nuisance_projector @ physical
        physical_u, physical_s, physical_vh = np.linalg.svd(
            residualized,
            full_matrices=True,
        )
        physical_tol = _singular_value_tolerance(
            physical_s,
            relative=relative,
            absolute=absolute,
        )
        physical_rank = _svd_rank(physical_s, physical_tol)
        physical_nullity = latent_count - physical_rank

        if physical_rank:
            physical_u_rank = physical_u[:, :physical_rank]
            physical_v_rank = physical_vh[:physical_rank, :].T
            inverse_singular_values = 1.0 / physical_s[:physical_rank]
            pseudoinverse = (
                physical_v_rank
                @ np.diag(inverse_singular_values)
                @ physical_u_rank.T
            )
        else:
            pseudoinverse = np.zeros((latent_count, row_count), dtype=np.float64)
        factor_operator = query @ pseudoinverse
        factorization_residual = query - factor_operator @ residualized

        query_s = np.linalg.svd(query, compute_uv=False)
        query_tol = _singular_value_tolerance(
            query_s,
            relative=relative,
            absolute=absolute,
        )
        query_rank = _svd_rank(query_s, query_tol)
        query_frobenius = float(np.linalg.norm(query, ord="fro"))

        null_basis = physical_vh[physical_rank:, :].T
        null_query = query @ null_basis
        null_query_s = np.linalg.svd(null_query, compute_uv=False)
        residual_frobenius = float(
            np.linalg.norm(factorization_residual, ord="fro")
        )
        residual_spectral = float(null_query_s[0]) if len(null_query_s) else 0.0
        residual_bound = absolute + identifiability * query_frobenius
        rank_increment = int(np.count_nonzero(null_query_s > residual_bound))
        augmented_rank = physical_rank + rank_increment
        normalized_residual = residual_frobenius / max(
            query_frobenius,
            np.finfo(np.float64).tiny,
        )
        if query_frobenius == 0.0:
            status = QueryIdentifiabilityStatus.TRIVIAL_QUERY
            energy_fraction = 1.0
        else:
            status = (
                QueryIdentifiabilityStatus.IDENTIFIABLE
                if residual_frobenius <= residual_bound
                else QueryIdentifiabilityStatus.NONIDENTIFIABLE
            )
            energy_fraction = float(
                np.clip(
                    1.0 - (residual_frobenius**2 / query_frobenius**2),
                    0.0,
                    1.0,
                )
            )

        metadata = frozen_finite_json_mapping(
            self.metadata,
            name="query identifiability certificate metadata",
        )
        for name, value in (
            ("whitened_physical_design", physical),
            ("whitened_nuisance_design", nuisance),
            ("query_map", query),
            ("nuisance_projector", nuisance_projector),
            ("residualized_physical_design", residualized),
            ("factor_operator", factor_operator),
            ("factorization_residual", factorization_residual),
            ("nuisance_singular_values", nuisance_s),
            ("physical_singular_values", physical_s),
            ("query_singular_values", query_s),
            ("null_query_singular_values", null_query_s),
        ):
            object.__setattr__(self, name, _immutable_float64(value))
        for name, value in (
            ("relative_rank_tolerance", relative),
            ("absolute_rank_tolerance", absolute),
            ("identifiability_tolerance", identifiability),
            ("physical_rank_tolerance", physical_tol),
            ("nuisance_rank_tolerance", nuisance_tol),
            ("query_rank_tolerance", query_tol),
            ("factorization_residual_frobenius", residual_frobenius),
            ("factorization_residual_spectral", residual_spectral),
            ("normalized_factorization_residual", normalized_residual),
            ("identifiable_query_energy_fraction", energy_fraction),
        ):
            object.__setattr__(self, name, value)
        for name, value in (
            ("nuisance_rank", nuisance_rank),
            ("physical_rank", physical_rank),
            ("query_rank", query_rank),
            ("physical_nullity", physical_nullity),
            ("rank_increment", rank_increment),
            ("augmented_rank", augmented_rank),
        ):
            object.__setattr__(self, name, value)
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "metadata", metadata)

        expected_id = cast(str, content_id(self.descriptor()))
        supplied_id = self.artifact_id
        if supplied_id is not None:
            supplied_id = cast(
                str,
                literal_lower_hex(
                    supplied_id,
                    name="artifact_id",
                    lengths={64},
                ),
            )
            if supplied_id != expected_id:
                raise ValueError(
                    "query identifiability certificate artifact_id does not match content"
                )
        object.__setattr__(self, "artifact_id", expected_id)

    @property
    def identifiable(self) -> bool:
        return self.status is not QueryIdentifiabilityStatus.NONIDENTIFIABLE

    @property
    def nontrivially_identifiable(self) -> bool:
        return self.status is QueryIdentifiabilityStatus.IDENTIFIABLE

    @property
    def observation_dimension(self) -> int:
        return int(self.whitened_physical_design.shape[0])

    @property
    def latent_dimension(self) -> int:
        return int(self.whitened_physical_design.shape[1])

    @property
    def nuisance_dimension(self) -> int:
        return int(self.whitened_nuisance_design.shape[1])

    @property
    def query_dimension(self) -> int:
        return int(self.query_map.shape[0])

    def arrays(self) -> Mapping[str, np.ndarray]:
        """Return immutable input and derived arrays bound by the certificate."""

        return {
            "whitened_physical_design": self.whitened_physical_design,
            "whitened_nuisance_design": self.whitened_nuisance_design,
            "query_map": self.query_map,
            "nuisance_projector": self.nuisance_projector,
            "residualized_physical_design": self.residualized_physical_design,
            "factor_operator": self.factor_operator,
            "factorization_residual": self.factorization_residual,
            "nuisance_singular_values": self.nuisance_singular_values,
            "physical_singular_values": self.physical_singular_values,
            "query_singular_values": self.query_singular_values,
            "null_query_singular_values": self.null_query_singular_values,
        }

    def descriptor(self) -> dict[str, object]:
        return {
            "schema": QUERY_IDENTIFIABILITY_CERTIFICATE_SCHEMA,
            "schema_version": QUERY_IDENTIFIABILITY_CERTIFICATE_VERSION,
            "semantics": QUERY_IDENTIFIABILITY_CERTIFICATE_SEMANTICS,
            "physical_response_id": self.physical_response_id,
            "observation_mapping_id": self.observation_mapping_id,
            "nuisance_design_id": self.nuisance_design_id,
            "query_id": self.query_id,
            "whitened_physical_design": _array_record(
                self.whitened_physical_design
            ),
            "whitened_nuisance_design": _array_record(
                self.whitened_nuisance_design
            ),
            "query_map": _array_record(self.query_map),
            "nuisance_projector": _array_record(self.nuisance_projector),
            "residualized_physical_design": _array_record(
                self.residualized_physical_design
            ),
            "factor_operator": _array_record(self.factor_operator),
            "factorization_residual": _array_record(self.factorization_residual),
            "nuisance_singular_values": _array_record(
                self.nuisance_singular_values
            ),
            "physical_singular_values": _array_record(
                self.physical_singular_values
            ),
            "query_singular_values": _array_record(self.query_singular_values),
            "null_query_singular_values": _array_record(
                self.null_query_singular_values
            ),
            "relative_rank_tolerance": self.relative_rank_tolerance,
            "absolute_rank_tolerance": self.absolute_rank_tolerance,
            "identifiability_tolerance": self.identifiability_tolerance,
            "physical_rank_tolerance": self.physical_rank_tolerance,
            "nuisance_rank_tolerance": self.nuisance_rank_tolerance,
            "query_rank_tolerance": self.query_rank_tolerance,
            "nuisance_rank": self.nuisance_rank,
            "physical_rank": self.physical_rank,
            "query_rank": self.query_rank,
            "physical_nullity": self.physical_nullity,
            "rank_increment": self.rank_increment,
            "augmented_rank": self.augmented_rank,
            "factorization_residual_frobenius": (
                self.factorization_residual_frobenius
            ),
            "factorization_residual_spectral": (
                self.factorization_residual_spectral
            ),
            "normalized_factorization_residual": (
                self.normalized_factorization_residual
            ),
            "identifiable_query_energy_fraction": (
                self.identifiable_query_energy_fraction
            ),
            "status": self.status.value,
            "metadata": plain_json(self.metadata),
            "claim_boundary": QUERY_IDENTIFIABILITY_CERTIFICATE_CLAIM_BOUNDARY,
        }

    def summary(self) -> dict[str, object]:
        return {
            "schema": QUERY_IDENTIFIABILITY_CERTIFICATE_SCHEMA,
            "schema_version": QUERY_IDENTIFIABILITY_CERTIFICATE_VERSION,
            "artifact_id": self.artifact_id,
            "status": self.status.value,
            "identifiable": self.identifiable,
            "nontrivially_identifiable": self.nontrivially_identifiable,
            "observation_dimension": self.observation_dimension,
            "latent_dimension": self.latent_dimension,
            "nuisance_dimension": self.nuisance_dimension,
            "query_dimension": self.query_dimension,
            "nuisance_rank": self.nuisance_rank,
            "physical_rank": self.physical_rank,
            "query_rank": self.query_rank,
            "physical_nullity": self.physical_nullity,
            "rank_increment": self.rank_increment,
            "augmented_rank": self.augmented_rank,
            "normalized_factorization_residual": (
                self.normalized_factorization_residual
            ),
            "identifiable_query_energy_fraction": (
                self.identifiable_query_energy_fraction
            ),
            "claim_boundary": QUERY_IDENTIFIABILITY_CERTIFICATE_CLAIM_BOUNDARY,
        }

    def to_record(self) -> dict[str, object]:
        return {**self.descriptor(), "artifact_id": self.artifact_id}


__all__ = [
    "QUERY_IDENTIFIABILITY_CERTIFICATE_CLAIM_BOUNDARY",
    "QUERY_IDENTIFIABILITY_CERTIFICATE_SCHEMA",
    "QUERY_IDENTIFIABILITY_CERTIFICATE_SEMANTICS",
    "QUERY_IDENTIFIABILITY_CERTIFICATE_VERSION",
    "QueryIdentifiabilityCertificateV2",
    "QueryIdentifiabilityStatus",
]
