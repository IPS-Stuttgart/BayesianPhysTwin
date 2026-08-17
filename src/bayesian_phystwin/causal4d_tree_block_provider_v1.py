"""Versioned claim-bearing tree-block covariance surface for Causal4D.

This provider binds an exact factorized linear-query covariance to the admitted
Prob4D update, the strict tree-block result identity, and a caller-owned query
identity.  It does not materialize the complete coefficient covariance and does
not widen the stable root-package API.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Final

import numpy as np

from ._canonical_contracts import (
    frozen_finite_json_mapping,
    immutable_array,
    plain_json,
)
from .contracts.provider import (
    installed_distribution_revision,
    installed_distribution_version,
)
from .tree_block_claim_contract import validate_tree_block_result
from .tree_block_posterior_operator import (
    TREE_BLOCK_POSTERIOR_OPERATOR_VERSION,
    TreeBlockPosteriorOperatorV1,
)
from .tree_block_sparse_gauge_belief import (
    TREE_BLOCK_GAUGE_AWARE_RESULT_VERSION,
    TREE_BLOCK_POSTERIOR_COVARIANCE_VERSION,
)
from .tree_block_sparse_prob4d import (
    CLAIM_BEARING_TREE_BLOCK_PROB4D_VERSION,
    ClaimBearingTreeBlockProb4DUpdateV1,
)

CAUSAL4D_TREE_BLOCK_PROVIDER_API_VERSION: Final = 1
CAUSAL4D_TREE_BLOCK_PROVIDER_PACKAGE_VERSION: Final = "0.4.0"
CAUSAL4D_TREE_BLOCK_QUERY_COVARIANCE_SCHEMA: Final = (
    "bayesian_phystwin.causal4d_tree_block_query_covariance"
)
CAUSAL4D_TREE_BLOCK_QUERY_COVARIANCE_VERSION: Final = 1
CAUSAL4D_TREE_BLOCK_PROVIDER_CAPABILITIES: Final = (
    "claim_bearing_tree_block_update_validation",
    "strict_tree_block_admission_binding",
    "factorized_linear_query_covariance",
    "query_identity_binding",
    "immutable_query_covariance",
    "no_dense_covariance_materialization",
)
CAUSAL4D_TREE_BLOCK_PROVIDER_ARTIFACT_SCHEMA_VERSIONS: Final = {
    "ClaimBearingTreeBlockProb4DUpdate": CLAIM_BEARING_TREE_BLOCK_PROB4D_VERSION,
    "TreeBlockGaugeAwareBeliefResult": TREE_BLOCK_GAUGE_AWARE_RESULT_VERSION,
    "TreeBlockPosteriorCovariance": TREE_BLOCK_POSTERIOR_COVARIANCE_VERSION,
    "TreeBlockPosteriorOperator": TREE_BLOCK_POSTERIOR_OPERATOR_VERSION,
    "Causal4DTreeBlockQueryCovariance": (CAUSAL4D_TREE_BLOCK_QUERY_COVARIANCE_VERSION),
}
CAUSAL4D_TREE_BLOCK_PROVIDER_INFERENCE_ROLE: Final = (
    "claim-bearing tree-block posterior linear-query covariance"
)
CAUSAL4D_TREE_BLOCK_PROVIDER_COMPATIBILITY: Final = (
    "additive provider; causal4d belief providers v1 and v2 are unchanged"
)
CAUSAL4D_TREE_BLOCK_PROVIDER_RAW_COVARIANCE_CLAIM: Final = (
    "exact query of the admitted working Gauss-Newton/IRLS covariance; empirical "
    "calibration and target-side coverage remain separate gates"
)
CAUSAL4D_TREE_BLOCK_PROVIDER_BOUNDARY: Final = (
    "The provider establishes factor integrity, strict-admission lineage, query "
    "identity, and exact numerical covariance application. It does not establish "
    "observation competence, uncertainty calibration, physical-query benefit, "
    "intervention benefit, deployment safety, or state of the art."
)


def _require(condition: bool | np.bool_, message: str) -> None:
    if not bool(condition):
        raise ValueError(message)


def _sha256(value: object, *, name: str) -> str:
    if type(value) is not str:
        raise TypeError(f"{name} must be a string")
    result = str(value)
    _require(
        len(result) == 64
        and all(character in "0123456789abcdef" for character in result),
        f"{name} must be a lowercase SHA-256 digest",
    )
    return result


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(json.dumps(array.shape, separators=(",", ":")).encode("ascii"))
    digest.update(array.view(np.uint8))
    return digest.hexdigest()


def _canonical_id(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            plain_json(value),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _query_matrix(value: object, *, coefficient_dimension: int) -> np.ndarray:
    raw = np.asarray(value)
    _require(
        raw.dtype.kind in {"i", "u", "f"},
        "query_matrix must be real numeric",
    )
    query = np.asarray(raw, dtype=np.float64)
    _require(query.ndim == 2, "query_matrix must have two dimensions")
    _require(
        query.shape[1] == coefficient_dimension,
        "query_matrix coefficient dimension changed",
    )
    _require(query.shape[0] >= 1, "query_matrix must contain at least one row")
    _require(np.all(np.isfinite(query)), "query_matrix must be finite")
    return query


def _revalidate_update(
    update: ClaimBearingTreeBlockProb4DUpdateV1,
) -> ClaimBearingTreeBlockProb4DUpdateV1:
    if not isinstance(update, ClaimBearingTreeBlockProb4DUpdateV1):
        raise TypeError("update must be a ClaimBearingTreeBlockProb4DUpdateV1")
    validate_tree_block_result(update.result, require_strict_admission=True)
    rebuilt = ClaimBearingTreeBlockProb4DUpdateV1(
        result=update.result,
        observation_artifact_id=update.observation_artifact_id,
        linearization_artifact_id=update.linearization_artifact_id,
        provider_manifest_id=update.provider_manifest_id,
        calibration_artifact_ids=update.calibration_artifact_ids,
        runtime_revision_source=update.runtime_revision_source,
        runtime_revision_independently_verified=(
            update.runtime_revision_independently_verified
        ),
    )
    _require(rebuilt.admission_id == update.admission_id, "update admission ID changed")
    _require(rebuilt.update_id == update.update_id, "update identity changed")
    _require(
        rebuilt.tree_block_result_id == update.tree_block_result_id,
        "tree-block result identity changed",
    )
    return update


@dataclass(frozen=True, slots=True)
class Causal4DTreeBlockQueryCovarianceV1:
    """Immutable covariance of one registered Causal4D linear query."""

    update_id: str
    tree_block_result_id: str
    query_id: str
    query_matrix_sha256: str
    coefficient_dimension: int
    inference_admissible: bool
    inference_reason: str
    covariance: np.ndarray
    _result_id: str = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        for name in (
            "update_id",
            "tree_block_result_id",
            "query_id",
            "query_matrix_sha256",
        ):
            object.__setattr__(self, name, _sha256(getattr(self, name), name=name))
        if (
            type(self.coefficient_dimension) is not int
            or self.coefficient_dimension < 1
        ):
            raise ValueError("coefficient_dimension must be a positive integer")
        if type(self.inference_admissible) is not bool:
            raise TypeError("inference_admissible must be a bool")
        if type(self.inference_reason) is not str or not self.inference_reason:
            raise ValueError("inference_reason must be a nonempty string")
        raw = np.asarray(self.covariance)
        _require(
            raw.dtype.kind in {"i", "u", "f"},
            "covariance must be real numeric",
        )
        covariance = np.asarray(raw, dtype=np.float64)
        _require(
            covariance.ndim == 2
            and covariance.shape[0] == covariance.shape[1]
            and len(covariance) >= 1,
            "covariance must be a nonempty square matrix",
        )
        _require(np.all(np.isfinite(covariance)), "covariance must be finite")
        _require(
            np.allclose(covariance, covariance.T, atol=1e-10, rtol=1e-10),
            "covariance must be symmetric",
        )
        _require(
            np.min(np.linalg.eigvalsh(0.5 * (covariance + covariance.T))) >= -1e-9,
            "covariance must be positive semidefinite",
        )
        object.__setattr__(
            self,
            "covariance",
            immutable_array(0.5 * (covariance + covariance.T), dtype=np.float64),
        )
        object.__setattr__(self, "_result_id", _canonical_id(self.descriptor()))

    @property
    def schema(self) -> str:
        return CAUSAL4D_TREE_BLOCK_QUERY_COVARIANCE_SCHEMA

    @property
    def schema_version(self) -> int:
        return CAUSAL4D_TREE_BLOCK_QUERY_COVARIANCE_VERSION

    @property
    def query_row_count(self) -> int:
        return len(self.covariance)

    @property
    def result_id(self) -> str:
        return self._result_id

    def descriptor(self) -> Mapping[str, Any]:
        return frozen_finite_json_mapping(
            {
                "schema": self.schema,
                "schema_version": self.schema_version,
                "update_id": self.update_id,
                "tree_block_result_id": self.tree_block_result_id,
                "query_id": self.query_id,
                "query_matrix_sha256": self.query_matrix_sha256,
                "coefficient_dimension": self.coefficient_dimension,
                "query_row_count": self.query_row_count,
                "inference_admissible": self.inference_admissible,
                "inference_reason": self.inference_reason,
                "covariance_sha256": _array_sha256(self.covariance),
                "raw_covariance_claim": (
                    CAUSAL4D_TREE_BLOCK_PROVIDER_RAW_COVARIANCE_CLAIM
                ),
                "claim_boundary": CAUSAL4D_TREE_BLOCK_PROVIDER_BOUNDARY,
            },
            name="Causal4D tree-block query covariance descriptor",
        )


def evaluate_claim_bearing_tree_block_query(
    update: ClaimBearingTreeBlockProb4DUpdateV1,
    query_matrix: object,
    *,
    query_id: str,
) -> Causal4DTreeBlockQueryCovarianceV1:
    """Evaluate a registered linear query without dense covariance allocation."""

    admitted = _revalidate_update(update)
    registered_query_id = _sha256(query_id, name="query_id")
    query = _query_matrix(
        query_matrix,
        coefficient_dimension=admitted.result.covariance.dimension,
    )
    covariance = TreeBlockPosteriorOperatorV1(
        admitted.result.covariance
    ).linear_covariance(query)
    return Causal4DTreeBlockQueryCovarianceV1(
        update_id=admitted.update_id,
        tree_block_result_id=admitted.tree_block_result_id,
        query_id=registered_query_id,
        query_matrix_sha256=_array_sha256(query),
        coefficient_dimension=admitted.result.covariance.dimension,
        inference_admissible=admitted.inference_admissible,
        inference_reason=admitted.result.reason,
        covariance=covariance,
    )


def causal4d_tree_block_provider_manifest(
    *,
    provider_revision: str | None = None,
) -> dict[str, object]:
    """Return the additive claim-bearing covariance-query descriptor."""

    if provider_revision is not None and (
        type(provider_revision) is not str or not provider_revision
    ):
        raise ValueError("provider_revision must be a nonempty string")
    revision = (
        provider_revision
        or os.environ.get("BAYESIAN_PHYSTWIN_REVISION")
        or installed_distribution_revision("bayesian-phystwin")
        or "unversioned-install"
    )
    return {
        "provider_name": "bayesian-phystwin",
        "provider_version": installed_distribution_version(
            "bayesian-phystwin",
            fallback=CAUSAL4D_TREE_BLOCK_PROVIDER_PACKAGE_VERSION,
        ),
        "provider_revision": revision,
        "schema_version": CAUSAL4D_TREE_BLOCK_PROVIDER_API_VERSION,
        "capabilities": list(CAUSAL4D_TREE_BLOCK_PROVIDER_CAPABILITIES),
        "artifact_schema_versions": dict(
            CAUSAL4D_TREE_BLOCK_PROVIDER_ARTIFACT_SCHEMA_VERSIONS
        ),
        "metadata": {
            "provider_api": ("bayesian_phystwin.causal4d_tree_block_provider_v1"),
            "provider_api_version": CAUSAL4D_TREE_BLOCK_PROVIDER_API_VERSION,
            "inference_role": CAUSAL4D_TREE_BLOCK_PROVIDER_INFERENCE_ROLE,
            "compatibility": CAUSAL4D_TREE_BLOCK_PROVIDER_COMPATIBILITY,
            "raw_covariance_claim": (CAUSAL4D_TREE_BLOCK_PROVIDER_RAW_COVARIANCE_CLAIM),
            "claim_boundary": CAUSAL4D_TREE_BLOCK_PROVIDER_BOUNDARY,
        },
    }


__all__ = [
    "CAUSAL4D_TREE_BLOCK_PROVIDER_API_VERSION",
    "CAUSAL4D_TREE_BLOCK_PROVIDER_ARTIFACT_SCHEMA_VERSIONS",
    "CAUSAL4D_TREE_BLOCK_PROVIDER_BOUNDARY",
    "CAUSAL4D_TREE_BLOCK_PROVIDER_CAPABILITIES",
    "CAUSAL4D_TREE_BLOCK_PROVIDER_COMPATIBILITY",
    "CAUSAL4D_TREE_BLOCK_PROVIDER_INFERENCE_ROLE",
    "CAUSAL4D_TREE_BLOCK_PROVIDER_PACKAGE_VERSION",
    "CAUSAL4D_TREE_BLOCK_PROVIDER_RAW_COVARIANCE_CLAIM",
    "CAUSAL4D_TREE_BLOCK_QUERY_COVARIANCE_SCHEMA",
    "CAUSAL4D_TREE_BLOCK_QUERY_COVARIANCE_VERSION",
    "Causal4DTreeBlockQueryCovarianceV1",
    "ClaimBearingTreeBlockProb4DUpdateV1",
    "causal4d_tree_block_provider_manifest",
    "evaluate_claim_bearing_tree_block_query",
]
