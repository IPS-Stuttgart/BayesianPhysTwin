"""Stable BayesianPhysTwin bridge to :mod:`prob4d.api.v2`.

The bridge imports Prob4D lazily so the lightweight BayesianPhysTwin base
installation remains usable without Prob4D. New provider-v2 integrations use
this module instead of depending on Prob4D implementation-module layout.
"""

from __future__ import annotations

import importlib
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

PROB4D_REQUIRED_API_VERSION = 2
PROB4D_REQUIRED_PROVIDER_API_VERSION = 2
PROB4D_REQUIRED_FACTOR_API_VERSION = 2
PROB4D_REQUIRED_PROJECT_ID = "github-repository-id:1295794737"
PROB4D_REQUIRED_PROVIDER_V2_CONTRACT_BUNDLE = "prob4d.provider_v2_factors.v1"
PROB4D_REQUIRED_PROVIDER_V2_CONTRACT_BUNDLE_SHA256 = (
    "fe0374f46319287e3709497de9cbb73f7497286cf4f157f246096f2c352e4446"
)
PROB4D_REQUIRED_PROVIDER_V2_STACK_SEMANTIC_SHA256 = (
    "58621710b5b22a64163c47b4756f200cea13e56491d85a3852af96ec1cb0f4fb"
)
PROB4D_REQUIRED_PROVIDER_V2_MINIMAL_PRIOR_ID = (
    "ddb97db5c953635eaa881c4d1b1fbe3e9508a72d0c0fb13a5d2a7f5727021dee"
)
PROB4D_REQUIRED_PROVIDER_V2_VALID_VECTOR_COUNT = 1
PROB4D_REQUIRED_PROVIDER_V2_INVALID_VECTOR_COUNT = 10
PROB4D_REQUIRED_PROVIDER_V2_NUMERICAL_ATOL = 1e-12
PROB4D_REQUIRED_PROVIDER_V2_NUMERICAL_RTOL = 1e-10


@dataclass(frozen=True, slots=True)
class Prob4DApiV2Compatibility:
    """Validated identity and version summary for an installed Prob4D API."""

    api_version: int
    provider_api_version: int
    provider_factor_api_version: int
    project_id: str
    canonical_repository: str
    frozen_artifact_repository: str


@dataclass(frozen=True, slots=True)
class Prob4DProviderV2ContractCompatibility:
    """Validated software-conformance summary for the installed corpus."""

    bundle_name: str
    bundle_sha256: str
    valid_vector_count: int
    invalid_vector_count: int
    minimal_prior_id: str
    minimal_stack_semantic_sha256: str
    numerical_atol: float
    numerical_rtol: float


def _exact_integer(value: object, *, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ImportError(f"installed Prob4D {name} must be an integer")
    return value


def _nonempty_string(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ImportError(f"installed Prob4D {name} must be a nonempty string")
    return value


def _mapping(value: object, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ImportError(f"installed Prob4D {name} must be a mapping")
    return value


def _finite_float(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ImportError(f"installed Prob4D {name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ImportError(f"installed Prob4D {name} must be a finite number")
    return result


def _required_callable(module: ModuleType, name: str) -> Any:
    value = getattr(module, name, None)
    if not callable(value):
        raise ImportError(f"installed Prob4D contract corpus lacks callable {name}")
    return value


def _load_prob4d_api_v2() -> ModuleType:
    try:
        api = importlib.import_module("prob4d.api.v2")
    except ImportError as error:
        raise ImportError(
            "claim-bearing Prob4D integration requires a compatible installation "
            "that exposes prob4d.api.v2"
        ) from error
    api_version = _exact_integer(
        getattr(api, "API_VERSION", None),
        name="API_VERSION",
    )
    provider_api_version = _exact_integer(
        getattr(api, "PROVIDER_API_VERSION", None),
        name="PROVIDER_API_VERSION",
    )
    factor_api_version = _exact_integer(
        getattr(api, "PROVIDER_FACTOR_API_VERSION", None),
        name="PROVIDER_FACTOR_API_VERSION",
    )
    if api_version != PROB4D_REQUIRED_API_VERSION:
        raise ImportError("installed Prob4D exposes an incompatible stable API version")
    if provider_api_version != PROB4D_REQUIRED_PROVIDER_API_VERSION:
        raise ImportError(
            "installed Prob4D exposes an incompatible provider API version"
        )
    if factor_api_version != PROB4D_REQUIRED_FACTOR_API_VERSION:
        raise ImportError("installed Prob4D exposes an incompatible factor API version")
    return api


def inspect_prob4d_api_v2() -> Prob4DApiV2Compatibility:
    """Validate and summarize the installed stable Prob4D provider-v2 API."""

    api = _load_prob4d_api_v2()
    identity_loader = getattr(api, "prob4d_project_identity", None)
    identity_validator = getattr(api, "validate_prob4d_project_identity", None)
    if not callable(identity_loader) or not callable(identity_validator):
        raise ImportError("installed Prob4D API v2 lacks project-identity validation")
    identity = identity_validator(identity_loader())
    if not isinstance(identity, dict):
        raise ImportError("installed Prob4D project identity must be a dictionary")
    project_id = _nonempty_string(identity.get("project_id"), name="project_id")
    if project_id != PROB4D_REQUIRED_PROJECT_ID:
        raise ImportError(
            "installed Prob4D project identity is not the supported project"
        )
    return Prob4DApiV2Compatibility(
        api_version=PROB4D_REQUIRED_API_VERSION,
        provider_api_version=PROB4D_REQUIRED_PROVIDER_API_VERSION,
        provider_factor_api_version=PROB4D_REQUIRED_FACTOR_API_VERSION,
        project_id=project_id,
        canonical_repository=_nonempty_string(
            identity.get("canonical_repository"),
            name="canonical_repository",
        ),
        frozen_artifact_repository=_nonempty_string(
            identity.get("frozen_artifact_repository"),
            name="frozen_artifact_repository",
        ),
    )


def inspect_prob4d_provider_v2_contract() -> Prob4DProviderV2ContractCompatibility:
    """Verify the installed provider-v2 corpus and its portable semantic identity.

    This is software interoperability evidence only. BayesianPhysTwin continues
    to independently validate every claim-bearing tree-sparse artifact before a
    physical update.
    """

    inspect_prob4d_api_v2()
    try:
        module = importlib.import_module("prob4d.provider_v2_contract_bundle")
    except ImportError as error:
        raise ImportError(
            "provider-v2 conformance inspection requires an installed Prob4D "
            "contract corpus"
        ) from error

    manifest_loader = _required_callable(
        module,
        "provider_v2_contract_bundle_manifest",
    )
    schema_loader = _required_callable(module, "provider_v2_contract_schema")
    vector_loader = _required_callable(module, "provider_v2_contract_vector")
    materializer = _required_callable(
        module,
        "materialize_provider_v2_contract_vector",
    )
    validator = _required_callable(
        module,
        "validate_provider_v2_contract_materialization",
    )
    semantic_hasher = _required_callable(
        module,
        "provider_v2_contract_stack_semantic_sha256",
    )
    invalid_loader = _required_callable(
        module,
        "invalid_provider_v2_contract_vectors",
    )

    manifest = _mapping(manifest_loader(), name="contract bundle manifest")
    bundle_name = _nonempty_string(
        manifest.get("bundle_name"),
        name="contract bundle_name",
    )
    if bundle_name != PROB4D_REQUIRED_PROVIDER_V2_CONTRACT_BUNDLE:
        raise ImportError("installed Prob4D provider-v2 contract bundle changed name")
    bundle_sha256 = _nonempty_string(
        manifest.get("bundle_sha256"),
        name="contract bundle_sha256",
    )
    if bundle_sha256 != PROB4D_REQUIRED_PROVIDER_V2_CONTRACT_BUNDLE_SHA256:
        raise ImportError("installed Prob4D provider-v2 contract bundle changed bytes")

    schema = _mapping(schema_loader(), name="contract schema")
    raw_vectors = schema.get("valid_vectors")
    if not isinstance(raw_vectors, list) or any(
        not isinstance(value, str) for value in raw_vectors
    ):
        raise ImportError("installed Prob4D contract valid_vectors must be strings")
    if raw_vectors != ["minimal"]:
        raise ImportError("installed Prob4D contract valid-vector roster changed")

    vector = vector_loader("minimal")
    materialization = materializer(vector)
    validator(vector, materialization)
    prior = getattr(materialization, "gauge_tree_prior", None)
    stack = getattr(materialization, "tree_sparse_stack", None)
    minimal_prior_id = _nonempty_string(
        getattr(prior, "prior_id", None),
        name="contract minimal prior_id",
    )
    if minimal_prior_id != PROB4D_REQUIRED_PROVIDER_V2_MINIMAL_PRIOR_ID:
        raise ImportError("installed Prob4D contract minimal prior identity changed")
    semantic_sha256 = _nonempty_string(
        semantic_hasher(stack),
        name="contract minimal stack semantic SHA-256",
    )
    if semantic_sha256 != PROB4D_REQUIRED_PROVIDER_V2_STACK_SEMANTIC_SHA256:
        raise ImportError("installed Prob4D contract stack semantic identity changed")

    invalid_vectors = invalid_loader()
    if not isinstance(invalid_vectors, tuple):
        raise ImportError("installed Prob4D invalid contract vectors must be a tuple")
    invalid_count = len(invalid_vectors)
    if invalid_count != PROB4D_REQUIRED_PROVIDER_V2_INVALID_VECTOR_COUNT:
        raise ImportError("installed Prob4D invalid-vector roster changed")

    numerical_atol = _finite_float(
        getattr(module, "PROVIDER_V2_CONTRACT_NUMERICAL_ATOL", None),
        name="contract numerical_atol",
    )
    numerical_rtol = _finite_float(
        getattr(module, "PROVIDER_V2_CONTRACT_NUMERICAL_RTOL", None),
        name="contract numerical_rtol",
    )
    if numerical_atol != PROB4D_REQUIRED_PROVIDER_V2_NUMERICAL_ATOL:
        raise ImportError("installed Prob4D contract numerical atol changed")
    if numerical_rtol != PROB4D_REQUIRED_PROVIDER_V2_NUMERICAL_RTOL:
        raise ImportError("installed Prob4D contract numerical rtol changed")

    return Prob4DProviderV2ContractCompatibility(
        bundle_name=bundle_name,
        bundle_sha256=bundle_sha256,
        valid_vector_count=len(raw_vectors),
        invalid_vector_count=invalid_count,
        minimal_prior_id=minimal_prior_id,
        minimal_stack_semantic_sha256=semantic_sha256,
        numerical_atol=numerical_atol,
        numerical_rtol=numerical_rtol,
    )


def load_claim_bearing_tree_sparse_prob4d(
    envelope_path: str | Path,
) -> Any:
    """Load one claim-bearing tree-sparse artifact through `prob4d.api.v2`."""

    api = _load_prob4d_api_v2()
    loader = getattr(api, "load_claim_bearing_tree_sparse_observation", None)
    if not callable(loader):
        raise ImportError(
            "installed Prob4D API v2 lacks the claim-bearing tree-sparse loader"
        )
    return loader(Path(envelope_path))


__all__ = [
    "PROB4D_REQUIRED_API_VERSION",
    "PROB4D_REQUIRED_FACTOR_API_VERSION",
    "PROB4D_REQUIRED_PROJECT_ID",
    "PROB4D_REQUIRED_PROVIDER_API_VERSION",
    "PROB4D_REQUIRED_PROVIDER_V2_CONTRACT_BUNDLE",
    "PROB4D_REQUIRED_PROVIDER_V2_CONTRACT_BUNDLE_SHA256",
    "PROB4D_REQUIRED_PROVIDER_V2_INVALID_VECTOR_COUNT",
    "PROB4D_REQUIRED_PROVIDER_V2_MINIMAL_PRIOR_ID",
    "PROB4D_REQUIRED_PROVIDER_V2_NUMERICAL_ATOL",
    "PROB4D_REQUIRED_PROVIDER_V2_NUMERICAL_RTOL",
    "PROB4D_REQUIRED_PROVIDER_V2_STACK_SEMANTIC_SHA256",
    "PROB4D_REQUIRED_PROVIDER_V2_VALID_VECTOR_COUNT",
    "Prob4DApiV2Compatibility",
    "Prob4DProviderV2ContractCompatibility",
    "inspect_prob4d_api_v2",
    "inspect_prob4d_provider_v2_contract",
    "load_claim_bearing_tree_sparse_prob4d",
]
