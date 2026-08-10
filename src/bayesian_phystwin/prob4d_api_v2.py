"""Stable BayesianPhysTwin bridge to :mod:`prob4d.api.v2`.

The bridge imports Prob4D lazily so the lightweight BayesianPhysTwin base
installation remains usable without Prob4D. New provider-v2 integrations use
this module instead of depending on Prob4D implementation-module layout.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

PROB4D_REQUIRED_API_VERSION = 2
PROB4D_REQUIRED_PROVIDER_API_VERSION = 2
PROB4D_REQUIRED_FACTOR_API_VERSION = 2
PROB4D_REQUIRED_PROJECT_ID = "github-repository-id:1295794737"


@dataclass(frozen=True, slots=True)
class Prob4DApiV2Compatibility:
    """Validated identity and version summary for an installed Prob4D API."""

    api_version: int
    provider_api_version: int
    provider_factor_api_version: int
    project_id: str
    canonical_repository: str
    frozen_artifact_repository: str


def _exact_integer(value: object, *, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ImportError(f"installed Prob4D {name} must be an integer")
    return value


def _nonempty_string(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ImportError(f"installed Prob4D {name} must be a nonempty string")
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
    "Prob4DApiV2Compatibility",
    "inspect_prob4d_api_v2",
    "load_claim_bearing_tree_sparse_prob4d",
]
