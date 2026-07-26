"""Typed Bayesian-PhysTwin provider contract for Causal4D.

Version 2 keeps all dependencies on the released PhysTwin implementation inside
Bayesian-PhysTwin. Causal4D receives validated immutable cases, spring graphs,
controller layouts, and a replay protocol instead of importing implementation
modules directly.
"""

from __future__ import annotations

import hashlib
import json
import os
from importlib.metadata import PackageNotFoundError, distribution, version

from . import causal4d_provider_v1 as _v1
from ._causal4d_provider_v2_case import (
    build_phystwin_spring_graph,
    controller_hand_count,
    infer_controller_groups,
    load_official_phystwin_case,
    released_controller_layout,
)
from ._causal4d_provider_v2_replay import (
    FIXED_INITIAL_STD_M,
    FIXED_INLIER_PRIOR,
    FIXED_OBSERVATION_STD_M,
    FIXED_OUTLIER_VARIANCE_MULTIPLIER,
    FIXED_PROCESS_STD_M,
    OfficialPhysTwinReplayProvider,
    PhysTwinReplayProvider,
    create_official_case_replay_provider,
    create_official_replay_provider,
    robust_random_walk_endpoint,
)
from ._causal4d_provider_v2_types import (
    PhysTwinCase,
    PhysTwinControllerLayout,
    PhysTwinSpringGraph,
    PhysTwinSpringGraphConfig,
)

CAUSAL4D_PROVIDER_API_VERSION = 2
try:
    CAUSAL4D_PROVIDER_PACKAGE_VERSION = version("bayesian-phystwin")
except PackageNotFoundError:
    CAUSAL4D_PROVIDER_PACKAGE_VERSION = "0+unknown"

CAUSAL4D_PROVIDER_CAPABILITIES = tuple(
    sorted(
        set(_v1.CAUSAL4D_PROVIDER_CAPABILITIES)
        | {
            "official_phystwin_case_v2",
            "phystwin_spring_graph_v2",
            "released_controller_layout_v2",
            "robust_random_walk_endpoint",
        }
    )
)
CAUSAL4D_ARTIFACT_SCHEMA_VERSIONS = {
    **_v1.CAUSAL4D_ARTIFACT_SCHEMA_VERSIONS,
    "PhysTwinCase": 2,
    "PhysTwinSpringGraph": 2,
}
CAUSAL4D_PROVIDER_CONTRACT = {
    "provider_api": "bayesian_phystwin.causal4d_provider_v2",
    "provider_api_version": CAUSAL4D_PROVIDER_API_VERSION,
    "artifacts": {
        "PhysTwinCase": {
            "version": 2,
            "coordinates": "metres",
            "object_points": "float32[T,N,3]",
            "controller_points": "float32[T,C,3]",
            "visibility": "bool[T,N]",
            "motion_validity": "bool[T,N]",
        },
        "PhysTwinControllerLayout": {
            "version": 2,
            "group_ids": "int32[C] contiguous from zero",
        },
        "PhysTwinSpringGraph": {
            "version": 2,
            "vertices": "float32[V,3] metres",
            "springs": "int32[S,2]",
            "rest_lengths": "float32[S] metres",
            "masses": "float32[V] kilograms",
        },
    },
    "operations": [
        "build_phystwin_spring_graph",
        "create_official_case_replay_provider",
        "load_official_phystwin_case",
        "released_controller_layout",
        "robust_random_walk_endpoint",
    ],
}
CAUSAL4D_PROVIDER_CONTRACT_SHA256 = hashlib.sha256(
    json.dumps(
        CAUSAL4D_PROVIDER_CONTRACT,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
).hexdigest()


def _installed_provider_revision() -> str | None:
    try:
        direct_url = distribution("bayesian-phystwin").read_text("direct_url.json")
    except PackageNotFoundError:
        return None
    if not direct_url:
        return None
    try:
        payload = json.loads(direct_url)
    except (TypeError, json.JSONDecodeError):
        return None
    commit_id = payload.get("vcs_info", {}).get("commit_id")
    return str(commit_id) if commit_id else None


def causal4d_provider_manifest(
    *, provider_revision: str | None = None
) -> dict[str, object]:
    """Return the semantic provider descriptor consumed by Causal4D."""

    revision = (
        provider_revision
        or os.environ.get("BAYESIAN_PHYSTWIN_REVISION")
        or _installed_provider_revision()
        or "unversioned-install"
    )
    return {
        "provider_name": "bayesian-phystwin",
        "provider_version": CAUSAL4D_PROVIDER_PACKAGE_VERSION,
        "provider_revision": revision,
        "schema_version": CAUSAL4D_PROVIDER_API_VERSION,
        "capabilities": list(CAUSAL4D_PROVIDER_CAPABILITIES),
        "artifact_schema_versions": dict(CAUSAL4D_ARTIFACT_SCHEMA_VERSIONS),
        "metadata": {
            "contract_sha256": CAUSAL4D_PROVIDER_CONTRACT_SHA256,
            "provider_api": "bayesian_phystwin.causal4d_provider_v2",
            "provider_api_version": CAUSAL4D_PROVIDER_API_VERSION,
        },
    }


build_lift_map = _v1.build_lift_map
lift_residual = _v1.lift_residual
released_self_collision_for_case = _v1.released_self_collision_for_case
sha256_file = _v1.sha256_file
target_validity = _v1.target_validity

__all__ = [
    "CAUSAL4D_ARTIFACT_SCHEMA_VERSIONS",
    "CAUSAL4D_PROVIDER_API_VERSION",
    "CAUSAL4D_PROVIDER_CAPABILITIES",
    "CAUSAL4D_PROVIDER_CONTRACT",
    "CAUSAL4D_PROVIDER_CONTRACT_SHA256",
    "CAUSAL4D_PROVIDER_PACKAGE_VERSION",
    "FIXED_INITIAL_STD_M",
    "FIXED_INLIER_PRIOR",
    "FIXED_OBSERVATION_STD_M",
    "FIXED_OUTLIER_VARIANCE_MULTIPLIER",
    "FIXED_PROCESS_STD_M",
    "OfficialPhysTwinReplayProvider",
    "PhysTwinCase",
    "PhysTwinControllerLayout",
    "PhysTwinReplayProvider",
    "PhysTwinSpringGraph",
    "PhysTwinSpringGraphConfig",
    "build_lift_map",
    "build_phystwin_spring_graph",
    "causal4d_provider_manifest",
    "controller_hand_count",
    "create_official_case_replay_provider",
    "create_official_replay_provider",
    "infer_controller_groups",
    "lift_residual",
    "load_official_phystwin_case",
    "released_controller_layout",
    "released_self_collision_for_case",
    "robust_random_walk_endpoint",
    "sha256_file",
    "target_validity",
]
