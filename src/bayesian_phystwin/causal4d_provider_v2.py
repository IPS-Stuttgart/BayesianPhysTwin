"""Typed Bayesian-PhysTwin replay and geometry surface for Causal4D.

Provider API v2 removes hidden mutable replay sequencing: every rollout carries
its controller trajectory, parameter values, state identity, and configuration
identity in one immutable request.  Legacy diagnostic and simulator-mutation
helpers remain confined to ``causal4d_provider_v1``.
"""

from __future__ import annotations

import os

from .contracts.provider import (
    installed_distribution_revision,
    installed_distribution_version,
)
from .contracts.replay import (
    InitialReplayRequestV1,
    PhysTwinReplayProviderV2,
    ReplayRequestV1,
    ReplayTrajectoryV1,
    RestartReplayRequestV1,
)
from .phystwin.artifacts import sha256_file
from .phystwin.geometry import build_lift_map, lift_residual, target_validity
from .phystwin.replay import (
    OfficialPhysTwinReplayProviderV2,
    create_official_replay_provider_v2,
)

CAUSAL4D_PROVIDER_API_VERSION = 2
CAUSAL4D_PROVIDER_PACKAGE_VERSION = "0.4.0"
CAUSAL4D_PROVIDER_CAPABILITIES = (
    "artifact_checksums",
    "immutable_replay_trajectories",
    "particle_endpoint_position",
    "particle_endpoint_velocity",
    "physical_parameter_particles",
    "phystwin_replay",
    "residual_lifting",
    "restart_velocity_history",
    "stateless_replay_requests",
    "target_validity",
    "typed_replay_requests",
)
CAUSAL4D_ARTIFACT_SCHEMA_VERSIONS = {
    "GraphBelief": 1,
    "TwinBelief": 1,
    "ReplayRequest": 1,
    "ReplayTrajectory": 1,
}

# Concise unversioned aliases within the explicitly versioned provider module.
PhysTwinReplayProvider = PhysTwinReplayProviderV2
OfficialPhysTwinReplayProvider = OfficialPhysTwinReplayProviderV2
create_official_replay_provider = create_official_replay_provider_v2


def causal4d_provider_manifest(
    *,
    provider_revision: str | None = None,
) -> dict[str, object]:
    """Return the provider-v2 descriptor consumed by Causal4D."""

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
            fallback=CAUSAL4D_PROVIDER_PACKAGE_VERSION,
        ),
        "provider_revision": revision,
        "schema_version": CAUSAL4D_PROVIDER_API_VERSION,
        "capabilities": list(CAUSAL4D_PROVIDER_CAPABILITIES),
        "artifact_schema_versions": dict(CAUSAL4D_ARTIFACT_SCHEMA_VERSIONS),
        "metadata": {
            "provider_api": "bayesian_phystwin.causal4d_provider_v2",
            "provider_api_version": CAUSAL4D_PROVIDER_API_VERSION,
            "legacy_provider_api": "bayesian_phystwin.causal4d_provider_v1",
        },
    }


__all__ = [
    "CAUSAL4D_ARTIFACT_SCHEMA_VERSIONS",
    "CAUSAL4D_PROVIDER_API_VERSION",
    "CAUSAL4D_PROVIDER_CAPABILITIES",
    "CAUSAL4D_PROVIDER_PACKAGE_VERSION",
    "InitialReplayRequestV1",
    "OfficialPhysTwinReplayProvider",
    "OfficialPhysTwinReplayProviderV2",
    "PhysTwinReplayProvider",
    "PhysTwinReplayProviderV2",
    "ReplayRequestV1",
    "ReplayTrajectoryV1",
    "RestartReplayRequestV1",
    "build_lift_map",
    "causal4d_provider_manifest",
    "create_official_replay_provider",
    "create_official_replay_provider_v2",
    "lift_residual",
    "sha256_file",
    "target_validity",
]
