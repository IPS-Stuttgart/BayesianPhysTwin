"""Stable legacy-artifact boundary for Causal4D integrations.

New cross-repository artifacts must use the versioned JSON/NPZ contracts.  This
module exists only for hash-locked released PhysTwin pickles that cannot be
migrated retroactively.
"""

from __future__ import annotations

from .legacy_artifacts import (
    LegacyPhysTwinArtifactKind,
    load_trusted_legacy_phystwin_pickle,
)

CAUSAL4D_ARTIFACT_API_VERSION = 1
CAUSAL4D_ARTIFACT_CAPABILITIES = (
    "digest_preflight_before_pickle",
    "top_level_artifact_contract",
)


def causal4d_artifact_provider_manifest() -> dict[str, object]:
    """Return the stable legacy-artifact provider descriptor."""

    return {
        "provider_api": "bayesian_phystwin.causal4d_artifacts_v1",
        "provider_api_version": CAUSAL4D_ARTIFACT_API_VERSION,
        "capabilities": list(CAUSAL4D_ARTIFACT_CAPABILITIES),
        "new_artifact_policy": "json-npz-only",
    }


__all__ = [
    "CAUSAL4D_ARTIFACT_API_VERSION",
    "CAUSAL4D_ARTIFACT_CAPABILITIES",
    "LegacyPhysTwinArtifactKind",
    "causal4d_artifact_provider_manifest",
    "load_trusted_legacy_phystwin_pickle",
]
