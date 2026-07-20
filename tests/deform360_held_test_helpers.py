from __future__ import annotations

from dataclasses import asdict
import hashlib

from bayesian_phystwin.deform360_frame_zero_assets import (
    FrameZeroAssetConfig,
    artifact_sha256,
)
from bayesian_phystwin.deform360_held_outcome_scoring import (
    OUTCOME_RECONSTRUCTION_CONTRACT,
)
from bayesian_phystwin.deform360_held_physical_prior import (
    HELD_PHYSICAL_NUMERIC_CONTRACT,
    UPSTREAM_FILE_SHA256,
    UPSTREAM_LOCK_BINDING_BY_PATH,
    UPSTREAM_RUNTIME_BUNDLE_CONTRACT,
)
from bayesian_phystwin.deform360_held_protocol import (
    REQUIRED_IMMUTABLE_BINDING_KEYS,
    SOURCE_FEASIBILITY_AMENDMENT_CONTRACT,
    held_contract_sha256,
)


def dummy_immutable_bindings() -> dict[str, str]:
    """Return a complete deterministic test-only held binding set."""

    bindings = {
        key: hashlib.sha256(f"test-only:{key}".encode()).hexdigest()
        for key in REQUIRED_IMMUTABLE_BINDING_KEYS
    }
    bindings["frame_zero_default_config"] = artifact_sha256(
        asdict(FrameZeroAssetConfig())
    )
    bindings["outcome_reconstruction_contract"] = held_contract_sha256(
        OUTCOME_RECONSTRUCTION_CONTRACT
    )
    bindings["held_physical_numeric_contract"] = held_contract_sha256(
        HELD_PHYSICAL_NUMERIC_CONTRACT
    )
    bindings["held_source_feasibility_amendment_contract"] = held_contract_sha256(
        SOURCE_FEASIBILITY_AMENDMENT_CONTRACT
    )
    bindings["upstream_runtime_bundle_tree"] = held_contract_sha256(
        UPSTREAM_RUNTIME_BUNDLE_CONTRACT
    )
    for path, binding_key in UPSTREAM_LOCK_BINDING_BY_PATH.items():
        bindings[binding_key] = UPSTREAM_FILE_SHA256[path]
    return bindings


def default_frame_zero_config() -> dict[str, object]:
    return asdict(FrameZeroAssetConfig())
