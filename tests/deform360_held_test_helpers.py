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
from bayesian_phystwin.deform360_held_protocol import (
    REQUIRED_IMMUTABLE_BINDING_KEYS,
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
    return bindings


def default_frame_zero_config() -> dict[str, object]:
    return asdict(FrameZeroAssetConfig())
