"""Compatibility boundary for Prob4D causal observation validation.

The semantic implementation lives in :mod:`prob4d_observation_contract`; this
module name remains the public boundary for frozen Bayesian-PhysTwin imports. It
also resolves the provider-specific stream-contract version without duplicating
the semantic validator.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .observation_belief import ObservationBeliefV1
from .prob4d_observation_contract import (
    PROB4D_CAUSAL_LINEAGE_VERSION,
    PROB4D_CAUSAL_STREAM_ID,
    PROB4D_FIXED_LAG_GAUGE_MODEL,
    PROB4D_GAUGE_FACTOR_NAMES,
    PROB4D_JOINT_GAUGE_FACTOR_PREFIX,
    PROB4D_JOINT_GAUGE_MODEL,
    PROB4D_LEGACY_GAUGE_FACTOR_NAMES,
    PROB4D_SOURCE_REPOSITORY,
    is_prob4d_causal_observation_belief,
    validate_prob4d_causal_observation_belief as _validate_prob4d_semantics,
)

PROB4D_LEGACY_CAUSAL_STREAM_CONTRACT_VERSION = 1
PROB4D_CAUSAL_STREAM_CONTRACT_VERSION = 2
PROB4D_LEGACY_COVARIANCE_SEMANTICS = "legacy_per_window_sim3_marginals_v1"


def _resolved_stream_contract(
    metadata: Mapping[str, Any],
    covariance_semantics: object,
) -> tuple[int | None, bool]:
    """Resolve an explicit or safely inferable provider stream version."""

    if covariance_semantics == PROB4D_LEGACY_COVARIANCE_SEMANTICS:
        expected: int | None = PROB4D_LEGACY_CAUSAL_STREAM_CONTRACT_VERSION
    elif covariance_semantics == PROB4D_JOINT_GAUGE_MODEL:
        expected = PROB4D_CAUSAL_STREAM_CONTRACT_VERSION
    elif covariance_semantics == PROB4D_FIXED_LAG_GAUGE_MODEL:
        # The fixed-lag product is a labelled reconstruction approximation, not
        # an uncertainty-preserving strict causal stream contract.
        expected = None
    else:
        raise ValueError(
            "Prob4D validation returned unknown covariance semantics"
        )

    declared = metadata.get("prob4d_causal_stream_contract_version")
    if declared is None:
        return expected, expected is not None
    if isinstance(declared, bool) or not isinstance(declared, int):
        raise ValueError(
            "Prob4D causal stream contract version must be an integer"
        )
    if expected is None:
        raise ValueError(
            "approximate fixed-lag covariance cannot declare a strict Prob4D "
            "causal stream contract version"
        )
    if declared != expected:
        raise ValueError(
            "Prob4D causal stream contract version disagrees with covariance "
            "semantics"
        )
    return expected, False


def validate_prob4d_causal_observation_belief(
    belief: ObservationBeliefV1,
) -> dict[str, object]:
    """Validate semantics, then bind the resolved provider stream version."""

    result = dict(_validate_prob4d_semantics(belief))
    version, inferred = _resolved_stream_contract(
        belief.metadata,
        result.get("covariance_semantics"),
    )
    result.update(
        stream_contract_version=version,
        stream_contract_version_inferred=inferred,
        strict_causal_stream_contract=version is not None,
    )
    return result


__all__ = [
    "PROB4D_CAUSAL_LINEAGE_VERSION",
    "PROB4D_CAUSAL_STREAM_CONTRACT_VERSION",
    "PROB4D_CAUSAL_STREAM_ID",
    "PROB4D_FIXED_LAG_GAUGE_MODEL",
    "PROB4D_GAUGE_FACTOR_NAMES",
    "PROB4D_JOINT_GAUGE_FACTOR_PREFIX",
    "PROB4D_JOINT_GAUGE_MODEL",
    "PROB4D_LEGACY_CAUSAL_STREAM_CONTRACT_VERSION",
    "PROB4D_LEGACY_COVARIANCE_SEMANTICS",
    "PROB4D_LEGACY_GAUGE_FACTOR_NAMES",
    "PROB4D_SOURCE_REPOSITORY",
    "is_prob4d_causal_observation_belief",
    "validate_prob4d_causal_observation_belief",
]
