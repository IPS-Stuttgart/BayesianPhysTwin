"""Compatibility import for Prob4D causal observation validation.

The implementation lives in :mod:`prob4d_observation_contract`; this module name
is retained because frozen Bayesian-PhysTwin code imports it directly.
"""

from .prob4d_observation_contract import (
    FIXED_EXTERNAL_CALIBRATION,
    PROB4D_CAUSAL_LINEAGE_VERSION,
    PROB4D_CAUSAL_STREAM_CONTRACT_VERSION,
    PROB4D_CAUSAL_STREAM_ID,
    PROB4D_FIXED_LAG_GAUGE_MODEL,
    PROB4D_GAUGE_FACTOR_NAMES,
    PROB4D_JOINT_GAUGE_FACTOR_PREFIX,
    PROB4D_JOINT_GAUGE_MODEL,
    PROB4D_LEGACY_CAUSAL_STREAM_CONTRACT_VERSION,
    PROB4D_LEGACY_GAUGE_FACTOR_NAMES,
    PROB4D_SOURCE_REPOSITORY,
    PROPAGATED_EXTERNAL_PRIOR,
    is_prob4d_causal_observation_belief,
    validate_prob4d_causal_observation_belief,
)

__all__ = [
    "FIXED_EXTERNAL_CALIBRATION",
    "PROB4D_CAUSAL_LINEAGE_VERSION",
    "PROB4D_CAUSAL_STREAM_CONTRACT_VERSION",
    "PROB4D_CAUSAL_STREAM_ID",
    "PROB4D_FIXED_LAG_GAUGE_MODEL",
    "PROB4D_GAUGE_FACTOR_NAMES",
    "PROB4D_JOINT_GAUGE_FACTOR_PREFIX",
    "PROB4D_JOINT_GAUGE_MODEL",
    "PROB4D_LEGACY_CAUSAL_STREAM_CONTRACT_VERSION",
    "PROB4D_LEGACY_GAUGE_FACTOR_NAMES",
    "PROB4D_SOURCE_REPOSITORY",
    "PROPAGATED_EXTERNAL_PRIOR",
    "is_prob4d_causal_observation_belief",
    "validate_prob4d_causal_observation_belief",
]
