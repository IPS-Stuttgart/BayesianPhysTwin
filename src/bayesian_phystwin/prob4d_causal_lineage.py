"""Compatibility boundary for Prob4D causal observation validation.

The semantic implementation lives in :mod:`prob4d_observation_contract`; this
module name remains the public boundary for frozen Bayesian-PhysTwin imports. It
also resolves the provider-specific stream-contract version and independently
validates a provider-v2 attestation whenever one is present.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .observation_belief import ObservationBeliefV1
from .prob4d_observation_contract import (
    FIXED_EXTERNAL_CALIBRATION,
    PROB4D_CAUSAL_LINEAGE_VERSION,
    PROB4D_CAUSAL_STREAM_ID,
    PROB4D_FIXED_LAG_GAUGE_MODEL,
    PROB4D_GAUGE_FACTOR_NAMES,
    PROB4D_JOINT_GAUGE_FACTOR_PREFIX,
    PROB4D_JOINT_GAUGE_MODEL,
    PROB4D_LEGACY_GAUGE_FACTOR_NAMES,
    PROB4D_SOURCE_REPOSITORY,
    PROPAGATED_EXTERNAL_PRIOR,
    is_prob4d_causal_observation_belief,
    validate_prob4d_causal_observation_belief as _validate_prob4d_semantics,
)
from .prob4d_provider_attestation import validate_prob4d_provider_attestation

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


def _provider_attestation_summary(
    belief: ObservationBeliefV1,
    *,
    require_claim_bearing: bool,
) -> dict[str, object] | None:
    raw = belief.metadata.get("prob4d_provider_attestation")
    if raw is None:
        if require_claim_bearing:
            raise ValueError(
                "a claim-bearing Prob4D provider-v2 attestation is required"
            )
        return None
    if not isinstance(raw, Mapping):
        raise ValueError("Prob4D provider attestation must be a mapping")
    validated = validate_prob4d_provider_attestation(
        raw,
        source_revision=belief.source_revision,
        require_claim_bearing=require_claim_bearing,
    )
    runtime = validated["runtime_revision"]
    return {
        "schema_name": validated["schema_name"],
        "schema_version": validated["schema_version"],
        "provider_api_version": validated["provider_api_version"],
        "provider_manifest_id": validated["provider_manifest_id"],
        "export_mode": validated["export_mode"],
        "claim_bearing": validated["claim_bearing"],
        "calibration_compatibility_validated": validated[
            "calibration_compatibility_validated"
        ],
        "calibration_artifact_ids": validated["calibration_artifact_ids"],
        "covariance_root_mode": validated["covariance_root_mode"],
        "composition_jacobian_mode": validated["composition_jacobian_mode"],
        "runtime_revision_source": runtime["source"],
        "runtime_revision_independently_verified": runtime[
            "independently_verified"
        ],
    }


def validate_prob4d_causal_observation_belief(
    belief: ObservationBeliefV1,
    *,
    require_claim_bearing_provider_v2: bool = False,
) -> dict[str, object]:
    """Validate causal semantics, stream version, and provider attestation.

    Frozen provider-v1 artifacts remain valid when no attestation is present. New
    prospective evidence can set ``require_claim_bearing_provider_v2=True`` to
    reject provider-v1 and exploratory provider-v2 artifacts.
    """

    result = dict(_validate_prob4d_semantics(belief))
    version, inferred = _resolved_stream_contract(
        belief.metadata,
        result.get("covariance_semantics"),
    )
    provider = _provider_attestation_summary(
        belief,
        require_claim_bearing=require_claim_bearing_provider_v2,
    )
    result.update(
        stream_contract_version=version,
        stream_contract_version_inferred=inferred,
        strict_causal_stream_contract=version is not None,
        provider_attestation_present=provider is not None,
        provider_attestation_validated=provider is not None,
        provider_attestation=provider,
    )
    return result


def validate_claim_bearing_prob4d_observation_belief(
    belief: ObservationBeliefV1,
) -> dict[str, object]:
    """Require the strict causal contract and a calibrated provider-v2 producer."""

    result = validate_prob4d_causal_observation_belief(
        belief,
        require_claim_bearing_provider_v2=True,
    )
    if result["strict_causal_stream_contract"] is not True:
        raise ValueError(
            "claim-bearing Prob4D observation requires a strict causal stream contract"
        )
    return result


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
    "PROB4D_LEGACY_COVARIANCE_SEMANTICS",
    "PROB4D_LEGACY_GAUGE_FACTOR_NAMES",
    "PROB4D_SOURCE_REPOSITORY",
    "PROPAGATED_EXTERNAL_PRIOR",
    "is_prob4d_causal_observation_belief",
    "validate_claim_bearing_prob4d_observation_belief",
    "validate_prob4d_causal_observation_belief",
]
