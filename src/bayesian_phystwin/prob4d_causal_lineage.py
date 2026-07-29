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
)
from .prob4d_observation_contract import (
    validate_prob4d_causal_observation_belief as _validate_prob4d_semantics,
)
from .prob4d_provider_attestation import validate_prob4d_provider_attestation

PROB4D_LEGACY_CAUSAL_STREAM_CONTRACT_VERSION = 1
PROB4D_CAUSAL_STREAM_CONTRACT_VERSION = 2
PROB4D_LEGACY_COVARIANCE_SEMANTICS = "legacy_per_window_sim3_marginals_v1"


def _require_mapping(value: object, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _require_sha256(value: object, *, name: str) -> str:
    digest = str(value)
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return digest


def _require_nonnegative_integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


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
        raise ValueError("Prob4D validation returned unknown covariance semantics")

    declared = metadata.get("prob4d_causal_stream_contract_version")
    if declared is None:
        return expected, expected is not None
    if isinstance(declared, bool) or not isinstance(declared, int):
        raise ValueError("Prob4D causal stream contract version must be an integer")
    if expected is None:
        raise ValueError(
            "approximate fixed-lag covariance cannot declare a strict Prob4D "
            "causal stream contract version"
        )
    if declared != expected:
        raise ValueError(
            "Prob4D causal stream contract version disagrees with covariance semantics"
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
        "runtime_revision_independently_verified": runtime["independently_verified"],
    }


def _claim_bearing_calibration_summary(
    belief: ObservationBeliefV1,
    provider: Mapping[str, Any],
) -> dict[str, object]:
    calibration = _require_mapping(
        belief.metadata.get("covariance_calibration"),
        name="claim-bearing Prob4D covariance calibration metadata",
    )
    if calibration.get("status") != "calibrated":
        raise ValueError(
            "claim-bearing Prob4D observation requires calibrated covariance metadata"
        )
    if calibration.get("uncalibrated_exploratory_covariance_allowed") is not False:
        raise ValueError(
            "claim-bearing Prob4D observation cannot allow uncalibrated covariance"
        )
    if calibration.get("pointwise_covariance_fallback_allowed") is not False:
        raise ValueError(
            "claim-bearing Prob4D observation cannot allow pointwise covariance fallback"
        )

    alignment_count = _require_nonnegative_integer(
        calibration.get("alignment_count"),
        name="claim-bearing Prob4D alignment_count",
    )
    calibrated_alignment_count = _require_nonnegative_integer(
        calibration.get("gauge_calibrated_alignment_count"),
        name="claim-bearing Prob4D gauge_calibrated_alignment_count",
    )
    if calibrated_alignment_count != alignment_count:
        raise ValueError(
            "claim-bearing Prob4D observation has uncalibrated gauge alignments"
        )
    fallback_counts = _require_mapping(
        calibration.get("covariance_fallback_counts"),
        name="claim-bearing Prob4D covariance fallback counts",
    )
    if fallback_counts:
        raise ValueError(
            "claim-bearing Prob4D observation reports covariance fallback use"
        )

    attested_ids = _require_mapping(
        provider.get("calibration_artifact_ids"),
        name="attested Prob4D calibration artifact IDs",
    )
    gauge_id = _require_sha256(
        calibration.get("gauge_artifact_id"),
        name="Prob4D gauge calibration artifact ID",
    )
    point_id = _require_sha256(
        calibration.get("point_artifact_id"),
        name="Prob4D point calibration artifact ID",
    )
    if gauge_id != _require_sha256(
        attested_ids.get("gauge_artifact_id"),
        name="attested Prob4D gauge calibration artifact ID",
    ):
        raise ValueError(
            "Prob4D gauge calibration metadata differs from its provider attestation"
        )
    if point_id != _require_sha256(
        attested_ids.get("point_artifact_id"),
        name="attested Prob4D point calibration artifact ID",
    ):
        raise ValueError(
            "Prob4D point calibration metadata differs from its provider attestation"
        )

    return {
        "status": "calibrated",
        "calibration_artifact_ids": {
            "gauge_artifact_id": gauge_id,
            "point_artifact_id": point_id,
        },
        "alignment_count": alignment_count,
        "gauge_calibrated_alignment_count": calibrated_alignment_count,
        "covariance_fallback_counts": {},
        "uncalibrated_exploratory_covariance_allowed": False,
        "pointwise_covariance_fallback_allowed": False,
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
    """Require explicit stream-v2 covariance and a calibrated provider-v2 producer."""

    result = validate_prob4d_causal_observation_belief(
        belief,
        require_claim_bearing_provider_v2=True,
    )
    if result.get("strict_causal_stream_contract") is not True:
        raise ValueError(
            "claim-bearing Prob4D observation requires a strict causal stream contract"
        )
    if (
        result.get("stream_contract_version") != PROB4D_CAUSAL_STREAM_CONTRACT_VERSION
        or result.get("stream_contract_version_inferred") is not False
    ):
        raise ValueError(
            "claim-bearing Prob4D observation requires explicit causal stream "
            "contract version 2"
        )
    if (
        result.get("covariance_semantics") != PROB4D_JOINT_GAUGE_MODEL
        or result.get("cross_window_covariance_preserved") is not True
    ):
        raise ValueError(
            "claim-bearing Prob4D observation requires the full joint cross-window "
            "gauge covariance"
        )
    provider = _require_mapping(
        result.get("provider_attestation"),
        name="validated claim-bearing Prob4D provider attestation",
    )
    calibration = _claim_bearing_calibration_summary(belief, provider)
    result.update(
        claim_bearing_provider_v2_validated=True,
        claim_bearing_covariance_calibration=calibration,
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
