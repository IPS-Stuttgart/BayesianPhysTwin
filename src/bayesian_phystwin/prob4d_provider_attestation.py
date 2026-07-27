"""Independently validate self-contained Prob4D provider-v2 attestations.

This module deliberately does not import Prob4D. It rechecks the neutral JSON/hash
contract before Bayesian-PhysTwin treats a provider-v2 artifact as claim-bearing.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

PROB4D_PROVIDER_ATTESTATION_SCHEMA = "prob4d.provider-attestation"
PROB4D_PROVIDER_ATTESTATION_VERSION = 1
PROB4D_PROVIDER_API_VERSION = 2
PROB4D_PROVIDER_IMPORT_BOUNDARY = "prob4d.provider_v2"
PROB4D_PROVIDER_SOURCE_REPOSITORY = "FlorianPfaff/Prob4D"

_REQUIRED_CAPABILITIES = frozenset(
    {
        "analytic_sim3_composition_jacobians",
        "canonical_repeated_eigenspace_covariance_root",
        "explicit_exploratory_and_claim_bearing_exports",
        "provider_attested_observation_artifacts",
        "runtime_revision_attestation",
        "strict_prediction_calibration_compatibility",
    }
)
_ATTESTATION_FIELDS = frozenset(
    {
        "schema_name",
        "schema_version",
        "provider_api_version",
        "provider_manifest_id",
        "provider_manifest",
        "provider_revision",
        "python_import_boundary",
        "export_mode",
        "claim_bearing",
        "calibration_compatibility_validated",
        "calibration_artifact_ids",
        "covariance_root_mode",
        "composition_jacobian_mode",
        "runtime_revision",
    }
)
_RUNTIME_FIELDS = frozenset(
    {
        "expected_revision",
        "observed_revision",
        "source",
        "clean_checkout",
        "matched",
        "independently_verified",
    }
)
_CALIBRATION_FIELDS = frozenset(
    {
        "gauge_artifact_id",
        "point_artifact_id",
    }
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _finite_json_mapping(value: Any, *, name: str) -> dict[str, Any]:
    _require(isinstance(value, Mapping), f"{name} must be a mapping")
    try:
        normalized = json.loads(
            json.dumps(
                dict(value),
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        )
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must contain finite JSON data") from error
    _require(isinstance(normalized, dict), f"{name} must be a JSON object")
    return normalized


def _require_exact_fields(
    value: Mapping[str, Any],
    expected: frozenset[str],
    *,
    name: str,
) -> None:
    missing = expected - value.keys()
    extra = value.keys() - expected
    _require(
        not missing and not extra,
        f"{name} fields changed; missing={sorted(missing)}, extra={sorted(extra)}",
    )


def _require_sha256(value: Any, *, name: str) -> str:
    result = str(value)
    _require(
        len(result) == 64
        and all(character in "0123456789abcdef" for character in result),
        f"{name} must be a lowercase SHA-256 digest",
    )
    return result


def _require_revision(value: Any, *, name: str) -> str:
    result = str(value)
    _require(
        len(result) in {40, 64}
        and all(character in "0123456789abcdef" for character in result),
        f"{name} must be an exact lowercase Git commit",
    )
    return result


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        dict(value),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def compute_prob4d_provider_manifest_id(manifest: Mapping[str, Any]) -> str:
    """Recompute the self-declared provider-manifest content address."""

    normalized = _finite_json_mapping(manifest, name="Prob4D provider manifest")
    normalized.pop("manifest_id", None)
    return hashlib.sha256(_canonical_json(normalized)).hexdigest()


def _validate_manifest(
    value: Any,
    *,
    expected_revision: str,
) -> dict[str, Any]:
    manifest = _finite_json_mapping(value, name="Prob4D provider manifest")
    manifest_id = _require_sha256(
        manifest.get("manifest_id", ""),
        name="Prob4D provider manifest_id",
    )
    _require(
        compute_prob4d_provider_manifest_id(manifest) == manifest_id,
        "Prob4D provider manifest ID does not match its descriptor",
    )
    revision = _require_revision(
        manifest.get("provider_revision", ""),
        name="Prob4D provider-manifest revision",
    )
    _require(
        revision == expected_revision,
        "Prob4D provider-manifest revision differs from observation source revision",
    )
    _require(
        manifest.get("provider_name") == "prob4d",
        "Prob4D provider manifest changed provider name",
    )
    _require(
        manifest.get("provider_api_version") == PROB4D_PROVIDER_API_VERSION,
        "Prob4D provider manifest is not API version 2",
    )

    capabilities = manifest.get("capabilities")
    _require(
        isinstance(capabilities, list)
        and all(isinstance(item, str) and item for item in capabilities)
        and len(capabilities) == len(set(capabilities)),
        "Prob4D provider capabilities must be unique nonempty strings",
    )
    _require(
        _REQUIRED_CAPABILITIES.issubset(capabilities),
        "Prob4D provider manifest lacks required claim-bearing capabilities",
    )

    schemas = manifest.get("artifact_schema_versions")
    _require(
        isinstance(schemas, Mapping),
        "Prob4D provider artifact schemas must be a mapping",
    )
    _require(
        schemas.get("ObservationBeliefV1") == 1
        and schemas.get("Prob4DCausalObservationStream") == 2,
        "Prob4D provider manifest declares unsupported observation schemas",
    )

    limitations = manifest.get("limitations")
    _require(
        isinstance(limitations, Mapping),
        "Prob4D provider limitations must be a mapping",
    )
    _require(
        limitations.get("uncalibrated_export_is_default") is False,
        "Prob4D provider-v2 manifest must not default to uncalibrated export",
    )
    _require(
        limitations.get("deployment_environment_revision_is_independent_vcs_evidence")
        is False,
        "Prob4D provider manifest misstates deployment revision evidence",
    )

    metadata = manifest.get("metadata")
    _require(
        isinstance(metadata, Mapping),
        "Prob4D provider metadata must be a mapping",
    )
    _require(
        metadata.get("source_repository") == PROB4D_PROVIDER_SOURCE_REPOSITORY,
        "Prob4D provider manifest source repository changed",
    )
    _require(
        metadata.get("python_import_boundary") == PROB4D_PROVIDER_IMPORT_BOUNDARY,
        "Prob4D provider manifest import boundary changed",
    )
    return manifest


def _validate_runtime(
    value: Any,
    *,
    provider_revision: str,
    claim_bearing: bool,
) -> dict[str, Any]:
    runtime = _finite_json_mapping(value, name="Prob4D runtime attestation")
    _require_exact_fields(
        runtime,
        _RUNTIME_FIELDS,
        name="Prob4D runtime attestation",
    )
    expected = _require_revision(
        runtime.get("expected_revision", ""),
        name="Prob4D runtime expected revision",
    )
    _require(
        expected == provider_revision,
        "Prob4D runtime expected revision differs from provider revision",
    )
    observed_value = runtime.get("observed_revision")
    observed = (
        None
        if observed_value is None
        else _require_revision(
            observed_value,
            name="Prob4D runtime observed revision",
        )
    )
    source = runtime.get("source")
    _require(
        source
        in {
            "installed_vcs_metadata",
            "source_checkout",
            "deployment_environment",
            "unavailable",
        },
        "Prob4D runtime evidence source is unsupported",
    )
    clean = runtime.get("clean_checkout")
    _require(
        clean is None or isinstance(clean, bool),
        "Prob4D runtime clean_checkout must be Boolean or null",
    )
    matched = runtime.get("matched")
    independent = runtime.get("independently_verified")
    _require(isinstance(matched, bool), "Prob4D runtime matched must be Boolean")
    _require(
        isinstance(independent, bool),
        "Prob4D runtime independently_verified must be Boolean",
    )
    _require(
        matched is (observed == expected),
        "Prob4D runtime matched flag disagrees with its revisions",
    )
    expected_independent = bool(
        matched
        and source in {"installed_vcs_metadata", "source_checkout"}
        and clean is not False
    )
    _require(
        independent is expected_independent,
        "Prob4D runtime verification flag disagrees with its evidence",
    )
    if source == "source_checkout":
        _require(
            isinstance(clean, bool),
            "Prob4D source-checkout evidence must declare cleanliness",
        )
    else:
        _require(
            clean is None,
            "non-checkout Prob4D runtime evidence cannot declare cleanliness",
        )
    if claim_bearing:
        _require(
            matched and independent,
            "claim-bearing Prob4D attestation requires independently matched code",
        )
    return runtime


def _validate_calibration_ids(
    value: Any,
    *,
    claim_bearing: bool,
) -> dict[str, Any]:
    calibration = _finite_json_mapping(
        value,
        name="Prob4D calibration artifact IDs",
    )
    _require_exact_fields(
        calibration,
        _CALIBRATION_FIELDS,
        name="Prob4D calibration artifact IDs",
    )
    for field in sorted(_CALIBRATION_FIELDS):
        artifact_id = calibration.get(field)
        if artifact_id is not None:
            calibration[field] = _require_sha256(
                artifact_id,
                name=f"Prob4D calibration {field}",
            )
    if claim_bearing:
        _require(
            all(calibration[field] is not None for field in _CALIBRATION_FIELDS),
            "claim-bearing Prob4D attestation requires both calibration IDs",
        )
    return calibration


def validate_prob4d_provider_attestation(
    attestation: Mapping[str, Any],
    *,
    source_revision: str,
    require_claim_bearing: bool = False,
) -> dict[str, Any]:
    """Validate and normalize a provider-v2 statement without importing Prob4D."""

    normalized = _finite_json_mapping(
        attestation,
        name="Prob4D provider attestation",
    )
    _require_exact_fields(
        normalized,
        _ATTESTATION_FIELDS,
        name="Prob4D provider attestation",
    )
    _require(
        normalized.get("schema_name") == PROB4D_PROVIDER_ATTESTATION_SCHEMA,
        "unsupported Prob4D provider-attestation schema",
    )
    _require(
        normalized.get("schema_version") == PROB4D_PROVIDER_ATTESTATION_VERSION,
        "unsupported Prob4D provider-attestation version",
    )
    _require(
        normalized.get("provider_api_version") == PROB4D_PROVIDER_API_VERSION,
        "Prob4D provider attestation is not API version 2",
    )

    revision = _require_revision(
        normalized.get("provider_revision", ""),
        name="Prob4D provider-attestation revision",
    )
    observation_revision = _require_revision(
        source_revision,
        name="observation source revision",
    )
    _require(
        revision == observation_revision,
        "Prob4D provider-attestation revision differs from observation source revision",
    )
    _require(
        normalized.get("python_import_boundary") == PROB4D_PROVIDER_IMPORT_BOUNDARY,
        "Prob4D provider-attestation import boundary changed",
    )

    manifest = _validate_manifest(
        normalized.get("provider_manifest"),
        expected_revision=revision,
    )
    declared_manifest_id = _require_sha256(
        normalized.get("provider_manifest_id", ""),
        name="Prob4D provider-attestation manifest ID",
    )
    _require(
        declared_manifest_id == manifest["manifest_id"],
        "Prob4D attestation manifest ID differs from embedded manifest",
    )

    export_mode = normalized.get("export_mode")
    _require(
        export_mode in {"calibrated", "exploratory"},
        "Prob4D provider export mode is unsupported",
    )
    claim_bearing = normalized.get("claim_bearing")
    _require(
        isinstance(claim_bearing, bool),
        "Prob4D claim_bearing must be Boolean",
    )
    _require(
        claim_bearing is (export_mode == "calibrated"),
        "Prob4D claim-bearing flag disagrees with export mode",
    )
    compatibility = normalized.get("calibration_compatibility_validated")
    _require(
        isinstance(compatibility, bool),
        "Prob4D calibration compatibility flag must be Boolean",
    )
    _require(
        compatibility is claim_bearing,
        "Prob4D calibration compatibility flag disagrees with export mode",
    )
    if require_claim_bearing:
        _require(
            claim_bearing,
            "a claim-bearing Prob4D provider-v2 artifact is required",
        )

    calibration = _validate_calibration_ids(
        normalized.get("calibration_artifact_ids"),
        claim_bearing=claim_bearing,
    )
    covariance_mode = normalized.get("covariance_root_mode")
    _require(
        covariance_mode in {"canonical_eigenspaces", "legacy_eigenvectors"},
        "Prob4D covariance-root mode is unsupported",
    )
    composition_mode = normalized.get("composition_jacobian_mode")
    _require(
        composition_mode in {"analytic", "legacy_finite_difference"},
        "Prob4D composition-Jacobian mode is unsupported",
    )
    if claim_bearing:
        _require(
            covariance_mode == "canonical_eigenspaces",
            "claim-bearing Prob4D artifact requires canonical covariance roots",
        )
        _require(
            composition_mode == "analytic",
            "claim-bearing Prob4D artifact requires analytic composition Jacobians",
        )

    runtime = _validate_runtime(
        normalized.get("runtime_revision"),
        provider_revision=revision,
        claim_bearing=claim_bearing,
    )
    normalized["provider_manifest"] = manifest
    normalized["calibration_artifact_ids"] = calibration
    normalized["runtime_revision"] = runtime
    return normalized


__all__ = [
    "PROB4D_PROVIDER_API_VERSION",
    "PROB4D_PROVIDER_ATTESTATION_SCHEMA",
    "PROB4D_PROVIDER_ATTESTATION_VERSION",
    "PROB4D_PROVIDER_IMPORT_BOUNDARY",
    "PROB4D_PROVIDER_SOURCE_REPOSITORY",
    "compute_prob4d_provider_manifest_id",
    "validate_prob4d_provider_attestation",
]
