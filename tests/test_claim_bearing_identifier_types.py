from __future__ import annotations

from copy import deepcopy

import pytest
from test_query_calibration_stable_core import (
    test_query_calibration_stable_core_coverage,
)

from bayesian_phystwin.prob4d_provider_attestation import (
    compute_prob4d_provider_manifest_id,
    validate_prob4d_provider_attestation,
)

_QUERY_CALIBRATION_STABLE_TESTS = (
    test_query_calibration_stable_core_coverage,
)
_REVISION = "1" * 40


def _provider_manifest() -> dict[str, object]:
    descriptor: dict[str, object] = {
        "provider_name": "prob4d",
        "provider_version": "0.3.0",
        "provider_revision": _REVISION,
        "provider_api_version": 2,
        "capabilities": [
            "analytic_sim3_composition_jacobians",
            "canonical_repeated_eigenspace_covariance_root",
            "explicit_exploratory_and_claim_bearing_exports",
            "provider_attested_observation_artifacts",
            "runtime_revision_attestation",
            "strict_prediction_calibration_compatibility",
        ],
        "artifact_schema_versions": {
            "ObservationBeliefV1": 1,
            "Prob4DCausalObservationStream": 2,
        },
        "limitations": {
            "uncalibrated_export_is_default": False,
            "deployment_environment_revision_is_independent_vcs_evidence": False,
        },
        "metadata": {
            "source_repository": "FlorianPfaff/Prob4D",
            "python_import_boundary": "prob4d.provider_v2",
        },
    }
    return {
        "manifest_id": compute_prob4d_provider_manifest_id(descriptor),
        **descriptor,
    }


def _provider_attestation() -> dict[str, object]:
    manifest = _provider_manifest()
    return {
        "schema_name": "prob4d.provider-attestation",
        "schema_version": 1,
        "provider_api_version": 2,
        "provider_manifest_id": manifest["manifest_id"],
        "provider_manifest": manifest,
        "provider_revision": _REVISION,
        "python_import_boundary": "prob4d.provider_v2",
        "export_mode": "calibrated",
        "claim_bearing": True,
        "calibration_compatibility_validated": True,
        "calibration_artifact_ids": {
            "gauge_artifact_id": "5" * 64,
            "point_artifact_id": "6" * 64,
        },
        "covariance_root_mode": "canonical_eigenspaces",
        "composition_jacobian_mode": "analytic",
        "runtime_revision": {
            "expected_revision": _REVISION,
            "observed_revision": _REVISION,
            "source": "source_checkout",
            "clean_checkout": True,
            "matched": True,
            "independently_verified": True,
        },
    }


def test_literal_provider_attestation_identifiers_validate() -> None:
    validated = validate_prob4d_provider_attestation(
        _provider_attestation(),
        source_revision=_REVISION,
        require_claim_bearing=True,
    )
    assert validated["provider_revision"] == _REVISION


def test_provider_attestation_rejects_integer_revision_that_stringifies_to_hex() -> (
    None
):
    attestation = _provider_attestation()
    attestation["provider_revision"] = int(_REVISION)

    with pytest.raises(ValueError, match="literal"):
        validate_prob4d_provider_attestation(
            attestation,
            source_revision=_REVISION,
            require_claim_bearing=True,
        )


def test_provider_attestation_rejects_integer_source_revision() -> None:
    with pytest.raises(ValueError, match="literal"):
        validate_prob4d_provider_attestation(
            _provider_attestation(),
            source_revision=int(_REVISION),  # type: ignore[arg-type]
            require_claim_bearing=True,
        )


def test_provider_attestation_rejects_integer_calibration_digest() -> None:
    attestation = deepcopy(_provider_attestation())
    calibration = attestation["calibration_artifact_ids"]
    assert isinstance(calibration, dict)
    calibration["gauge_artifact_id"] = int("5" * 64)

    with pytest.raises(ValueError, match="literal"):
        validate_prob4d_provider_attestation(
            attestation,
            source_revision=_REVISION,
            require_claim_bearing=True,
        )
