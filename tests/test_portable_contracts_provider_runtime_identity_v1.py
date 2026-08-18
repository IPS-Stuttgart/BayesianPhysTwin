from __future__ import annotations

from dataclasses import dataclass

import pytest

from bayesian_phystwin.inference.v1 import GuardedCandidateInference
from bayesian_phystwin.prob4d_provider_attestation import (
    compute_prob4d_provider_manifest_id,
)
from bayesian_phystwin.provider_runtime_identity_v1 import (
    Prob4DRuntimeIdentityV1,
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


def _attestation() -> dict[str, object]:
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


def test_runtime_identity_keeps_commit_and_evidence_source_separate() -> None:
    identity = Prob4DRuntimeIdentityV1.from_provider_attestation(_attestation())

    assert identity.runtime_revision == _REVISION
    assert identity.runtime_revision_source == "source_checkout"
    assert identity.runtime_revision != identity.runtime_revision_source
    assert Prob4DRuntimeIdentityV1.from_record(identity.to_record()) == identity


def test_runtime_identity_rejects_source_label_as_revision() -> None:
    with pytest.raises(ValueError, match="Git commit"):
        Prob4DRuntimeIdentityV1(
            project_id="prob4d",
            source_repository="IPS-Stuttgart/Prob4D",
            provider_manifest_id="a" * 64,
            expected_revision="installed_vcs_metadata",
            observed_revision="installed_vcs_metadata",
            revision_evidence_source="installed_vcs_metadata",
            clean_checkout=None,
            independently_verified=True,
        )


def test_runtime_identity_rejects_mismatched_or_unverified_code() -> None:
    with pytest.raises(ValueError, match="differs"):
        Prob4DRuntimeIdentityV1(
            project_id="prob4d",
            source_repository="IPS-Stuttgart/Prob4D",
            provider_manifest_id="a" * 64,
            expected_revision="1" * 40,
            observed_revision="2" * 40,
            revision_evidence_source="installed_vcs_metadata",
            clean_checkout=None,
            independently_verified=True,
        )
    with pytest.raises(ValueError, match="independently verified"):
        Prob4DRuntimeIdentityV1(
            project_id="prob4d",
            source_repository="IPS-Stuttgart/Prob4D",
            provider_manifest_id="a" * 64,
            expected_revision="1" * 40,
            observed_revision="1" * 40,
            revision_evidence_source="installed_vcs_metadata",
            clean_checkout=None,
            independently_verified=False,
        )


@dataclass(frozen=True)
class _TreeUpdateShape:
    update_id: str
    inference_admissible: bool

    @property
    def candidate_id(self) -> str:
        return self.update_id


def test_tree_update_protocol_shape_is_guardable() -> None:
    update = _TreeUpdateShape(update_id="a" * 64, inference_admissible=True)
    assert isinstance(update, GuardedCandidateInference)
