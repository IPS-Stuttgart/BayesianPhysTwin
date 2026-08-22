from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

import bayesian_phystwin.mapped_prob4d_candidate_v1 as mapped_module
from bayesian_phystwin import GaugeAwareBeliefResult
from bayesian_phystwin.mapped_prob4d_candidate_v1 import (
    MAPPED_PROB4D_CANDIDATE_CLAIM_BOUNDARY,
    MAPPED_PROB4D_CANDIDATE_SCHEMA,
    MAPPED_PROB4D_CANDIDATE_VERSION,
    MappedClaimBearingProb4DCandidateV1,
    bind_provider_mapping_to_prob4d_candidate,
    infer_mapped_claim_bearing_prob4d_candidate_from_artifacts,
)
from bayesian_phystwin.observation_belief import ObservationBeliefV1
from bayesian_phystwin.physical_linearization import PhysicalLinearizationV1
from bayesian_phystwin.prospective_prob4d_update import (
    ClaimBearingProb4DCandidateV1,
    ClaimBearingProb4DUpdateV1,
    bind_claim_bearing_prob4d_candidate,
)
from bayesian_phystwin.provider_physical_mapping_audit_v1 import (
    ProviderPhysicalMappingAuditV1,
)

OBSERVATION_ID = "a" * 64
LINEARIZATION_ID = "b" * 64
PROVIDER_ID = "c" * 64
PHYSICAL_QUERY_ID = "d" * 64
MAPPING_PROTOCOL_ID = "e" * 64
CALIBRATION_IDS = {"gauge": "f" * 64, "point": "0" * 64}


def _lineage() -> dict[str, object]:
    return {
        "observation_artifact_id": OBSERVATION_ID,
        "linearization_artifact_id": LINEARIZATION_ID,
        "prob4d_claim_bearing_provider_manifest_id": PROVIDER_ID,
        "prob4d_claim_bearing_calibration_artifact_ids": CALIBRATION_IDS,
        "prob4d_claim_bearing_runtime_revision_source": "independent-vcs-check",
        "prob4d_claim_bearing_runtime_revision_independently_verified": True,
    }


def _result(
    *,
    inference_admissible: bool = True,
    reason: str = "accepted",
) -> GaugeAwareBeliefResult:
    coefficient = 0.1 if inference_admissible else 0.0
    return GaugeAwareBeliefResult(
        inference_admissible=inference_admissible,
        reason=reason,
        state_coefficients=np.array([coefficient]),
        gauge_delta=np.zeros(0),
        shared_bias_coefficients=np.zeros(0),
        view_bias_coefficients=np.zeros(0),
        anchor_bias_coefficients=np.zeros(0),
        posterior_covariance=np.array([[0.2]]),
        identifiable_state_transform=np.array([[1.0]]),
        identifiable_fractions=np.array([1.0]),
        query_sensitivity_fractions=np.array([1.0]),
        robust_weights=np.array([1.0]),
        anchor_robust_weights=np.zeros(0),
        diagnostics={"solver": "test"},
        input_lineage=_lineage(),
    )


def _update(
    *,
    inference_admissible: bool = True,
    reason: str = "accepted",
) -> ClaimBearingProb4DUpdateV1:
    return ClaimBearingProb4DUpdateV1(
        result=_result(
            inference_admissible=inference_admissible,
            reason=reason,
        ),
        observation_artifact_id=OBSERVATION_ID,
        linearization_artifact_id=LINEARIZATION_ID,
        provider_manifest_id=PROVIDER_ID,
        calibration_artifact_ids=CALIBRATION_IDS,
        runtime_revision_source="independent-vcs-check",
        runtime_revision_independently_verified=True,
    )


def _candidate(
    *,
    inference_admissible: bool = True,
    reason: str = "accepted",
) -> ClaimBearingProb4DCandidateV1:
    return bind_claim_bearing_prob4d_candidate(
        _update(
            inference_admissible=inference_admissible,
            reason=reason,
        )
    )


def _audit(**updates: Any) -> ProviderPhysicalMappingAuditV1:
    values: dict[str, Any] = {
        "case_id": "source-case-01",
        "case_artifact_id": "1" * 64,
        "provider_artifact_id": OBSERVATION_ID,
        "physical_query_id": PHYSICAL_QUERY_ID,
        "mapping_protocol_id": MAPPING_PROTOCOL_ID,
        "provider_frame": "camera_native",
        "physical_frame": "robot_world",
        "policy_id": "2" * 64,
        "mapping_admissible": True,
        "technical_valid": True,
        "provider_support_complete": True,
        "query_support_sufficient": True,
        "result_reason": "provider-physical-mapping-admissible",
        "rejection_reasons": (),
        "diagnostics": {"source_only": True},
    }
    values.update(updates)
    return ProviderPhysicalMappingAuditV1(**values)


def _rejected_audit() -> ProviderPhysicalMappingAuditV1:
    return _audit(
        mapping_admissible=False,
        query_support_sufficient=False,
        result_reason="insufficient-physical-query-overlap",
        rejection_reasons=("insufficient-physical-query-overlap",),
    )


def _fake_observation(
    artifact_id: str = OBSERVATION_ID,
) -> ObservationBeliefV1:
    return cast(
        ObservationBeliefV1,
        SimpleNamespace(artifact_id=artifact_id),
    )


def _fake_linearization() -> PhysicalLinearizationV1:
    return cast(
        PhysicalLinearizationV1,
        SimpleNamespace(artifact_id=LINEARIZATION_ID),
    )


def test_bound_candidate_binds_mapping_query_and_candidate_identities() -> None:
    candidate = _candidate()
    audit = _audit()

    mapped = bind_provider_mapping_to_prob4d_candidate(
        candidate,
        audit,
        physical_query_id=PHYSICAL_QUERY_ID,
        metadata={"information_split": "source-only"},
    )

    assert mapped.candidate is candidate
    assert mapped.mapping_audit is audit
    assert mapped.mapping_audit_id == audit.audit_id
    assert mapped.physical_query_id == PHYSICAL_QUERY_ID
    assert mapped.inference_admissible is True
    assert mapped.reason == "accepted"
    assert mapped.covariance_semantics is candidate.covariance_semantics
    assert len(cast(str, mapped.artifact_id)) == 64

    record = mapped.to_record()
    assert record["schema"] == MAPPED_PROB4D_CANDIDATE_SCHEMA
    assert record["schema_version"] == MAPPED_PROB4D_CANDIDATE_VERSION
    assert record["candidate_id"] == candidate.candidate_id
    assert record["mapping_audit_id"] == audit.audit_id
    assert record["claim_boundary"] == MAPPED_PROB4D_CANDIDATE_CLAIM_BOUNDARY
    assert record["candidate"] == candidate.to_record()
    assert record["mapping_audit"] == audit.to_dict()
    json.dumps(record, sort_keys=True, allow_nan=False)


def test_bound_candidate_preserves_rejected_solver_fallback_semantics() -> None:
    candidate = _candidate(
        inference_admissible=False,
        reason="strict-v2-fixed-point-not-converged",
    )

    mapped = bind_provider_mapping_to_prob4d_candidate(
        candidate,
        _audit(),
        physical_query_id=PHYSICAL_QUERY_ID,
    )

    assert mapped.inference_admissible is False
    assert mapped.reason == "strict-v2-fixed-point-not-converged"
    assert mapped.covariance_semantics.method == "exact_prior_fallback"


def test_binding_rejects_inadmissible_mapping() -> None:
    with pytest.raises(ValueError, match="mapping is inadmissible"):
        bind_provider_mapping_to_prob4d_candidate(
            _candidate(),
            _rejected_audit(),
            physical_query_id=PHYSICAL_QUERY_ID,
        )


def test_binding_rejects_provider_artifact_mismatch() -> None:
    with pytest.raises(ValueError, match="provider_artifact_id"):
        bind_provider_mapping_to_prob4d_candidate(
            _candidate(),
            _audit(provider_artifact_id="3" * 64),
            physical_query_id=PHYSICAL_QUERY_ID,
        )


def test_binding_rejects_physical_query_mismatch() -> None:
    with pytest.raises(ValueError, match="physical_query_id"):
        bind_provider_mapping_to_prob4d_candidate(
            _candidate(),
            _audit(),
            physical_query_id="4" * 64,
        )


def test_mapping_identity_changes_the_bound_candidate_identity() -> None:
    candidate = _candidate()
    first = bind_provider_mapping_to_prob4d_candidate(
        candidate,
        _audit(),
        physical_query_id=PHYSICAL_QUERY_ID,
    )
    second = bind_provider_mapping_to_prob4d_candidate(
        candidate,
        _audit(mapping_protocol_id="5" * 64),
        physical_query_id=PHYSICAL_QUERY_ID,
    )

    assert first.candidate.candidate_id == second.candidate.candidate_id
    assert first.mapping_audit_id != second.mapping_audit_id
    assert first.artifact_id != second.artifact_id


def test_inadmissible_mapping_stops_before_candidate_inference(monkeypatch) -> None:
    events: list[str] = []

    def infer(*args: object, **kwargs: object) -> ClaimBearingProb4DCandidateV1:
        events.append("infer")
        raise AssertionError("candidate inference must not run")

    monkeypatch.setattr(
        mapped_module,
        "infer_claim_bearing_prob4d_candidate_from_artifacts",
        infer,
    )

    with pytest.raises(ValueError, match="mapping is inadmissible"):
        infer_mapped_claim_bearing_prob4d_candidate_from_artifacts(
            _fake_observation(),
            _fake_linearization(),
            mapping_audit=_rejected_audit(),
            physical_query_id=PHYSICAL_QUERY_ID,
            physical_prediction_xyz_m=np.zeros((1, 3)),
        )
    assert events == []


def test_mapping_identity_mismatch_stops_before_candidate_inference(
    monkeypatch,
) -> None:
    events: list[str] = []

    def infer(*args: object, **kwargs: object) -> ClaimBearingProb4DCandidateV1:
        events.append("infer")
        raise AssertionError("candidate inference must not run")

    monkeypatch.setattr(
        mapped_module,
        "infer_claim_bearing_prob4d_candidate_from_artifacts",
        infer,
    )

    with pytest.raises(ValueError, match="provider_artifact_id"):
        infer_mapped_claim_bearing_prob4d_candidate_from_artifacts(
            _fake_observation(),
            _fake_linearization(),
            mapping_audit=_audit(provider_artifact_id="6" * 64),
            physical_query_id=PHYSICAL_QUERY_ID,
            physical_prediction_xyz_m=np.zeros((1, 3)),
        )
    with pytest.raises(ValueError, match="physical_query_id"):
        infer_mapped_claim_bearing_prob4d_candidate_from_artifacts(
            _fake_observation(),
            _fake_linearization(),
            mapping_audit=_audit(),
            physical_query_id="7" * 64,
            physical_prediction_xyz_m=np.zeros((1, 3)),
        )
    assert events == []


def test_passing_mapping_runs_candidate_inference_once_and_binds_result(
    monkeypatch,
) -> None:
    events: list[str] = []
    expected = _candidate()
    observation = _fake_observation()
    linearization = _fake_linearization()

    def infer(*args: object, **kwargs: object) -> ClaimBearingProb4DCandidateV1:
        events.append("infer")
        assert args == (observation, linearization)
        prediction = cast(np.ndarray, kwargs["physical_prediction_xyz_m"])
        np.testing.assert_array_equal(prediction, np.zeros((1, 3)))
        assert kwargs["shared_bias_jacobian"] is None
        assert kwargs["covariance_semantics"] is None
        assert kwargs["custom_anchor_flag"] is True
        return expected

    monkeypatch.setattr(
        mapped_module,
        "infer_claim_bearing_prob4d_candidate_from_artifacts",
        infer,
    )

    mapped = infer_mapped_claim_bearing_prob4d_candidate_from_artifacts(
        observation,
        linearization,
        mapping_audit=_audit(),
        physical_query_id=PHYSICAL_QUERY_ID,
        physical_prediction_xyz_m=np.zeros((1, 3)),
        metadata={"route": "mapped"},
        custom_anchor_flag=True,
    )

    assert events == ["infer"]
    assert mapped.candidate is expected
    assert mapped.metadata == {"route": "mapped"}
    assert mapped.mapping_audit_id == _audit().audit_id


def test_contract_rejects_wrong_types_and_forged_identity() -> None:
    candidate = _candidate()
    audit = _audit()
    with pytest.raises(TypeError, match="candidate"):
        MappedClaimBearingProb4DCandidateV1(
            candidate=cast(Any, object()),
            mapping_audit=audit,
            physical_query_id=PHYSICAL_QUERY_ID,
        )
    with pytest.raises(TypeError, match="mapping_audit"):
        MappedClaimBearingProb4DCandidateV1(
            candidate=candidate,
            mapping_audit=cast(Any, object()),
            physical_query_id=PHYSICAL_QUERY_ID,
        )
    with pytest.raises(ValueError, match="physical_query_id"):
        bind_provider_mapping_to_prob4d_candidate(
            candidate,
            audit,
            physical_query_id="not-a-digest",
        )
    with pytest.raises(ValueError, match="artifact_id"):
        MappedClaimBearingProb4DCandidateV1(
            candidate=candidate,
            mapping_audit=audit,
            physical_query_id=PHYSICAL_QUERY_ID,
            artifact_id="8" * 64,
        )
