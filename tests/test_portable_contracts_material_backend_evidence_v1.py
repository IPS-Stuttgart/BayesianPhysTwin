from __future__ import annotations

from typing import Any

import pytest

from bayesian_phystwin._portable_contracts import content_id
from bayesian_phystwin.evidence_decision_v1 import (
    DecisionMetricV1,
    EvidenceDecisionV1,
)
from bayesian_phystwin.material_backend_evidence_v1 import (
    MATERIAL_BACKEND_EVIDENCE_CLAIM_BOUNDARY,
    MATERIAL_BACKEND_EVIDENCE_SCHEMA,
    MATERIAL_BACKEND_EVIDENCE_STAGES,
    MATERIAL_BACKEND_EVIDENCE_VERSION,
    MaterialBackendEvidenceStatusV1,
    build_material_backend_evidence_status_v1,
    describe_material_backend_evidence_stages,
    load_material_backend_evidence_status_v1,
    require_material_backend_evidence_stage,
    save_material_backend_evidence_status_v1,
    verify_material_backend_evidence_status_v1,
)
from bayesian_phystwin.material_backend_qualification_v1 import (
    MaterialBackendQualificationV1,
)
from bayesian_phystwin.repository_provenance import RepositoryState

DIGEST_A = "a" * 64
DIGEST_B = "b" * 64
DIGEST_C = "c" * 64
DIGEST_D = "d" * 64
DIGEST_E = "e" * 64
REVISION = "1" * 40


def _qualification(**changes: Any) -> MaterialBackendQualificationV1:
    values: dict[str, Any] = {
        "canonical_profile_id": "jax-fem-quasistatic-v1",
        "producer_profile_id": "jax-fem-quasistatic-v1",
        "transport": "lagrangian-export-v1",
        "runtime_id": DIGEST_A,
        "qualification_protocol_id": DIGEST_B,
        "source_evidence_id": DIGEST_C,
        "source_group_ids": ("source-b", "source-a", "source-c"),
        "incumbent_runtime_id": DIGEST_D,
        "units_coordinate_entity_order_valid": True,
        "deterministic_replay_valid": True,
        "maximum_zero_action_drift_m": 0.0005,
        "allowed_zero_action_drift_m": 0.001,
        "maximum_rigid_equivariance_error_m": 0.0002,
        "allowed_rigid_equivariance_error_m": 0.001,
        "time_step_refinement_relative_error": 0.02,
        "allowed_time_step_refinement_relative_error": 0.05,
        "topology_identity_preserved": True,
        "physical_sanity_violations": 0,
        "gradient_claimed": True,
        "maximum_jacobian_relative_error": 0.03,
        "allowed_jacobian_relative_error": 0.05,
        "source_query_parity_rmse_m": 0.002,
        "allowed_source_query_parity_rmse_m": 0.005,
        "exact_fallback_verified": True,
        "protocol_frozen_before_source_outcomes": True,
        "target_outcomes_used": False,
        "metadata": {"evidence_role": "source-only"},
    }
    values.update(changes)
    return MaterialBackendQualificationV1(**values)


def _decision(
    *,
    role: str,
    parent_key: str,
    parent_id: str,
    salt: str,
    confirmatory: bool,
    authorized: bool,
    metadata_changes: dict[str, object] | None = None,
) -> EvidenceDecisionV1:
    metadata: dict[str, object] = {
        "evidence_role": role,
        "canonical_profile_id": "jax-fem-quasistatic-v1",
        "producer_profile_id": "jax-fem-quasistatic-v1",
        "runtime_id": DIGEST_A,
        parent_key: parent_id,
    }
    if metadata_changes:
        metadata.update(metadata_changes)
    return EvidenceDecisionV1(
        claim_id=f"material-backend-{role}-{salt}",
        protocol_id=f"material-backend-{role}-protocol-{salt}",
        status="pass",
        run_classification="confirmatory" if confirmatory else "controlled",
        claim_authorized=authorized,
        evidence_level=2,
        metric=DecisionMetricV1(
            name="object-balanced-regret",
            comparison="candidate-minus-incumbent",
            rule="upper-bound-nonpositive",
            observed_value=-0.1,
            threshold_value=0.0,
            unit="dimensionless",
        ),
        run_manifest_id=(salt * 64)[:64],
        evidence_fingerprint=((chr(ord(salt) + 1)) * 64)[:64],
        evidence_summary_sha256=((chr(ord(salt) + 2)) * 64)[:64],
        repositories=(
            RepositoryState(
                repository="IPS-Stuttgart/BayesianPhysTwin",
                revision=REVISION,
                dirty=False,
                role="primary",
            ),
        ),
        metadata=metadata,
        created_utc="2026-08-18T00:00:00+00:00",
    )


def _complete_evidence() -> tuple[
    MaterialBackendQualificationV1,
    EvidenceDecisionV1,
    EvidenceDecisionV1,
    EvidenceDecisionV1,
]:
    qualification = _qualification()
    qualification_id = qualification.artifact_id
    assert qualification_id is not None
    source = _decision(
        role="source-competence",
        parent_key="qualification_artifact_id",
        parent_id=qualification_id,
        salt="2",
        confirmatory=False,
        authorized=False,
    )
    target = _decision(
        role="fresh-object-validation",
        parent_key="source_decision_id",
        parent_id=source.decision_id,
        salt="3",
        confirmatory=True,
        authorized=True,
    )
    downstream = _decision(
        role="downstream-query-benefit",
        parent_key="target_decision_id",
        parent_id=target.decision_id,
        salt="4",
        confirmatory=True,
        authorized=True,
    )
    return qualification, source, target, downstream


def test_stage_description_is_contiguous_and_machine_readable() -> None:
    description = describe_material_backend_evidence_stages()

    assert [stage.code for stage in MATERIAL_BACKEND_EVIDENCE_STAGES] == [
        "T0",
        "T1",
        "T2",
        "T3",
        "T4",
        "T5",
        "T6",
    ]
    assert [item["code"] for item in description["stages"]] == [
        "T0",
        "T1",
        "T2",
        "T3",
        "T4",
        "T5",
        "T6",
    ]
    assert description["claim_boundary"] == MATERIAL_BACKEND_EVIDENCE_CLAIM_BOUNDARY


def test_registered_adapter_and_native_stages_are_distinct() -> None:
    registered = build_material_backend_evidence_status_v1(
        canonical_profile_id="jax-fem-quasistatic-v1",
        producer_profile_id="jax-fem-quasistatic-v1",
    )
    adapter = build_material_backend_evidence_status_v1(
        canonical_profile_id="jax-fem-quasistatic-v1",
        producer_profile_id="jax-fem-quasistatic-v1",
        adapter_evidence_id=DIGEST_B,
    )
    native = build_material_backend_evidence_status_v1(
        canonical_profile_id="jax-fem-quasistatic-v1",
        producer_profile_id="jax-fem-quasistatic-v1",
        adapter_evidence_id=DIGEST_B,
        runtime_id=DIGEST_A,
        native_replay_evidence_id=DIGEST_C,
    )

    assert (registered.stage_code, registered.stage) == (
        "T0",
        "transport-registered",
    )
    assert (adapter.stage_code, adapter.stage) == ("T1", "adapter-tested")
    assert (native.stage_code, native.stage) == (
        "T2",
        "native-runtime-replayed",
    )
    assert registered.artifact_id != adapter.artifact_id != native.artifact_id


def test_complete_evidence_builds_t6_and_roundtrips(tmp_path: Any) -> None:
    qualification, source, target, downstream = _complete_evidence()
    status = build_material_backend_evidence_status_v1(
        canonical_profile_id="jax-fem-quasistatic-v1",
        producer_profile_id="jax-fem-quasistatic-v1",
        adapter_evidence_id=DIGEST_B,
        runtime_id=DIGEST_A,
        native_replay_evidence_id=DIGEST_C,
        qualification=qualification,
        source_decision=source,
        target_decision=target,
        downstream_decision=downstream,
        target_group_ids=("target-c", "target-a", "target-b"),
        metadata={"policy": "contiguous-promotion-v1"},
    )

    assert status.stage == "downstream-query-benefit"
    assert status.stage_code == "T6"
    assert status.source_group_ids == ("source-a", "source-b", "source-c")
    assert status.target_group_ids == ("target-a", "target-b", "target-c")
    assert status.exact_fallback_verified
    assert status.artifact_id == content_id(status.descriptor())
    payload = status.to_payload()
    assert payload["schema"] == MATERIAL_BACKEND_EVIDENCE_SCHEMA
    assert payload["schema_version"] == MATERIAL_BACKEND_EVIDENCE_VERSION

    path = tmp_path / "material-backend-evidence.json"
    save_material_backend_evidence_status_v1(status, path)
    loaded = load_material_backend_evidence_status_v1(path)
    assert loaded == status
    assert loaded.to_payload() == payload
    with pytest.raises(FileExistsError):
        save_material_backend_evidence_status_v1(status, path)


def test_every_promotion_gap_fails_closed() -> None:
    with pytest.raises(ValueError, match="adapter evidence"):
        MaterialBackendEvidenceStatusV1(
            canonical_profile_id="jax-fem-quasistatic-v1",
            producer_profile_id="jax-fem-quasistatic-v1",
            transport="lagrangian-export-v1",
            runtime_id=DIGEST_A,
            native_replay_evidence_id=DIGEST_C,
        )
    with pytest.raises(ValueError, match="native runtime replay"):
        MaterialBackendEvidenceStatusV1(
            canonical_profile_id="jax-fem-quasistatic-v1",
            producer_profile_id="jax-fem-quasistatic-v1",
            transport="lagrangian-export-v1",
            adapter_evidence_id=DIGEST_B,
            qualification_artifact_id=DIGEST_D,
            source_group_ids=("source-a", "source-b"),
            exact_fallback_verified=True,
        )
    with pytest.raises(ValueError, match="source competence"):
        MaterialBackendEvidenceStatusV1(
            canonical_profile_id="jax-fem-quasistatic-v1",
            producer_profile_id="jax-fem-quasistatic-v1",
            transport="lagrangian-export-v1",
            adapter_evidence_id=DIGEST_B,
            runtime_id=DIGEST_A,
            native_replay_evidence_id=DIGEST_C,
            qualification_artifact_id=DIGEST_D,
            target_decision_id=DIGEST_E,
            source_group_ids=("source-a", "source-b"),
            target_group_ids=("target-a",),
            exact_fallback_verified=True,
        )


def test_qualification_and_group_boundaries_fail_closed() -> None:
    failed = _qualification(deterministic_replay_valid=False)
    with pytest.raises(ValueError, match="not qualified"):
        build_material_backend_evidence_status_v1(
            canonical_profile_id="jax-fem-quasistatic-v1",
            producer_profile_id="jax-fem-quasistatic-v1",
            adapter_evidence_id=DIGEST_B,
            runtime_id=DIGEST_A,
            native_replay_evidence_id=DIGEST_C,
            qualification=failed,
        )

    qualification, source, target, _ = _complete_evidence()
    with pytest.raises(ValueError, match="must be disjoint"):
        build_material_backend_evidence_status_v1(
            canonical_profile_id="jax-fem-quasistatic-v1",
            producer_profile_id="jax-fem-quasistatic-v1",
            adapter_evidence_id=DIGEST_B,
            runtime_id=DIGEST_A,
            native_replay_evidence_id=DIGEST_C,
            qualification=qualification,
            source_decision=source,
            target_decision=target,
            target_group_ids=("source-a", "target-a"),
        )


def test_decision_metadata_and_authorization_are_replayed() -> None:
    qualification = _qualification()
    qualification_id = qualification.artifact_id
    assert qualification_id is not None
    wrong_runtime = _decision(
        role="source-competence",
        parent_key="qualification_artifact_id",
        parent_id=qualification_id,
        salt="5",
        confirmatory=False,
        authorized=False,
        metadata_changes={"runtime_id": DIGEST_E},
    )
    with pytest.raises(ValueError, match="runtime_id"):
        build_material_backend_evidence_status_v1(
            canonical_profile_id="jax-fem-quasistatic-v1",
            producer_profile_id="jax-fem-quasistatic-v1",
            adapter_evidence_id=DIGEST_B,
            runtime_id=DIGEST_A,
            native_replay_evidence_id=DIGEST_C,
            qualification=qualification,
            source_decision=wrong_runtime,
        )

    qualification, source, _, _ = _complete_evidence()
    unauthorized_target = _decision(
        role="fresh-object-validation",
        parent_key="source_decision_id",
        parent_id=source.decision_id,
        salt="6",
        confirmatory=True,
        authorized=False,
    )
    with pytest.raises(ValueError, match="authorize"):
        build_material_backend_evidence_status_v1(
            canonical_profile_id="jax-fem-quasistatic-v1",
            producer_profile_id="jax-fem-quasistatic-v1",
            adapter_evidence_id=DIGEST_B,
            runtime_id=DIGEST_A,
            native_replay_evidence_id=DIGEST_C,
            qualification=qualification,
            source_decision=source,
            target_decision=unauthorized_target,
            target_group_ids=("target-a", "target-b"),
        )


def test_stage_requirement_replays_claim_bearing_evidence() -> None:
    qualification, source, target, downstream = _complete_evidence()
    status = build_material_backend_evidence_status_v1(
        canonical_profile_id="jax-fem-quasistatic-v1",
        producer_profile_id="jax-fem-quasistatic-v1",
        adapter_evidence_id=DIGEST_B,
        runtime_id=DIGEST_A,
        native_replay_evidence_id=DIGEST_C,
        qualification=qualification,
        source_decision=source,
        target_decision=target,
        downstream_decision=downstream,
        target_group_ids=("target-a", "target-b"),
    )

    assert (
        require_material_backend_evidence_stage(
            status,
            "fresh-object-validated",
            qualification=qualification,
            source_decision=source,
            target_decision=target,
            downstream_decision=downstream,
        )
        is status
    )
    with pytest.raises(ValueError, match="qualification is required"):
        require_material_backend_evidence_stage(
            status,
            "numerically-qualified",
        )
    adapter = build_material_backend_evidence_status_v1(
        canonical_profile_id="jax-fem-quasistatic-v1",
        producer_profile_id="jax-fem-quasistatic-v1",
        adapter_evidence_id=DIGEST_B,
    )
    with pytest.raises(ValueError, match="is below"):
        require_material_backend_evidence_stage(
            adapter,
            "native-runtime-replayed",
        )


def test_payload_tampering_and_external_substitution_are_rejected() -> None:
    qualification, source, target, downstream = _complete_evidence()
    status = build_material_backend_evidence_status_v1(
        canonical_profile_id="jax-fem-quasistatic-v1",
        producer_profile_id="jax-fem-quasistatic-v1",
        adapter_evidence_id=DIGEST_B,
        runtime_id=DIGEST_A,
        native_replay_evidence_id=DIGEST_C,
        qualification=qualification,
        source_decision=source,
        target_decision=target,
        downstream_decision=downstream,
        target_group_ids=("target-a", "target-b"),
    )

    payload = status.to_payload()
    payload["stage"] = "adapter-tested"
    with pytest.raises(ValueError, match="does not replay"):
        MaterialBackendEvidenceStatusV1.from_payload(payload)

    payload = status.to_payload()
    payload["runtime_id"] = DIGEST_E
    with pytest.raises(ValueError, match="artifact_id"):
        MaterialBackendEvidenceStatusV1.from_payload(payload)

    other_qualification = _qualification(source_evidence_id=DIGEST_E)
    with pytest.raises(ValueError, match="qualification artifact"):
        verify_material_backend_evidence_status_v1(
            status,
            qualification=other_qualification,
            source_decision=source,
            target_decision=target,
            downstream_decision=downstream,
        )
