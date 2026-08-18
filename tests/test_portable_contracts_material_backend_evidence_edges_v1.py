from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest

from bayesian_phystwin.evidence_decision_v1 import (
    DecisionMetricV1,
    EvidenceDecisionV1,
)
from bayesian_phystwin.material_backend_evidence_v1 import (
    MATERIAL_BACKEND_EVIDENCE_CLAIM_BOUNDARY,
    MATERIAL_BACKEND_EVIDENCE_SCHEMA,
    MATERIAL_BACKEND_EVIDENCE_VERSION,
    MaterialBackendEvidenceStatusV1,
    build_material_backend_evidence_status_v1,
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
    status: str = "pass",
    run_classification: str = "controlled",
    authorized: bool = False,
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
        status=cast(Any, status),
        run_classification=cast(Any, run_classification),
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
    )
    target = _decision(
        role="fresh-object-validation",
        parent_key="source_decision_id",
        parent_id=source.decision_id,
        salt="3",
        run_classification="confirmatory",
        authorized=True,
    )
    downstream = _decision(
        role="downstream-query-benefit",
        parent_key="target_decision_id",
        parent_id=target.decision_id,
        salt="4",
        run_classification="confirmatory",
        authorized=True,
    )
    return qualification, source, target, downstream


def _status_kwargs() -> dict[str, Any]:
    return {
        "canonical_profile_id": "jax-fem-quasistatic-v1",
        "producer_profile_id": "jax-fem-quasistatic-v1",
        "transport": "lagrangian-export-v1",
    }


def _qualified_status(
    qualification: MaterialBackendQualificationV1,
    **changes: Any,
) -> MaterialBackendEvidenceStatusV1:
    qualification_id = qualification.artifact_id
    assert qualification_id is not None
    values = _status_kwargs()
    values.update(
        {
            "adapter_evidence_id": DIGEST_B,
            "runtime_id": DIGEST_A,
            "native_replay_evidence_id": DIGEST_C,
            "qualification_artifact_id": qualification_id,
            "source_group_ids": qualification.source_group_ids,
            "exact_fallback_verified": True,
        }
    )
    values.update(changes)
    return MaterialBackendEvidenceStatusV1(**values)


@pytest.mark.parametrize(
    ("changes", "message"),
    (
        (
            {"canonical_profile_id": "another-family-v1"},
            "does not belong",
        ),
        (
            {"transport": cast(Any, "native-particle-state-v1")},
            "transport does not match",
        ),
        (
            {"adapter_evidence_id": DIGEST_B, "runtime_id": DIGEST_A},
            "must be supplied together",
        ),
        (
            {
                "adapter_evidence_id": DIGEST_B,
                "qualification_artifact_id": DIGEST_D,
                "source_group_ids": ("source-a", "source-b"),
                "exact_fallback_verified": True,
            },
            "requires native runtime replay",
        ),
        (
            {
                "adapter_evidence_id": DIGEST_B,
                "runtime_id": DIGEST_A,
                "native_replay_evidence_id": DIGEST_C,
                "source_decision_id": DIGEST_D,
            },
            "requires numerical qualification",
        ),
        (
            {
                "adapter_evidence_id": DIGEST_B,
                "runtime_id": DIGEST_A,
                "native_replay_evidence_id": DIGEST_C,
                "qualification_artifact_id": DIGEST_D,
                "source_decision_id": DIGEST_E,
                "downstream_decision_id": DIGEST_B,
                "source_group_ids": ("source-a", "source-b"),
                "exact_fallback_verified": True,
            },
            "requires fresh-object validation",
        ),
        (
            {"source_group_ids": ("source-a", "source-b")},
            "admitted only with numerical qualification",
        ),
        (
            {
                "adapter_evidence_id": DIGEST_B,
                "runtime_id": DIGEST_A,
                "native_replay_evidence_id": DIGEST_C,
                "qualification_artifact_id": DIGEST_D,
                "source_group_ids": ("source-a",),
                "exact_fallback_verified": True,
            },
            "at least two source groups",
        ),
        (
            {
                "adapter_evidence_id": DIGEST_B,
                "runtime_id": DIGEST_A,
                "native_replay_evidence_id": DIGEST_C,
                "qualification_artifact_id": DIGEST_D,
                "source_group_ids": ("source-a", "source-b"),
                "target_group_ids": ("target-a",),
                "exact_fallback_verified": True,
            },
            "admitted only with fresh-object validation",
        ),
        (
            {
                "adapter_evidence_id": DIGEST_B,
                "runtime_id": DIGEST_A,
                "native_replay_evidence_id": DIGEST_C,
                "qualification_artifact_id": DIGEST_D,
                "source_decision_id": DIGEST_E,
                "target_decision_id": DIGEST_B,
                "source_group_ids": ("source-a", "source-b"),
                "exact_fallback_verified": True,
            },
            "requires target_group_ids",
        ),
        (
            {"exact_fallback_verified": True},
            "stage-bearing only with qualification",
        ),
        (
            {
                "adapter_evidence_id": DIGEST_B,
                "runtime_id": DIGEST_A,
                "native_replay_evidence_id": DIGEST_C,
                "qualification_artifact_id": DIGEST_D,
                "source_group_ids": ("source-a", "source-b"),
            },
            "requires exact fallback",
        ),
        (
            {"artifact_id": DIGEST_E},
            "artifact_id does not match",
        ),
    ),
)
def test_status_constructor_rejects_noncontiguous_bindings(
    changes: dict[str, Any],
    message: str,
) -> None:
    values = _status_kwargs()
    values.update(changes)
    with pytest.raises(ValueError, match=message):
        MaterialBackendEvidenceStatusV1(**values)


@pytest.mark.parametrize(
    ("groups", "message"),
    (
        ((" source-a",), "canonical strings"),
        (("source-a", "source-a"), "unique groups"),
    ),
)
def test_status_constructor_rejects_noncanonical_group_rosters(
    groups: tuple[str, ...],
    message: str,
) -> None:
    values = _status_kwargs()
    values["source_group_ids"] = groups
    with pytest.raises(ValueError, match=message):
        MaterialBackendEvidenceStatusV1(**values)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("schema", "wrong-schema", "schema changed"),
        ("schema_version", 2, "version changed"),
        ("claim_boundary", "wrong-boundary", "claim boundary changed"),
    ),
)
def test_payload_header_tampering_is_rejected(
    field: str,
    value: object,
    message: str,
) -> None:
    status = MaterialBackendEvidenceStatusV1(**_status_kwargs())
    payload = status.to_payload()
    payload[field] = value
    with pytest.raises(ValueError, match=message):
        MaterialBackendEvidenceStatusV1.from_payload(payload)


def test_verifier_rejects_evidence_supplied_before_its_stage() -> None:
    qualification, source, target, downstream = _complete_evidence()
    registered = MaterialBackendEvidenceStatusV1(**_status_kwargs())
    with pytest.raises(ValueError, match="pre-qualification"):
        verify_material_backend_evidence_status_v1(
            registered,
            qualification=qualification,
        )

    qualified = _qualified_status(qualification)
    with pytest.raises(ValueError, match="before source competence"):
        verify_material_backend_evidence_status_v1(
            qualified,
            qualification=qualification,
            source_decision=source,
        )

    source_status = _qualified_status(
        qualification,
        source_decision_id=source.decision_id,
    )
    with pytest.raises(ValueError, match="before fresh-object validation"):
        verify_material_backend_evidence_status_v1(
            source_status,
            qualification=qualification,
            source_decision=source,
            target_decision=target,
        )

    target_status = _qualified_status(
        qualification,
        source_decision_id=source.decision_id,
        target_decision_id=target.decision_id,
        target_group_ids=("target-a", "target-b"),
    )
    with pytest.raises(ValueError, match="before downstream-query benefit"):
        verify_material_backend_evidence_status_v1(
            target_status,
            qualification=qualification,
            source_decision=source,
            target_decision=target,
            downstream_decision=downstream,
        )


@pytest.mark.parametrize(
    ("stage", "message"),
    (
        ("source", "source_decision is required"),
        ("target", "target_decision is required"),
        ("downstream", "downstream_decision is required"),
    ),
)
def test_verifier_requires_each_claim_bearing_decision(
    stage: str,
    message: str,
) -> None:
    qualification, source, target, downstream = _complete_evidence()
    changes: dict[str, Any] = {"source_decision_id": source.decision_id}
    supplied: dict[str, Any] = {"qualification": qualification}
    if stage in {"target", "downstream"}:
        changes.update(
            {
                "target_decision_id": target.decision_id,
                "target_group_ids": ("target-a", "target-b"),
            }
        )
        supplied["source_decision"] = source
    if stage == "downstream":
        changes["downstream_decision_id"] = downstream.decision_id
        supplied["target_decision"] = target
    status = _qualified_status(qualification, **changes)
    with pytest.raises(ValueError, match=message):
        verify_material_backend_evidence_status_v1(status, **supplied)


def test_verifier_rejects_group_and_decision_substitution() -> None:
    qualification, source, target, downstream = _complete_evidence()

    wrong_groups = _qualified_status(
        qualification,
        source_group_ids=("other-a", "other-b"),
    )
    with pytest.raises(ValueError, match="source groups"):
        verify_material_backend_evidence_status_v1(
            wrong_groups,
            qualification=qualification,
        )

    wrong_source = _qualified_status(
        qualification,
        source_decision_id=DIGEST_E,
    )
    with pytest.raises(ValueError, match="source decision does not match"):
        verify_material_backend_evidence_status_v1(
            wrong_source,
            qualification=qualification,
            source_decision=source,
        )

    wrong_target = _qualified_status(
        qualification,
        source_decision_id=source.decision_id,
        target_decision_id=DIGEST_E,
        target_group_ids=("target-a",),
    )
    with pytest.raises(ValueError, match="target decision does not match"):
        verify_material_backend_evidence_status_v1(
            wrong_target,
            qualification=qualification,
            source_decision=source,
            target_decision=target,
        )

    wrong_downstream = _qualified_status(
        qualification,
        source_decision_id=source.decision_id,
        target_decision_id=target.decision_id,
        downstream_decision_id=DIGEST_E,
        target_group_ids=("target-a",),
    )
    with pytest.raises(ValueError, match="downstream decision does not match"):
        verify_material_backend_evidence_status_v1(
            wrong_downstream,
            qualification=qualification,
            source_decision=source,
            target_decision=target,
            downstream_decision=downstream,
        )


def test_source_decision_must_be_passing_scientific_non_authorizing_evidence() -> None:
    qualification = _qualification()
    qualification_id = qualification.artifact_id
    assert qualification_id is not None

    failed = _decision(
        role="source-competence",
        parent_key="qualification_artifact_id",
        parent_id=qualification_id,
        salt="5",
        status="fail",
    )
    failed_status = _qualified_status(
        qualification,
        source_decision_id=failed.decision_id,
    )
    with pytest.raises(ValueError, match="must pass"):
        verify_material_backend_evidence_status_v1(
            failed_status,
            qualification=qualification,
            source_decision=failed,
        )

    infrastructure = _decision(
        role="source-competence",
        parent_key="qualification_artifact_id",
        parent_id=qualification_id,
        salt="6",
        run_classification="infrastructure",
    )
    infrastructure_status = _qualified_status(
        qualification,
        source_decision_id=infrastructure.decision_id,
    )
    with pytest.raises(ValueError, match="scientific evidence"):
        verify_material_backend_evidence_status_v1(
            infrastructure_status,
            qualification=qualification,
            source_decision=infrastructure,
        )

    authorized = _decision(
        role="source-competence",
        parent_key="qualification_artifact_id",
        parent_id=qualification_id,
        salt="7",
        run_classification="confirmatory",
        authorized=True,
    )
    authorized_status = _qualified_status(
        qualification,
        source_decision_id=authorized.decision_id,
    )
    with pytest.raises(ValueError, match="cannot authorize"):
        verify_material_backend_evidence_status_v1(
            authorized_status,
            qualification=qualification,
            source_decision=authorized,
        )


def test_target_decision_must_be_confirmatory() -> None:
    qualification, source, _, _ = _complete_evidence()
    nonconfirmatory = _decision(
        role="fresh-object-validation",
        parent_key="source_decision_id",
        parent_id=source.decision_id,
        salt="5",
    )
    status = _qualified_status(
        qualification,
        source_decision_id=source.decision_id,
        target_decision_id=nonconfirmatory.decision_id,
        target_group_ids=("target-a",),
    )
    with pytest.raises(ValueError, match="must be confirmatory"):
        verify_material_backend_evidence_status_v1(
            status,
            qualification=qualification,
            source_decision=source,
            target_decision=nonconfirmatory,
        )


def test_verifier_rejects_wrong_runtime_types() -> None:
    qualification = _qualification()
    status = _qualified_status(
        qualification,
        source_decision_id=DIGEST_E,
    )
    with pytest.raises(TypeError, match="must be EvidenceDecisionV1"):
        verify_material_backend_evidence_status_v1(
            status,
            qualification=qualification,
            source_decision=cast(Any, object()),
        )
    with pytest.raises(TypeError, match="status must be"):
        verify_material_backend_evidence_status_v1(cast(Any, object()))


def test_builder_rejects_profile_mismatch_and_qualification_without_runtime() -> None:
    with pytest.raises(ValueError, match="does not belong"):
        build_material_backend_evidence_status_v1(
            canonical_profile_id="another-family-v1",
            producer_profile_id="jax-fem-quasistatic-v1",
        )
    with pytest.raises(ValueError, match="qualification requires runtime_id"):
        build_material_backend_evidence_status_v1(
            canonical_profile_id="jax-fem-quasistatic-v1",
            producer_profile_id="jax-fem-quasistatic-v1",
            adapter_evidence_id=DIGEST_B,
            native_replay_evidence_id=DIGEST_C,
            qualification=_qualification(),
        )


def test_minimum_stage_and_save_api_validate_argument_types(tmp_path: Path) -> None:
    adapter = build_material_backend_evidence_status_v1(
        canonical_profile_id="jax-fem-quasistatic-v1",
        producer_profile_id="jax-fem-quasistatic-v1",
        adapter_evidence_id=DIGEST_B,
    )
    assert (
        require_material_backend_evidence_stage(adapter, "adapter-tested") is adapter
    )
    with pytest.raises(ValueError, match="minimum_stage"):
        require_material_backend_evidence_stage(
            adapter,
            cast(Any, "not-a-stage"),
        )
    with pytest.raises(TypeError, match="status must be"):
        require_material_backend_evidence_stage(
            cast(Any, object()),
            "adapter-tested",
        )
    with pytest.raises(TypeError, match="status must be"):
        save_material_backend_evidence_status_v1(
            cast(Any, object()),
            tmp_path / "invalid.json",
        )
    with pytest.raises(ValueError, match="literal Boolean"):
        save_material_backend_evidence_status_v1(
            adapter,
            tmp_path / "invalid-overwrite.json",
            overwrite=cast(Any, 1),
        )


def test_payload_constants_remain_bound_to_roundtrip_contract() -> None:
    status = MaterialBackendEvidenceStatusV1(**_status_kwargs())
    payload = status.to_payload()
    assert payload["schema"] == MATERIAL_BACKEND_EVIDENCE_SCHEMA
    assert payload["schema_version"] == MATERIAL_BACKEND_EVIDENCE_VERSION
    assert payload["claim_boundary"] == MATERIAL_BACKEND_EVIDENCE_CLAIM_BOUNDARY
