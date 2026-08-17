from __future__ import annotations

from typing import Any, cast

import pytest

from bayesian_phystwin._portable_contracts import content_id
from bayesian_phystwin.material_backend_qualification_v1 import (
    MATERIAL_BACKEND_QUALIFICATION_CLAIM_BOUNDARY,
    MATERIAL_BACKEND_QUALIFICATION_SCHEMA,
    MATERIAL_BACKEND_QUALIFICATION_VERSION,
    MaterialBackendQualificationV1,
    load_material_backend_qualification_v1,
    require_qualified_material_backend_runtime,
    save_material_backend_qualification_v1,
)

DIGEST_A = "a" * 64
DIGEST_B = "b" * 64
DIGEST_C = "c" * 64
DIGEST_D = "d" * 64


def _qualification(**changes: Any) -> MaterialBackendQualificationV1:
    values: dict[str, Any] = {
        "canonical_profile_id": "jax-fem-quasistatic-v1",
        "producer_profile_id": "jax-fem-quasistatic-v1",
        "transport": "lagrangian-export-v1",
        "runtime_id": DIGEST_A,
        "qualification_protocol_id": DIGEST_B,
        "source_evidence_id": DIGEST_C,
        "source_group_ids": ("object-c", "object-a", "object-b"),
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


def test_passing_qualification_roundtrips_and_binds_exact_runtime(
    tmp_path: Any,
) -> None:
    qualification = _qualification()

    assert qualification.qualified
    assert qualification.failure_reasons == ()
    assert qualification.source_group_ids == (
        "object-a",
        "object-b",
        "object-c",
    )
    assert qualification.artifact_id == content_id(qualification.descriptor())
    payload = qualification.to_payload()
    assert payload["schema"] == MATERIAL_BACKEND_QUALIFICATION_SCHEMA
    assert payload["schema_version"] == MATERIAL_BACKEND_QUALIFICATION_VERSION
    assert payload["claim_boundary"] == MATERIAL_BACKEND_QUALIFICATION_CLAIM_BOUNDARY
    assert payload["qualified"] is True
    with pytest.raises(TypeError):
        qualification.metadata["evidence_role"] = "tampered"  # type: ignore[index]

    path = tmp_path / "qualification.json"
    save_material_backend_qualification_v1(qualification, path)
    loaded = load_material_backend_qualification_v1(path)
    assert loaded == qualification
    assert loaded.to_payload() == payload
    assert (
        require_qualified_material_backend_runtime(
            profile_id=qualification.canonical_profile_id,
            producer_profile_id=qualification.producer_profile_id,
            runtime_id=qualification.runtime_id,
            qualification=qualification,
        )
        is qualification
    )
    with pytest.raises(FileExistsError):
        save_material_backend_qualification_v1(qualification, path)
    save_material_backend_qualification_v1(
        qualification,
        path,
        overwrite=True,
    )


@pytest.mark.parametrize(
    ("changes", "reason"),
    [
        (
            {"units_coordinate_entity_order_valid": False},
            "units-coordinate",
        ),
        ({"deterministic_replay_valid": False}, "deterministic-replay"),
        (
            {"maximum_zero_action_drift_m": 0.002},
            "zero-action-equilibrium",
        ),
        (
            {"maximum_rigid_equivariance_error_m": 0.002},
            "rigid-transform-equivariance",
        ),
        (
            {"time_step_refinement_relative_error": 0.06},
            "time-step-refinement",
        ),
        ({"topology_identity_preserved": False}, "topology-identity"),
        ({"physical_sanity_violations": 1}, "physical-sanity"),
        (
            {"maximum_jacobian_relative_error": 0.06},
            "finite-difference-jacobian",
        ),
        (
            {"source_query_parity_rmse_m": 0.006},
            "source-query-parity",
        ),
        ({"exact_fallback_verified": False}, "exact-fallback"),
        (
            {"protocol_frozen_before_source_outcomes": False},
            "protocol-not-frozen",
        ),
        ({"target_outcomes_used": True}, "target-outcomes-used"),
    ],
)
def test_every_qualification_gate_fails_closed(
    changes: dict[str, Any],
    reason: str,
) -> None:
    qualification = _qualification(**changes)

    assert not qualification.qualified
    assert any(reason in item for item in qualification.failure_reasons)
    with pytest.raises(ValueError, match="not qualified"):
        require_qualified_material_backend_runtime(
            profile_id=qualification.canonical_profile_id,
            producer_profile_id=qualification.producer_profile_id,
            runtime_id=qualification.runtime_id,
            qualification=qualification,
        )


def test_threshold_equality_and_non_gradient_runtime_pass() -> None:
    equality = _qualification(
        maximum_zero_action_drift_m=0.001,
        maximum_rigid_equivariance_error_m=0.001,
        time_step_refinement_relative_error=0.05,
        maximum_jacobian_relative_error=0.05,
        source_query_parity_rmse_m=0.005,
    )
    assert equality.qualified

    no_gradient = _qualification(
        gradient_claimed=False,
        maximum_jacobian_relative_error=None,
        allowed_jacobian_relative_error=None,
    )
    assert no_gradient.qualified


def test_legacy_transport_binds_to_canonical_family() -> None:
    qualification = _qualification(
        canonical_profile_id="genesis-mpm-v1",
        producer_profile_id="genesis-world-mpm-v1",
        transport="lagrangian-export-v1",
    )

    assert qualification.canonical_profile_id == "genesis-mpm-v1"
    assert qualification.producer_profile_id == "genesis-world-mpm-v1"
    assert (
        require_qualified_material_backend_runtime(
            profile_id="genesis-mpm-v1",
            producer_profile_id="genesis-world-mpm-v1",
            runtime_id=qualification.runtime_id,
            qualification=qualification,
        )
        is qualification
    )


def test_multiple_failures_are_retained_in_stable_order() -> None:
    failed = _qualification(
        deterministic_replay_valid=False,
        topology_identity_preserved=False,
        target_outcomes_used=True,
    )

    assert failed.failure_reasons == (
        "deterministic-replay",
        "topology-identity",
        "target-outcomes-used",
    )


def test_invalid_inputs_and_identity_fail_closed(tmp_path: Any) -> None:
    with pytest.raises(ValueError, match="canonical_profile_id"):
        _qualification(canonical_profile_id="unknown-v1")
    with pytest.raises(ValueError, match="does not belong"):
        _qualification(canonical_profile_id="genesis-mpm-v1")
    with pytest.raises(ValueError, match="transport"):
        _qualification(transport="material-trajectory-v1")
    with pytest.raises(ValueError, match="at least two"):
        _qualification(source_group_ids=("object-a",))
    with pytest.raises(ValueError, match="canonical"):
        _qualification(source_group_ids=(" object-a", "object-b"))
    with pytest.raises(ValueError, match="unique"):
        _qualification(source_group_ids=("object-a", "object-a"))
    with pytest.raises(ValueError, match="boolean"):
        _qualification(exact_fallback_verified=cast(Any, 1))
    with pytest.raises(ValueError, match="nonnegative"):
        _qualification(maximum_zero_action_drift_m=-1.0)
    with pytest.raises(ValueError, match="physical_sanity_violations"):
        _qualification(physical_sanity_violations=cast(Any, True))
    with pytest.raises(ValueError, match="required"):
        _qualification(maximum_jacobian_relative_error=None)
    with pytest.raises(ValueError, match="must be null"):
        _qualification(
            gradient_claimed=False,
            maximum_jacobian_relative_error=0.0,
            allowed_jacobian_relative_error=None,
        )
    with pytest.raises(ValueError, match="artifact_id"):
        _qualification(artifact_id="f" * 64)

    qualification = _qualification()
    with pytest.raises(TypeError, match="qualification"):
        save_material_backend_qualification_v1(
            cast(Any, object()),
            tmp_path / "invalid.json",
        )
    with pytest.raises(ValueError, match="literal Boolean"):
        save_material_backend_qualification_v1(
            qualification,
            tmp_path / "invalid.json",
            overwrite=cast(Any, 1),
        )


def test_payload_tampering_and_runtime_mismatches_are_rejected() -> None:
    qualification = _qualification()

    tampered = qualification.to_payload()
    tampered["qualified"] = False
    with pytest.raises(ValueError, match="does not replay"):
        MaterialBackendQualificationV1.from_payload(tampered)

    tampered = qualification.to_payload()
    tampered["runtime_id"] = "e" * 64
    with pytest.raises(ValueError, match="artifact_id"):
        MaterialBackendQualificationV1.from_payload(tampered)

    tampered = qualification.to_payload()
    tampered["schema"] = "other"
    with pytest.raises(ValueError, match="schema changed"):
        MaterialBackendQualificationV1.from_payload(tampered)

    with pytest.raises(TypeError, match="qualification"):
        require_qualified_material_backend_runtime(
            profile_id=qualification.canonical_profile_id,
            producer_profile_id=qualification.producer_profile_id,
            runtime_id=qualification.runtime_id,
            qualification=cast(Any, object()),
        )
    with pytest.raises(ValueError, match="requested profile_id"):
        require_qualified_material_backend_runtime(
            profile_id="genesis-mpm-v1",
            producer_profile_id=qualification.producer_profile_id,
            runtime_id=qualification.runtime_id,
            qualification=qualification,
        )
    with pytest.raises(ValueError, match="canonical profile"):
        other_profile = _qualification(
            canonical_profile_id="genesis-mpm-v1",
            producer_profile_id="genesis-world-mpm-v1",
        )
        require_qualified_material_backend_runtime(
            profile_id=other_profile.canonical_profile_id,
            producer_profile_id=other_profile.producer_profile_id,
            runtime_id=other_profile.runtime_id,
            qualification=qualification,
        )
    with pytest.raises(ValueError, match="runtime_id"):
        require_qualified_material_backend_runtime(
            profile_id=qualification.canonical_profile_id,
            producer_profile_id=qualification.producer_profile_id,
            runtime_id="e" * 64,
            qualification=qualification,
        )
