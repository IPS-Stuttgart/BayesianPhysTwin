from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest

from bayesian_phystwin.claim_bearing_prob4d import (
    build_claim_bearing_gauge_aware_batch_from_artifacts,
    build_claim_bearing_gauge_aware_batch_from_observation_belief,
)
from bayesian_phystwin.observation_belief import ObservationBeliefV1
from bayesian_phystwin.observation_belief_gauge_adapter import (
    build_gauge_aware_batch_from_observation_belief,
)
from bayesian_phystwin.physical_linearization import PhysicalLinearizationV1
from bayesian_phystwin.prob4d_causal_lineage import (
    validate_claim_bearing_prob4d_observation_belief,
    validate_prob4d_causal_observation_belief,
)
from bayesian_phystwin.prob4d_provider_attestation import (
    compute_prob4d_provider_manifest_id,
    validate_prob4d_provider_attestation,
)


def _metadata() -> dict[str, object]:
    return {
        "metric_coordinates": True,
        "metric_units": "m",
        "coordinate_frame": "phystwin-world",
        "metric_gauge_anchor": {
            "artifact_id": "a" * 64,
            "window_id": "window-0",
            "world_frame_id": "phystwin-world",
            "source_artifact_sha256": "1" * 64,
            "calibration_artifact_sha256": "b" * 64,
            "covariance_treatment": "fixed_external_calibration",
        },
        "causal_source_lineage": {
            "schema_version": 1,
            "producer": "Prob4D",
            "motioncrafter_lineage_schema_version": 1,
            "motioncrafter_windowing_model": ("motioncrafter_sliding_window_v1"),
            "source_product": ("independently_decoded_overlap_windows"),
            "causal_frame_stop_exclusive": 6,
            "admissibility_rule": ("source_frame_max < causal_frame_stop_exclusive"),
            "future_prediction_payloads_opened": 0,
            "source_artifact_sha256": "c" * 64,
            "selected_windows": [
                {
                    "window_id": "window-0",
                    "source_frame_start": 0,
                    "source_frame_stop_exclusive": 3,
                    "source_frame_max": 2,
                    "frame_indices_sha256": "2" * 64,
                    "payload_sha256": "1" * 64,
                },
                {
                    "window_id": "window-1",
                    "source_frame_start": 2,
                    "source_frame_stop_exclusive": 5,
                    "source_frame_max": 4,
                    "frame_indices_sha256": "3" * 64,
                    "payload_sha256": "4" * 64,
                },
            ],
        },
    }


def _provider_manifest() -> dict[str, object]:
    descriptor: dict[str, object] = {
        "provider_name": "prob4d",
        "provider_version": "0.2.0",
        "provider_revision": "d" * 40,
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
    manifest_id = hashlib.sha256(
        json.dumps(
            descriptor,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    return {"manifest_id": manifest_id, **descriptor}


def _provider_attestation() -> dict[str, object]:
    manifest = _provider_manifest()
    return {
        "schema_name": "prob4d.provider-attestation",
        "schema_version": 1,
        "provider_api_version": 2,
        "provider_manifest_id": manifest["manifest_id"],
        "provider_manifest": manifest,
        "provider_revision": "d" * 40,
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
            "expected_revision": "d" * 40,
            "observed_revision": "d" * 40,
            "source": "source_checkout",
            "clean_checkout": True,
            "matched": True,
            "independently_verified": True,
        },
    }


def _belief() -> ObservationBeliefV1:
    local = np.repeat(np.eye(3)[None], 4, axis=0) * 1e-5
    factors = np.zeros((4, 3, 7))
    factors[:2, 0, 0] = 0.002
    factors[2:, 1, 1] = 0.003
    return ObservationBeliefV1(
        case_id="case",
        stream_id="prob4d:causal-overlap-window-points",
        causal_frame_stop=6,
        view_names=("camera-0",),
        window_names=("window-0", "window-1"),
        factor_names=tuple(f"gauge_latent_{index}" for index in range(7)),
        source_repository="FlorianPfaff/Prob4D",
        source_revision="d" * 40,
        source_artifact_sha256="c" * 64,
        declared_frame_ids=np.asarray([1, 2, 3, 4]),
        mean_xyz_m=np.asarray(
            [
                [0.0, 0.0, 1.0],
                [0.1, 0.0, 1.0],
                [0.0, 0.1, 1.0],
                [0.1, 0.1, 1.0],
            ]
        ),
        frame_ids=np.asarray([1, 2, 3, 4]),
        entity_ids=np.asarray([0, 1, 0, 1]),
        view_indices=np.zeros(4, dtype=np.int64),
        window_indices=np.asarray([0, 0, 1, 1]),
        correlation_group_ids=np.asarray([0, 0, 1, 1]),
        factor_group_ids=np.asarray([0, 0, 1, 1]),
        prior_reliability=np.asarray([0.9, 0.8, 0.7, 0.6]),
        association_probability=np.ones(4),
        local_covariance_m2=local,
        low_rank_factor_m=factors,
        group_ids=np.asarray([0, 1]),
        group_prior_nominal_probability=np.asarray([0.85, 0.65]),
        group_composite_weight=np.asarray([0.5, 0.5]),
        metadata=_metadata(),
    )


def _attested_belief() -> ObservationBeliefV1:
    belief = _belief()
    metadata = deepcopy(dict(belief.metadata))
    metadata.update(
        {
            "prob4d_causal_stream_contract_version": 2,
            "metric_anchor_covariance_in_joint_factor": True,
            "factor_definition": "one shared joint gauge latent vector",
            "factor_group_semantics": (
                "all rows use one factor group; each window contributes its block "
                "of the same joint gauge covariance root"
            ),
            "joint_cross_window_gauge_covariance_represented": True,
            "gauge_posterior": {
                "model": "sequential_joint_spanning_tree_v1",
                "window_count": 2,
                "full_dimension": 14,
                "exported_factor_rank": 7,
                "retained_covariance_trace_fraction": 1.0,
                "minimum_retained_gauge_trace": 0.999,
                "max_gauge_rank": 64,
                "cross_window_covariance_preserved": True,
                "parent_window_ids": [None, "window-0"],
                "fixed_lag_boundary_covariance_is_approximate": False,
            },
            "covariance_calibration": {
                "status": "calibrated",
                "gauge_artifact_id": "5" * 64,
                "point_artifact_id": "6" * 64,
                "uncalibrated_exploratory_covariance_allowed": False,
                "pointwise_covariance_fallback_allowed": False,
                "alignment_count": 1,
                "gauge_calibrated_alignment_count": 1,
                "covariance_fallback_counts": {},
            },
            "prob4d_provider_attestation": _provider_attestation(),
        }
    )
    metadata["metric_gauge_anchor"].update(
        {
            "schema_name": "prob4d.metric-gauge-anchor",
            "schema_version": 1,
            "case_id": belief.case_id,
            "coordinate_frame": "phystwin-world",
            "world_frame_id": "phystwin-world",
            "metric_units": "m",
            "source_kind": "prefix_registration",
        }
    )
    return replace(
        belief,
        factor_names=tuple(f"joint_gauge_latent_{index:04d}" for index in range(7)),
        factor_group_ids=np.zeros(4, dtype=np.int64),
        metadata=metadata,
    )


def _state_design(belief: ObservationBeliefV1) -> np.ndarray:
    state = np.zeros((belief.observation_count, 3, 2), dtype=np.float64)
    state[:, 0, 0] = 1.0
    state[:, 1, 1] = 1.0
    return state


def _adapt(belief: ObservationBeliefV1):
    state = np.zeros((belief.observation_count, 3, 1))
    state[:, 0, 0] = 1.0
    return build_gauge_aware_batch_from_observation_belief(
        belief,
        physical_prediction_xyz_m=np.zeros_like(belief.mean_xyz_m),
        state_jacobian=state,
        query_state_jacobian=state[:1],
        physical_response_scale_m=0.05,
    )


def _adapt_claim_bearing(belief: ObservationBeliefV1):
    state = _state_design(belief)
    return build_claim_bearing_gauge_aware_batch_from_observation_belief(
        belief,
        physical_prediction_xyz_m=np.zeros_like(belief.mean_xyz_m),
        state_jacobian=state,
        query_state_jacobian=state[:2],
        physical_response_scale_m=0.05,
        state_prior_covariance_m2=np.eye(2) * 1e-3,
    )


def test_valid_prob4d_causal_lineage_is_bound_before_adaptation() -> None:
    belief = _belief()
    validation = validate_prob4d_causal_observation_belief(belief)
    adapted = _adapt(belief)

    assert validation["validated"] is True
    assert validation["window_count"] == 2
    assert validation["provider_attestation_present"] is False
    assert adapted.summary()["prob4d_causal_lineage_validated"] is True
    assert adapted.batch.metadata["prob4d_causal_lineage"] == validation


def test_claim_bearing_provider_v2_attestation_is_independently_validated() -> None:
    belief = _attested_belief()
    validation = validate_claim_bearing_prob4d_observation_belief(belief)
    provider = validation["provider_attestation"]

    assert validation["provider_attestation_present"] is True
    assert validation["provider_attestation_validated"] is True
    assert validation["stream_contract_version"] == 2
    assert validation["stream_contract_version_inferred"] is False
    assert validation["claim_bearing_provider_v2_validated"] is True
    assert provider["claim_bearing"] is True
    assert provider["provider_api_version"] == 2
    assert provider["runtime_revision_independently_verified"] is True
    assert provider["calibration_artifact_ids"] == {
        "gauge_artifact_id": "5" * 64,
        "point_artifact_id": "6" * 64,
    }


def test_strict_provider_v2_validation_rejects_frozen_provider_v1_artifact() -> None:
    with pytest.raises(ValueError, match="provider-v2 attestation is required"):
        validate_claim_bearing_prob4d_observation_belief(_belief())


def test_provider_manifest_payload_tampering_is_rejected_before_adaptation() -> None:
    belief = _attested_belief()
    metadata = deepcopy(dict(belief.metadata))
    metadata["prob4d_provider_attestation"]["provider_manifest"]["provider_version"] = (
        "999"
    )

    with pytest.raises(ValueError, match="manifest ID does not match"):
        _adapt(replace(belief, metadata=metadata))


def test_rehashed_provider_capability_removal_is_rejected() -> None:
    belief = _attested_belief()
    metadata = deepcopy(dict(belief.metadata))
    attestation = metadata["prob4d_provider_attestation"]
    manifest = attestation["provider_manifest"]
    manifest["capabilities"].remove("runtime_revision_attestation")
    manifest["manifest_id"] = compute_prob4d_provider_manifest_id(manifest)
    attestation["provider_manifest_id"] = manifest["manifest_id"]

    with pytest.raises(ValueError, match="required claim-bearing capabilities"):
        validate_prob4d_causal_observation_belief(replace(belief, metadata=metadata))


def test_prob4d_causal_lineage_rejects_changed_cutoff() -> None:
    belief = _belief()
    metadata = deepcopy(dict(belief.metadata))
    metadata["causal_source_lineage"]["causal_frame_stop_exclusive"] = 7

    with pytest.raises(ValueError, match="cutoff differs"):
        _adapt(replace(belief, metadata=metadata))


def test_prob4d_causal_lineage_rejects_future_payload_access() -> None:
    belief = _belief()
    metadata = deepcopy(dict(belief.metadata))
    metadata["causal_source_lineage"]["future_prediction_payloads_opened"] = 1

    with pytest.raises(ValueError, match="opening future payloads"):
        _adapt(replace(belief, metadata=metadata))


def test_prob4d_causal_lineage_rejects_window_mismatch() -> None:
    belief = _belief()
    metadata = deepcopy(dict(belief.metadata))
    metadata["causal_source_lineage"]["selected_windows"][1]["window_id"] = (
        "another-window"
    )

    with pytest.raises(ValueError, match="window order differs"):
        _adapt(replace(belief, metadata=metadata))


def test_prob4d_causal_lineage_rejects_uncertain_anchor_claim() -> None:
    belief = _belief()
    metadata = deepcopy(dict(belief.metadata))
    metadata["metric_gauge_anchor"]["covariance_treatment"] = (
        "marginalized_global_anchor"
    )

    with pytest.raises(ValueError, match="requires a fixed metric anchor"):
        _adapt(replace(belief, metadata=metadata))


def test_prob4d_causal_lineage_rejects_source_digest_mismatch() -> None:
    belief = _belief()
    metadata = deepcopy(dict(belief.metadata))
    metadata["causal_source_lineage"]["source_artifact_sha256"] = "e" * 64

    with pytest.raises(ValueError, match="differs from the descriptor"):
        _adapt(replace(belief, metadata=metadata))


def test_exploratory_provider_v2_attestation_is_valid_but_not_claim_bearing() -> None:
    attestation = _provider_attestation()
    attestation.update(
        export_mode="exploratory",
        claim_bearing=False,
        calibration_compatibility_validated=False,
        calibration_artifact_ids={
            "gauge_artifact_id": None,
            "point_artifact_id": None,
        },
        covariance_root_mode="legacy_eigenvectors",
        composition_jacobian_mode="legacy_finite_difference",
        runtime_revision={
            "expected_revision": "d" * 40,
            "observed_revision": None,
            "source": "unavailable",
            "clean_checkout": None,
            "matched": False,
            "independently_verified": False,
        },
    )

    validated = validate_prob4d_provider_attestation(
        attestation,
        source_revision="d" * 40,
    )

    assert validated["claim_bearing"] is False
    assert validated["calibration_artifact_ids"] == {
        "gauge_artifact_id": None,
        "point_artifact_id": None,
    }
    assert validated["runtime_revision"]["source"] == "unavailable"


def test_provider_attestation_rejects_nonfinite_json() -> None:
    attestation = _provider_attestation()
    attestation["provider_manifest"]["metadata"]["nonfinite"] = float("nan")

    with pytest.raises(ValueError, match="finite JSON data"):
        validate_prob4d_provider_attestation(
            attestation,
            source_revision="d" * 40,
        )


def test_prob4d_causal_lineage_rejects_nonmapping_attestation() -> None:
    belief = _belief()
    metadata = deepcopy(dict(belief.metadata))
    metadata["prob4d_provider_attestation"] = "not-a-mapping"

    with pytest.raises(ValueError, match="attestation must be a mapping"):
        validate_prob4d_causal_observation_belief(replace(belief, metadata=metadata))


def test_claim_bearing_entry_rejects_non_strict_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_validate(*_args: object, **_kwargs: object) -> dict[str, object]:
        return {"strict_causal_stream_contract": False}

    monkeypatch.setattr(
        "bayesian_phystwin.prob4d_causal_lineage."
        "validate_prob4d_causal_observation_belief",
        fake_validate,
    )

    with pytest.raises(ValueError, match="strict causal stream contract"):
        validate_claim_bearing_prob4d_observation_belief(_belief())


def test_claim_bearing_entry_rejects_inferred_stream_v2() -> None:
    belief = _attested_belief()
    metadata = deepcopy(dict(belief.metadata))
    del metadata["prob4d_causal_stream_contract_version"]

    with pytest.raises(ValueError, match="explicit causal stream contract version 2"):
        validate_claim_bearing_prob4d_observation_belief(
            replace(belief, metadata=metadata)
        )


def test_claim_bearing_entry_rejects_attested_legacy_stream_v1() -> None:
    belief = _attested_belief()
    metadata = deepcopy(dict(belief.metadata))
    metadata["prob4d_causal_stream_contract_version"] = 1
    legacy = replace(
        belief,
        factor_names=tuple(f"gauge_latent_{index}" for index in range(7)),
        factor_group_ids=belief.window_indices,
        metadata=metadata,
    )

    with pytest.raises(ValueError, match="explicit causal stream contract version 2"):
        validate_claim_bearing_prob4d_observation_belief(legacy)


def test_claim_bearing_entry_rejects_calibration_identity_drift() -> None:
    belief = _attested_belief()
    metadata = deepcopy(dict(belief.metadata))
    metadata["covariance_calibration"]["gauge_artifact_id"] = "7" * 64

    with pytest.raises(ValueError, match="differs from its provider attestation"):
        validate_claim_bearing_prob4d_observation_belief(
            replace(belief, metadata=metadata)
        )


def test_claim_bearing_entry_rejects_pointwise_fallback_permission() -> None:
    belief = _attested_belief()
    metadata = deepcopy(dict(belief.metadata))
    metadata["covariance_calibration"]["pointwise_covariance_fallback_allowed"] = True

    with pytest.raises(ValueError, match="pointwise covariance fallback"):
        validate_claim_bearing_prob4d_observation_belief(
            replace(belief, metadata=metadata)
        )


def test_claim_bearing_adapter_records_validated_admission() -> None:
    adapted = _adapt_claim_bearing(_attested_belief())
    metadata = adapted.batch.metadata

    assert metadata["prob4d_claim_bearing_provider_v2_validated"] is True
    assert metadata["prob4d_claim_bearing_stream_contract_version"] == 2
    assert (
        metadata["prob4d_claim_bearing_provider_manifest_id"]
        == (_provider_manifest()["manifest_id"])
    )
    assert metadata["prob4d_claim_bearing_calibration_artifact_ids"] == {
        "gauge_artifact_id": "5" * 64,
        "point_artifact_id": "6" * 64,
    }


def test_claim_bearing_adapter_validates_before_innovation_shape() -> None:
    belief = _attested_belief()
    metadata = deepcopy(dict(belief.metadata))
    metadata["covariance_calibration"]["point_artifact_id"] = "8" * 64
    invalid = replace(belief, metadata=metadata)
    state = _state_design(invalid)

    with pytest.raises(ValueError, match="differs from its provider attestation"):
        build_claim_bearing_gauge_aware_batch_from_observation_belief(
            invalid,
            physical_prediction_xyz_m=np.zeros((1, 3)),
            state_jacobian=state,
            query_state_jacobian=state[:2],
            physical_response_scale_m=0.05,
        )


def test_claim_bearing_artifact_adapter_preserves_linearization_binding() -> None:
    belief = _attested_belief()
    state = _state_design(belief)
    linearization = PhysicalLinearizationV1(
        observation_artifact_id=belief.artifact_id,
        baseline_belief_id="8" * 64,
        action_prefix_id="9" * 64,
        simulator_revision="simulator-revision",
        frame_ids=belief.frame_ids,
        entity_ids=belief.entity_ids,
        view_indices=belief.view_indices,
        window_indices=belief.window_indices,
        state_jacobian=state,
        query_state_jacobian=state[:2],
        physical_response_m=np.asarray([[0.01, 0.0, 0.0], [0.0, 0.01, 0.0]]),
    )

    adapted = build_claim_bearing_gauge_aware_batch_from_artifacts(
        belief,
        linearization,
        physical_prediction_xyz_m=np.zeros_like(belief.mean_xyz_m),
        state_prior_covariance_m2=np.eye(2) * 1e-3,
    )

    assert adapted.batch.metadata["prob4d_claim_bearing_provider_v2_validated"]
    assert adapted.batch.metadata["row_alignment_verified"] is True
    assert adapted.batch.metadata["linearization_artifact_id"] == (
        linearization.artifact_id
    )


def test_claim_bearing_adapter_rejects_malformed_validation_summary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "bayesian_phystwin.claim_bearing_prob4d."
        "validate_claim_bearing_prob4d_observation_belief",
        lambda _belief: {
            "provider_attestation": "not-a-mapping",
            "claim_bearing_covariance_calibration": {},
        },
    )

    with pytest.raises(ValueError, match="provider attestation must be a mapping"):
        _adapt_claim_bearing(_attested_belief())
