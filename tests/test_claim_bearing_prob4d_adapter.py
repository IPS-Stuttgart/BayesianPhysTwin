from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from bayesian_phystwin.claim_bearing_prob4d import (
    build_claim_bearing_gauge_aware_batch_from_artifacts,
    build_claim_bearing_gauge_aware_batch_from_observation_belief,
)
from bayesian_phystwin.observation_belief import ObservationBeliefV1
from bayesian_phystwin.physical_linearization import PhysicalLinearizationV1
from bayesian_phystwin.prob4d_causal_lineage import (
    PROB4D_CAUSAL_STREAM_CONTRACT_VERSION,
    validate_claim_bearing_prob4d_observation_belief,
)

FIXTURE_PATH = (
    Path(__file__).parent / "fixtures" / "prob4d_joint_observation_v1.json"
)
PROB4D_REVISION = "d" * 40
GAUGE_CALIBRATION_ID = "5" * 64
POINT_CALIBRATION_ID = "6" * 64


def _provider_manifest() -> dict[str, object]:
    descriptor: dict[str, object] = {
        "provider_name": "prob4d",
        "provider_version": "0.2.0",
        "provider_revision": PROB4D_REVISION,
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
        "provider_revision": PROB4D_REVISION,
        "python_import_boundary": "prob4d.provider_v2",
        "export_mode": "calibrated",
        "claim_bearing": True,
        "calibration_compatibility_validated": True,
        "calibration_artifact_ids": {
            "gauge_artifact_id": GAUGE_CALIBRATION_ID,
            "point_artifact_id": POINT_CALIBRATION_ID,
        },
        "covariance_root_mode": "canonical_eigenspaces",
        "composition_jacobian_mode": "analytic",
        "runtime_revision": {
            "expected_revision": PROB4D_REVISION,
            "observed_revision": PROB4D_REVISION,
            "source": "source_checkout",
            "clean_checkout": True,
            "matched": True,
            "independently_verified": True,
        },
    }


def _fixture_belief() -> ObservationBeliefV1:
    payload = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    descriptor = payload["descriptor"]
    arrays = {
        name: np.asarray(record["values"], dtype=np.dtype(record["dtype"]))
        for name, record in payload["arrays"].items()
    }
    return ObservationBeliefV1(
        case_id=descriptor["case_id"],
        stream_id=descriptor["stream_id"],
        causal_frame_stop=descriptor["causal_frame_stop"],
        view_names=tuple(descriptor["view_names"]),
        window_names=tuple(descriptor["window_names"]),
        factor_names=tuple(descriptor["factor_names"]),
        source_repository=descriptor["source_repository"],
        source_revision=descriptor["source_revision"],
        source_artifact_sha256=descriptor["source_artifact_sha256"],
        metadata=descriptor["metadata"],
        **arrays,
    )


def _claim_bearing_belief() -> ObservationBeliefV1:
    belief = _fixture_belief()
    metadata = deepcopy(dict(belief.metadata))
    metadata.update(
        {
            "prob4d_causal_stream_contract_version": 2,
            "metric_anchor_covariance_in_joint_factor": True,
            "covariance_calibration": {
                "status": "calibrated",
                "gauge_artifact_id": GAUGE_CALIBRATION_ID,
                "point_artifact_id": POINT_CALIBRATION_ID,
                "uncalibrated_exploratory_covariance_allowed": False,
                "pointwise_covariance_fallback_allowed": False,
                "alignment_count": 2,
                "gauge_calibrated_alignment_count": 2,
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
            "calibration_artifact_sha256": "b" * 64,
            "covariance_treatment": "propagated_external_prior",
        }
    )
    return replace(belief, metadata=metadata)


def _state_design(belief: ObservationBeliefV1) -> np.ndarray:
    state = np.zeros((belief.observation_count, 3, 2), dtype=np.float64)
    state[:, 0, 0] = 1.0
    state[:, 1, 1] = 1.0
    return state


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


def test_claim_bearing_validation_requires_explicit_stream_v2() -> None:
    belief = _claim_bearing_belief()
    metadata = deepcopy(dict(belief.metadata))
    del metadata["prob4d_causal_stream_contract_version"]

    with pytest.raises(ValueError, match="explicit causal stream contract version 2"):
        validate_claim_bearing_prob4d_observation_belief(
            replace(belief, metadata=metadata)
        )


def test_claim_bearing_validation_rejects_attested_legacy_covariance() -> None:
    belief = _claim_bearing_belief()
    metadata = deepcopy(dict(belief.metadata))
    metadata["prob4d_causal_stream_contract_version"] = 1
    factors = np.zeros((belief.observation_count, 3, 7), dtype=np.float64)
    factors[:, 0, 0] = 0.001
    legacy = replace(
        belief,
        factor_names=tuple(f"gauge_latent_{index}" for index in range(7)),
        factor_group_ids=belief.window_indices,
        low_rank_factor_m=factors,
        metadata=metadata,
    )

    with pytest.raises(ValueError, match="explicit causal stream contract version 2"):
        validate_claim_bearing_prob4d_observation_belief(legacy)


def test_claim_bearing_validation_cross_checks_calibration_ids() -> None:
    belief = _claim_bearing_belief()
    metadata = deepcopy(dict(belief.metadata))
    metadata["covariance_calibration"]["gauge_artifact_id"] = "7" * 64

    with pytest.raises(ValueError, match="differs from its provider attestation"):
        validate_claim_bearing_prob4d_observation_belief(
            replace(belief, metadata=metadata)
        )


def test_claim_bearing_validation_rejects_covariance_fallback_permission() -> None:
    belief = _claim_bearing_belief()
    metadata = deepcopy(dict(belief.metadata))
    metadata["covariance_calibration"][
        "pointwise_covariance_fallback_allowed"
    ] = True

    with pytest.raises(ValueError, match="pointwise covariance fallback"):
        validate_claim_bearing_prob4d_observation_belief(
            replace(belief, metadata=metadata)
        )


def test_claim_bearing_adapter_validates_before_forming_innovation() -> None:
    belief = _claim_bearing_belief()
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


def test_claim_bearing_adapter_records_validated_provenance() -> None:
    belief = _claim_bearing_belief()
    adapted = _adapt_claim_bearing(belief)
    metadata = adapted.batch.metadata

    assert metadata["prob4d_claim_bearing_provider_v2_validated"] is True
    assert metadata["prob4d_claim_bearing_stream_contract_version"] == (
        PROB4D_CAUSAL_STREAM_CONTRACT_VERSION
    )
    assert metadata["prob4d_claim_bearing_provider_manifest_id"] == (
        _provider_manifest()["manifest_id"]
    )
    assert metadata["prob4d_claim_bearing_calibration_artifact_ids"] == {
        "gauge_artifact_id": GAUGE_CALIBRATION_ID,
        "point_artifact_id": POINT_CALIBRATION_ID,
    }
    assert metadata[
        "prob4d_claim_bearing_runtime_revision_independently_verified"
    ] is True


def test_claim_bearing_artifact_adapter_retains_linearization_binding() -> None:
    belief = _claim_bearing_belief()
    state = _state_design(belief)
    linearization = PhysicalLinearizationV1(
        observation_artifact_id=belief.artifact_id,
        baseline_belief_id="a" * 64,
        action_prefix_id="b" * 64,
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
