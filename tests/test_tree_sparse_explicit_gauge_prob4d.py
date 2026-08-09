from __future__ import annotations

import subprocess
import sys
from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from bayesian_phystwin.explicit_gauge_prob4d import (
    PROB4D_FROZEN_FACTOR_REPOSITORY,
)
from bayesian_phystwin.physical_linearization import PhysicalLinearizationV1
from bayesian_phystwin.prior_aware_gauge_belief import PriorAwareGaugeConfigV1
from bayesian_phystwin.sparse_prior_aware_gauge_belief import (
    SparseGaugeDesignV1,
    TreeSparseGaugeDesignV1,
    update_sparse_prior_aware_gauge_belief,
)
from bayesian_phystwin.tree_sparse_explicit_gauge_prob4d import (
    PROB4D_TREE_PRIOR_SEMANTICS,
    build_claim_bearing_tree_sparse_prob4d_batch,
    load_claim_bearing_tree_sparse_prob4d,
    update_claim_bearing_tree_sparse_prob4d_from_artifacts,
)

_ARTIFACT_ID = "a" * 64
_OBSERVATION_ID = "b" * 64
_PROVIDER_ID = "c" * 64
_PRIOR_ARTIFACT_ID = "d" * 64
_PRIOR_ID = "e" * 64
_GAUGE_CALIBRATION_ID = "f" * 64
_POINT_CALIBRATION_ID = "1" * 64
_REVISION = "2" * 40


def _tree_arrays() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    parents = np.asarray([-1, 0], dtype=np.int64)
    transitions = np.zeros((2, 7, 7), dtype=np.float64)
    transitions[1] = np.eye(7, dtype=np.float64) * 0.25
    scales = np.zeros((2, 7, 7), dtype=np.float64)
    scales[0] = np.eye(7, dtype=np.float64) * 0.035
    scales[1] = np.eye(7, dtype=np.float64) * 0.020
    return parents, transitions, scales


def _dense_tree_covariance(
    transitions: np.ndarray,
    scales: np.ndarray,
) -> np.ndarray:
    root = scales[0] @ scales[0].T
    innovation = scales[1] @ scales[1].T
    child = transitions[1] @ root @ transitions[1].T + innovation
    cross = root @ transitions[1].T
    return np.block([[root, cross], [cross.T, child]])


def _provider_attestation() -> dict[str, Any]:
    calibration_ids = {
        "gauge_artifact_id": _GAUGE_CALIBRATION_ID,
        "point_artifact_id": _POINT_CALIBRATION_ID,
    }
    provider_manifest = {
        "manifest_id": _PROVIDER_ID,
        "provider_api_version": 2,
        "capabilities": [
            "content_addressed_tree_sparse_observation_artifacts",
            "strict_claim_bearing_tree_sparse_observation_loading",
        ],
        "artifact_schema_versions": {
            "TreeSparseObservationArtifactV1": 1,
            "ClaimBearingTreeSparseObservationEnvelopeV1": 1,
        },
    }
    return {
        "claim_bearing": True,
        "export_mode": "calibrated",
        "provider_revision": _REVISION,
        "provider_manifest_id": _PROVIDER_ID,
        "provider_manifest": provider_manifest,
        "calibration_artifact_ids": calibration_ids,
        "runtime_revision": {
            "source": "source_checkout",
            "independently_verified": True,
        },
    }


def _validated_observation(*, second_frame: int = 2) -> SimpleNamespace:
    parents, transitions, scales = _tree_arrays()
    local = np.zeros((4, 3, 7), dtype=np.float64)
    local[:, :, 4:7] = np.eye(3, dtype=np.float64)[None]
    gauge_ids = ("window-0", "window-1")
    factors = SimpleNamespace(
        world_mean_m=np.asarray(
            [
                [0.004, 0.000, 0.000],
                [0.000, -0.003, 0.000],
                [0.000, 0.000, 0.002],
                [0.003, -0.001, 0.001],
            ],
            dtype=np.float64,
        ),
        conditional_world_covariance_m2=np.repeat(
            (np.eye(3, dtype=np.float64) * 5.0e-4)[None],
            4,
            axis=0,
        ),
        local_gauge_jacobian=local,
        gauge_indices=np.asarray([0, 0, 1, 1], dtype=np.int64),
        association_probability=np.asarray([0.9, 0.8, 0.85, 0.75]),
        prior_reliability=np.asarray([0.95, 0.9, 0.88, 0.82]),
        prior_nominal_probability=np.asarray([0.94, 0.94, 0.91, 0.91]),
        composite_weight=np.asarray([0.5, 0.5, 0.4, 0.4]),
        point_ids=np.asarray([10, 11, 20, 21], dtype=np.int64),
        frame_indices=np.asarray([0, 1, second_frame, 3], dtype=np.int64),
        view_ids=("camera-0", "camera-0", "camera-0", "camera-0"),
        factor_ids=("factor-0", "factor-0", "factor-1", "factor-1"),
        correlation_group_ids=(
            "factor-0:camera-0",
            "factor-0:camera-0",
            "factor-1:camera-0",
            "factor-1:camera-0",
        ),
        gauge_ids=gauge_ids,
        causal_frame_stop=5,
        gauge_tree_prior=SimpleNamespace(
            gauge_ids=gauge_ids,
            parent_indices=parents,
            transition_matrices=transitions,
            innovation_scale_tril=scales,
            prior_id=_PRIOR_ID,
            representation_semantics=PROB4D_TREE_PRIOR_SEMANTICS,
        ),
    )
    manifest = SimpleNamespace(
        artifact_id=_OBSERVATION_ID,
        sequence_id="sequence-a",
        case_id="case-a",
        stream_id="prob4d:tree-sparse:camera-0",
        source_repository=PROB4D_FROZEN_FACTOR_REPOSITORY,
        source_revision=_REVISION,
        causal_frame_stop=5,
        observation_count=4,
        gauge_ids=gauge_ids,
        gauge_tree_prior_artifact_id=_PRIOR_ARTIFACT_ID,
        gauge_tree_prior_id=_PRIOR_ID,
    )
    calibration_ids = {
        "gauge_artifact_id": _GAUGE_CALIBRATION_ID,
        "point_artifact_id": _POINT_CALIBRATION_ID,
    }
    envelope = SimpleNamespace(
        artifact_id=_ARTIFACT_ID,
        observation_artifact_id=_OBSERVATION_ID,
        observation_artifact_schema_version=1,
        sequence_id="sequence-a",
        case_id="case-a",
        stream_id="prob4d:tree-sparse:camera-0",
        source_repository=PROB4D_FROZEN_FACTOR_REPOSITORY,
        source_revision=_REVISION,
        causal_frame_stop=5,
        observation_count=4,
        gauge_ids=gauge_ids,
        gauge_tree_prior_artifact_id=_PRIOR_ARTIFACT_ID,
        gauge_tree_prior_id=_PRIOR_ID,
        causal_source_lineage={
            "schema_version": 1,
            "producer": "Prob4D",
            "causal_frame_stop_exclusive": 5,
            "future_prediction_payloads_opened": 0,
            "selected_windows": [
                {
                    "window_id": "window-0",
                    "source_frame_start": 0,
                    "source_frame_stop_exclusive": 2,
                },
                {
                    "window_id": "window-1",
                    "source_frame_start": 2,
                    "source_frame_stop_exclusive": 5,
                },
            ],
        },
        provider_manifest_id=_PROVIDER_ID,
        calibration_artifact_ids=calibration_ids,
        runtime_revision_source="source_checkout",
        runtime_revision_independently_verified=True,
        provider_attestation=_provider_attestation(),
    )
    return SimpleNamespace(
        artifact_id=_ARTIFACT_ID,
        envelope=envelope,
        observation=SimpleNamespace(manifest=manifest, factors=factors),
    )


def _linearization() -> PhysicalLinearizationV1:
    state = np.zeros((4, 3, 2), dtype=np.float64)
    state[0, 0, 0] = 1.0
    state[1, 1, 1] = 1.0
    state[2, 2, 0] = 0.8
    state[3, 0, 1] = 0.7
    query = np.zeros((2, 3, 2), dtype=np.float64)
    query[0, 0, 0] = 1.0
    query[1, 1, 1] = 1.0
    return PhysicalLinearizationV1(
        observation_artifact_id=_ARTIFACT_ID,
        baseline_belief_id="3" * 64,
        action_prefix_id="4" * 64,
        simulator_revision="simulator-revision-a",
        frame_ids=np.asarray([0, 1, 2, 3], dtype=np.int64),
        entity_ids=np.asarray([10, 11, 20, 21], dtype=np.int64),
        view_indices=np.zeros(4, dtype=np.int64),
        window_indices=np.asarray([0, 0, 1, 1], dtype=np.int64),
        state_jacobian=state,
        query_state_jacobian=query,
        physical_response_m=np.asarray(
            [[0.1, 0.0, 0.0], [0.0, 0.1, 0.0]],
            dtype=np.float64,
        ),
    )


def _config() -> PriorAwareGaugeConfigV1:
    return replace(
        PriorAwareGaugeConfigV1(),
        minimum_conditional_information_fraction=0.0,
        minimum_identifiable_fraction=1.0e-8,
        minimum_query_sensitivity_fraction=0.0,
        maximum_state_update_m=1.0,
        maximum_update_to_physical_response_ratio=100.0,
    )


def test_tree_precision_matches_materialized_covariance_inverse() -> None:
    parents, transitions, scales = _tree_arrays()
    local = np.zeros((4, 3, 7), dtype=np.float64)
    design = TreeSparseGaugeDesignV1(
        local_gauge_jacobian=local,
        gauge_indices=np.asarray([0, 0, 1, 1], dtype=np.int64),
        parent_indices=parents,
        transition_matrices=transitions,
        innovation_scale_tril=scales,
        gauge_ids=("window-0", "window-1"),
        prior_id=_PRIOR_ID,
    )

    covariance = _dense_tree_covariance(transitions, scales)
    np.testing.assert_allclose(
        design.prior_information_matrix(),
        np.linalg.inv(covariance),
        atol=2.0e-10,
        rtol=2.0e-10,
    )
    assert design.dense_gauge_prior_avoided_bytes == covariance.nbytes


def test_claim_bearing_tree_adapter_matches_dense_prior_solver() -> None:
    validated = _validated_observation()
    linearization = _linearization()
    physical_prediction = np.zeros((4, 3), dtype=np.float64)
    adapted = build_claim_bearing_tree_sparse_prob4d_batch(
        validated,
        linearization,
        physical_prediction_xyz_m=physical_prediction,
    )
    np.testing.assert_allclose(
        adapted.batch.association_probability,
        np.asarray([0.9, 0.8, 0.85, 0.75]),
    )
    np.testing.assert_allclose(
        adapted.batch.composite_weight,
        np.asarray([0.5, 0.5, 0.4, 0.4]),
    )
    parents, transitions, scales = _tree_arrays()
    dense_design = SparseGaugeDesignV1(
        local_gauge_jacobian=adapted.tree_gauge_design.local_gauge_jacobian,
        gauge_indices=adapted.tree_gauge_design.gauge_indices,
        gauge_prior_covariance=_dense_tree_covariance(transitions, scales),
        gauge_ids=adapted.gauge_ids,
    )

    tree_result = update_sparse_prior_aware_gauge_belief(
        adapted.batch,
        adapted.tree_gauge_design,
        config=_config(),
    )
    dense_result = update_sparse_prior_aware_gauge_belief(
        adapted.batch,
        dense_design,
        config=_config(),
    )

    assert tree_result.inference_admissible
    assert dense_result.inference_admissible
    for name in (
        "state_coefficients",
        "gauge_delta",
        "posterior_covariance",
        "robust_weights",
    ):
        np.testing.assert_allclose(
            getattr(tree_result, name),
            getattr(dense_result, name),
            atol=5.0e-9,
            rtol=5.0e-8,
        )
    assert tree_result.diagnostics["dense_gauge_prior_covariance_materialized"] is False
    assert tree_result.diagnostics["gauge_prior_representation"] == (
        "tree-transition-innovation-information-v1"
    )
    assert (
        adapted.batch.metadata["prob4d_dense_gauge_prior_covariance_materialized"]
        is False
    )


def test_claim_bearing_tree_update_retains_evidence_identities() -> None:
    result = update_claim_bearing_tree_sparse_prob4d_from_artifacts(
        _validated_observation(),
        _linearization(),
        physical_prediction_xyz_m=np.zeros((4, 3), dtype=np.float64),
        config=_config(),
    )

    assert result.observation_artifact_id == _ARTIFACT_ID
    assert result.provider_manifest_id == _PROVIDER_ID
    assert result.calibration_artifact_ids == {
        "gauge_artifact_id": _GAUGE_CALIBRATION_ID,
        "point_artifact_id": _POINT_CALIBRATION_ID,
    }
    assert result.runtime_revision_independently_verified is True
    assert (
        result.result.input_lineage["prob4d_claim_bearing_tree_sparse_bridge_version"]
        == 1
    )


def test_tree_adapter_rejects_row_outside_its_causal_source_window() -> None:
    with pytest.raises(ValueError, match="outside its causal source window"):
        build_claim_bearing_tree_sparse_prob4d_batch(
            _validated_observation(second_frame=1),
            _linearization(),
            physical_prediction_xyz_m=np.zeros((4, 3), dtype=np.float64),
        )


def test_tree_adapter_rejects_provider_capability_substitution() -> None:
    validated = _validated_observation()
    manifest = validated.envelope.provider_attestation["provider_manifest"]
    manifest["capabilities"].remove(
        "strict_claim_bearing_tree_sparse_observation_loading"
    )

    with pytest.raises(ValueError, match="lacks tree-sparse claim capabilities"):
        build_claim_bearing_tree_sparse_prob4d_batch(
            validated,
            _linearization(),
            physical_prediction_xyz_m=np.zeros((4, 3), dtype=np.float64),
        )


def test_tree_adapter_rejects_prior_identity_substitution() -> None:
    validated = _validated_observation()
    validated.observation.factors.gauge_tree_prior.prior_id = "9" * 64

    with pytest.raises(ValueError, match="identity differs"):
        build_claim_bearing_tree_sparse_prob4d_batch(
            validated,
            _linearization(),
            physical_prediction_xyz_m=np.zeros((4, 3), dtype=np.float64),
        )


def test_tree_loader_is_lazy_and_uses_prob4d_strict_surface(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentinel = object()
    calls: list[str] = []

    class Provider:
        @staticmethod
        def load_claim_bearing_tree_sparse_observation(path: object) -> object:
            calls.append(str(path))
            return sentinel

    monkeypatch.setattr(
        "bayesian_phystwin.tree_sparse_explicit_gauge_prob4d.importlib.import_module",
        lambda name: Provider if name == "prob4d.provider_v2_factors" else None,
    )

    assert load_claim_bearing_tree_sparse_prob4d("claim.json") is sentinel
    assert calls == ["claim.json"]


def test_importing_tree_adapter_does_not_import_prob4d() -> None:
    code = """
import sys
import bayesian_phystwin.tree_sparse_explicit_gauge_prob4d
loaded = sorted(name for name in sys.modules if name == "prob4d" or name.startswith("prob4d."))
if loaded:
    raise SystemExit(f"Prob4D imported eagerly: {loaded}")
"""
    subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )
