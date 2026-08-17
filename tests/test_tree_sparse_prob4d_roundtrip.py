from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("prob4d")

from prob4d.gauge_tree_prior import GaugeTreeSquareRootPriorV1
from prob4d.provider_attestation import build_provider_attestation
from prob4d.provider_v2_tree_sparse_artifact import (
    seal_claim_bearing_tree_sparse_observation,
)
from prob4d.provider_v2_tree_sparse_manifest import (
    prob4d_tree_sparse_provider_manifest,
)
from prob4d.tree_sparse_observation_factors import (
    build_tree_sparse_observation_factors,
)

from bayesian_phystwin.physical_linearization import PhysicalLinearizationV1
from bayesian_phystwin.prior_aware_gauge_belief import PriorAwareGaugeConfigV1
from bayesian_phystwin.tree_sparse_explicit_gauge_prob4d import (
    load_claim_bearing_tree_sparse_prob4d,
    update_claim_bearing_tree_sparse_prob4d_from_path,
)

_PROB4D_REVISION = "b2953319e9b7afea04013c214c502b38c5a83489"
_GAUGE_CALIBRATION_ID = "a" * 64
_POINT_CALIBRATION_ID = "b" * 64


def _runtime() -> dict[str, object]:
    return {
        "expected_revision": _PROB4D_REVISION,
        "observed_revision": _PROB4D_REVISION,
        "source": "source_checkout",
        "clean_checkout": True,
        "matched": True,
        "independently_verified": True,
    }


def _attestation() -> dict[str, object]:
    return build_provider_attestation(
        provider_manifest=prob4d_tree_sparse_provider_manifest(
            provider_revision=_PROB4D_REVISION
        ),
        provider_revision=_PROB4D_REVISION,
        export_mode="calibrated",
        calibration_compatibility_validated=True,
        calibration_artifact_ids={
            "gauge_artifact_id": _GAUGE_CALIBRATION_ID,
            "point_artifact_id": _POINT_CALIBRATION_ID,
        },
        covariance_root_mode="canonical_eigenspaces",
        composition_jacobian_mode="analytic",
        runtime_revision=_runtime(),
    )


def _lineage() -> dict[str, object]:
    return {
        "schema_version": 1,
        "producer": "Prob4D",
        "motioncrafter_lineage_schema_version": 1,
        "motioncrafter_windowing_model": "motioncrafter-sliding-window-v1",
        "source_product": "independently_decoded_overlap_windows",
        "causal_frame_stop_exclusive": 5,
        "admissibility_rule": "source_frame_max < causal_frame_stop_exclusive",
        "future_prediction_payloads_opened": 0,
        "selected_windows": [
            {
                "window_id": "window-0",
                "source_frame_start": 0,
                "source_frame_stop_exclusive": 2,
                "source_frame_max": 1,
                "frame_indices_sha256": "c" * 64,
                "payload_sha256": "d" * 64,
            },
            {
                "window_id": "window-1",
                "source_frame_start": 2,
                "source_frame_stop_exclusive": 5,
                "source_frame_max": 4,
                "frame_indices_sha256": "e" * 64,
                "payload_sha256": "f" * 64,
            },
        ],
        "source_artifact_sha256": "1" * 64,
        "source_digest_scope": "real-cross-repository-roundtrip-fixture",
    }


def _prior() -> GaugeTreeSquareRootPriorV1:
    transitions = np.zeros((2, 7, 7), dtype=np.float64)
    transitions[1] = np.eye(7, dtype=np.float64) * 0.2
    innovations = np.stack(
        (
            np.eye(7, dtype=np.float64) * 2.0e-4,
            np.eye(7, dtype=np.float64) * 3.0e-4,
        )
    )
    return GaugeTreeSquareRootPriorV1.from_transition_covariances(
        gauge_ids=("window-0", "window-1"),
        parent_indices=np.asarray([-1, 0], dtype=np.int64),
        transition_matrices=transitions,
        innovation_covariances=innovations,
    )


def _factors():
    local_jacobian = np.zeros((4, 3, 7), dtype=np.float64)
    local_jacobian[:, :, 4:7] = np.eye(3, dtype=np.float64)[None]
    return build_tree_sparse_observation_factors(
        _prior(),
        world_mean_m=np.asarray(
            [
                [0.0, 0.0, 1.0],
                [0.2, 0.0, 1.1],
                [0.1, 0.2, 1.2],
                [0.3, 0.1, 1.3],
            ],
            dtype=np.float64,
        ),
        conditional_world_covariance_m2=np.repeat(
            np.eye(3, dtype=np.float64)[None] * 1.0e-3,
            4,
            axis=0,
        ),
        local_gauge_jacobian=local_jacobian,
        gauge_indices=np.asarray([0, 0, 1, 1], dtype=np.int64),
        association_probability=np.asarray([0.9, 0.8, 0.85, 0.75]),
        prior_reliability=np.asarray([0.95, 0.9, 0.88, 0.82]),
        prior_nominal_probability=np.asarray([0.94, 0.94, 0.91, 0.91]),
        composite_weight=np.asarray([0.5, 0.5, 0.4, 0.4]),
        point_ids=np.asarray([10, 11, 20, 21], dtype=np.int64),
        frame_indices=np.asarray([0, 0, 2, 2], dtype=np.int64),
        view_ids=("camera-0", "camera-0", "camera-0", "camera-0"),
        factor_ids=("factor-0", "factor-0", "factor-1", "factor-1"),
        correlation_group_ids=(
            "factor-0:camera-0",
            "factor-0:camera-0",
            "factor-1:camera-0",
            "factor-1:camera-0",
        ),
        causal_frame_stop=5,
    )


def _linearization(artifact_id: str) -> PhysicalLinearizationV1:
    state = np.zeros((4, 3, 2), dtype=np.float64)
    state[0, 0, 0] = 1.0
    state[1, 0, 0] = -1.0
    state[2, 1, 1] = 1.0
    state[3, 1, 1] = -1.0
    query = np.zeros((2, 3, 2), dtype=np.float64)
    query[0, 0, 0] = 1.0
    query[1, 1, 1] = 1.0
    return PhysicalLinearizationV1(
        observation_artifact_id=artifact_id,
        baseline_belief_id="2" * 64,
        action_prefix_id="3" * 64,
        simulator_revision="real-prob4d-roundtrip-simulator-v1",
        frame_ids=np.asarray([0, 0, 2, 2], dtype=np.int64),
        entity_ids=np.asarray([10, 11, 20, 21], dtype=np.int64),
        view_indices=np.zeros(4, dtype=np.int64),
        window_indices=np.asarray([0, 0, 1, 1], dtype=np.int64),
        state_jacobian=state,
        query_state_jacobian=query,
        physical_response_m=np.asarray(
            [[0.02, 0.0, 0.0], [0.0, 0.02, 0.0]],
            dtype=np.float64,
        ),
    )


def _config() -> PriorAwareGaugeConfigV1:
    return PriorAwareGaugeConfigV1(
        minimum_conditional_information_fraction=0.0,
        minimum_identifiable_fraction=1.0e-8,
        minimum_query_sensitivity_fraction=0.0,
        maximum_state_update_m=1.0,
        maximum_update_to_physical_response_ratio=100.0,
    )


def test_real_prob4d_serialized_tree_sparse_roundtrip(tmp_path: Path) -> None:
    factors = _factors()
    envelope_path = tmp_path / "claim.json"
    produced = seal_claim_bearing_tree_sparse_observation(
        factors,
        envelope_path,
        sequence_id="sequence-a",
        case_id="case-a",
        stream_id="prob4d:tree-sparse:camera-0",
        source_revision=_PROB4D_REVISION,
        causal_source_lineage=_lineage(),
        provider_attestation=_attestation(),
        artifact_metadata={"split": "cross-repository-integration"},
        metadata={"protocol": "prob4d-to-bayesian-phystwin-tree-sparse-v1"},
    )

    admitted = load_claim_bearing_tree_sparse_prob4d(envelope_path)
    assert admitted.artifact_id == produced.artifact_id
    assert admitted.envelope.source_revision == _PROB4D_REVISION

    physical_prediction = np.asarray(factors.world_mean_m).copy()
    physical_prediction[0, 0] -= 0.006
    physical_prediction[1, 0] += 0.006
    physical_prediction[2, 1] -= 0.005
    physical_prediction[3, 1] += 0.005
    result = update_claim_bearing_tree_sparse_prob4d_from_path(
        envelope_path,
        _linearization(produced.artifact_id),
        physical_prediction_xyz_m=physical_prediction,
        config=_config(),
        metadata={"integration_source": "real-prob4d-serialized-artifact"},
    )

    assert result.inference_admissible
    assert result.observation_artifact_id == produced.artifact_id
    assert result.provider_manifest_id == produced.provider_manifest_id
    assert result.runtime_revision_independently_verified is True
    assert result.result.input_lineage["integration_source"] == (
        "real-prob4d-serialized-artifact"
    )
    assert (
        result.result.diagnostics["dense_gauge_prior_covariance_materialized"] is False
    )
    assert result.result.diagnostics["gauge_prior_representation"] == (
        "tree-transition-innovation-information-v1"
    )
