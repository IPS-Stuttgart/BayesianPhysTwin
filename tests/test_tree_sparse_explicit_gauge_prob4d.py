from __future__ import annotations

import copy
import subprocess
import sys
from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

import bayesian_phystwin.sparse_prior_aware_gauge_belief as sparse_module
import bayesian_phystwin.tree_sparse_explicit_gauge_prob4d as tree_module
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


def _build(validated: Any, **kwargs: Any) -> Any:
    return build_claim_bearing_tree_sparse_prob4d_batch(
        validated,
        _linearization(),
        physical_prediction_xyz_m=np.zeros((4, 3), dtype=np.float64),
        **kwargs,
    )


def _lineage_bounds() -> dict[str, tuple[int, int]]:
    return {"window-0": (0, 2), "window-1": (2, 5)}


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
        _build(_validated_observation(second_frame=1))


def test_tree_adapter_rejects_provider_capability_substitution() -> None:
    validated = _validated_observation()
    manifest = validated.envelope.provider_attestation["provider_manifest"]
    manifest["capabilities"].remove(
        "strict_claim_bearing_tree_sparse_observation_loading"
    )

    with pytest.raises(ValueError, match="lacks tree-sparse claim capabilities"):
        _build(validated)


def test_tree_adapter_rejects_prior_identity_substitution() -> None:
    validated = _validated_observation()
    validated.observation.factors.gauge_tree_prior.prior_id = "9" * 64

    with pytest.raises(ValueError, match="identity differs"):
        _build(validated)


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


@pytest.mark.parametrize(
    ("operation", "error_type", "message"),
    [
        (
            lambda: tree_module._sequence("not-a-sequence", name="value"),
            TypeError,
            "must be a sequence",
        ),
        (
            lambda: tree_module._integer_vector(
                np.asarray([True]),
                name="values",
                count=1,
            ),
            TypeError,
            "integer vector",
        ),
        (
            lambda: tree_module._integer_vector(
                np.asarray([-1]),
                name="values",
                count=1,
                minimum=0,
            ),
            ValueError,
            "at least 0",
        ),
        (
            lambda: tree_module._float_array(
                np.zeros(2),
                name="values",
                shape=(1,),
            ),
            ValueError,
            "shape",
        ),
        (
            lambda: tree_module._float_array(
                np.asarray([np.nan]),
                name="values",
                shape=(1,),
            ),
            ValueError,
            "finite",
        ),
        (
            lambda: tree_module._design_array(
                np.zeros((1, 2, 1)),
                name="design",
                count=1,
            ),
            ValueError,
            "shape",
        ),
        (
            lambda: tree_module._design_array(
                np.full((1, 3, 1), np.nan),
                name="design",
                count=1,
            ),
            ValueError,
            "finite",
        ),
        (
            lambda: tree_module._probability_vector(
                np.asarray([0.0]),
                name="probability",
                count=1,
                strictly_positive=True,
            ),
            ValueError,
            "must lie in",
        ),
    ],
)
def test_tree_low_level_validators_reject_malformed_values(
    operation: Any,
    error_type: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error_type, match=message):
        operation()


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("manifest_id", "9" * 64, "identity differs"),
        ("provider_api_version", 1, "API version 2"),
        ("capabilities", "not-a-sequence", "must be a sequence"),
    ],
)
def test_extended_provider_manifest_rejects_identity_api_and_shape(
    field: str,
    value: object,
    message: str,
) -> None:
    validated = _validated_observation()
    manifest = validated.envelope.provider_attestation["provider_manifest"]
    manifest[field] = value
    with pytest.raises((TypeError, ValueError), match=message):
        tree_module._validate_extended_provider_manifest(
            validated.envelope,
            provider_manifest_id=_PROVIDER_ID,
        )


@pytest.mark.parametrize(
    ("schema_name", "message"),
    [
        ("TreeSparseObservationArtifactV1", "artifact version"),
        ("ClaimBearingTreeSparseObservationEnvelopeV1", "envelope version"),
    ],
)
def test_extended_provider_manifest_rejects_schema_version_substitution(
    schema_name: str,
    message: str,
) -> None:
    validated = _validated_observation()
    versions = validated.envelope.provider_attestation["provider_manifest"][
        "artifact_schema_versions"
    ]
    versions[schema_name] = 2
    with pytest.raises(ValueError, match=message):
        tree_module._validate_extended_provider_manifest(
            validated.envelope,
            provider_manifest_id=_PROVIDER_ID,
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("schema_version", 2, "schema version"),
        ("producer", "Other", "producer"),
        ("causal_frame_stop_exclusive", 4, "envelope cutoff"),
        ("future_prediction_payloads_opened", 1, "future prediction"),
    ],
)
def test_tree_lineage_rejects_header_substitution(
    field: str,
    value: object,
    message: str,
) -> None:
    lineage = copy.deepcopy(_validated_observation().envelope.causal_source_lineage)
    lineage[field] = value
    with pytest.raises(ValueError, match=message):
        tree_module._lineage_bounds(
            lineage,
            gauge_ids=("window-0", "window-1"),
            causal_frame_stop=5,
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("duplicate", "duplicate windows"),
        ("invalid", "invalid window"),
        ("missing", "gauges differ"),
    ],
)
def test_tree_lineage_rejects_invalid_window_registry(
    mutation: str,
    message: str,
) -> None:
    lineage = copy.deepcopy(_validated_observation().envelope.causal_source_lineage)
    windows = lineage["selected_windows"]
    if mutation == "duplicate":
        windows.append(copy.deepcopy(windows[0]))
    elif mutation == "invalid":
        windows[1]["source_frame_start"] = 5
        windows[1]["source_frame_stop_exclusive"] = 5
    else:
        windows.pop()
    with pytest.raises(ValueError, match=message):
        tree_module._lineage_bounds(
            lineage,
            gauge_ids=("window-0", "window-1"),
            causal_frame_stop=5,
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("gauge-order", "gauge order"),
        ("semantics", "semantics changed"),
        ("root", "unique root"),
        ("second-root", "only the first"),
        ("future-parent", "parent must precede"),
        ("root-transition", "root transition"),
        ("upper-triangular", "lower triangular"),
        ("nonpositive-diagonal", "positive diagonal"),
    ],
)
def test_tree_prior_rejects_invalid_topology_and_factors(
    mutation: str,
    message: str,
) -> None:
    prior = copy.deepcopy(_validated_observation().observation.factors.gauge_tree_prior)
    if mutation == "gauge-order":
        prior.gauge_ids = tuple(reversed(prior.gauge_ids))
    elif mutation == "semantics":
        prior.representation_semantics = "other"
    elif mutation == "root":
        prior.parent_indices[0] = 0
    elif mutation == "second-root":
        prior.parent_indices[1] = -1
    elif mutation == "future-parent":
        prior.parent_indices[1] = 1
    elif mutation == "root-transition":
        prior.transition_matrices[0, 0, 0] = 1.0
    elif mutation == "upper-triangular":
        prior.innovation_scale_tril[1, 0, 1] = 0.1
    else:
        prior.innovation_scale_tril[1, 0, 0] = 0.0
    with pytest.raises(ValueError, match=message):
        tree_module._validate_tree_prior(
            prior,
            gauge_ids=("window-0", "window-1"),
            prior_id=_PRIOR_ID,
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("asymmetric", "symmetric"),
        ("indefinite", "positive definite"),
        ("unknown-gauge", "unknown gauge"),
        ("future-frame", "exclusive causal"),
        ("short-identities", "one value per row"),
        ("row-gauges", "row gauges differ"),
        ("row-cutoff", "envelope cutoff"),
        ("group-settings", "changed nominal probability"),
    ],
)
def test_tree_rows_reject_invalid_covariance_identity_and_power(
    mutation: str,
    message: str,
) -> None:
    factors = copy.deepcopy(_validated_observation().observation.factors)
    if mutation == "asymmetric":
        factors.conditional_world_covariance_m2[0, 0, 1] = 0.1
    elif mutation == "indefinite":
        factors.conditional_world_covariance_m2[0, 0, 0] = -1.0
    elif mutation == "unknown-gauge":
        factors.gauge_indices[0] = 2
    elif mutation == "future-frame":
        factors.frame_indices[0] = 5
    elif mutation == "short-identities":
        factors.view_ids = factors.view_ids[:-1]
    elif mutation == "row-gauges":
        factors.gauge_ids = tuple(reversed(factors.gauge_ids))
    elif mutation == "row-cutoff":
        factors.causal_frame_stop = 4
    else:
        factors.prior_nominal_probability[1] = 0.5
    with pytest.raises(ValueError, match=message):
        tree_module._validate_rows(
            factors,
            observation_count=4,
            gauge_ids=("window-0", "window-1"),
            causal_frame_stop=5,
            lineage_bounds=_lineage_bounds(),
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("validated-id", "differs from its envelope"),
        ("schema", "schema version 1"),
        ("repository", "frozen Prob4D identity"),
        ("duplicate-gauges", "must be unique"),
        ("runtime", "not independently verified"),
        ("manifest-field", "field case_id"),
        ("manifest-gauges", "gauge order differs"),
    ],
)
def test_claim_envelope_rejects_cross_layer_substitution(
    mutation: str,
    message: str,
) -> None:
    validated = _validated_observation()
    if mutation == "validated-id":
        validated.artifact_id = "9" * 64
    elif mutation == "schema":
        validated.envelope.observation_artifact_schema_version = 2
    elif mutation == "repository":
        validated.envelope.source_repository = "Other/Prob4D"
    elif mutation == "duplicate-gauges":
        validated.envelope.gauge_ids = ("window-0", "window-0")
    elif mutation == "runtime":
        validated.envelope.runtime_revision_independently_verified = False
    elif mutation == "manifest-field":
        validated.observation.manifest.case_id = "other-case"
    else:
        validated.observation.manifest.gauge_ids = ("window-1", "window-0")
    with pytest.raises(ValueError, match=message):
        _build(validated)


@pytest.mark.parametrize(
    ("mutation", "error_type", "message"),
    [
        ("batch", TypeError, "batch must"),
        ("design", TypeError, "tree_gauge_design must"),
        ("gauge-ids", ValueError, "gauge IDs differ"),
    ],
)
def test_adapter_result_revalidates_composed_types_and_gauge_order(
    mutation: str,
    error_type: type[Exception],
    message: str,
) -> None:
    adapted = _build(_validated_observation())
    with pytest.raises(error_type, match=message):
        if mutation == "batch":
            replace(adapted, batch=object())
        elif mutation == "design":
            replace(adapted, tree_gauge_design=object())
        else:
            replace(adapted, gauge_ids=tuple(reversed(adapted.gauge_ids)))


def test_adapter_rejects_reserved_metadata_collision() -> None:
    with pytest.raises(ValueError, match="overrides reserved"):
        _build(
            _validated_observation(),
            metadata={"observation_artifact_id": "not-authoritative"},
        )


def test_adapter_rejects_nonmapping_canonical_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(tree_module, "plain_json", lambda value: [])
    with pytest.raises(TypeError, match="lost its mapping type"):
        _build(_validated_observation(), metadata={"note": "test"})


def test_tree_design_rejects_nonstring_prior_id() -> None:
    parents, transitions, scales = _tree_arrays()
    with pytest.raises(TypeError, match="prior_id must be a string"):
        TreeSparseGaugeDesignV1(
            local_gauge_jacobian=np.zeros((4, 3, 7)),
            gauge_indices=np.asarray([0, 0, 1, 1]),
            parent_indices=parents,
            transition_matrices=transitions,
            innovation_scale_tril=scales,
            gauge_ids=("window-0", "window-1"),
            prior_id=object(),
        )


def test_tree_rejection_materializes_lazy_prior_only_for_fallback_result() -> None:
    adapted = _build(_validated_observation())
    unsupported = replace(
        adapted.batch,
        prior_reliability=np.zeros(4),
    )
    result = update_sparse_prior_aware_gauge_belief(
        unsupported,
        adapted.tree_gauge_design,
        config=_config(),
    )
    assert result.inference_admissible is False
    assert result.reason == "no-identifiable-query-state"
    assert result.posterior_covariance.shape == (16, 16)
    assert np.all(np.isfinite(result.posterior_covariance))


def test_tree_prior_keeps_optional_nuisance_blocks_in_precision_form() -> None:
    shared = np.zeros((4, 3, 1), dtype=np.float64)
    shared[:, 0, 0] = 1.0
    view = np.zeros((4, 3, 1), dtype=np.float64)
    view[:, 1, 0] = 1.0
    anchor_state = np.zeros((1, 3, 2), dtype=np.float64)
    anchor_state[0, 0, 0] = 1.0
    anchor_bias = np.zeros((1, 3, 1), dtype=np.float64)
    anchor_bias[0, 2, 0] = 1.0
    adapted = _build(
        _validated_observation(),
        shared_bias_jacobian=shared,
        view_bias_jacobian=view,
        anchor_innovation_m=np.zeros((1, 3), dtype=np.float64),
        anchor_covariance_m2=np.eye(3, dtype=np.float64)[None] * 1.0e-3,
        anchor_state_jacobian=anchor_state,
        anchor_correlation_group_ids=("anchor",),
        anchor_prior_reliability=np.ones(1),
        anchor_prior_nominal_probability=np.ones(1),
        anchor_composite_weight=np.ones(1),
        anchor_bias_jacobian=anchor_bias,
        anchor_bias_prior_covariance=np.eye(1) * 1.0e-2,
    )
    state, nuisance_precision, lazy_prior = sparse_module._prior_covariances(
        adapted.batch,
        adapted.tree_gauge_design,
        _config(),
    )
    assert state.shape == (2, 2)
    assert nuisance_precision.shape == (17, 17)
    assert np.asarray(lazy_prior).shape == (19, 19)


def test_tree_loader_reports_missing_prob4d_installation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing_provider(name: str) -> Any:
        raise ImportError(name)

    monkeypatch.setattr(tree_module.importlib, "import_module", missing_provider)
    with pytest.raises(ImportError, match="requires a compatible Prob4D"):
        load_claim_bearing_tree_sparse_prob4d("claim.json")


def test_tree_loader_reports_missing_strict_loader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(tree_module.importlib, "import_module", lambda name: object())
    with pytest.raises(ImportError, match="lacks the claim-bearing"):
        load_claim_bearing_tree_sparse_prob4d("claim.json")


def test_update_from_path_loads_then_runs_claim_bearing_update(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        tree_module,
        "load_claim_bearing_tree_sparse_prob4d",
        lambda path: _validated_observation(),
    )
    result = tree_module.update_claim_bearing_tree_sparse_prob4d_from_path(
        "claim.json",
        _linearization(),
        physical_prediction_xyz_m=np.zeros((4, 3), dtype=np.float64),
        config=_config(),
    )
    assert result.observation_artifact_id == _ARTIFACT_ID


def test_structured_rejection_does_not_materialize_dense_covariance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from bayesian_phystwin.sparse_prior_aware_gauge_belief import (
        update_sparse_prior_aware_gauge_belief_structured,
    )
    from bayesian_phystwin.structured_gauge_aware_result import (
        PRECISION_BACKED_COVARIANCE_REPRESENTATION,
        PrecisionBackedCovarianceV1,
    )

    adapted = _build(_validated_observation())
    unsupported = replace(adapted.batch, prior_reliability=np.zeros(4))

    def fail_materialization(self, *, maximum_bytes=None):
        raise AssertionError("structured rejection materialized covariance")

    monkeypatch.setattr(
        PrecisionBackedCovarianceV1,
        "materialize",
        fail_materialization,
    )
    result = update_sparse_prior_aware_gauge_belief_structured(
        unsupported,
        adapted.tree_gauge_design,
        config=_config(),
    )

    assert not result.inference_admissible
    assert result.covariance_representation == (
        PRECISION_BACKED_COVARIANCE_REPRESENTATION
    )
    assert result.dense_covariance_materialized is False
    assert result.diagnostics["result_dense_covariance_materialized"] is False
    assert len(result.result_id) == 64


def test_structured_rejection_materializes_only_through_explicit_conversion() -> None:
    from bayesian_phystwin.sparse_prior_aware_gauge_belief import (
        update_sparse_prior_aware_gauge_belief,
        update_sparse_prior_aware_gauge_belief_structured,
    )

    adapted = _build(_validated_observation())
    unsupported = replace(adapted.batch, prior_reliability=np.zeros(4))
    structured = update_sparse_prior_aware_gauge_belief_structured(
        unsupported,
        adapted.tree_gauge_design,
        config=_config(),
    )
    legacy = update_sparse_prior_aware_gauge_belief(
        unsupported,
        adapted.tree_gauge_design,
        config=_config(),
    )

    with pytest.raises(MemoryError, match="exceeding"):
        structured.materialize_posterior_covariance(
            maximum_bytes=structured.estimated_dense_covariance_bytes - 1
        )
    converted = structured.to_legacy()
    np.testing.assert_allclose(
        converted.posterior_covariance,
        legacy.posterior_covariance,
        atol=1e-12,
        rtol=1e-12,
    )
    assert converted.reason == legacy.reason


def test_structured_acceptance_is_numerically_identical_to_legacy() -> None:
    from bayesian_phystwin.sparse_prior_aware_gauge_belief import (
        update_sparse_prior_aware_gauge_belief,
        update_sparse_prior_aware_gauge_belief_structured,
    )
    from bayesian_phystwin.structured_gauge_aware_result import DenseCovarianceV1

    adapted = _build(_validated_observation())
    structured = update_sparse_prior_aware_gauge_belief_structured(
        adapted.batch,
        adapted.tree_gauge_design,
        config=_config(),
    )
    legacy = update_sparse_prior_aware_gauge_belief(
        adapted.batch,
        adapted.tree_gauge_design,
        config=_config(),
    )

    assert structured.inference_admissible
    assert isinstance(structured.covariance, DenseCovarianceV1)
    assert structured.dense_covariance_materialized is True
    converted = structured.to_legacy()
    for name in (
        "state_coefficients",
        "gauge_delta",
        "posterior_covariance",
        "robust_weights",
    ):
        np.testing.assert_allclose(
            getattr(converted, name),
            getattr(legacy, name),
            atol=1e-12,
            rtol=1e-12,
        )


def test_structured_and_legacy_solver_modes_are_context_local() -> None:
    from bayesian_phystwin._gauge_aware_contracts import GaugeAwareBeliefResult
    from bayesian_phystwin.sparse_prior_aware_gauge_belief import (
        update_sparse_prior_aware_gauge_belief,
        update_sparse_prior_aware_gauge_belief_structured,
    )
    from bayesian_phystwin.structured_gauge_aware_result import (
        StructuredGaugeAwareBeliefResultV1,
    )

    adapted = _build(_validated_observation())
    structured = update_sparse_prior_aware_gauge_belief_structured(
        adapted.batch,
        adapted.tree_gauge_design,
        config=_config(),
    )
    legacy = update_sparse_prior_aware_gauge_belief(
        adapted.batch,
        adapted.tree_gauge_design,
        config=_config(),
    )
    assert isinstance(structured, StructuredGaugeAwareBeliefResultV1)
    assert isinstance(legacy, GaugeAwareBeliefResult)


def test_claim_bearing_structured_rejection_binds_without_materialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from bayesian_phystwin.structured_gauge_aware_result import (
        PrecisionBackedCovarianceV1,
    )
    from bayesian_phystwin.tree_sparse_structured_gauge_prob4d import (
        update_claim_bearing_tree_sparse_prob4d_structured_from_artifacts,
    )

    def fail_materialization(self, *, maximum_bytes=None):
        raise AssertionError("claim identity materialized covariance")

    monkeypatch.setattr(
        PrecisionBackedCovarianceV1,
        "materialize",
        fail_materialization,
    )
    update = update_claim_bearing_tree_sparse_prob4d_structured_from_artifacts(
        _validated_observation(),
        _linearization(),
        physical_prediction_xyz_m=np.zeros((4, 3), dtype=np.float64),
        state_prior_covariance_m2=np.zeros((2, 2), dtype=np.float64),
        config=_config(),
    )

    assert not update.inference_admissible
    assert update.dense_covariance_materialized is False
    assert len(update.admission_id) == 64
    assert len(update.structured_result_id) == 64
    assert len(update.update_id) == 64
    assert update.descriptor()["structured_result_id"] == update.structured_result_id


def test_claim_bearing_structured_conversion_is_explicit_and_budgeted() -> None:
    from bayesian_phystwin.tree_sparse_structured_gauge_prob4d import (
        update_claim_bearing_tree_sparse_prob4d_structured_from_artifacts,
    )

    update = update_claim_bearing_tree_sparse_prob4d_structured_from_artifacts(
        _validated_observation(),
        _linearization(),
        physical_prediction_xyz_m=np.zeros((4, 3), dtype=np.float64),
        state_prior_covariance_m2=np.zeros((2, 2), dtype=np.float64),
        config=_config(),
    )
    with pytest.raises(MemoryError, match="exceeding"):
        update.to_legacy(maximum_covariance_bytes=1)
    legacy = update.to_legacy()
    assert legacy.result.reason == update.result.reason
    assert legacy.observation_artifact_id == update.observation_artifact_id


def test_legacy_sparse_result_diagnostics_remain_content_compatible() -> None:
    from bayesian_phystwin.sparse_prior_aware_gauge_belief import (
        update_sparse_prior_aware_gauge_belief,
        update_sparse_prior_aware_gauge_belief_structured,
    )

    adapted = _build(_validated_observation())
    legacy = update_sparse_prior_aware_gauge_belief(
        adapted.batch,
        adapted.tree_gauge_design,
        config=_config(),
    )
    structured = update_sparse_prior_aware_gauge_belief_structured(
        adapted.batch,
        adapted.tree_gauge_design,
        config=_config(),
    )

    assert not any(key.startswith("result_") for key in legacy.diagnostics)
    assert structured.diagnostics["result_dense_covariance_materialized"] is True
    assert structured.diagnostics["result_covariance_representation"] == (
        "dense-covariance-v1"
    )


def test_structured_covariance_contract_validation_branches() -> None:
    import bayesian_phystwin.structured_gauge_aware_result as structured_module
    from bayesian_phystwin.structured_gauge_aware_result import (
        DenseCovarianceV1,
        StructuredGaugeAwareBeliefResultV1,
    )

    empty = DenseCovarianceV1(np.zeros((0, 0), dtype=np.float64))
    assert empty.dimension == 0
    assert empty.materialize().shape == (0, 0)
    assert empty.materialize(maximum_bytes=0).shape == (0, 0)
    np.testing.assert_array_equal(
        np.asarray(empty, dtype=np.float32),
        np.zeros((0, 0), dtype=np.float32),
    )
    with pytest.raises(ValueError, match="nonnegative integer"):
        empty.materialize(maximum_bytes=True)
    with pytest.raises(ValueError, match="nonnegative integer"):
        empty.materialize(maximum_bytes=-1)

    neutral = structured_module._symmetric_matrix(
        np.eye(1),
        name="neutral",
        positive_semidefinite=False,
        positive_definite=False,
    )
    np.testing.assert_array_equal(neutral, np.eye(1))

    adapted = _build(_validated_observation())
    from bayesian_phystwin.sparse_prior_aware_gauge_belief import (
        update_sparse_prior_aware_gauge_belief_structured,
    )

    valid = update_sparse_prior_aware_gauge_belief_structured(
        adapted.batch,
        adapted.tree_gauge_design,
        config=_config(),
    )
    with pytest.raises(TypeError, match="inference_admissible"):
        replace(valid, inference_admissible=1)
    with pytest.raises(ValueError, match="reason"):
        replace(valid, reason="")
    with pytest.raises(TypeError, match="unsupported representation"):
        replace(valid, covariance=object())
    with pytest.raises(TypeError, match="GaugeAwareBeliefResult"):
        StructuredGaugeAwareBeliefResultV1.from_legacy(object())


def _rejected_structured_claim_update():
    from bayesian_phystwin.tree_sparse_structured_gauge_prob4d import (
        update_claim_bearing_tree_sparse_prob4d_structured_from_artifacts,
    )

    return update_claim_bearing_tree_sparse_prob4d_structured_from_artifacts(
        _validated_observation(),
        _linearization(),
        physical_prediction_xyz_m=np.zeros((4, 3), dtype=np.float64),
        state_prior_covariance_m2=np.zeros((2, 2), dtype=np.float64),
        config=_config(),
    )


def test_structured_claim_update_rejects_invalid_wrapper_fields() -> None:
    update = _rejected_structured_claim_update()
    with pytest.raises(TypeError, match="StructuredGaugeAwareBeliefResultV1"):
        replace(update, result=object())
    with pytest.raises(TypeError, match="independently_verified"):
        replace(update, runtime_revision_independently_verified=1)
    with pytest.raises(ValueError, match="must be True"):
        replace(update, runtime_revision_independently_verified=False)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("observation", "observation_artifact_id"),
        ("calibration", "calibration_artifact_ids"),
        ("runtime", "independently verified runtime"),
    ],
)
def test_structured_claim_update_rejects_lineage_substitution(
    mutation: str,
    message: str,
) -> None:
    update = _rejected_structured_claim_update()
    lineage = dict(update.result.input_lineage)
    if mutation == "observation":
        lineage["observation_artifact_id"] = "9" * 64
    elif mutation == "calibration":
        calibration = dict(lineage["prob4d_claim_bearing_calibration_artifact_ids"])
        calibration["gauge_artifact_id"] = "8" * 64
        lineage["prob4d_claim_bearing_calibration_artifact_ids"] = calibration
    else:
        lineage["prob4d_claim_bearing_runtime_revision_independently_verified"] = False
    bad_result = replace(update.result, input_lineage=lineage)
    with pytest.raises(ValueError, match=message):
        replace(update, result=bad_result)
