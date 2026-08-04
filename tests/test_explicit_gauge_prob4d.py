from __future__ import annotations

import subprocess
import sys
from dataclasses import replace
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest

from bayesian_phystwin import explicit_gauge_prob4d as _explicit
from bayesian_phystwin._gauge_aware_contracts import (
    COMPOSITE_WEIGHT_MODE_PROVIDER_FINAL,
)
from bayesian_phystwin.explicit_gauge_prob4d import (
    build_claim_bearing_explicit_gauge_batch,
    update_claim_bearing_explicit_gauge_from_artifacts,
)
from bayesian_phystwin.physical_linearization import PhysicalLinearizationV1

ARTIFACT_ID = "a" * 64
PROVIDER_MANIFEST_ID = "b" * 64
GAUGE_CALIBRATION_ID = "c" * 64
POINT_CALIBRATION_ID = "d" * 64
SOURCE_REVISION = "e" * 40


def _fixture() -> tuple[
    SimpleNamespace,
    SimpleNamespace,
    PhysicalLinearizationV1,
    np.ndarray,
]:
    gauge_ids = ("window-0", "window-1")
    frame_indices = np.asarray([0, 0, 1, 1], dtype=np.int64)
    point_ids = np.asarray([0, 1, 0, 1], dtype=np.int64)
    gauge_indices = np.asarray([0, 0, 1, 1], dtype=np.int64)
    view_ids = ("camera-b", "camera-a", "camera-b", "camera-a")
    world_mean = np.asarray(
        [
            [0.0, 0.0, 1.000],
            [0.1, 0.0, 1.000],
            [0.0, 0.1, 1.005],
            [0.1, 0.1, 1.005],
        ],
        dtype=np.float64,
    )
    conditional = np.repeat(
        (np.eye(3, dtype=np.float64) * 1e-4)[None],
        len(world_mean),
        axis=0,
    )
    marginal = conditional + np.eye(3, dtype=np.float64)[None] * 1e-5
    local_gauge = np.zeros((len(world_mean), 3, 7), dtype=np.float64)
    local_gauge[:, 0, 4] = 1.0
    local_gauge[:, 1, 5] = 1.0
    association = np.asarray([1.0, 0.8, 0.9, 0.7])
    reliability = np.asarray([0.95, 0.90, 0.85, 0.80])
    nominal = np.asarray([0.98, 0.98, 0.90, 0.90])
    composite = np.asarray([1.0, 0.5, 1.0, 0.5])
    calibration_ids = {
        "gauge_artifact_id": GAUGE_CALIBRATION_ID,
        "point_artifact_id": POINT_CALIBRATION_ID,
    }
    runtime = {
        "source": "source_checkout",
        "independently_verified": True,
    }
    attestation = {
        "claim_bearing": True,
        "export_mode": "calibrated",
        "provider_revision": SOURCE_REVISION,
        "provider_manifest_id": PROVIDER_MANIFEST_ID,
        "calibration_artifact_ids": calibration_ids,
        "runtime_revision": runtime,
    }
    envelope = SimpleNamespace(
        artifact_id=ARTIFACT_ID,
        bundle_schema_version=4,
        sequence_id="sequence-a",
        case_id="case-a",
        stream_id="prob4d:explicit-gauge-factors",
        source_repository="FlorianPfaff/Prob4D",
        source_revision=SOURCE_REVISION,
        causal_frame_stop=2,
        factor_count=2,
        observation_count=len(world_mean),
        gauge_ids=gauge_ids,
        gauge_covariance_semantics="joint-cross-window",
        cross_window_gauge_covariance_preserved=True,
        provider_manifest_id=PROVIDER_MANIFEST_ID,
        calibration_artifact_ids=calibration_ids,
        runtime_revision_source="source_checkout",
        runtime_revision_independently_verified=True,
        provider_attestation=attestation,
    )
    bundle = SimpleNamespace(
        sequence_id=envelope.sequence_id,
        case_id=envelope.case_id,
        stream_id=envelope.stream_id,
        source_repository=envelope.source_repository,
        source_revision=envelope.source_revision,
        causal_frame_stop=envelope.causal_frame_stop,
        factors=(object(), object()),
        gauges=tuple(SimpleNamespace(window_id=gauge_id) for gauge_id in gauge_ids),
    )
    validated = SimpleNamespace(
        bundle=bundle,
        envelope=envelope,
        artifact_id=ARTIFACT_ID,
    )
    stack = SimpleNamespace(
        world_mean_m=world_mean,
        conditional_world_covariance_m2=conditional,
        marginal_world_covariance_m2=marginal,
        local_gauge_jacobian=local_gauge,
        gauge_indices=gauge_indices,
        gauge_prior_covariance=np.eye(14, dtype=np.float64) * 1e-6,
        association_probability=association,
        prior_reliability=reliability,
        prior_nominal_probability=nominal,
        composite_weight=composite,
        point_ids=point_ids,
        frame_indices=frame_indices,
        view_ids=view_ids,
        factor_ids=("factor-0", "factor-0", "factor-1", "factor-1"),
        correlation_group_ids=("g0", "g1", "g2", "g3"),
        gauge_ids=gauge_ids,
        causal_frame_stop=2,
    )
    state_jacobian = np.zeros((len(world_mean), 3, 1), dtype=np.float64)
    state_jacobian[:, 2, 0] = 1.0
    query_jacobian = np.zeros((1, 3, 1), dtype=np.float64)
    query_jacobian[0, 2, 0] = 1.0
    linearization = PhysicalLinearizationV1(
        observation_artifact_id=ARTIFACT_ID,
        baseline_belief_id="f" * 64,
        action_prefix_id="1" * 64,
        simulator_revision="simulator-revision-a",
        frame_ids=frame_indices,
        entity_ids=point_ids,
        view_indices=np.asarray([1, 0, 1, 0], dtype=np.int64),
        window_indices=gauge_indices,
        state_jacobian=state_jacobian,
        query_state_jacobian=query_jacobian,
        physical_response_m=np.asarray([[0.0, 0.0, 0.02]]),
        metadata={"case_id": "case-a"},
    )
    physical_prediction = world_mean.copy()
    physical_prediction[:, 2] -= 0.005
    return validated, stack, linearization, physical_prediction


def _changed_array(
    value: np.ndarray,
    index: tuple[int, ...],
    replacement: float,
) -> np.ndarray:
    changed = np.asarray(value).copy()
    changed[index] = replacement
    return changed


def test_explicit_gauge_bridge_preserves_factor_semantics() -> None:
    validated, stack, linearization, physical_prediction = _fixture()

    adapted = build_claim_bearing_explicit_gauge_batch(
        validated,
        stack,
        linearization,
        physical_prediction_xyz_m=physical_prediction,
    )

    batch = adapted.batch
    np.testing.assert_array_equal(
        batch.observation_covariance_m2,
        stack.conditional_world_covariance_m2,
    )
    assert not np.array_equal(
        batch.observation_covariance_m2,
        stack.marginal_world_covariance_m2,
    )
    np.testing.assert_array_equal(
        batch.gauge_prior_covariance,
        stack.gauge_prior_covariance,
    )
    np.testing.assert_array_equal(
        batch.prior_reliability,
        stack.prior_reliability,
    )
    np.testing.assert_array_equal(
        batch.prior_nominal_probability,
        stack.prior_nominal_probability,
    )
    np.testing.assert_allclose(
        batch.composite_weight,
        stack.association_probability * stack.composite_weight,
    )
    assert batch.composite_weight_mode == COMPOSITE_WEIGHT_MODE_PROVIDER_FINAL

    assert batch.gauge_jacobian.shape == (4, 3, 14)
    np.testing.assert_array_equal(
        batch.gauge_jacobian[:2, :, :7],
        stack.local_gauge_jacobian[:2],
    )
    np.testing.assert_array_equal(
        batch.gauge_jacobian[:2, :, 7:],
        np.zeros((2, 3, 7)),
    )
    np.testing.assert_array_equal(
        batch.gauge_jacobian[2:, :, :7],
        np.zeros((2, 3, 7)),
    )
    np.testing.assert_array_equal(
        batch.gauge_jacobian[2:, :, 7:],
        stack.local_gauge_jacobian[2:],
    )
    assert batch.metadata["prob4d_marginal_point_covariance_consumed"] is False
    assert batch.metadata["row_alignment_verified"] is True
    assert adapted.observation_artifact_id == ARTIFACT_ID


def test_explicit_gauge_bridge_accepts_optional_nuisance_inputs() -> None:
    validated, stack, linearization, physical_prediction = _fixture()
    shared = np.zeros((4, 3, 1), dtype=np.float64)
    view = np.zeros((4, 3, 1), dtype=np.float64)

    adapted = build_claim_bearing_explicit_gauge_batch(
        validated,
        stack,
        linearization,
        physical_prediction_xyz_m=physical_prediction,
        shared_bias_jacobian=shared,
        view_bias_jacobian=view,
        state_prior_covariance_m2=np.eye(1, dtype=np.float64),
        metadata={"registered_arm": "explicit-gauge"},
    )

    np.testing.assert_array_equal(adapted.batch.shared_bias_jacobian, shared)
    np.testing.assert_array_equal(adapted.batch.view_bias_jacobian, view)
    assert adapted.batch.metadata["registered_arm"] == "explicit-gauge"


def test_explicit_gauge_bridge_fails_before_excessive_expansion() -> None:
    validated, stack, linearization, physical_prediction = _fixture()
    required_bytes = 4 * 3 * 14 * np.dtype(np.float64).itemsize

    with pytest.raises(MemoryError, match="exceeding the declared"):
        build_claim_bearing_explicit_gauge_batch(
            validated,
            stack,
            linearization,
            physical_prediction_xyz_m=physical_prediction,
            maximum_dense_gauge_design_bytes=required_bytes - 1,
        )


def test_explicit_gauge_bridge_rejects_row_identity_drift() -> None:
    validated, stack, linearization, physical_prediction = _fixture()
    changed = replace(
        linearization,
        window_indices=np.asarray([0, 1, 1, 1], dtype=np.int64),
    )

    with pytest.raises(ValueError, match="window_indices differ"):
        build_claim_bearing_explicit_gauge_batch(
            validated,
            stack,
            changed,
            physical_prediction_xyz_m=physical_prediction,
        )


@pytest.mark.parametrize(
    ("attestation_field", "value", "match"),
    (
        ("claim_bearing", False, "not claim-bearing"),
        ("export_mode", "exploratory", "not calibrated"),
        ("provider_revision", "9" * 40, "revision differs"),
        ("provider_manifest_id", "9" * 64, "manifest differs"),
    ),
)
def test_explicit_gauge_bridge_rejects_attestation_identity_drift(
    attestation_field: str,
    value: object,
    match: str,
) -> None:
    validated, stack, linearization, physical_prediction = _fixture()
    validated.envelope.provider_attestation = {
        **validated.envelope.provider_attestation,
        attestation_field: value,
    }

    with pytest.raises(ValueError, match=match):
        build_claim_bearing_explicit_gauge_batch(
            validated,
            stack,
            linearization,
            physical_prediction_xyz_m=physical_prediction,
        )


@pytest.mark.parametrize(
    ("runtime_field", "value", "match"),
    (
        ("source", "installed_vcs_metadata", "runtime source differs"),
        ("independently_verified", False, "not independently verified"),
    ),
)
def test_explicit_gauge_bridge_rejects_attested_runtime_drift(
    runtime_field: str,
    value: object,
    match: str,
) -> None:
    validated, stack, linearization, physical_prediction = _fixture()
    attestation = dict(validated.envelope.provider_attestation)
    attestation["runtime_revision"] = {
        **attestation["runtime_revision"],
        runtime_field: value,
    }
    validated.envelope.provider_attestation = attestation

    with pytest.raises(ValueError, match=match):
        build_claim_bearing_explicit_gauge_batch(
            validated,
            stack,
            linearization,
            physical_prediction_xyz_m=physical_prediction,
        )


def test_explicit_gauge_bridge_rejects_contradictory_attestation() -> None:
    validated, stack, linearization, physical_prediction = _fixture()
    validated.envelope.provider_attestation = {
        **validated.envelope.provider_attestation,
        "calibration_artifact_ids": {
            "gauge_artifact_id": "9" * 64,
            "point_artifact_id": POINT_CALIBRATION_ID,
        },
    }

    with pytest.raises(ValueError, match="calibration IDs differ"):
        build_claim_bearing_explicit_gauge_batch(
            validated,
            stack,
            linearization,
            physical_prediction_xyz_m=physical_prediction,
        )


@pytest.mark.parametrize(
    ("mutator", "match"),
    (
        (
            lambda validated: setattr(validated, "artifact_id", "9" * 64),
            "artifact ID differs",
        ),
        (
            lambda validated: setattr(
                validated.envelope,
                "bundle_schema_version",
                3,
            ),
            "schema version 4",
        ),
        (
            lambda validated: setattr(
                validated.envelope,
                "source_repository",
                "IPS-Stuttgart/Prob4D",
            ),
            "frozen Prob4D producer identity",
        ),
        (
            lambda validated: setattr(
                validated.envelope,
                "gauge_ids",
                ("window-0", "window-0"),
            ),
            "gauge_ids must be unique",
        ),
        (
            lambda validated: setattr(
                validated.envelope,
                "gauge_covariance_semantics",
                "marginal-blocks-only",
            ),
            "joint cross-window covariance",
        ),
        (
            lambda validated: setattr(
                validated.envelope,
                "cross_window_gauge_covariance_preserved",
                False,
            ),
            "lost cross-window covariance",
        ),
        (
            lambda validated: setattr(
                validated.envelope,
                "runtime_revision_independently_verified",
                False,
            ),
            "literally True",
        ),
        (
            lambda validated: setattr(
                validated.bundle,
                "sequence_id",
                "changed-sequence",
            ),
            "field sequence_id",
        ),
        (
            lambda validated: setattr(
                validated.bundle,
                "causal_frame_stop",
                3,
            ),
            "causal_frame_stop",
        ),
        (
            lambda validated: setattr(
                validated.bundle,
                "factors",
                (object(),),
            ),
            "factor_count",
        ),
        (
            lambda validated: setattr(
                validated.bundle,
                "gauges",
                (SimpleNamespace(window_id="window-0"),),
            ),
            "gauges differ",
        ),
    ),
)
def test_explicit_gauge_bridge_rejects_envelope_bundle_drift(
    mutator: object,
    match: str,
) -> None:
    validated, stack, linearization, physical_prediction = _fixture()
    cast(object, mutator)(validated)  # type: ignore[operator]

    with pytest.raises(ValueError, match=match):
        build_claim_bearing_explicit_gauge_batch(
            validated,
            stack,
            linearization,
            physical_prediction_xyz_m=physical_prediction,
        )


@pytest.mark.parametrize(
    ("field", "factory", "match"),
    (
        (
            "world_mean_m",
            lambda stack: np.zeros((4, 2)),
            "world_mean_m",
        ),
        (
            "conditional_world_covariance_m2",
            lambda stack: np.zeros((4, 2, 2)),
            "conditional_world_covariance",
        ),
        (
            "marginal_world_covariance_m2",
            lambda stack: np.zeros((4, 2, 2)),
            "marginal_world_covariance",
        ),
        (
            "local_gauge_jacobian",
            lambda stack: np.zeros((4, 3, 6)),
            "local_gauge_jacobian",
        ),
        (
            "gauge_indices",
            lambda stack: np.zeros((4, 1), dtype=np.int64),
            "gauge_indices",
        ),
        (
            "gauge_ids",
            lambda stack: ("window-0",),
            "gauges differ",
        ),
        (
            "gauge_indices",
            lambda stack: np.asarray([0, 0, 2, 1], dtype=np.int64),
            "unknown gauge",
        ),
        (
            "gauge_prior_covariance",
            lambda stack: np.eye(7),
            "prior has changed shape",
        ),
        (
            "gauge_prior_covariance",
            lambda stack: _changed_array(
                stack.gauge_prior_covariance,
                (0, 0),
                np.nan,
            ),
            "prior must be finite",
        ),
        (
            "gauge_prior_covariance",
            lambda stack: _changed_array(
                stack.gauge_prior_covariance,
                (0, 1),
                0.1,
            ),
            "prior must be symmetric",
        ),
        (
            "gauge_prior_covariance",
            lambda stack: _changed_array(
                stack.gauge_prior_covariance,
                (0, 0),
                -1.0,
            ),
            "positive semidefinite",
        ),
        (
            "association_probability",
            lambda stack: np.ones((4, 1)),
            "association_probability must have shape",
        ),
        (
            "prior_nominal_probability",
            lambda stack: np.asarray([-0.1, 0.9, 0.9, 0.9]),
            "prior_nominal_probability must lie",
        ),
        (
            "point_ids",
            lambda stack: np.asarray([-1, 1, 0, 1], dtype=np.int64),
            "point_ids must be nonnegative",
        ),
        (
            "frame_indices",
            lambda stack: np.asarray([0, 0, 2, 1], dtype=np.int64),
            "causal frame stop",
        ),
        (
            "view_ids",
            lambda stack: ("camera-a",),
            "string identities",
        ),
        (
            "causal_frame_stop",
            lambda stack: 3,
            "stack differs from envelope",
        ),
        (
            "world_mean_m",
            lambda stack: _changed_array(
                stack.world_mean_m,
                (0, 0),
                np.nan,
            ),
            "non-finite values",
        ),
        (
            "conditional_world_covariance_m2",
            lambda stack: _changed_array(
                stack.conditional_world_covariance_m2,
                (0, 0, 1),
                0.1,
            ),
            "conditional point covariances must be symmetric",
        ),
        (
            "conditional_world_covariance_m2",
            lambda stack: np.concatenate(
                (
                    np.zeros((1, 3, 3)),
                    stack.conditional_world_covariance_m2[1:],
                )
            ),
            "positive definite",
        ),
        (
            "marginal_world_covariance_m2",
            lambda stack: _changed_array(
                stack.marginal_world_covariance_m2,
                (0, 0, 1),
                0.1,
            ),
            "marginal point covariances must be symmetric",
        ),
        (
            "marginal_world_covariance_m2",
            lambda stack: _changed_array(
                stack.marginal_world_covariance_m2,
                (0, 0, 0),
                -1.0,
            ),
            "positive semidefinite",
        ),
    ),
)
def test_explicit_gauge_bridge_rejects_sparse_stack_drift(
    field: str,
    factory: object,
    match: str,
) -> None:
    validated, stack, linearization, physical_prediction = _fixture()
    value = cast(object, factory)(stack)  # type: ignore[operator]
    setattr(stack, field, value)

    with pytest.raises(ValueError, match=match):
        build_claim_bearing_explicit_gauge_batch(
            validated,
            stack,
            linearization,
            physical_prediction_xyz_m=physical_prediction,
        )


def test_explicit_gauge_bridge_keeps_association_out_of_reliability() -> None:
    validated, stack, linearization, physical_prediction = _fixture()
    original_reliability = stack.prior_reliability.copy()

    adapted = build_claim_bearing_explicit_gauge_batch(
        validated,
        stack,
        linearization,
        physical_prediction_xyz_m=physical_prediction,
    )

    np.testing.assert_array_equal(
        adapted.batch.prior_reliability,
        original_reliability,
    )
    assert adapted.batch.metadata[
        "prob4d_association_probability_semantics"
    ].startswith("generalized-Bayes-row-power")


def test_explicit_gauge_bridge_rejects_zero_association_rows() -> None:
    validated, stack, linearization, physical_prediction = _fixture()
    stack.association_probability = stack.association_probability.copy()
    stack.association_probability[0] = 0.0

    with pytest.raises(ValueError, match="association_probability"):
        build_claim_bearing_explicit_gauge_batch(
            validated,
            stack,
            linearization,
            physical_prediction_xyz_m=physical_prediction,
        )


@pytest.mark.parametrize(
    ("replacement", "match"),
    (
        (object(), "PhysicalLinearizationV1"),
        (
            replace(
                _fixture()[2],
                observation_artifact_id="9" * 64,
            ),
            "does not identify",
        ),
        (
            replace(
                _fixture()[2],
                frame_ids=np.asarray([0, 1, 1, 1], dtype=np.int64),
            ),
            "frame_ids differ",
        ),
        (
            replace(
                _fixture()[2],
                entity_ids=np.asarray([0, 2, 0, 1], dtype=np.int64),
            ),
            "entity_ids differ",
        ),
        (
            replace(
                _fixture()[2],
                view_indices=np.asarray([0, 0, 1, 0], dtype=np.int64),
            ),
            "view_indices differ",
        ),
    ),
)
def test_explicit_gauge_bridge_rejects_linearization_drift(
    replacement: object,
    match: str,
) -> None:
    validated, stack, _linearization, physical_prediction = _fixture()

    with pytest.raises((TypeError, ValueError), match=match):
        build_claim_bearing_explicit_gauge_batch(
            validated,
            stack,
            replacement,  # type: ignore[arg-type]
            physical_prediction_xyz_m=physical_prediction,
        )


@pytest.mark.parametrize(
    ("prediction", "match"),
    (
        (np.zeros((4, 2)), "shape"),
        (
            _changed_array(_fixture()[3], (0, 0), np.nan),
            "must be finite",
        ),
    ),
)
def test_explicit_gauge_bridge_rejects_invalid_physical_prediction(
    prediction: np.ndarray,
    match: str,
) -> None:
    validated, stack, linearization, _physical_prediction = _fixture()

    with pytest.raises(ValueError, match=match):
        build_claim_bearing_explicit_gauge_batch(
            validated,
            stack,
            linearization,
            physical_prediction_xyz_m=prediction,
        )


def test_explicit_gauge_helper_contracts_fail_closed() -> None:
    calls = (
        (lambda: _explicit._require_string(1, name="value"), TypeError),
        (lambda: _explicit._require_string("", name="value"), ValueError),
        (lambda: _explicit._require_sha256("A" * 64, name="digest"), ValueError),
        (lambda: _explicit._require_revision("g" * 40, name="revision"), ValueError),
        (lambda: _explicit._require_integer(True, name="value"), TypeError),
        (
            lambda: _explicit._require_integer(0, name="value", minimum=1),
            ValueError,
        ),
        (lambda: _explicit._require_mapping([], name="mapping"), TypeError),
        (lambda: _explicit._require_mapping({1: "x"}, name="mapping"), TypeError),
        (
            lambda: _explicit._calibration_ids(
                {"gauge_artifact_id": GAUGE_CALIBRATION_ID}
            ),
            ValueError,
        ),
        (lambda: _explicit._string_tuple([], name="items"), TypeError),
        (lambda: _explicit._string_tuple([""], name="items"), ValueError),
    )
    for call, error in calls:
        with pytest.raises(error):
            call()


def test_explicit_gauge_adapter_result_rejects_non_batch() -> None:
    validated, stack, linearization, physical_prediction = _fixture()
    adapted = build_claim_bearing_explicit_gauge_batch(
        validated,
        stack,
        linearization,
        physical_prediction_xyz_m=physical_prediction,
    )

    with pytest.raises(TypeError, match="GaugeAwareObservationBatch"):
        _explicit.ExplicitGaugeFactorAdapterResult(
            batch=object(),  # type: ignore[arg-type]
            observation_artifact_id=adapted.observation_artifact_id,
            linearization_artifact_id=adapted.linearization_artifact_id,
            provider_manifest_id=adapted.provider_manifest_id,
            calibration_artifact_ids=adapted.calibration_artifact_ids,
            runtime_revision_source=adapted.runtime_revision_source,
            dense_gauge_design_bytes=adapted.dense_gauge_design_bytes,
            dense_gauge_design_limit_bytes=(adapted.dense_gauge_design_limit_bytes),
            gauge_ids=adapted.gauge_ids,
            view_ids=adapted.view_ids,
        )


def test_explicit_gauge_one_call_update_binds_all_lineage() -> None:
    validated, stack, linearization, physical_prediction = _fixture()

    update = update_claim_bearing_explicit_gauge_from_artifacts(
        validated,
        stack,
        linearization,
        physical_prediction_xyz_m=physical_prediction,
    )

    lineage = update.result.input_lineage
    assert update.observation_artifact_id == ARTIFACT_ID
    assert update.linearization_artifact_id == linearization.artifact_id
    assert update.provider_manifest_id == PROVIDER_MANIFEST_ID
    assert lineage["observation_artifact_id"] == ARTIFACT_ID
    assert lineage["linearization_artifact_id"] == linearization.artifact_id
    assert lineage["prob4d_claim_bearing_provider_manifest_id"] == PROVIDER_MANIFEST_ID
    assert (
        lineage["prob4d_claim_bearing_runtime_revision_independently_verified"] is True
    )
    assert len(update.update_id) == 64
    retained = cast(dict[str, str], update.calibration_artifact_ids)
    with pytest.raises(TypeError):
        retained["gauge_artifact_id"] = "8" * 64


def test_explicit_gauge_metadata_cannot_override_provenance() -> None:
    validated, stack, linearization, physical_prediction = _fixture()

    with pytest.raises(ValueError, match="reserved explicit-gauge fields"):
        build_claim_bearing_explicit_gauge_batch(
            validated,
            stack,
            linearization,
            physical_prediction_xyz_m=physical_prediction,
            metadata={"observation_artifact_id": "2" * 64},
        )


def test_explicit_gauge_module_does_not_import_prob4d() -> None:
    code = """
import sys
import bayesian_phystwin.explicit_gauge_prob4d
if "prob4d" in sys.modules:
    raise SystemExit("consumer imported the producer implementation")
"""
    subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )
