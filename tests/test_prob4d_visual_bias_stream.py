from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from bayesian_phystwin.prob4d_visual_bias_stream import (
    PROB4D_VISUAL_BIAS_MODEL_SCHEMA,
    PROB4D_VISUAL_BIAS_STREAM_UPDATE_SCHEMA,
    Prob4DVisualBiasStreamConsumptionBindingV1,
    Prob4DVisualBiasStreamUpdateBindingV1,
    ValidatedProb4DVisualBiasStreamV1,
    prob4d_visual_bias_nuisance_family_id,
)
from bayesian_phystwin.prob4d_visual_bias_update import (
    PROB4D_VISUAL_BIAS_ORTHOGONALIZATION,
    _array_descriptor,
    _canonical_id,
)


def _sha(character: str) -> str:
    return character * 64


def _model_id(covariance: np.ndarray) -> str:
    return _canonical_id(
        {
            "schema": PROB4D_VISUAL_BIAS_MODEL_SCHEMA,
            "bias_ids": ["session"],
            "basis_names": ["tx", "ty", "tz"],
            "joint_bias_covariance": _array_descriptor(covariance),
            "orthogonalization_semantics": (
                PROB4D_VISUAL_BIAS_ORTHOGONALIZATION
            ),
            "gauge_projection_tolerance": 1e-6,
            "model_metadata": {"calibration": "source-only"},
        }
    )


def _validated_stream(*, update_count: int = 2) -> ValidatedProb4DVisualBiasStreamV1:
    covariance = np.diag(np.array([0.04, 0.01, 0.01], dtype=np.float64))
    model_id = _model_id(covariance)
    updates: list[Prob4DVisualBiasStreamUpdateBindingV1] = []
    previous: str | None = None
    for index in range(update_count):
        row_start = 2 * index
        update = Prob4DVisualBiasStreamUpdateBindingV1(
            bias_model_id=model_id,
            observation_stream_update_id=_sha(str(index + 1)),
            visual_bias_artifact_id=_sha(chr(ord("a") + index)),
            observation_artifact_id=_sha(chr(ord("c") + index)),
            observation_identity_sha256=_sha(chr(ord("e") + index)),
            frame_start=2 * index,
            frame_stop_exclusive=2 * index + 2,
            row_start=row_start,
            row_stop_exclusive=row_start + 2,
            maximum_gauge_projection=0.0,
            previous_update_id=previous,
        )
        updates.append(update)
        previous = update.update_id
    row_update_indices = np.repeat(
        np.arange(update_count, dtype=np.int64),
        2,
    )
    row_bias_indices = np.zeros(2 * update_count, dtype=np.int64)
    bias_jacobian = np.repeat(
        np.eye(3, dtype=np.float64)[None],
        2 * update_count,
        axis=0,
    )
    return ValidatedProb4DVisualBiasStreamV1(
        stream_key="case-a/session-a",
        bias_ids=("session",),
        basis_names=("tx", "ty", "tz"),
        orthogonalization_semantics=PROB4D_VISUAL_BIAS_ORTHOGONALIZATION,
        gauge_projection_tolerance=1e-6,
        updates=tuple(updates),
        row_update_indices=row_update_indices,
        row_bias_indices=row_bias_indices,
        bias_jacobian=bias_jacobian,
        joint_bias_covariance=covariance,
        model_metadata={"calibration": "source-only"},
        metadata={"protocol": "contract-test"},
        bias_model_id=model_id,
    )


def _consumption(
    stream: ValidatedProb4DVisualBiasStreamV1,
) -> Prob4DVisualBiasStreamConsumptionBindingV1:
    return Prob4DVisualBiasStreamConsumptionBindingV1(
        visual_bias_stream=stream,
        factor_stream_artifact_id=_sha("f"),
        factor_stream_update_ids=tuple(
            update.observation_stream_update_id for update in stream.updates
        ),
        observation_binding_ids=tuple(
            _sha(chr(ord("8") + index)) for index in range(len(stream.updates))
        ),
        recursive_nuisance_policy_id=_sha("9"),
        nuisance_family_id=stream.nuisance_family_id,
    )


def test_update_identity_is_recomputed_and_tamper_evident() -> None:
    stream = _validated_stream()
    update = stream.updates[0]

    assert update.identity_record()["schema"] == (
        PROB4D_VISUAL_BIAS_STREAM_UPDATE_SCHEMA
    )
    assert update.update_id == _canonical_id(update.identity_record())
    with pytest.raises(ValueError, match="update ID mismatch"):
        replace(update, update_id=_sha("0"))


def test_stream_arrays_are_immutable_and_model_scoped() -> None:
    stream = _validated_stream()

    assert stream.artifact_id is not None
    assert stream.bias_model_id is not None
    assert stream.nuisance_family_id == prob4d_visual_bias_nuisance_family_id(
        stream.bias_model_id
    )
    assert stream.global_design().shape == (4, 3, 3)
    assert not stream.row_update_indices.flags.writeable
    assert not stream.row_bias_indices.flags.writeable
    assert not stream.bias_jacobian.flags.writeable
    assert not stream.joint_bias_covariance.flags.writeable


def test_multi_update_execution_fails_closed_instead_of_reusing_prior() -> None:
    binding = _consumption(_validated_stream())

    assert not binding.claim_bearing_execution_admissible
    assert binding.execution_reason == "persistent_visual_bias_state_solver_required"
    with pytest.raises(ValueError, match="duplicate the prior"):
        binding.require_claim_bearing_execution()

    single = _consumption(_validated_stream(update_count=1))
    assert single.claim_bearing_execution_admissible
    single.require_claim_bearing_execution()


def test_stream_rejects_replayed_observation_artifacts() -> None:
    stream = _validated_stream()
    first, second = stream.updates
    replayed = Prob4DVisualBiasStreamUpdateBindingV1(
        bias_model_id=second.bias_model_id,
        observation_stream_update_id=second.observation_stream_update_id,
        visual_bias_artifact_id=second.visual_bias_artifact_id,
        observation_artifact_id=first.observation_artifact_id,
        observation_identity_sha256=second.observation_identity_sha256,
        frame_start=second.frame_start,
        frame_stop_exclusive=second.frame_stop_exclusive,
        row_start=second.row_start,
        row_stop_exclusive=second.row_stop_exclusive,
        maximum_gauge_projection=second.maximum_gauge_projection,
        previous_update_id=first.update_id,
    )

    with pytest.raises(ValueError, match="observation_artifact_id values"):
        replace(
            stream,
            updates=(first, replayed),
            artifact_id=None,
        )


def test_stream_requires_exact_row_to_update_assignment() -> None:
    stream = _validated_stream()
    tampered = np.array([0, 1, 1, 1], dtype=np.int64)

    with pytest.raises(ValueError, match="row_update_indices"):
        replace(
            stream,
            row_update_indices=tampered,
            artifact_id=None,
        )
