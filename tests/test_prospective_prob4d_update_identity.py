import hashlib
import json

import numpy as np
import pytest

from bayesian_phystwin import GaugeAwareBeliefResult
from bayesian_phystwin.prospective_prob4d_update import (
    CLAIM_BEARING_PROB4D_INFERENCE_RESULT_VERSION,
    CLAIM_BEARING_PROB4D_UPDATE_IDENTITY_VERSION,
    ClaimBearingProb4DUpdateV1,
)

OBSERVATION_ID = "a" * 64
LINEARIZATION_ID = "b" * 64
PROVIDER_ID = "c" * 64
CALIBRATION_IDS = {"gauge": "d" * 64, "point": "e" * 64}
RUNTIME_SOURCE = "independent-vcs-check"


def _lineage(*, extra: dict[str, object] | None = None) -> dict[str, object]:
    return {
        "observation_artifact_id": OBSERVATION_ID,
        "linearization_artifact_id": LINEARIZATION_ID,
        "prob4d_claim_bearing_provider_manifest_id": PROVIDER_ID,
        "prob4d_claim_bearing_calibration_artifact_ids": CALIBRATION_IDS,
        "prob4d_claim_bearing_runtime_revision_source": RUNTIME_SOURCE,
        "prob4d_claim_bearing_runtime_revision_independently_verified": True,
        **(extra or {}),
    }


def _result(
    *,
    state: float = 0.1,
    covariance: float = 0.2,
    robust_weight: float = 1.0,
    diagnostics: dict[str, object] | None = None,
    lineage: dict[str, object] | None = None,
    reason: str = "accepted",
    inference_admissible: bool = True,
) -> GaugeAwareBeliefResult:
    return GaugeAwareBeliefResult(
        inference_admissible=inference_admissible,
        reason=reason,
        state_coefficients=np.array([state]),
        gauge_delta=np.zeros(0),
        shared_bias_coefficients=np.zeros(0),
        view_bias_coefficients=np.zeros(0),
        anchor_bias_coefficients=np.zeros(0),
        posterior_covariance=np.array([[covariance]]),
        identifiable_state_transform=np.array([[1.0]]),
        identifiable_fractions=np.array([1.0]),
        query_sensitivity_fractions=np.array([1.0]),
        robust_weights=np.array([robust_weight]),
        anchor_robust_weights=np.zeros(0),
        diagnostics={"solver": "test"} if diagnostics is None else diagnostics,
        input_lineage=_lineage() if lineage is None else lineage,
    )


def _update(result: GaugeAwareBeliefResult) -> ClaimBearingProb4DUpdateV1:
    return ClaimBearingProb4DUpdateV1(
        result=result,
        observation_artifact_id=OBSERVATION_ID,
        linearization_artifact_id=LINEARIZATION_ID,
        provider_manifest_id=PROVIDER_ID,
        calibration_artifact_ids=CALIBRATION_IDS,
        runtime_revision_source=RUNTIME_SOURCE,
        runtime_revision_independently_verified=True,
    )


def _historical_update_id(result: GaugeAwareBeliefResult) -> str:
    payload = {
        "schema": "bayesian_phystwin.claim_bearing_prob4d_update",
        "schema_version": 1,
        "observation_artifact_id": OBSERVATION_ID,
        "linearization_artifact_id": LINEARIZATION_ID,
        "provider_manifest_id": PROVIDER_ID,
        "calibration_artifact_ids": CALIBRATION_IDS,
        "runtime_revision_source": RUNTIME_SOURCE,
        "runtime_revision_independently_verified": True,
        "inference_admissible": result.inference_admissible,
        "reason": result.reason,
    }
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def test_strengthened_identity_preserves_historical_admission_id() -> None:
    result = _result()
    update = _update(result)

    assert CLAIM_BEARING_PROB4D_UPDATE_IDENTITY_VERSION == 2
    assert CLAIM_BEARING_PROB4D_INFERENCE_RESULT_VERSION == 1
    assert update.admission_id == _historical_update_id(result)
    assert update.legacy_update_id == update.admission_id
    assert update.update_id != update.admission_id
    assert len(update.inference_result_id) == len(update.update_id) == 64


@pytest.mark.parametrize(
    "changed",
    [
        _result(state=0.2),
        _result(covariance=0.3),
        _result(robust_weight=0.8),
        _result(diagnostics={"solver": "different"}),
        _result(lineage=_lineage(extra={"factor_stream_id": "f" * 64})),
    ],
)
def test_update_id_binds_complete_numerical_result(
    changed: GaugeAwareBeliefResult,
) -> None:
    reference = _update(_result())
    candidate = _update(changed)

    assert candidate.admission_id == reference.admission_id
    assert candidate.inference_result_id != reference.inference_result_id
    assert candidate.update_id != reference.update_id


def test_result_identity_is_canonical_for_metadata_order() -> None:
    first = _update(
        _result(
            diagnostics={"outer": {"b": 2, "a": 1}, "items": [3, 4]},
            lineage=_lineage(extra={"z": 2, "y": 1}),
        )
    )
    second = _update(
        _result(
            diagnostics={"items": [3, 4], "outer": {"a": 1, "b": 2}},
            lineage=_lineage(extra={"y": 1, "z": 2}),
        )
    )

    assert first.inference_result_id == second.inference_result_id
    assert first.update_id == second.update_id


def test_result_identity_is_immutable_after_input_mutation() -> None:
    state = np.array([0.1])
    covariance = np.array([[0.2]])
    diagnostics = {"nested": {"value": 1}}
    lineage = _lineage()
    result = GaugeAwareBeliefResult(
        inference_admissible=True,
        reason="accepted",
        state_coefficients=state,
        gauge_delta=np.zeros(0),
        shared_bias_coefficients=np.zeros(0),
        view_bias_coefficients=np.zeros(0),
        anchor_bias_coefficients=np.zeros(0),
        posterior_covariance=covariance,
        identifiable_state_transform=np.array([[1.0]]),
        identifiable_fractions=np.array([1.0]),
        query_sensitivity_fractions=np.array([1.0]),
        robust_weights=np.array([1.0]),
        anchor_robust_weights=np.zeros(0),
        diagnostics=diagnostics,
        input_lineage=lineage,
    )
    update = _update(result)
    identity = update.update_id

    state[0] = 9.0
    covariance[0, 0] = 9.0
    diagnostics["nested"]["value"] = 9  # type: ignore[index]
    lineage["extra"] = "changed"

    assert update.update_id == identity
    assert update.result.state_coefficients.tolist() == [0.1]
    assert update.result.posterior_covariance.tolist() == [[0.2]]


def test_claim_bearing_identity_rejects_untyped_decision_fields() -> None:
    result = _result()
    object.__setattr__(result, "inference_admissible", np.bool_(True))
    with pytest.raises(TypeError, match="inference_admissible"):
        _update(result)

    empty_reason = _result()
    object.__setattr__(empty_reason, "reason", "")
    with pytest.raises(ValueError, match="reason"):
        _update(empty_reason)
