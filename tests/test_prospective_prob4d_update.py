from types import SimpleNamespace

import numpy as np
import pytest

import bayesian_phystwin.prospective_prob4d_update as update_module
from bayesian_phystwin import GaugeAwareBeliefResult
from bayesian_phystwin.prospective_prob4d_update import (
    ClaimBearingProb4DUpdateV1,
    update_claim_bearing_prob4d_from_artifacts,
)

OBSERVATION_ID = "a" * 64
LINEARIZATION_ID = "b" * 64
PROVIDER_ID = "c" * 64
CALIBRATION_IDS = {"gauge": "d" * 64, "point": "e" * 64}


def _lineage(*, runtime_verified: bool = True) -> dict[str, object]:
    return {
        "observation_artifact_id": OBSERVATION_ID,
        "linearization_artifact_id": LINEARIZATION_ID,
        "prob4d_claim_bearing_provider_manifest_id": PROVIDER_ID,
        "prob4d_claim_bearing_calibration_artifact_ids": CALIBRATION_IDS,
        "prob4d_claim_bearing_runtime_revision_source": "independent-vcs-check",
        "prob4d_claim_bearing_runtime_revision_independently_verified": (
            runtime_verified
        ),
    }


def _result(lineage: dict[str, object]) -> GaugeAwareBeliefResult:
    return GaugeAwareBeliefResult(
        inference_admissible=True,
        reason="accepted",
        state_coefficients=np.array([0.1]),
        gauge_delta=np.zeros(0),
        shared_bias_coefficients=np.zeros(0),
        view_bias_coefficients=np.zeros(0),
        anchor_bias_coefficients=np.zeros(0),
        posterior_covariance=np.array([[0.2]]),
        identifiable_state_transform=np.array([[1.0]]),
        identifiable_fractions=np.array([1.0]),
        query_sensitivity_fractions=np.array([1.0]),
        robust_weights=np.array([1.0]),
        anchor_robust_weights=np.zeros(0),
        diagnostics={"solver": "test"},
        input_lineage=lineage,
    )


def test_one_call_path_admits_before_solving_and_binds_lineage(monkeypatch) -> None:
    events: list[str] = []
    lineage = _lineage()
    adapted = SimpleNamespace(
        batch=SimpleNamespace(metadata=lineage),
        observation_artifact_id=OBSERVATION_ID,
    )
    linearization = SimpleNamespace(artifact_id=LINEARIZATION_ID)
    expected_result = _result(lineage)
    config = object()

    def adapt(*args, **kwargs):
        events.append("adapt")
        assert kwargs["physical_prediction_xyz_m"].shape == (1, 3)
        return adapted

    def solve(batch, *, config=None):
        events.append("solve")
        assert batch is adapted.batch
        assert config is config_value
        return expected_result

    config_value = config
    monkeypatch.setattr(
        update_module,
        "build_claim_bearing_gauge_aware_batch_from_artifacts",
        adapt,
    )
    monkeypatch.setattr(
        update_module,
        "update_prior_aware_gauge_belief",
        solve,
    )

    update = update_claim_bearing_prob4d_from_artifacts(
        object(),
        linearization,
        physical_prediction_xyz_m=np.zeros((1, 3)),
        config=config_value,
    )

    assert events == ["adapt", "solve"]
    assert update.result is expected_result
    assert update.inference_admissible
    assert update.observation_artifact_id == OBSERVATION_ID
    assert update.linearization_artifact_id == LINEARIZATION_ID
    assert update.provider_manifest_id == PROVIDER_ID
    assert dict(update.calibration_artifact_ids) == CALIBRATION_IDS
    assert len(update.update_id) == 64


def test_unverified_runtime_fails_before_solver(monkeypatch) -> None:
    events: list[str] = []
    lineage = _lineage(runtime_verified=False)
    adapted = SimpleNamespace(
        batch=SimpleNamespace(metadata=lineage),
        observation_artifact_id=OBSERVATION_ID,
    )

    def adapt(*args, **kwargs):
        events.append("adapt")
        return adapted

    def solve(*args, **kwargs):
        events.append("solve")
        raise AssertionError("solver must not run")

    monkeypatch.setattr(
        update_module,
        "build_claim_bearing_gauge_aware_batch_from_artifacts",
        adapt,
    )
    monkeypatch.setattr(
        update_module,
        "update_prior_aware_gauge_belief",
        solve,
    )

    with pytest.raises(ValueError, match="not independently verified"):
        update_claim_bearing_prob4d_from_artifacts(
            object(),
            SimpleNamespace(artifact_id=LINEARIZATION_ID),
            physical_prediction_xyz_m=np.zeros((1, 3)),
        )
    assert events == ["adapt"]


def test_update_contract_rejects_lineage_mismatch() -> None:
    lineage = _lineage()
    result = _result({**lineage, "observation_artifact_id": "f" * 64})
    with pytest.raises(ValueError, match="observation_artifact_id"):
        ClaimBearingProb4DUpdateV1(
            result=result,
            observation_artifact_id=OBSERVATION_ID,
            linearization_artifact_id=LINEARIZATION_ID,
            provider_manifest_id=PROVIDER_ID,
            calibration_artifact_ids=CALIBRATION_IDS,
            runtime_revision_source="independent-vcs-check",
            runtime_revision_independently_verified=True,
        )


def test_update_contract_rejects_invalid_calibration_digest() -> None:
    lineage = _lineage()
    with pytest.raises(ValueError, match="calibration artifact"):
        ClaimBearingProb4DUpdateV1(
            result=_result(lineage),
            observation_artifact_id=OBSERVATION_ID,
            linearization_artifact_id=LINEARIZATION_ID,
            provider_manifest_id=PROVIDER_ID,
            calibration_artifact_ids={"gauge": "invalid"},
            runtime_revision_source="independent-vcs-check",
            runtime_revision_independently_verified=True,
        )
