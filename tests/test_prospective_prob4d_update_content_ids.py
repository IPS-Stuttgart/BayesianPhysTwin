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


def _lineage() -> dict[str, object]:
    return {
        "observation_artifact_id": OBSERVATION_ID,
        "linearization_artifact_id": LINEARIZATION_ID,
        "prob4d_claim_bearing_provider_manifest_id": PROVIDER_ID,
        "prob4d_claim_bearing_calibration_artifact_ids": CALIBRATION_IDS,
        "prob4d_claim_bearing_runtime_revision_source": "independent-vcs-check",
        "prob4d_claim_bearing_runtime_revision_independently_verified": True,
    }


def _result() -> GaugeAwareBeliefResult:
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
        input_lineage=_lineage(),
    )


@pytest.mark.parametrize(
    "field",
    [
        "observation_artifact_id",
        "linearization_artifact_id",
        "provider_manifest_id",
    ],
)
def test_update_contract_rejects_nonstring_content_ids(field: str) -> None:
    arguments: dict[str, object] = {
        "result": _result(),
        "observation_artifact_id": OBSERVATION_ID,
        "linearization_artifact_id": LINEARIZATION_ID,
        "provider_manifest_id": PROVIDER_ID,
        "calibration_artifact_ids": CALIBRATION_IDS,
        "runtime_revision_source": "independent-vcs-check",
        "runtime_revision_independently_verified": True,
    }
    arguments[field] = 7

    with pytest.raises(TypeError, match=rf"{field} must be a string"):
        ClaimBearingProb4DUpdateV1(**arguments)  # type: ignore[arg-type]


def test_nonstring_provider_manifest_id_fails_before_solver(monkeypatch) -> None:
    events: list[str] = []
    lineage = {
        **_lineage(),
        "prob4d_claim_bearing_provider_manifest_id": 7,
    }
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

    with pytest.raises(TypeError, match="provider_manifest_id must be a string"):
        update_claim_bearing_prob4d_from_artifacts(
            object(),
            SimpleNamespace(artifact_id=LINEARIZATION_ID),
            physical_prediction_xyz_m=np.zeros((1, 3)),
        )

    assert events == ["adapt"]
