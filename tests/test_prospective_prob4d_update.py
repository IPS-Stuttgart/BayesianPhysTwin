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


def _update(
    *,
    result: GaugeAwareBeliefResult | None = None,
    calibration_artifact_ids: object = CALIBRATION_IDS,
    runtime_revision_source: object = "independent-vcs-check",
    runtime_revision_independently_verified: object = True,
) -> ClaimBearingProb4DUpdateV1:
    return ClaimBearingProb4DUpdateV1(
        result=_result(_lineage()) if result is None else result,
        observation_artifact_id=OBSERVATION_ID,
        linearization_artifact_id=LINEARIZATION_ID,
        provider_manifest_id=PROVIDER_ID,
        calibration_artifact_ids=calibration_artifact_ids,  # type: ignore[arg-type]
        runtime_revision_source=runtime_revision_source,  # type: ignore[arg-type]
        runtime_revision_independently_verified=(
            runtime_revision_independently_verified  # type: ignore[arg-type]
        ),
    )


def test_claim_update_identity_is_backed_by_irreversibly_immutable_result() -> None:
    update = _update()
    identities = (
        update.admission_id,
        update.inference_result_id,
        update.update_id,
    )

    for name in (
        "state_coefficients",
        "gauge_delta",
        "shared_bias_coefficients",
        "view_bias_coefficients",
        "anchor_bias_coefficients",
        "posterior_covariance",
        "identifiable_state_transform",
        "identifiable_fractions",
        "query_sensitivity_fractions",
        "robust_weights",
        "anchor_robust_weights",
    ):
        with pytest.raises(ValueError):
            getattr(update.result, name).setflags(write=True)

    assert (
        update.admission_id,
        update.inference_result_id,
        update.update_id,
    ) == identities


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


def test_nonstring_runtime_source_fails_before_solver(monkeypatch) -> None:
    events: list[str] = []
    lineage = {
        **_lineage(),
        "prob4d_claim_bearing_runtime_revision_source": 7,
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

    with pytest.raises(TypeError, match="runtime_revision_source must be a string"):
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
        _update(result=result)


def test_update_contract_rejects_calibration_lineage_mismatch() -> None:
    with pytest.raises(ValueError, match="calibration_artifact_ids"):
        _update(
            calibration_artifact_ids={
                "gauge": "d" * 64,
                "point": "f" * 64,
            }
        )


def test_update_contract_rejects_runtime_source_lineage_mismatch() -> None:
    with pytest.raises(ValueError, match="runtime_revision_source"):
        _update(runtime_revision_source="different-independent-check")


def test_update_contract_requires_true_verification_flag() -> None:
    with pytest.raises(ValueError, match="must be True"):
        _update(runtime_revision_independently_verified=False)


def test_update_contract_freezes_calibration_mapping() -> None:
    supplied = dict(CALIBRATION_IDS)
    update = _update(calibration_artifact_ids=supplied)
    update_id = update.update_id

    supplied["gauge"] = "f" * 64

    assert dict(update.calibration_artifact_ids) == CALIBRATION_IDS
    assert update.update_id == update_id
    with pytest.raises(TypeError):
        update.calibration_artifact_ids["gauge"] = "f" * 64  # type: ignore[index]


def test_update_contract_exposes_immutable_result_lineage() -> None:
    supplied = dict(CALIBRATION_IDS)
    lineage = {
        **_lineage(),
        "prob4d_claim_bearing_calibration_artifact_ids": supplied,
    }
    update = _update(result=_result(lineage))
    update_id = update.update_id

    supplied["gauge"] = "f" * 64

    assert (
        update.result.input_lineage["prob4d_claim_bearing_calibration_artifact_ids"]
        == CALIBRATION_IDS
    )
    assert update.update_id == update_id
    with pytest.raises(TypeError, match="immutable"):
        update.result.input_lineage["prob4d_claim_bearing_calibration_artifact_ids"][
            "gauge"
        ] = "f" * 64


def test_update_contract_rejects_invalid_calibration_digest() -> None:
    with pytest.raises(ValueError, match="calibration artifact"):
        _update(calibration_artifact_ids={"gauge": "invalid"})


def test_update_contract_rejects_missing_and_unnamed_calibration_ids() -> None:
    for calibration_ids, message in (({}, "missing"), ({"": "d" * 64}, "nonempty")):
        with pytest.raises(ValueError, match=message):
            _update(calibration_artifact_ids=calibration_ids)


@pytest.mark.parametrize(
    ("calibration_ids", "message"),
    [
        ({1: "d" * 64}, "names must be strings"),
        ({"gauge": 1}, "digest must be a string"),
    ],
)
def test_update_contract_rejects_nonstring_calibration_entries(
    calibration_ids: object,
    message: str,
) -> None:
    with pytest.raises(TypeError, match=message):
        _update(calibration_artifact_ids=calibration_ids)


def test_update_contract_rejects_wrong_result_and_runtime_types() -> None:
    with pytest.raises(TypeError, match="GaugeAwareBeliefResult"):
        _update(result=object())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="runtime_revision_source"):
        _update(runtime_revision_source="")
    with pytest.raises(TypeError, match="runtime_revision_source must be a string"):
        _update(runtime_revision_source=7)
    with pytest.raises(TypeError, match="must be a bool"):
        _update(runtime_revision_independently_verified=1)


def test_update_contract_rejects_unverified_result_lineage() -> None:
    lineage = _lineage(runtime_verified=False)
    with pytest.raises(ValueError, match="lacks independently verified"):
        _update(result=_result(lineage))


def test_update_contract_exposes_versioned_result_identities() -> None:
    update = _update()

    assert len(update.admission_id) == 64
    assert update.legacy_update_id == update.admission_id
    assert len(update.inference_result_id) == 64
    assert update.update_id != update.admission_id


def test_update_contract_rejects_untyped_result_decision_fields() -> None:
    untyped_decision = _result(_lineage())
    object.__setattr__(
        untyped_decision,
        "inference_admissible",
        np.bool_(True),
    )
    with pytest.raises(TypeError, match="inference_admissible"):
        _update(result=untyped_decision)

    empty_reason = _result(_lineage())
    object.__setattr__(empty_reason, "reason", "")
    with pytest.raises(ValueError, match="reason"):
        _update(result=empty_reason)
