from __future__ import annotations

import inspect
from dataclasses import replace

import numpy as np
import pytest

import bayesian_phystwin.deform360_registered_residual_history_v1 as subject
from bayesian_phystwin.deform360_registered_residual_history_v1 import (
    _common as common,
)
from bayesian_phystwin.deform360_registered_residual_history_v1 import (
    _execution as execution,
)
from bayesian_phystwin._portable_contracts import content_id
from bayesian_phystwin.endpoint_model_average import (
    DEFAULT_MODEL_AVERAGED_ENDPOINT_CONFIG_V1,
    infer_model_averaged_endpoint,
    predict_model_averaged_endpoint,
)

SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SHA_D = "d" * 64
SHA_E = "e" * 64


def _provenance() -> subject.ResidualHistorySourceProvenanceV1:
    return subject.ResidualHistorySourceProvenanceV1(
        source_inventory_id=SHA_A,
        provider_reconstruction_id=SHA_B,
        scoring_reconstruction_id=SHA_C,
        provider_implementation_revision="1" * 40,
        scoring_implementation_revision="2" * 40,
        provider_configuration_id=SHA_D,
        scoring_configuration_id=SHA_E,
        provider_camera_family_ids=("recorder-00", "recorder-02"),
        scoring_camera_family_ids=("recorder-01", "recorder-03"),
        provider_input_artifact_ids=("1" * 64, "3" * 64),
        scoring_input_artifact_ids=("2" * 64, "4" * 64),
        provider_parent_reconstruction_ids=("5" * 64,),
        scoring_parent_reconstruction_ids=("6" * 64,),
        metadata={"role": "opened-source-only"},
    )


def _arrays(*, future_count: int = 6) -> dict[str, np.ndarray]:
    physical_prefix = np.zeros((3, 4, 3), dtype=np.float64)
    observation = np.full_like(physical_prefix, np.nan)
    validity = np.zeros((3, 4), dtype=bool)
    validity[0] = True
    observation[0] = np.asarray(
        [
            [0.010, 0.000, 0.000],
            [0.011, 0.000, 0.000],
            [0.012, 0.000, 0.000],
            [0.013, 0.000, 0.000],
        ],
        dtype=np.float64,
    )
    validity[1, :2] = True
    observation[1, :2] = np.asarray(
        [[0.020, 0.001, 0.000], [0.021, 0.001, 0.000]],
        dtype=np.float64,
    )
    validity[2, 2:] = True
    observation[2, 2:] = np.asarray(
        [[0.030, 0.002, 0.000], [0.031, 0.002, 0.000]],
        dtype=np.float64,
    )
    physical_future = np.zeros((future_count, 4, 3), dtype=np.float64)
    physical_future[..., 2] = (
        np.arange(future_count, dtype=np.float64)[:, None] * 0.001
    )
    reference_covariance = np.zeros(
        physical_future.shape + (3,),
        dtype=np.float64,
    )
    arrays = {
        "physical_prefix": physical_prefix,
        "observation": observation,
        "validity": validity,
        "physical_future": physical_future,
        "reference_covariance": reference_covariance,
    }
    arrays["registered_mean"] = _registered_mean(arrays)
    return arrays


def _registered_mean(arrays: dict[str, np.ndarray]) -> np.ndarray:
    result = np.array(arrays["physical_future"], copy=True, order="C")
    for track in range(arrays["validity"].shape[1]):
        support = np.flatnonzero(arrays["validity"][:, track])
        if len(support):
            frame = int(support[-1])
            result[:, track] += (
                arrays["observation"][frame, track]
                - arrays["physical_prefix"][frame, track]
            )
    return result


def _run(
    arrays: dict[str, np.ndarray],
    *,
    registered_mean: object | None = None,
    provenance: object | None = None,
    metadata: dict[str, object] | None = None,
    source_unit_id: str = "opened-source-object-session-001",
) -> subject.RegisteredResidualHistoryPredictionV1:
    return subject.run_registered_residual_history_v1(
        arrays["physical_prefix"],
        arrays["observation"],
        arrays["validity"],
        arrays["physical_future"],
        (
            arrays["registered_mean"]
            if registered_mean is None
            else registered_mean
        ),
        arrays["reference_covariance"],
        source_unit_id=source_unit_id,
        provenance=_provenance() if provenance is None else provenance,
        metadata=metadata,
    )


def _donor_covariance(arrays: dict[str, np.ndarray]) -> np.ndarray:
    residual = np.zeros_like(arrays["physical_prefix"])
    validity = arrays["validity"]
    residual[validity] = (
        arrays["observation"][validity] - arrays["physical_prefix"][validity]
    )
    posterior = infer_model_averaged_endpoint(
        residual,
        validity,
        end_frame=len(residual),
        config=DEFAULT_MODEL_AVERAGED_ENDPOINT_CONFIG_V1,
    )
    return np.stack(
        [
            predict_model_averaged_endpoint(
                posterior,
                horizon_steps=horizon,
            ).covariance_m2
            for horizon in range(1, len(arrays["physical_future"]) + 1)
        ],
        axis=0,
    )


def test_registered_execution_reproduces_donor_and_preserves_mean() -> None:
    arrays = _arrays()
    registered = arrays["registered_mean"]
    result = _run(arrays, metadata={"opened_source_only": True})
    bins = np.empty(len(registered), dtype=np.int64)
    for label, chunk in enumerate(np.array_split(np.arange(len(bins)), 3)):
        bins[chunk] = label
    expected = _donor_covariance(arrays) * np.asarray(
        subject.REGISTERED_COVARIANCE_SCALES
    )[bins, None, None, None]

    assert result.accepted
    assert result.mean_m is registered
    assert result.hybrid is not None
    assert result.hybrid.mean_m is registered
    assert result.hybrid.covariance_m2 is result.covariance_m2
    np.testing.assert_allclose(result.covariance_m2, expected)
    assert result.decision.future_horizon_bins == tuple(int(x) for x in bins)
    assert result.decision.future_horizon_steps == tuple(range(1, 7))
    assert result.decision.endpoint_config_id == common._frozen_endpoint_config_id()
    assert len(result.decision.endpoint_prediction_ids) == 6
    assert result.decision.metadata["opened_source_only"] is True
    assert not result.covariance_m2.flags.writeable
    descriptor = result.decision.descriptor()
    assert descriptor["reference_predictor_id"] == "last_residual"
    assert descriptor["covariance_donor_id"] == "independent_endpoint_v1"
    assert descriptor["fixed_anchor_contract_version"] == 1


def test_public_execution_has_no_injection_surface() -> None:
    parameters = inspect.signature(
        subject.run_registered_residual_history_v1
    ).parameters
    prohibited = {
        "donor_covariance_m2",
        "covariance_scale",
        "future_horizon_steps",
        "future_horizon_bins",
        "target_roster",
        "target_path",
    }
    assert not prohibited & set(parameters)


def test_execution_is_deterministic_and_changed_source_changes_ids() -> None:
    first = _run(_arrays())
    second = _run(_arrays())
    changed_arrays = _arrays()
    changed_arrays["observation"][0, 0, 0] += 0.001
    changed_arrays["registered_mean"] = _registered_mean(changed_arrays)
    changed = _run(changed_arrays)

    assert first.decision.decision_id == second.decision.decision_id
    assert first.decision.endpoint_posterior_id == second.decision.endpoint_posterior_id
    assert changed.decision.decision_id != first.decision.decision_id
    assert (
        changed.decision.endpoint_posterior_id
        != first.decision.endpoint_posterior_id
    )


def test_support_and_mean_mismatch_return_exact_reference_objects() -> None:
    arrays = _arrays()
    arrays["validity"][2, 3] = False
    arrays["observation"][2, 3] = np.nan
    arrays["registered_mean"] = _registered_mean(arrays)
    support_result = _run(arrays)

    assert not support_result.accepted
    assert support_result.mean_m is arrays["registered_mean"]
    assert support_result.covariance_m2 is arrays["reference_covariance"]
    assert support_result.decision.fallback_reasons == (
        "insufficient-per-track-support",
    )
    assert support_result.decision.endpoint_posterior_id is None

    arrays = _arrays()
    mismatch = np.array(arrays["registered_mean"], copy=True)
    mismatch[0, 0, 0] += 0.001
    mismatch_result = _run(arrays, registered_mean=mismatch)
    assert not mismatch_result.accepted
    assert mismatch_result.mean_m is mismatch
    assert mismatch_result.covariance_m2 is arrays["reference_covariance"]
    assert mismatch_result.decision.fallback_reasons == (
        "registered-mean-mismatch",
    )


def test_support_and_mean_mismatch_can_be_reported_together() -> None:
    arrays = _arrays()
    arrays["validity"][2, 3] = False
    arrays["observation"][2, 3] = np.nan
    mismatch = np.array(arrays["registered_mean"], copy=True)
    mismatch[0, 0, 0] += 0.001
    result = _run(arrays, registered_mean=mismatch)
    assert result.decision.fallback_reasons == (
        "insufficient-per-track-support",
        "registered-mean-mismatch",
    )


def test_inference_failure_returns_exact_fallback_without_donor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def reject(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise ValueError("inference unavailable")

    monkeypatch.setattr(execution, "infer_model_averaged_endpoint", reject)
    arrays = _arrays()
    result = _run(arrays, metadata={"source": "kept"})

    assert not result.accepted
    assert result.mean_m is arrays["registered_mean"]
    assert result.covariance_m2 is arrays["reference_covariance"]
    assert result.decision.fallback_reasons == (
        "covariance-contract-rejection",
    )
    assert result.decision.endpoint_posterior_id is None
    assert result.decision.endpoint_prediction_ids == ()
    assert result.decision.donor_covariance_sha256 is None
    assert result.decision.metadata["source_metadata"]["source"] == "kept"
    assert result.decision.metadata["covariance_rejection_type"] == "ValueError"


def test_composition_failure_returns_fallback_with_complete_donor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def reject(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise RuntimeError("composition unavailable")

    monkeypatch.setattr(execution, "compose_covariance_only_hybrid", reject)
    arrays = _arrays()
    result = _run(arrays)

    assert not result.accepted
    assert result.mean_m is arrays["registered_mean"]
    assert result.covariance_m2 is arrays["reference_covariance"]
    assert result.decision.endpoint_posterior_id is not None
    assert len(result.decision.endpoint_prediction_ids) == 6
    assert result.decision.donor_covariance_sha256 is not None
    assert result.decision.metadata["covariance_rejection_type"] == "RuntimeError"


def test_default_endpoint_configuration_is_explicit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[object] = []
    original = execution.infer_model_averaged_endpoint

    def capture(*args: object, **kwargs: object) -> object:
        seen.append(kwargs.get("config"))
        return original(*args, **kwargs)

    monkeypatch.setattr(execution, "infer_model_averaged_endpoint", capture)
    result = _run(_arrays())
    assert result.accepted
    assert seen == [DEFAULT_MODEL_AVERAGED_ENDPOINT_CONFIG_V1]


def test_missingness_reference_covariance_and_short_future_fail_closed() -> None:
    arrays = _arrays()
    invalid = tuple(np.argwhere(~arrays["validity"])[0])
    arrays["observation"][invalid] = 0.0
    with pytest.raises(ValueError, match="explicit NaN"):
        _run(arrays)

    arrays = _arrays()
    arrays["reference_covariance"][0, 0, 0, 0] = 1e-12
    with pytest.raises(ValueError, match="exact zero covariance"):
        _run(arrays)

    with pytest.raises(ValueError, match="H>=3"):
        _run(_arrays(future_count=2))


def test_identity_bearing_inputs_reject_implicit_copies() -> None:
    arrays = _arrays()
    with pytest.raises(ValueError, match="C-contiguous"):
        _run(
            arrays,
            registered_mean=np.asfortranarray(arrays["registered_mean"]),
        )
    arrays = _arrays()
    arrays["reference_covariance"] = np.asfortranarray(
        arrays["reference_covariance"]
    )
    with pytest.raises(ValueError, match="C-contiguous"):
        _run(arrays)
