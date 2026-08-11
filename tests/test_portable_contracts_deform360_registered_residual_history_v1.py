from __future__ import annotations

import inspect
from dataclasses import replace

import numpy as np
import pytest

from bayesian_phystwin.deform360_registered_residual_history_v1 import (
    REGISTERED_COVARIANCE_DONOR_ID,
    REGISTERED_COVARIANCE_SCALES,
    REGISTERED_REFERENCE_PREDICTOR_ID,
    RegisteredResidualHistoryPredictionV1,
    ResidualHistorySourceProvenanceV1,
    run_registered_residual_history_v1,
)
from bayesian_phystwin.endpoint_model_average import (
    DEFAULT_MODEL_AVERAGED_ENDPOINT_CONFIG_V1,
    infer_model_averaged_endpoint,
    predict_model_averaged_endpoint,
)


def _provenance() -> ResidualHistorySourceProvenanceV1:
    return ResidualHistorySourceProvenanceV1(
        source_inventory_id="a" * 64,
        provider_reconstruction_id="b" * 64,
        scoring_reconstruction_id="c" * 64,
        provider_implementation_revision="1" * 40,
        scoring_implementation_revision="2" * 40,
        provider_configuration_id="d" * 64,
        scoring_configuration_id="e" * 64,
        provider_camera_family_ids=("recorder-family-00", "recorder-family-02"),
        scoring_camera_family_ids=("recorder-family-01", "recorder-family-03"),
        provider_input_artifact_ids=("1" * 64, "3" * 64),
        scoring_input_artifact_ids=("2" * 64, "4" * 64),
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
    physical_future[..., 2] = np.arange(future_count, dtype=np.float64)[:, None] * 0.001
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
    physical_prefix = arrays["physical_prefix"]
    observation = arrays["observation"]
    validity = arrays["validity"]
    result = np.array(arrays["physical_future"], copy=True, order="C")
    for track in range(validity.shape[1]):
        support = np.flatnonzero(validity[:, track])
        if len(support):
            frame = int(support[-1])
            result[:, track] += (
                observation[frame, track] - physical_prefix[frame, track]
            )
    return result


def _run(
    arrays: dict[str, np.ndarray],
    *,
    registered_mean: np.ndarray | None = None,
    metadata: dict[str, object] | None = None,
) -> RegisteredResidualHistoryPredictionV1:
    return run_registered_residual_history_v1(
        arrays["physical_prefix"],
        arrays["observation"],
        arrays["validity"],
        arrays["physical_future"],
        arrays["registered_mean"] if registered_mean is None else registered_mean,
        arrays["reference_covariance"],
        source_unit_id="opened-source-object-session-001",
        provenance=_provenance(),
        metadata=metadata,
    )


def _unscaled_endpoint_covariance(arrays: dict[str, np.ndarray]) -> np.ndarray:
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


def test_public_execution_has_no_donor_or_schedule_injection_surface() -> None:
    parameters = inspect.signature(run_registered_residual_history_v1).parameters
    prohibited = {
        "donor_covariance_m2",
        "covariance_scale",
        "future_horizon_steps",
        "future_horizon_bins",
        "target_roster",
        "target_path",
    }
    assert not prohibited & set(parameters)
    assert "registered_last_residual_mean_m" in parameters
    assert "reference_covariance_m2" in parameters


def test_registered_execution_reproduces_frozen_donor_and_preserves_mean() -> None:
    arrays = _arrays()
    registered_mean = arrays["registered_mean"]
    result = _run(
        arrays,
        registered_mean=registered_mean,
        metadata={"opened_source_only": True},
    )
    unscaled = _unscaled_endpoint_covariance(arrays)
    bins = np.empty(len(registered_mean), dtype=np.int64)
    for label, chunk in enumerate(np.array_split(np.arange(len(bins)), 3)):
        bins[chunk] = label
    expected = (
        unscaled * np.asarray(REGISTERED_COVARIANCE_SCALES)[bins, None, None, None]
    )

    assert result.accepted
    assert result.mean_m is registered_mean
    assert result.hybrid is not None
    assert result.hybrid.mean_m is registered_mean
    assert result.hybrid.covariance_m2 is result.covariance_m2
    np.testing.assert_allclose(result.covariance_m2, expected, rtol=0.0, atol=1e-15)
    assert result.decision.future_horizon_bins == tuple(int(value) for value in bins)
    assert len(result.decision.endpoint_prediction_ids) == len(registered_mean)
    assert result.decision.descriptor()["reference_predictor_id"] == (
        REGISTERED_REFERENCE_PREDICTOR_ID
    )
    assert result.decision.descriptor()["covariance_donor_id"] == (
        REGISTERED_COVARIANCE_DONOR_ID
    )
    assert result.decision.metadata["opened_source_only"] is True
    assert not result.covariance_m2.flags.writeable


def test_execution_is_deterministic_and_source_changes_change_identity() -> None:
    first = _run(_arrays())
    second = _run(_arrays())
    changed_arrays = _arrays()
    changed_arrays["observation"][0, 0, 0] += 0.001
    changed_arrays["registered_mean"] = _registered_mean(changed_arrays)
    changed = _run(changed_arrays)

    assert first.provenance.provenance_id == second.provenance.provenance_id
    assert first.decision.endpoint_config_id == second.decision.endpoint_config_id
    assert first.decision.endpoint_posterior_id == second.decision.endpoint_posterior_id
    assert (
        first.decision.endpoint_prediction_ids
        == second.decision.endpoint_prediction_ids
    )
    assert first.decision.decision_id == second.decision.decision_id
    assert (
        changed.decision.endpoint_posterior_id != first.decision.endpoint_posterior_id
    )
    assert changed.decision.decision_id != first.decision.decision_id


def test_insufficient_support_returns_exact_registered_reference_objects() -> None:
    arrays = _arrays()
    arrays["validity"][2, 3] = False
    arrays["observation"][2, 3] = np.nan
    arrays["registered_mean"] = _registered_mean(arrays)
    registered_mean = arrays["registered_mean"]
    reference_covariance = arrays["reference_covariance"]

    result = _run(arrays, registered_mean=registered_mean)

    assert not result.accepted
    assert result.mean_m is registered_mean
    assert result.covariance_m2 is reference_covariance
    assert result.hybrid is None
    assert result.decision.fallback_reasons == ("insufficient-per-track-support",)
    assert result.decision.endpoint_config_id is None
    assert result.decision.endpoint_prediction_ids == ()


def test_registered_mean_mismatch_returns_exact_reference_objects() -> None:
    arrays = _arrays()
    registered_mean = np.array(arrays["registered_mean"], copy=True, order="C")
    registered_mean[0, 0, 0] += 0.001

    result = _run(arrays, registered_mean=registered_mean)

    assert not result.accepted
    assert result.mean_m is registered_mean
    assert result.covariance_m2 is arrays["reference_covariance"]
    assert result.decision.fallback_reasons == ("registered-mean-mismatch",)


def test_missingness_and_reference_covariance_fail_closed() -> None:
    arrays = _arrays()
    invalid = np.argwhere(~arrays["validity"])[0]
    arrays["observation"][tuple(invalid)] = 0.0
    with pytest.raises(ValueError, match="explicit NaN"):
        _run(arrays)

    arrays = _arrays()
    arrays["reference_covariance"][0, 0, 0, 0] = 1e-12
    with pytest.raises(ValueError, match="exact zero covariance"):
        _run(arrays)


def test_identity_bearing_inputs_reject_implicit_copies() -> None:
    arrays = _arrays()
    with pytest.raises(ValueError, match="C-contiguous"):
        _run(
            arrays,
            registered_mean=np.asfortranarray(arrays["registered_mean"]),
        )

    arrays = _arrays()
    arrays["reference_covariance"] = np.asfortranarray(arrays["reference_covariance"])
    with pytest.raises(ValueError, match="C-contiguous"):
        _run(arrays)


def test_provenance_rejects_shared_families_and_input_bytes() -> None:
    provenance = _provenance()
    with pytest.raises(ValueError, match="camera families must be disjoint"):
        replace(
            provenance,
            scoring_camera_family_ids=(
                "recorder-family-00",
                "recorder-family-03",
            ),
            provenance_id=None,
        )
    with pytest.raises(ValueError, match="input artifacts must be disjoint"):
        replace(
            provenance,
            scoring_input_artifact_ids=("1" * 64, "4" * 64),
            provenance_id=None,
        )


def test_content_id_and_derived_decision_tampering_is_rejected() -> None:
    result = _run(_arrays())
    with pytest.raises(ValueError, match="provenance_id"):
        replace(result.provenance, provenance_id="0" * 64)
    with pytest.raises(ValueError, match="decision_id"):
        replace(result.decision, decision_id="0" * 64)
    with pytest.raises(ValueError, match="canonical partition"):
        replace(
            result.decision,
            future_horizon_bins=(0, 0, 0, 1, 2, 2),
            decision_id=None,
        )


def test_frozen_metadata_and_claim_boundary_are_retained() -> None:
    result = _run(_arrays(), metadata={"protocol": "source-only-v1"})
    with pytest.raises(TypeError, match="immutable"):
        result.decision.metadata["protocol"] = "changed"
    descriptor = result.decision.descriptor()
    assert descriptor["claim_boundary"]
    assert (
        "target" not in inspect.signature(run_registered_residual_history_v1).parameters
    )
