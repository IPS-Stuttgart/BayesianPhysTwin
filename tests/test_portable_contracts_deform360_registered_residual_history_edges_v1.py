from __future__ import annotations

import inspect
from dataclasses import replace

import numpy as np
import pytest

import bayesian_phystwin.deform360_registered_residual_history_v1 as subject
from bayesian_phystwin._portable_contracts import content_id
from bayesian_phystwin.deform360_registered_residual_history_v1 import (
    _common as common,
)
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
        (arrays["registered_mean"] if registered_mean is None else registered_mean),
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


def test_result_constructor_rejects_mismatches() -> None:
    result = _run(_arrays())
    with pytest.raises(TypeError, match="provenance"):
        replace(result, provenance="bad")
    with pytest.raises(TypeError, match="decision"):
        replace(result, decision="bad")
    with pytest.raises(ValueError, match="provenance differ"):
        other = replace(
            result.provenance,
            source_inventory_id="9" * 64,
            provenance_id=None,
        )
        replace(result, provenance=other)
    with pytest.raises(ValueError, match="mean shape"):
        replace(result, mean_m=result.mean_m[:-1])
    with pytest.raises(ValueError, match="covariance shape"):
        replace(result, covariance_m2=result.covariance_m2[:-1])
    with pytest.raises(ValueError, match="mean content"):
        changed = np.array(result.mean_m, copy=True)
        changed[0, 0, 0] += 1.0
        replace(result, mean_m=changed)
    with pytest.raises(ValueError, match="covariance content"):
        changed_cov = np.array(result.covariance_m2, copy=True)
        changed_cov[0, 0, 0, 0] += 1.0
        replace(result, covariance_m2=changed_cov)
    with pytest.raises(ValueError, match="missing the covariance hybrid"):
        replace(result, hybrid=None)


def test_fallback_result_rejects_hybrid() -> None:
    arrays = _arrays()
    mismatch = np.array(arrays["registered_mean"], copy=True)
    mismatch[0, 0, 0] += 1.0
    fallback = _run(arrays, registered_mean=mismatch)
    accepted = _run(_arrays())
    with pytest.raises(ValueError, match="must not retain"):
        replace(fallback, hybrid=accepted.hybrid)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("physical_prefix", "bad", "NumPy array"),
        ("physical_prefix", np.zeros((3, 4, 3), dtype=np.float32), "float64"),
        ("physical_prefix", np.zeros((3, 4), dtype=np.float64), "dimensions"),
        ("validity", np.zeros((3, 4), dtype=np.int64), "Boolean"),
        ("validity", np.zeros((3, 4, 1), dtype=bool), "two dimensions"),
        ("physical_future", np.zeros((6, 4), dtype=np.float64), "dimensions"),
        ("registered_mean", np.zeros((6, 4, 3), dtype=np.float32), "float64"),
        ("reference_covariance", np.zeros((6, 4, 3), dtype=np.float64), "dimensions"),
    ],
)
def test_execution_array_types_fail_closed(
    field: str,
    value: object,
    match: str,
) -> None:
    arrays = _arrays()
    arrays[field] = value  # type: ignore[assignment]
    with pytest.raises((TypeError, ValueError), match=match):
        _run(arrays)


def test_execution_array_shapes_and_values_fail_closed() -> None:
    arrays = _arrays()
    arrays["observation"] = arrays["observation"][:, :3].copy()
    with pytest.raises(ValueError, match="matching shape"):
        _run(arrays)

    arrays = _arrays()
    arrays["validity"] = arrays["validity"][:, :3].copy()
    with pytest.raises(ValueError, match="matching shape"):
        _run(arrays)

    arrays = _arrays()
    arrays["physical_future"] = arrays["physical_future"][:, :3].copy()
    arrays["registered_mean"] = arrays["registered_mean"][:, :3].copy()
    arrays["reference_covariance"] = arrays["reference_covariance"][:, :3].copy()
    with pytest.raises(ValueError, match="track rosters"):
        _run(arrays)

    arrays = _arrays()
    arrays["registered_mean"] = arrays["registered_mean"][:-1].copy()
    with pytest.raises(ValueError, match="shape changed"):
        _run(arrays)

    arrays = _arrays()
    arrays["reference_covariance"] = arrays["reference_covariance"][:-1].copy()
    with pytest.raises(ValueError, match="shape changed"):
        _run(arrays)

    arrays = _arrays()
    arrays["physical_prefix"][0, 0, 0] = np.inf
    with pytest.raises(ValueError, match="finite"):
        _run(arrays)

    arrays = _arrays()
    arrays["observation"][0, 0, 0] = np.inf
    with pytest.raises(ValueError, match="valid provider"):
        _run(arrays)


def test_noncontiguous_nonfinite_and_bad_source_inputs_fail_closed() -> None:
    arrays = _arrays()
    arrays["physical_prefix"] = np.asfortranarray(arrays["physical_prefix"])
    with pytest.raises(ValueError, match="C-contiguous"):
        _run(arrays)

    arrays = _arrays()
    arrays["validity"] = np.asfortranarray(arrays["validity"])
    with pytest.raises(ValueError, match="C-contiguous"):
        _run(arrays)

    arrays = _arrays()
    arrays["physical_future"][0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        _run(arrays)

    arrays = _arrays()
    with pytest.raises(ValueError, match="canonical string"):
        _run(arrays, source_unit_id=" bad")
    with pytest.raises(TypeError, match="provenance"):
        _run(arrays, provenance="bad")
    with pytest.raises(ValueError, match="metadata"):
        _run(arrays, metadata={"bad": np.inf})


def test_private_descriptor_type_guards() -> None:
    with pytest.raises(TypeError, match="config"):
        common._endpoint_config_descriptor("bad")
    with pytest.raises(TypeError, match="posterior"):
        common._endpoint_posterior_descriptor(
            "bad",
            residual_history_sha256=SHA_A,
            validity_sha256=SHA_B,
            config_id=SHA_C,
        )
    with pytest.raises(TypeError, match="prediction"):
        common._endpoint_prediction_descriptor("bad", posterior_id=SHA_A)


def test_metadata_and_claim_boundary_are_immutable() -> None:
    result = _run(_arrays(), metadata={"protocol": "source-only-v1"})
    with pytest.raises(TypeError, match="immutable"):
        result.decision.metadata["protocol"] = "changed"  # type: ignore[index]
    assert result.decision.descriptor()["claim_boundary"] == subject.CLAIM_BOUNDARY
    assert (
        "target"
        not in inspect.signature(subject.run_registered_residual_history_v1).parameters
    )


def test_config_descriptor_is_content_addressed() -> None:
    descriptor = common._endpoint_config_descriptor(
        DEFAULT_MODEL_AVERAGED_ENDPOINT_CONFIG_V1
    )
    assert descriptor["endpoint_contract_version"] == 1
    assert descriptor["fixed_anchor_contract_version"] == 1
    assert common._frozen_endpoint_config_id() == content_id(descriptor)
