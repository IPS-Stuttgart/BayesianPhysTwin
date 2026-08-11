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



@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("provider_reconstruction_id", SHA_C, "must differ"),
        (
            "scoring_camera_family_ids",
            ("recorder-00", "recorder-03"),
            "families must be disjoint",
        ),
        (
            "scoring_input_artifact_ids",
            ("1" * 64, "4" * 64),
            "input artifacts must be disjoint",
        ),
        (
            "provider_parent_reconstruction_ids",
            (SHA_B,),
            "own parent",
        ),
        (
            "scoring_parent_reconstruction_ids",
            ("5" * 64,),
            "lineages overlap",
        ),
        ("provider_implementation_revision", "x" * 40, "lowercase"),
        (
            "provider_camera_family_ids",
            ("recorder-02", "recorder-00"),
            "sorted and unique",
        ),
        (
            "provider_input_artifact_ids",
            ("3" * 64, "1" * 64),
            "sorted and unique",
        ),
    ],
)
def test_provenance_rejects_invalid_separation(
    field: str,
    value: object,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        replace(_provenance(), **{field: value}, provenance_id=None)


def test_provenance_tamper_metadata_and_descriptor() -> None:
    provenance = _provenance()
    with pytest.raises(ValueError, match="provenance_id"):
        replace(provenance, provenance_id="0" * 64)
    with pytest.raises(TypeError, match="immutable"):
        provenance.metadata["role"] = "changed"  # type: ignore[index]
    descriptor = provenance.descriptor()
    assert descriptor["provider_parent_reconstruction_ids"] == ["5" * 64]
    assert descriptor["scoring_parent_reconstruction_ids"] == ["6" * 64]


@pytest.mark.parametrize(
    ("value", "name", "minimum", "match"),
    [
        (True, "value", 0, "integer"),
        (1.5, "value", 0, "integer"),
        (-1, "value", 0, "at least"),
    ],
)
def test_integer_validator_rejects_bad_values(
    value: object,
    name: str,
    minimum: int,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        common._integer(value, name=name, minimum=minimum)


@pytest.mark.parametrize("value", ["", " padded", "bad\nline"])
def test_canonical_string_rejects_bad_values(value: str) -> None:
    with pytest.raises(ValueError):
        common._canonical_string(value, name="value")


def test_tuple_validators_cover_empty_duplicate_and_non_tuple() -> None:
    for value in ([], (), ("b", "a"), ("a", "a")):
        with pytest.raises(ValueError):
            common._canonical_string_tuple(value, name="strings")
    with pytest.raises(ValueError):
        common._digest_tuple([], name="digests")
    with pytest.raises(ValueError):
        common._digest_tuple((), name="digests")
    with pytest.raises(ValueError):
        common._digest_tuple(("b" * 64, "a" * 64), name="digests")
    assert common._digest_tuple((), name="parents", allow_empty=True) == ()


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("endpoint_config_id", "0" * 64, "configuration identity"),
        ("endpoint_contract_version", 2, "endpoint contract"),
        ("fixed_anchor_contract_version", 2, "fixed-anchor"),
        ("future_horizon_bins", (0, 0, 0, 1, 2, 2), "canonical partition"),
        ("future_horizon_steps", (1, 2, 4, 5, 6, 7), "consecutive"),
        ("accepted", 1, "Booleans"),
        ("registered_mean_identity_preserved", False, "always be preserved"),
    ],
)
def test_decision_rejects_frozen_contract_tampering(
    field: str,
    value: object,
    match: str,
) -> None:
    decision = _run(_arrays()).decision
    with pytest.raises(ValueError, match=match):
        replace(decision, **{field: value}, decision_id=None)


def test_decision_rejects_id_reason_and_lineage_tampering() -> None:
    decision = _run(_arrays()).decision
    with pytest.raises(ValueError, match="decision_id"):
        replace(decision, decision_id="0" * 64)
    with pytest.raises(ValueError, match="unsupported reason"):
        replace(
            decision,
            accepted=False,
            fallback_reasons=("unknown",),
            endpoint_posterior_id=None,
            endpoint_prediction_ids=(),
            donor_covariance_sha256=None,
            hybrid_artifact_id=None,
            reference_covariance_identity_preserved=True,
            output_covariance_sha256=decision.reference_covariance_sha256,
            decision_id=None,
        )
    with pytest.raises(ValueError, match="complete or absent"):
        replace(decision, endpoint_posterior_id=None, decision_id=None)
    with pytest.raises(ValueError, match="unique"):
        replace(
            decision,
            endpoint_prediction_ids=(decision.endpoint_prediction_ids[0],) * 6,
            decision_id=None,
        )
    with pytest.raises(ValueError, match="accepted decision"):
        replace(
            decision,
            fallback_reasons=("covariance-contract-rejection",),
            decision_id=None,
        )


def test_fallback_decision_rejects_inconsistent_claims() -> None:
    arrays = _arrays()
    mismatch = np.array(arrays["registered_mean"], copy=True)
    mismatch[0, 0, 0] += 1.0
    decision = _run(arrays, registered_mean=mismatch).decision
    with pytest.raises(ValueError, match="fallback must preserve"):
        replace(
            decision,
            reference_covariance_identity_preserved=False,
            decision_id=None,
        )
    with pytest.raises(ValueError, match="fallback covariance"):
        replace(
            decision,
            output_covariance_sha256="f" * 64,
            decision_id=None,
        )
    with pytest.raises(ValueError, match="admission fallback"):
        replace(
            decision,
            endpoint_posterior_id="1" * 64,
            endpoint_prediction_ids=tuple(f"{x:064x}" for x in range(2, 8)),
            donor_covariance_sha256="8" * 64,
            decision_id=None,
        )


def test_covariance_rejection_rejects_partial_lineage() -> None:
    arrays = _arrays()
    original = execution.compose_covariance_only_hybrid
    try:
        execution.compose_covariance_only_hybrid = lambda *a, **k: (_ for _ in ()).throw(
            ValueError("reject")
        )
        decision = _run(arrays).decision
    finally:
        execution.compose_covariance_only_hybrid = original
    with pytest.raises(ValueError, match="complete or absent"):
        replace(decision, donor_covariance_sha256=None, decision_id=None)
