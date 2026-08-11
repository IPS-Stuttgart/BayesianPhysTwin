from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

import bayesian_phystwin.deform360_registered_residual_history_v1 as source_module
from bayesian_phystwin.deform360_registered_residual_history_v1 import (
    RegisteredResidualHistoryDecisionV1,
    RegisteredResidualHistoryPredictionV1,
    ResidualHistorySourceProvenanceV1,
    run_registered_residual_history_v1,
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


def _arrays(*, future_count: int = 6) -> dict[str, Any]:
    physical_prefix = np.zeros((3, 2, 3), dtype=np.float64)
    observation = np.full_like(physical_prefix, np.nan)
    validity = np.zeros((3, 2), dtype=bool)
    validity[:2] = True
    observation[:2] = np.asarray(
        [
            [[0.010, 0.001, 0.000], [0.020, 0.002, 0.000]],
            [[0.011, 0.001, 0.000], [0.021, 0.002, 0.000]],
        ],
        dtype=np.float64,
    )
    physical_future = np.zeros((future_count, 2, 3), dtype=np.float64)
    physical_future[..., 2] = (
        np.arange(future_count, dtype=np.float64)[:, None] * 0.001
    )
    reference_covariance = np.zeros(
        physical_future.shape + (3,),
        dtype=np.float64,
    )
    arrays: dict[str, Any] = {
        "physical_prefix": physical_prefix,
        "observation": observation,
        "validity": validity,
        "physical_future": physical_future,
        "reference_covariance": reference_covariance,
    }
    arrays["registered_mean"] = _registered_mean(arrays)
    return arrays


def _registered_mean(arrays: dict[str, Any]) -> np.ndarray:
    physical_prefix = cast(np.ndarray, arrays["physical_prefix"])
    observation = cast(np.ndarray, arrays["observation"])
    validity = cast(np.ndarray, arrays["validity"])
    physical_future = cast(np.ndarray, arrays["physical_future"])
    result = np.array(physical_future, copy=True, order="C")
    for track in range(validity.shape[1]):
        support = np.flatnonzero(validity[:, track])
        if len(support):
            frame = int(support[-1])
            result[:, track] += (
                observation[frame, track] - physical_prefix[frame, track]
            )
    return result


def _execute(
    arrays: dict[str, Any],
    *,
    provenance: object | None = None,
    registered_mean: object | None = None,
) -> RegisteredResidualHistoryPredictionV1:
    selected_provenance = _provenance() if provenance is None else provenance
    selected_mean = (
        arrays["registered_mean"] if registered_mean is None else registered_mean
    )
    return run_registered_residual_history_v1(
        cast(np.ndarray, arrays["physical_prefix"]),
        cast(np.ndarray, arrays["observation"]),
        cast(np.ndarray, arrays["validity"]),
        cast(np.ndarray, arrays["physical_future"]),
        cast(np.ndarray, selected_mean),
        cast(np.ndarray, arrays["reference_covariance"]),
        source_unit_id="opened-source-object-session-001",
        provenance=cast(ResidualHistorySourceProvenanceV1, selected_provenance),
    )


def _support_fallback() -> RegisteredResidualHistoryPredictionV1:
    arrays = _arrays()
    arrays["validity"][:, 1] = False
    arrays["observation"][:, 1] = np.nan
    arrays["registered_mean"] = _registered_mean(arrays)
    return _execute(arrays)


def _mean_mismatch_fallback() -> RegisteredResidualHistoryPredictionV1:
    arrays = _arrays()
    registered_mean = np.array(arrays["registered_mean"], copy=True, order="C")
    registered_mean[0, 0, 0] += 0.001
    return _execute(arrays, registered_mean=registered_mean)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        (
            {"provider_reconstruction_id": "c" * 64},
            "provider and scoring reconstructions must differ",
        ),
        (
            {"provider_camera_family_ids": ["recorder-family-00"]},
            "must be a canonical tuple",
        ),
        ({"provider_camera_family_ids": ()}, "must be nonempty"),
        (
            {
                "provider_camera_family_ids": (
                    "recorder-family-02",
                    "recorder-family-00",
                )
            },
            "must be sorted and unique",
        ),
        (
            {"provider_camera_family_ids": ("bad\nfamily",)},
            "must be a single canonical line",
        ),
        (
            {"provider_input_artifact_ids": ["1" * 64]},
            "must be a canonical tuple",
        ),
        ({"provider_input_artifact_ids": ()}, "must be nonempty"),
        (
            {"provider_input_artifact_ids": ("3" * 64, "1" * 64)},
            "must be sorted and unique",
        ),
    ],
)
def test_provenance_canonicalization_failures(
    changes: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        replace(_provenance(), **changes, provenance_id=None)


@pytest.mark.parametrize("future_count", [True, 2])
def test_canonical_horizon_count_rejects_invalid_values(
    future_count: object,
) -> None:
    with pytest.raises(ValueError, match="at least three future frames"):
        source_module._canonical_horizon_bins(cast(int, future_count))


def test_canonical_horizon_partition_rejects_empty_chunks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def empty_middle_chunk(
        indices: np.ndarray,
        sections: int,
    ) -> list[np.ndarray]:
        assert sections == 3
        return [indices[:1], indices[1:1], indices[1:]]

    monkeypatch.setattr(source_module.np, "array_split", empty_middle_chunk)
    with pytest.raises(AssertionError, match="empty bin"):
        source_module._canonical_horizon_bins(3)


def test_endpoint_config_descriptor_rejects_wrong_type() -> None:
    with pytest.raises(TypeError, match="ModelAveragedEndpointConfigV1"):
        source_module._endpoint_config_descriptor(object())


@pytest.mark.parametrize(
    ("kind", "error_type", "message"),
    [
        ("not-array", TypeError, "must be a NumPy array"),
        ("dtype", ValueError, "must have dtype float64"),
        ("dimensions", ValueError, "must have 3 dimensions"),
        ("storage", ValueError, "must be C-contiguous"),
        ("identity-not-array", TypeError, "to preserve identity"),
    ],
)
def test_float64_execution_array_contracts(
    kind: str,
    error_type: type[Exception],
    message: str,
) -> None:
    arrays = _arrays()
    registered_mean: object | None = None
    if kind == "not-array":
        arrays["physical_prefix"] = arrays["physical_prefix"].tolist()
    elif kind == "dtype":
        arrays["physical_prefix"] = arrays["physical_prefix"].astype(np.float32)
    elif kind == "dimensions":
        arrays["physical_prefix"] = np.zeros((3, 2), dtype=np.float64)
    elif kind == "storage":
        arrays["physical_prefix"] = np.zeros(
            (3, 2, 6),
            dtype=np.float64,
        )[..., ::2]
    else:
        registered_mean = arrays["registered_mean"].tolist()

    with pytest.raises(error_type, match=message):
        _execute(arrays, registered_mean=registered_mean)


@pytest.mark.parametrize(
    ("kind", "error_type", "message"),
    [
        ("not-array", TypeError, "must be a NumPy array"),
        ("dtype", ValueError, "must have Boolean dtype"),
        ("dimensions", ValueError, "must have 2 dimensions"),
        ("storage", ValueError, "must be C-contiguous"),
    ],
)
def test_validity_array_contracts(
    kind: str,
    error_type: type[Exception],
    message: str,
) -> None:
    arrays = _arrays()
    if kind == "not-array":
        arrays["validity"] = arrays["validity"].tolist()
    elif kind == "dtype":
        arrays["validity"] = arrays["validity"].astype(np.int64)
    elif kind == "dimensions":
        arrays["validity"] = np.zeros((3, 2, 1), dtype=bool)
    else:
        arrays["validity"] = np.zeros((3, 4), dtype=bool)[:, ::2]

    with pytest.raises(error_type, match=message):
        _execute(arrays)


@pytest.mark.parametrize(
    ("kind", "message"),
    [
        ("prefix-shape", "prefix arrays must have matching shape"),
        ("prefix-coordinate", "prefix arrays must have matching shape"),
        ("empty-prefix", "prefix arrays must have matching shape"),
        ("future-coordinate", "physical_future_m must have shape"),
        ("future-short", "physical_future_m must have shape"),
        ("roster", "prefix and future track rosters differ"),
        ("registered-shape", "registered_last_residual_mean_m shape changed"),
        ("covariance-shape", "reference_covariance_m2 shape changed"),
        ("prefix-nonfinite", "physical_prefix_m must be finite"),
        ("future-nonfinite", "physical_future_m must be finite"),
        ("registered-nonfinite", "registered_last_residual_mean_m must be finite"),
        ("covariance-nonfinite", "reference_covariance_m2 must be finite"),
        ("valid-observation-nonfinite", "valid provider observations must be finite"),
    ],
)
def test_execution_shape_and_finiteness_contracts(
    kind: str,
    message: str,
) -> None:
    arrays = _arrays()
    if kind == "prefix-shape":
        arrays["observation"] = arrays["observation"][:-1]
    elif kind == "prefix-coordinate":
        arrays["physical_prefix"] = np.zeros((3, 2, 2), dtype=np.float64)
        arrays["observation"] = np.zeros((3, 2, 2), dtype=np.float64)
    elif kind == "empty-prefix":
        arrays["physical_prefix"] = np.zeros((0, 2, 3), dtype=np.float64)
        arrays["observation"] = np.zeros((0, 2, 3), dtype=np.float64)
        arrays["validity"] = np.zeros((0, 2), dtype=bool)
    elif kind == "future-coordinate":
        arrays["physical_future"] = np.zeros((6, 2, 2), dtype=np.float64)
        arrays["registered_mean"] = np.zeros((6, 2, 2), dtype=np.float64)
        arrays["reference_covariance"] = np.zeros((6, 2, 2, 3), dtype=np.float64)
    elif kind == "future-short":
        arrays["physical_future"] = np.zeros((2, 2, 3), dtype=np.float64)
        arrays["registered_mean"] = np.zeros((2, 2, 3), dtype=np.float64)
        arrays["reference_covariance"] = np.zeros((2, 2, 3, 3), dtype=np.float64)
    elif kind == "roster":
        arrays["physical_future"] = np.zeros((6, 1, 3), dtype=np.float64)
        arrays["registered_mean"] = np.zeros((6, 1, 3), dtype=np.float64)
        arrays["reference_covariance"] = np.zeros((6, 1, 3, 3), dtype=np.float64)
    elif kind == "registered-shape":
        arrays["registered_mean"] = arrays["registered_mean"][:-1]
    elif kind == "covariance-shape":
        arrays["reference_covariance"] = arrays["reference_covariance"][:-1]
    elif kind == "prefix-nonfinite":
        arrays["physical_prefix"][0, 0, 0] = np.inf
    elif kind == "future-nonfinite":
        arrays["physical_future"][0, 0, 0] = np.nan
    elif kind == "registered-nonfinite":
        arrays["registered_mean"][0, 0, 0] = np.inf
    elif kind == "covariance-nonfinite":
        arrays["reference_covariance"][0, 0, 0, 0] = np.nan
    else:
        arrays["observation"][0, 0, 0] = np.nan

    with pytest.raises(ValueError, match=message):
        _execute(arrays)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        (
            {"valid_observation_count_by_track": [2, 2]},
            "must be a canonical tuple",
        ),
        (
            {"valid_observation_count_by_track": (True, 2)},
            "must be an integer",
        ),
        (
            {"valid_observation_count_by_track": (-1, 2)},
            "must be nonnegative",
        ),
        (
            {"valid_observation_count_by_track": ()},
            "must be nonempty",
        ),
        (
            {"future_horizon_count": True},
            "future_horizon_count must be an integer",
        ),
        (
            {"future_horizon_bins": [0, 0, 1, 1, 2, 2]},
            "future_horizon_bins must be a canonical tuple",
        ),
        ({"accepted": 1}, "accepted must be a Boolean"),
        (
            {"fallback_reasons": ["registered-mean-mismatch"]},
            "fallback_reasons must be a canonical tuple",
        ),
        (
            {
                "fallback_reasons": (
                    "registered-mean-mismatch",
                    "registered-mean-mismatch",
                )
            },
            "fallback_reasons must be sorted and unique",
        ),
        (
            {"fallback_reasons": ("unsupported-reason",)},
            "fallback_reasons contain an unsupported reason",
        ),
        (
            {"endpoint_config_id": "not-a-digest"},
            "endpoint_config_id",
        ),
        (
            {"endpoint_prediction_ids": ["1" * 64] * 6},
            "endpoint_prediction_ids must be a canonical tuple",
        ),
        (
            {"endpoint_prediction_ids": ("1" * 64,) * 6},
            "endpoint_prediction_ids must be unique",
        ),
        (
            {"registered_mean_identity_preserved": 1},
            "identity-preservation fields must be Booleans",
        ),
        (
            {"reference_covariance_identity_preserved": 1},
            "identity-preservation fields must be Booleans",
        ),
        (
            {"registered_mean_identity_preserved": False},
            "registered mean identity must always be preserved",
        ),
    ],
)
def test_decision_field_contracts(
    changes: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        replace(_execute(_arrays()).decision, **changes, decision_id=None)


def test_decision_support_and_mean_reasons_are_content_derived() -> None:
    decision = _execute(_arrays()).decision
    with pytest.raises(ValueError, match="support fallback reason differs"):
        replace(
            decision,
            valid_observation_count_by_track=(1, 2),
            decision_id=None,
        )
    with pytest.raises(ValueError, match="mean mismatch fallback reason differs"):
        replace(
            decision,
            reconstructed_reference_mean_sha256="f" * 64,
            decision_id=None,
        )


def test_accepted_decision_requires_empty_reasons_and_complete_lineage() -> None:
    decision = _execute(_arrays()).decision
    with pytest.raises(ValueError, match="cannot contain fallback reasons"):
        replace(
            decision,
            valid_observation_count_by_track=(1, 2),
            fallback_reasons=("insufficient-per-track-support",),
            decision_id=None,
        )
    with pytest.raises(ValueError, match="lacks endpoint donor lineage"):
        replace(decision, endpoint_config_id=None, decision_id=None)
    with pytest.raises(ValueError, match="did not deploy reference covariance"):
        replace(
            decision,
            reference_covariance_identity_preserved=True,
            decision_id=None,
        )


def test_fallback_decision_requires_reason_no_donor_and_exact_covariance() -> None:
    mismatch = _mean_mismatch_fallback().decision
    with pytest.raises(ValueError, match="require at least one reason"):
        replace(
            mismatch,
            reconstructed_reference_mean_sha256=mismatch.registered_mean_sha256,
            fallback_reasons=(),
            decision_id=None,
        )

    fallback = _support_fallback().decision
    with pytest.raises(ValueError, match="must not retain donor execution"):
        replace(fallback, endpoint_config_id="a" * 64, decision_id=None)
    with pytest.raises(ValueError, match="preserve reference covariance identity"):
        replace(
            fallback,
            reference_covariance_identity_preserved=False,
            decision_id=None,
        )
    with pytest.raises(ValueError, match="differs from the reference"):
        replace(fallback, output_covariance_sha256="f" * 64, decision_id=None)


def test_source_unit_id_is_canonical() -> None:
    arrays = _arrays()
    with pytest.raises(ValueError, match="source_unit_id"):
        run_registered_residual_history_v1(
            arrays["physical_prefix"],
            arrays["observation"],
            arrays["validity"],
            arrays["physical_future"],
            arrays["registered_mean"],
            arrays["reference_covariance"],
            source_unit_id="bad\nsource",
            provenance=_provenance(),
        )


def test_result_rejects_wrong_contract_types_and_provenance() -> None:
    result = _execute(_arrays())
    with pytest.raises(TypeError, match="provenance"):
        RegisteredResidualHistoryPredictionV1(
            result.mean_m,
            result.covariance_m2,
            cast(Any, object()),
            result.decision,
            result.hybrid,
        )
    with pytest.raises(TypeError, match="decision"):
        RegisteredResidualHistoryPredictionV1(
            result.mean_m,
            result.covariance_m2,
            result.provenance,
            cast(Any, object()),
            result.hybrid,
        )
    other_provenance = replace(
        result.provenance,
        source_inventory_id="f" * 64,
        provenance_id=None,
    )
    with pytest.raises(ValueError, match="decision and source provenance differ"):
        RegisteredResidualHistoryPredictionV1(
            result.mean_m,
            result.covariance_m2,
            other_provenance,
            result.decision,
            result.hybrid,
        )


def test_result_rejects_shape_content_and_hybrid_mismatches() -> None:
    result = _execute(_arrays())
    with pytest.raises(ValueError, match="mean shape"):
        RegisteredResidualHistoryPredictionV1(
            result.mean_m[:-1],
            result.covariance_m2,
            result.provenance,
            result.decision,
            result.hybrid,
        )
    with pytest.raises(ValueError, match="covariance shape"):
        RegisteredResidualHistoryPredictionV1(
            result.mean_m,
            result.covariance_m2[:-1],
            result.provenance,
            result.decision,
            result.hybrid,
        )

    changed_mean = np.array(result.mean_m, copy=True)
    changed_mean[0, 0, 0] += 1.0
    with pytest.raises(ValueError, match="mean content"):
        RegisteredResidualHistoryPredictionV1(
            changed_mean,
            result.covariance_m2,
            result.provenance,
            result.decision,
            result.hybrid,
        )

    changed_covariance = np.array(result.covariance_m2, copy=True)
    changed_covariance[0, 0, 0, 0] += 1.0
    with pytest.raises(ValueError, match="covariance content"):
        RegisteredResidualHistoryPredictionV1(
            result.mean_m,
            changed_covariance,
            result.provenance,
            result.decision,
            result.hybrid,
        )

    with pytest.raises(ValueError, match="missing the covariance hybrid"):
        RegisteredResidualHistoryPredictionV1(
            result.mean_m,
            result.covariance_m2,
            result.provenance,
            result.decision,
            None,
        )
    with pytest.raises(ValueError, match="retain hybrid objects"):
        RegisteredResidualHistoryPredictionV1(
            np.array(result.mean_m, copy=True),
            result.covariance_m2,
            result.provenance,
            result.decision,
            result.hybrid,
        )

    fallback = _support_fallback()
    with pytest.raises(ValueError, match="must not retain a covariance hybrid"):
        RegisteredResidualHistoryPredictionV1(
            fallback.mean_m,
            fallback.covariance_m2,
            fallback.provenance,
            fallback.decision,
            result.hybrid,
        )


def test_zero_support_and_combined_fallback_reasons() -> None:
    arrays = _arrays()
    arrays["validity"][:, 1] = False
    arrays["observation"][:, 1] = np.nan
    arrays["registered_mean"] = _registered_mean(arrays)
    zero_support = _execute(arrays)
    assert zero_support.decision.valid_observation_count_by_track == (2, 0)

    changed_mean = np.array(arrays["registered_mean"], copy=True, order="C")
    changed_mean[0, 0, 0] += 0.001
    combined = _execute(arrays, registered_mean=changed_mean)
    assert combined.decision.fallback_reasons == (
        "insufficient-per-track-support",
        "registered-mean-mismatch",
    )


def test_run_rejects_wrong_provenance_type() -> None:
    with pytest.raises(TypeError, match="provenance"):
        _execute(_arrays(), provenance=object())


def test_endpoint_update_count_assertion_is_retained(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def mismatched_posterior(*args: object, **kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(update_count=np.zeros(2, dtype=np.int64))

    monkeypatch.setattr(
        source_module,
        "infer_model_averaged_endpoint",
        mismatched_posterior,
    )
    with pytest.raises(AssertionError, match="different causal observations"):
        _execute(_arrays())


def test_registered_mean_identity_assertion_is_retained(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def copied_mean_hybrid(*args: object, **kwargs: object) -> SimpleNamespace:
        registered_mean = cast(np.ndarray, args[0])
        return SimpleNamespace(mean_m=np.array(registered_mean, copy=True))

    monkeypatch.setattr(
        source_module,
        "compose_covariance_only_hybrid",
        copied_mean_hybrid,
    )
    with pytest.raises(AssertionError, match="copied the reference mean"):
        _execute(_arrays())


@pytest.mark.parametrize("copied", ["mean", "covariance"])
def test_fallback_identity_assertions_are_retained(
    copied: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def copied_result(**kwargs: object) -> SimpleNamespace:
        mean = cast(np.ndarray, kwargs["mean_m"])
        covariance = cast(np.ndarray, kwargs["covariance_m2"])
        return SimpleNamespace(
            mean_m=np.array(mean, copy=True) if copied == "mean" else mean,
            covariance_m2=(
                np.array(covariance, copy=True)
                if copied == "covariance"
                else covariance
            ),
        )

    monkeypatch.setattr(
        source_module,
        "RegisteredResidualHistoryPredictionV1",
        copied_result,
    )
    message = (
        "fallback copied the registered mean"
        if copied == "mean"
        else "fallback copied the reference covariance"
    )
    with pytest.raises(AssertionError, match=message):
        _support_fallback()
