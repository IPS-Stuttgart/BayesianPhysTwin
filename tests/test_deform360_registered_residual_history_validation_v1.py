from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

import bayesian_phystwin.deform360_registered_residual_history_v1 as subject


def _provenance() -> subject.ResidualHistorySourceProvenanceV1:
    return subject.ResidualHistorySourceProvenanceV1(
        source_inventory_id="a" * 64,
        provider_reconstruction_id="b" * 64,
        scoring_reconstruction_id="c" * 64,
        provider_implementation_revision="1" * 40,
        scoring_implementation_revision="2" * 40,
        provider_configuration_id="d" * 64,
        scoring_configuration_id="e" * 64,
        provider_camera_family_ids=("provider-00", "provider-02"),
        scoring_camera_family_ids=("scoring-01", "scoring-03"),
        provider_input_artifact_ids=("1" * 64, "3" * 64),
        scoring_input_artifact_ids=("2" * 64, "4" * 64),
    )


def _arrays(*, future_count: int = 6) -> dict[str, np.ndarray]:
    physical_prefix = np.zeros((3, 4, 3), dtype=np.float64)
    observation = np.full_like(physical_prefix, np.nan)
    validity = np.zeros((3, 4), dtype=bool)
    validity[0] = True
    validity[1, :2] = True
    validity[2, 2:] = True
    observation[0] = np.asarray(
        [
            [0.010, 0.000, 0.000],
            [0.011, 0.000, 0.000],
            [0.012, 0.000, 0.000],
            [0.013, 0.000, 0.000],
        ],
        dtype=np.float64,
    )
    observation[1, :2] = np.asarray(
        [[0.020, 0.001, 0.000], [0.021, 0.001, 0.000]],
        dtype=np.float64,
    )
    observation[2, 2:] = np.asarray(
        [[0.030, 0.002, 0.000], [0.031, 0.002, 0.000]],
        dtype=np.float64,
    )
    physical_future = np.zeros((future_count, 4, 3), dtype=np.float64)
    registered_mean = np.array(physical_future, copy=True, order="C")
    for track in range(validity.shape[1]):
        support = np.flatnonzero(validity[:, track])
        if len(support):
            frame = int(support[-1])
            registered_mean[:, track] += observation[frame, track]
    return {
        "physical_prefix": physical_prefix,
        "observation": observation,
        "validity": validity,
        "physical_future": physical_future,
        "registered_mean": registered_mean,
        "reference_covariance": np.zeros(
            physical_future.shape + (3,),
            dtype=np.float64,
        ),
    }


def _recompute_registered_mean(arrays: dict[str, np.ndarray]) -> None:
    result = np.array(arrays["physical_future"], copy=True, order="C")
    for track in range(arrays["validity"].shape[1]):
        support = np.flatnonzero(arrays["validity"][:, track])
        if len(support):
            frame = int(support[-1])
            result[:, track] += (
                arrays["observation"][frame, track]
                - arrays["physical_prefix"][frame, track]
            )
    arrays["registered_mean"] = result


def _run(
    arrays: dict[str, np.ndarray],
    *,
    provenance: object | None = None,
    source_unit_id: str = "source-object-session-001",
) -> subject.RegisteredResidualHistoryPredictionV1:
    return subject.run_registered_residual_history_v1(
        arrays["physical_prefix"],
        arrays["observation"],
        arrays["validity"],
        arrays["physical_future"],
        arrays["registered_mean"],
        arrays["reference_covariance"],
        source_unit_id=source_unit_id,
        provenance=_provenance() if provenance is None else provenance,  # type: ignore[arg-type]
    )


def _fallback() -> subject.RegisteredResidualHistoryPredictionV1:
    arrays = _arrays()
    arrays["validity"][:, 3] = False
    arrays["observation"][:, 3] = np.nan
    _recompute_registered_mean(arrays)
    return _run(arrays)


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ("", "nonempty canonical string"),
        (" leading", "nonempty canonical string"),
        ("two\nlines", "single canonical line"),
    ],
)
def test_canonical_string_rejects_ambiguous_values(
    value: object,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        subject._canonical_string(value, name="value")


@pytest.mark.parametrize(
    ("value", "message"),
    [
        (["a"], "canonical tuple"),
        ((), "nonempty"),
        (("b", "a"), "sorted and unique"),
        (("a", "a"), "sorted and unique"),
    ],
)
def test_canonical_string_tuple_rejects_noncanonical_values(
    value: object,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        subject._canonical_string_tuple(value, name="value")


@pytest.mark.parametrize(
    ("value", "message"),
    [
        (["a" * 64], "canonical tuple"),
        ((), "nonempty"),
        (("b" * 64, "a" * 64), "sorted and unique"),
        (("a" * 64, "a" * 64), "sorted and unique"),
    ],
)
def test_digest_tuple_rejects_noncanonical_values(
    value: object,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        subject._digest_tuple(value, name="value")


@pytest.mark.parametrize(
    ("value", "message"),
    [
        (object(), "must be a NumPy array"),
        (np.zeros((2, 3), dtype=np.float32), "dtype float64"),
        (np.zeros(3, dtype=np.float64), "must have 2 dimensions"),
        (np.asfortranarray(np.zeros((2, 3))), "C-contiguous"),
    ],
)
def test_float_array_validator_rejects_lossy_inputs(
    value: object,
    message: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        subject._float64_array(value, name="value", ndim=2)


def test_identity_array_type_error_names_identity_requirement() -> None:
    with pytest.raises(TypeError, match="to preserve identity"):
        subject._float64_array(
            object(),
            name="value",
            ndim=2,
            preserve_identity=True,
        )


@pytest.mark.parametrize(
    ("value", "message"),
    [
        (object(), "must be a NumPy array"),
        (np.zeros((2, 3), dtype=np.int64), "Boolean dtype"),
        (np.zeros(3, dtype=bool), "must have 2 dimensions"),
        (np.asfortranarray(np.zeros((2, 3), dtype=bool)), "C-contiguous"),
    ],
)
def test_boolean_array_validator_rejects_lossy_inputs(
    value: object,
    message: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        subject._boolean_array(value, name="value", ndim=2)


@pytest.mark.parametrize("future_count", [True, 2])
def test_horizon_partition_requires_three_genuine_frames(
    future_count: object,
) -> None:
    with pytest.raises(ValueError, match="at least three future frames"):
        subject._canonical_horizon_bins(future_count)  # type: ignore[arg-type]


def test_horizon_partition_rejects_empty_internal_chunk(monkeypatch: Any) -> None:
    def split_with_empty_chunk(
        _indices: np.ndarray,
        _sections: int,
    ) -> list[np.ndarray]:
        return [
            np.asarray([0], dtype=np.int64),
            np.asarray([], dtype=np.int64),
            np.asarray([1, 2], dtype=np.int64),
        ]

    monkeypatch.setattr(subject.np, "array_split", split_with_empty_chunk)
    with pytest.raises(AssertionError, match="empty bin"):
        subject._canonical_horizon_bins(3)


def test_endpoint_config_descriptor_rejects_wrong_contract_type() -> None:
    with pytest.raises(TypeError, match="ModelAveragedEndpointConfigV1"):
        subject._endpoint_config_descriptor(object())  # type: ignore[arg-type]


def test_provenance_rejects_identical_reconstructions() -> None:
    provenance = _provenance()
    with pytest.raises(ValueError, match="reconstructions must differ"):
        replace(
            provenance,
            scoring_reconstruction_id=provenance.provider_reconstruction_id,
            provenance_id=None,
        )


def _replace_decision(
    decision: subject.RegisteredResidualHistoryDecisionV1,
    **updates: object,
) -> subject.RegisteredResidualHistoryDecisionV1:
    updates.setdefault("decision_id", None)
    return replace(decision, **updates)


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"valid_observation_count_by_track": [2, 2, 2, 2]}, "canonical tuple"),
        ({"valid_observation_count_by_track": (True, 2, 2, 2)}, "must be an integer"),
        ({"valid_observation_count_by_track": (-1, 2, 2, 2)}, "nonnegative"),
        ({"valid_observation_count_by_track": ()}, "must be nonempty"),
        ({"future_horizon_count": True}, "must be an integer"),
        ({"future_horizon_bins": [0, 0, 1, 1, 2, 2]}, "canonical tuple"),
        ({"accepted": 1}, "accepted must be a Boolean"),
        ({"fallback_reasons": []}, "canonical tuple"),
        (
            {
                "fallback_reasons": (
                    "registered-mean-mismatch",
                    "registered-mean-mismatch",
                )
            },
            "sorted and unique",
        ),
        ({"fallback_reasons": ("unsupported",)}, "unsupported reason"),
        ({"endpoint_prediction_ids": []}, "canonical tuple"),
        ({"registered_mean_identity_preserved": 1}, "must be Booleans"),
        ({"registered_mean_identity_preserved": False}, "must always be preserved"),
        ({"endpoint_config_id": None}, "lacks endpoint donor lineage"),
        (
            {"reference_covariance_identity_preserved": True},
            "did not deploy reference covariance",
        ),
    ],
)
def test_decision_rejects_noncanonical_or_inconsistent_fields(
    updates: dict[str, object],
    message: str,
) -> None:
    decision = _run(_arrays()).decision
    with pytest.raises(ValueError, match=message):
        _replace_decision(decision, **updates)


def test_decision_rejects_duplicate_prediction_ids() -> None:
    decision = _run(_arrays()).decision
    duplicate = (decision.endpoint_prediction_ids[0],) * len(
        decision.endpoint_prediction_ids
    )
    with pytest.raises(ValueError, match="must be unique"):
        _replace_decision(decision, endpoint_prediction_ids=duplicate)


def test_decision_recomputes_support_and_mean_failure_reasons() -> None:
    decision = _run(_arrays()).decision
    with pytest.raises(ValueError, match="support fallback reason"):
        _replace_decision(
            decision,
            valid_observation_count_by_track=(1, 2, 2, 2),
        )
    with pytest.raises(ValueError, match="mean mismatch fallback reason"):
        _replace_decision(
            decision,
            reconstructed_reference_mean_sha256="f" * 64,
        )
    with pytest.raises(ValueError, match="accepted decisions cannot contain"):
        _replace_decision(
            decision,
            reconstructed_reference_mean_sha256="f" * 64,
            fallback_reasons=("registered-mean-mismatch",),
        )


def test_fallback_decision_rejects_missing_reason_and_retained_donor() -> None:
    accepted = _run(_arrays()).decision
    fallback_fields = {
        "accepted": False,
        "fallback_reasons": (),
        "endpoint_config_id": None,
        "endpoint_posterior_id": None,
        "endpoint_prediction_ids": (),
        "donor_covariance_sha256": None,
        "hybrid_artifact_id": None,
        "output_covariance_sha256": accepted.reference_covariance_sha256,
        "reference_covariance_identity_preserved": True,
    }
    with pytest.raises(ValueError, match="require at least one reason"):
        _replace_decision(accepted, **fallback_fields)

    fallback = _fallback().decision
    with pytest.raises(ValueError, match="must not retain donor execution"):
        _replace_decision(fallback, endpoint_config_id="f" * 64)
    with pytest.raises(ValueError, match="preserve reference covariance identity"):
        _replace_decision(
            fallback,
            reference_covariance_identity_preserved=False,
        )
    with pytest.raises(ValueError, match="differs from the reference"):
        _replace_decision(fallback, output_covariance_sha256="f" * 64)


def test_prediction_contract_rejects_mismatched_objects_and_content() -> None:
    accepted = _run(_arrays())
    fallback = _fallback()
    different_provenance = replace(
        accepted.provenance,
        source_inventory_id="f" * 64,
        provenance_id=None,
    )

    with pytest.raises(TypeError, match="provenance must be"):
        replace(accepted, provenance=object())
    with pytest.raises(TypeError, match="decision must be"):
        replace(accepted, decision=object())
    with pytest.raises(ValueError, match="provenance differ"):
        replace(accepted, provenance=different_provenance)
    with pytest.raises(ValueError, match="mean shape differs"):
        replace(accepted, mean_m=np.zeros((1, 1, 3), dtype=np.float64))
    with pytest.raises(ValueError, match="covariance shape differs"):
        replace(accepted, covariance_m2=np.zeros((1, 1, 3, 3), dtype=np.float64))

    changed_mean = np.array(accepted.mean_m, copy=True)
    changed_mean[0, 0, 0] += 1.0
    with pytest.raises(ValueError, match="mean content differs"):
        replace(accepted, mean_m=changed_mean)
    changed_covariance = np.array(accepted.covariance_m2, copy=True)
    changed_covariance[0, 0, 0, 0] += 1.0
    with pytest.raises(ValueError, match="covariance content differs"):
        replace(accepted, covariance_m2=changed_covariance)
    with pytest.raises(ValueError, match="missing the covariance hybrid"):
        replace(accepted, hybrid=None)
    with pytest.raises(ValueError, match="does not retain hybrid objects"):
        replace(accepted, mean_m=np.array(accepted.mean_m, copy=True))
    with pytest.raises(ValueError, match="must not retain a covariance hybrid"):
        replace(fallback, hybrid=accepted.hybrid)


def _call_validation(arrays: dict[str, Any]) -> None:
    subject._validate_execution_arrays(
        arrays["physical_prefix"],
        arrays["observation"],
        arrays["validity"],
        arrays["physical_future"],
        arrays["registered_mean"],
        arrays["reference_covariance"],
    )


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        ("physical_prefix", np.zeros((2, 4, 3)), "matching shape"),
        ("physical_future", np.zeros((2, 4, 3)), "H>=3"),
        ("physical_future", np.zeros((6, 5, 3)), "track rosters differ"),
        ("registered_mean", np.zeros((5, 4, 3)), "shape changed"),
        ("reference_covariance", np.zeros((5, 4, 3, 3)), "shape changed"),
    ],
)
def test_execution_array_shapes_fail_closed(
    field: str,
    replacement: np.ndarray,
    message: str,
) -> None:
    arrays: dict[str, Any] = _arrays()
    arrays[field] = replacement
    with pytest.raises(ValueError, match=message):
        _call_validation(arrays)


@pytest.mark.parametrize(
    ("field", "index", "message"),
    [
        ("physical_prefix", (0, 0, 0), "physical_prefix_m must be finite"),
        ("physical_future", (0, 0, 0), "physical_future_m must be finite"),
        ("registered_mean", (0, 0, 0), "registered_last_residual_mean_m must be finite"),
        ("reference_covariance", (0, 0, 0, 0), "reference_covariance_m2 must be finite"),
        ("observation", (0, 0, 0), "valid provider observations must be finite"),
    ],
)
def test_execution_nonfinite_values_fail_closed(
    field: str,
    index: tuple[int, ...],
    message: str,
) -> None:
    arrays: dict[str, Any] = _arrays()
    arrays[field] = np.array(arrays[field], copy=True, order="C")
    arrays[field][index] = np.inf
    with pytest.raises(ValueError, match=message):
        _call_validation(arrays)


def test_execution_accepts_no_invalid_rows_and_handles_zero_support_track() -> None:
    arrays = _arrays()
    arrays["validity"][:] = True
    arrays["observation"] = np.nan_to_num(arrays["observation"])
    _recompute_registered_mean(arrays)
    assert _run(arrays).accepted

    arrays = _arrays()
    arrays["validity"][:, 0] = False
    arrays["observation"][:, 0] = np.nan
    _recompute_registered_mean(arrays)
    result = _run(arrays)
    assert not result.accepted
    assert result.decision.valid_observation_count_by_track[0] == 0


def test_execution_rejects_invalid_source_identity_and_provenance_type() -> None:
    with pytest.raises(ValueError, match="single canonical line"):
        _run(_arrays(), source_unit_id="two\nlines")
    with pytest.raises(TypeError, match="provenance must be"):
        _run(_arrays(), provenance=object())


def test_combined_support_and_mean_failure_is_preserved() -> None:
    arrays = _arrays()
    arrays["validity"][:, 0] = False
    arrays["observation"][:, 0] = np.nan
    arrays["registered_mean"][0, 1, 0] += 1.0
    result = _run(arrays)
    assert result.decision.fallback_reasons == (
        "insufficient-per-track-support",
        "registered-mean-mismatch",
    )


def test_endpoint_observation_count_mismatch_fails_closed(monkeypatch: Any) -> None:
    monkeypatch.setattr(
        subject,
        "infer_model_averaged_endpoint",
        lambda *_args, **_kwargs: SimpleNamespace(
            update_count=np.zeros(4, dtype=np.int64)
        ),
    )
    with pytest.raises(AssertionError, match="different causal observations"):
        _run(_arrays())


def test_hybrid_mean_ownership_mismatch_fails_closed(monkeypatch: Any) -> None:
    def copied_mean_hybrid(
        mean: np.ndarray,
        donor_covariance: np.ndarray,
        **_kwargs: object,
    ) -> SimpleNamespace:
        return SimpleNamespace(
            mean_m=np.array(mean, copy=True),
            covariance_m2=np.zeros(donor_covariance.shape, dtype=np.float64),
            record=SimpleNamespace(artifact_id="f" * 64),
        )

    monkeypatch.setattr(subject, "compose_covariance_only_hybrid", copied_mean_hybrid)
    with pytest.raises(AssertionError, match="copied the reference mean"):
        _run(_arrays())


@pytest.mark.parametrize("copy_covariance", [False, True])
def test_fallback_internal_identity_assertions(
    monkeypatch: Any,
    copy_covariance: bool,
) -> None:
    def copied_fallback(**kwargs: object) -> SimpleNamespace:
        mean = kwargs["mean_m"]
        covariance = kwargs["covariance_m2"]
        return SimpleNamespace(
            mean_m=mean if copy_covariance else np.array(mean, copy=True),
            covariance_m2=(
                np.array(covariance, copy=True) if copy_covariance else covariance
            ),
        )

    monkeypatch.setattr(subject, "RegisteredResidualHistoryPredictionV1", copied_fallback)
    message = (
        "fallback copied the reference covariance"
        if copy_covariance
        else "fallback copied the registered mean"
    )
    with pytest.raises(AssertionError, match=message):
        _fallback()
