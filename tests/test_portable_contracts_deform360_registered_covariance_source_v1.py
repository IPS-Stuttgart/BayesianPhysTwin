from __future__ import annotations

import inspect

import numpy as np
import pytest

from bayesian_phystwin.deform360_registered_covariance_source_v1 import (
    COVARIANCE_SCALES,
    run_registered_deform360_covariance_source_v1,
)
from bayesian_phystwin.endpoint_model_average import (
    infer_model_averaged_endpoint,
    predict_model_averaged_endpoint,
)

SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64


def _case() -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    residual = np.asarray(
        [
            [[0.001, 0.0, 0.0], [0.0, 0.001, 0.0]],
            [[0.002, 0.0, 0.0], [0.0, 0.002, 0.0]],
            [[0.003, 0.0, 0.0], [0.0, 0.003, 0.0]],
        ],
        dtype=np.float64,
    )
    valid = np.ones((3, 2), dtype=bool)
    physical_mean = np.zeros((3, 2, 3), dtype=np.float64)
    registered_mean = np.array(
        physical_mean + residual[-1][None, ...],
        dtype=np.float64,
        copy=True,
        order="C",
    )
    physical_covariance = np.repeat(
        np.eye(3, dtype=np.float64)[None, None, ...] * 0.01,
        3,
        axis=0,
    )
    physical_covariance = np.repeat(physical_covariance, 2, axis=1)
    bins = np.asarray([0, 1, 2], dtype=np.int64)
    return (
        residual,
        valid,
        registered_mean,
        physical_mean,
        physical_covariance,
        bins,
    )


def _run(
    residual: np.ndarray,
    valid: np.ndarray,
    registered_mean: np.ndarray,
    physical_mean: np.ndarray,
    physical_covariance: np.ndarray,
    bins: np.ndarray,
):
    return run_registered_deform360_covariance_source_v1(
        residual,
        valid,
        registered_mean,
        physical_mean,
        physical_covariance,
        end_frame=3,
        future_horizon_bins=bins,
        source_unit_id="opened-source-object-session-01",
        source_residual_artifact_id=SHA_A,
        registered_reference_artifact_id=SHA_B,
        physical_fallback_belief_id=SHA_C,
        metadata={"protocol": "source-only-v1"},
    )


def test_registered_source_path_preserves_mean_and_matches_frozen_donor() -> None:
    (
        residual,
        valid,
        registered_mean,
        physical_mean,
        physical_covariance,
        bins,
    ) = _case()

    result = _run(
        residual,
        valid,
        registered_mean,
        physical_mean,
        physical_covariance,
        bins,
    )

    assert result.accepted
    assert result.mean_m is registered_mean
    assert result.hybrid is not None
    assert result.record.reason == "accepted"
    assert result.record.descriptor()["covariance_donor_id"] == (
        "independent_endpoint_v1"
    )
    posterior = infer_model_averaged_endpoint(
        residual,
        valid,
        end_frame=3,
    )
    donor = np.stack(
        [
            predict_model_averaged_endpoint(
                posterior,
                horizon_steps=horizon,
            ).covariance_m2
            for horizon in range(1, 4)
        ]
    )
    scale = np.asarray(COVARIANCE_SCALES, dtype=np.float64)[bins]
    np.testing.assert_allclose(
        result.covariance_m2,
        donor * scale[:, None, None, None],
        atol=1e-18,
        rtol=1e-15,
    )
    assert not result.covariance_m2.flags.writeable


def test_registered_mean_mismatch_returns_exact_physical_fallback() -> None:
    case = list(_case())
    case[2] = np.array(case[2], copy=True)
    case[2][0, 0, 0] += 0.001

    result = _run(*case)

    assert not result.accepted
    assert result.record.reason == "registered-reference-mean-mismatch"
    assert result.mean_m is case[3]
    assert result.covariance_m2 is case[4]
    assert result.record.exact_fallback_identity_preserved


def test_one_unsupported_material_returns_whole_case_fallback() -> None:
    case = list(_case())
    valid = np.array(case[1], copy=True)
    valid[1:, 1] = False
    residual = np.array(case[0], copy=True)
    residual[~valid] = 0.0
    case[0] = residual
    case[1] = valid

    result = _run(*case)

    assert not result.accepted
    assert result.record.reason == "insufficient-material-support"
    assert result.mean_m is case[3]
    assert result.covariance_m2 is case[4]


def test_invalid_rows_cannot_hide_residual_values() -> None:
    case = list(_case())
    valid = np.array(case[1], copy=True)
    valid[0, 0] = False
    case[1] = valid

    with pytest.raises(ValueError, match="exact zero"):
        _run(*case)


def test_record_identity_changes_with_source_artifact() -> None:
    case = _case()
    first = _run(*case)
    second = run_registered_deform360_covariance_source_v1(
        case[0],
        case[1],
        case[2],
        case[3],
        case[4],
        end_frame=3,
        future_horizon_bins=case[5],
        source_unit_id="opened-source-object-session-01",
        source_residual_artifact_id="d" * 64,
        registered_reference_artifact_id=SHA_B,
        physical_fallback_belief_id=SHA_C,
        metadata={"protocol": "source-only-v1"},
    )

    assert first.record.record_id != second.record.record_id


def test_post_cutoff_residuals_cannot_change_registered_output() -> None:
    case = list(_case())
    suffix = np.full((2, 2, 3), 10.0, dtype=np.float64)
    case[0] = np.concatenate([case[0], suffix], axis=0)
    case[1] = np.concatenate(
        [case[1], np.ones((2, 2), dtype=bool)],
        axis=0,
    )

    first = _run(*_case())
    second = _run(*case)

    np.testing.assert_array_equal(first.mean_m, second.mean_m)
    np.testing.assert_array_equal(
        first.covariance_m2,
        second.covariance_m2,
    )


def test_caller_cannot_change_registered_donor_or_scale() -> None:
    signature = inspect.signature(run_registered_deform360_covariance_source_v1)

    assert "covariance_donor_id" not in signature.parameters
    assert "covariance_scale" not in signature.parameters
