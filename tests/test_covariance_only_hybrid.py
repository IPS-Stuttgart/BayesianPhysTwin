from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from bayesian_phystwin.covariance_only_hybrid import (
    CovarianceOnlyHybridRecordV1,
    compose_covariance_only_hybrid,
)


def _mean() -> np.ndarray:
    return np.arange(24, dtype=np.float64).reshape(2, 4, 3) / 1000.0


def _covariance() -> np.ndarray:
    covariance = np.zeros((2, 4, 3, 3), dtype=np.float64)
    covariance[..., 0, 0] = 1.0e-4
    covariance[..., 1, 1] = 2.0e-4
    covariance[..., 2, 2] = 3.0e-4
    covariance[..., 0, 1] = 1.0e-5
    covariance[..., 1, 0] = 1.0e-5
    return covariance


def test_composition_preserves_exact_mean_object_and_scales_covariance() -> None:
    mean = _mean()
    covariance = _covariance()
    schedule = np.asarray([[0.5], [2.0]], dtype=np.float64)

    prediction = compose_covariance_only_hybrid(
        mean,
        covariance,
        reference_predictor_id="last_residual",
        covariance_donor_id="independent_endpoint_v1",
        covariance_scale=schedule,
        metadata={"horizons": ["early", "late"]},
    )

    assert prediction.mean_m is mean
    np.testing.assert_array_equal(prediction.mean_m, mean)
    np.testing.assert_allclose(
        prediction.covariance_m2,
        covariance * np.broadcast_to(schedule, (2, 4))[..., None, None],
    )
    assert not prediction.covariance_m2.flags.writeable
    assert prediction.record.mean_object_identity_preserved
    assert not prediction.record.point_prediction_changed
    assert prediction.record.minimum_covariance_scale == 0.5
    assert prediction.record.maximum_covariance_scale == 2.0
    assert prediction.record.artifact_id is not None


def test_composition_does_not_mutate_donor_or_reference() -> None:
    mean = _mean()
    covariance = _covariance()
    mean_snapshot = mean.copy()
    covariance_snapshot = covariance.copy()

    prediction = compose_covariance_only_hybrid(
        mean,
        covariance,
        reference_predictor_id="last_residual",
        covariance_donor_id="dynamic_endpoint_v2",
        covariance_scale=4.0,
    )

    np.testing.assert_array_equal(mean, mean_snapshot)
    np.testing.assert_array_equal(covariance, covariance_snapshot)
    np.testing.assert_allclose(prediction.covariance_m2, 4.0 * covariance)


def test_record_identity_is_deterministic_and_tamper_evident() -> None:
    first = compose_covariance_only_hybrid(
        _mean(),
        _covariance(),
        reference_predictor_id="last_residual",
        covariance_donor_id="independent_endpoint_v1",
        covariance_scale=1.0,
    ).record
    second = compose_covariance_only_hybrid(
        _mean(),
        _covariance(),
        reference_predictor_id="last_residual",
        covariance_donor_id="independent_endpoint_v1",
        covariance_scale=1.0,
    ).record

    assert second.artifact_id == first.artifact_id
    with pytest.raises(ValueError, match="artifact_id"):
        replace(first, maximum_covariance_scale=2.0)


def test_record_rejects_claim_that_point_prediction_changed() -> None:
    record = compose_covariance_only_hybrid(
        _mean(),
        _covariance(),
        reference_predictor_id="last_residual",
        covariance_donor_id="independent_endpoint_v1",
    ).record

    values = record.descriptor()
    with pytest.raises(ValueError, match="point_prediction_changed"):
        CovarianceOnlyHybridRecordV1(
            reference_predictor_id=str(values["reference_predictor_id"]),
            covariance_donor_id=str(values["covariance_donor_id"]),
            mean_shape=tuple(values["mean_shape"]),
            covariance_shape=tuple(values["covariance_shape"]),
            reference_mean_sha256=str(values["reference_mean_sha256"]),
            donor_covariance_sha256=str(values["donor_covariance_sha256"]),
            scale_schedule_sha256=str(values["scale_schedule_sha256"]),
            output_covariance_sha256=str(values["output_covariance_sha256"]),
            minimum_covariance_scale=float(values["minimum_covariance_scale"]),
            maximum_covariance_scale=float(values["maximum_covariance_scale"]),
            mean_object_identity_preserved=True,
            point_prediction_changed=True,
            metadata={},
        )


@pytest.mark.parametrize(
    ("mean", "covariance", "scale", "match"),
    [
        (np.zeros((2, 3), dtype=np.float32), np.zeros((2, 3, 3)), 1.0, "float64"),
        (np.zeros((2, 3)), np.zeros((2, 2, 2)), 1.0, "shape"),
        (
            np.zeros((2, 3)),
            np.asarray(
                [[[1.0, 2.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]] * 2
            ),
            1.0,
            "symmetric",
        ),
        (
            np.zeros((2, 3)),
            np.asarray(
                [[[1.0, 2.0, 0.0], [2.0, 1.0, 0.0], [0.0, 0.0, 1.0]]] * 2
            ),
            1.0,
            "positive semidefinite",
        ),
        (np.zeros((2, 3)), np.zeros((2, 3, 3)), 0.0, "strictly positive"),
        (np.zeros((2, 3)), np.zeros((2, 3, 3)), np.ones((4,)), "broadcast"),
    ],
)
def test_composition_rejects_malformed_inputs(
    mean: np.ndarray,
    covariance: np.ndarray,
    scale: object,
    match: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=match):
        compose_covariance_only_hybrid(
            mean,
            covariance,
            reference_predictor_id="last_residual",
            covariance_donor_id="independent_endpoint_v1",
            covariance_scale=scale,
        )


def test_reference_mean_must_be_numpy_array_for_object_identity() -> None:
    with pytest.raises(TypeError, match="NumPy array"):
        compose_covariance_only_hybrid(
            [[0.0, 0.0, 0.0]],  # type: ignore[arg-type]
            np.zeros((1, 3, 3)),
            reference_predictor_id="last_residual",
            covariance_donor_id="independent_endpoint_v1",
        )
