from __future__ import annotations

import json

import numpy as np
import pytest

from bayesian_phystwin.contracts.fixed_anchor import FixedBayesianAnchorConfigV1
from bayesian_phystwin.endpoint_model_average import (
    ModelAveragedEndpointConfigV1,
    infer_model_averaged_endpoint,
)
from bayesian_phystwin.horizon_discrepancy import (
    HorizonDiscrepancyCalibrationV1,
    fit_horizon_discrepancy_calibration,
    load_horizon_discrepancy_calibration,
    mean_retention_at_horizon,
    predict_horizon_conditioned_endpoint,
    save_horizon_discrepancy_calibration,
)


def _posterior():
    config = ModelAveragedEndpointConfigV1(
        components=(
            FixedBayesianAnchorConfigV1(
                process_std_m=0.001,
                observation_std_m=0.0025,
            ),
            FixedBayesianAnchorConfigV1(
                process_std_m=0.004,
                observation_std_m=0.005,
            ),
        )
    )
    residual = np.array(
        [
            [[0.010, 0.000, 0.000], [0.000, 0.008, 0.000]],
            [[0.012, 0.001, 0.000], [0.001, 0.009, 0.000]],
            [[0.011, 0.002, 0.001], [0.002, 0.010, 0.001]],
        ]
    )
    return infer_model_averaged_endpoint(
        residual,
        np.ones((3, 2), dtype=bool),
        end_frame=3,
        config=config,
    )


def _source_fixture():
    groups = ("source-c", "source-a", "source-f", "source-b", "source-e", "source-d")
    horizon = np.array([4, 8, 16], dtype=np.int64)
    endpoint = np.array(
        [
            [0.010, -0.004, 0.002],
            [0.006, 0.003, -0.002],
            [-0.008, 0.005, 0.004],
            [0.004, -0.006, 0.003],
            [-0.005, -0.003, 0.006],
            [0.009, 0.002, -0.004],
        ]
    )
    retention = 0.25 + 0.75 * np.power(2.0, -horizon / 8.0)
    future = retention[None, :, None] * endpoint[:, None, :]
    return groups, horizon, endpoint, future


def test_fit_selects_source_supported_mean_reversion() -> None:
    groups, horizon, endpoint, future = _source_fixture()

    calibration = fit_horizon_discrepancy_calibration(
        groups,
        endpoint,
        future,
        horizon,
        half_life_candidates=(None, 4.0, 8.0, 16.0),
        minimum_retention_candidates=(0.0, 0.25, 0.5),
    )

    assert calibration.source_group_ids == tuple(sorted(groups))
    assert calibration.mean_reversion_half_life_steps == 8.0
    assert calibration.minimum_mean_retention == 0.25
    assert np.all(calibration.process_std_m_per_sqrt_step > 0.0)
    assert calibration.source_outcomes_used
    assert not calibration.interval_calibration_outcomes_used
    assert not calibration.confirmation_outcomes_used
    assert not calibration.target_outcomes_used
    assert np.isclose(mean_retention_at_horizon(calibration, 8), 0.625)


def test_fit_is_invariant_to_source_group_order() -> None:
    groups, horizon, endpoint, future = _source_fixture()
    forward = fit_horizon_discrepancy_calibration(
        groups,
        endpoint,
        future,
        horizon,
        half_life_candidates=(None, 8.0),
        minimum_retention_candidates=(0.25,),
    )
    reverse = fit_horizon_discrepancy_calibration(
        tuple(reversed(groups)),
        endpoint[::-1],
        future[::-1],
        horizon,
        half_life_candidates=(None, 8.0),
        minimum_retention_candidates=(0.25,),
    )

    assert reverse.artifact_id == forward.artifact_id
    assert reverse.source_summary_sha256 == forward.source_summary_sha256


def test_horizon_zero_preserves_model_averaged_posterior() -> None:
    groups, horizon, endpoint, future = _source_fixture()
    calibration = fit_horizon_discrepancy_calibration(
        groups,
        endpoint,
        future,
        horizon,
        half_life_candidates=(8.0,),
        minimum_retention_candidates=(0.25,),
    )
    posterior = _posterior()

    prediction = predict_horizon_conditioned_endpoint(
        posterior,
        calibration,
        horizon_steps=0,
    )

    assert prediction.mean_retention == 1.0
    assert np.allclose(prediction.mean_m, posterior.mean_m)
    assert np.allclose(prediction.covariance_m2, posterior.covariance_m2)
    assert np.allclose(prediction.component_mean_m, posterior.component_mean_m)
    assert np.allclose(
        prediction.component_variance_m2,
        posterior.component_variance_m2,
    )
    assert np.allclose(prediction.additional_axis_variance_m2, 0.0)


def test_future_prediction_reverts_mean_and_expands_covariance() -> None:
    groups, horizon, endpoint, future = _source_fixture()
    calibration = fit_horizon_discrepancy_calibration(
        groups,
        endpoint,
        future,
        horizon,
        half_life_candidates=(8.0,),
        minimum_retention_candidates=(0.25,),
        minimum_process_std_m_per_sqrt_step=0.0001,
    )
    posterior = _posterior()
    now = predict_horizon_conditioned_endpoint(
        posterior,
        calibration,
        horizon_steps=0,
    )
    future_prediction = predict_horizon_conditioned_endpoint(
        posterior,
        calibration,
        horizon_steps=16,
    )

    expected_retention = 0.25 + 0.75 * 0.25
    assert np.isclose(future_prediction.mean_retention, expected_retention)
    assert np.allclose(
        future_prediction.mean_m,
        expected_retention * posterior.mean_m,
    )
    assert np.all(future_prediction.additional_axis_variance_m2 > 0.0)
    assert np.all(
        np.trace(future_prediction.covariance_m2, axis1=1, axis2=2)
        > np.trace(now.covariance_m2, axis1=1, axis2=2)
    )
    assert np.min(
        np.linalg.eigvalsh(future_prediction.covariance_m2),
        initial=0.0,
    ) >= -1e-12
    assert not future_prediction.mean_m.flags.writeable
    assert not future_prediction.covariance_m2.flags.writeable


def test_calibration_round_trip_and_tamper_detection(tmp_path) -> None:
    groups, horizon, endpoint, future = _source_fixture()
    calibration = fit_horizon_discrepancy_calibration(
        groups,
        endpoint,
        future,
        horizon,
        half_life_candidates=(8.0,),
        minimum_retention_candidates=(0.25,),
        metadata={"producer": "source-only-test"},
    )
    path = tmp_path / "horizon-calibration.json"

    save_horizon_discrepancy_calibration(path, calibration)
    loaded = load_horizon_discrepancy_calibration(path)

    assert loaded.artifact_id == calibration.artifact_id
    assert loaded.metadata["producer"] == "source-only-test"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["minimum_mean_retention"] = 0.5
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="artifact_id"):
        load_horizon_discrepancy_calibration(path)


def test_target_or_interval_outcomes_fail_closed() -> None:
    kwargs = {
        "source_group_ids": ("source-a", "source-b"),
        "source_summary_sha256": "a" * 64,
        "horizon_steps": (4, 8),
        "mean_reversion_half_life_steps": 8.0,
        "minimum_mean_retention": 0.25,
        "stationary_std_m": np.zeros(3),
        "process_std_m_per_sqrt_step": np.full(3, 1e-6),
    }
    with pytest.raises(ValueError, match="interval-calibration"):
        HorizonDiscrepancyCalibrationV1(
            **kwargs,
            interval_calibration_outcomes_used=True,
        )
    with pytest.raises(ValueError, match="target outcomes"):
        HorizonDiscrepancyCalibrationV1(
            **kwargs,
            confirmation_outcomes_used=True,
        )
    with pytest.raises(ValueError, match="target outcomes"):
        HorizonDiscrepancyCalibrationV1(
            **kwargs,
            target_outcomes_used=True,
        )


def test_invalid_source_shapes_and_groups_fail_closed() -> None:
    groups, horizon, endpoint, future = _source_fixture()
    with pytest.raises(ValueError, match="independent source groups"):
        fit_horizon_discrepancy_calibration(
            ("source-a",),
            endpoint[:1],
            future[:1],
            horizon,
        )
    with pytest.raises(ValueError, match="unique"):
        fit_horizon_discrepancy_calibration(
            ("source-a", "source-a"),
            endpoint[:2],
            future[:2],
            horizon,
        )
    with pytest.raises(ValueError, match="future_mean_m"):
        fit_horizon_discrepancy_calibration(
            groups,
            endpoint,
            future[:, :-1],
            horizon,
        )
    with pytest.raises(ValueError, match="strictly increasing"):
        fit_horizon_discrepancy_calibration(
            groups,
            endpoint,
            future,
            (8, 4, 16),
        )
