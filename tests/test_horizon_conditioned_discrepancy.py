from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest

from bayesian_phystwin.contracts.fixed_anchor import FixedBayesianAnchorConfigV1
from bayesian_phystwin.endpoint_model_average import (
    ModelAveragedEndpointConfigV1,
    infer_model_averaged_endpoint,
)
from bayesian_phystwin.horizon_conditioned_discrepancy import (
    HorizonDiscrepancyCalibrationV1,
    fit_horizon_discrepancy_calibration,
    load_horizon_discrepancy_calibration,
    predict_horizon_conditioned_endpoint,
    save_horizon_discrepancy_calibration,
)


def _posterior():
    residual = np.array(
        [
            [[0.010, -0.004, 0.002], [0.004, 0.003, -0.001]],
            [[0.012, -0.003, 0.003], [0.005, 0.004, -0.002]],
            [[0.011, -0.002, 0.004], [0.006, 0.004, -0.002]],
        ],
        dtype=np.float64,
    )
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
        ),
        component_prior_probability=(0.65, 0.35),
    )
    return infer_model_averaged_endpoint(
        residual,
        np.ones(residual.shape[:2], dtype=bool),
        end_frame=len(residual),
        config=config,
    )


def _calibration(**updates: object) -> HorizonDiscrepancyCalibrationV1:
    values: dict[str, object] = {
        "source_group_ids": ("object-a", "object-b", "object-c"),
        "source_summary_sha256": "a" * 64,
        "horizon_steps": (1, 5, 10),
        "mean_reversion_half_life_steps": 10.0,
        "minimum_mean_retention": 0.25,
        "stationary_std_m": np.array([0.003, 0.004, 0.005]),
        "additional_process_std_m_per_sqrt_step": np.array([0.0001, 0.0002, 0.0003]),
        "component_process_variance_scale": 0.75,
        "metadata": {"split": "source-only"},
    }
    values.update(updates)
    return HorizonDiscrepancyCalibrationV1(**values)  # type: ignore[arg-type]


def test_horizon_zero_preserves_model_averaged_endpoint_exactly() -> None:
    posterior = _posterior()
    prediction = predict_horizon_conditioned_endpoint(
        posterior,
        _calibration(),
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


def test_mean_reversion_and_axis_process_uncertainty_are_applied() -> None:
    posterior = _posterior()
    calibration = _calibration(
        mean_reversion_half_life_steps=8.0,
        minimum_mean_retention=0.2,
    )
    prediction = predict_horizon_conditioned_endpoint(
        posterior,
        calibration,
        horizon_steps=8,
    )

    assert np.isclose(prediction.mean_retention, 0.6)
    assert np.allclose(prediction.mean_m, 0.6 * posterior.mean_m)
    assert np.all(prediction.additional_axis_variance_m2 > 0.0)
    assert np.all(np.linalg.eigvalsh(prediction.covariance_m2) >= -1e-12)
    assert not prediction.mean_m.flags.writeable
    assert not prediction.covariance_m2.flags.writeable


def test_no_reversion_mode_keeps_mean_and_accumulates_variance() -> None:
    posterior = _posterior()
    calibration = _calibration(
        mean_reversion_half_life_steps=None,
        minimum_mean_retention=1.0,
        stationary_std_m=np.zeros(3),
    )
    prediction = predict_horizon_conditioned_endpoint(
        posterior,
        calibration,
        horizon_steps=12,
    )

    assert prediction.mean_retention == 1.0
    assert np.allclose(prediction.mean_m, posterior.mean_m)
    assert np.trace(prediction.covariance_m2[0]) > np.trace(posterior.covariance_m2[0])


def test_fit_recovers_source_mean_reversion_and_is_order_invariant() -> None:
    groups = ("c", "a", "d", "b")
    endpoint = np.array(
        [
            [0.010, -0.004, 0.002],
            [0.006, 0.002, -0.003],
            [-0.008, 0.004, 0.001],
            [0.004, -0.006, 0.005],
        ]
    )
    horizons = np.array([4, 8, 16, 32])
    expected_retention = 0.25 + 0.75 * np.power(2.0, -horizons / 8.0)
    deterministic = expected_retention[None, :, None] * endpoint[:, None, :]
    noise = np.array(
        [
            [[1, 0, 0], [0, 1, 0], [-1, 0, 1], [0, -1, 0]],
            [[-1, 0, 0], [0, -1, 0], [1, 0, -1], [0, 1, 0]],
            [[0, 1, 0], [1, 0, 0], [0, -1, 1], [-1, 0, 0]],
            [[0, -1, 0], [-1, 0, 0], [0, 1, -1], [1, 0, 0]],
        ],
        dtype=float,
    )
    future = deterministic + 2e-4 * noise

    fitted = fit_horizon_discrepancy_calibration(
        groups,
        endpoint,
        future,
        horizons,
        half_life_candidates=(None, 4.0, 8.0, 16.0),
        minimum_retention_candidates=(0.0, 0.25, 0.5),
    )
    permutation = np.array([2, 0, 3, 1])
    reordered = fit_horizon_discrepancy_calibration(
        tuple(groups[index] for index in permutation),
        endpoint[permutation],
        future[permutation],
        horizons,
        half_life_candidates=(16.0, 8.0, None, 4.0),
        minimum_retention_candidates=(0.5, 0.25, 0.0),
    )

    assert fitted.mean_reversion_half_life_steps == 8.0
    assert fitted.minimum_mean_retention == 0.25
    assert fitted.source_group_ids == tuple(sorted(groups))
    assert reordered.source_group_ids == fitted.source_group_ids
    assert reordered.source_summary_sha256 == fitted.source_summary_sha256
    assert reordered.mean_reversion_half_life_steps == 8.0
    assert reordered.minimum_mean_retention == 0.25
    assert reordered.artifact_id == fitted.artifact_id
    assert np.all(fitted.additional_process_std_m_per_sqrt_step > 0.0)


def test_fit_retains_persistent_source_when_it_is_best() -> None:
    groups = ("a", "b", "c")
    endpoint = np.array(
        [[0.01, 0.0, 0.0], [-0.006, 0.003, 0.0], [0.004, -0.002, 0.001]]
    )
    horizons = (2, 6, 12)
    future = np.repeat(endpoint[:, None, :], len(horizons), axis=1)

    fitted = fit_horizon_discrepancy_calibration(
        groups,
        endpoint,
        future,
        horizons,
        half_life_candidates=(None, 4.0, 8.0),
        minimum_retention_candidates=(0.0, 0.5),
    )

    assert fitted.mean_reversion_half_life_steps is None
    assert fitted.minimum_mean_retention == 1.0
    assert np.all(fitted.additional_process_std_m_per_sqrt_step >= 1e-6)


def test_calibration_roundtrip_is_content_addressed_and_strict(tmp_path) -> None:
    calibration = _calibration()
    path = tmp_path / "horizon-calibration.json"
    save_horizon_discrepancy_calibration(calibration, path)
    loaded = load_horizon_discrepancy_calibration(path)

    assert loaded.to_record() == calibration.to_record()
    assert loaded.artifact_id == calibration.artifact_id
    assert not loaded.stationary_std_m.flags.writeable
    with pytest.raises(FileExistsError):
        save_horizon_discrepancy_calibration(calibration, path)

    record = loaded.to_record()
    record["minimum_mean_retention"] = 0.5
    path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match="artifact_id"):
        load_horizon_discrepancy_calibration(path)

    path.write_text('{"schema": "x", "schema": "y"}', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate JSON object key"):
        load_horizon_discrepancy_calibration(path)


def test_calibration_and_fit_fail_closed_on_invalid_information_boundaries() -> None:
    cases = (
        ({"source_group_ids": ("only",)}, "at least two"),
        ({"source_group_ids": ("a", "a")}, "unique"),
        ({"horizon_steps": (1, 1)}, "strictly increasing"),
        ({"mean_reversion_half_life_steps": None}, "minimum_mean_retention=1"),
        (
            {
                "mean_reversion_half_life_steps": None,
                "minimum_mean_retention": 1.0,
                "additional_process_std_m_per_sqrt_step": np.zeros(3),
            },
            "positive floor",
        ),
        ({"interval_calibration_outcomes_used": True}, "cannot select"),
        ({"confirmation_outcomes_used": True}, "before target"),
        ({"target_outcomes_used": True}, "before target"),
    )
    for updates, message in cases:
        with pytest.raises(ValueError, match=message):
            _calibration(**updates)

    with pytest.raises(ValueError, match="shape"):
        fit_horizon_discrepancy_calibration(
            ("a", "b"),
            np.zeros((2, 2)),
            np.zeros((2, 2, 3)),
            (1, 2),
        )
    with pytest.raises(ValueError, match="finite"):
        future = np.zeros((2, 2, 3))
        future[0, 0, 0] = np.nan
        fit_horizon_discrepancy_calibration(
            ("a", "b"),
            np.zeros((2, 3)),
            future,
            (1, 2),
        )
    with pytest.raises(ValueError, match="candidate grid"):
        fit_horizon_discrepancy_calibration(
            ("a", "b"),
            np.zeros((2, 3)),
            np.zeros((2, 2, 3)),
            (1, 2),
            half_life_candidates=(2.0,),
            minimum_retention_candidates=(1.0,),
        )


def test_prediction_and_mapping_contracts_reject_tampering() -> None:
    posterior = _posterior()
    with pytest.raises(TypeError, match="posterior"):
        predict_horizon_conditioned_endpoint(  # type: ignore[arg-type]
            object(),
            _calibration(),
            horizon_steps=1,
        )
    with pytest.raises(ValueError, match="integer"):
        predict_horizon_conditioned_endpoint(
            posterior,
            _calibration(),
            horizon_steps=1.5,  # type: ignore[arg-type]
        )

    record = _calibration().to_record()
    record["extra"] = True
    with pytest.raises(ValueError, match="fields changed"):
        HorizonDiscrepancyCalibrationV1.from_mapping(record)
    record = _calibration().to_record()
    record["schema_version"] = 2
    with pytest.raises(ValueError, match="unsupported"):
        HorizonDiscrepancyCalibrationV1.from_mapping(record)

    with pytest.raises(ValueError, match="finite mean reversion"):
        replace(_calibration(), minimum_mean_retention=1.0)


def test_causal4d_provider_v2_advertises_horizon_calibration() -> None:
    from bayesian_phystwin.causal4d_belief_provider_v2 import (
        causal4d_belief_provider_v2_manifest,
    )

    manifest = causal4d_belief_provider_v2_manifest(provider_revision="f" * 40)
    assert "source_calibrated_horizon_discrepancy" in manifest["capabilities"]
    assert "mean_reverting_discrepancy_prediction" in manifest["capabilities"]
    assert manifest["artifact_schema_versions"]["HorizonDiscrepancyCalibration"] == 1
    assert (
        manifest["artifact_schema_versions"]["HorizonConditionedEndpointPrediction"]
        == 1
    )
