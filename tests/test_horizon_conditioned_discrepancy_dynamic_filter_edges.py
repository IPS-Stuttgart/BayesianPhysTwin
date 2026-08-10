from __future__ import annotations

import numpy as np
import pytest

import bayesian_phystwin._dynamic_endpoint_filter as filter_module
from bayesian_phystwin.contracts.fixed_anchor import FixedBayesianAnchorConfigV1
from bayesian_phystwin.dynamic_endpoint_model_average import (
    DynamicEndpointModelAverageConfigV2,
    DynamicEndpointNumericalError,
    PersistenceEndpointComponentV2,
    infer_dynamic_endpoint_model_average,
    predict_dynamic_endpoint_model_average,
)


def _single(component: object) -> DynamicEndpointModelAverageConfigV2:
    return DynamicEndpointModelAverageConfigV2(
        components=(component,),  # type: ignore[arg-type]
    )


def test_input_and_type_failures() -> None:
    with pytest.raises(ValueError, match="shape"):
        infer_dynamic_endpoint_model_average(
            np.zeros((2, 3)),
            np.zeros(2, dtype=bool),
            end_frame=1,
        )
    with pytest.raises(ValueError, match="must match"):
        infer_dynamic_endpoint_model_average(
            np.zeros((2, 1, 3)),
            np.zeros((2, 2), dtype=bool),
            end_frame=1,
        )
    residual = np.zeros((2, 1, 3))
    residual[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        infer_dynamic_endpoint_model_average(
            residual,
            np.ones((2, 1), dtype=bool),
            end_frame=1,
        )
    for cutoff in (True, 0, 3, 1.5):
        with pytest.raises(ValueError):
            infer_dynamic_endpoint_model_average(
                np.zeros((2, 1, 3)),
                np.ones((2, 1), dtype=bool),
                end_frame=cutoff,  # type: ignore[arg-type]
            )
    with pytest.raises(TypeError, match="config"):
        infer_dynamic_endpoint_model_average(
            np.zeros((2, 1, 3)),
            np.ones((2, 1), dtype=bool),
            end_frame=1,
            config=object(),  # type: ignore[arg-type]
        )
    posterior = infer_dynamic_endpoint_model_average(
        np.zeros((1, 1, 3)),
        np.ones((1, 1), dtype=bool),
        end_frame=1,
    )
    with pytest.raises(TypeError, match="posterior"):
        predict_dynamic_endpoint_model_average(
            object(),  # type: ignore[arg-type]
            horizon_steps=1,
        )
    for horizon in (True, -1, 0.5):
        with pytest.raises(ValueError, match="nonnegative integer"):
            predict_dynamic_endpoint_model_average(
                posterior,
                horizon_steps=horizon,  # type: ignore[arg-type]
            )


def test_result_arrays_are_defensively_owned_and_read_only() -> None:
    residual = np.zeros((1, 1, 3))
    posterior = infer_dynamic_endpoint_model_average(
        residual,
        np.ones((1, 1), dtype=bool),
        end_frame=1,
    )
    prediction = predict_dynamic_endpoint_model_average(posterior, horizon_steps=2)
    residual[0, 0, 0] = 99.0

    assert posterior.mean_m[0, 0] != 99.0
    arrays = (
        posterior.mean_m,
        posterior.covariance_m2,
        posterior.component_weights,
        posterior.component_state_mean,
        prediction.mean_m,
        prediction.component_velocity_mean_m_per_step,
    )
    assert all(not value.flags.writeable for value in arrays)
    with pytest.raises(ValueError):
        posterior.mean_m[0, 0] = 1.0


def test_nonfinite_internal_evidence_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = np.logaddexp

    def broken(*args: object, **kwargs: object) -> np.ndarray:
        result = original(*args, **kwargs)
        return np.full_like(result, np.nan)

    monkeypatch.setattr(np, "logaddexp", broken)
    with pytest.raises(ValueError, match="non-finite numerics"):
        infer_dynamic_endpoint_model_average(
            np.zeros((1, 1, 3)),
            np.ones((1, 1), dtype=bool),
            end_frame=1,
            config=_single(PersistenceEndpointComponentV2()),
        )


def test_nonfinite_weight_normalization_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = DynamicEndpointModelAverageConfigV2(
        components=(PersistenceEndpointComponentV2(),),
    )
    original = np.exp

    def broken(value: object, *args: object, **kwargs: object) -> np.ndarray:
        array = original(value, *args, **kwargs)
        return np.full_like(array, np.nan)

    monkeypatch.setattr(np, "exp", broken)
    with pytest.raises(ValueError, match="non-finite weights"):
        filter_module._normalized_component_weights(
            np.zeros((1, 1)),
            np.ones(1),
            config,
        )


def test_covariance_admission_never_clips_or_repairs() -> None:
    accepted = np.array([[[1.0, 1e-14], [0.0, -5e-13]]])
    symmetric = filter_module._admit_psd_2x2(accepted, name="roundoff probe")
    assert np.isclose(symmetric[0, 1, 0], 5e-15)
    assert symmetric[0, 1, 1] == accepted[0, 1, 1]

    with pytest.raises(DynamicEndpointNumericalError, match="non-finite"):
        filter_module._admit_psd_2x2(
            np.full((1, 2, 2), np.nan),
            name="nonfinite probe",
        )
    with pytest.raises(
        DynamicEndpointNumericalError,
        match="positive semidefinite",
    ):
        filter_module._admit_psd_2x2(
            np.array([[[1.0, 0.0], [0.0, -1e-6]]]),
            name="indefinite probe",
        )


def test_inference_detects_component_observation_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = filter_module._filter_component
    call_count = 0

    def inconsistent(*args: object, **kwargs: object) -> tuple[np.ndarray, ...]:
        nonlocal call_count
        result = original(*args, **kwargs)  # type: ignore[arg-type]
        call_count += 1
        if call_count == 2:
            values = list(result)
            values[3] = np.asarray(values[3]) + 1
            return tuple(values)
        return result

    monkeypatch.setattr(filter_module, "_filter_component", inconsistent)
    config = DynamicEndpointModelAverageConfigV2(
        components=(
            PersistenceEndpointComponentV2(),
            FixedBayesianAnchorConfigV1(),
        )
    )
    with pytest.raises(AssertionError, match="different observations"):
        infer_dynamic_endpoint_model_average(
            np.zeros((1, 1, 3)),
            np.ones((1, 1), dtype=bool),
            end_frame=1,
            config=config,
        )
