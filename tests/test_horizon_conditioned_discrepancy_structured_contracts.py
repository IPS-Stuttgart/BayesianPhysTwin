from typing import Any, cast

import numpy as np
import pytest

from bayesian_phystwin._structured_discrepancy_inference import _solve_spd
from bayesian_phystwin.contracts.fixed_anchor import FixedBayesianAnchorConfigV1
from bayesian_phystwin.endpoint_model_average import ModelAveragedEndpointConfigV1
from bayesian_phystwin.structured_discrepancy import (
    StructuredDiscrepancyConfigV1,
    StructuredDiscrepancyPosteriorV1,
    StructuredDiscrepancyPredictionV1,
    StructuredDiscrepancyQueryMomentsV1,
    infer_structured_discrepancy,
    predict_structured_discrepancy,
    structured_discrepancy_query_moments,
)


def _single_component() -> ModelAveragedEndpointConfigV1:
    return ModelAveragedEndpointConfigV1(
        components=(FixedBayesianAnchorConfigV1(),),
    )


def _small_posterior(*, component_count: int = 1) -> StructuredDiscrepancyPosteriorV1:
    components = tuple(
        FixedBayesianAnchorConfigV1(
            process_std_m=0.001 * index,
            observation_std_m=0.002 + 0.001 * index,
        )
        for index in range(component_count)
    )
    endpoint_config = ModelAveragedEndpointConfigV1(
        components=components,
        component_prior_probability=tuple(
            1.0 / component_count for _ in range(component_count)
        ),
    )
    return infer_structured_discrepancy(
        np.zeros((2, 2, 3)),
        np.ones((2, 2), dtype=bool),
        np.eye(2),
        end_frame=2,
        config=StructuredDiscrepancyConfigV1(
            endpoint_config=endpoint_config,
        ),
    )


def _posterior_arguments(
    posterior: StructuredDiscrepancyPosteriorV1,
) -> dict[str, Any]:
    return {
        "spatial_basis": posterior.spatial_basis,
        "component_coefficient_mean_m": posterior.component_coefficient_mean_m,
        "component_coefficient_covariance_m2": (
            posterior.component_coefficient_covariance_m2
        ),
        "component_local_variance_m2": posterior.component_local_variance_m2,
        "component_weights": posterior.component_weights,
        "component_log_score": posterior.component_log_score,
        "component_final_nominal_probability": (
            posterior.component_final_nominal_probability
        ),
        "update_count": posterior.update_count,
        "component_process_variance_m2": posterior.component_process_variance_m2,
        "config": posterior.config,
        "end_frame": posterior.end_frame,
    }


def _prediction_arguments(
    prediction: StructuredDiscrepancyPredictionV1,
) -> dict[str, Any]:
    return {
        "spatial_basis": prediction.spatial_basis,
        "component_coefficient_mean_m": prediction.component_coefficient_mean_m,
        "component_coefficient_covariance_m2": (
            prediction.component_coefficient_covariance_m2
        ),
        "component_local_variance_m2": prediction.component_local_variance_m2,
        "component_weights": prediction.component_weights,
        "component_process_variance_m2": prediction.component_process_variance_m2,
        "config": prediction.config,
        "source_end_frame": prediction.source_end_frame,
        "horizon_steps": prediction.horizon_steps,
    }


def test_config_contract_rejects_invalid_settings() -> None:
    with pytest.raises(TypeError, match="endpoint_config"):
        StructuredDiscrepancyConfigV1(
            endpoint_config=cast(ModelAveragedEndpointConfigV1, object())
        )
    with pytest.raises(TypeError, match="real number"):
        StructuredDiscrepancyConfigV1(basis_orthonormal_atol=True)
    with pytest.raises(ValueError, match="finite"):
        StructuredDiscrepancyConfigV1(basis_orthonormal_atol=np.inf)
    with pytest.raises(ValueError, match="positive"):
        StructuredDiscrepancyConfigV1(basis_orthonormal_atol=0.0)


def test_inference_input_contracts_fail_closed() -> None:
    residual = np.zeros((2, 3, 3))
    valid = np.ones((2, 3), dtype=bool)
    basis = np.eye(3)
    config = StructuredDiscrepancyConfigV1(
        endpoint_config=_single_component(),
    )

    with pytest.raises(TypeError, match="residual_m"):
        infer_structured_discrepancy(
            residual.astype(bool), valid, basis, end_frame=2, config=config
        )
    with pytest.raises(ValueError, match="shape"):
        infer_structured_discrepancy(
            np.zeros((2, 3, 2)), valid, basis, end_frame=2, config=config
        )
    invalid_residual = residual.copy()
    invalid_residual[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        infer_structured_discrepancy(
            invalid_residual, valid, basis, end_frame=2, config=config
        )
    with pytest.raises(TypeError, match="boolean"):
        infer_structured_discrepancy(
            residual, valid.astype(float), basis, end_frame=2, config=config
        )
    with pytest.raises(ValueError, match="match"):
        infer_structured_discrepancy(
            residual, valid[:, :2], basis, end_frame=2, config=config
        )
    with pytest.raises(TypeError, match="prior_reliability"):
        infer_structured_discrepancy(
            residual,
            valid,
            basis,
            prior_reliability=valid,
            end_frame=2,
            config=config,
        )
    with pytest.raises(ValueError, match="match"):
        infer_structured_discrepancy(
            residual,
            valid,
            basis,
            prior_reliability=np.ones((2, 2)),
            end_frame=2,
            config=config,
        )
    invalid_reliability = np.ones((2, 3))
    invalid_reliability[0, 0] = np.nan
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        infer_structured_discrepancy(
            residual,
            valid,
            basis,
            prior_reliability=invalid_reliability,
            end_frame=2,
            config=config,
        )
    with pytest.raises(TypeError, match="end_frame"):
        infer_structured_discrepancy(
            residual, valid, basis, end_frame=True, config=config
        )
    with pytest.raises(ValueError, match="integer"):
        infer_structured_discrepancy(
            residual, valid, basis, end_frame=1.5, config=config
        )
    with pytest.raises(ValueError, match="inside"):
        infer_structured_discrepancy(residual, valid, basis, end_frame=0, config=config)
    with pytest.raises(TypeError, match="config"):
        infer_structured_discrepancy(
            residual,
            valid,
            basis,
            end_frame=2,
            config=cast(StructuredDiscrepancyConfigV1, object()),
        )


def test_basis_and_spd_numeric_boundaries_fail_closed() -> None:
    with pytest.raises(TypeError, match="numeric matrix"):
        infer_structured_discrepancy(
            np.zeros((1, 2, 3)),
            np.ones((1, 2), dtype=bool),
            np.ones((2, 1), dtype=bool),
            end_frame=1,
            config=StructuredDiscrepancyConfigV1(
                endpoint_config=_single_component(),
            ),
        )
    with pytest.raises(ValueError, match="positive definite"):
        _solve_spd(np.zeros((1, 1)), np.ones((1, 1)))


@pytest.mark.parametrize(
    ("changes", "error", "message"),
    [
        ({"spatial_basis": np.ones(2)}, ValueError, "nonempty matrix"),
        (
            {"component_coefficient_mean_m": np.array([[["x", "x", "x"]]])},
            TypeError,
            "real numeric",
        ),
        (
            {"component_coefficient_mean_m": np.full((1, 2, 3), np.nan)},
            ValueError,
            "finite",
        ),
        (
            {"component_coefficient_mean_m": np.zeros((1, 2))},
            ValueError,
            "coefficient_mean",
        ),
        (
            {"component_coefficient_covariance_m2": np.zeros((1, 2, 1))},
            ValueError,
            "covariance_m2 shape",
        ),
        (
            {"component_local_variance_m2": np.zeros((1, 1))},
            ValueError,
            "local_variance",
        ),
        ({"component_weights": np.ones(2)}, ValueError, "weights shape"),
        (
            {"component_process_variance_m2": np.ones(2)},
            ValueError,
            "process_variance_m2 shape",
        ),
        (
            {"component_local_variance_m2": -np.ones((1, 2))},
            ValueError,
            "nonnegative",
        ),
        ({"component_weights": np.array([0.5])}, ValueError, "normalized"),
        (
            {
                "component_coefficient_covariance_m2": np.array(
                    [[[1.0, 1.0], [0.0, 1.0]]]
                )
            },
            ValueError,
            "symmetric",
        ),
        (
            {
                "component_coefficient_covariance_m2": np.array(
                    [[[-1.0, 0.0], [0.0, 1.0]]]
                )
            },
            ValueError,
            "positive semidefinite",
        ),
    ],
)
def test_component_state_contract_rejects_tampering(
    changes: dict[str, Any],
    error: type[Exception],
    message: str,
) -> None:
    arguments = _posterior_arguments(_small_posterior())
    arguments.update(changes)
    with pytest.raises(error, match=message):
        StructuredDiscrepancyPosteriorV1(**arguments)


def test_posterior_contract_rejects_identity_and_summary_tampering() -> None:
    posterior = _small_posterior()

    arguments = _posterior_arguments(posterior)
    arguments["config"] = cast(StructuredDiscrepancyConfigV1, object())
    with pytest.raises(TypeError, match="config"):
        StructuredDiscrepancyPosteriorV1(**arguments)

    for invalid_end_frame, error in ((True, TypeError), (0, ValueError)):
        arguments = _posterior_arguments(posterior)
        arguments["end_frame"] = invalid_end_frame
        with pytest.raises(error, match="end_frame"):
            StructuredDiscrepancyPosteriorV1(**arguments)

    arguments = _posterior_arguments(posterior)
    arguments["config"] = StructuredDiscrepancyConfigV1(
        endpoint_config=ModelAveragedEndpointConfigV1(
            components=(
                FixedBayesianAnchorConfigV1(process_std_m=0.0),
                FixedBayesianAnchorConfigV1(process_std_m=0.001),
            )
        )
    )
    with pytest.raises(ValueError, match="component count"):
        StructuredDiscrepancyPosteriorV1(**arguments)

    arguments = _posterior_arguments(posterior)
    arguments["component_process_variance_m2"] = np.array([1.0])
    with pytest.raises(ValueError, match="differs from config"):
        StructuredDiscrepancyPosteriorV1(**arguments)

    arguments = _posterior_arguments(posterior)
    arguments["component_log_score"] = np.array([np.nan])
    with pytest.raises(ValueError, match="log_score"):
        StructuredDiscrepancyPosteriorV1(**arguments)

    two_component = _small_posterior(component_count=2)
    arguments = _posterior_arguments(two_component)
    arguments["component_weights"] = np.array([0.75, 0.25])
    with pytest.raises(ValueError, match="scores and prior"):
        StructuredDiscrepancyPosteriorV1(**arguments)

    arguments = _posterior_arguments(posterior)
    arguments["component_final_nominal_probability"] = np.zeros((1, 1))
    with pytest.raises(ValueError, match="probability shape"):
        StructuredDiscrepancyPosteriorV1(**arguments)

    arguments = _posterior_arguments(posterior)
    arguments["component_final_nominal_probability"] = np.full((1, 2), 2.0)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        StructuredDiscrepancyPosteriorV1(**arguments)

    arguments = _posterior_arguments(posterior)
    arguments["update_count"] = np.ones(2)
    with pytest.raises(ValueError, match="integers"):
        StructuredDiscrepancyPosteriorV1(**arguments)

    arguments = _posterior_arguments(posterior)
    arguments["update_count"] = -np.ones(2, dtype=np.int64)
    with pytest.raises(ValueError, match="nonnegative"):
        StructuredDiscrepancyPosteriorV1(**arguments)


def test_prediction_contract_and_public_horizon_validation_fail_closed() -> None:
    posterior = _small_posterior()
    prediction = predict_structured_discrepancy(posterior, horizon_steps=1)

    with pytest.raises(TypeError, match="posterior"):
        predict_structured_discrepancy(
            cast(StructuredDiscrepancyPosteriorV1, object()), horizon_steps=1
        )
    with pytest.raises(TypeError, match="horizon_steps"):
        predict_structured_discrepancy(posterior, horizon_steps=True)
    with pytest.raises(ValueError, match="nonnegative"):
        predict_structured_discrepancy(posterior, horizon_steps=-1)
    with pytest.raises(ValueError, match="nonnegative"):
        predict_structured_discrepancy(posterior, horizon_steps=1.5)

    arguments = _prediction_arguments(prediction)
    arguments["config"] = cast(StructuredDiscrepancyConfigV1, object())
    with pytest.raises(TypeError, match="config"):
        StructuredDiscrepancyPredictionV1(**arguments)

    for field_name, invalid_value, error in (
        ("source_end_frame", True, TypeError),
        ("source_end_frame", 0, ValueError),
        ("horizon_steps", True, TypeError),
        ("horizon_steps", -1, ValueError),
    ):
        arguments = _prediction_arguments(prediction)
        arguments[field_name] = invalid_value
        with pytest.raises(error, match=field_name):
            StructuredDiscrepancyPredictionV1(**arguments)

    arguments = _prediction_arguments(prediction)
    arguments["config"] = StructuredDiscrepancyConfigV1(
        endpoint_config=ModelAveragedEndpointConfigV1(
            components=(
                FixedBayesianAnchorConfigV1(process_std_m=0.0),
                FixedBayesianAnchorConfigV1(process_std_m=0.001),
            )
        )
    )
    with pytest.raises(ValueError, match="component count"):
        StructuredDiscrepancyPredictionV1(**arguments)

    arguments = _prediction_arguments(prediction)
    arguments["component_process_variance_m2"] = np.array([1.0])
    with pytest.raises(ValueError, match="differs from config"):
        StructuredDiscrepancyPredictionV1(**arguments)


def test_query_contracts_fail_closed() -> None:
    posterior = _small_posterior()
    with pytest.raises(TypeError, match="belief"):
        structured_discrepancy_query_moments(
            cast(StructuredDiscrepancyPosteriorV1, object()), np.eye(6)
        )
    with pytest.raises(TypeError, match="numeric"):
        structured_discrepancy_query_moments(posterior, np.ones((1, 6), dtype=bool))
    with pytest.raises(ValueError, match="finite and nonempty"):
        structured_discrepancy_query_moments(posterior, np.empty((0, 6)))
    invalid_query = np.zeros((1, 6))
    invalid_query[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite and nonempty"):
        structured_discrepancy_query_moments(posterior, invalid_query)

    invalid_moments = (
        (np.empty(0), np.empty((0, 0)), "nonempty"),
        (np.zeros(2), np.eye(1), "shape"),
        (np.array([np.nan]), np.eye(1), "finite"),
        (np.zeros(2), np.array([[1.0, 1.0], [0.0, 1.0]]), "symmetric"),
        (np.zeros(1), np.array([[-1.0]]), "positive semidefinite"),
    )
    for mean, covariance, message in invalid_moments:
        with pytest.raises(ValueError, match=message):
            StructuredDiscrepancyQueryMomentsV1(
                mean=mean,
                covariance=covariance,
            )
