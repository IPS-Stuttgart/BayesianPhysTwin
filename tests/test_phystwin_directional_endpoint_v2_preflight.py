from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pytest

import bayesian_phystwin._phystwin_directional_endpoint_v2_solver as solver


def _component_inputs() -> dict[str, Any]:
    config = solver.DirectionalEndpointConfigV2()
    prior = solver.admit_spd_system(
        np.eye(3, dtype=np.float64),
        name="prior",
        config=config,
    )
    return {
        "mean": np.zeros(3, dtype=np.float64),
        "prior": prior,
        "innovation": np.zeros(1, dtype=np.float64),
        "projected_covariance": np.ones((1, 1), dtype=np.float64),
        "observation_matrix": np.asarray([[1.0, 0.0, 0.0]]),
        "observation_variance": 1.0,
        "name": "component",
        "config": config,
    }


def _robust_inputs() -> dict[str, Any]:
    return {
        "mean": np.zeros((1, 3), dtype=np.float64),
        "covariance": np.eye(3, dtype=np.float64)[None],
        "observation": np.zeros((1, 3), dtype=np.float64),
        "observation_matrix": np.eye(3, dtype=np.float64)[None],
        "observation_variance": 1e-4,
        "inlier_prior": 0.95,
        "outlier_variance_multiplier": 100.0,
        "name": "preflight",
        "config": solver.DirectionalEndpointConfigV2(),
    }


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("mean", np.asarray([np.inf, 0.0, 0.0])),
        ("innovation", np.asarray([np.inf])),
        ("projected_covariance", np.asarray([[np.inf]])),
        ("observation_matrix", np.asarray([[np.inf, 0.0, 0.0]])),
    ],
)
def test_component_preflight_rejects_before_matrix_arithmetic(
    field: str,
    value: np.ndarray,
) -> None:
    inputs = _component_inputs()
    inputs[field] = value

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        with pytest.raises(solver.SPDSolveError, match="non-finite"):
            solver._component_update(**inputs)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("mean", np.asarray([[np.inf, 0.0, 0.0]])),
        ("observation", np.asarray([[np.inf, 0.0, 0.0]])),
        (
            "observation_matrix",
            np.asarray(
                [
                    [
                        [np.inf, 0.0, 0.0],
                        [0.0, 1.0, 0.0],
                        [0.0, 0.0, 1.0],
                    ]
                ]
            ),
        ),
    ],
)
def test_robust_update_rejects_nonfinite_rows_without_runtime_warnings(
    field: str,
    value: np.ndarray,
) -> None:
    inputs = _robust_inputs()
    inputs[field] = value

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        with pytest.raises(
            solver.DirectionalEndpointNumericalError,
            match="failed SPD admission",
        ):
            solver.robust_linear_update_v2(**inputs)
