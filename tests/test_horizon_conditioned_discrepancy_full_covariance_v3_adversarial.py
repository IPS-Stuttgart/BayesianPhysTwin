from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import bayesian_phystwin._full_covariance_dynamic_endpoint_v3 as v3_module
from bayesian_phystwin.dynamic_endpoint_model_average import (
    DynamicEndpointNumericalError,
    infer_full_covariance_dynamic_endpoint_model_average,
    predict_full_covariance_dynamic_endpoint_model_average,
)


def _metric_covariance(
    frame_count: int,
    track_count: int,
    covariance: np.ndarray,
) -> np.ndarray:
    return np.broadcast_to(
        covariance,
        (frame_count, track_count, 3, 3),
    ).copy()


@pytest.mark.parametrize(
    ("covariance", "message"),
    [
        (np.zeros((1, 1, 2, 2)), "shape"),
        (np.full((1, 1, 3, 3), np.nan), "non-finite"),
        (
            np.array(
                [[[[1.0, 0.1, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]]]
            ),
            "not symmetric",
        ),
        (-np.eye(3)[None, None], "not positive semidefinite"),
    ],
)
def test_invalid_observation_covariance_fails_closed(
    covariance: np.ndarray,
    message: str,
) -> None:
    with pytest.raises((ValueError, DynamicEndpointNumericalError), match=message):
        infer_full_covariance_dynamic_endpoint_model_average(
            np.zeros((1, 1, 3)),
            np.ones((1, 1), dtype=np.bool_),
            covariance,
            end_frame=1,
        )


def test_invalid_type_and_boundary_inputs_fail_closed() -> None:
    residual = np.zeros((2, 1, 3))
    valid = np.ones((2, 1), dtype=np.bool_)
    covariance = _metric_covariance(2, 1, np.eye(3) * 1e-5)
    with pytest.raises(ValueError, match="booleans"):
        infer_full_covariance_dynamic_endpoint_model_average(
            residual,
            np.ones((2, 1), dtype=np.int64),
            covariance,
            end_frame=1,
        )
    with pytest.raises(ValueError, match="integer"):
        infer_full_covariance_dynamic_endpoint_model_average(
            residual,
            valid,
            covariance,
            end_frame=1.5,
        )
    with pytest.raises(ValueError, match="inside"):
        infer_full_covariance_dynamic_endpoint_model_average(
            residual,
            valid,
            covariance,
            end_frame=0,
        )
    with pytest.raises(TypeError, match="config"):
        infer_full_covariance_dynamic_endpoint_model_average(
            residual,
            valid,
            covariance,
            end_frame=1,
            config=object(),  # type: ignore[arg-type]
        )
    with pytest.raises(TypeError, match="posterior"):
        predict_full_covariance_dynamic_endpoint_model_average(
            object(),  # type: ignore[arg-type]
            horizon_steps=0,
        )
    posterior = infer_full_covariance_dynamic_endpoint_model_average(
        residual,
        valid,
        covariance,
        end_frame=1,
    )
    with pytest.raises(ValueError, match="nonnegative integer"):
        predict_full_covariance_dynamic_endpoint_model_average(
            posterior,
            horizon_steps=-1,
        )
    with pytest.raises(ValueError, match="nonnegative integer"):
        predict_full_covariance_dynamic_endpoint_model_average(
            posterior,
            horizon_steps=True,
        )
    with pytest.raises(ValueError, match="real numeric"):
        infer_full_covariance_dynamic_endpoint_model_average(
            [[[{"not": "numeric"}]]],
            [[True]],
            np.zeros((1, 1, 3, 3)),
            end_frame=1,
        )


def test_implementation_uses_no_explicit_matrix_inverse() -> None:
    source = Path(v3_module.__file__).read_text(encoding="utf-8")
    assert "np.linalg.inv" not in source
    assert "np.linalg.pinv" not in source
