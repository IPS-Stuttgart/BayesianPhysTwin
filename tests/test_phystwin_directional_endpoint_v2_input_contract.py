from __future__ import annotations

import numpy as np
import pytest

from bayesian_phystwin.phystwin_directional_endpoint_v2 import (
    robust_directional_endpoint_v2,
)


def _inputs() -> dict[str, object]:
    source = np.zeros((1, 1, 3), dtype=np.float64)
    valid = np.ones((1, 1), dtype=np.bool_)
    return {
        "source_residual": source,
        "source_valid": valid,
        "multiview_residual": np.zeros_like(source),
        "multiview_valid": valid.copy(),
        "tangent_projectors": np.diag([1.0, 1.0, 0.0])[None],
        "priority_identities": np.asarray([False], dtype=np.bool_),
        "end_frame": 1,
        "process_variance": 0.0,
        "observation_variance": 1e-4,
        "initial_variance": 1e-3,
        "inlier_prior": 0.95,
        "outlier_variance_multiplier": 100.0,
    }


def _call(**changes: object) -> object:
    inputs = _inputs()
    inputs.update(changes)
    return robust_directional_endpoint_v2(**inputs)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "value",
    [
        np.asarray([[1]], dtype=np.int64),
        np.asarray([[0.5]], dtype=np.float64),
        np.asarray([[np.nan]], dtype=np.float64),
    ],
)
def test_source_validity_rejects_non_boolean_values(value: np.ndarray) -> None:
    with pytest.raises(ValueError, match="source validity must contain only booleans"):
        _call(source_valid=value)


@pytest.mark.parametrize(
    "value",
    [
        np.asarray([[1]], dtype=np.int64),
        np.asarray([[-1.0]], dtype=np.float64),
        np.asarray([[np.nan]], dtype=np.float64),
    ],
)
def test_multiview_validity_rejects_non_boolean_values(value: np.ndarray) -> None:
    with pytest.raises(
        ValueError,
        match="multiview validity must contain only booleans",
    ):
        _call(multiview_valid=value)


@pytest.mark.parametrize(
    "value",
    [
        np.asarray([1], dtype=np.int64),
        np.asarray([0.5], dtype=np.float64),
        np.asarray([np.nan], dtype=np.float64),
    ],
)
def test_priority_identities_reject_non_boolean_values(value: np.ndarray) -> None:
    with pytest.raises(ValueError, match="priority identities must contain only booleans"):
        _call(priority_identities=value)


def test_falsey_non_config_does_not_silently_select_default_policy() -> None:
    with pytest.raises(TypeError, match="config must be a DirectionalEndpointConfigV2"):
        _call(config=0)


def test_empty_point_axis_is_rejected() -> None:
    with pytest.raises(ValueError, match="point>=1"):
        _call(
            source_residual=np.zeros((1, 0, 3), dtype=np.float64),
            source_valid=np.zeros((1, 0), dtype=np.bool_),
            multiview_residual=np.zeros((1, 0, 3), dtype=np.float64),
            multiview_valid=np.zeros((1, 0), dtype=np.bool_),
            tangent_projectors=np.zeros((0, 3, 3), dtype=np.float64),
            priority_identities=np.zeros(0, dtype=np.bool_),
        )


def test_boolean_inputs_retain_existing_behavior() -> None:
    result = _call()

    np.testing.assert_array_equal(result.update_count, [1])
    assert np.all(np.isfinite(result.mean))
    assert np.all(np.isfinite(result.covariance))
