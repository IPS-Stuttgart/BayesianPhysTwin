from __future__ import annotations

import inspect

import numpy as np
import pytest

import bayesian_phystwin.phystwin_directional_endpoint as endpoint_module
from bayesian_phystwin.phystwin_directional_endpoint import (
    robust_directional_endpoint,
)


def _tangent_projectors(point_count: int) -> np.ndarray:
    return np.repeat(np.diag([1.0, 1.0, 0.0])[None], point_count, axis=0)


def test_directional_endpoint_avoids_explicit_matrix_inverse() -> None:
    source = inspect.getsource(endpoint_module)
    assert "np.linalg.inv" not in source


def test_directional_endpoint_covariance_stays_psd_under_stiff_updates() -> None:
    rng = np.random.default_rng(4)
    frame_count = 500
    point_count = 8
    source = rng.normal(scale=1e-3, size=(frame_count, point_count, 3))
    multiview = rng.normal(scale=1e-3, size=(frame_count, point_count, 3))
    valid = np.ones((frame_count, point_count), dtype=bool)

    result = robust_directional_endpoint(
        source,
        valid,
        multiview,
        valid,
        _tangent_projectors(point_count),
        np.ones(point_count, dtype=bool),
        end_frame=frame_count,
        process_variance=1e-16,
        observation_variance=1e-18,
        initial_variance=1e6,
        inlier_prior=0.95,
        outlier_variance_multiplier=1e6,
    )

    eigenvalues = np.linalg.eigvalsh(result.covariance)
    assert np.all(np.isfinite(result.mean))
    assert np.all(np.isfinite(result.covariance))
    assert np.min(eigenvalues) >= -1e-12
    np.testing.assert_allclose(
        result.covariance,
        np.swapaxes(result.covariance, 1, 2),
        atol=1e-15,
    )
    np.testing.assert_allclose(
        result.variance,
        eigenvalues[:, -1],
        rtol=1e-12,
        atol=1e-18,
    )


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("process_variance", np.nan),
        ("observation_variance", np.inf),
        ("initial_variance", np.nan),
        ("inlier_prior", np.nan),
        ("outlier_variance_multiplier", np.inf),
    ],
)
def test_directional_endpoint_rejects_nonfinite_filter_parameters(
    name: str,
    value: float,
) -> None:
    source = np.zeros((1, 1, 3), dtype=np.float64)
    valid = np.ones((1, 1), dtype=bool)
    parameters = {
        "end_frame": 1,
        "process_variance": 0.0,
        "observation_variance": 1e-4,
        "initial_variance": 1e-3,
        "inlier_prior": 0.95,
        "outlier_variance_multiplier": 100.0,
    }
    parameters[name] = value

    with pytest.raises(ValueError, match="finite"):
        robust_directional_endpoint(
            source,
            valid,
            source,
            valid,
            _tangent_projectors(1),
            np.ones(1, dtype=bool),
            **parameters,
        )


def test_roundoff_psd_repair_clips_only_tiny_negative_modes() -> None:
    covariance = np.array(
        [
            [[-1e-16, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 2.0]],
            [[1.0, 1e-16, 0.0], [-1e-16, 2.0, 0.0], [0.0, 0.0, 3.0]],
        ],
        dtype=np.float64,
    )

    repaired = endpoint_module._repair_roundoff_psd(covariance)  # noqa: SLF001

    assert np.min(np.linalg.eigvalsh(repaired)) >= 0.0
    np.testing.assert_allclose(
        repaired,
        np.swapaxes(repaired, 1, 2),
        atol=0.0,
    )
    np.testing.assert_allclose(repaired[1], np.diag([1.0, 2.0, 3.0]))


def test_roundoff_psd_repair_fails_on_nonfinite_or_material_defects() -> None:
    empty = endpoint_module._repair_roundoff_psd(  # noqa: SLF001
        np.zeros((0, 3, 3), dtype=np.float64)
    )
    assert empty.shape == (0, 3, 3)

    with pytest.raises(FloatingPointError, match="non-finite"):
        endpoint_module._repair_roundoff_psd(  # noqa: SLF001
            np.full((1, 3, 3), np.nan)
        )
    with pytest.raises(np.linalg.LinAlgError, match="semidefiniteness"):
        endpoint_module._repair_roundoff_psd(  # noqa: SLF001
            np.diag([-1e-3, 1.0, 1.0])[None]
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("source_valid", np.ones((1, 1), dtype=np.int64)),
        ("multiview_valid", np.array([[np.nan]])),
        ("priority_identities", np.ones(1, dtype=np.int64)),
    ],
)
def test_directional_endpoint_rejects_nonboolean_support_inputs(
    field: str,
    value: np.ndarray,
) -> None:
    source = np.zeros((1, 1, 3), dtype=np.float64)
    arguments: dict[str, object] = {
        "source_residual": source,
        "source_valid": np.ones((1, 1), dtype=np.bool_),
        "multiview_residual": source,
        "multiview_valid": np.ones((1, 1), dtype=np.bool_),
        "tangent_projectors": _tangent_projectors(1),
        "priority_identities": np.ones(1, dtype=np.bool_),
        "end_frame": 1,
        "process_variance": 0.0,
        "observation_variance": 1e-4,
        "initial_variance": 1e-3,
        "inlier_prior": 0.95,
        "outlier_variance_multiplier": 100.0,
    }
    arguments[field] = value

    with pytest.raises(ValueError, match="boolean dtype"):
        robust_directional_endpoint(**arguments)  # type: ignore[arg-type]
