from __future__ import annotations

import numpy as np
import pytest

from bayesian_phystwin.causal4d_belief_provider_v1 import (
    infer_fixed_bayesian_anchor_endpoint,
)
from bayesian_phystwin.dynamic_endpoint_model_average import (
    infer_dynamic_endpoint_model_average,
)
from bayesian_phystwin.endpoint_model_average import infer_model_averaged_endpoint


class _RejectArrayCoercion:
    def __array__(self, dtype=None, copy=None):
        del dtype, copy
        raise TypeError("array coercion rejected")


def _provider_inputs() -> tuple[np.ndarray, np.ndarray]:
    residual = np.zeros((3, 2, 3), dtype=np.float64)
    residual[:, 0, 0] = (0.001, 0.002, 0.003)
    residual[:, 1, 1] = (-0.001, -0.002, -0.003)
    valid = np.asarray(
        (
            (True, True),
            (False, True),
            (True, False),
        ),
        dtype=bool,
    )
    return residual, valid


def test_fixed_anchor_provider_rejects_complex_residual_without_lossy_cast() -> None:
    residual, valid = _provider_inputs()
    complex_residual = residual.astype(np.complex128)
    complex_residual[0, 0, 0] += 0.5j

    with pytest.raises(ValueError, match="residual_m must contain real numeric values"):
        infer_fixed_bayesian_anchor_endpoint(
            complex_residual,
            valid,
            end_frame=len(residual),
        )


def test_fixed_anchor_provider_normalizes_residual_coercion_errors() -> None:
    _, valid = _provider_inputs()

    with pytest.raises(ValueError, match="residual_m must contain real numeric values"):
        infer_fixed_bayesian_anchor_endpoint(
            _RejectArrayCoercion(),
            valid,
            end_frame=1,
        )


@pytest.mark.parametrize(
    "invalid_validity",
    (
        np.asarray(((1, 1), (0, 2), (1, 0)), dtype=np.int64),
        np.asarray(((1.0, 1.0), (0.0, np.nan), (1.0, 0.0))),
        np.asarray(((1, 1), (-1, 1), (1, 0)), dtype=np.int64),
        np.asarray((("1", "1"), ("0", "1"), ("1", "0")), dtype=object),
        np.ones((3, 2), dtype=np.complex128),
    ),
)
def test_fixed_anchor_provider_rejects_nonbinary_validity(
    invalid_validity: np.ndarray,
) -> None:
    residual, _ = _provider_inputs()

    with pytest.raises(
        ValueError,
        match="valid must contain booleans or exact 0/1 values",
    ):
        infer_fixed_bayesian_anchor_endpoint(
            residual,
            invalid_validity,
            end_frame=len(residual),
        )


def test_fixed_anchor_provider_normalizes_validity_coercion_errors() -> None:
    residual, _ = _provider_inputs()

    with pytest.raises(
        ValueError,
        match="valid must contain booleans or exact 0/1 values",
    ):
        infer_fixed_bayesian_anchor_endpoint(
            residual,
            _RejectArrayCoercion(),
            end_frame=len(residual),
        )


def test_fixed_anchor_provider_rejects_mismatched_validity_shape() -> None:
    residual, _ = _provider_inputs()

    with pytest.raises(
        ValueError,
        match="valid must match the residual frame and track dimensions",
    ):
        infer_fixed_bayesian_anchor_endpoint(
            residual,
            np.ones((3, 1), dtype=bool),
            end_frame=len(residual),
        )


@pytest.mark.parametrize("end_frame", (True, np.bool_(True), 1.0))
def test_fixed_anchor_provider_rejects_noninteger_cutoff(end_frame: object) -> None:
    residual, valid = _provider_inputs()

    with pytest.raises(ValueError, match="end_frame must be an integer"):
        infer_fixed_bayesian_anchor_endpoint(
            residual,
            valid,
            end_frame=end_frame,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("dtype", (np.uint8, np.int64, np.float32, np.float64))
def test_fixed_anchor_provider_accepts_exact_zero_one_validity(dtype: type) -> None:
    residual, valid = _provider_inputs()
    expected = infer_fixed_bayesian_anchor_endpoint(
        residual,
        valid,
        end_frame=len(residual),
    )
    actual = infer_fixed_bayesian_anchor_endpoint(
        residual,
        valid.astype(dtype),
        end_frame=len(residual),
    )

    np.testing.assert_array_equal(actual.mean_m, expected.mean_m)
    np.testing.assert_array_equal(actual.variance_m2, expected.variance_m2)
    np.testing.assert_array_equal(
        actual.final_nominal_probability,
        expected.final_nominal_probability,
    )
    np.testing.assert_array_equal(actual.update_count, expected.update_count)


def test_model_average_rejects_complex_residual_without_lossy_cast() -> None:
    residual, valid = _provider_inputs()
    complex_residual = residual.astype(np.complex128)
    complex_residual[1, 0, 2] += 0.25j

    with pytest.raises(ValueError, match="residual_m must contain real numeric values"):
        infer_model_averaged_endpoint(
            complex_residual,
            valid,
            end_frame=len(residual),
        )


@pytest.mark.parametrize(
    "invalid_validity",
    (
        np.asarray(((1, 1), (0, 2), (1, 0)), dtype=np.int64),
        np.asarray((("1", "1"), ("0", "1"), ("1", "0")), dtype=object),
    ),
)
def test_model_average_rejects_nonbinary_validity(
    invalid_validity: np.ndarray,
) -> None:
    residual, _ = _provider_inputs()

    with pytest.raises(
        ValueError,
        match="valid must contain booleans or exact 0/1 values",
    ):
        infer_model_averaged_endpoint(
            residual,
            invalid_validity,
            end_frame=len(residual),
        )


def test_model_average_accepts_exact_zero_one_validity() -> None:
    residual, valid = _provider_inputs()
    expected = infer_model_averaged_endpoint(
        residual,
        valid,
        end_frame=len(residual),
    )
    actual = infer_model_averaged_endpoint(
        residual,
        valid.astype(np.int64),
        end_frame=len(residual),
    )

    np.testing.assert_array_equal(actual.mean_m, expected.mean_m)
    np.testing.assert_array_equal(actual.covariance_m2, expected.covariance_m2)
    np.testing.assert_array_equal(actual.component_weights, expected.component_weights)


def test_model_average_rejects_noninteger_cutoff() -> None:
    residual, valid = _provider_inputs()

    with pytest.raises(ValueError, match="end_frame must be an integer"):
        infer_model_averaged_endpoint(
            residual,
            valid,
            end_frame=1.0,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("residual", (np.ones((2, 1, 3), dtype=np.complex128), _RejectArrayCoercion()))
def test_dynamic_model_average_rejects_lossy_or_failed_residual_coercion(
    residual: object,
) -> None:
    valid = np.ones((2, 1), dtype=bool)

    with pytest.raises(ValueError, match="residual_m must contain real numeric values"):
        infer_dynamic_endpoint_model_average(
            residual,  # type: ignore[arg-type]
            valid,
            end_frame=2,
        )
